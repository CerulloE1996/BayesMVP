

//// MVOP_PartialLog_helper_fns.hpp
////
//// Log-scale ("PartialLog") helper functions for the LC-MVOP / MVOP models.
////
//// Design notes (differs deliberately from the LC-MVP PartialLog machinery):
////  - The MVP PartialLog grad path uses VECTORISED signed-log containers + the
////    fn_MVP_*_grad_log_scale family. Those functions form their L_Omega diagonal
////    seeds internally as (y-quantity * Bound_Z), which is WRONG for ordinal tests
////    (ordinal seeds are two-term:  phi(lb)*lb - phi(ub)*ub  etc.), so they cannot
////    be reused for MVOP. Instead of duplicating ~2000 lines of vectorised signed-log
////    machinery, the MVOP log-scale gradient fix-up below is SCALAR and PER-ROW:
////    it is only ever run on the (few) "problem" rows detected by the masks, which
////    is the same performance profile as the "Stan" scalar fallback philosophy.
////  - All chain-rule recursions below follow the *general* accumulation form
////    (weight_ii = prev * prob_n_recip * prod_{t' != t0+ii} prob(t')), i.e. the
////    form used by the FIXED fn_MVOP_compute_cutpoint_grad general case.
////
//// REQUIREMENTS: stan-math >= 4.3 (for stan::math::std_normal_log_qf).
//// Include AFTER MVOP_manual_grad_calc_fns.hpp / MVP helper headers.

#pragma once

#include <Eigen/Dense>
#include <limits>
#include <cmath>
#include <vector>

static constexpr double SL_NEG_INF        = -std::numeric_limits<double>::infinity();
static constexpr double HALF_LOG_TWO_PI   =  0.91893853320467274178;  // 0.5*log(2*pi)




//// -----------------------------------------------------------------------------
//// Scalar dispatchers - route to the SAME functions the vectorised paths use.
//// NB: the codebase has NO custom log_Phi / inv_Phi_from_log_prob; the log-scale
//// paths everywhere else use the APPROX pair (log_Phi_approx + inv_Phi_approx_from_
//// logit_prob), so this file must too, or it disagrees with the standard-scale pass
//// that produced its inputs.
//// 2026-09-22 (assistant): the note above now applies to the Phi_approx setting only. For
//// Phi_type = "Phi" / inv_Phi_type = "inv_Phi" the log-scale tail paths use the exact pair
//// fast_log_Phi / fast_inv_Phi_from_log_p (double_fns.hpp); see
//// fn_MVOP_compute_lp_GHK_cols_log_scale_ordinal and MVP_log_scale_grad_calc_fns_T.hpp.
//// -----------------------------------------------------------------------------
ALWAYS_INLINE double sl_exp(const double x, const bool S)   { return S ? stan::math::exp(x)   : fast_exp_1(x); }
ALWAYS_INLINE double sl_log(const double x, const bool S)   { return S ? stan::math::log(x)   : fast_log_1(x); }
ALWAYS_INLINE double sl_log1m(const double x, const bool S) { return S ? stan::math::log1m(x) : fast_log1m_1(x); }
////
//// log Phi(x) ~=~ log_inv_logit(0.07056*x^3 + 1.5976*x)   [matches "log_Phi_approx"]:
ALWAYS_INLINE double sl_log_Phi(const double x, const bool S) {
      const double t = 0.07056*x*x*x + 1.5976*x;
      return S ? stan::math::log_inv_logit(t) : fast_log_inv_logit(t);
}
////
ALWAYS_INLINE double sl_inv_Phi_from_logit(const double logit_p, const bool S) {
      return S ? inv_Phi_approx_from_logit_prob_std(logit_p) : fast_inv_Phi_approx_from_logit_prob(logit_p);
}
////
//// log(1 - exp(x)), x < 0. expm1 has no custom/SIMD counterpart in this codebase, so
//// libm is used for the near-zero branch (exact, scalar, problem-rows only):
ALWAYS_INLINE double sl_log1m_exp(const double x, const bool S) {
      if (x > -0.6931471805599453) return std::log(-std::expm1(x));
      return sl_log1m(sl_exp(x, S), S);
}
ALWAYS_INLINE double sl_log_diff_exp(const double a, const double b, const bool S) {   //// log(e^a - e^b), a > b
      return a + sl_log1m_exp(b - a, S);
}




//// -----------------------------------------------------------------------------
//// Signed-log scalar arithmetic:  x  <->  { lg = log|x|, sg = sign(x) }
//// lg == -Inf  represents exact zero.
//// -----------------------------------------------------------------------------
struct SignedLog {
  double lg;
  double sg;
};

ALWAYS_INLINE SignedLog sl_zero() {
  return { SL_NEG_INF, 1.0 };
}

ALWAYS_INLINE bool sl_is_zero(const SignedLog &a) {
  return !(a.lg > SL_NEG_INF);
}

//// make a SignedLog from (log|x|, sign) - returns exact zero if lg is -Inf/NaN:
ALWAYS_INLINE SignedLog sl_make(const double lg, const double sg) {
  if (!(lg > SL_NEG_INF)) return sl_zero();
  return { lg, (sg < 0.0) ? -1.0 : 1.0 };
}

// ALWAYS_INLINE SignedLog sl_from_val(const double x) {
//   if ( (x == 0.0) || (!std::isfinite(x)) ) return sl_zero();  //// nonfinite guarded to zero (call sites guard Inf bounds explicitly)
//   return { std::log(std::abs(x)), (x < 0.0) ? -1.0 : 1.0 };
// }
// 
// ALWAYS_INLINE SignedLog sl_neg(const SignedLog &a) {
//   if (sl_is_zero(a)) return sl_zero();
//   return { a.lg, -a.sg };
// }
// 
// ALWAYS_INLINE SignedLog sl_mul(const SignedLog &a, const SignedLog &b) {
//   if (sl_is_zero(a) || sl_is_zero(b)) return sl_zero();
//   return { a.lg + b.lg, a.sg * b.sg };
// }
// 
// ALWAYS_INLINE SignedLog sl_add(const SignedLog &a, const SignedLog &b) {
//   if (sl_is_zero(a)) return b;
//   if (sl_is_zero(b)) return a;
//   const double m = (a.lg > b.lg) ? a.lg : b.lg;
//   const double s = a.sg * std::exp(a.lg - m) + b.sg * std::exp(b.lg - m);
//   if (s == 0.0) return sl_zero();
//   return { m + std::log(std::abs(s)), (s < 0.0) ? -1.0 : 1.0 };
// }
// 
// ALWAYS_INLINE double sl_val(const SignedLog &a) {
//   if (sl_is_zero(a)) return 0.0;
//   return a.sg * std::exp(a.lg);
// }
// 
// //// log of standard-normal pdf; -Inf at +-Inf argument (phi = 0):
// ALWAYS_INLINE double log_phi_std_norm(const double x) {
//   if (std::isinf(x)) return SL_NEG_INF;
//   return -0.5 * x * x - HALF_LOG_TWO_PI;
// }
// 
// //// numerically-safe log-sum-exp of two logs (either may be -Inf):
// ALWAYS_INLINE double lse_2(const double a, const double b) {
//   if (!(a > SL_NEG_INF)) return b;
//   if (!(b > SL_NEG_INF)) return a;
//   const double m = (a > b) ? a : b;
//   return m + stan::math::log1p_exp( ((a > b) ? b : a) - m );
// }

ALWAYS_INLINE SignedLog sl_from_val(const double x, const bool S) {
  if ( (x == 0.0) || (!std::isfinite(x)) ) return sl_zero();  //// nonfinite guarded to zero (call sites guard Inf bounds explicitly)
  return { sl_log(std::abs(x), S), (x < 0.0) ? -1.0 : 1.0 };
}

ALWAYS_INLINE SignedLog sl_neg(const SignedLog &a) {
  if (sl_is_zero(a)) return sl_zero();
  return { a.lg, -a.sg };
}

ALWAYS_INLINE SignedLog sl_mul(const SignedLog &a, const SignedLog &b) {
  if (sl_is_zero(a) || sl_is_zero(b)) return sl_zero();
  return { a.lg + b.lg, a.sg * b.sg };
}

ALWAYS_INLINE SignedLog sl_add(const SignedLog &a, const SignedLog &b, const bool S) {
  if (sl_is_zero(a)) return b;
  if (sl_is_zero(b)) return a;
  const double m = (a.lg > b.lg) ? a.lg : b.lg;
  const double s = a.sg * sl_exp(a.lg - m, S) + b.sg * sl_exp(b.lg - m, S);
  if (s == 0.0) return sl_zero();
  return { m + sl_log(std::abs(s), S), (s < 0.0) ? -1.0 : 1.0 };
}

ALWAYS_INLINE double sl_val(const SignedLog &a, const bool S) {
  if (sl_is_zero(a)) return 0.0;
  return a.sg * sl_exp(a.lg, S);
}

//// log of standard-normal pdf; -Inf at +-Inf argument (phi = 0). Analytic - no dispatch needed:
ALWAYS_INLINE double log_phi_std_norm(const double x) {
  if (std::isinf(x)) return SL_NEG_INF;
  return -0.5 * x * x - HALF_LOG_TWO_PI;
}

//// numerically-safe log-sum-exp of two logs (either may be -Inf):
ALWAYS_INLINE double lse_2(const double a, const double b, const bool S) {
  if (!(a > SL_NEG_INF)) return b;
  if (!(b > SL_NEG_INF)) return a;
  const double m = (a > b) ? a : b;
  const double d = ((a > b) ? b : a) - m;
  return m + (S ? stan::math::log1p_exp(d) : fast_log1p_exp_1(d));
}



//// -----------------------------------------------------------------------------
//// fn_MVOP_compute_lp_GHK_cols_log_scale_ordinal
////
//// Log-scale fix-up of the ordinal GHK column quantities for the "problem"
//// observations of test t (both truncation bounds deep in the SAME tail, so
//// prob = Phi(ub) - Phi(lb) underflows / catastrophically cancels).
////
//// Recomputes (exactly, on the log scale) and OVERWRITES for the given indices:
////    y1_log_prob(i, t)          = log( Phi(ub) - Phi(lb) )        [exact]
////    prob(i, t)                 = exp(y1_log_prob)                [may be 0.0]
////    Z_std_norm(i, t)           = inv_Phi( (1-u)*Phi(lb) + u*Phi(ub) )  [exact, via std_normal_log_qf]
////    Phi_Z(i, t), Bound_U_Phi_Bound_Z(i, t)  back-filled for consistency.
////
//// Lower-tail case (lb <= 0):  work with log Phi directly.
//// Upper-tail case (lb  > 0):  work with survival fns  Phi_bar(x) = Phi(-x)
////                             and recover Z as  -std_normal_log_qf(log q).
//// -----------------------------------------------------------------------------
ALWAYS_INLINE void fn_MVOP_compute_lp_GHK_cols_log_scale_ordinal(   const int t,
                                                                    const std::vector<int> &index,
                                                                    Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
                                                                    Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
                                                                    Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                                                    Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
                                                                    Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,        //// lower bounds (may be -Inf)
                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Upper_Bound_Z,  //// upper bounds (may be +Inf)
                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array,
                                                                    const bool S,
                                                                    const KernelChoice &kernel_choice
) {
  
      const int index_size = index.size();
      ////
      //// ---- 2026-09-22 (assistant, approved change "native exact tails"):
      ////
      //// Previously these rows ALWAYS used the Phi_approx tail pair (sl_log_Phi = cubic-logistic log Phi,
      //// sl_inv_Phi_from_logit = cubic inverse), whatever Phi_type was, while fn_MVOP_row_grads_log_scale
      //// below differentiates them with the EXACT Gaussian densities (log_phi_std_norm, lphZr = Z^2/2 + 0.5 log 2pi).
      //// For Phi_type = "Phi" / inv_Phi_type = "inv_Phi" that was both a splice in the target (exact interior,
      //// Phi_approx tails) and a value/gradient mismatch on these rows. Now:
      ////   use_exact_log_Phi (Phi_type = "Phi"):        log Phi(x) = fast_log_Phi(x)     (erfc body, Mills-ratio tail; exact)
      ////   use_exact_inv_Phi (inv_Phi_type = "inv_Phi"): Z = fast_inv_Phi_from_log_p(log Phi_Z, log(1 - Phi_Z))  (AS241; exact)
      //// The row gradients need no change: they are already the exact derivatives
      ////   d/dx Phi(x) = phi(x),  dZ/dPhi_Z = 1/phi(Z),
      //// i.e. dprob/dmu = (phi(lb) - phi(ub)) / L_tt and -dZ/dmu = [(1-u) phi(lb) + u phi(ub)] / (phi(Z) L_tt).
      //// The Phi_approx setting keeps the old tail functions (and hence its pre-existing mismatch with the exact
      //// row gradients - reported separately, not changed here).
      ////
      const bool use_exact_log_Phi = !kernel_choice.Phi_approx;
      const bool use_exact_inv_Phi = !kernel_choice.inv_Phi_approx;
      
      // for (int ii = 0; ii < index_size; ++ii) {
      //   
      //       const int i = index[ii];
      //       const double lb = Bound_Z(i, t);
      //       const double ub = Upper_Bound_Z(i, t);
      //       const double u  = u_array(i, t);
      //       //// u is strictly in (0, 1) from the nuisance transform:
      //       const double log_u   = sl_log(u, S); // std::log(u);
      //       const double log_1mu = sl_log1m(u, S); // stan::math::log1m(u);
      //       
      //       double log_prob;
      //       double Z;
      //       double log_Phi_lb;   //// log Phi(lb) - for back-filling Bound_U_Phi_Bound_Z
      //       
      //       if (lb <= 0.0) {
      //         
      //             //// ---------------- LOWER-TAIL (or straddling) case ----------------
      //             log_Phi_lb                = std::isinf(lb) ? SL_NEG_INF : sl_log_Phi(lb, S); // stan::math::std_normal_lcdf(lb);
      //             const double log_Phi_ub   = std::isinf(ub) ? 0.0        : sl_log_Phi(ub, S); // stan::math::std_normal_lcdf(ub);
      //             
      //             if (!(log_Phi_lb > SL_NEG_INF)) {
      //               log_prob = log_Phi_ub;
      //             } else {
      //               log_prob = sl_log_diff_exp(log_Phi_ub, log_Phi_lb, S); // stan::math::log_diff_exp(log_Phi_ub, log_Phi_lb);
      //             }
      //             
      //             //// Phi(Z) = (1-u)*Phi(lb) + u*Phi(ub)   (tiny, or fine - either way exact in logs):
      //             const double log_Phi_Z = lse_2(log_1mu + log_Phi_lb, log_u + log_Phi_ub);
      //             const double logit_Phi_Z = log_Phi_Z - sl_log1m_exp(log_Phi_Z, S);
      //             Z = sl_inv_Phi_from_logit(logit_Phi_Z, S);
      //             // Z = stan::math::std_normal_log_qf(log_Phi_Z);
      //             
      //             Phi_Z(i, t) = std::exp(log_Phi_Z);
      //         
      //       } else {
      //         
      //             //// ---------------- UPPER-TAIL case (lb > 0, so ub > 0 too) ----------------
      //             //// survival fns:  Phi_bar(x) = Phi(-x)
      //             const double log_Sb_lb = stan::math::std_normal_lcdf(-lb);
      //             const double log_Sb_ub = std::isinf(ub) ? SL_NEG_INF : stan::math::std_normal_lcdf(-ub);
      //             
      //             if (!(log_Sb_ub > SL_NEG_INF)) {
      //               log_prob = log_Sb_lb;
      //             } else {
      //               log_prob = sl_log_diff_exp(log_Sb_lb, log_Sb_ub, S); // stan::math::log_diff_exp(log_Sb_lb, log_Sb_ub);
      //             }
      //             
      //             //// q = 1 - Phi(Z) = (1-u)*Phi_bar(lb) + u*Phi_bar(ub)  (tiny):
      //             const double log_q = lse_2(log_1mu + log_Sb_lb, log_u + log_Sb_ub);
      //             // Z = -stan::math::std_normal_log_qf(log_q);
      //             const double logit_Phi_Z = log_Phi_Z - sl_log1m_exp(log_Phi_Z, S);
      //             Z = sl_inv_Phi_from_logit(logit_Phi_Z, S);
      //             
      //             log_Phi_lb  = stan::math::log1m_exp(log_Sb_lb);  //// log(1 - Phi_bar(lb)); fine since Phi_bar(lb) tiny... (lb > 0)
      //             Phi_Z(i, t) = 1.0 - std::exp(log_q);
      //         
      //       }
      //       
      //       //// clamp to avoid -Inf propagating into rowwise log-lik sums (matches -700 sentinel convention):
      //       if (!(log_prob > -700.0)) log_prob = -700.0;
      //       
      //       y1_log_prob(i, t) = log_prob;
      //       prob(i, t)        = std::exp(log_prob);   //// may underflow to 0.0 - grads for these rows use the log-scale path
      //       Z_std_norm(i, t)  = Z;
      //       Bound_U_Phi_Bound_Z(i, t) = (log_Phi_lb > SL_NEG_INF) ? std::exp(log_Phi_lb) : 0.0;
      //       
      // }
      for (int ii = 0; ii < index_size; ++ii) {
        
            const int i = index[ii];
            const double lb = Bound_Z(i, t);
            const double ub = Upper_Bound_Z(i, t);
            const double u  = u_array(i, t);
            //// u is strictly in (0, 1) from the nuisance transform:
            const double log_u   = sl_log(u, S);
            const double log_1mu = sl_log1m(u, S);
            
            double log_prob;
            double Z;
            double log_Phi_lb;   //// log Phi(lb) - for back-filling Bound_U_Phi_Bound_Z
            
            if (lb <= 0.0) {
              
                  //// ---------------- LOWER-TAIL (or straddling) case ----------------
                  log_Phi_lb              = std::isinf(lb) ? SL_NEG_INF : (use_exact_log_Phi ? fast_log_Phi(lb) : sl_log_Phi(lb, S));
                  const double log_Phi_ub = std::isinf(ub) ? 0.0        : (use_exact_log_Phi ? fast_log_Phi(ub) : sl_log_Phi(ub, S));
                  
                  if (!(log_Phi_lb > SL_NEG_INF)) {
                    log_prob = log_Phi_ub;
                  } else {
                    log_prob = sl_log_diff_exp(log_Phi_ub, log_Phi_lb, S);
                  }
                  
                  //// Phi(Z) = (1-u)*Phi(lb) + u*Phi(ub)   (tiny, but exact in logs):
                  const double log_Phi_Z = lse_2(log_1mu + log_Phi_lb, log_u + log_Phi_ub, S);
                  ////
                  //// logit(Phi_Z) = log Phi_Z - log(1 - Phi_Z). Phi_Z is TINY here, so
                  //// log1m_exp is stable and the logit is large-negative:
                  ////
                  const double log_1m_Phi_Z = sl_log1m_exp(log_Phi_Z, S);
                  if (use_exact_inv_Phi) {
                    Z = fast_inv_Phi_from_log_p(log_Phi_Z, log_1m_Phi_Z);        //// exact; lower tail driven by log Phi_Z
                  } else {
                    const double logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
                    Z = sl_inv_Phi_from_logit(logit_Phi_Z, S);
                  }
                  ////
                  Phi_Z(i, t) = sl_exp(log_Phi_Z, S);
              
            } else {
              
                  //// ---------------- UPPER-TAIL case (lb > 0, so ub > 0 too) ----------------
                  //// survival fns:  Phi_bar(x) = Phi(-x)
                  const double log_Sb_lb = use_exact_log_Phi ? fast_log_Phi(-lb) : sl_log_Phi(-lb, S);
                  const double log_Sb_ub = std::isinf(ub) ? SL_NEG_INF : (use_exact_log_Phi ? fast_log_Phi(-ub) : sl_log_Phi(-ub, S));
                  
                  if (!(log_Sb_ub > SL_NEG_INF)) {
                    log_prob = log_Sb_lb;
                  } else {
                    log_prob = sl_log_diff_exp(log_Sb_lb, log_Sb_ub, S);
                  }
                  
                  //// q = 1 - Phi(Z) = (1-u)*Phi_bar(lb) + u*Phi_bar(ub)   (tiny):
                  const double log_q = lse_2(log_1mu + log_Sb_lb, log_u + log_Sb_ub, S);
                  ////
                  //// logit(Phi_Z) = log Phi_Z - log(1 - Phi_Z) = log(1 - q) - log(q).
                  //// q is TINY, so log1m_exp(log_q) is stable and the logit is large-POSITIVE:
                  ////
                  const double log_Phi_Z_upper = sl_log1m_exp(log_q, S);
                  if (use_exact_inv_Phi) {
                    Z = fast_inv_Phi_from_log_p(log_Phi_Z_upper, log_q);         //// exact; upper tail driven by log(1 - Phi_Z) = log q
                  } else {
                    const double logit_Phi_Z = log_Phi_Z_upper - log_q;
                    Z = sl_inv_Phi_from_logit(logit_Phi_Z, S);
                  }
                  ////
                  log_Phi_lb  = sl_log1m_exp(log_Sb_lb, S);   //// log(1 - Phi_bar(lb)); fine since Phi_bar(lb) tiny (lb > 0)
                  Phi_Z(i, t) = 1.0 - sl_exp(log_q, S);
              
            }
            
            //// clamp to avoid -Inf propagating into rowwise log-lik sums (matches -700 sentinel convention):
            if (!(log_prob > -700.0)) log_prob = -700.0;
            
            y1_log_prob(i, t) = log_prob;
            prob(i, t)        = sl_exp(log_prob, S);   //// may underflow to 0.0 - grads for these rows use the log-scale path
            Z_std_norm(i, t)  = Z;
            Bound_U_Phi_Bound_Z(i, t) = (log_Phi_lb > SL_NEG_INF) ? sl_exp(log_Phi_lb, S) : 0.0;
            
      }
  
}


//// -----------------------------------------------------------------------------
//// sl_chain_accumulate
////
//// Scalar signed-log version of the GHK chain-rule propagation used by
//// fn_MVOP_compute_cutpoint_grad / fn_MOVP_compute_L_Omega_grad_v3 /
//// fn_MVP_compute_coefficients_grad_v3 (all share the same template):
////
////    gp[0] = gp0 ;  z[0] = z0
////    prod  = z[0] * L(t0+1, t0)
////    gp[1] = dphi[t0+1] * prod
////    for ii = 1 .. :
////        z[ii]    = -dZneg[t0+ii] * prod
////        prod     = sum_{k=0..ii} z[k] * L(t0+ii+1, t0+k)
////        gp[ii+1] = dphi[t0+ii+1] * prod
////
//// and accumulates   sum_ii  gp[ii] * exp(log_W[t0+ii])   in signed-log form,
//// where log_W[tau] is the per-observation accumulation weight:
////   n_class > 1 :  log_prev + log_prob_n_recip + sum_{t' != tau} y1_log_prob(t')
////   n_class == 1:  -y1_log_prob(tau)
//// -----------------------------------------------------------------------------
ALWAYS_INLINE SignedLog sl_chain_accumulate(  const int t0,
                                              const SignedLog gp0,
                                              const SignedLog z0,
                                              const std::vector<SignedLog> &sl_dphi,   //// dprob_t / d(mu_eff_t)
                                              const std::vector<SignedLog> &sl_dZneg,  //// -dZ_t / d(mu_eff_t)   ( >= 0 )
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                              const std::vector<double> &log_W,
                                              const int n_tests,
                                              std::vector<SignedLog> &z_ws,             //// workspace, size >= n_tests
                                              const bool S
) {
  
      SignedLog acc = sl_mul(gp0, sl_make(log_W[t0], 1.0));
      
      if (t0 == n_tests - 1) return acc;
      
      z_ws[0] = z0;
      SignedLog prod = sl_mul(z0, sl_from_val(L_Omega_double(t0 + 1, t0), S));
      SignedLog gp   = sl_mul(sl_dphi[t0 + 1], prod);
      acc = sl_add(acc, sl_mul(gp, sl_make(log_W[t0 + 1], 1.0)), S);
      
      const int n_future = n_tests - t0 - 1;
      
      for (int ii = 1; ii < n_future; ++ii) {
        
            z_ws[ii] = sl_mul(sl_neg(sl_dZneg[t0 + ii]), prod);
            
            prod = sl_zero();
            for (int k = 0; k < ii + 1; ++k) {
              prod = sl_add(prod, sl_mul(z_ws[k], sl_from_val(L_Omega_double(t0 + ii + 1, t0 + k), S)), S);
            }
            
            gp = sl_mul(sl_dphi[t0 + ii + 1], prod);
            acc = sl_add(acc, sl_mul(gp, sl_make(log_W[t0 + ii + 1], 1.0)), S);
        
      }
      
      return acc;
  
}


//// -----------------------------------------------------------------------------
//// fn_MVOP_row_grads_log_scale
////
//// Exact log-scale gradient contributions of ONE (problem) observation, ONE class,
//// for: nuisance u's, coefficients (incl. covariates), L_Omega (diag + off-diag),
//// and cutpoints. Adds directly into the scalar accumulators / u-grad chunk array.
////
//// PRECONDITION: the standard-scale gradient pass must have had this row's
//// contributions ZEROED (see fn in the lp_grad file which zeroes problem rows of
//// common_grad_term_1 / prob_recip / dphi_* / dZ_* / prob etc.), so the additions
//// here do not double count.
////
//// All quantities are rebuilt from the EXACT stored per-class state:
////   y1_log_prob[c] (exact logs), Z_std_norm[c] (exact), Bound_Z[c]/Upper_Bound_Z[c],
////   u_array, y_chunk  -  log(phi(.)) terms are recomputed analytically.
//// -----------------------------------------------------------------------------
ALWAYS_INLINE void fn_MVOP_row_grads_log_scale(   const int n,          //// row within chunk
                                                  const int n_global,   //// global row (for X)
                                                  const int c,
                                                  const bool do_us,
                                                  const bool do_coeff,
                                                  const bool do_corr,
                                                  const bool do_cut,
                                                  const int n_tests,
                                                  // const int n_binary_tests,
                                                  const Eigen::Matrix<int, -1, 1> &ord_idx_of_test,
                                                  const int n_class,
                                                  const int n_covariates_max,
                                                  const Eigen::Ref<const Eigen::Matrix<int, -1, 1>> n_cov_vec_c,
                                                  const Eigen::Ref<const Eigen::Matrix<int, -1, 1>> n_cat_per_ord_test,
                                                  const std::vector<Eigen::Matrix<double, -1, -1>> &X_c,
                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_chunk,
                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array,
                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z_c,
                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Upper_Bound_Z_c,
                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Z_std_norm_c,
                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y1_log_prob_c,
                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double_c,
                                                  const double log_prev_nc,          //// log prev(pop(n), c);  ignored if n_class == 1
                                                  const double log_prob_n_recip_n,   //// -log_lik(n)
                                                  //// outputs (accumulated into):
                                                  Eigen::Ref<Eigen::Matrix<double, -1, -1>> u_grad_array_CM_chunk,
                                                  Eigen::Ref<Eigen::Matrix<double, -1, -1>> beta_grad_array_c,
                                                  Eigen::Ref<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array_c,
                                                  Eigen::Ref<Eigen::Matrix<double, -1, -1>> cutpoint_grad_array_c,
                                                  const bool S
) {
  
      //// ---------------- per-test signed-log seeds + weights ----------------
      std::vector<SignedLog> sl_dphi(n_tests);     //// dprob_t / d(mu_eff_t)      == dphi_over_L
      std::vector<SignedLog> sl_dZneg(n_tests);    //// -dZ_t / d(mu_eff_t)        == dZ_dmu_neg
      std::vector<SignedLog> sl_dphibz(n_tests);   //// L_Omega-diag prob seed     == dphi_times_bz
      std::vector<SignedLog> sl_dZbz(n_tests);     //// L_Omega-diag z    seed     == dZ_times_bz
      std::vector<double>    lphZr(n_tests);       //// log( 1 / phi(Z_t) )
      std::vector<double>    lph_lb(n_tests);      //// log phi(lower bound)  (binary: log phi(Bound_Z))
      std::vector<double>    lph_ub(n_tests);      //// log phi(upper bound)  (ordinal only)
      std::vector<double>    log_W(n_tests);
      std::vector<SignedLog> z_ws(n_tests);        //// chain workspace
      
      double S_ylp = 0.0;
      for (int t = 0; t < n_tests; ++t) S_ylp += y1_log_prob_c(n, t);
      
      for (int t = 0; t < n_tests; ++t) {
        if (n_class > 1) {
          log_W[t] = log_prev_nc + log_prob_n_recip_n + (S_ylp - y1_log_prob_c(n, t));
        } else {
          log_W[t] = -y1_log_prob_c(n, t);
        }
      }
      
      // double S = 0.0;
      // for (int t = 0; t < n_tests; ++t) S += y1_log_prob_c(n, t);
      // 
      // for (int t = 0; t < n_tests; ++t) {
      //   if (n_class > 1) {
      //     log_W[t] = log_prev_nc + log_prob_n_recip_n + (S - y1_log_prob_c(n, t));
      //   } else {
      //     log_W[t] = -y1_log_prob_c(n, t);
      //   }
      // }
      
      for (int t = 0; t < n_tests; ++t) {
            
            const double L_tt      = L_Omega_double_c(t, t);
            // const double log_L_recip = -std::log(L_tt);   //// L(t,t) > 0
            const double log_L_recip = -sl_log(L_tt, S);
            const double Z = Z_std_norm_c(n, t);
            lphZr[t] = 0.5 * Z * Z + HALF_LOG_TWO_PI;
            
            // if (t < n_binary_tests) {
            if (ord_idx_of_test(t) < 0) {
              
                  //// ---------------- binary test ----------------
                  const double BZ    = Bound_Z_c(n, t);
                  const double y     = y_chunk(n, t);
                  const double ysign = 2.0 * y - 1.0;
                  const double lphB  = log_phi_std_norm(BZ);
                  lph_lb[t] = lphB;
                  lph_ub[t] = SL_NEG_INF;
                  
                  sl_dphi[t]  = sl_make(lphB + log_L_recip, ysign);                    //// == dprob_t/dmu (see derivation in file header of lp_grad)
                  const double ymy = y - ysign * u_array(n, t);                        //// y - y_sign*u  >= 0 always
                  // sl_dZneg[t] = (ymy > 0.0) ? sl_make(std::log(ymy) + lphZr[t] + lphB + log_L_recip, 1.0) : sl_zero();
                  // sl_dphibz[t] = sl_mul(sl_dphi[t],  sl_from_val(BZ));
                  // sl_dZbz[t]   = sl_mul(sl_dZneg[t], sl_from_val(BZ));
                  sl_dZneg[t] = (ymy > 0.0) ? sl_make(sl_log(ymy, S) + lphZr[t] + lphB + log_L_recip, 1.0) : sl_zero();
                  sl_dphibz[t] = sl_mul(sl_dphi[t],  sl_from_val(BZ, S));
                  sl_dZbz[t]   = sl_mul(sl_dZneg[t], sl_from_val(BZ, S));
              
            } else {
              
                  //// ---------------- ordinal test ----------------
                  // const double lb = Bound_Z_c(n, t);
                  // const double ub = Upper_Bound_Z_c(n, t);
                  // const double u  = u_array(n, t);
                  // const double log_u   = std::log(u);
                  // const double log_1mu = stan::math::log1m(u);
                  // lph_lb[t] = log_phi_std_norm(lb);   //// -Inf if lb = -Inf
                  // lph_ub[t] = log_phi_std_norm(ub);   //// -Inf if ub = +Inf
                  // 
                  // //// dprob/dmu = (phi(lb) - phi(ub)) / L_tt :
                  // sl_dphi[t] = sl_add( sl_make(lph_lb[t] + log_L_recip,  1.0),
                  //                      sl_make(lph_ub[t] + log_L_recip, -1.0) );
                  // 
                  // //// -dZ/dmu = [ (1-u)*phi(lb) + u*phi(ub) ] / (phi(Z) * L_tt)  ( >= 0 ):
                  // const double lnum = lse_2(log_1mu + lph_lb[t], log_u + lph_ub[t]);
                  // sl_dZneg[t] = sl_make(lnum + lphZr[t] + log_L_recip, 1.0);
                  // 
                  // //// dphi_times_bz = ( phi(lb)*lb - phi(ub)*ub ) / L_tt   (x*phi(x) -> 0 as x -> +-Inf):
                  // SignedLog term_lb = std::isinf(lb) ? sl_zero() : sl_make(lph_lb[t] + std::log(std::abs(lb) > 0.0 ? std::abs(lb) : 1e-300) + log_L_recip, (lb < 0.0) ? -1.0 : 1.0);
                  // SignedLog term_ub = std::isinf(ub) ? sl_zero() : sl_make(lph_ub[t] + std::log(std::abs(ub) > 0.0 ? std::abs(ub) : 1e-300) + log_L_recip, (ub < 0.0) ? -1.0 : 1.0);
                  // sl_dphibz[t] = sl_add(term_lb, sl_neg(term_ub));
                  // 
                  // //// dZ_times_bz = [ (1-u)*phi(lb)*lb + u*phi(ub)*ub ] / (phi(Z) * L_tt):
                  // SignedLog n_lb = std::isinf(lb) ? sl_zero() : sl_make(log_1mu + lph_lb[t] + std::log(std::abs(lb) > 0.0 ? std::abs(lb) : 1e-300), (lb < 0.0) ? -1.0 : 1.0);
                  // SignedLog n_ub = std::isinf(ub) ? sl_zero() : sl_make(log_u   + lph_ub[t] + std::log(std::abs(ub) > 0.0 ? std::abs(ub) : 1e-300), (ub < 0.0) ? -1.0 : 1.0);
                  // sl_dZbz[t] = sl_mul( sl_add(n_lb, n_ub), sl_make(lphZr[t] + log_L_recip, 1.0) );
                  
                  const double lb = Bound_Z_c(n, t);
                  const double ub = Upper_Bound_Z_c(n, t);
                  const double u  = u_array(n, t);
                  const double log_u   = sl_log(u, S);
                  const double log_1mu = sl_log1m(u, S);
                  lph_lb[t] = log_phi_std_norm(lb);   //// -Inf if lb = -Inf
                  lph_ub[t] = log_phi_std_norm(ub);   //// -Inf if ub = +Inf
                
                  sl_dphi[t] = sl_add( sl_make(lph_lb[t] + log_L_recip,  1.0),
                                       sl_make(lph_ub[t] + log_L_recip, -1.0), S );
                  ////
                  const double lnum = lse_2(log_1mu + lph_lb[t], log_u + lph_ub[t], S);
                  sl_dZneg[t] = sl_make(lnum + lphZr[t] + log_L_recip, 1.0);
                  ////
                  SignedLog term_lb = std::isinf(lb) ? sl_zero() : sl_make(lph_lb[t] + sl_log(std::abs(lb) > 0.0 ? std::abs(lb) : 1e-300, S) + log_L_recip, (lb < 0.0) ? -1.0 : 1.0);
                  SignedLog term_ub = std::isinf(ub) ? sl_zero() : sl_make(lph_ub[t] + sl_log(std::abs(ub) > 0.0 ? std::abs(ub) : 1e-300, S) + log_L_recip, (ub < 0.0) ? -1.0 : 1.0);
                  sl_dphibz[t] = sl_add(term_lb, sl_neg(term_ub), S);
                  ////
                  SignedLog n_lb = std::isinf(lb) ? sl_zero() : sl_make(log_1mu + lph_lb[t] + sl_log(std::abs(lb) > 0.0 ? std::abs(lb) : 1e-300, S), (lb < 0.0) ? -1.0 : 1.0);
                  SignedLog n_ub = std::isinf(ub) ? sl_zero() : sl_make(log_u   + lph_ub[t] + sl_log(std::abs(ub) > 0.0 ? std::abs(ub) : 1e-300, S), (ub < 0.0) ? -1.0 : 1.0);
                  sl_dZbz[t] = sl_mul( sl_add(n_lb, n_ub, S), sl_make(lphZr[t] + log_L_recip, 1.0) );
                  
              
            }
        
      }
      
      //// ---------------- nuisance / u grads ----------------
      //// prob_t does NOT depend on u_t directly (binary and ordinal alike);
      //// u_t enters only through Z_t:   dZ_t/du_t = prob_t / phi(Z_t)   ( > 0 ).
      // if (do_us) {
      //     for (int t = 0; t < n_tests - 1; ++t) {
      //       const SignedLog z0 = sl_make(y1_log_prob_c(n, t) + lphZr[t], 1.0);
      //       const SignedLog acc = sl_chain_accumulate(t, sl_zero(), z0, sl_dphi, sl_dZneg,
      //                                                 L_Omega_double_c, log_W, n_tests, z_ws);
      //       u_grad_array_CM_chunk(n, t) += sl_val(acc);
      //     }
      //     //// t = n_tests-1 has no future tests -> zero gradient contribution.
      // }
      if (do_us) {
          for (int t = 0; t < n_tests - 1; ++t) {
            const SignedLog z0 = sl_make(y1_log_prob_c(n, t) + lphZr[t], 1.0);
            const SignedLog acc = sl_chain_accumulate(t, sl_zero(), z0, sl_dphi, sl_dZneg,
                                                      L_Omega_double_c, log_W, n_tests, z_ws, S);
            u_grad_array_CM_chunk(n, t) += sl_val(acc, S);
          }
      }
      
      //// ---------------- coefficient grads (incl. covariates) ----------------
      //// mu_t = X*beta + inc, so d/dbeta_k = X(n,k) * d/dmu_t :
      // if (do_coeff) {
      //     for (int t = 0; t < n_tests; ++t) {
      //       const SignedLog acc_mu = sl_chain_accumulate(t, sl_dphi[t], sl_neg(sl_dZneg[t]), sl_dphi, sl_dZneg,
      //                                                    L_Omega_double_c, log_W, n_tests, z_ws);
      //       const double val_mu = sl_val(acc_mu);
      //       if (n_covariates_max > 1) {
      //         for (int k = 0; k < n_cov_vec_c(t); ++k) {
      //           beta_grad_array_c(k, t) += X_c[t](n_global, k) * val_mu;
      //         }
      //       } else {
      //         beta_grad_array_c(0, t) += val_mu;
      //       }
      //     }
      // }
      if (do_coeff) {
          for (int t = 0; t < n_tests; ++t) {
            const SignedLog acc_mu = sl_chain_accumulate(t, sl_dphi[t], sl_neg(sl_dZneg[t]), sl_dphi, sl_dZneg,
                                                         L_Omega_double_c, log_W, n_tests, z_ws, S);
            const double val_mu = sl_val(acc_mu, S);
            if (n_covariates_max > 1) {
              for (int k = 0; k < n_cov_vec_c(t); ++k) {
                beta_grad_array_c(k, t) += X_c[t](n_global, k) * val_mu;
              }
            } else {
              beta_grad_array_c(0, t) += val_mu;
            }
          }
      }
      
      //// ---------------- L_Omega grads ----------------
      // if (do_corr) {
      //     //// diagonals (seeds:  gp0 = dphi_times_bz(t1),  z0 = -dZ_times_bz(t1) ):
      //     for (int t1 = 0; t1 < n_tests; ++t1) {
      //       const SignedLog acc = sl_chain_accumulate(t1, sl_dphibz[t1], sl_neg(sl_dZbz[t1]), sl_dphi, sl_dZneg,
      //                                                 L_Omega_double_c, log_W, n_tests, z_ws);
      //       U_Omega_grad_array_c(t1, t1) += sl_val(acc);
      //     }
      //     //// off-diagonals (seeds:  gp0 = dphi_over_L(t1)*Z(t2),  z0 = -dZ_dmu_neg(t1)*Z(t2) ):
      //     for (int t1 = 1; t1 < n_tests; ++t1) {
      //       for (int t2 = 0; t2 < t1; ++t2) {
      //         const SignedLog Z_t2 = sl_from_val(Z_std_norm_c(n, t2));
      //         const SignedLog gp0  = sl_mul(sl_dphi[t1], Z_t2);
      //         const SignedLog z0   = sl_mul(sl_neg(sl_dZneg[t1]), Z_t2);
      //         const SignedLog acc  = sl_chain_accumulate(t1, gp0, z0, sl_dphi, sl_dZneg,
      //                                                    L_Omega_double_c, log_W, n_tests, z_ws);
      //         U_Omega_grad_array_c(t1, t2) += sl_val(acc);
      //       }
      //     }
      // }
      if (do_corr) {
          //// diagonals (seeds:  gp0 = dphi_times_bz(t1),  z0 = -dZ_times_bz(t1) ):
          for (int t1 = 0; t1 < n_tests; ++t1) {
            const SignedLog acc = sl_chain_accumulate(t1, sl_dphibz[t1], sl_neg(sl_dZbz[t1]), sl_dphi, sl_dZneg,
                                                      L_Omega_double_c, log_W, n_tests, z_ws, S);
            U_Omega_grad_array_c(t1, t1) += sl_val(acc, S);
          }
          //// off-diagonals (seeds:  gp0 = dphi_over_L(t1)*Z(t2),  z0 = -dZ_dmu_neg(t1)*Z(t2) ):
          for (int t1 = 1; t1 < n_tests; ++t1) {
            for (int t2 = 0; t2 < t1; ++t2) {
              const SignedLog Z_t2 = sl_from_val(Z_std_norm_c(n, t2), S);
              const SignedLog gp0  = sl_mul(sl_dphi[t1], Z_t2);
              const SignedLog z0   = sl_mul(sl_neg(sl_dZneg[t1]), Z_t2);
              const SignedLog acc  = sl_chain_accumulate(t1, gp0, z0, sl_dphi, sl_dZneg,
                                                         L_Omega_double_c, log_W, n_tests, z_ws, S);
              U_Omega_grad_array_c(t1, t2) += sl_val(acc, S);
            }
          }
      }
      // //// ---------------- cutpoint grads ----------------
      // //// each observation touches at most 2 cutpoints of its own test:
      // ////   y = j+1  ->  C_j is the UPPER bound;   y = j+2  ->  C_j is the LOWER bound.
      // if (do_cut) {
      //     for (int t = 0; t < n_tests; ++t) {
      //       
      //       const int t_ord = ord_idx_of_test(t);
      //       if (t_ord < 0) continue;   //// binary slot -> no cutpoints
      //       
      //   // if (do_cut) {
      //   //     for (int t = n_binary_tests; t < n_tests; ++t) {
      //   //       
      //   //           const int t_ord = t - n_binary_tests;
      //             const int K_t   = n_cat_per_ord_test(t_ord);
      //             const int y_i   = static_cast<int>(y_chunk(n, t));
      //             const double log_L_recip = -std::log(L_Omega_double_c(t, t));
      //             const double log_u   = std::log(u_array(n, t));
      //             const double log_1mu = stan::math::log1m(u_array(n, t));
      //             
      //             if (y_i < K_t) {  //// C_{y-1} is the upper bound:
      //                 const int j = y_i - 1;
      //                 const SignedLog gp0 = sl_make(lph_ub[t] + log_L_recip, 1.0);
      //                 const SignedLog z0  = sl_make(log_u + lph_ub[t] + lphZr[t] + log_L_recip, 1.0);
      //                 const SignedLog acc = sl_chain_accumulate(t, gp0, z0, sl_dphi, sl_dZneg,
      //                                                           L_Omega_double_c, log_W, n_tests, z_ws);
      //                 cutpoint_grad_array_c(j, t_ord) += sl_val(acc);
      //             }
      //             if (y_i >= 2) {   //// C_{y-2} is the lower bound:
      //                 const int j = y_i - 2;
      //                 const SignedLog gp0 = sl_make(lph_lb[t] + log_L_recip, -1.0);
      //                 const SignedLog z0  = sl_make(log_1mu + lph_lb[t] + lphZr[t] + log_L_recip, 1.0);
      //                 const SignedLog acc = sl_chain_accumulate(t, gp0, z0, sl_dphi, sl_dZneg,
      //                                                           L_Omega_double_c, log_W, n_tests, z_ws);
      //                 cutpoint_grad_array_c(j, t_ord) += sl_val(acc);
      //             }
      //         
      //       }
      // }
      //// ---------------- cutpoint grads ----------------
      //// each observation touches at most 2 cutpoints of its own test:
      ////   y = j+1  ->  C_j is the UPPER bound;   y = j+2  ->  C_j is the LOWER bound.
      if (do_cut) {
        
          for (int t = 0; t < n_tests; ++t) {
            
                const int t_ord = ord_idx_of_test(t);
                if (t_ord < 0) continue;   //// binary slot -> no cutpoints
                ////
                const int K_t   = n_cat_per_ord_test(t_ord);
                const int y_i   = static_cast<int>(y_chunk(n, t));
                const double log_L_recip = -sl_log(L_Omega_double_c(t, t), S);
                const double log_u       =  sl_log(u_array(n, t), S);
                const double log_1mu     =  sl_log1m(u_array(n, t), S);
                
                if (y_i < K_t) {  //// C_{y-1} is the upper bound:
                    const int j = y_i - 1;
                    const SignedLog gp0 = sl_make(lph_ub[t] + log_L_recip, 1.0);
                    const SignedLog z0  = sl_make(log_u + lph_ub[t] + lphZr[t] + log_L_recip, 1.0);
                    const SignedLog acc = sl_chain_accumulate(t, gp0, z0, sl_dphi, sl_dZneg,
                                                              L_Omega_double_c, log_W, n_tests, z_ws, S);
                    cutpoint_grad_array_c(j, t_ord) += sl_val(acc, S);
                }
                if (y_i >= 2) {   //// C_{y-2} is the lower bound:
                    const int j = y_i - 2;
                    const SignedLog gp0 = sl_make(lph_lb[t] + log_L_recip, -1.0);
                    const SignedLog z0  = sl_make(log_1mu + lph_lb[t] + lphZr[t] + log_L_recip, 1.0);
                    const SignedLog acc = sl_chain_accumulate(t, gp0, z0, sl_dphi, sl_dZneg,
                                                              L_Omega_double_c, log_W, n_tests, z_ws, S);
                    cutpoint_grad_array_c(j, t_ord) += sl_val(acc, S);
                }
            
          }
        
      }
  
}











