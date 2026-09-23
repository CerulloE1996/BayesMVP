
#pragma once

#ifndef STAN_MATH_PRIM_META_MVP_DOUBLE_FNS_HPP
#define STAN_MATH_PRIM_META_MVP_DOUBLE_FNS_HPP



 

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/prim/err/check_pos_definite.hpp>
#include <stan/math/prim/err/check_square.hpp>
#include <stan/math/prim/err/check_symmetric.hpp>


#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/sqrt.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/transpose.hpp>
#include <stan/math/prim/fun/dot_product.hpp>
#include <stan/math/prim/fun/norm2.hpp>
#include <stan/math/prim/fun/diagonal.hpp>
#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/eigenvalues_sym.hpp>
#include <stan/math/prim/fun/diag_post_multiply.hpp>

#include <stan/math/prim/fun/log_inv_logit.hpp>
#include <stan/math/prim/fun/fma.hpp>


#include <stan/math/prim/prob/multi_normal_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/lkj_corr_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/weibull_lpdf.hpp>
#include <stan/math/prim/prob/gamma_lpdf.hpp>
#include <stan/math/prim/prob/beta_lpdf.hpp>



 

#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/KroneckerProduct>
 

#include <immintrin.h>
 
using namespace Eigen;

 
 
#define EIGEN_DONT_PARALLELIZE
 
 


 
 
 

template <typename T, int N>
inline std::array<Eigen::Matrix<T, -1, -1>, N> array_of_mats( int rows, 
                                                              int cols) {
  
       std::array<Eigen::Matrix<T, -1, -1>, N> arr;
       for (int c = 0; c < N; ++c) {
         arr[c] = Eigen::Matrix<T, -1, -1>::Zero(rows, cols);
       }
       
       return arr;
 
}


// ---------------------------------------------------------------------------------------------
//  Relative-accurate normal tail kernels for the GHK log-scale branches.
//  Include AFTER the fast_and_approx_*_fns.hpp headers (uses fast_exp_1_*, fast_log_1_*, fast_Phi_*,
//  fast_inv_Phi_wo_checks_case_{1,2a,2b}_* and _mm256_abs_pd / _mm512_abs_pd).
//
//  Everything is built on ONE function, the Mills ratio
//        R(x) = (1 - Phi(x)) / phi(x)  =  sqrt(pi/2) * erfcx(x / sqrt(2)),      x >= 0,
//  which is smooth, O(1/x) and has no cancellation. Then:
//        Phi(x)          = phi(x) * R(-x)                               (x < 0)
//        log Phi(x)      = -x^2/2 - 0.5*log(2*pi) + log R(-x)           (x < 0)
//        log(1 - Phi(x)) = log Phi(-x)
//        d/dx log Phi(x) = phi(x)/Phi(x) = 1 / R(-x)                     (x < 0)
//        inv_Phi from log p : AS241 tail case needs only r = sqrt(-log(min(p, 1-p)))
//        dZ/dlog_p    =  Phi(Z)/phi(Z)     = exp(log_p    + Z^2/2 + 0.5 log 2pi)  (=  R(-Z) for Z<0, stable side)
//        dZ/dlog_1m_p = -(1-Phi(Z))/phi(Z) = -exp(log_1m_p + Z^2/2 + 0.5 log 2pi)  (= -R( Z) for Z>0, stable side)
//
//  R(x) is evaluated with the Laplace continued fraction  R(x) = 1/(x + 1/(x + 2/(x + 3/(x + ...))))
//  by backward recurrence of depth MILLS_CF_DEPTH. Depth 40 is full double precision for x >= 2.5
//  (run normal_tail_self_test() once; lower the depth after benchmarking if you like — the tail
//  path is only taken for |x| > NORMAL_TAIL_THRESH, and a rational minimax in 1/x can replace the
//  fraction later; the self-test validates any replacement).
//
//  Body (|x| <= NORMAL_TAIL_THRESH) keeps your existing fast_Phi (Abramowitz-Stegun). Its absolute
//  error 7.5e-8 is a relative error of 1.2e-5 at the threshold and smaller inside; the tail side is
//  exact, so the join is continuous to ~1e-5 in log Phi. The remaining tail error is your fast_log's
//  (~1e-8); swap in the double-precision log constant if you want machine precision.
// ---------------------------------------------------------------------------------------------

static constexpr double NORMAL_TAIL_THRESH  = 2.5;
static constexpr int    MILLS_CF_DEPTH      = 40;
static constexpr double NEG_HALF_LOG_2PI    = -0.91893853320467274178;   // -0.5*log(2*pi)
static constexpr double INV_SQRT_2PI        =  0.39894228040143267794;   //  1/sqrt(2*pi)

// =============================================================================================
//  SCALAR
// =============================================================================================
inline double fast_mills_ratio_tail(const double x) {                 // x >= NORMAL_TAIL_THRESH
  
      double f = x;
      for (int k = MILLS_CF_DEPTH; k >= 1; --k) f = x + static_cast<double>(k) / f;
      return 1.0 / f;
      
}




inline double fast_log_Phi_tail_only(const double x) {                // x < -NORMAL_TAIL_THRESH
      return -0.5 * x * x + NEG_HALF_LOG_2PI + std::log(fast_mills_ratio_tail(-x));
}




// scalar body uses std::erfc directly (already relative-accurate); vector bodies use fast_Phi
inline double fast_log_Phi(const double x) {
  
      if (x < -NORMAL_TAIL_THRESH) return fast_log_Phi_tail_only(x);
      return std::log(0.5 * std::erfc(-x * M_SQRT1_2));
      
}




inline double fast_log_1m_Phi(const double x) { return fast_log_Phi(-x); }




inline double fast_dlog_Phi_dx(const double x) {                      // phi(x)/Phi(x)
  
      if (x < -NORMAL_TAIL_THRESH) return 1.0 / fast_mills_ratio_tail(-x);
      return INV_SQRT_2PI * std::exp(-0.5 * x * x) / (0.5 * std::erfc(-x * M_SQRT1_2));
      
}
// d/dx log(1 - Phi(x)) = -fast_dlog_Phi_dx(-x)




// AS241 rationals (identical constants to your AVX code)
inline double fast_inv_Phi_case_1(const double q) {                   // |q| <= 0.425
  
      const double r = 0.180625 - q * q;
      double num = 2509.0809287301226727 * r + 33430.575583588128105;
      num = num * r + 67265.770927008700853;  num = num * r + 45921.953931549871457;
      num = num * r + 13731.693765509461125;  num = num * r + 1971.5909503065514427;
      num = num * r + 133.14166789178437745;  num = num * r + 3.387132872796366608;
      double den = 5226.495278852854561 * r + 28729.085735721942674;
      den = den * r + 39307.89580009271061;   den = den * r + 21213.794301586595867;
      den = den * r + 5394.1960214247511077;  den = den * r + 687.1870074920579083;
      den = den * r + 42.313330701600911252;  den = den * r + 1.0;
      return q * num / den;
      
}




inline double fast_inv_Phi_case_2a(double r) {                        // r <= 5
  
      r -= 1.60;
      double num = 0.00077454501427834140764;
      num = r * num + 0.0227238449892691845833; num = r * num + 0.24178072517745061177;
      num = r * num + 1.27045825245236838258;   num = r * num + 3.64784832476320460504;
      num = r * num + 5.7694972214606914055;    num = r * num + 4.6303378461565452959;
      num = r * num + 1.42343711074968357734;
      double den = 0.00000000105075007164441684324;
      den = r * den + 0.0005475938084995344946; den = r * den + 0.0151986665636164571966;
      den = r * den + 0.14810397642748007459;   den = r * den + 0.68976733498510000455;
      den = r * den + 1.6763848301838038494;    den = r * den + 2.05319162663775882187;
      den = r * den + 1.0;
      return num / den;
      
}




inline double fast_inv_Phi_case_2b(double r) {                        // r > 5
  
      r -= 5.0;
      double num = 0.000000201033439929228813265;
      num = r * num + 0.0000271155556874348757815; num = r * num + 0.0012426609473880784386;
      num = r * num + 0.026532189526576123093;     num = r * num + 0.29656057182850489123;
      num = r * num + 1.7848265399172913358;       num = r * num + 5.4637849111641143699;
      num = r * num + 6.6579046435011037772;
      double den = 0.00000000000000204426310338993978564;
      den = r * den + 0.00000014215117583164458887; den = r * den + 0.000018463183175100546818;
      den = r * den + 0.0007868691311456132591;     den = r * den + 0.0148753612908506148525;
      den = r * den + 0.13692988092273580531;       den = r * den + 0.59983220655588793769;
      den = r * den + 1.0;
      return num / den;
      
}




// inv_Phi given log(p) and log(1-p). No exp/log of an extreme probability anywhere.
inline double fast_inv_Phi_from_log_p(const double log_p, 
                                      const double log_1m_p) {
  
      const double p = std::exp(log_p);                                 // only consumed by the central case
      const double q = p - 0.5;
      if (std::fabs(q) <= 0.425) return fast_inv_Phi_case_1(q);
      const double r = std::sqrt(-std::min(log_p, log_1m_p));           // sqrt(-log(min(p, 1-p)))
      const double v = (r <= 5.0) ? fast_inv_Phi_case_2a(r) : fast_inv_Phi_case_2b(r);
      return (log_p < log_1m_p) ? -v : v;                                // p < 0.5 -> Z < 0
      
}




// Total derivative dZ/dv through the STABLE side: log_p when Z<0, log_1m_p when Z>=0.
// Both are exact identities (p = exp(log_p) = 1 - exp(log_1m_p)); the blend only picks the
// evaluation that cannot overflow.
inline double fast_dZ_dv_from_log_p(const double Z, 
                                    const double log_p, 
                                    const double log_1m_p,
                                    const double dlog_p_dv, 
                                    const double dlog_1m_p_dv) {
  
      const double log_inv_phi = 0.5 * Z * Z - NEG_HALF_LOG_2PI;           // log(1/phi(Z))
      if (Z < 0.0) return  std::exp(log_p    + log_inv_phi) * dlog_p_dv;       //  (Phi/phi)     * dlog_p/dv
      else         return -std::exp(log_1m_p + log_inv_phi) * dlog_1m_p_dv;   // -((1-Phi)/phi) * dlog_1m_p/dv

}




////
//// ---- Exact normal tails for the autodiff (stan::math::var) reference paths:
////
//// Added 2026-09-22 (assistant, approved change "native exact tails"). Used by the Phi_type = "Phi" branches of
//// MVP_lp_grad_AD_fns.hpp, std_MVP_lp_grad_AD_fns.hpp, MVOP_lp_grad_AD_fns.hpp and LC_LT_lp_grad_AD_fns.hpp,
//// which previously switched to the Phi_approx (cubic-logistic) tails beyond overflow_threshold / underflow_threshold
//// whatever Phi_type was. The values come from the SAME double kernels as the manual-gradient paths (fast_log_Phi,
//// fast_inv_Phi_from_log_p above), and the derivatives are supplied analytically through precomputed_gradients:
////
////   log Phi(x):                 d/dx log Phi(x) = phi(x) / Phi(x)                    (= fast_dlog_Phi_dx(x) = 1 / R(-x) for x < -2.5)
////   Z = Phi^{-1}(p), p = exp(log_p):
////                               dZ/dlog_p    =  p / phi(Z)       =  exp(log_p    + Z^2/2 + 0.5 log(2 pi))
////                               dZ/dlog_1m_p = -(1 - p) / phi(Z)  = -exp(log_1m_p + Z^2/2 + 0.5 log(2 pi))
////
//// log_p and log_1m_p are two representations of the same probability, so the total derivative of Z may be routed
//// through EITHER of them; we route it through the side that cannot overflow (log_p when Z < 0, log_1m_p when Z >= 0),
//// exactly as fast_dZ_dv_from_log_p does. This is only valid when the caller computes log_p and log_1m_p as
//// consistent functions of the same upstream vars (every call site in the AD files does: one is log1m_exp of the other,
//// or both are exact expressions in the same (Bound_Z, u)).
////
//// Requires stan/math/rev (var_fns.hpp, which includes it, is always included before this file).
////
inline stan::math::var log_Phi_exact_var(const stan::math::var &x_var) {

      const double x_value = x_var.val();
      const double log_Phi_value = fast_log_Phi(x_value);
      const double derivative_of_log_Phi_wrt_x = fast_dlog_Phi_dx(x_value);
      return stan::math::precomputed_gradients(log_Phi_value,
                                               std::vector<stan::math::var>{x_var},
                                               std::vector<double>{derivative_of_log_Phi_wrt_x});

}




inline stan::math::var inv_Phi_from_log_p_exact_var(const stan::math::var &log_p_var,
                                                    const stan::math::var &log_1m_p_var) {

      const double log_p_value    = log_p_var.val();
      const double log_1m_p_value = log_1m_p_var.val();
      const double Z_value        = fast_inv_Phi_from_log_p(log_p_value, log_1m_p_value);
      const double log_inv_phi_Z  = 0.5 * Z_value * Z_value - NEG_HALF_LOG_2PI;    // log(1/phi(Z))
      if (Z_value < 0.0) {
            const double derivative_of_Z_wrt_log_p = std::exp(log_p_value + log_inv_phi_Z);
            return stan::math::precomputed_gradients(Z_value,
                                                     std::vector<stan::math::var>{log_p_var},
                                                     std::vector<double>{derivative_of_Z_wrt_log_p});
      } else {
            const double derivative_of_Z_wrt_log_1m_p = -std::exp(log_1m_p_value + log_inv_phi_Z);
            return stan::math::precomputed_gradients(Z_value,
                                                     std::vector<stan::math::var>{log_1m_p_var},
                                                     std::vector<double>{derivative_of_Z_wrt_log_1m_p});
      }

}


 
 
//// -------------------------------------------------------------------------------------------------------------------------------------------------------------
 
 
// The scalar version for comparison
inline double test_simple_double(const double x) {
 
   const double res = 2.0*x;
   return res;
 
} 
  
 
 
 // Other Misc. functions -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 
 
inline bool Eigen_any_NaNs_process_return(Eigen::Ref<Eigen::Matrix<double, -1, -1>>  x_Ref) {
   return !((x_Ref.array() == x_Ref.array()).all());  // NaN  =/=  itself, so this works to check for NaNs
 }
 
 
 
inline bool Eigen_any_NaNs(Eigen::Matrix<double, -1, -1>  &x_R_val) {
   Eigen::Ref<Eigen::Matrix<double, -1, -1>> x_Ref(x_R_val);  // Directly create Eigen::Ref from R-value
   return  Eigen_any_NaNs_process_return(x_Ref);
 }


 inline bool Eigen_any_NaNs(Eigen::Matrix<double, -1, -1> x_L_val) {
   Eigen::Ref<Eigen::Matrix<double, -1, -1>> x_Ref(x_L_val);  // Directly create Eigen::Ref from L-value
   return  Eigen_any_NaNs_process_return(x_Ref);
 }


template <typename MatrixType>
inline bool Eigen_any_NaNs(Eigen::Ref<Eigen::Block<MatrixType, -1, -1>> x_Ref) {
  return  Eigen_any_NaNs_process_return(x_Ref);
}


inline bool Eigen_any_NaNs(Eigen::Array<double, -1, -1> x) {
  Eigen::Matrix<double, -1, -1>  x_matrix = x.matrix();
  return  Eigen_any_NaNs_process_return(x_matrix);
}



// 
// inline Eigen::Matrix<double, -1, 1> fn_colvec_double(Eigen::Array<double, -1, 1> x,
//                                                      const std::string &fn,
//                                                      const std::string &vect_type = "Eigen",
//                                                      const bool &skip_checks = false) {
//   
//   Eigen::Matrix<double, -1, 1>  x_matrix = x.matrix();
//   return fn_return_colvec_Ref_double_all(x_matrix, fn, vect_type, skip_checks);
//   
//   
// }



// 
// inline bool Eigen_any_NaNs(Eigen::Array<double, -1, -1> x_array) {
//  return !((x_array == x_array).all());
// }
// 
// 
// 
// template <typename BlockType>
// inline bool Eigen_any_NaNs(Eigen::Ref<Eigen::Block<BlockType>> &block) {
//   return !((block.array() == block.array()).all());
// }
// 




// template <typename Derived>
// inline bool Eigen_any_NaNs(const Eigen::DenseBase<Derived>& x) {
//   return !((x.array() == x.array()).all());  // NaN check
// }
// 
// 
// inline bool Eigen_any_NaNs(Eigen::Block<Eigen::MatrixXd>& x) {
//   return !((x.array() == x.array()).all());  // NaN check for non-const block expressions
// }



 ////////////// douuble fn's -------------------------------------------------------------------------------------------------

 

  
 
 
//// 
//// Function to manually construct a cutpoint vector from raw unconstrained parameters - using exp() / log-differences.
////
inline Eigen::Matrix<double, -1, 1>	  construct_C( const Eigen::Matrix<double, -1, 1> &C_raw_vec, // parameter - log_diffs
                                                   bool softplus) {
  
       const int n_total_cutpoints = C_raw_vec.size();
       Eigen::Matrix<double, -1, 1> C_vec(n_total_cutpoints);
       C_vec(0) = C_raw_vec(0); // first cutpoint is same as first raw_C/log_diff
      
       if (softplus == true) { 
           Eigen::Matrix<double, -1, 1> softplus_C_vec = stan::math::log1p_exp(C_raw_vec.segment(1, n_total_cutpoints - 1));
           for (int k = 2; k <= n_total_cutpoints; ++k) {
               C_vec(k - 1) = C_vec(k - 2) + softplus_C_vec(k - 2);
           }
       } else { 
           Eigen::Matrix<double, -1, 1> exp_C_vec = stan::math::exp(C_raw_vec.segment(1, n_total_cutpoints - 1));
           for (int k = 2; k <= n_total_cutpoints; ++k) {
               C_vec(k - 1) = C_vec(k - 2) + exp_C_vec(k - 2);
           }
       }
      
       return C_vec;
   
}


inline double raw_C_to_C_log_det_J_lp( const Eigen::Matrix<double, -1, 1> &raw_C, // parameter - log_diffs
                                       bool softplus) {
  
       const int n_cutpoints = raw_C.size();
       double log_det_J = 0.0;
       if (softplus == true) log_det_J += (stan::math::log_inv_logit(raw_C.segment(1, n_cutpoints - 1))).sum();
       else                  log_det_J += raw_C.segment(1, n_cutpoints - 1).sum();  
       return log_det_J;
       
}

//// ---------------------------------------------------------------------------------------
//// Induced-Dirichlet ("ind_dir") log-density function:
//// NOTE: You can use this for both ind_dir PRIORS and ind_dir MODELS:
//// NOTE: adapted from: Betancourt et al (see: https://betanalpha.github.io/assets/case_studies/ordinal_regression.html),
//// HOWEVER my version has a (much) more computationally efficient (lower-trianglar) Jacobian computation, which is 
//// mathematically still valid. 
////
inline double induced_dirichlet_given_C_lpdf( const Eigen::Matrix<double, -1, 1> &p_ord,
                                              const Eigen::Matrix<double, -1, 1> &C,
                                              const Eigen::Matrix<double, -1, 1> &alpha,
                                              bool use_probit_link) {
  
       const int n_cat = p_ord.size();
       const int n_thr = n_cat - 1;
        
       double log_prob = 0.0;
        
       for (int k = 1; k <= n_thr; ++k) {
          if (use_probit_link == true)  log_prob += stan::math::normal_lpdf(C(k - 1), 0.0, 1.0);
          else                          log_prob += stan::math::log_inv_logit(C(k - 1)) + stan::math::log1m_inv_logit(C(k - 1));
       }
       log_prob += stan::math::dirichlet_lpdf(p_ord, alpha);
        
       return log_prob;
  
}


////
//// Convert from cumul_probs -> ord_probs:
////
inline Eigen::Matrix<double, -1, 1> cumul_probs_to_ord_probs( const Eigen::Matrix<double, -1, 1> &cumul_probs) {
  
       const int n_thr = cumul_probs.size();
       const int n_cat = n_thr + 1;
       Eigen::Matrix<double, -1, 1> ord_probs(n_cat);
        
       ord_probs(0) = cumul_probs(0) - 0.0;
       for (int k = 2; k <= n_thr; ++k) {
          ord_probs(k - 1) = cumul_probs(k - 1) - cumul_probs(k - 2); // since probs are INCREASING with k
       }
       ord_probs(n_cat - 1) =  1.0 - cumul_probs(n_cat - 2);
        
       return ord_probs;
  
}







////
//// Convert from ord_probs -> cumul_probs:
////
inline Eigen::Matrix<double, -1, 1> ord_probs_to_cumul_probs( const Eigen::Matrix<double, -1, 1> &ord_probs) {
  
       const int n_cat = ord_probs.size();
       const int n_thr = n_cat - 1;
       Eigen::Matrix<double, -1, 1> cumul_probs(n_thr);
      
       cumul_probs(0) = ord_probs(0);
       for (int k = 2; k <= n_thr; ++k) {
         cumul_probs(k - 1) = cumul_probs(k - 2) + ord_probs(k - 1); // since probs are INCREASING with k
       }
      
       return cumul_probs;
  
}


////
//// Convert from cumul_probs -> C (w/ induced dirichlet):
////
// vector ID_cumul_probs_to_C( const Eigen::Matrix<double, -1, 1> &cumul_probs, 
//                             int use_probit_link) {
// 
//       int n_thr = stan::math::num_elements(cumul_probs);
//       int n_cat = n_thr + 1;
//       Eigen::Matrix<double, -1, 1> C(n_thr);
//       
//       for (k in 1:n_thr) {
//         if (cumul_probs[k] < 1e-300) {
//           C[k] = -37.5; ////  prob = 1e-38;
//         } else if (cumul_probs[k] > 0.99999999999999999) {
//           C[k] = +8.20; ////  prob = 0.9999999999999;
//         } else {
//           if (use_probit_link == true) C[k] = stan::math::inv_Phi(cumul_probs[k]); 
//           else                         C[k] = stan::math::logit(cumul_probs[k]); 
//         }
//       }
//       
//       return C;
//     
// }

inline Eigen::Matrix<double, -1, 1> ID_cumul_probs_to_C( const Eigen::Matrix<double, -1, 1> &cumul_probs, 
                                                         bool use_probit_link) {
  
       const int n_thr = cumul_probs.size();
       const int n_cat = n_thr + 1;
       Eigen::Matrix<double, -1, 1> C(n_thr);
      
      for (int k = 0; k < n_thr; ++k) {
         if (cumul_probs(k) < 1e-300) {
           C(k) = -37.5; ////  prob = 1e-38;
         } else if (cumul_probs(k) > 0.99999999999999999) {
           C(k) = +8.20; ////  prob = 0.9999999999999;
         } else {
           if (use_probit_link == true) C(k) = stan::math::inv_Phi(cumul_probs(k)); 
           else                         C(k) = stan::math::logit(cumul_probs(k)); 
         }
       }
      
       return C;
  
}
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

inline std::array<Eigen::Matrix<double, -1, -1>, 1>      array_of_mats_1d(  int n_rows,
                                                                            int n_cols) {
   
   
       std::array<Eigen::Matrix<double, -1, -1 >, 1> my_1d_array;
       Eigen::Matrix<double, -1, -1> my_mat = Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols);
       my_1d_array[0] = my_mat;
     
       return my_1d_array;
   
}
 
 
inline std::array<Eigen::Matrix<double, -1, -1>, 2> array_of_mats_2d( int n_rows,
                                                                      int n_cols) {
   
       std::array<Eigen::Matrix<double, -1, -1>, 2> my_2d_array = {
                                 Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols),
                                 Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols)
                               };
     
       return  my_2d_array;
   
}
 
  
// convert std vec to eigen vec - double
inline Eigen::Matrix<double, -1, 1> std_vec_to_Eigen_vec( const std::vector<double> &std_vec) {
   
       Eigen::Matrix<double, -1, 1>  Eigen_vec(std_vec.size());
       
       for (int i = 0; i < std_vec.size(); ++i) {
         Eigen_vec(i) = std_vec[i];
       }
       
       return(Eigen_vec);
   
}
 
 
 inline std::vector<double> Eigen_vec_to_std_vec( const Eigen::Matrix<double, -1, 1> &Eigen_vec) {
   
       std::vector<double>  std_vec(Eigen_vec.rows());
       
       for (int i = 0; i < Eigen_vec.rows(); ++i) {
         std_vec[i] = Eigen_vec(i);
       }
       
       return(std_vec);
   
}
 
 
inline Eigen::Matrix<double, 1, -1>     fn_first_element_neg_rest_pos( Eigen::Matrix<double, 1, -1>  &row_vec) {
   
       row_vec(0) = - row_vec(0);
       
       return(row_vec);
       
}
 
 
 
 
 
 
 // 
 // 
 // inline  std::unique_ptr<size_t[]> get_commutation_unequal_vec
 // (unsigned const n, unsigned const m, bool const transpose){
 //   
 //   unsigned const nm = n * m,
 //     nnm_p1 = n * nm + 1L,
 //     nm_pm = nm + m;
 //   std::unique_ptr<size_t[]> out(new size_t[nm]);
 //   size_t * const o_begin = out.get();
 //   size_t idx = 0L;
 //   for(unsigned i = 0; i < n; ++i, idx += nm_pm){
 //     size_t idx1 = idx;
 //     for(unsigned j = 0; j < m; ++j, idx1 += nnm_p1)
 //       if(transpose)
 //         *(o_begin + idx1 / nm) = (idx1 % nm);
 //       else
 //         *(o_begin + idx1 % nm) = (idx1 / nm);
 //   }
 //   
 //   return out;
 //   
 // }
 // 
 // 
 // 
 // inline  Rcpp::NumericVector commutation_dot
 // (unsigned const n, unsigned const m, Rcpp::NumericVector x,
 //  bool const transpose){
 //   size_t const nm = n * m;
 //   Rcpp::NumericVector out(nm);
 //   auto const indices = get_commutation_unequal_vec(n, m, transpose);
 //   
 //   for(size_t i = 0; i < nm; ++i)
 //     out[i] = x[*(indices.get() +i )];
 //   
 //   return out;
 // }
 // 
 // inline  Rcpp::NumericMatrix get_commutation_unequal
 // (unsigned const n, unsigned const m){
 //   
 //   unsigned const nm = n * m,
 //     nnm_p1 = n * nm + 1L,
 //     nm_pm = nm + m;
 //   Rcpp::NumericMatrix out(nm, nm);
 //   double * o = &out[0];
 //   for(unsigned i = 0; i < n; ++i, o += nm_pm){
 //     double *o1 = o;
 //     for(unsigned j = 0; j < m; ++j, o1 += nnm_p1)
 //       *o1 = 1.;
 //   }
 //   
 //   return out;
 // }
 // 
 // inline  Rcpp::NumericMatrix get_commutation_equal(unsigned const m){
 //   unsigned const mm = m * m,
 //     mmm = mm * m,
 //     mmm_p1 = mmm + 1L,
 //     mm_pm = mm + m;
 //   Rcpp::NumericMatrix out(mm, mm);
 //   double * const o = &out[0];
 //   unsigned inc_i(0L);
 //   for(unsigned i = 0; i < m; ++i, inc_i += m){
 //     double *o1 = o + inc_i + i * mm,
 //       *o2 = o + i     + inc_i * mm;
 //     for(unsigned j = 0; j < i; ++j, o1 += mmm_p1, o2 += mm_pm){
 //       *o1 = 1.;
 //       *o2 = 1.;
 //     }
 //     *o1 += 1.;
 //   }
 //   return out;
 // }
 // 
 // 
 
 // 
 // 
 // inline  Eigen::Matrix<double, -1, -1  >  get_commutation(unsigned const n, unsigned const m) {
 //   
 //   if (n == m)  {
 //     
 //     Rcpp::NumericMatrix commutation_mtx_Nuemric_Matrix =  get_commutation_equal(n);
 //     
 //     double n_rows = commutation_mtx_Nuemric_Matrix.nrow();
 //     double n_cols = commutation_mtx_Nuemric_Matrix.ncol();
 //     
 //     Eigen::Matrix<double, -1, -1>  commutation_mtx_Eigen   =  Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols);
 //     
 //     
 //     for (int i = 0; i < n_rows; ++i) {
 //       for (int j = 0; j < n_cols; ++j) {
 //         commutation_mtx_Eigen(i, j) = commutation_mtx_Nuemric_Matrix(i, j) ;
 //       }
 //     }
 //     
 //     return commutation_mtx_Eigen;
 //     
 //     
 //   } else {
 //     
 //     Rcpp::NumericMatrix commutation_mtx_Nuemric_Matrix =  get_commutation_unequal(n, m);
 //     
 //     double n_rows = commutation_mtx_Nuemric_Matrix.nrow();
 //     double n_cols = commutation_mtx_Nuemric_Matrix.ncol();
 //     
 //     Eigen::Matrix<double, -1, -1>  commutation_mtx_Eigen   =  Eigen::Matrix<double, -1, -1>::Zero(n_rows, n_cols);
 //     
 //     
 //     for (int i = 0; i < n_rows; ++i) {
 //       for (int j = 0; j < n_cols; ++j) {
 //         commutation_mtx_Eigen(i, j) = commutation_mtx_Nuemric_Matrix(i, j) ;
 //       }
 //     }
 //     
 //     return commutation_mtx_Eigen;
 //     
 //     
 //   }
 //   
 //   
 // }
 // 
 // 
 // 
 // 
 // 
 // 
 // 
 // 
 // inline Eigen::Matrix<double, -1, -1  > elimination_matrix(const int &n) {
 //   
 //   Eigen::Matrix<double, -1, -1> out   =  Eigen::Matrix<double, -1, -1>::Zero((n*(n+1))/2,  n*n);
 //   
 //   for (int j = 0; j < n; ++j) {
 //     Eigen::Matrix<double, 1, -1> e_j   =  Eigen::Matrix<double, 1, -1>::Zero(n);
 //     
 //     e_j(j) = 1.0;
 //     
 //     for (int i = j; i < n; ++i) {
 //       Eigen::Matrix<double, -1, 1> u   =  Eigen::Matrix<double, -1, 1>::Zero((n*(n+1))/2);
 //       u(j*n+i-((j+1)*j)/2) = 1.0;
 //       Eigen::Matrix<double, 1, -1> e_i   =  Eigen::Matrix<double, 1, -1>::Zero(n);
 //       e_i(i) = 1.0;
 //       
 //       out += Eigen::kroneckerProduct(u, Eigen::kroneckerProduct(e_j, e_i));
 //     }
 //   }
 //   
 //   return out;
 // }
 // 
 // 
 // 
 // 
 // 
 // 
 // 
 // 
 // 
 // inline  Eigen::Matrix<double, -1, -1  > duplication_matrix(const int &n) {
 //   
 //   //arma::mat out((n*(n+1))/2, n*n, arma::fill::zeros);
 //   Eigen::Matrix<double, -1, -1> out   =  Eigen::Matrix<double, -1, -1>::Zero((n*(n+1))/2,  n*n);
 //   
 //   for (int j = 0; j < n; ++j) {
 //     for (int i = j; i < n; ++i) {
 //       // arma::vec u((n*(n+1))/2, arma::fill::zeros);
 //       Eigen::Matrix<double, -1, 1> u   =  Eigen::Matrix<double, -1, 1>::Zero((n*(n+1))/2);
 //       u(j*n+i-((j+1)*j)/2) = 1.0;
 //       
 //       //       arma::mat T(n,n, arma::fill::zeros);
 //       Eigen::Matrix<double, -1, -1> T   =  Eigen::Matrix<double, -1, -1>::Zero(n, n);
 //       T(i,j) = 1.0;
 //       T(j,i) = 1.0;
 //       
 //       Eigen::Map<Eigen::Matrix<double, -1, 1> > T_vec(T.data(), n*n);
 //       
 //       out += u * T_vec.transpose();
 //     }
 //   }
 //   
 //   return out.transpose();
 //   
 // }
 // 
 // 
 // 
 // 
 // 
 // 
 // 
 
 
 
 
 
 
 
 
 inline  Eigen::Matrix<double, -1, -1  > inv_Phi_approx_from_logit_prob_Stan(  Eigen::Matrix<double, -1, -1  > logit_p)   {
   using namespace stan::math;
   Eigen::Array<double, -1, -1  > x_i = 0.34176618822627863*logit_p.array();
   Eigen::Array<double, -1, -1  > asinh_stuff_div_3 =  0.33333333333333331483 *   log(  ( x_i.array()  +  sqrt(fma(x_i.matrix(), x_i.matrix(), 1.0)).array()  ).matrix() ).array() ;          // now do arc_sinh part
   Eigen::Array<double, -1, -1  > exp_x_i = exp(asinh_stuff_div_3.matrix() ).array();
   return (  2.7472242570765295 * (  fma(exp_x_i.matrix(), exp_x_i.matrix(), -1.0).array()   / exp_x_i ).array()   ).matrix()    ;  //   now do sinh parth part
   
 }
 

 
 
  
  inline  Eigen::Matrix<double, -1, -1  > inv_Phi_approx_from_logit_prob_Eigen_double(  Eigen::Matrix<double, -1, -1  > logit_p)   {
   Eigen::Array<double, -1, -1  > x_i = 0.34176618822627863*logit_p.array();
   Eigen::Array<double, -1, -1  > asinh_stuff_div_3 =  0.33333333333333331483 *  ( x_i  + (x_i*x_i + 1.0 ).sqrt() ).log() ;   // now do arc_sinh part
   Eigen::Array<double, -1, -1  > exp_x_i =  (asinh_stuff_div_3).exp();
   return  (  2.7472242570765295 * ( (exp_x_i*exp_x_i  - 1.0) / exp_x_i )  ).matrix()   ;  //   now do sinh parth part
 }
 
 
 
 
 
 
 
 
 
 // // NOTE: Compiler needs  to auto-vectorise for the following to be fast!!
 inline  Eigen::Matrix<double, -1, -1  > fn_mat_loop_dbl_Eigen(  Eigen::Matrix<double, -1, -1  > x, 
                                                                 std::function<double(double)> fn_double) {
   int  N = x.rows();     ///// Eigen::Matrix<double, -1, 1  >  out(N); /// NOTABLY slower if store in new array!!!!!! (making arrays is taxing !!!!)
   int  M = x.cols();
   for (int j = 0; j < M; ++j) {
     for (int i = 0; i < N; ++i) {
       x(i, j) = fn_double(x(i, j));
     }
   }
   return  x;  
 }


 // // NOTE: Compiler needs  to auto-vectorise for the following to be fast!!
 inline  Eigen::Matrix<double, -1, 1  > fn_colvec_loop_dbl_Eigen(  Eigen::Matrix<double, -1, 1  > x, 
                                                                   std::function<double(double)> fn_double) {
   for (int i = 0; i < x.rows(); ++i) {
     x(i) = fn_double(x(i));
   }  
   return  x;
 }
  
 
 
 
 
 



 


 
 
 
 
// 
// // -------------------------------------------------------------------------------------------------------------------------------------------------------------







double __int_as_double (int64_t a) {  double r; memcpy (&r, &a, sizeof r); return r;}
int64_t __double_as_int (double a) { int64_t r; memcpy (&r, &a, sizeof r); return r;}




 

// checks specifically for COL vectors  -  i.e., true if = Eigen::Matrix<-1, 1, double> [COMPILE-time]
template<typename T>
struct is_eigen_col_vector {
  static constexpr bool value = std::is_base_of_v<Eigen::MatrixBase<typename std::decay<T>::type>, typename std::decay<T>::type> && 
    T::ColsAtCompileTime == 1;
}; 

// checks specifically for ROW vectors  -  i.e., true if = Eigen::Matrix<1, -1, double> [COMPILE-time]
template<typename T>
struct is_eigen_row_vector {
  static constexpr bool value = std::is_base_of_v<Eigen::MatrixBase<typename std::decay<T>::type>, typename std::decay<T>::type> && 
    T::RowsAtCompileTime == 1; 
}; 

// checks specifically for [DYNAMIC] MATRICES (more than one row AND more than one column) - i.e., true if = Eigen::Matrix<-1, -1, double> 
template<typename T>
struct is_dynamic_matrix {
  static constexpr bool value = 
    std::is_base_of_v<Eigen::MatrixBase<typename std::decay<T>::type>, 
                      typename std::decay<T>::type> && 
                        T::RowsAtCompileTime == -1 && 
                        T::ColsAtCompileTime == -1 &&
                        std::is_same_v<typename T::Scalar, double>;
}; 
 

///////////////////  fns - exp   -----------------------------------------------------------------------------------------------------------------------------





// see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
/* For a in [0.5, 4), compute a * 2**i, -250 < i < 250 */
// Note: this function is  the same as the one in the link above, but for * double * instead of * float *.
// not handling edge cases  - should work for a in [0.5, 2.0). 
inline double fast_ldexp(const double a, 
                         const int i) {
  int64_t ia = ((uint64_t)i << 52ULL) + __double_as_int(a);
  return __int_as_double(ia);
}

/// this is the same as fast_ldexp but handles edge-cases (will have wider input range )
/// if i is too large, splits the calc. into two steps
inline double fast_ldexp_2(const double a, 
                           const int i) {
  // In IEEE 754 double precision format:
  // Bits 0-51:  Fraction (mantissa)
  // Bits 52-62: Exponent (11 bits)
  // Bit 63:     Sign
  
  // First attempt: try to directly manipulate the exponent bits
  // Shifting i by 52 positions puts it in the exponent field
  int64_t ia = ((uint64_t)i << 52ULL) + __double_as_int(a);
  
  // Check if |i| is too large to prevent exponent overflow
  // For IEEE 754 doubles:
  // - Minimum exponent is -1022 (subnormal numbers)
  // - Maximum exponent is +1023 (before infinity)
  // We use ±1000 as a safe threshold
  if ((unsigned int)(i + 1000) > 2000) { // This checks if |i| > 1000
    // If |i| is too large, split the scaling into two steps
    // to avoid exponent overflow/underflow
     
    // Scale by +/- 1000 (depending on sign of i)
    int i1 = (i < 0) ? -1000 : 1000;
    // Scale by the remaining amount
    int i2 = i - i1;
    
    // Perform the two-step scaling
    // first scale by i1
    int64_t step1 = ((uint64_t)i1 << 52ULL) + __double_as_int(a);
    double mid = __int_as_double(step1);
    
    // then scale by i2
    // int64_t step2 = ((uint64_t)i2 << 52ULL) + __double_as_int(mid);
    // return __int_as_double(step2);
    return fast_ldexp(mid, i2);
    
  }
  
  
  return fast_ldexp(a, i); // if doesnt exceed threshold just use 'standard' fast_ldexp()
  
}






// Adapted from: https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx
// added (optional) extra degree(s) for poly approx (oroginal float fn had 4 degrees) - using "minimaxApprox" R package to find coefficient terms
// R code:   minimaxApprox::minimaxApprox(fn = exp, lower = -0.346573590279972643113, upper = 0.346573590279972643113, degree = 5, basis ="Chebyshev")
inline double fast_exp_1_wo_checks (const double x)  {
  
  const double input = x;
  const double l2e =   (1.442695040888963387); /* log2(e) */
  const double l2h =   (-0.693145751999999948367); /* -log(2)_hi */
  const double l2l =   (-0.00000142860676999999996193); /* -log(2)_lo */

  // /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
   // // ///// 9-degree approx:
   const double c0 =     (0.00000276479776161191821278);
   const double c1 =     (0.0000248844480527491290235);
   const double c2 =     (0.000198411488032534342194);
   const double c3 =     (0.00138888017711994078175);
   const double c4 =     (0.00833333340524595143906);
   const double c5 =     (0.0416666670404215802592);
   const double c6 =     (0.166666666664891632843);
   const double c7 =     (0.499999999994389376923);
   const double c8 =     (1.00000000000001221245);
   const double c9 =     (1.00000000000001332268);
   
  /* exp(x) = 2^i * e^f; i = rint (log2(e) * x), f = x - log(2) * i */
  const double  t = x*l2e ; // _mm512_mul_pd (x, l2e);      /* t = log2(e) * x */
  const int i = (int)rint(t) ; // i = _mm512_cvtpd_epi64(t);       /* i = (int)rint(t) */
  const double x_2 = std::round(t) ; //  _mm512_roundscale_pd(t,((0<<4)| _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC|_MM_FROUND_NO_EXC));
  const double f = std::fma (x_2, l2l, std::fma(x_2, l2h, input)); /* x - log(2)_hi * r */    /* f = x - log(2)_hi * r - log(2)_lo * r */
  
  /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
  double  p;
  p = c0;
  p = std::fma (p, f, c1);
  p = std::fma (p, f, c2);
  p = std::fma (p, f, c3);
  p = std::fma (p, f, c4);
  p = std::fma (p, f, c5);
  p = std::fma (p, f, c6);
  p = std::fma (p, f, c7);
  p = std::fma (p, f, c8);
  p = std::fma (p, f, c9);
  
  return fast_ldexp_2(p, i) ;
  
}


 
 
 
 
 



// see https://stackoverflow.com/questions/39587752/difference-between-ldexp1-x-and-exp2x
// Note: this function is  the same as the one in the link above, but for * double * instead of * float *.
inline double fast_exp_1(const double a)    {
  if (std::fabs (a) < 708.4) {
    return  fast_exp_1_wo_checks(a);
  } else {
   // const double large =  8.98846567431157953864652595394512366808e307;
    double r = INFINITY - INFINITY; //  // this handles NaNs
    if (a < 0.0) r = 0.0;
    if (a > 0.0) r = INFINITY; // + INF
    return r;
  }
}
 






// 
///////////////////  fns - log   -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//https://stackoverflow.com/a/65537754/9007125
inline  double fast_log_1_wo_checks (const double a)  {
  
        double  r,  t;
  
        const int64_t e = (__double_as_int (a) - 0x3fe5555555555555 )    &   0xFFF0000000000000   ;
        const double i = fma ((double)e, 0.000000000000000222044604925031308085, 0.0); // 0x1.0p-52
        const double m = __int_as_double (__double_as_int (a) - e) - 1.0 ;  /* m in [2/3, 4/3] */
        const double s = m * m;
        
        /* Compute log1p(m) for m in [-1/3, 1/3] */
        r =             -0.13031005859375;  // -0x1.0ae000p-3
        t =              0.140869140625;  //  0x1.208000p-3
        r = fma (r, s, -0.121483512222766876221); // -0x1.f198b2p-4
        t = fma (t, s,  0.139814853668212890625); //  0x1.1e5740p-3
        r = fma (r, s, -0.166846126317977905273); // -0x1.55b36cp-3
        t = fma (t, s,  0.200120344758033752441); //  0x1.99d8b2p-3
        r = fma (r, s, -0.249996200203895568848); // -0x1.fffe02p-3
        r = fma (t, m, r);
        r = fma (r, m,  0.333331972360610961914); //  0x1.5554fap-2
        r = fma (r, m, -0.5); // -0x1.000000p-1
        r = fma (r, s, m);
        
        return fma(i,  0.693147182464599609375, r); //  0x1.62e430p-1 // log(2)
        
}

//  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
inline   double fast_log1p_1_wo_checks(const double x)   {
  
  if (fabs(x) > 1e-4)   return fast_log_1_wo_checks(1.0 + x);     // x is large enough that the obvious evaluation is OK
  return (-0.5*x + 1.0)*x;     // Use Taylor approx. log(1 + x) = x - x^2/2 with error roughly x^3/3     // Since |x| < 10^-4, |x|^3 < 10^-12, relative error less than 10^-8
  
}

//  compute log(1-x) without losing precision for small values of x.
inline   double fast_log1m_1_wo_checks(const double x)   {
  
  return fast_log1p_1_wo_checks(-x);
  
}

//  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
inline   double fast_log1p_exp_1_wo_checks(const double x)   {
  
  if (x > 0.0) return x + fast_log1p_1_wo_checks(fast_exp_1_wo_checks(-x));
  return fast_log1p_1_wo_checks(fast_exp_1_wo_checks(x));
  
}
 
 
 

//https://stackoverflow.com/a/65537754/9007125 /* compute natural logarithm, maximum error 0.85089 ulps */
inline   double fast_log_1  (const double a)  {
  
      if (!((a > 0.0) && (a < INFINITY))) {
        
        double r = a + a;  // silence NaNs if necessary
        if (a  < 0.0) r =  INFINITY - INFINITY; //  NaN
        if (a == 0.0) r = -INFINITY;
        return r;
        
      }
      
      return  fast_log_1_wo_checks(a);
      
}

//  compute log(1+x) without losing precision for small values of x.
//  see: https://www.johndcook.com/cpp_log_one_plus_x.html
inline   double fast_log1p_1(const double x)   {
  
  if (fabs(x) > 1e-4)   return fast_log_1(1.0 + x);     // x is large enough that the obvious evaluation is OK
  
  if (x <= -1.0)   {
    
    std::stringstream os;
    os << "Invalid input argument (" << x
       << "); must be greater than -1.0";
    throw std::invalid_argument( os.str() );
    
  }
  
  return (-0.5*x + 1.0)*x;  // Use Taylor approx. log(1 + x) = x - x^2/2 with error roughly x^3/3     // Since |x| < 10^-4, |x|^3 < 10^-12, relative error less than 10^-8
  
}

//  compute log(1-x) without losing precision for small values of x.
inline  double fast_log1m_1(const double x)   {
  
  return fast_log1p_1(-x);
  
}

//  compute log(1+x) without losing precision for small values of x. //  see: https://www.johndcook.com/cpp_log_one_plus_x.html
inline   double fast_log1p_exp_1(const double x)   {
  
  if (x > 0.0) return x + fast_log1p_1(fast_exp_1(-x));
  return fast_log1p_1(fast_exp_1(x));
  
}
 












///////////////////  fns - inv_logit   ----------------------------------------------------------------------------------------------------------------


inline  double  fast_inv_logit_for_x_pos(const double x )  {
  return     (1.0 /  (1.0 + fast_exp_1(-x)))  ;
}

inline  double  fast_inv_logit_for_x_neg(const double x )  {
  const double log_eps = -18.420680743952367;
  const double   exp_x =  fast_exp_1(x) ;
  if (x > log_eps) return  exp_x / (1.0 + exp_x);
  return  exp_x;
}

inline  double  fast_inv_logit(const double x )  {
  
  double val;
  
  if (x > 0.0) { 
    val =   fast_inv_logit_for_x_pos(x);
  } else { 
    val =  fast_inv_logit_for_x_neg(x);
  }
 
  if  ((val > 0) && (val < 1))  { 
    return val;
  } else if (val > 1) { 
    return 1;
  } else { 
    return 0;
  }
  
}
 










inline  double  fast_inv_logit_for_x_pos_wo_checks(const double x )  {
  double     exp_m_x =  fast_exp_1_wo_checks(-x) ;
  return     (1.0 /  (1.0 + exp_m_x))  ;
}

inline   double  fast_inv_logit_for_x_neg_wo_checks(const double x )  {
  double log_eps = -18.420680743952367;
  double   exp_x =  fast_exp_1_wo_checks(x) ;
  if (x > log_eps) return   exp_x / (1.0 + exp_x);
  return  exp_x;
}

inline   double  fast_inv_logit_wo_checks(const double x )  {
  if (x > 0) return  fast_inv_logit_for_x_pos_wo_checks(x);
  return  fast_inv_logit_for_x_neg_wo_checks(x);
  
  double val;
  
  if (x > 0.0) { 
    val =   fast_inv_logit_for_x_pos_wo_checks(x);
  } else { 
    val =  fast_inv_logit_for_x_neg_wo_checks(x);
  }
  
  if  ((val > 0) && (val < 1))  { 
    return val;
  } else if (val > 1) {  
    return 1;
  } else {  
    return 0;
  }
  
}
 
















///////////////////  fns - log_inv_logit   ----------------------------------------------------------------------------------------------------------------

inline   double  fast_log_inv_logit_for_x_pos(const double x )  {
  return    - fast_log1p_exp_1(-x);  // return    - fast_log1p_1(fast_exp_1(-x));
}

inline   double  fast_log_inv_logit_for_x_neg(const double x )  {
  const double log_eps = -18.420680743952367;
  if (x > log_eps) return  x - fast_log1p_1(fast_exp_1(x));
  return x;
}

inline   double  fast_log_inv_logit(const double x )  {
  if (x > 0.0) return  fast_log_inv_logit_for_x_pos(x);
  return  fast_log_inv_logit_for_x_neg(x);
}

 
















inline   double  fast_log_inv_logit_for_x_pos_wo_checks(const double x )  {
   return    - fast_log1p_exp_1_wo_checks((-x));
}

inline   double  fast_log_inv_logit_for_x_neg_wo_checks(const double x )  {
  const double log_eps = -18.420680743952367;
  if (x > log_eps) return x - fast_log1p_1_wo_checks(fast_exp_1_wo_checks(x));
  return x;
}

inline   double  fast_log_inv_logit_wo_checks(const double x )  {
  if (x > 0.0) return  fast_log_inv_logit_for_x_pos_wo_checks(x);
  return  fast_log_inv_logit_for_x_neg_wo_checks(x);
}
 




 
 
//  ///////////////////  fns - Phi_approx   ----------------------------------------------------------------------------------------------------------------




inline  double  fast_Phi_approx_wo_checks(const double x )  {
  
  const double    a =       (0.07056);
  const double    b =       (1.5976);
  const double    x_sq = x*x;
  const double    a_x_sq_plus_b = fma(a, x_sq, b);
  const double    stuff_to_inv_logit =  x*a_x_sq_plus_b;
  
  return    fast_inv_logit_wo_checks(stuff_to_inv_logit);
  
}
 


 


inline   double  fast_Phi_approx(const double x )  {
  
  const double    a =       (0.07056);
  const double    b =       (1.5976);
  const double    x_sq = x*x;
  const double    a_x_sq_plus_b = fma(a, x_sq, b);
  const double    stuff_to_inv_logit =  x*a_x_sq_plus_b;
  
  return    fast_inv_logit(stuff_to_inv_logit);
  
}
 

 




///////////////////  fns - inv_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------






inline  Eigen::Matrix<double, -1, -1  > inv_Phi_approx_Eigen_double(  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> x)   {
  
  Eigen::Matrix<double, -1, -1  > x_i = -0.34176618822627863*( 1.0/x.array()  - 1.0 ).log();
  Eigen::Matrix<double, -1, -1  > asinh_stuff_div_3 = ( 0.33333333333333331483 *  ( x_i.array()   + (x_i.array() *x_i.array()  + 1.0 ).sqrt() ).log()  ).matrix() ;          // now do arc_sinh part
  Eigen::Matrix<double, -1, -1  > exp_x_i =  (asinh_stuff_div_3).exp();
  return  2.7472242570765295 * ( ( exp_x_i.array() *exp_x_i.array()   - 1.0).array()  / exp_x_i.array()  ).matrix() ;  //   now do sinh parth part
  
} 




inline  double  inv_Phi_approx_std( const double x )  {
  const double m_logit_p =   std::log( 1.0/x  - 1.0 )  ; // log first
  const double x_i = -0.34176618822627863*m_logit_p;
  const double asinh_stuff_div_3 =  0.33333333333333331483 * std::log( x_i  +  std::sqrt(  fma(x_i, x_i, 1.0) ) )  ;          // now do arc_sinh part
  const double exp_x_i =  std::exp(asinh_stuff_div_3);
  return  2.7472242570765295 * ( fma(exp_x_i, exp_x_i , -1.0) / exp_x_i ) ;  //   now do sinh parth part
}








inline Eigen::Matrix<double, -1, -1> inv_Phi_approx_Stan(const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> x)   {
  
            Eigen::Matrix<double, -1, -1  > x_i = ( -0.34176618822627863*stan::math::log(  ( 1.0/x.array() - 1.0 ).matrix()  ).array()  ).matrix() ;
            Eigen::Matrix<double, -1, -1  > asinh_stuff_div_3 = (  0.33333333333333331483 *  stan::math::log(  ( x_i.array()  + stan::math::sqrt( (x_i.array()*x_i.array() + 1.0 ).matrix() ).array()  ).matrix()  ).array() ).matrix() ;          // now do arc_sinh part
            Eigen::Matrix<double, -1, -1  > exp_x_i =  stan::math::exp(asinh_stuff_div_3.matrix() ).matrix();
            
            return  ( 2.7472242570765295 * ( ( exp_x_i.array()*exp_x_i.array()  - 1.0) / exp_x_i.array() ).array() ).matrix() ;  //   now do sinh parth part
}




 
inline Eigen::Matrix<double, -1, -1  > inv_Phi_approx_Eigen(  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  x)   {
  
            Eigen::Array<double, -1, -1  > x_i = -0.34176618822627863*( 1.0/x.array() - 1.0 ).log();
            Eigen::Array<double, -1, -1  > asinh_stuff_div_3 =  0.33333333333333331483 *  ( x_i  + (x_i*x_i + 1.0 ).sqrt() ).log() ;          // now do arc_sinh part
            Eigen::Array<double, -1, -1  > exp_x_i =  (asinh_stuff_div_3).exp();
            
            return   (  2.7472242570765295 * ( ( exp_x_i*exp_x_i  - 1.0) / exp_x_i )  ).matrix() ;  //   now do sinh parth part
            
}






 
inline  double  fast_inv_Phi_approx_wo_checks( const double x )  {
  
  const double m_logit_p = fast_log_1_wo_checks( 1.0/x - 1.0); // log first
  const double x_i = -0.34176618822627863*m_logit_p;
  const double asinh_stuff_div_3 =  0.33333333333333331483 * fast_log_1_wo_checks( x_i  +  std::sqrt(  fma(x_i, x_i, 1.0) ) ) ;          // now do arc_sinh part
  const double exp_x_i = fast_exp_1_wo_checks(asinh_stuff_div_3);
  return  2.7472242570765295 * ( ((exp_x_i*exp_x_i)  - 1.0) / exp_x_i ) ;  //   now do sinh parth part
  
}
 


inline  double  fast_inv_Phi_approx( const double x )  {
  
      if ( (x > 0) &&  (x < 1)) {
            const double m_logit_p = fast_log_1(1.0/x - 1.0); // log first
            const double x_i = -0.34176618822627863*m_logit_p;
            const double asinh_stuff_div_3 =  0.33333333333333331483 * fast_log_1(x_i + std::sqrt(fma(x_i, x_i, 1.0))) ;          // now do arc_sinh part
            const double exp_x_i = fast_exp_1(asinh_stuff_div_3);
            return  2.7472242570765295 * (((exp_x_i*exp_x_i)  - 1.0) / exp_x_i) ;  //   now do sinh parth part
      }  else {
            if ((x < 0) || (x > 1)) {
              // return std::numeric_limits<double>::quiet_NaN();
                std::stringstream os;
                os << "Invalid input argument for inv_Phi_approx (" << x
                   << "); must be in [0, 1]";
                throw std::invalid_argument( os.str() );
            }   else if  (x == 0) {
              return -INFINITY;
            }   else { // if x  == 1
              return INFINITY;
            }
      }
      
}
 






// 




// need to add citation to this (slight modification from a forum post)
inline    double inv_Phi_approx_from_logit_prob_std(const double logit_p) {
  
  double x_i = 0.34176618822627863*logit_p;
  double asinh_stuff_div_3 =  0.33333333333333331483 *  std::log( x_i  +   std::sqrt(  fma(x_i, x_i, 1.0) ) ) ;          // now do arc_sinh part
  double exp_x_i =  std::exp(asinh_stuff_div_3);
  return  2.7472242570765295 * ( fma(exp_x_i, exp_x_i , -1.0) / exp_x_i ) ;  //   now do sinh parth part
  
}


 



// need to add citation to this (slight modification from a forum post)
inline    double fast_inv_Phi_approx_from_logit_prob_wo_checks(const double  logit_p) {
  
  double x_i = 0.34176618822627863*logit_p;
  double asinh_stuff_div_3 =  0.33333333333333331483 *  fast_log_1(x_i  +   std::sqrt(x_i*x_i + 1.0)) ;          // now do arc_sinh part
  double exp_x_i =  fast_exp_1_wo_checks(asinh_stuff_div_3);
  return  2.7472242570765295 * ( (exp_x_i*exp_x_i  - 1.0) / exp_x_i ) ;  //   now do sinh parth part
  
}
 

// need to add citation to this (slight modification from a forum post)
inline   double fast_inv_Phi_approx_from_logit_prob(const double  logit_p) {
  
  double x_i = 0.34176618822627863*logit_p;
  double asinh_stuff_div_3 =  0.33333333333333331483 *  fast_log_1_wo_checks(x_i  +   std::sqrt(x_i*x_i + 1.0)) ;          // now do arc_sinh part
  double exp_x_i =  fast_exp_1_wo_checks(asinh_stuff_div_3);
  return  2.7472242570765295 * ( (exp_x_i*exp_x_i  - 1.0) / exp_x_i ) ;  //   now do sinh parth part
  
}
 
// 
///////////////////  fns - log_Phi_approx   ----------------------------------------------------------------------------------------------------------------------------------------


inline Eigen::Matrix<double, -1, 1>  log_Phi_approx_Eigen_double( const  Eigen::Matrix<double, -1, 1> &q )  {
  return  ( - ((-(0.07056*q.array()*q.array()*q.array() + 1.5976*q.array())).exp()).log1p() ).matrix() ;
}


inline Eigen::Matrix<double, -1, 1>  log_Phi_approx_Stan_double( const  Eigen::Matrix<double, -1, 1> &q )  {
  return stan::math::log_inv_logit( (q.array()*(stan::math::fma(0.07056*q.matrix(), q.matrix(), 1.5976)).array()  ).matrix() ) ;
}


 
 
 

inline  double  fast_log_Phi_approx_wo_checks( const double x )  {
  
 const double    a =       (0.07056);
 const double    b =       (1.5976);
 const double    x_sq = x*x;
 const double    a_x_sq_plus_b = fma(a, x_sq, b);
 const double    stuff_to_inv_logit =  x*a_x_sq_plus_b;
 return fast_log_inv_logit_wo_checks(stuff_to_inv_logit);
 
}
 







/// this still has no bounds checking as it uses log_inv_logit 
inline   double  fast_log_Phi_approx(const double x )  {
 
      const double    a =       (0.07056);
      const double    b =       (1.5976);
      const double    x_sq = x*x;
      const double    a_x_sq_plus_b = fma(a, x_sq, b);
      const double    stuff_to_inv_logit =  x*a_x_sq_plus_b;
      return fast_log_inv_logit(stuff_to_inv_logit);
      
}
 






//////////////////// ------------- tanh  --------------------------------------------------------------------------------------------------------------------------------------

inline   double    fast_tanh(const   double x  )   {
  return    ( 2.0 / (1.0 + fast_exp_1(-2.0*x   )  ) - 1.0 )  ;
}
 



inline  double    fast_tanh_wo_checks(const   double x  )   {
  return    ( 2.0 / (1.0 + fast_exp_1_wo_checks(-2.0*x   )  ) - 1.0 )  ;
}
 



 
 
 
 
 
 
 
 
 
 
//////////////////////  fns - error functions   ----------------------------------------------------------------------------------------------------------------------------------------
// 
// 
//  // see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function
//  inline  double fast_erf_wo_checks_part_1_upper(const double a,
//                                                 const double t,
//                                                 const double s) {
//      double r, u;
// 
//      // max ulp error = 0.97749 (USE_EXPM1 = 1); 1.05364 (USE_EXPM1 = 0)
//      r = fma (-5.6271698391213282e-18, t, 4.8565951797366214e-16); // -0x1.9f363ba3b515dp-58, 0x1.17f6b1d68f44bp-51
//      u = fma (-1.9912968283386570e-14, t, 5.1614612434698227e-13); // -0x1.66b85b7fbd01ap-46, 0x1.22907eebc22e0p-41
//      r = fma (r, s, u);
//      u = fma (-9.4934693745934645e-12, t, 1.3183034417605052e-10); // -0x1.4e0591fd97592p-37, 0x1.21e5e2d8544d1p-33
//      r = fma (r, s, u);
//      u = fma (-1.4354030030292210e-09, t, 1.2558925114413972e-08); // -0x1.8a8f81b7e0e84p-30, 0x1.af85793b93d2fp-27
//      r = fma (r, s, u);
//      u = fma (-8.9719702096303798e-08, t, 5.2832013824348913e-07); // -0x1.8157db0edbfa8p-24, 0x1.1ba3c453738fdp-21
//      r = fma (r, s, u);
//      u = fma (-2.5730580226082933e-06, t, 1.0322052949676148e-05); // -0x1.595999b7e922dp-19, 0x1.5a59c27b3b856p-17
//      r = fma (r, s, u);
//      u = fma (-3.3555264836700767e-05, t, 8.4667486930266041e-05); // -0x1.197b61ee37123p-15, 0x1.631f0597f62b8p-14
//      r = fma (r, s, u);
//      u = fma (-1.4570926486271945e-04, t, 7.1877160107954648e-05); // -0x1.319310dfb8583p-13, 0x1.2d798353da894p-14
//      r = fma (r, s, u);
//      u = fma ( 4.9486959714661590e-04, t,-1.6221099717135270e-03); //  0x1.037445e25d3e5p-11,-0x1.a939f51db8c06p-10
//      r = fma (r, s, u);
//      u = fma ( 1.6425707149019379e-04, t, 1.9148914196620660e-02); //  0x1.5878d80188695p-13, 0x1.39bc5e0e9e09ap-6
//      r = fma (r, s, u);
//      r = fma (r, t, -1.0277918343487560e-1); // -0x1.a4fbc8f8ff7dap-4
//      r = fma (r, t, -6.3661844223699315e-1); // -0x1.45f2da3ae06f8p-1
//      r = fma (r, t, -1.2837929411398119e-1); // -0x1.06ebb92d9ffa8p-3
//      r = fma (r, t, -t);
//      r = 1.0 - fast_exp_1(r);
// 
//      return copysign (r, a);
// 
//  }
// 
// // see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function
// inline double fast_erf_wo_checks_part_2_lower(const double a,
//                                               const double t,
//                                               const double s) {
//     double r, u;
// 
//     // max ulp error = 1.01912
//     r =            -7.7794684889591997e-10; // -0x1.abae491c44131p-31
//     r = fma (r, s,  1.3710980398024347e-8); //  0x1.d71b0f1b10071p-27
//     r = fma (r, s, -1.6206313758492398e-7); // -0x1.5c0726f04dbc7p-23
//     r = fma (r, s,  1.6447131571278227e-6); //  0x1.b97fd3d9927cap-20
//     r = fma (r, s, -1.4924712302009488e-5); // -0x1.f4ca4d6f3e232p-17
//     r = fma (r, s,  1.2055293576900605e-4); //  0x1.f9a2baa8fedc2p-14
//     r = fma (r, s, -8.5483259293144627e-4); // -0x1.c02db03dd71bbp-11
//     r = fma (r, s,  5.2239776061185055e-3); //  0x1.565bccf92b31ep-8
//     r = fma (r, s, -2.6866170643111514e-2); // -0x1.b82ce311fa94bp-6
//     r = fma (r, s,  1.1283791670944182e-1); //  0x1.ce2f21a040d14p-4
//     r = fma (r, s, -3.7612638903183515e-1); // -0x1.812746b0379bcp-2
//     r = fma (r, s,  1.2837916709551256e-1); //  0x1.06eba8214db68p-3
// 
//     return fma (r, a, a);
// 
// }
// 
// // see: https://math.stackexchange.com/questions/42920/efficient-and-accurate-approximation-of-error-function
// inline double fast_erf_wo_checks (const double a) {
// 
//   double r, s, t, u;
//   t = fabs (a);
//   s = a * a;
//   if (t >= 1.0)    return fast_erf_wo_checks_part_1_upper(a, t, s);
//   else             return fast_erf_wo_checks_part_2_lower(a, t, s);
// 
// }
// 
// 




// ///////////////////  fns - Phi functions   ----------------------------------------------------------------------------------------------------------------------------------------
 
 
// inline double fast_Phi_wo_checks(const double x) {
// 
//    const double sqrt_2_recip = 0.707106781186547461715;
//    return 0.5 * (1.0 + fast_erf_wo_checks(x * sqrt_2_recip));
// 
// }
//  
 


//// ---- fast_Phi: relative-minimax rational approximation of the Mills ratio (2026-09-22, replaces Abramowitz & Stegun 26.2.17):
////
////   Phi(-z) = phi(z) * R(z),   R(z) = Phi(-z) / phi(z) = sqrt(pi/2) * erfcx(z / sqrt(2)),   z = |x|,
////   R(z) ~= P6(z) / Q7(z)  on z in [0, 37.5],   max relative error 7.8e-12 (measured on a dense grid in mpmath, double coefficients),
////   so  Phi(-z) = exp(-z^2/2) * N6(z) / Q7(z)  with  N6 = P6 / sqrt(2 pi)  (the 1/sqrt(2 pi) of phi is folded into N6),
////   and Phi(x) = 1 - Phi(-x) for x > 0 (the same reflection as before).
////
//// Why: A&S 26.2.17 (Zelen & Severo 1964 / Hastings 1955 form, published) has absolute error <= 7.5e-8 but RELATIVE error 5e-5 at
//// x = -3, 8e-3 at -7.5 and 0.16 at -37.5, while every caller registers the exact density phi(x) as the gradient, so value and
//// gradient disagreed in the lower tail (and log(fast_Phi) was off by up to 0.17 nats). The rational form is relative-accurate for
//// every x, branch-free, and costs the same as A&S: one exp, one divide and 13 FMAs (A&S: 1 FMA + 9 multiplies + 4 adds).
////
//// Provenance: the Mills-ratio/rational form is standard (e.g. Cody 1969, Math. Comp. 23:631-637, uses rationals for erfc);
//// THESE coefficients were fitted by an assistant (Claude, 2026-09-22) with a relative-error Sanathanan-Koerner / Lawson
//// near-minimax fit in 40-digit mpmath, constrained to N6(0) = 0.5 exactly so that fast_Phi(0) = 0.5 with no jump at x = 0.
//// FAST_PHI_MILLS_Z_CAP caps only the rational's argument (the exp still sees the true z): it keeps N and Q finite for
//// |x| = inf / huge |x| (exp(-z^2/2) = 0 there, so the result is 0 / 1 exactly as before) and does not change any x in [-37.5, 8.25].
#ifndef BAYESMVP_FAST_PHI_MILLS_COEFFICIENTS_DEFINED
#define BAYESMVP_FAST_PHI_MILLS_COEFFICIENTS_DEFINED
static constexpr double FAST_PHI_MILLS_Z_CAP = 38.5;
static constexpr double FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[7] = {   //// N6(z) = sum_k c[k] z^k   (includes 1/sqrt(2 pi))
        0.5,
        0.5968126365007661,
        0.34755100079419643,
        0.12199983678527791,
        0.027010741025906854,
        0.0035954377698120227,
        0.0002302422023077616 };
static constexpr double FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[8] = {   //// Q7(z) = sum_k d[k] z^k
        1.0,
        1.9915098343653488,
        1.7840969388752417,
        0.9377097714656843,
        0.314821789667328,
        0.06828298351478591,
        0.009012426603260822,
        0.0005771316095331759 };
static_assert(FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[0] == 0.5, "fast_Phi Mills rational: N(0) must be exactly 0.5 so that fast_Phi(0) = 0.5");
static_assert(FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[0] == 1.0, "fast_Phi Mills rational: Q(0) must be exactly 1");
static_assert(FAST_PHI_MILLS_Z_CAP >= 37.5, "fast_Phi Mills rational: the cap must not bind inside fast_Phi's range [-37.5, 8.25]");
#endif




//// Phi(x) = exp(-z^2/2) * N(z) / Q(z) for x < 0, 1 minus that for x >= 0 (z = |x|): see the FAST_PHI_MILLS_* block above (2026-09-22).
//// Same rational as fast_Phi_wo_checks_AVX2 / _AVX512. It uses std::exp, not the scalar fast_exp_1: fast_exp_1_wo_checks's
//// -log(2) split (l2h + l2l) is off by 4.7e-11, which gives exp(-z^2/2) a relative error of ~|i| * 4.7e-11 (i = the binary exponent),
//// i.e. 2e-9 at z = 7.5 and 5e-8 at z = 37.5; std::exp keeps the scalar kernel within ~1e-14 of the AVX kernels
//// (fast_exp_1_AVX2 / _AVX512 use the exact split -0.693145751953125 + -1.42860682030941723212e-6).
inline double fast_Phi_wo_checks(double x) {

      const double z = std::fabs(x);
      //// NaN x: the comparison is false, so z_rational = the cap (as _mm512_min_pd), and the NaN still propagates through std::exp.
      const double z_rational = (z < FAST_PHI_MILLS_Z_CAP) ? z : FAST_PHI_MILLS_Z_CAP;

      double numerator = FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[6];
      numerator = std::fma(numerator, z_rational, FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[5]);
      numerator = std::fma(numerator, z_rational, FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[4]);
      numerator = std::fma(numerator, z_rational, FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[3]);
      numerator = std::fma(numerator, z_rational, FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[2]);
      numerator = std::fma(numerator, z_rational, FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[1]);
      numerator = std::fma(numerator, z_rational, FAST_PHI_MILLS_NUMERATOR_COEFFICIENTS[0]);

      double denominator = FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[7];
      denominator = std::fma(denominator, z_rational, FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[6]);
      denominator = std::fma(denominator, z_rational, FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[5]);
      denominator = std::fma(denominator, z_rational, FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[4]);
      denominator = std::fma(denominator, z_rational, FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[3]);
      denominator = std::fma(denominator, z_rational, FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[2]);
      denominator = std::fma(denominator, z_rational, FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[1]);
      denominator = std::fma(denominator, z_rational, FAST_PHI_MILLS_DENOMINATOR_COEFFICIENTS[0]);

      const double lower_tail_probability = std::exp(-0.5 * z * z) * (numerator / denominator);   //// Phi(-z) = exp(-z^2/2) * N(z) / Q(z), same order as the AVX kernels
      double val;

      if (x >= 0.0) {

         val =  1.0 - lower_tail_probability;

      } else {

         val = lower_tail_probability;

      }


      /// ensure output is between 0 and 1
      if  ((val > 0) && (val < 1))  {
        return val;
      } else if (val >= 1) {   //// >= (2026-09-22): val == 1.0 (x > ~8.29) used to fall to the final branch and return 0
        return 1;
      } else {
        return 0;
      }

}




inline double fast_Phi(const double x) {

    const double sqrt_2_recip = 0.707106781186547461715;

    if (std::isnan(x)) return x;   //// preserve NaN (was returned as 1.0), as fast_Phi_AVX2 / _AVX512 already do (2026-09-22)

    if  ((x >= -37.5) && (x < 8.25)) {   //// >= : x == -37.5 used to fall through to the final branch and return 1.0 (2026-09-22)
      
          return  fast_Phi_wo_checks(x) ; // 0.5 * (1.0 + fast_erf_wo_checks(x * sqrt_2_recip));
      
    } else if (x < -37.5) {
      
          return 0.0;
      
    } else {  // if x > 8.25
      
          return 1.0;
       
    }

}









// ///////////////////  fns - inverse-error functions   ----------------------------------------------------------------------------------------------------------------------------------------
//  //  compute inverse error functions with maximum error of 2.35793 ulp  // see: https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
// 
//  inline double fast_inv_erf_wo_checks_part_1_upper(const double a,
//                                                    const double t) {
// 
//      double p =            3.03697567e-10; //  0x1.4deb44p-32
// 
//      p = fma (p, t,  2.93243101e-8); //  0x1.f7c9aep-26
//      p = fma (p, t,  1.22150334e-6); //  0x1.47e512p-20
//      p = fma (p, t,  2.84108955e-5); //  0x1.dca7dep-16
//      p = fma (p, t,  3.93552968e-4); //  0x1.9cab92p-12
//      p = fma (p, t,  3.02698812e-3); //  0x1.8cc0dep-9
//      p = fma (p, t,  4.83185798e-3); //  0x1.3ca920p-8
//      p = fma (p, t, -2.64646143e-1); // -0x1.0eff66p-2
//      p = fma (p, t,  8.40016484e-1); //  0x1.ae16a4p-1
// 
//      return a * p;
// 
//  }
// 
// //  compute inverse error functions with maximum error of 2.35793 ulp  // see: https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
// 
//  inline double fast_inv_erf_wo_checks_part_2_lower(const double a,
//                                                    const double t) {
// 
//      double p =             5.43877832e-9;  //  0x1.75c000p-28
// 
//      p = fma (p, t,  1.43285448e-7); //  0x1.33b402p-23
//      p = fma (p, t,  1.22774793e-6); //  0x1.499232p-20
//      p = fma (p, t,  1.12963626e-7); //  0x1.e52cd2p-24
//      p = fma (p, t, -5.61530760e-5); // -0x1.d70bd0p-15
//      p = fma (p, t, -1.47697632e-4); // -0x1.35be90p-13
//      p = fma (p, t,  2.31468678e-3); //  0x1.2f6400p-9
//      p = fma (p, t,  1.15392581e-2); //  0x1.7a1e50p-7
//      p = fma (p, t, -2.32015476e-1); // -0x1.db2aeep-3
//      p = fma (p, t,  8.86226892e-1); //  0x1.c5bf88p-1
// 
//      return a * p;
// 
//  }
// 
// //  compute inverse error functions with maximum error of 2.35793 ulp  // see: https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
// 
//  inline double fast_inv_erf_wo_checks(const double a) {
// 
//   double t = fma(a, 0.0 - a, 1.0);
//   t = fast_log_1(t);
// 
//   if (fabs(t) > 6.125) return fast_inv_erf_wo_checks_part_1_upper(a, t);
//   else                 return fast_inv_erf_wo_checks_part_2_lower(a, t);
// 
// }
// 
// 
// 
// 
// 
// ///////////////////  fns - inv_Phi functions   ----------------------------------------------------------------------------------------------------------------------------------------
// 
// inline double fast_inv_Phi_wo_checks(const double x) {
// 
//    const double sqrt_2 = 1.41421356237309514547;
//    return sqrt_2 * fast_inv_erf_wo_checks(2.0*x - 1.0);
// 
// }


/// FMA version of the  Stan fn provided by Sean Pinkney here:  https://github.com/stan-dev/math/issues/2555 
inline double  fast_inv_Phi_wo_checks(double p) {
  
  double r; 
  double val;
  const double q = p - 0.5;
  
  if (stan::math::abs(q) <= 0.425) { // CASE 1
    
    r = 0.180625 - q * q;
    
    double numerator =  fma(r, 2509.0809287301226727, 33430.575583588128105);
    numerator =  fma(numerator, r, 67265.770927008700853);
    numerator =  fma(numerator, r, 45921.953931549871457); 
    numerator =  fma(numerator, r, 13731.693765509461125); /// (fma_3 * r + 13731.693765509461125);
    numerator =  fma(numerator, r, 1971.5909503065514427); ///  (fma_4 * r +  1971.5909503065514427);
    numerator =  fma(numerator, r, 133.14166789178437745); /// (fma_5 * r + 133.14166789178437745);
    numerator =  fma(numerator, r, 3.387132872796366608); ///  (fma_6 * r + 3.387132872796366608)e 
    
    
    double denominator =  fma(r, 5226.495278852854561, 28729.085735721942674);
    denominator =  fma(denominator, r, 39307.89580009271061);
    denominator =  fma(denominator, r, 21213.794301586595867);
    denominator =  fma(denominator, r, 5394.1960214247511077);
    denominator =  fma(denominator, r, 687.1870074920579083);
    denominator =  fma(denominator, r, 42.313330701600911252);
    denominator =  fma(denominator, r, 1.0);
    
    val =  numerator / denominator;
    val = q*val;
    
    return val;
    
  } else { /* closer than 0.075 from {0,1} boundary */  //// CASE 2 
    
    //// first, compute r 
    if (q > 0.0) r = 1.0 - p;
    else r = p;
    r = stan::math::sqrt(-fast_log_1(r));
    
    
    /// then compute val 
    if (r <= 5.0) { /* <==> min(p,1-p) >= exp(-25) ~= 1.3888e-11 */  //// CASE 2(a)
      
      r += -1.60;
      
      double numerator = (0.00077454501427834140764);
      numerator = fma(r, numerator, (0.0227238449892691845833));
      numerator = fma(r, numerator, (0.24178072517745061177));
      numerator = fma(r, numerator, (1.27045825245236838258));
      numerator = fma(r, numerator, (3.64784832476320460504));
      numerator = fma(r, numerator, (5.7694972214606914055));
      numerator = fma(r, numerator, (4.6303378461565452959));
      numerator = fma(r, numerator, (1.42343711074968357734));
      
      double  denominator = (0.00000000105075007164441684324);
      denominator = fma(r,  denominator, (0.0005475938084995344946));
      denominator = fma(r,  denominator, (0.0151986665636164571966));
      denominator = fma(r,  denominator, (0.14810397642748007459));
      denominator = fma(r,  denominator, (0.68976733498510000455));
      denominator = fma(r,  denominator, (1.6763848301838038494));
      denominator = fma(r,  denominator, (2.05319162663775882187));
      denominator = fma(r, denominator,  (1.0));
      
      val = numerator / denominator;
      
      
    } else { /* very close to  0 or 1 */      //// CASE 2(b)
       
      r += -5.0;
      
      double numerator =  (0.000000201033439929228813265);
      numerator = fma(r, numerator,  (0.0000271155556874348757815));
      numerator = fma(r, numerator,  (0.0012426609473880784386));
      numerator = fma(r, numerator,  (0.026532189526576123093));
      numerator = fma(r, numerator,  (0.29656057182850489123));
      numerator = fma(r, numerator,  (1.7848265399172913358));
      numerator = fma(r, numerator,  (5.4637849111641143699));
      numerator = fma(r, numerator,  (6.6579046435011037772));
      
      double denominator =  (0.00000000000000204426310338993978564);
      denominator = fma(r, denominator,  (0.00000014215117583164458887));
      denominator = fma(r, denominator,  (0.000018463183175100546818));
      denominator = fma(r, denominator,  (0.0007868691311456132591));
      denominator = fma(r, denominator,  (0.0148753612908506148525));
      denominator = fma(r, denominator,  (0.13692988092273580531));
      denominator = fma(r, denominator,  (0.59983220655588793769));
      denominator = fma(r, denominator,  (1.0));
      
      val = numerator / denominator;
      
    }
    
    if (q < 0.0) val = -val;
    
    return val;
    
  } 
  
  
}


 
 
 
 /////////////////// other  ----------------------------------------------------------------------------------------------------------------------------------------
  
 
inline Eigen::Matrix<double, 1, -1>     fn_first_element_neg_rest_pos( Eigen::Ref<Eigen::Matrix<double, 1, -1>>  row_vec    ) {

   row_vec(0) = - row_vec(0);

   return(row_vec);

 }

 

 
 
 
#endif
 
 
 
 
 
 
 
 
 
 
 
 
 