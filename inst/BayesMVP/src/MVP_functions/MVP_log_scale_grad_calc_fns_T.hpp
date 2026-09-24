
//// MVP_log_scale_grad_calc_fns_T.hpp
////
//// Templated (<Vec vec>) versions of everything in MVP_log_scale_grad_calc_fns.hpp, plus the
//// three signed log-sum-exp utilities those functions need. Every function here ends in _T and
//// takes NO Model_fn_args_struct / vect_type strings; your originals are untouched.
////
//// Requires: fn_dispatch_templated.hpp, MVP_helpers_migrated.hpp (KernelChoice, apply_*).
////
//// Maths is identical to your originals. Only the element-wise calls changed:
////   fn_EIGEN_double(x, "log", vt)  ->  apply_inplace<vec, Fn::log>(x)      (in place)
////   log_abs_sum_exp_general_v2(A, S, vt, vt, out_log, out_sign, mx, sm) -> log_abs_sum_exp_general_v2_T<vec>(A, S, out_log, out_sign, mx, sm)
////   log_sum_vec_signed_v1(la, sg, vt)  ->  log_sum_vec_signed_T<vec>(la, sg)
////   fn_log_sum_exp_2d_double(M, vt)    ->  fn_log_sum_exp_2d_T<vec>(M)

#pragma once
// #include "MVP_helpers_migrated.hpp"

//// =====================================================================================
//// 0. Signed log-sum-exp utilities
//// =====================================================================================

//// Row-wise: out_log = log|sum_j sign_j exp(log_j)|, out_sign = sign(sum). All k columns of
//// log_terms/sign_terms are used (pass .leftCols(k) at the call site). out_log doubles as scratch.
template <Vec vec>
inline void log_abs_sum_exp_general_v2_T(  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_terms,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_terms,
                                           Eigen::Matrix<double, -1, 1> &out_log,
                                           Eigen::Matrix<double, -1, 1> &out_sign,
                                           Eigen::Matrix<double, -1, 1> &container_max_logs,
                                           Eigen::Matrix<double, -1, 1> &container_sum_exp_signed
) {
        
        const int n = static_cast<int>(log_terms.rows());
        const int k = static_cast<int>(log_terms.cols());
        container_max_logs = log_terms.rowwise().maxCoeff();
        container_sum_exp_signed.setZero();
        for (int j = 0; j < k; ++j) {
          out_log = log_terms.col(j) - container_max_logs;                 //// scratch
          apply_inplace<vec, Fn::exp>(out_log);
          container_sum_exp_signed.array() += sign_terms.col(j).array() * out_log.array();
        }
        for (int i = 0; i < n; ++i) {
          const double s = container_sum_exp_signed(i);
          out_sign(i) = (s > 0.0) ? 1.0 : ((s < 0.0) ? -1.0 : 1.0);
          out_log(i)  = std::abs(s);
        }
        apply_inplace<vec, Fn::log>(out_log);                                //// log(0) -> -Inf is fine (Stan log)
        for (int i = 0; i < n; ++i) if (!std::isfinite(out_log(i))) out_log(i) = -700.0;
        out_log.array() += container_max_logs.array();
  
}




//// Whole-vector signed sum: log|sum_i sign_i exp(log_i)| and its sign.
struct LogSumSignedResult_T { double log_sum; double sign; };

template <Vec vec>
inline LogSumSignedResult_T log_sum_vec_signed_T(  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> log_abs,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> sign
) {
  
        const double mx = log_abs.maxCoeff();
        //// every term is exactly zero (log_abs = -Inf throughout, e.g. the per-row gradients w.r.t. L_Omega(0, 0)
        //// when test 1's bound is exactly 0). Before this guard, -Inf - (-Inf) = NaN made the whole sum NaN, and the chain rule then turned the
        //// correlation gradient into NaN (0 * NaN). Return the same "log zero" as the s == 0 case below.
        if (std::isinf(mx) && (mx < 0.0)) {
          LogSumSignedResult_T r_all_zero;
          r_all_zero.sign    = 1.0;
          r_all_zero.log_sum = -700.0;
          return r_all_zero;
        }
        Eigen::Matrix<double, -1, 1> tmp = log_abs.array() - mx;
        apply_inplace<vec, Fn::exp>(tmp);
        const double s = (sign.array() * tmp.array()).sum();
        LogSumSignedResult_T r;
        r.sign    = (s < 0.0) ? -1.0 : 1.0;
        r.log_sum = (s == 0.0) ? -700.0 : std::log(std::abs(s)) + mx;
        return r;
  
}




//// Row-wise log-sum-exp of an n x 2 matrix (all positive terms).
template <Vec vec>
inline Eigen::Matrix<double, -1, 1> fn_log_sum_exp_2d_T( const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> M) {
  
        Eigen::Matrix<double, -1, 1> mx = M.rowwise().maxCoeff();
        Eigen::Matrix<double, -1, 1> a = M.col(0) - mx;  apply_inplace<vec, Fn::exp>(a);
        Eigen::Matrix<double, -1, 1> b = M.col(1) - mx;  apply_inplace<vec, Fn::exp>(b);
        Eigen::Matrix<double, -1, 1> out = a + b;
        apply_inplace<vec, Fn::log>(out);
        out.array() += mx.array();
        return out;
  
}




//// =====================================================================================
//// 1a. Exact normal-tail kernels over a gathered vector (tail rows only).
////
//// The tail fix-ups below
//// previously used the Phi_approx pair (log_Phi_approx + inv_Phi_approx_from_logit_prob) for
//// EVERY Phi_type, which spliced the exact interior (Phi_type = "Phi") onto cubic-logistic tails
//// at overflow_threshold / underflow_threshold (a jump of ~10.7 nats in log Phi at -7.5).
//// For the exact setting these two helpers supply the exact log-space functions instead:
////
////   fn_apply_log_Phi_exact_inplace_T:      x -> log Phi(x)       (fast_log_Phi_AVX512 / _AVX2 / scalar fast_log_Phi;
////                                                                 Mills-ratio continued fraction for x < -2.5, so
////                                                                 relative-accurate at every tail argument)
////   fn_inv_Phi_from_log_p_exact_T:         (log p, log(1-p)) -> Phi^{-1}(p)
////                                                                (AS241 / Wichura 1988, fed r = sqrt(-log min(p, 1-p))
////                                                                 directly, so no exp/log of an extreme probability)
////
//// The kernels are the existing ones in inst/include/BayesMVP/math/fast_and_approx_AVX*_fns.hpp and
//// general_functions/double_fns.hpp. The level falls back AVX512 -> AVX2 -> scalar exactly as apply_raw
//// does; the scalar remainder (and Vec::Scalar) uses the double versions.
//// =====================================================================================
template <Vec vec>
inline void fn_apply_log_Phi_exact_inplace_T(Eigen::Matrix<double, -1, 1> &values_in_log_Phi_out) {
  
        double *data_pointer = values_in_log_Phi_out.data();
        const int n_values = static_cast<int>(values_in_log_Phi_out.size());
        int index_value = 0;
        if constexpr (vec == Vec::AVX512) {
          #if BMVP_HAS_AVX512
            for (; index_value + 8 <= n_values; index_value += 8) {
              _mm512_storeu_pd(data_pointer + index_value, fast_log_Phi_AVX512(_mm512_loadu_pd(data_pointer + index_value)));
            }
          #elif BMVP_HAS_AVX2
            for (; index_value + 4 <= n_values; index_value += 4) {
              _mm256_storeu_pd(data_pointer + index_value, fast_log_Phi_AVX2(_mm256_loadu_pd(data_pointer + index_value)));
            }
          #endif
        } else if constexpr (vec == Vec::AVX2) {
          #if BMVP_HAS_AVX2
            for (; index_value + 4 <= n_values; index_value += 4) {
              _mm256_storeu_pd(data_pointer + index_value, fast_log_Phi_AVX2(_mm256_loadu_pd(data_pointer + index_value)));
            }
          #endif
        }
        for (; index_value < n_values; ++index_value) {
          data_pointer[index_value] = fast_log_Phi(data_pointer[index_value]);
        }
  
}




template <Vec vec>
inline void fn_inv_Phi_from_log_p_exact_T(  const Eigen::Matrix<double, -1, 1> &log_p_vec,
                                            const Eigen::Matrix<double, -1, 1> &log_1m_p_vec,
                                            Eigen::Matrix<double, -1, 1> &Z_out_vec
) {
  
        const int n_values = static_cast<int>(log_p_vec.size());
        Z_out_vec.resize(n_values);
        const double *log_p_pointer = log_p_vec.data();
        const double *log_1m_p_pointer = log_1m_p_vec.data();
        double *Z_out_pointer = Z_out_vec.data();
        int index_value = 0;
        if constexpr (vec == Vec::AVX512) {
          #if BMVP_HAS_AVX512
            for (; index_value + 8 <= n_values; index_value += 8) {
              _mm512_storeu_pd(Z_out_pointer + index_value,
                               fast_inv_Phi_from_log_p_wo_checks_AVX512(_mm512_loadu_pd(log_p_pointer + index_value),
                                                                        _mm512_loadu_pd(log_1m_p_pointer + index_value)));
            }
          #elif BMVP_HAS_AVX2
            for (; index_value + 4 <= n_values; index_value += 4) {
              _mm256_storeu_pd(Z_out_pointer + index_value,
                               fast_inv_Phi_from_log_p_wo_checks_AVX2(_mm256_loadu_pd(log_p_pointer + index_value),
                                                                      _mm256_loadu_pd(log_1m_p_pointer + index_value)));
            }
          #endif
        } else if constexpr (vec == Vec::AVX2) {
          #if BMVP_HAS_AVX2
            for (; index_value + 4 <= n_values; index_value += 4) {
              _mm256_storeu_pd(Z_out_pointer + index_value,
                               fast_inv_Phi_from_log_p_wo_checks_AVX2(_mm256_loadu_pd(log_p_pointer + index_value),
                                                                      _mm256_loadu_pd(log_1m_p_pointer + index_value)));
            }
          #endif
        }
        for (; index_value < n_values; ++index_value) {
          Z_out_pointer[index_value] = fast_inv_Phi_from_log_p(log_p_pointer[index_value], log_1m_p_pointer[index_value]);
        }
  
}




//// =====================================================================================
//// 1. GHK log-scale fix-ups for the problem rows (index). Eigen indexed views need a
////    temporary per expression; those allocations are unavoidable and small (|index| rows).
////
////    Both functions now take the KernelChoice.
////     - Phi_approx AND inv_Phi_approx (the fully approximate setting): the original code,
////       unchanged, runs first and returns.
////     - otherwise the tail CDF follows kernel_choice.Phi_approx and the tail inverse follows
////       kernel_choice.inv_Phi_approx, each with its OWN matching derivative, so the value and
////       the manual gradient stay consistent in every combination:
////
////       exact CDF (Phi_type = "Phi"):
////         underflow (y = 0, Bound_Z < underflow_threshold):  y1 = log Phi(Bound_Z)
////         overflow  (y = 1, Bound_Z > overflow_threshold):   y1 = log(1 - Phi(Bound_Z)) = log Phi(-Bound_Z)
////         |d prob / d Bound_Z| = phi(Bound_Z)   =>  log_phi_Bound_Z = -Bound_Z^2/2 - 0.5 log(2 pi)
////       exact inverse (inv_Phi_type = "inv_Phi"):
////         Z = Phi^{-1}(Phi_Z) from (log Phi_Z, log(1 - Phi_Z))
////         dZ / d Phi_Z = 1 / phi(Z)             =>  log_phi_Z_recip = +Z^2/2 + 0.5 log(2 pi)
////
////       These are the same derivative quantities the standard-scale pass uses for the exact
////       setting (fn_MVP_compute_phi_Bound_Z_cols_T / fn_MVP_compute_phi_Z_recip_cols_T), and the
////       downstream GHK chain rule consumes only y1_log_prob, log_phi_Bound_Z, log_phi_Z_recip,
////       Z_std_norm and u, so no other gradient code needs to change:
////         d y1 / d Bound_Z   = (1 - 2y) phi(Bound_Z) / prob                [underflow: +phi(B)/Phi(B); overflow: -phi(B)/(1-Phi(B))]
////         d Z / d Bound_Z    = (y - (2y - 1) u) phi(Bound_Z) / phi(Z)      [underflow: u phi(B)/phi(Z); overflow: (1-u) phi(B)/phi(Z)]
////         d Z / d u          = prob / phi(Z)                               [underflow: Phi(B)/phi(Z); overflow: (1-Phi(B))/phi(Z)]
//// =====================================================================================
template <Vec vec>
inline void fn_MVP_compute_lp_GHK_cols_log_scale_underflow_T(  const int t,
                                                               const std::vector<int> &index,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_Z_std_norm,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_phi_Bound_Z,
                                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array,
                                                               const KernelChoice &kernel_choice
) {
  
        const double a_times_3 = 3.0 * 0.07056, b = 1.5976;
        typedef Eigen::Matrix<double, -1, 1> V;
        
        if (kernel_choice.Phi_approx && kernel_choice.inv_Phi_approx) {
          
                //// ---- fully approximate setting: original code, unchanged ----
                V log_Bound_U_Phi_Bound_Z = Bound_Z(index, t);
                apply_inplace<vec, Fn::log_Phi_approx>(log_Bound_U_Phi_Bound_Z);
                V tmp = log_Bound_U_Phi_Bound_Z;  apply_inplace<vec, Fn::exp>(tmp);
                Bound_U_Phi_Bound_Z(index, t) = tmp;
              
                V u_log = u_array(index, t);  apply_inplace<vec, Fn::log>(u_log);
                V log_Phi_Z = u_log + log_Bound_U_Phi_Bound_Z;
                tmp = log_Phi_Z;  apply_inplace<vec, Fn::exp>(tmp);
                Phi_Z(index, t) = tmp;
              
                V log_1m_Phi_Z = Bound_U_Phi_Bound_Z(index, t).array() * u_array(index, t).array();
                apply_inplace<vec, Fn::log1m>(log_1m_Phi_Z);
                V logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
                apply_inplace<vec, Fn::inv_Phi_approx_from_logit_prob>(logit_Phi_Z);
                Z_std_norm(index, t) = logit_Phi_Z;
              
                tmp = logit_Phi_Z.cwiseAbs();  apply_inplace<vec, Fn::log>(tmp);
                log_Z_std_norm(index, t) = tmp;
              
                y1_log_prob(index, t) = log_Bound_U_Phi_Bound_Z;
                prob(index, t)        = Bound_U_Phi_Bound_Z(index, t);
              
                V log_Bound_U_Phi_Bound_Z_1m = Bound_U_Phi_Bound_Z(index, t);
                apply_inplace<vec, Fn::log1m>(log_Bound_U_Phi_Bound_Z_1m);
              
                tmp = Bound_Z(index, t);
                tmp.array() = a_times_3 * tmp.array().square() + b;
                apply_inplace<vec, Fn::log>(tmp);
                tmp.array() += log_Bound_U_Phi_Bound_Z.array() + log_Bound_U_Phi_Bound_Z_1m.array();
                log_phi_Bound_Z(index, t) = tmp;
              
                tmp = Z_std_norm(index, t);
                tmp.array() = a_times_3 * tmp.array().square() + b;
                apply_inplace<vec, Fn::log>(tmp);
                tmp.array() += log_Phi_Z.array() + log_1m_Phi_Z.array();
                log_phi_Z_recip(index, t) = -tmp;
                
                return;
          
        }
        
        ////
        //// ---- exact CDF and/or exact inverse (Phi_type = "Phi" and/or inv_Phi_type = "inv_Phi"):
        ////
        const double half_log_two_pi = 0.91893853320467274178;   //// 0.5 * log(2 pi)
        
        {
          
                ////
                //// ---- log Phi(Bound_Z): exact log-space normal CDF, or the Phi_approx tail:
                ////
                V log_Phi_Bound_Z_vec = Bound_Z(index, t);
                if (!kernel_choice.Phi_approx) fn_apply_log_Phi_exact_inplace_T<vec>(log_Phi_Bound_Z_vec);
                else                           apply_inplace<vec, Fn::log_Phi_approx>(log_Phi_Bound_Z_vec);
                V Phi_Bound_Z_vec = log_Phi_Bound_Z_vec;
                apply_inplace<vec, Fn::exp>(Phi_Bound_Z_vec);
                Bound_U_Phi_Bound_Z(index, t) = Phi_Bound_Z_vec;
                
                ////
                //// ---- Phi_Z = u * Phi(Bound_Z), held on the log scale:
                ////
                V log_u_vec = u_array(index, t);
                apply_inplace<vec, Fn::log>(log_u_vec);
                V log_Phi_Z_vec = log_u_vec + log_Phi_Bound_Z_vec;
                V Phi_Z_vec = log_Phi_Z_vec;
                apply_inplace<vec, Fn::exp>(Phi_Z_vec);
                Phi_Z(index, t) = Phi_Z_vec;
                V log_1m_Phi_Z_vec = Phi_Bound_Z_vec.array() * u_array(index, t).array();    //// tiny argument, so log1m is accurate
                apply_inplace<vec, Fn::log1m>(log_1m_Phi_Z_vec);
                
                ////
                //// ---- Z = inverse CDF of Phi_Z: exact from (log Phi_Z, log(1 - Phi_Z)), or the Phi_approx inverse:
                ////
                V Z_vec;
                if (!kernel_choice.inv_Phi_approx) {
                  fn_inv_Phi_from_log_p_exact_T<vec>(log_Phi_Z_vec, log_1m_Phi_Z_vec, Z_vec);
                } else {
                  Z_vec = log_Phi_Z_vec - log_1m_Phi_Z_vec;
                  apply_inplace<vec, Fn::inv_Phi_approx_from_logit_prob>(Z_vec);
                }
                Z_std_norm(index, t) = Z_vec;
                V log_abs_Z_vec = Z_vec.cwiseAbs();
                apply_inplace<vec, Fn::log>(log_abs_Z_vec);
                log_Z_std_norm(index, t) = log_abs_Z_vec;
                
                y1_log_prob(index, t) = log_Phi_Bound_Z_vec;
                prob(index, t)        = Phi_Bound_Z_vec;
                
                ////
                //// ---- log |d prob / d Bound_Z|: exact log phi(Bound_Z), or the Phi_approx derivative:
                ////
                V log_phi_Bound_Z_vec = Bound_Z(index, t);
                if (!kernel_choice.Phi_approx) {
                  log_phi_Bound_Z_vec.array() = -0.5 * log_phi_Bound_Z_vec.array().square() - half_log_two_pi;
                } else {
                  V log_1m_Phi_Bound_Z_vec = Phi_Bound_Z_vec;
                  apply_inplace<vec, Fn::log1m>(log_1m_Phi_Bound_Z_vec);
                  log_phi_Bound_Z_vec.array() = a_times_3 * log_phi_Bound_Z_vec.array().square() + b;
                  apply_inplace<vec, Fn::log>(log_phi_Bound_Z_vec);
                  log_phi_Bound_Z_vec.array() += log_Phi_Bound_Z_vec.array() + log_1m_Phi_Bound_Z_vec.array();
                }
                log_phi_Bound_Z(index, t) = log_phi_Bound_Z_vec;
                
                ////
                //// ---- log(dZ / d Phi_Z): exact -log phi(Z), or the Phi_approx inverse derivative:
                ////
                V log_phi_Z_recip_vec = Z_vec;
                if (!kernel_choice.inv_Phi_approx) {
                  log_phi_Z_recip_vec.array() = 0.5 * log_phi_Z_recip_vec.array().square() + half_log_two_pi;
                } else {
                  log_phi_Z_recip_vec.array() = a_times_3 * log_phi_Z_recip_vec.array().square() + b;
                  apply_inplace<vec, Fn::log>(log_phi_Z_recip_vec);
                  log_phi_Z_recip_vec.array() += log_Phi_Z_vec.array() + log_1m_Phi_Z_vec.array();
                  log_phi_Z_recip_vec = -log_phi_Z_recip_vec;
                }
                log_phi_Z_recip(index, t) = log_phi_Z_recip_vec;
          
        }
  
}




template <Vec vec>
inline void fn_MVP_compute_lp_GHK_cols_log_scale_overflow_T(  const int t,
                                                              const int num_overflows,
                                                              const std::vector<int> &index,
                                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
                                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
                                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_Z_std_norm,
                                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
                                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_phi_Bound_Z,
                                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
                                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array,
                                                              const KernelChoice &kernel_choice
) {
  
        const double a_times_3 = 3.0 * 0.07056, b = 1.5976;
        typedef Eigen::Matrix<double, -1, 1> V;
        
        if (kernel_choice.Phi_approx && kernel_choice.inv_Phi_approx) {
          
                //// ---- fully approximate setting: original code, unchanged ----
                V log_Bound_U_Phi_Bound_Z_1m = -Bound_Z(index, t);
                apply_inplace<vec, Fn::log_Phi_approx>(log_Bound_U_Phi_Bound_Z_1m);
                V Bound_U_Phi_Bound_Z_1m = log_Bound_U_Phi_Bound_Z_1m;  apply_inplace<vec, Fn::exp>(Bound_U_Phi_Bound_Z_1m);
                V log_Bound_U_Phi_Bound_Z = Bound_U_Phi_Bound_Z_1m;     apply_inplace<vec, Fn::log1m>(log_Bound_U_Phi_Bound_Z);
              
                Bound_U_Phi_Bound_Z(index, t).array() = 1.0 - Bound_U_Phi_Bound_Z_1m.array();
              
                Eigen::Matrix<double, -1, -1> lse2(num_overflows, 2);
                V u_log = u_array(index, t);  apply_inplace<vec, Fn::log>(u_log);
                lse2.col(0) = log_Bound_U_Phi_Bound_Z_1m + u_log;
                lse2.col(1) = log_Bound_U_Phi_Bound_Z;
                V log_Phi_Z = fn_log_sum_exp_2d_T<vec>(lse2);
              
                V tmp = log_Phi_Z;  apply_inplace<vec, Fn::exp>(tmp);
                Phi_Z(index, t) = tmp;
              
                V log_1m_Phi_Z = u_array(index, t);  apply_inplace<vec, Fn::log1m>(log_1m_Phi_Z);
                log_1m_Phi_Z.array() += log_Bound_U_Phi_Bound_Z_1m.array();
              
                V logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
                apply_inplace<vec, Fn::inv_Phi_approx_from_logit_prob>(logit_Phi_Z);
                Z_std_norm(index, t) = logit_Phi_Z;
                tmp = logit_Phi_Z.cwiseAbs();  apply_inplace<vec, Fn::log>(tmp);
                log_Z_std_norm(index, t) = tmp;
              
                y1_log_prob(index, t) = log_Bound_U_Phi_Bound_Z_1m;
                prob(index, t)        = Bound_U_Phi_Bound_Z_1m;
              
                tmp = Bound_Z(index, t);
                tmp.array() = a_times_3 * tmp.array().square() + b;
                apply_inplace<vec, Fn::log>(tmp);
                tmp.array() += log_Bound_U_Phi_Bound_Z.array() + log_Bound_U_Phi_Bound_Z_1m.array();
                log_phi_Bound_Z(index, t) = tmp;
              
                tmp = Z_std_norm(index, t);
                tmp.array() = a_times_3 * tmp.array().square() + b;
                apply_inplace<vec, Fn::log>(tmp);
                tmp.array() += log_Phi_Z.array() + log_1m_Phi_Z.array();
                log_phi_Z_recip(index, t) = -tmp;
                
                return;
          
        }
        
        ////
        //// ---- exact CDF and/or exact inverse (Phi_type = "Phi" and/or inv_Phi_type = "inv_Phi"):
        ////
        const double half_log_two_pi = 0.91893853320467274178;   //// 0.5 * log(2 pi)
        
        {
          
                ////
                //// ---- log(1 - Phi(Bound_Z)) = log Phi(-Bound_Z) (reflection, so no 1 - Phi cancellation): exact, or the Phi_approx tail:
                ////
                V log_1m_Phi_Bound_Z_vec = -Bound_Z(index, t);
                if (!kernel_choice.Phi_approx) fn_apply_log_Phi_exact_inplace_T<vec>(log_1m_Phi_Bound_Z_vec);
                else                           apply_inplace<vec, Fn::log_Phi_approx>(log_1m_Phi_Bound_Z_vec);
                V one_m_Phi_Bound_Z_vec = log_1m_Phi_Bound_Z_vec;
                apply_inplace<vec, Fn::exp>(one_m_Phi_Bound_Z_vec);
                V log_Phi_Bound_Z_vec = one_m_Phi_Bound_Z_vec;      //// log(1 - tiny): accurate
                apply_inplace<vec, Fn::log1m>(log_Phi_Bound_Z_vec);
                Bound_U_Phi_Bound_Z(index, t).array() = 1.0 - one_m_Phi_Bound_Z_vec.array();
                
                ////
                //// ---- Phi_Z = Phi(Bound_Z) + u (1 - Phi(Bound_Z)):  log Phi_Z = LSE(log(1 - Phi(B)) + log u, log Phi(B)),
                ////      log(1 - Phi_Z) = log(1 - u) + log(1 - Phi(B))   (no cancellation):
                ////
                Eigen::Matrix<double, -1, -1> log_sum_exp_input_mat(num_overflows, 2);
                V log_u_vec = u_array(index, t);
                apply_inplace<vec, Fn::log>(log_u_vec);
                log_sum_exp_input_mat.col(0) = log_1m_Phi_Bound_Z_vec + log_u_vec;
                log_sum_exp_input_mat.col(1) = log_Phi_Bound_Z_vec;
                V log_Phi_Z_vec = fn_log_sum_exp_2d_T<vec>(log_sum_exp_input_mat);
                V Phi_Z_vec = log_Phi_Z_vec;
                apply_inplace<vec, Fn::exp>(Phi_Z_vec);
                Phi_Z(index, t) = Phi_Z_vec;
                V log_1m_Phi_Z_vec = u_array(index, t);
                apply_inplace<vec, Fn::log1m>(log_1m_Phi_Z_vec);
                log_1m_Phi_Z_vec.array() += log_1m_Phi_Bound_Z_vec.array();
                
                ////
                //// ---- Z = inverse CDF of Phi_Z: exact from (log Phi_Z, log(1 - Phi_Z)) (upper-tail case driven by log(1 - Phi_Z)),
                ////      or the Phi_approx inverse:
                ////
                V Z_vec;
                if (!kernel_choice.inv_Phi_approx) {
                  fn_inv_Phi_from_log_p_exact_T<vec>(log_Phi_Z_vec, log_1m_Phi_Z_vec, Z_vec);
                } else {
                  Z_vec = log_Phi_Z_vec - log_1m_Phi_Z_vec;
                  apply_inplace<vec, Fn::inv_Phi_approx_from_logit_prob>(Z_vec);
                }
                Z_std_norm(index, t) = Z_vec;
                V log_abs_Z_vec = Z_vec.cwiseAbs();
                apply_inplace<vec, Fn::log>(log_abs_Z_vec);
                log_Z_std_norm(index, t) = log_abs_Z_vec;
                
                y1_log_prob(index, t) = log_1m_Phi_Bound_Z_vec;
                prob(index, t)        = one_m_Phi_Bound_Z_vec;
                
                ////
                //// ---- log |d prob / d Bound_Z| = log phi(Bound_Z) (exact), or the Phi_approx derivative:
                ////
                V log_phi_Bound_Z_vec = Bound_Z(index, t);
                if (!kernel_choice.Phi_approx) {
                  log_phi_Bound_Z_vec.array() = -0.5 * log_phi_Bound_Z_vec.array().square() - half_log_two_pi;
                } else {
                  log_phi_Bound_Z_vec.array() = a_times_3 * log_phi_Bound_Z_vec.array().square() + b;
                  apply_inplace<vec, Fn::log>(log_phi_Bound_Z_vec);
                  log_phi_Bound_Z_vec.array() += log_Phi_Bound_Z_vec.array() + log_1m_Phi_Bound_Z_vec.array();
                }
                log_phi_Bound_Z(index, t) = log_phi_Bound_Z_vec;
                
                ////
                //// ---- log(dZ / d Phi_Z) = -log phi(Z) (exact), or the Phi_approx inverse derivative:
                ////
                V log_phi_Z_recip_vec = Z_vec;
                if (!kernel_choice.inv_Phi_approx) {
                  log_phi_Z_recip_vec.array() = 0.5 * log_phi_Z_recip_vec.array().square() + half_log_two_pi;
                } else {
                  log_phi_Z_recip_vec.array() = a_times_3 * log_phi_Z_recip_vec.array().square() + b;
                  apply_inplace<vec, Fn::log>(log_phi_Z_recip_vec);
                  log_phi_Z_recip_vec.array() += log_Phi_Z_vec.array() + log_1m_Phi_Z_vec.array();
                  log_phi_Z_recip_vec = -log_phi_Z_recip_vec;
                }
                log_phi_Z_recip(index, t) = log_phi_Z_recip_vec;
          
        }
  
}

//// =====================================================================================
//// 2. grad prep on the log scale
//// =====================================================================================
template <Vec vec>
inline void fn_MVP_grad_prep_log_scale_T(  Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_prob_rowwise_prod_temp,
                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_prob_recip_rowwise_prod_temp,
                                           Eigen::Ref<Eigen::Matrix<double, -1, 1>>  log_prob_rowwise_prod_temp_all,
                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_common_grad_term_1,
                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y1_log_prob_recip,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  log_prob_n_recip,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  log_prev_per_obs_given_c,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Bound_Z,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_recip_double,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_L_Omega_recip_double,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_sign_chunk,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_m_y_sign_x_u,
                                           const int n_class
) {
  
        const int n_tests = y1_log_prob.cols();
        const int chunk_size = y1_log_prob.rows();
      
        for (int i = 0; i < n_tests; i++) {
          const int t = n_tests - (i + 1);
          log_prob_rowwise_prod_temp.col(t)       = y1_log_prob.block(0, t, chunk_size, i + 1).rowwise().sum();
          log_prob_recip_rowwise_prod_temp.col(t) = y1_log_prob_recip.block(0, t, chunk_size, i + 1).rowwise().sum();
        }
        log_prob_rowwise_prod_temp_all = y1_log_prob.rowwise().sum();
      
        if (n_class > 1) {
          for (int i = 0; i < n_tests; i++) {
            const int t = n_tests - (i + 1);
            log_common_grad_term_1.col(t).array() = log_prob_n_recip.array() + log_prev_per_obs_given_c.array()
                                                  + log_prob_rowwise_prod_temp_all.array()
                                                  + log_prob_recip_rowwise_prod_temp.col(t).array();
          }
        } else {
          //// was log_common_grad_term_1.setConstant(-700.0), a sentinel that the problem-row
          //// replacement functions then used as if it were the real term, zeroing the log-scale nuisance / L_Omega
          //// gradients of every problem row of tests 2..n_tests in the standard (single-class) MVP. With one class,
          //// prev = 1 and the likelihood prob_n is the product of ALL the test probabilities, so the latent-class
          //// expression above, log(1 / prob_n) + log(prev) + log(prod_all) + log(prod_{s >= t} 1 / prob_s), reduces
          //// exactly to log(prod_{s >= t} 1 / prob_s). That is used directly here (it avoids the -lp + lp cancellation,
          //// which would give NaN if a row's lp were -inf).
          for (int i = 0; i < n_tests; i++) {
            const int t = n_tests - (i + 1);
            log_common_grad_term_1.col(t) = log_prob_recip_rowwise_prod_temp.col(t);
          }
        }
      
        Eigen::Matrix<double, -1, 1> log_abs_col(chunk_size);
        for (int t = 0; t < n_tests; t++) {
          const double L_log_abs = log_abs_L_Omega_recip_double(t, t);
          const double L_sign    = sign_L_Omega_recip_double(t, t);
      
          log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array() = log_phi_Bound_Z.col(t).array() + L_log_abs;
      
          log_abs_col = y_m_y_sign_x_u.col(t).cwiseAbs();
          apply_inplace<vec, Fn::log>(log_abs_col);
          log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array() =
              log_abs_col.array() + log_phi_Z_recip.col(t).array() + log_phi_Bound_Z.col(t).array() + L_log_abs;
      
          sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t) = y_sign_chunk.col(t).array().sign() * L_sign;
          sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t) = y_m_y_sign_x_u.col(t).array().sign() * L_sign;
        }
  
}




//// Small helper used by the three grad functions: exp() of an indexed view, times a sign, into an indexed view.
template <Vec vec>
ALWAYS_INLINE void exp_indexed_times_sign_T(Eigen::Matrix<double, -1, -1> &dst, const int col, const std::vector<int> &idx,
                                            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_src, const int src_col,
                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> sign) {
  
        Eigen::Matrix<double, -1, 1> tmp = log_abs_src.col(src_col)(idx);
        apply_inplace<vec, Fn::exp>(tmp);
        dst.col(col)(idx) = tmp.array() * sign.array();
  
}




//// =====================================================================================
//// 3. nuisance grad on the log scale (problem rows only)
//// =====================================================================================
template <Vec vec>
inline void fn_MVP_compute_nuisance_grad_log_scale_T(  const std::vector<int> &n_problem_array,
                                                       const std::vector<std::vector<int>> &problem_index_array,
                                                       Eigen::Matrix<double, -1, -1> &log_abs_u_grad_array_CM_chunk,
                                                       Eigen::Matrix<double, -1, -1> &u_grad_array_CM_chunk,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_double,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_prob_recip,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_prob_rowwise_prod_temp,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_ys,     //// log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_ys,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_ym,     //// log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_ym,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_common_grad_term_1,
                                                       Eigen::Matrix<double, -1, -1> &log_abs_z_grad_term,
                                                       Eigen::Matrix<double, -1, -1> &sign_z_grad_term,
                                                       Eigen::Matrix<double, -1, -1> &log_abs_grad_prob,
                                                       Eigen::Matrix<double, -1, -1> &sign_grad_prob,
                                                       Eigen::Matrix<double, -1, 1>  &log_abs_prod,
                                                       Eigen::Matrix<double, -1, 1>  &sign_prod,
                                                       Eigen::Matrix<double, -1, 1>  &log_sum_result,
                                                       Eigen::Matrix<double, -1, 1>  &sign_sum_result,
                                                       Eigen::Matrix<double, -1, -1> &log_terms,
                                                       Eigen::Matrix<double, -1, -1> &sign_terms,
                                                       Eigen::Matrix<double, -1, 1>  &log_abs_a,
                                                       Eigen::Matrix<double, -1, 1>  &log_abs_b,
                                                       Eigen::Matrix<double, -1, 1>  &sign_a,
                                                       Eigen::Matrix<double, -1, 1>  &sign_b,
                                                       Eigen::Matrix<double, -1, 1>  &container_max_logs,
                                                       Eigen::Matrix<double, -1, 1>  &container_sum_exp_signed
) {
  
        const int chunk_size = u_grad_array_CM_chunk.rows();
        const int n_tests    = u_grad_array_CM_chunk.cols();
      
        auto resize_all = [&](const int m) {
          log_abs_z_grad_term.resize(m, n_tests); sign_z_grad_term.resize(m, n_tests);
          log_abs_grad_prob.resize(m, n_tests);   sign_grad_prob.resize(m, n_tests);
          log_terms.resize(m, n_tests);           sign_terms.resize(m, n_tests);
          log_abs_prod.resize(m); sign_prod.resize(m); log_abs_a.resize(m); log_abs_b.resize(m); sign_a.resize(m); sign_b.resize(m);
          log_sum_result.resize(m); sign_sum_result.resize(m); container_max_logs.resize(m); container_sum_exp_signed.resize(m);
          log_abs_z_grad_term.setConstant(-700.0); log_abs_grad_prob.setConstant(-700.0); log_abs_prod.setConstant(-700.0);
          log_sum_result.setConstant(-700.0); log_terms.setConstant(-700.0); log_abs_a.setConstant(-700.0); log_abs_b.setConstant(-700.0);
          container_max_logs.setConstant(-700.0);
          sign_z_grad_term.setOnes(); sign_grad_prob.setOnes(); sign_prod.setOnes(); sign_sum_result.setOnes(); sign_terms.setOnes();
          sign_a.setOnes(); sign_b.setOnes(); container_sum_exp_signed.setZero();
        };
        resize_all(chunk_size);
      
        {  //// second-to-last column
          const int t = n_tests - 1;
          if (n_problem_array[t] > 0) {
            const std::vector<int> &idx = problem_index_array[t];
            log_abs_u_grad_array_CM_chunk.col(n_tests - 2)(idx) =
                log_common_grad_term_1.col(t)(idx).array() + log_abs_ys.col(t)(idx).array() + log_abs_L_Omega_double(t, t - 1)
                + log_phi_Z_recip.col(t - 1)(idx).array() + y1_log_prob.col(t - 1)(idx).array();
            Eigen::Matrix<double, -1, 1> sg = sign_ys.col(t)(idx).array() * stan::math::sign(L_Omega_double(t, t - 1));
            exp_indexed_times_sign_T<vec>(u_grad_array_CM_chunk, n_tests - 2, idx, log_abs_u_grad_array_CM_chunk, n_tests - 2, sg);
          }
        }
      
        {  //// third-to-last column
          const int t = n_tests - 2;
          if (n_problem_array[t] > 0) {
            const std::vector<int> &idx = problem_index_array[t];
            resize_all(n_problem_array[t]);
      
            log_abs_z_grad_term.col(0) = log_phi_Z_recip.col(t - 1)(idx) + y1_log_prob.col(t - 1)(idx);
            sign_z_grad_term.col(0).setOnes();
            log_abs_z_grad_term.col(1).array() = log_abs_L_Omega_double(t, t - 1) + log_abs_z_grad_term.col(0).array() + log_abs_ym.col(t)(idx).array();
            sign_z_grad_term.col(1).array()    = stan::math::sign(L_Omega_double(t, t - 1)) * sign_z_grad_term.col(0).array() * sign_ym.col(t)(idx).array();
      
            log_abs_prod.array() = log_abs_z_grad_term.col(0).array() + log_abs_L_Omega_double(t, t - 1);
            sign_prod.array()    = sign_z_grad_term.col(0).array() * stan::math::sign(L_Omega_double(t, t - 1));
            log_abs_grad_prob.col(0).array() = log_abs_ys.col(t)(idx).array() + log_abs_prod.array();
            sign_grad_prob.col(0).array()    = sign_ys.col(t)(idx).array() * sign_prod.array();
      
            log_abs_a.array() = log_abs_z_grad_term.col(0).array() + log_abs_L_Omega_double(t + 1, t - 1);
            log_abs_b.array() = log_abs_z_grad_term.col(1).array() + log_abs_L_Omega_double(t + 1, t);
            sign_a.array() =  sign_z_grad_term.col(0).array() * stan::math::sign(L_Omega_double(t + 1, t - 1));
            sign_b.array() = -sign_z_grad_term.col(1).array() * stan::math::sign(L_Omega_double(t + 1, t));
            log_terms.col(0) = log_abs_a; log_terms.col(1) = log_abs_b; sign_terms.col(0) = sign_a; sign_terms.col(1) = sign_b;
            log_abs_sum_exp_general_v2_T<vec>(log_terms.leftCols(2), sign_terms.leftCols(2), log_abs_prod, sign_prod, container_max_logs, container_sum_exp_signed);
      
            log_abs_grad_prob.col(1).array() = log_abs_ys.col(t + 1)(idx).array() + log_abs_prod.array();
            sign_grad_prob.col(1).array()    = sign_ys.col(t + 1)(idx).array() * sign_prod.array();
      
            log_abs_a = log_abs_grad_prob.col(0) + y1_log_prob.col(t + 1)(idx);
            log_abs_b = log_abs_grad_prob.col(1) + y1_log_prob.col(t)(idx);
            sign_a = sign_grad_prob.col(0); sign_b = sign_grad_prob.col(1);
            log_terms.col(0) = log_abs_a; log_terms.col(1) = log_abs_b; sign_terms.col(0) = sign_a; sign_terms.col(1) = sign_b;
            log_abs_sum_exp_general_v2_T<vec>(log_terms.leftCols(2), sign_terms.leftCols(2), log_sum_result, sign_sum_result, container_max_logs, container_sum_exp_signed);
      
            log_abs_u_grad_array_CM_chunk.col(n_tests - 3)(idx) = log_common_grad_term_1.col(t)(idx) + log_sum_result;
            exp_indexed_times_sign_T<vec>(u_grad_array_CM_chunk, n_tests - 3, idx, log_abs_u_grad_array_CM_chunk, n_tests - 3, sign_sum_result);
          }
        }
      
        for (int i = 1; i < n_tests - 2; i++) {
          const int t = n_tests - (i + 2);
          if (n_problem_array[t] > 0) {
            const std::vector<int> &idx = problem_index_array[t];
            resize_all(n_problem_array[t]);
      
            log_abs_z_grad_term.col(0) = log_phi_Z_recip.col(t - 1)(idx) + y1_log_prob.col(t - 1)(idx);
            sign_z_grad_term.col(0).setOnes();
            log_abs_prod = log_abs_z_grad_term.col(0).array() + log_abs_L_Omega_double(t, t - 1);
            sign_prod    = sign_z_grad_term.col(0).array() * stan::math::sign(L_Omega_double(t, t - 1));
            log_abs_grad_prob.col(0) = log_abs_ys.col(t)(idx).array() + log_abs_prod.array();
            sign_grad_prob.col(0)    = sign_ys.col(t)(idx).array() * sign_prod.array();
      
            for (int ii = 1; ii < i + 2; ii++) {
              log_abs_z_grad_term.col(ii).array() = log_abs_ym.col(t + ii - 1)(idx).array() + log_abs_prod.array();
              sign_z_grad_term.col(ii).array()    = sign_ym.col(t + ii - 1)(idx).array() * sign_prod.array();
              log_terms.col(0).array()  = log_abs_z_grad_term.col(0).array() + log_abs_L_Omega_double(t + ii, t - 1);
              sign_terms.col(0).array() = sign_z_grad_term.col(0).array() * stan::math::sign(L_Omega_double(t + ii, t - 1));
              for (int j = 1; j <= ii; j++) {
                log_terms.col(j).array()  = log_abs_z_grad_term.col(j).array() + log_abs_L_Omega_double(t + ii, t + j - 1);
                sign_terms.col(j).array() = -sign_z_grad_term.col(j).array() * stan::math::sign(L_Omega_double(t + ii, t + j - 1));
              }
              //// leftCols(ii + 1), was leftCols(ii). Columns 0..ii are filled just above (the positive
              //// L(t + ii, t - 1) term and the ii negative terms j = 1..ii), matching Enzo's natural-scale recursion in
              //// fn_MVP_compute_nuisance_grad_v2 (zg0 * L(t + ii, t - 1) - zg[1..ii] . L(t + ii, t..t + ii - 1)). leftCols(ii) dropped the
              //// j = ii term, so every problem row of the nuisance columns 0..n_tests - 4 was wrong for n_tests >= 4 (the 3-test fixtures
              //// never reach this loop). The same line is in Enzo's original fn_MVP_compute_nuisance_grad_log_scale (fixed there too).
              log_abs_sum_exp_general_v2_T<vec>(log_terms.leftCols(ii + 1), sign_terms.leftCols(ii + 1), log_abs_prod, sign_prod, container_max_logs, container_sum_exp_signed);
              log_abs_grad_prob.col(ii).array() = log_abs_ys.col(t + ii)(idx).array() + log_abs_prod.array();
              sign_grad_prob.col(ii).array()    = sign_ys.col(t + ii)(idx).array() * sign_prod.array();
            }
      
            log_terms.setConstant(-700.0); sign_terms.setOnes();
            for (int ii = 0; ii < i + 2; ii++) {
              log_terms.col(ii)  = log_abs_grad_prob.col(ii).array() + log_prob_rowwise_prod_temp.col(t)(idx).array() + log_prob_recip.col(t + ii)(idx).array();
              sign_terms.col(ii) = sign_grad_prob.col(ii);
            }
            log_abs_sum_exp_general_v2_T<vec>(log_terms.leftCols(i + 2), sign_terms.leftCols(i + 2), log_sum_result, sign_sum_result, container_max_logs, container_sum_exp_signed);
      
            log_abs_u_grad_array_CM_chunk.col(n_tests - (i + 3))(idx) = log_common_grad_term_1.col(t)(idx).array() + log_sum_result.array();
            exp_indexed_times_sign_T<vec>(u_grad_array_CM_chunk, n_tests - (i + 3), idx, log_abs_u_grad_array_CM_chunk, n_tests - (i + 3), sign_sum_result);
          }
        }
      
        resize_all(chunk_size);
  
}




//// =====================================================================================
//// 4. coefficients grad on the log scale (intercept-only, problem rows only)
//// =====================================================================================
template <Vec vec>
inline void fn_MVP_compute_coefficients_grad_log_scale_T(  const std::vector<int> &n_problem_array,
                                                           const std::vector<std::vector<int>> &problem_index_array,
                                                           Eigen::Matrix<double, -1, -1> &beta_grad_array,
                                                           std::vector<Eigen::Matrix<double, -1, -1>> &sign_beta_grad_array_for_each_n,
                                                           std::vector<Eigen::Matrix<double, -1, -1>> &log_abs_beta_grad_array_for_each_n,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_double,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_prob_rowwise_prod_temp,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_ys,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_ys,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_ym,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_ym,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_common_grad_term_1,
                                                           Eigen::Matrix<double, -1, -1> &log_abs_z_grad_term,
                                                           Eigen::Matrix<double, -1, -1> &sign_z_grad_term,
                                                           Eigen::Matrix<double, -1, -1> &log_abs_grad_prob,
                                                           Eigen::Matrix<double, -1, -1> &sign_grad_prob,
                                                           Eigen::Matrix<double, -1, 1>  &log_abs_prod,
                                                           Eigen::Matrix<double, -1, 1>  &sign_prod,
                                                           Eigen::Matrix<double, -1, -1> &log_abs_prod_comp,
                                                           Eigen::Matrix<double, -1, -1> &sign_prod_comp,
                                                           Eigen::Matrix<double, -1, 1>  &log_sum_result,
                                                           Eigen::Matrix<double, -1, 1>  &sign_sum_result,
                                                           Eigen::Matrix<double, -1, -1> &log_terms,
                                                           Eigen::Matrix<double, -1, -1> &sign_terms,
                                                           Eigen::Matrix<double, -1, 1>  &container_max_logs,
                                                           Eigen::Matrix<double, -1, 1>  &container_sum_exp_signed
) {
  
        const int chunk_size = log_phi_Z_recip.rows();
        const int n_tests    = log_phi_Z_recip.cols();
        (void)beta_grad_array;
      
        auto resize_all = [&](const int m) {
          log_abs_z_grad_term.resize(m, n_tests); sign_z_grad_term.resize(m, n_tests);
          log_abs_grad_prob.resize(m, n_tests);   sign_grad_prob.resize(m, n_tests);
          log_terms.resize(m, n_tests);           sign_terms.resize(m, n_tests);
          log_abs_prod.resize(m); sign_prod.resize(m); log_sum_result.resize(m); sign_sum_result.resize(m);
          container_max_logs.resize(m); container_sum_exp_signed.resize(m);
          log_abs_prod_comp.resize(m, n_tests); sign_prod_comp.resize(m, n_tests);
          log_abs_z_grad_term.setConstant(-700.0); log_abs_grad_prob.setConstant(-700.0); log_abs_prod.setConstant(-700.0);
          log_sum_result.setConstant(-700.0); log_terms.setConstant(-700.0); container_max_logs.setConstant(-700.0);
          log_abs_prod_comp.setConstant(-700.0);
          sign_z_grad_term.setOnes(); sign_grad_prob.setOnes(); sign_prod.setOnes(); sign_sum_result.setOnes(); sign_terms.setOnes();
          sign_prod_comp.setOnes(); container_sum_exp_signed.setZero();
        };
        resize_all(chunk_size);
      
        {
          const int t = n_tests - 1;
          if (n_problem_array[t] > 0) {
            const std::vector<int> &idx = problem_index_array[t];
            log_abs_beta_grad_array_for_each_n[0].col(t)(idx).array() = log_common_grad_term_1.col(t)(idx).array() + log_abs_ys.col(t)(idx).array();
            sign_beta_grad_array_for_each_n[0].col(t)(idx).array()    = sign_ys.col(t)(idx).array();
          }
        }
      
        for (int i = 0; i < n_tests - 1; i++) {
          const int t = n_tests - (i + 2);
          if (n_problem_array[t] > 0) {
            const std::vector<int> &idx = problem_index_array[t];
            resize_all(n_problem_array[t]);
      
            log_abs_grad_prob.col(0) = log_abs_ys.col(t)(idx);
            sign_grad_prob.col(0)    = sign_ys.col(t)(idx);
            log_abs_z_grad_term.col(0).array() = log_abs_ym.col(t)(idx).array();
            sign_z_grad_term.col(0).array()    = -sign_ym.col(t)(idx).array();
      
            {
              const int ii = 0;
              for (int iii = 0; iii < ii + 1; iii++) {
                const double lo = log_abs_L_Omega_double(t + ii + 1, t + iii);
                const double so = stan::math::sign(L_Omega_double(t + ii + 1, t + iii));
                log_abs_prod_comp.col(iii).array() = log_abs_z_grad_term.col(iii).array() + lo;
                sign_prod_comp.col(iii).array()    = sign_z_grad_term.col(iii).array() * so;
              }
            }
            sign_prod    = sign_prod_comp.col(0);
            log_abs_prod = log_abs_prod_comp.col(0);
      
            log_abs_grad_prob.col(1).array() = log_abs_ys.col(t + 1)(idx).array() + log_abs_prod.array();
            sign_grad_prob.col(1).array()    = sign_ys.col(t + 1)(idx).array() * sign_prod.array();
      
            if (i > 0) {
              for (int ii = 1; ii < i + 1; ii++) {
                log_abs_z_grad_term.col(ii).array() = log_abs_ym.col(t + ii)(idx).array() + log_abs_prod.array();
                sign_z_grad_term.col(ii).array()    = -sign_ym.col(t + ii)(idx).array() * sign_prod.array();
                for (int iii = 0; iii < ii + 1; iii++) {
                  const double lo = log_abs_L_Omega_double(t + ii + 1, t + iii);
                  const double so = stan::math::sign(L_Omega_double(t + ii + 1, t + iii));
                  log_abs_prod_comp.col(iii).array() = log_abs_z_grad_term.col(iii).array() + lo;
                  sign_prod_comp.col(iii).array()    = sign_z_grad_term.col(iii).array() * so;
                }
                log_abs_sum_exp_general_v2_T<vec>(log_abs_prod_comp.leftCols(ii + 1), sign_prod_comp.leftCols(ii + 1), log_abs_prod, sign_prod, container_max_logs, container_sum_exp_signed);
                log_abs_grad_prob.col(ii + 1) = log_abs_ys.col(t + ii + 1)(idx).array() + log_abs_prod.array();
                sign_grad_prob.col(ii + 1)    = sign_ys.col(t + ii + 1)(idx).array() * sign_prod.array();
              }
            }
      
            for (int ii = 0; ii < i + 2; ii++) {
              log_terms.col(ii).array()  = log_abs_grad_prob.col(ii).array() + log_prob_rowwise_prod_temp.col(t)(idx).array() - y1_log_prob.col(t + ii)(idx).array();
              sign_terms.col(ii).array() = sign_grad_prob.col(ii).array();
            }
            log_abs_sum_exp_general_v2_T<vec>(log_terms.leftCols(i + 2), sign_terms.leftCols(i + 2), log_sum_result, sign_sum_result, container_max_logs, container_sum_exp_signed);
      
            log_abs_beta_grad_array_for_each_n[0].col(t)(idx).array() = log_common_grad_term_1.col(t)(idx).array() + log_sum_result.array();
            sign_beta_grad_array_for_each_n[0].col(t)(idx).array()    = sign_sum_result.array();
          }
        }
      
        resize_all(chunk_size);
  
}




//// =====================================================================================
//// 5. L_Omega grad on the log scale (problem rows only)
//// =====================================================================================
template <Vec vec>
inline void fn_MVP_compute_L_Omega_grad_log_scale_T(  const std::vector<int> &n_problem_array,
                                                      const std::vector<std::vector<int>> &problem_index_array,
                                                      Eigen::Ref<Eigen::Matrix<double, -1, -1>> L_Omega_grad_array,
                                                      std::vector<Eigen::Matrix<double, -1, -1>> &sign_LO,      //// sign_L_Omega_grad_array_col_for_each_n
                                                      std::vector<Eigen::Matrix<double, -1, -1>> &log_abs_LO,   //// log_abs_L_Omega_grad_array_col_for_each_n
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_Bound_Z,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_Bound_Z,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_Z_std_norm,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_Z_std_norm,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_double,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_phi_Z_recip,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_prop_rowwise_prod_temp,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_ys,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_ys,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_ym,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_ym,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_common_grad_term_1,
                                                      Eigen::Matrix<double, -1, -1> &log_abs_z_grad_term,
                                                      Eigen::Matrix<double, -1, -1> &sign_z_grad_term,
                                                      Eigen::Matrix<double, -1, -1> &log_abs_grad_prob,
                                                      Eigen::Matrix<double, -1, -1> &sign_grad_prob,
                                                      Eigen::Matrix<double, -1, 1>  &log_abs_prod,
                                                      Eigen::Matrix<double, -1, 1>  &sign_prod,
                                                      Eigen::Matrix<double, -1, -1> &log_abs_prod_comp,
                                                      Eigen::Matrix<double, -1, -1> &sign_prod_comp,
                                                      Eigen::Matrix<double, -1, -1> &log_abs_dcc,      //// log_abs_derivs_chain_container_vec_comp
                                                      Eigen::Matrix<double, -1, -1> &sign_dcc,
                                                      Eigen::Matrix<double, -1, 1>  &log_sum_result,
                                                      Eigen::Matrix<double, -1, 1>  &sign_sum_result,
                                                      Eigen::Matrix<double, -1, -1> &log_terms,
                                                      Eigen::Matrix<double, -1, -1> &sign_terms,
                                                      Eigen::Matrix<double, -1, 1>  &log_abs_a,
                                                      Eigen::Matrix<double, -1, 1>  &log_abs_b,
                                                      Eigen::Matrix<double, -1, 1>  &sign_a,
                                                      Eigen::Matrix<double, -1, 1>  &sign_b,
                                                      Eigen::Matrix<double, -1, 1>  &container_max_logs,
                                                      Eigen::Matrix<double, -1, 1>  &container_sum_exp_signed
) {
  
        const int n_tests    = log_Bound_Z.cols();
        const int chunk_size = log_Bound_Z.rows();
        (void)L_Omega_grad_array;
      
        auto resize_all = [&](const int m) {
          log_abs_z_grad_term.resize(m, n_tests); sign_z_grad_term.resize(m, n_tests);
          log_abs_grad_prob.resize(m, n_tests);   sign_grad_prob.resize(m, n_tests);
          log_terms.resize(m, n_tests);           sign_terms.resize(m, n_tests);
          log_abs_prod.resize(m); sign_prod.resize(m); log_sum_result.resize(m); sign_sum_result.resize(m);
          container_max_logs.resize(m); container_sum_exp_signed.resize(m);
          log_abs_prod_comp.resize(m, n_tests); sign_prod_comp.resize(m, n_tests);
          log_abs_dcc.resize(m, n_tests); sign_dcc.resize(m, n_tests);
          log_abs_a.resize(m); log_abs_b.resize(m); sign_a.resize(m); sign_b.resize(m);
          log_abs_z_grad_term.setConstant(-700.0); log_abs_grad_prob.setConstant(-700.0); log_abs_prod.setConstant(-700.0);
          log_sum_result.setConstant(-700.0); log_terms.setConstant(-700.0); container_max_logs.setConstant(-700.0);
          log_abs_prod_comp.setConstant(-700.0); log_abs_dcc.setConstant(-700.0); log_abs_a.setConstant(-700.0); log_abs_b.setConstant(-700.0);
          sign_z_grad_term.setOnes(); sign_grad_prob.setOnes(); sign_prod.setOnes(); sign_sum_result.setOnes(); sign_terms.setOnes();
          sign_prod_comp.setOnes(); sign_dcc.setOnes(); sign_a.setOnes(); sign_b.setOnes(); container_sum_exp_signed.setZero();
        };
      
        //// ---- last diagonal ----
        {
          const int t1 = n_tests - 1;
          const std::vector<int> &idx = problem_index_array[t1];
          sign_LO[t1].col(t1)(idx).array()    = sign_ys.col(t1)(idx).array() * sign_Bound_Z.col(t1)(idx).array();
          log_abs_LO[t1].col(t1)(idx).array() = log_common_grad_term_1.col(t1)(idx).array() + log_abs_ys.col(t1)(idx).array() + log_Bound_Z.col(t1)(idx).array();
        }
      
        //// ---- second-to-last diagonal ----
        {
          const int t1 = n_tests - 2;
          if (n_problem_array[t1] > 0) {
            const std::vector<int> &idx = problem_index_array[t1];
            resize_all(n_problem_array[t1]);
      
            log_abs_grad_prob.col(0)         = log_abs_ys.col(t1)(idx) + log_Bound_Z.col(t1)(idx);
            sign_grad_prob.col(0).array()    = sign_ys.col(t1)(idx).array() * sign_Bound_Z.col(t1)(idx).array();
            log_abs_z_grad_term.col(0)       = log_abs_ym.col(t1)(idx) + log_Bound_Z.col(t1)(idx);
            sign_z_grad_term.col(0).array()  = -sign_ym.col(t1)(idx).array() * sign_Bound_Z.col(t1)(idx).array();
            log_abs_prod.array() = log_abs_z_grad_term.col(0).array() + log_abs_L_Omega_double(t1 + 1, t1);
            sign_prod            = sign_z_grad_term.col(0) * stan::math::sign(L_Omega_double(t1 + 1, t1));
            log_abs_grad_prob.col(1)         = log_abs_ys.col(t1 + 1)(idx) + log_abs_prod;
            sign_grad_prob.col(1).array()    = sign_ys.col(t1 + 1)(idx).array() * sign_prod.array();
      
            log_abs_a = log_abs_grad_prob.col(1) + y1_log_prob.col(t1)(idx);      sign_a = sign_grad_prob.col(1);
            log_abs_b = log_abs_grad_prob.col(0) + y1_log_prob.col(t1 + 1)(idx);  sign_b = sign_grad_prob.col(0);
            log_terms.col(0) = log_abs_a; log_terms.col(1) = log_abs_b; sign_terms.col(0) = sign_a; sign_terms.col(1) = sign_b;
            log_abs_sum_exp_general_v2_T<vec>(log_terms.leftCols(2), sign_terms.leftCols(2), log_sum_result, sign_sum_result, container_max_logs, container_sum_exp_signed);
      
            log_abs_LO[t1].col(t1)(idx) = log_common_grad_term_1.col(t1)(idx) + log_sum_result;
            sign_LO[t1].col(t1)(idx)    = sign_sum_result;
          }
        }
      
        //// ---- remaining diagonals ----
        for (int i = 3; i < n_tests + 1; i++) {
          const int t1 = n_tests - i;
          if (n_problem_array[t1] > 0) {
            const std::vector<int> &idx = problem_index_array[t1];
            resize_all(n_problem_array[t1]);
      
            log_abs_grad_prob.col(0)        = log_abs_ys.col(t1)(idx) + log_Bound_Z.col(t1)(idx);
            sign_grad_prob.col(0).array()   = sign_ys.col(t1)(idx).array() * sign_Bound_Z.col(t1)(idx).array();
            log_abs_z_grad_term.col(0)      = log_abs_ym.col(t1)(idx) + log_Bound_Z.col(t1)(idx);
            sign_z_grad_term.col(0).array() = -sign_ym.col(t1)(idx).array() * sign_Bound_Z.col(t1)(idx).array();
            log_abs_prod.array() = log_abs_L_Omega_double(t1 + 1, t1) + log_abs_z_grad_term.col(0).array();
            sign_prod            = stan::math::sign(L_Omega_double(t1 + 1, t1)) * sign_z_grad_term.col(0);
            log_abs_grad_prob.col(1)        = log_abs_ys.col(t1 + 1)(idx) + log_abs_prod;
            sign_grad_prob.col(1).array()   = sign_ys.col(t1 + 1)(idx).array() * sign_prod.array();
      
            for (int ii = 1; ii < i - 1; ii++) {
              log_abs_z_grad_term.col(ii)      = log_abs_ym.col(t1 + ii)(idx) + log_abs_prod;
              sign_z_grad_term.col(ii).array() = -sign_ym.col(t1 + ii)(idx).array() * sign_prod.array();
              for (int iii = 0; iii < ii + 1; iii++) {
                log_abs_prod_comp.col(iii).array() = log_abs_z_grad_term.col(iii).array() + log_abs_L_Omega_double(t1 + ii + 1, t1 + iii);
                sign_prod_comp.col(iii)            = sign_z_grad_term.col(iii) * stan::math::sign(L_Omega_double(t1 + ii + 1, t1 + iii));
              }
              log_abs_sum_exp_general_v2_T<vec>(log_abs_prod_comp.leftCols(ii + 1), sign_prod_comp.leftCols(ii + 1), log_abs_prod, sign_prod, container_max_logs, container_sum_exp_signed);
              log_abs_grad_prob.col(ii + 1)        = log_abs_ys.col(t1 + ii + 1)(idx) + log_abs_prod;
              sign_grad_prob.col(ii + 1).array()   = sign_ys.col(t1 + ii + 1)(idx).array() * sign_prod.array();
            }
            for (int iii = 0; iii < i; iii++) {
              log_abs_dcc.col(iii) = log_abs_grad_prob.col(iii) + log_prop_rowwise_prod_temp.col(t1)(idx) - y1_log_prob.col(t1 + iii)(idx);
              sign_dcc.col(iii)    = sign_grad_prob.col(iii);
            }
            log_abs_sum_exp_general_v2_T<vec>(log_abs_dcc.leftCols(i), sign_dcc.leftCols(i), log_sum_result, sign_sum_result, container_max_logs, container_sum_exp_signed);
            log_abs_LO[t1].col(t1)(idx) = log_common_grad_term_1.col(t1)(idx) + log_sum_result;
            sign_LO[t1].col(t1)(idx)    = sign_sum_result;
          }
        }
      
        //// ---- off-diagonals: last row ----
        {
          const int t1 = n_tests - 1;
          if (n_problem_array[t1] > 0) {
            const std::vector<int> &idx = problem_index_array[t1];
            auto fill = [&](const int t2) {
              sign_LO[t1].col(t2)(idx).array() = sign_ys.col(t1)(idx).array() * sign_Z_std_norm.col(t2)(idx).array();
              log_abs_LO[t1].col(t2)(idx)      = log_common_grad_term_1.col(t1)(idx) + log_abs_ys.col(t1)(idx) + log_Z_std_norm.col(t2)(idx);
            };
            fill(n_tests - 2);
            if (t1 > 1) fill(n_tests - 3);
            if (t1 > 2) {
              for (int t2_dash = 3; t2_dash < n_tests; t2_dash++) {
                const int t2 = n_tests - (t2_dash + 1);
                if (t2 < n_tests - 1) fill(t2);
              }
            }
          }
        }
      
        //// ---- off-diagonals: remaining rows ----
        for (int t1_dash = 1; t1_dash < n_tests - 1; t1_dash++) {
          const int t1 = n_tests - (t1_dash + 1);
          if (n_problem_array[t1] > 0) {
            const std::vector<int> &idx = problem_index_array[t1];
            resize_all(n_problem_array[t1]);
            for (int t2_dash = t1_dash + 1; t2_dash < n_tests; t2_dash++) {
              const int t2 = n_tests - (t2_dash + 1);
              log_abs_prod = log_Z_std_norm.col(t2)(idx);
              sign_prod    = sign_Z_std_norm.col(t2)(idx);
              log_abs_grad_prob.col(0)        = log_abs_ys.col(t1)(idx) + log_abs_prod;
              sign_grad_prob.col(0).array()   = sign_ys.col(t1)(idx).array() * sign_prod.array();
              log_abs_z_grad_term.col(0)      = log_abs_ym.col(t1)(idx) + log_abs_prod;
              sign_z_grad_term.col(0).array() = -sign_ym.col(t1)(idx).array() * sign_prod.array();
              for (int t1_dash_dash = 1; t1_dash_dash < t1_dash + 1; t1_dash_dash++) {
                if (t1_dash_dash > 1) {
                  log_abs_z_grad_term.col(t1_dash_dash - 1)      = log_abs_ym.col(t1 + t1_dash_dash - 1)(idx) + log_abs_prod;
                  sign_z_grad_term.col(t1_dash_dash - 1).array() = -sign_ym.col(t1 + t1_dash_dash - 1)(idx).array() * sign_prod.array();
                }
                for (int iii = 0; iii < t1_dash_dash; iii++) {
                  log_abs_prod_comp.col(iii).array() = log_abs_z_grad_term.col(iii).array() + log_abs_L_Omega_double(t1 + t1_dash_dash, t1 + iii);
                  sign_prod_comp.col(iii).array()    = sign_z_grad_term.col(iii).array() * stan::math::sign(L_Omega_double(t1 + t1_dash_dash, t1 + iii));
                }
                log_abs_sum_exp_general_v2_T<vec>(log_abs_prod_comp.leftCols(t1_dash_dash), sign_prod_comp.leftCols(t1_dash_dash), log_abs_prod, sign_prod, container_max_logs, container_sum_exp_signed);
                log_abs_grad_prob.col(t1_dash_dash)        = log_abs_ys.col(t1 + t1_dash_dash)(idx) + log_abs_prod;
                sign_grad_prob.col(t1_dash_dash).array()   = sign_ys.col(t1 + t1_dash_dash)(idx).array() * sign_prod.array();
              }
              for (int ii = 0; ii < t1_dash + 1; ii++) {
                log_abs_dcc.col(ii) = log_abs_grad_prob.col(ii) + log_prop_rowwise_prod_temp.col(t1)(idx) - y1_log_prob.col(t1 + ii)(idx);
                sign_dcc.col(ii)    = sign_grad_prob.col(ii);
              }
              log_abs_sum_exp_general_v2_T<vec>(log_abs_dcc.leftCols(t1_dash + 1), sign_dcc.leftCols(t1_dash + 1), log_sum_result, sign_sum_result, container_max_logs, container_sum_exp_signed);
              log_abs_LO[t1].col(t2)(idx) = log_common_grad_term_1.col(t1)(idx) + log_sum_result;
              sign_LO[t1].col(t2)(idx)    = sign_sum_result;
            }
          }
        }
      
        resize_all(chunk_size);
  
}



