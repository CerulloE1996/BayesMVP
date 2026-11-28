//// -------------------------------------------------------------------------------------
//// CLC_MVOP_lp_grad_MD_AD_fns.hpp
////
//// Manual-gradient drivers for Model_type = "CLC_MVOP" (and alias "CLC-MVOP"):
////   - NoLog (serial and WCP OpenMP)
////   - PartialLog (log-scale tail handling, with chunk relayout)
////   - CI (closed-form conditional-independence path when CI == true or n_us == 0)
////   - Multi-attempts dispatcher (NoLog -> PartialLog -> whole-model AD fallback)
////
//// Reuses shared MVOP / MVP GHK forward, backward, PartialLog, and missing-data kernels.
//// -------------------------------------------------------------------------------------

#pragma once

#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "CLC_MVOP_lp_grad_common_T.hpp"
#pragma GCC push_options
#pragma GCC optimize ("O1")
#include "CLC_MVOP_lp_grad_AD_fns.hpp"
#pragma GCC pop_options


//// -------------------------------------------------------------------------------------
//// Resize per-thread workspace matrices for a given chunk size and n_class (4 classes)
//// -------------------------------------------------------------------------------------
inline void fn_CLC_MVOP_resize_ws_for_chunk(
      LC_MVP_workspace_struct &ws_cur,
      const int chunk_size,
      const int n_tests,
      const int n_class
) {
      if (static_cast<int>(ws_cur.Z_std_norm.size()) < n_class) {
            ws_cur.Z_std_norm.resize(n_class);
      }
      if (static_cast<int>(ws_cur.Bound_Z.size()) < n_class) {
            ws_cur.Bound_Z.resize(n_class);
      }
      if (static_cast<int>(ws_cur.Bound_U_Phi_Bound_Z.size()) < n_class) {
            ws_cur.Bound_U_Phi_Bound_Z.resize(n_class);
      }
      if (static_cast<int>(ws_cur.prob.size()) < n_class) {
            ws_cur.prob.resize(n_class);
      }
      if (static_cast<int>(ws_cur.Phi_Z.size()) < n_class) {
            ws_cur.Phi_Z.resize(n_class);
      }
      if (static_cast<int>(ws_cur.Upper_Bound_Z.size()) < n_class) {
            ws_cur.Upper_Bound_Z.resize(n_class);
      }
      for (int class_index = 0; class_index < n_class; class_index++) {
            ws_cur.Z_std_norm[class_index].resize(
                  chunk_size,
                  n_tests
            );
            ws_cur.Bound_Z[class_index].resize(
                  chunk_size,
                  n_tests
            );
            ws_cur.Bound_U_Phi_Bound_Z[class_index].resize(
                  chunk_size,
                  n_tests
            );
            ws_cur.prob[class_index].resize(
                  chunk_size,
                  n_tests
            );
            ws_cur.Phi_Z[class_index].resize(
                  chunk_size,
                  n_tests
            );
            ws_cur.Upper_Bound_Z[class_index].resize(
                  chunk_size,
                  n_tests
            );
      }
      ws_cur.y1_log_prob.resize(
            chunk_size,
            n_tests
      );
      ws_cur.phi_Z_recip.resize(
            chunk_size,
            n_tests
      );
      ws_cur.phi_Bound_Z.resize(
            chunk_size,
            n_tests
      );
      ws_cur.u_grad_array_CM_chunk.resize(
            chunk_size,
            n_tests
      );
      ws_cur.common_grad_term_1.resize(
            chunk_size,
            n_tests
      );
      ws_cur.y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(
            chunk_size,
            n_tests
      );
      ws_cur.y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(
            chunk_size,
            n_tests
      );
      ws_cur.prob_rowwise_prod_temp.resize(
            chunk_size,
            n_tests
      );
      ws_cur.prob_recip_rowwise_prod_temp.resize(
            chunk_size,
            n_tests
      );
      ws_cur.prod_container_or_inc_array.resize(chunk_size);
      ws_cur.derivs_chain_container_vec.resize(chunk_size);
      ws_cur.prob_rowwise_prod_temp_all.resize(chunk_size);
      ws_cur.dphi_direct_Cj.resize(chunk_size);
      ws_cur.dZ_direct_Cj.resize(chunk_size);
      ws_cur.grad_prob.resize(
            chunk_size,
            n_tests
      );
      ws_cur.z_grad_term.resize(
            chunk_size,
            n_tests
      );
      ws_cur.y_chunk.resize(
            chunk_size,
            n_tests
      );
      ws_cur.u_array.resize(
            chunk_size,
            n_tests
      );
      ws_cur.y_sign.resize(
            chunk_size,
            n_tests
      );
      ws_cur.y_m_y_sign_x_u.resize(
            chunk_size,
            n_tests
      );
      ws_cur.u_grad_array_CM_chunk_block.resize(
            chunk_size,
            n_tests
      );
      ws_cur.u_unc_vec_chunk.resize(chunk_size * n_tests);
      ws_cur.u_vec_chunk.resize(chunk_size * n_tests);
      ws_cur.du_wrt_duu_chunk.resize(chunk_size * n_tests);
      ws_cur.d_J_wrt_duu_chunk.resize(chunk_size * n_tests);
      ws_cur.lp_array.resize(
            chunk_size,
            n_class
      );
      ws_cur.prob_n.resize(chunk_size);
      ws_cur.prob_n_recip.resize(chunk_size);
      ws_cur.log_sum_result.resize(chunk_size);
      ws_cur.container_max_logs.resize(chunk_size);
      ws_cur.rowwise_log_sum.resize(chunk_size);
      ws_cur.rowwise_prod.resize(chunk_size);
      ws_cur.rowwise_sum.resize(chunk_size);
      ws_cur.log_lik_chunk.resize(chunk_size);
      ws_cur.prob_recip.resize(
            chunk_size,
            n_tests
      );
      ws_cur.phi_Upper_Bound_Z.resize(
            chunk_size,
            n_tests
      );
      ws_cur.dphi_over_L.resize(
            chunk_size,
            n_tests
      );
      ws_cur.dZ_dmu_neg.resize(
            chunk_size,
            n_tests
      );
      ws_cur.dphi_times_bz.resize(
            chunk_size,
            n_tests
      );
      ws_cur.dZ_times_bz.resize(
            chunk_size,
            n_tests
      );
      ws_cur.log_prev_per_obs_given_c.resize(chunk_size);
      ws_cur.prev_per_obs_given_c.resize(chunk_size);
}


//// -------------------------------------------------------------------------------------
//// Ensure per-thread workspace vector has enough allocated slots for CLC-MVOP
//// -------------------------------------------------------------------------------------
inline void fn_CLC_MVOP_ensure_workspaces(
      std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
      const CLC_MVOP_prep &prep,
      const int n_threads_WCP
) {
      const int required_ws = (n_threads_WCP > 1) ? n_threads_WCP : 1;
      if (static_cast<int>(LC_MVP_ws_structs.size()) < required_ws) {
            LC_MVP_ws_structs.resize(required_ws);
      }
      for (int thread_index = 0; thread_index < required_ws; thread_index++) {
            LC_MVP_workspace_struct &ws_cur = LC_MVP_ws_structs[thread_index];
            if (!ws_cur.is_allocated
                || ws_cur.stored_n_class < prep.n_class
                || ws_cur.allocated_chunk_size < prep.normal_chunk_size
                || ws_cur.stored_n_tests != prep.n_tests) {
                  ws_cur.allocate(
                        prep.normal_chunk_size,
                        prep.n_tests,
                        prep.n_class,
                        std::max(1, prep.n_covariates_max),
                        std::max(1, prep.n_corrs_total),
                        std::max(1, prep.n_covariates_total)
                  );
            }
            if ((ws_cur.y_chunk.rows() != prep.normal_chunk_size)
                || (ws_cur.y_chunk.cols() != prep.n_tests)
                || (static_cast<int>(ws_cur.Upper_Bound_Z.size()) < prep.n_class)
                || (static_cast<int>(ws_cur.Z_std_norm.size()) < prep.n_class)
                || (ws_cur.lp_array.cols() != prep.n_class)) {
                  fn_CLC_MVOP_resize_ws_for_chunk(
                        ws_cur,
                        prep.normal_chunk_size,
                        prep.n_tests,
                        prep.n_class
                  );
            }
      }
}


//// -------------------------------------------------------------------------------------
//// Nuisance anchor when prior_only == true (matches standard normal u_raw under Phi)
//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_CLC_MVOP_prior_only_nuisance_anchor_T(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const std::string &grad_option,
      CLC_MVOP_prep &prep,
      double &log_jac_u
) {
      const int n_us = prep.n_us;
      if (n_us <= 0) {
            return;
      }
      if (prep.prior_only_u_unc_buf.rows() != n_us) {
            prep.prior_only_u_unc_buf.resize(n_us);
            prep.prior_only_u_buf.resize(n_us);
            prep.prior_only_du_wrt_duu_buf.resize(n_us);
            prep.prior_only_d_J_wrt_duu_buf.resize(n_us);
      }
      Eigen::Matrix<double, -1, 1> &u_unc_copy = prep.prior_only_u_unc_buf;
      Eigen::Matrix<double, -1, 1> &u_vec = prep.prior_only_u_buf;
      Eigen::Matrix<double, -1, 1> &du_wrt_duu = prep.prior_only_du_wrt_duu_buf;
      Eigen::Matrix<double, -1, 1> &d_J_wrt_duu = prep.prior_only_d_J_wrt_duu_buf;
      u_unc_copy = theta_us_vec_ref;
      fn_MVP_compute_nuisance_T<vec>(
            u_vec,
            u_unc_copy,
            prep.kchoice
      );
      log_jac_u += fn_MVP_compute_nuisance_log_jac_u_T<vec>(
            u_vec,
            u_unc_copy,
            d_J_wrt_duu,
            prep.kchoice
      );
      const bool do_us = (grad_option == "us_only") || (grad_option == "all");
      if (do_us) {
            fn_MVP_nuisance_first_deriv_T<vec>(
                  du_wrt_duu,
                  u_vec,
                  u_unc_copy,
                  prep.kchoice
            );
            fn_MVP_nuisance_deriv_of_log_det_J_T<vec>(
                  d_J_wrt_duu,
                  u_vec,
                  u_unc_copy,
                  du_wrt_duu,
                  prep.kchoice
            );
            out_mat.segment(
                  1,
                  n_us
            ) = d_J_wrt_duu;
      }
}


//// -------------------------------------------------------------------------------------
//// 1. Closed-form conditional-independence (CI / n_us == 0) path
////    Handles binary, ordinal, interval-censored, missing, and log-scale tails
//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_lp_grad_CLC_MVOP_CI_Inplace_process(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const std::string &grad_option,
      const Model_fn_args_struct &Model_args_as_cpp_struct
) {
      CLC_MVOP_prep prep = fn_CLC_MVOP_prep(
            out_mat,
            theta_main_vec_ref,
            theta_us_vec_ref,
            y_ref,
            Model_args_as_cpp_struct,
            1
      );
      if (prep.label_constraint_violated) {
            out_mat.setZero();
            out_mat(0) = -std::numeric_limits<double>::infinity();
            return;
      }
      const int n_obs = prep.n_obs;
      const int n_tests = prep.n_tests;
      const int n_class = prep.n_class;
      const int n_params = prep.n_params;
      const int K_max = std::max(1, prep.n_covariates_max);

      std::vector<Eigen::Matrix<double, -1, -1>> beta_4_grad_lik(
            n_class,
            Eigen::Matrix<double, -1, -1>::Zero(
                  K_max,
                  n_tests
            )
      );
      std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array = vec_of_mats<double>(
            n_tests,
            n_tests,
            n_class
      );
      std::vector<Eigen::Matrix<double, -1, -1>> cutpoint_grad_4 = vec_of_mats<double>(
            std::max(1, prep.n_cutpoints_max),
            std::max(1, prep.n_ordinal_tests),
            n_class
      );
      Eigen::Matrix<double, -1, -1> d_lp_d_log_prev = Eigen::Matrix<double, -1, -1>::Zero(
            n_obs,
            n_class
      );
      double log_jac_u = 0.0;
      fn_CLC_MVOP_prior_only_nuisance_anchor_T<nico::math::Vec::Scalar>(
            out_mat,
            theta_us_vec_ref,
            grad_option,
            prep,
            log_jac_u
      );

      if (!prep.prior_only) {
            const bool do_grad = (grad_option != "none");
            const bool use_approx = prep.kchoice.Phi_approx;

            std::vector<Eigen::Matrix<double, -1, -1>> Xbeta_lik(
                  n_class,
                  Eigen::Matrix<double, -1, -1>::Zero(
                        n_obs,
                        n_tests
                  )
            );
            for (int class_index = 0; class_index < n_class; class_index++) {
                  for (int test_index = 0; test_index < n_tests; test_index++) {
                        if ((*prep.is_perfect_ref)(test_index) == 1) {
                              Xbeta_lik[class_index].col(test_index).setConstant(
                                    prep.beta_4[class_index](0, test_index)
                              );
                        } else {
                              const int n_covs_ct = (*prep.n_covs_per_outcome)(class_index, test_index);
                              if (n_covs_ct > 0) {
                                    Xbeta_lik[class_index].col(test_index).noalias() =
                                          (*prep.X_class_test)[prep.class_index_of_X_to_read(class_index)][prep.test_index_of_X_to_read(test_index)].leftCols(n_covs_ct)
                                          * prep.beta_4[class_index].col(test_index).head(n_covs_ct);
                              }
                        }
                  }
            }

            Eigen::Matrix<double, -1, -1> lp_mat = prep.log_prev_subject;
            std::vector<Eigen::Matrix<double, -1, -1>> dlogp_dmu(
                  n_class,
                  Eigen::Matrix<double, -1, -1>::Zero(
                        n_obs,
                        n_tests
                  )
            );
            std::vector<Eigen::Matrix<double, -1, -1>> dlogp_dClo(
                  n_class,
                  Eigen::Matrix<double, -1, -1>::Zero(
                        n_obs,
                        n_tests
                  )
            );
            std::vector<Eigen::Matrix<double, -1, -1>> dlogp_dChi(
                  n_class,
                  Eigen::Matrix<double, -1, -1>::Zero(
                        n_obs,
                        n_tests
                  )
            );

            //// ---- Per class and test, the whole column of n_obs rows through his NicoSIMD kernels at this vect_type
            ////      (AVX512, AVX2 or Scalar): log Phi (exact, or Phi_approx), exp, log1m and log on the column at once.
            ////      The columns of dlogp_dClo / dlogp_dChi / dlogp_dmu are the buffers. log P = log(Phi(A) - Phi(B))
            ////      = log Phi(A) + log1m(exp(log Phi(B) - log Phi(A))), with A, B as before (Phi or survival form by the
            ////      sign of the lower bound); an open bound gets B = -40 (log Phi(-40) is about -805, so the exp of the
            ////      difference is exactly 0 in double) and its gradient ratio the log argument -1000 (exp is exactly 0).
            ////      A missing value (y < 0) and a category spanning every cutpoint add 0 to the log-likelihood and gradient.
            const double a_Phi_approx = 0.07056;
            const double b_Phi_approx = 1.5976;
            for (int class_index = 0; class_index < n_class; class_index++) {
                  for (int test_index = 0; test_index < n_tests; test_index++) {
                        const int ord_index = prep.ord_idx_of_test(test_index);
                        const int n_cat_test = (ord_index >= 0) ? (*prep.n_cat_per_ord_test)(ord_index) : 2;
                        double *log_Phi_A = dlogp_dClo[class_index].col(test_index).data();
                        double *log_Phi_B = dlogp_dChi[class_index].col(test_index).data();
                        double *log_prob = dlogp_dmu[class_index].col(test_index).data();
                        //// 1. the arguments A and B of log Phi, row by row (no maths but + - *)
                        for (int row_index = 0; row_index < n_obs; row_index++) {
                              const int y_lo = y_ref(row_index, test_index);
                              const double mu_val = Xbeta_lik[class_index](row_index, test_index);
                              log_Phi_A[row_index] = 0.0;
                              log_Phi_B[row_index] = -40.0;
                              if (y_lo < 0) {
                                    continue;
                              }
                              if (ord_index < 0) {
                                    log_Phi_A[row_index] = ((y_lo == 1) ? 1.0 : -1.0) * mu_val;
                                    continue;
                              }
                              const int y_hi = prep.has_y_upper ? (*prep.y_upper_mat)(row_index, test_index) : y_lo;
                              const bool open_lo = (y_lo == 1);
                              const bool open_hi = (y_hi == n_cat_test);
                              if (open_lo && open_hi) {
                                    continue;
                              }
                              const double lower_bound = open_lo ? 0.0 : (prep.C_4[class_index](y_lo - 2, ord_index) - mu_val);
                              const double upper_bound = open_hi ? 0.0 : (prep.C_4[class_index](y_hi - 1, ord_index) - mu_val);
                              if (open_lo) {
                                    log_Phi_A[row_index] = upper_bound;
                              } else if (open_hi) {
                                    log_Phi_A[row_index] = -lower_bound;
                              } else if (lower_bound <= 0.0) {
                                    log_Phi_A[row_index] = upper_bound;
                                    log_Phi_B[row_index] = lower_bound;
                              } else {
                                    log_Phi_A[row_index] = -lower_bound;
                                    log_Phi_B[row_index] = -upper_bound;
                              }
                        }
                        //// 2. log Phi of both columns (NicoSIMD)
                        if (use_approx) {
                              apply_raw<vec, Fn::log_Phi_approx, true>(log_Phi_A, n_obs);
                              apply_raw<vec, Fn::log_Phi_approx, true>(log_Phi_B, n_obs);
                        } else {
                              fn_apply_log_Phi_exact_raw_T<vec>(log_Phi_A, n_obs);
                              fn_apply_log_Phi_exact_raw_T<vec>(log_Phi_B, n_obs);
                        }
                        //// 3. log P = log Phi(A) + log1m(exp(log Phi(B) - log Phi(A))) (NicoSIMD exp and log1m)
                        for (int row_index = 0; row_index < n_obs; row_index++) {
                              log_prob[row_index] = log_Phi_B[row_index] - log_Phi_A[row_index];
                        }
                        apply_raw<vec, Fn::exp, true>(log_prob, n_obs);
                        apply_raw<vec, Fn::log1m, true>(log_prob, n_obs);
                        //// 4. add log P to the class log-likelihood (rows that count only); the log arguments of the two
                        ////    gradient ratios, log density(bound) - log P, go into log_Phi_A (lower) and log_Phi_B (upper)
                        for (int row_index = 0; row_index < n_obs; row_index++) {
                              log_prob[row_index] += log_Phi_A[row_index];
                              const int y_lo = y_ref(row_index, test_index);
                              const double mu_val = Xbeta_lik[class_index](row_index, test_index);
                              log_Phi_A[row_index] = -1000.0;
                              log_Phi_B[row_index] = -1000.0;
                              if (y_lo < 0) {
                                    continue;
                              }
                              double x_lo = 0.0;
                              double x_hi = 0.0;
                              bool has_lo = false;
                              bool has_hi = false;
                              if (ord_index < 0) {
                                    x_lo = ((y_lo == 1) ? 1.0 : -1.0) * mu_val;
                                    has_lo = true;
                              } else {
                                    const int y_hi = prep.has_y_upper ? (*prep.y_upper_mat)(row_index, test_index) : y_lo;
                                    const bool open_lo = (y_lo == 1);
                                    const bool open_hi = (y_hi == n_cat_test);
                                    if (open_lo && open_hi) {
                                          continue;
                                    }
                                    has_lo = !open_lo;
                                    has_hi = !open_hi;
                                    x_lo = open_lo ? 0.0 : (prep.C_4[class_index](y_lo - 2, ord_index) - mu_val);
                                    x_hi = open_hi ? 0.0 : (prep.C_4[class_index](y_hi - 1, ord_index) - mu_val);
                              }
                              lp_mat(row_index, class_index) += log_prob[row_index];
                              if (!do_grad) {
                                    continue;
                              }
                              //// exact: log phi(x) = -x^2 / 2 - log(2 pi) / 2; Phi_approx: the density without the log
                              ////        (3 a x^2 + b) / (1 + exp(-|p|))^2, p = x (a x^2 + b); its log and the - |p| below
                              if (has_lo) {
                                    log_Phi_A[row_index] = use_approx
                                          ? -std::abs(x_lo * (a_Phi_approx * x_lo * x_lo + b_Phi_approx))
                                          : (-0.5 * x_lo * x_lo - HALF_LOG_TWO_PI - log_prob[row_index]);
                              }
                              if (has_hi) {
                                    log_Phi_B[row_index] = use_approx
                                          ? -std::abs(x_hi * (a_Phi_approx * x_hi * x_hi + b_Phi_approx))
                                          : (-0.5 * x_hi * x_hi - HALF_LOG_TWO_PI - log_prob[row_index]);
                              }
                        }
                        if (do_grad && use_approx) {
                              //// Phi_approx density: exp(-|p|) (NicoSIMD), then (3 a x^2 + b) / (1 + exp(-|p|))^2, its log
                              //// (NicoSIMD), then - |p| - log P; rows without the bound keep -1000
                              apply_raw<vec, Fn::exp, true>(log_Phi_A, n_obs);
                              apply_raw<vec, Fn::exp, true>(log_Phi_B, n_obs);
                              for (int row_index = 0; row_index < n_obs; row_index++) {
                                    const int y_lo = y_ref(row_index, test_index);
                                    const double mu_val = Xbeta_lik[class_index](row_index, test_index);
                                    double x_lo = 0.0;
                                    double x_hi = 0.0;
                                    if (y_lo >= 0 && ord_index < 0) {
                                          x_lo = ((y_lo == 1) ? 1.0 : -1.0) * mu_val;
                                    } else if (y_lo >= 0) {
                                          const int y_hi = prep.has_y_upper ? (*prep.y_upper_mat)(row_index, test_index) : y_lo;
                                          x_lo = (y_lo == 1) ? 0.0 : (prep.C_4[class_index](y_lo - 2, ord_index) - mu_val);
                                          x_hi = (y_hi == n_cat_test) ? 0.0 : (prep.C_4[class_index](y_hi - 1, ord_index) - mu_val);
                                    }
                                    const double one_plus_e_lo = 1.0 + log_Phi_A[row_index];
                                    const double one_plus_e_hi = 1.0 + log_Phi_B[row_index];
                                    log_Phi_A[row_index] = (3.0 * a_Phi_approx * x_lo * x_lo + b_Phi_approx) / (one_plus_e_lo * one_plus_e_lo);
                                    log_Phi_B[row_index] = (3.0 * a_Phi_approx * x_hi * x_hi + b_Phi_approx) / (one_plus_e_hi * one_plus_e_hi);
                              }
                              apply_raw<vec, Fn::log, true>(log_Phi_A, n_obs);
                              apply_raw<vec, Fn::log, true>(log_Phi_B, n_obs);
                              for (int row_index = 0; row_index < n_obs; row_index++) {
                                    const int y_lo = y_ref(row_index, test_index);
                                    const double mu_val = Xbeta_lik[class_index](row_index, test_index);
                                    bool has_lo = false;
                                    bool has_hi = false;
                                    double x_lo = 0.0;
                                    double x_hi = 0.0;
                                    if (y_lo >= 0 && ord_index < 0) {
                                          x_lo = ((y_lo == 1) ? 1.0 : -1.0) * mu_val;
                                          has_lo = true;
                                    } else if (y_lo >= 0) {
                                          const int y_hi = prep.has_y_upper ? (*prep.y_upper_mat)(row_index, test_index) : y_lo;
                                          const bool open_lo = (y_lo == 1);
                                          const bool open_hi = (y_hi == n_cat_test);
                                          has_lo = !open_lo && !(open_lo && open_hi);
                                          has_hi = !open_hi && !(open_lo && open_hi);
                                          x_lo = open_lo ? 0.0 : (prep.C_4[class_index](y_lo - 2, ord_index) - mu_val);
                                          x_hi = open_hi ? 0.0 : (prep.C_4[class_index](y_hi - 1, ord_index) - mu_val);
                                    }
                                    log_Phi_A[row_index] = has_lo
                                          ? (log_Phi_A[row_index] - std::abs(x_lo * (a_Phi_approx * x_lo * x_lo + b_Phi_approx)) - log_prob[row_index])
                                          : -1000.0;
                                    log_Phi_B[row_index] = has_hi
                                          ? (log_Phi_B[row_index] - std::abs(x_hi * (a_Phi_approx * x_hi * x_hi + b_Phi_approx)) - log_prob[row_index])
                                          : -1000.0;
                              }
                        }
                        //// 5. the gradient ratios (NicoSIMD exp); d/dmu = ratio_lo - ratio_hi (binary: sign * ratio),
                        ////    d/dC_lo = -ratio_lo, d/dC_hi = ratio_hi
                        if (do_grad) {
                              apply_raw<vec, Fn::exp, true>(log_Phi_A, n_obs);
                              apply_raw<vec, Fn::exp, true>(log_Phi_B, n_obs);
                        }
                        for (int row_index = 0; row_index < n_obs; row_index++) {
                              const double ratio_lo = do_grad ? log_Phi_A[row_index] : 0.0;
                              const double ratio_hi = do_grad ? log_Phi_B[row_index] : 0.0;
                              if (ord_index < 0) {
                                    const int y_lo = y_ref(row_index, test_index);
                                    log_prob[row_index] = ((y_lo == 1) ? 1.0 : -1.0) * ratio_lo;
                                    log_Phi_A[row_index] = 0.0;
                                    log_Phi_B[row_index] = 0.0;
                              } else {
                                    log_prob[row_index] = ratio_lo - ratio_hi;
                                    log_Phi_A[row_index] = -ratio_lo;
                                    log_Phi_B[row_index] = ratio_hi;
                              }
                        }
                  }
            }
            //// Log-sum-exp over classes and posterior class responsibilities, a column at a time (NicoSIMD exp and log):
            //// out_mat's log-likelihood segment first holds each row's max, the columns of d_lp_d_log_prev hold
            //// exp(lp - max), lp_mat's first column (not used after this) holds their sum, then its log
            const int ll_start = 1 + n_params;
            double *lse_of_row = out_mat.data() + ll_start;
            for (int row_index = 0; row_index < n_obs; row_index++) {
                  double max_lp = lp_mat(row_index, 0);
                  for (int class_index = 1; class_index < n_class; class_index++) {
                        if (lp_mat(row_index, class_index) > max_lp) {
                              max_lp = lp_mat(row_index, class_index);
                        }
                  }
                  lse_of_row[row_index] = max_lp;
            }
            for (int class_index = 0; class_index < n_class; class_index++) {
                  double *exp_col = d_lp_d_log_prev.col(class_index).data();
                  for (int row_index = 0; row_index < n_obs; row_index++) {
                        exp_col[row_index] = lp_mat(row_index, class_index) - lse_of_row[row_index];
                  }
                  apply_raw<vec, Fn::exp, true>(exp_col, n_obs);
            }
            double *sum_exp = lp_mat.col(0).data();
            for (int row_index = 0; row_index < n_obs; row_index++) {
                  double sum_val = 0.0;
                  for (int class_index = 0; class_index < n_class; class_index++) {
                        sum_val += d_lp_d_log_prev(row_index, class_index);
                  }
                  sum_exp[row_index] = sum_val;
                  if (do_grad) {
                        for (int class_index = 0; class_index < n_class; class_index++) {
                              d_lp_d_log_prev(row_index, class_index) /= sum_val;
                        }
                  }
            }
            apply_raw<vec, Fn::log, true>(sum_exp, n_obs);
            for (int row_index = 0; row_index < n_obs; row_index++) {
                  lse_of_row[row_index] += sum_exp[row_index];
            }
            if (!do_grad) {
                  d_lp_d_log_prev.setZero();
            }

            if (do_grad) {
                  for (int class_index = 0; class_index < n_class; class_index++) {
                        for (int test_index = 0; test_index < n_tests; test_index++) {
                              const Eigen::Matrix<double, -1, 1> dmu_vec =
                                    d_lp_d_log_prev.col(class_index).array()
                                    * dlogp_dmu[class_index].col(test_index).array();
                              const int n_covs_ct = (*prep.n_covs_per_outcome)(class_index, test_index);
                              if ((n_covs_ct > 0) && ((*prep.is_perfect_ref)(test_index) == 0)) {
                                    beta_4_grad_lik[class_index].col(test_index).head(n_covs_ct).noalias() +=
                                          (*prep.X_class_test)[prep.class_index_of_X_to_read(class_index)][prep.test_index_of_X_to_read(test_index)].leftCols(n_covs_ct).transpose()
                                          * dmu_vec;
                              }
                              const int ord_index = prep.ord_idx_of_test(test_index);
                              if (ord_index >= 0) {
                                    const int n_cat_test = (*prep.n_cat_per_ord_test)(ord_index);
                                    for (int row_index = 0; row_index < n_obs; row_index++) {
                                          const int y_lo = y_ref(row_index, test_index);
                                          if (y_lo < 0) {
                                                continue;
                                          }
                                          const int y_hi = prep.has_y_upper
                                                           ? (*prep.y_upper_mat)(row_index, test_index)
                                                           : y_lo;
                                          const double resp_nc = d_lp_d_log_prev(row_index, class_index);
                                          if ((y_lo >= 2) && (y_lo <= n_cat_test)) {
                                                cutpoint_grad_4[class_index](y_lo - 2, ord_index) +=
                                                      resp_nc * dlogp_dClo[class_index](row_index, test_index);
                                          }
                                          if ((y_hi >= 1) && (y_hi < n_cat_test)) {
                                                cutpoint_grad_4[class_index](y_hi - 1, ord_index) +=
                                                      resp_nc * dlogp_dChi[class_index](row_index, test_index);
                                          }
                                    }
                              }
                        }
                  }
            }
      }

      fn_CLC_MVOP_assemble(
            out_mat,
            theta_main_vec_ref,
            prep,
            Model_args_as_cpp_struct,
            log_jac_u,
            beta_4_grad_lik,
            U_Omega_grad_array,
            cutpoint_grad_4,
            d_lp_d_log_prev
      );
}


//// -------------------------------------------------------------------------------------
//// 2. Shared NoLog GHK chunk processor for CLC-MVOP
//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_lp_grad_CLC_MVOP_NoLog_process_chunk_T(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const std::string &grad_option,
      const int chunk_counter,
      const int chunk_size,
      const CLC_MVOP_prep &prep,
      double &log_jac_u,
      std::vector<Eigen::Matrix<double, -1, -1>> &beta_4_grad_lik,
      std::vector<Eigen::Matrix<double, -1, -1>> &U_Omega_grad_array,
      std::vector<Eigen::Matrix<double, -1, -1>> &cutpoint_grad_4,
      Eigen::Matrix<double, -1, -1> &d_lp_d_log_prev_subject,
      LC_MVP_workspace_struct &ws_cur,
      const Model_fn_args_struct &Model_args_as_cpp_struct
) {
      const double sqrt_2_pi_recip = 1.0 / std::sqrt(2.0 * M_PI);
      const double pos_inf = std::numeric_limits<double>::infinity();
      const int n_params = prep.n_params;
      const int n_tests = prep.n_tests;
      const int n_class = prep.n_class;
      const int row_start = prep.chunk_size_orig * chunk_counter;
      const KernelChoice &kernel_choice = prep.kchoice;
      const Eigen::Matrix<int, -1, 1> &ord_idx_of_test = prep.ord_idx_of_test;
      const Eigen::Matrix<int, -1, 1> &n_cat_per_ord_test = *prep.n_cat_per_ord_test;
      const Eigen::Matrix<int, -1, 1> &n_thr_per_ord_test = *prep.n_thr_per_ord_test;
      const std::vector<Eigen::Matrix<double, -1, -1>> &cutpoint_mats = prep.C_4;
      const std::vector<Eigen::Matrix<double, -1, -1>> &L_Omega_mats = prep.L_Omega_double;
      const std::vector<Eigen::Matrix<double, -1, -1>> &L_Omega_recip_mats = prep.L_Omega_recip_double;

      if ((ws_cur.y_chunk.rows() != chunk_size)
          || (ws_cur.y_chunk.cols() != n_tests)
          || (static_cast<int>(ws_cur.Upper_Bound_Z.size()) < n_class)
          || (static_cast<int>(ws_cur.Z_std_norm.size()) < n_class)
          || (ws_cur.Z_std_norm[0].rows() != chunk_size)
          || (ws_cur.lp_array.cols() != n_class)) {
            fn_CLC_MVOP_resize_ws_for_chunk(
                  ws_cur,
                  chunk_size,
                  n_tests,
                  n_class
            );
      }

      std::vector<Eigen::Matrix<double, -1, -1>> &Z_std_norm = ws_cur.Z_std_norm;
      std::vector<Eigen::Matrix<double, -1, -1>> &Bound_Z = ws_cur.Bound_Z;
      std::vector<Eigen::Matrix<double, -1, -1>> &Bound_U_Phi_Bound_Z = ws_cur.Bound_U_Phi_Bound_Z;
      std::vector<Eigen::Matrix<double, -1, -1>> &prob = ws_cur.prob;
      std::vector<Eigen::Matrix<double, -1, -1>> &Phi_Z = ws_cur.Phi_Z;
      std::vector<Eigen::Matrix<double, -1, -1>> &Upper_Bound_Z = ws_cur.Upper_Bound_Z;
      Eigen::Matrix<double, -1, -1> &y1_log_prob = ws_cur.y1_log_prob;
      Eigen::Matrix<double, -1, -1> &phi_Z_recip = ws_cur.phi_Z_recip;
      Eigen::Matrix<double, -1, -1> &phi_Bound_Z = ws_cur.phi_Bound_Z;
      Eigen::Matrix<double, -1, -1> &u_grad_array_CM_chunk = ws_cur.u_grad_array_CM_chunk;
      Eigen::Matrix<double, -1, -1> &common_grad_term_1 = ws_cur.common_grad_term_1;
      Eigen::Matrix<double, -1, -1> &prob_rowwise_prod_temp = ws_cur.prob_rowwise_prod_temp;
      Eigen::Matrix<double, -1, -1> &prob_recip_rowwise_prod_temp = ws_cur.prob_recip_rowwise_prod_temp;
      Eigen::Matrix<double, -1, 1> &prod_container_or_inc_array = ws_cur.prod_container_or_inc_array;
      Eigen::Matrix<double, -1, 1> &derivs_chain_container_vec = ws_cur.derivs_chain_container_vec;
      Eigen::Matrix<double, -1, 1> &prob_rowwise_prod_temp_all = ws_cur.prob_rowwise_prod_temp_all;
      Eigen::Matrix<double, -1, 1> &dphi_direct_Cj = ws_cur.dphi_direct_Cj;
      Eigen::Matrix<double, -1, 1> &dZ_direct_Cj = ws_cur.dZ_direct_Cj;
      Eigen::Matrix<double, -1, -1> &grad_prob = ws_cur.grad_prob;
      Eigen::Matrix<double, -1, -1> &z_grad_term = ws_cur.z_grad_term;
      Eigen::Matrix<double, -1, -1> &y_chunk = ws_cur.y_chunk;
      static thread_local Eigen::Matrix<double, -1, -1> y_upper_chunk;
      if (prep.has_y_upper && ((y_upper_chunk.rows() != chunk_size) || (y_upper_chunk.cols() != n_tests))) {
            y_upper_chunk.resize(
                  chunk_size,
                  n_tests
            );
      }
      Eigen::Matrix<double, -1, -1> &u_array = ws_cur.u_array;
      Eigen::Matrix<double, -1, -1> &y_sign = ws_cur.y_sign;
      Eigen::Matrix<double, -1, -1> &y_m_y_sign_x_u = ws_cur.y_m_y_sign_x_u;
      Eigen::Matrix<double, -1, -1> &u_grad_array_CM_chunk_block = ws_cur.u_grad_array_CM_chunk_block;
      Eigen::Matrix<double, -1, 1> &u_unc_vec_chunk = ws_cur.u_unc_vec_chunk;
      Eigen::Matrix<double, -1, 1> &u_vec_chunk = ws_cur.u_vec_chunk;
      Eigen::Matrix<double, -1, 1> &du_wrt_duu_chunk = ws_cur.du_wrt_duu_chunk;
      Eigen::Matrix<double, -1, 1> &d_J_wrt_duu_chunk = ws_cur.d_J_wrt_duu_chunk;
      Eigen::Matrix<double, -1, -1> &lp_array = ws_cur.lp_array;
      Eigen::Matrix<double, -1, 1> &prob_n = ws_cur.prob_n;
      Eigen::Matrix<double, -1, 1> &prob_n_recip = ws_cur.prob_n_recip;
      Eigen::Matrix<double, -1, 1> &log_sum_result = ws_cur.log_sum_result;
      Eigen::Matrix<double, -1, 1> &container_max_logs = ws_cur.container_max_logs;
      Eigen::Matrix<double, -1, 1> &rowwise_prod = ws_cur.rowwise_prod;
      Eigen::Matrix<double, -1, 1> &rowwise_sum = ws_cur.rowwise_sum;
      Eigen::Matrix<double, -1, 1> &log_lik_chunk = ws_cur.log_lik_chunk;
      Eigen::Matrix<double, -1, -1> &prob_recip = ws_cur.prob_recip;
      Eigen::Matrix<double, -1, -1> &phi_Upper_Bound_Z = ws_cur.phi_Upper_Bound_Z;
      Eigen::Matrix<double, -1, -1> &dphi_over_L = ws_cur.dphi_over_L;
      Eigen::Matrix<double, -1, -1> &dZ_dmu_neg = ws_cur.dZ_dmu_neg;
      Eigen::Matrix<double, -1, -1> &dphi_times_bz = ws_cur.dphi_times_bz;
      Eigen::Matrix<double, -1, -1> &dZ_times_bz = ws_cur.dZ_times_bz;
      Eigen::Matrix<double, -1, 1> &log_prev_per_obs_given_c = ws_cur.log_prev_per_obs_given_c;
      Eigen::Matrix<double, -1, 1> &prev_per_obs_given_c = ws_cur.prev_per_obs_given_c;

      u_grad_array_CM_chunk.setZero();
      y_chunk = y_ref.middleRows(
            row_start,
            chunk_size
      ).cast<double>();
      if (prep.has_y_upper) {
            y_upper_chunk = prep.y_upper_mat->middleRows(
                  row_start,
                  chunk_size
            ).cast<double>();
      }
      u_unc_vec_chunk = theta_us_vec_ref.segment(
            row_start * n_tests,
            chunk_size * n_tests
      );
      fn_MVP_compute_nuisance_T<vec>(
            u_vec_chunk,
            u_unc_vec_chunk,
            kernel_choice
      );
      log_jac_u += fn_MVP_compute_nuisance_log_jac_u_T<vec>(
            u_vec_chunk,
            u_unc_vec_chunk,
            d_J_wrt_duu_chunk,
            kernel_choice
      );
      u_array = u_vec_chunk.reshaped(
            chunk_size,
            n_tests
      );

      //// Missing results (y == -1) via shared GHK helpers (thread_local reusable buffers)
      const bool chunk_has_missing_results = fn_MVP_rows_of_y_have_a_missing_result(
            y_ref,
            row_start,
            chunk_size
      );
      static thread_local Eigen::Matrix<double, -1, -1> is_missing_result;
      static thread_local Eigen::Matrix<double, -1, -1> Z_untruncated;
      if (chunk_has_missing_results) {
            if ((is_missing_result.rows() != chunk_size) || (is_missing_result.cols() != n_tests)) {
                  is_missing_result.resize(chunk_size, n_tests);
                  Z_untruncated.resize(chunk_size, n_tests);
            }
            fn_MVP_missing_result_indicator_and_recode_missing_results_to_y_equals_1(
                  y_chunk,
                  is_missing_result
            );
            if (prep.has_y_upper) {
                  for (int test_index = 0; test_index < n_tests; test_index++) {
                        const int ord_index = ord_idx_of_test(test_index);
                        const double fill_hi = (ord_index >= 0)
                                               ? static_cast<double>(n_cat_per_ord_test(ord_index))
                                               : 1.0;
                        y_upper_chunk.col(test_index).array() =
                              (is_missing_result.col(test_index).array() > 0.5).select(
                                    fill_hi,
                                    y_upper_chunk.col(test_index).array()
                              );
                  }
            }
            fn_MVP_compute_untruncated_GHK_Z_of_missing_results_T<vec>(
                  Z_untruncated,
                  u_array,
                  is_missing_result,
                  kernel_choice
            );
      }
      y_sign.array() = 2.0 * y_chunk.array() - 1.0;
      y_m_y_sign_x_u.array() = y_chunk.array() - (y_sign.array() * u_array.array());

      static thread_local std::vector<Eigen::Matrix<double, -1, -1>> Xbeta_chunk;
      if (static_cast<int>(Xbeta_chunk.size()) != n_class) {
            Xbeta_chunk.resize(static_cast<std::size_t>(n_class));
      }
      for (int class_index = 0; class_index < n_class; class_index++) {
            if ((Xbeta_chunk[class_index].rows() != chunk_size)
                || (Xbeta_chunk[class_index].cols() != n_tests)) {
                  Xbeta_chunk[class_index].resize(
                        chunk_size,
                        n_tests
                  );
            }
            for (int test_index = 0; test_index < n_tests; test_index++) {
                  if ((*prep.is_perfect_ref)(test_index) == 1) {
                        Xbeta_chunk[class_index].col(test_index).setConstant(
                              prep.beta_4[class_index](0, test_index)
                        );
                  } else {
                        const int n_covs_ct = (*prep.n_covs_per_outcome)(class_index, test_index);
                        if (n_covs_ct > 0) {
                              Xbeta_chunk[class_index].col(test_index).noalias() =
                                    (*prep.X_class_test)[prep.class_index_of_X_to_read(class_index)][prep.test_index_of_X_to_read(test_index)].block(
                                          row_start,
                                          0,
                                          chunk_size,
                                          n_covs_ct
                                    )
                                    * prep.beta_4[class_index].col(test_index).head(n_covs_ct);
                        } else {
                              Xbeta_chunk[class_index].col(test_index).setZero();
                        }
                  }
            }
      }

      //// -------------------------------------------------------------------------------------
      //// Forward likelihood
      //// -------------------------------------------------------------------------------------
      static thread_local Eigen::Matrix<double, -1, 1> mu_vec;
      if (mu_vec.rows() != chunk_size) {
            mu_vec.resize(chunk_size);
      }
      for (int class_index = 0; class_index < n_class; class_index++) {
            log_prev_per_obs_given_c = prep.log_prev_subject.col(class_index).segment(
                  row_start,
                  chunk_size
            );
            prev_per_obs_given_c = prep.prev_subject.col(class_index).segment(
                  row_start,
                  chunk_size
            );
            prod_container_or_inc_array.setZero();
            for (int test_index = 0; test_index < n_tests; test_index++) {
                  const int ord_index = ord_idx_of_test(test_index);
                  const double L_tt_inv = L_Omega_recip_mats[class_index](test_index, test_index);
                  mu_vec.array() = Xbeta_chunk[class_index].col(test_index).array()
                                   + prod_container_or_inc_array.array();
                  if (ord_index < 0) {
                        Bound_Z[class_index].col(test_index).array() = -L_tt_inv * mu_vec.array();
                  } else {
                        const int n_cat_test = n_cat_per_ord_test(ord_index);
                        for (int row_index = 0; row_index < chunk_size; row_index++) {
                              const int y_lo = static_cast<int>(y_chunk(row_index, test_index));
                              const int y_hi = prep.has_y_upper
                                               ? static_cast<int>(y_upper_chunk(row_index, test_index))
                                               : y_lo;
                              Bound_Z[class_index](row_index, test_index) =
                                    (y_lo == 1)
                                    ? -pos_inf
                                    : (cutpoint_mats[class_index](y_lo - 2, ord_index) - mu_vec(row_index)) * L_tt_inv;
                              Upper_Bound_Z[class_index](row_index, test_index) =
                                    (y_hi == n_cat_test)
                                    ? pos_inf
                                    : (cutpoint_mats[class_index](y_hi - 1, ord_index) - mu_vec(row_index)) * L_tt_inv;
                        }
                        if (chunk_has_missing_results) {
                              fn_MVOP_set_bounds_of_missing_results_to_minus_and_plus_infinity(
                                    test_index,
                                    is_missing_result,
                                    Bound_Z[class_index],
                                    Upper_Bound_Z[class_index]
                              );
                        }
                  }
                  fn_MVOP_compute_lp_GHK_cols_T<vec>(
                        test_index,
                        Bound_U_Phi_Bound_Z[class_index],
                        Phi_Z[class_index],
                        Z_std_norm[class_index],
                        prob[class_index],
                        y1_log_prob,
                        Bound_Z[class_index],
                        y_chunk,
                        u_array,
                        Upper_Bound_Z[class_index],
                        (ord_index < 0),
                        kernel_choice
                  );
                  if (chunk_has_missing_results) {
                        fn_MVP_set_GHK_col_of_missing_results_to_the_untruncated_draw(
                              test_index,
                              is_missing_result,
                              u_array,
                              Z_untruncated,
                              Bound_U_Phi_Bound_Z[class_index],
                              Phi_Z[class_index],
                              Z_std_norm[class_index],
                              prob[class_index],
                              y1_log_prob
                        );
                  }
                  if (test_index < n_tests - 1) {
                        prod_container_or_inc_array =
                              Z_std_norm[class_index].leftCols(test_index + 1)
                              * L_Omega_mats[class_index].row(test_index + 1).head(test_index + 1).transpose();
                  }
            }
            rowwise_sum = y1_log_prob.rowwise().sum();
            rowwise_sum.array() += log_prev_per_obs_given_c.array();
            lp_array.col(class_index) = rowwise_sum;
      }

      log_sum_exp_general_T<vec>(
            lp_array,
            log_sum_result,
            container_max_logs
      );
      out_mat.segment(
            1 + n_params + row_start,
            chunk_size
      ) = log_sum_result;
      log_lik_chunk = log_sum_result;
      prob_n = log_lik_chunk;
      apply_inplace<vec, Fn::exp>(prob_n);
      prob_n_recip.array() = 1.0 / prob_n.array();

      //// -------------------------------------------------------------------------------------
      //// Backward gradients via shared MVOP and MVP manual-gradient kernels
      //// -------------------------------------------------------------------------------------
      const bool do_us = (grad_option == "us_only") || (grad_option == "all");
      const bool do_coeff = (grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only");
      const bool do_corr = (grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only");
      const bool do_cut = (grad_option == "main_only") || (grad_option == "all") || (grad_option == "cutpoints_only");
      const bool do_prev = (grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only");
      const Eigen::Matrix<double, -1, -1> *y_upper_ptr = prep.has_y_upper ? &y_upper_chunk : nullptr;
      static thread_local Eigen::Matrix<int, -1, 1> n_cov_vec_c;
      if (n_cov_vec_c.rows() != n_tests) {
            n_cov_vec_c.resize(n_tests);
      }

      for (int class_index = 0; class_index < n_class; class_index++) {
            prev_per_obs_given_c = prep.prev_subject.col(class_index).segment(
                  row_start,
                  chunk_size
            );
            prob_recip.array() = 1.0 / prob[class_index].array();
            for (int test_index = 0; test_index < n_tests; test_index++) {
                  fn_MVP_compute_phi_Z_recip_cols_T<vec>(
                        test_index,
                        phi_Z_recip,
                        Phi_Z[class_index],
                        Z_std_norm[class_index],
                        kernel_choice
                  );
                  if (ord_idx_of_test(test_index) < 0) {
                        fn_MVP_compute_phi_Bound_Z_cols_T<vec>(
                              test_index,
                              phi_Bound_Z,
                              Bound_U_Phi_Bound_Z[class_index],
                              Bound_Z[class_index],
                              kernel_choice
                        );
                        if (chunk_has_missing_results) {
                              fn_MVP_set_phi_Bound_Z_col_of_missing_results_to_zero(
                                    test_index,
                                    is_missing_result,
                                    phi_Bound_Z
                              );
                        }
                  } else {
                        if (kernel_choice.Phi_approx) {
                              phi_Bound_Z.col(test_index) = Bound_Z[class_index].col(test_index);
                              Phi_col_inplace<vec>(
                                    phi_Bound_Z,
                                    test_index,
                                    kernel_choice
                              );
                              phi_Upper_Bound_Z.col(test_index) = Upper_Bound_Z[class_index].col(test_index);
                              Phi_col_inplace<vec>(
                                    phi_Upper_Bound_Z,
                                    test_index,
                                    kernel_choice
                              );
                              Eigen::Matrix<double, -1, -1> neg_bounds(
                                    Bound_Z[class_index].rows(),
                                    2
                              );
                              neg_bounds.col(0) = -Bound_Z[class_index].col(test_index);
                              neg_bounds.col(1) = -Upper_Bound_Z[class_index].col(test_index);
                              Phi_col_inplace<vec>(
                                    neg_bounds,
                                    0,
                                    kernel_choice
                              );
                              Phi_col_inplace<vec>(
                                    neg_bounds,
                                    1,
                                    kernel_choice
                              );
                              phi_Bound_Z.col(test_index).array() =
                                    (3.0 * 0.07056 * Bound_Z[class_index].col(test_index).array().square() + 1.5976)
                                    * phi_Bound_Z.col(test_index).array()
                                    * neg_bounds.col(0).array();
                              phi_Upper_Bound_Z.col(test_index).array() =
                                    (3.0 * 0.07056 * Upper_Bound_Z[class_index].col(test_index).array().square() + 1.5976)
                                    * phi_Upper_Bound_Z.col(test_index).array()
                                    * neg_bounds.col(1).array();
                        } else {
                              phi_Bound_Z.col(test_index).array() =
                                    -0.5 * Bound_Z[class_index].col(test_index).array().square();
                              apply_col_inplace<vec, Fn::exp>(
                                    phi_Bound_Z,
                                    test_index
                              );
                              phi_Bound_Z.col(test_index).array() *= sqrt_2_pi_recip;
                              phi_Upper_Bound_Z.col(test_index).array() =
                                    -0.5 * Upper_Bound_Z[class_index].col(test_index).array().square();
                              apply_col_inplace<vec, Fn::exp>(
                                    phi_Upper_Bound_Z,
                                    test_index
                              );
                              phi_Upper_Bound_Z.col(test_index).array() *= sqrt_2_pi_recip;
                        }
                        phi_Bound_Z.col(test_index).array() =
                              Bound_Z[class_index].col(test_index).array().isInf().select(
                                    0.0,
                                    phi_Bound_Z.col(test_index).array()
                              );
                        phi_Upper_Bound_Z.col(test_index).array() =
                              Upper_Bound_Z[class_index].col(test_index).array().isInf().select(
                                    0.0,
                                    phi_Upper_Bound_Z.col(test_index).array()
                              );
                  }
            }
            if (grad_option != "none") {
                  fn_MVOP_grad_prep(
                        prob[class_index],
                        y_sign,
                        y_m_y_sign_x_u,
                        u_array,
                        L_Omega_recip_mats[class_index],
                        prev_per_obs_given_c,
                        prob_n_recip,
                        phi_Z_recip,
                        phi_Bound_Z,
                        phi_Upper_Bound_Z,
                        Bound_Z[class_index],
                        Upper_Bound_Z[class_index],
                        prob_recip,
                        prob_rowwise_prod_temp,
                        prob_recip_rowwise_prod_temp,
                        prob_rowwise_prod_temp_all,
                        common_grad_term_1,
                        dphi_over_L,
                        dZ_dmu_neg,
                        dphi_times_bz,
                        dZ_times_bz,
                        ord_idx_of_test,
                        Model_args_as_cpp_struct
                  );
            }
            if (do_us) {
                  u_grad_array_CM_chunk_block = u_grad_array_CM_chunk.block(
                        0,
                        0,
                        chunk_size,
                        n_tests
                  );
                  fn_MVP_compute_nuisance_grad_v2(
                        u_grad_array_CM_chunk_block,
                        phi_Z_recip,
                        common_grad_term_1,
                        L_Omega_mats[class_index],
                        prob[class_index],
                        prob_recip,
                        prob_rowwise_prod_temp,
                        dphi_over_L,
                        dZ_dmu_neg,
                        z_grad_term,
                        grad_prob,
                        prod_container_or_inc_array,
                        derivs_chain_container_vec,
                        Model_args_as_cpp_struct
                  );
                  u_grad_array_CM_chunk.block(
                        0,
                        0,
                        chunk_size,
                        n_tests
                  ).array() += u_grad_array_CM_chunk_block.array();
                  if (class_index == n_class - 1) {
                        const int start_index = 1 + row_start * n_tests;
                        const int block_length = chunk_size * n_tests;
                        out_mat.segment(
                              start_index,
                              block_length
                        ) = u_grad_array_CM_chunk.reshaped();
                        fn_MVP_nuisance_first_deriv_T<vec>(
                              du_wrt_duu_chunk,
                              u_vec_chunk,
                              u_unc_vec_chunk,
                              kernel_choice
                        );
                        fn_MVP_nuisance_deriv_of_log_det_J_T<vec>(
                              d_J_wrt_duu_chunk,
                              u_vec_chunk,
                              u_unc_vec_chunk,
                              du_wrt_duu_chunk,
                              kernel_choice
                        );
                        out_mat.segment(
                              start_index,
                              block_length
                        ).array() *= du_wrt_duu_chunk.array();
                        out_mat.segment(
                              start_index,
                              block_length
                        ).array() += d_J_wrt_duu_chunk.array();
                  }
            }
            if (do_coeff) {
                  n_cov_vec_c = prep.n_covs_per_outcome->row(class_index).transpose();
                  fn_MVP_compute_coefficients_grad_v3(
                        class_index,
                        beta_4_grad_lik[class_index],
                        (*prep.X_class_test)[prep.class_index_of_X_to_read(class_index)],
                        n_cov_vec_c,
                        row_start,
                        prep.n_covariates_max,
                        common_grad_term_1,
                        L_Omega_mats[class_index],
                        prob[class_index],
                        prob_recip,
                        prob_rowwise_prod_temp,
                        dphi_over_L,
                        dZ_dmu_neg,
                        z_grad_term,
                        grad_prob,
                        prod_container_or_inc_array,
                        derivs_chain_container_vec,
                        true,
                        Model_args_as_cpp_struct
                  );
                  for (int test_index = 0; test_index < n_tests; test_index++) {
                        if ((*prep.is_perfect_ref)(test_index) == 1) {
                              beta_4_grad_lik[class_index].col(test_index).setZero();
                        }
                  }
            }
            if (do_corr) {
                  fn_MOVP_compute_L_Omega_grad_v3(
                        U_Omega_grad_array[class_index],
                        common_grad_term_1,
                        L_Omega_mats[class_index],
                        prob[class_index],
                        prob_recip,
                        Z_std_norm[class_index],
                        prob_rowwise_prod_temp,
                        dphi_over_L,
                        dZ_dmu_neg,
                        dphi_times_bz,
                        dZ_times_bz,
                        z_grad_term,
                        grad_prob,
                        prod_container_or_inc_array,
                        derivs_chain_container_vec,
                        true,
                        Model_args_as_cpp_struct
                  );
            }
            if (do_cut) {
                  for (int test_index = 0; test_index < n_tests; test_index++) {
                        const int ord_index = ord_idx_of_test(test_index);
                        if (ord_index < 0) {
                              continue;
                        }
                        fn_MVOP_compute_cutpoint_grad_from_one_unit_dZ_seed_propagation_and_category_scatter(
                              class_index,
                              cutpoint_grad_4[class_index],
                              test_index,
                              ord_index,
                              n_thr_per_ord_test(ord_index) + 1,
                              common_grad_term_1,
                              L_Omega_mats[class_index],
                              prob[class_index],
                              prob_recip,
                              prob_rowwise_prod_temp,
                              phi_Bound_Z,
                              phi_Upper_Bound_Z,
                              phi_Z_recip,
                              u_array,
                              y_chunk,
                              dphi_over_L,
                              dZ_dmu_neg,
                              z_grad_term,
                              grad_prob,
                              prod_container_or_inc_array,
                              derivs_chain_container_vec,
                              dphi_direct_Cj,
                              dZ_direct_Cj,
                              true,
                              Model_args_as_cpp_struct
                        );
                  }
            }
            if (do_prev) {
                  rowwise_prod = prob[class_index].rowwise().prod();
                  d_lp_d_log_prev_subject.col(class_index).segment(
                        row_start,
                        chunk_size
                  ).array() = prev_per_obs_given_c.array()
                              * rowwise_prod.array()
                              * prob_n_recip.array();
            }
      }
}


struct CLC_MVOP_NoLog_persistent_state {
      CLC_MVOP_prep prep;
      //// what fn_lp_grad_MVOP_LC_Pinkney_NoLog_process_chunk_T is handed, with each subject as its own population
      MVOP_prep prep_of_LC_MVOP_chunk_function;
      Eigen::Matrix<int, -1, 1> subject_index_as_pop_ind;
      Eigen::Matrix<double, -1, -1> prev_grad_per_subject;
      std::vector<Eigen::Matrix<double, -1, -1>> beta_4_grad_lik;
      std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array;
      std::vector<Eigen::Matrix<double, -1, -1>> cutpoint_grad_4;
      Eigen::Matrix<double, -1, -1> d_lp_d_log_prev_subject;

      void ensure_and_zero(
            const int n_obs,
            const int n_tests,
            const int n_class,
            const int K_max,
            const int n_cut_max,
            const int n_ord_alloc
      ) {
            if (static_cast<int>(beta_4_grad_lik.size()) != n_class) {
                  beta_4_grad_lik.resize(static_cast<std::size_t>(n_class));
                  U_Omega_grad_array.resize(static_cast<std::size_t>(n_class));
                  cutpoint_grad_4.resize(static_cast<std::size_t>(n_class));
            }
            for (int class_index = 0; class_index < n_class; class_index++) {
                  if ((beta_4_grad_lik[class_index].rows() != K_max)
                      || (beta_4_grad_lik[class_index].cols() != n_tests)) {
                        beta_4_grad_lik[class_index].resize(K_max, n_tests);
                  }
                  beta_4_grad_lik[class_index].setZero();

                  if ((U_Omega_grad_array[class_index].rows() != n_tests)
                      || (U_Omega_grad_array[class_index].cols() != n_tests)) {
                        U_Omega_grad_array[class_index].resize(n_tests, n_tests);
                  }
                  U_Omega_grad_array[class_index].setZero();

                  if ((cutpoint_grad_4[class_index].rows() != n_cut_max)
                      || (cutpoint_grad_4[class_index].cols() != n_ord_alloc)) {
                        cutpoint_grad_4[class_index].resize(n_cut_max, n_ord_alloc);
                  }
                  cutpoint_grad_4[class_index].setZero();
            }
            if ((d_lp_d_log_prev_subject.rows() != n_obs)
                || (d_lp_d_log_prev_subject.cols() != n_class)) {
                  d_lp_d_log_prev_subject.resize(n_obs, n_class);
            }
            d_lp_d_log_prev_subject.setZero();
      }
};


template <Vec vec>
inline void fn_lp_grad_CLC_MVOP_NoLog_Inplace_process_T(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const std::string &grad_option,
      const Model_fn_args_struct &Model_args_as_cpp_struct,
      std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
      const int n_threads_WCP
) {
      static thread_local CLC_MVOP_NoLog_persistent_state state;
      CLC_MVOP_prep &prep = state.prep;
      fn_CLC_MVOP_prep_inplace(
            prep,
            out_mat,
            theta_main_vec_ref,
            theta_us_vec_ref,
            y_ref,
            Model_args_as_cpp_struct,
            1
      );
      if (prep.label_constraint_violated) {
            out_mat.setZero();
            out_mat(0) = -std::numeric_limits<double>::infinity();
            return;
      }
      const int n_obs = prep.n_obs;
      const int n_tests = prep.n_tests;
      const int n_class = prep.n_class;
      const int K_max = std::max(1, prep.n_covariates_max);
      const int n_cut_max = std::max(1, prep.n_cutpoints_max);
      const int n_ord_alloc = std::max(1, prep.n_ordinal_tests);

      state.ensure_and_zero(
            n_obs,
            n_tests,
            n_class,
            K_max,
            n_cut_max,
            n_ord_alloc
      );
      std::vector<Eigen::Matrix<double, -1, -1>> &beta_4_grad_lik = state.beta_4_grad_lik;
      std::vector<Eigen::Matrix<double, -1, -1>> &U_Omega_grad_array = state.U_Omega_grad_array;
      std::vector<Eigen::Matrix<double, -1, -1>> &cutpoint_grad_4 = state.cutpoint_grad_4;
      Eigen::Matrix<double, -1, -1> &d_lp_d_log_prev_subject = state.d_lp_d_log_prev_subject;
      double log_jac_u = 0.0;

      if (prep.prior_only) {
            fn_CLC_MVOP_prior_only_nuisance_anchor_T<vec>(
                  out_mat,
                  theta_us_vec_ref,
                  grad_option,
                  prep,
                  log_jac_u
            );
      } else {
            fn_CLC_MVOP_ensure_workspaces(
                  LC_MVP_ws_structs,
                  prep,
                  n_threads_WCP
            );
            //// the LC_MVOP chunk function is called as it is unless the data have interval results or a perfect reference test
            const bool call_LC_MVOP_chunk_function = (n_threads_WCP <= 1)
                                                     && (!prep.has_y_upper)
                                                     && (!(prep.is_perfect_ref->array() == 1).any());
            if (call_LC_MVOP_chunk_function) {
                  //// the remainder chunk has its own workspace, made once at the remainder size, so neither
                  //// workspace is resized during sampling
                  const bool has_remainder_chunk = (prep.n_total_chunks > prep.n_full_chunks);
                  if (has_remainder_chunk && (static_cast<int>(LC_MVP_ws_structs.size()) < 2)) {
                        LC_MVP_ws_structs.resize(2);
                  }
                  LC_MVP_workspace_struct &ws_cur = LC_MVP_ws_structs[0];
                  if (has_remainder_chunk) {
                        LC_MVP_workspace_struct &ws_remainder = LC_MVP_ws_structs[1];
                        if (!ws_remainder.is_allocated
                            || ws_remainder.stored_n_class < prep.n_class
                            || ws_remainder.stored_n_tests != prep.n_tests) {
                              ws_remainder.allocate(prep.last_chunk_size,
                                                    prep.n_tests,
                                                    prep.n_class,
                                                    std::max(1, prep.n_covariates_max),
                                                    std::max(1, prep.n_corrs_total),
                                                    std::max(1, prep.n_covariates_total));
                        }
                        if ((ws_remainder.y_chunk.rows() != prep.last_chunk_size)
                            || (ws_remainder.y_chunk.cols() != n_tests)
                            || (static_cast<int>(ws_remainder.Upper_Bound_Z.size()) < n_class)
                            || (static_cast<int>(ws_remainder.Z_std_norm.size()) < n_class)
                            || (ws_remainder.lp_array.cols() != n_class)) {
                              fn_CLC_MVOP_resize_ws_for_chunk(ws_remainder,
                                                              prep.last_chunk_size,
                                                              n_tests,
                                                              n_class);
                        }
                  }
                  MVOP_prep &P = state.prep_of_LC_MVOP_chunk_function;
                  if (state.subject_index_as_pop_ind.rows() != n_obs) {
                        state.subject_index_as_pop_ind.resize(n_obs);
                        for (int row_index = 0; row_index < n_obs; row_index++) {
                              state.subject_index_as_pop_ind(row_index) = row_index;
                        }
                        state.prev_grad_per_subject.resize(
                              n_obs,
                              n_class
                        );
                  }
                  state.prev_grad_per_subject.setZero();
                  P.N = n_obs;
                  P.n_tests = n_tests;
                  P.n_us = prep.n_us;
                  P.n_params_main = prep.n_params_main;
                  P.n_params = prep.n_params;
                  P.n_class = n_class;
                  P.n_pops = n_obs;
                  P.n_covariates_max = prep.n_covariates_max;
                  P.n_binary_tests = prep.n_binary_tests;
                  P.n_ordinal_tests = prep.n_ordinal_tests;
                  P.n_cutpoints_max = prep.n_cutpoints_max;
                  P.exclude_priors = prep.exclude_priors;
                  P.chunk_size_orig = prep.chunk_size_orig;
                  P.normal_chunk_size = prep.normal_chunk_size;
                  P.last_chunk_size = prep.last_chunk_size;
                  P.n_total_chunks = prep.n_total_chunks;
                  P.n_full_chunks = prep.n_full_chunks;
                  P.kchoice = prep.kchoice;
                  P.X = prep.X_class_test;
                  P.n_covariates_per_outcome_vec = prep.n_covs_per_outcome;
                  P.n_cat_per_ord_test = prep.n_cat_per_ord_test;
                  P.n_thr_per_ord_test = prep.n_thr_per_ord_test;
                  P.pop_ind = &state.subject_index_as_pop_ind;
                  P.ord_idx_of_test = prep.ord_idx_of_test;
                  //// handed over by swap (no copy), and swapped back after the chunk loop
                  P.beta_double_array.swap(prep.beta_4);
                  P.C.swap(prep.C_4);
                  P.L_Omega_double.swap(prep.L_Omega_double);
                  P.L_Omega_recip_double.swap(prep.L_Omega_recip_double);
                  P.prev_mat.swap(prep.prev_subject);
                  P.log_prev_mat_small.swap(prep.log_prev_subject);
                  for (int chunk_index = 0; chunk_index < prep.n_total_chunks; chunk_index++) {
                        const int chunk_size_cur = (chunk_index == prep.n_full_chunks)
                                                   ? prep.last_chunk_size
                                                   : prep.normal_chunk_size;
                        LC_MVP_workspace_struct &ws_of_chunk = (chunk_index == prep.n_full_chunks)
                                                               ? LC_MVP_ws_structs[1]
                                                               : ws_cur;
                        //// the workspace is sized here for every class, so the chunk function finds it the right size
                        if ((ws_of_chunk.y_chunk.rows() != chunk_size_cur) || (ws_of_chunk.y_chunk.cols() != n_tests)) {
                              fn_CLC_MVOP_resize_ws_for_chunk(ws_of_chunk,
                                                              chunk_size_cur,
                                                              n_tests,
                                                              n_class);
                        }
                        fn_lp_grad_MVOP_LC_Pinkney_NoLog_process_chunk_T<vec>(
                              out_mat,
                              theta_us_vec_ref,
                              y_ref,
                              grad_option,
                              chunk_index,
                              chunk_size_cur,
                              P,
                              log_jac_u,
                              beta_4_grad_lik,
                              U_Omega_grad_array,
                              cutpoint_grad_4,
                              state.prev_grad_per_subject,
                              ws_of_chunk,
                              Model_args_as_cpp_struct
                        );
                  }
                  P.beta_double_array.swap(prep.beta_4);
                  P.C.swap(prep.C_4);
                  P.L_Omega_double.swap(prep.L_Omega_double);
                  P.L_Omega_recip_double.swap(prep.L_Omega_recip_double);
                  P.prev_mat.swap(prep.prev_subject);
                  P.log_prev_mat_small.swap(prep.log_prev_subject);
                  //// derivative with respect to the log prevalence of each subject and class
                  d_lp_d_log_prev_subject.array() = prep.prev_subject.array() * state.prev_grad_per_subject.array();
            } else if (n_threads_WCP <= 1) {
                  LC_MVP_workspace_struct &ws_cur = LC_MVP_ws_structs[0];
                  for (int chunk_index = 0; chunk_index < prep.n_total_chunks; chunk_index++) {
                        const int chunk_size_cur = (chunk_index == prep.n_full_chunks)
                                                   ? prep.last_chunk_size
                                                   : prep.normal_chunk_size;
                        fn_lp_grad_CLC_MVOP_NoLog_process_chunk_T<vec>(
                              out_mat,
                              theta_us_vec_ref,
                              y_ref,
                              grad_option,
                              chunk_index,
                              chunk_size_cur,
                              prep,
                              log_jac_u,
                              beta_4_grad_lik,
                              U_Omega_grad_array,
                              cutpoint_grad_4,
                              d_lp_d_log_prev_subject,
                              ws_cur,
                              Model_args_as_cpp_struct
                        );
                  }
            } else {
                  std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> beta_4_per_chunk(prep.n_full_chunks);
                  std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> U_Omega_per_chunk(prep.n_full_chunks);
                  std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> cut_per_chunk(prep.n_full_chunks);
                  std::vector<double> log_jac_u_per_chunk(
                        prep.n_full_chunks,
                        0.0
                  );
                  std::vector<std::exception_ptr> exception_in_chunk(prep.n_full_chunks);
                  #pragma omp parallel num_threads(n_threads_WCP)
                  {
                        std::exception_ptr ws_ex;
#ifdef _OPENMP
                        const int thread_index = omp_get_thread_num();
#else
                        const int thread_index = 0;
#endif
                        LC_MVP_workspace_struct &ws_cur = LC_MVP_ws_structs[thread_index];
                        try {
                              fn_CLC_MVOP_resize_ws_for_chunk(
                                    ws_cur,
                                    prep.normal_chunk_size,
                                    prep.n_tests,
                                    prep.n_class
                              );
                        } catch (...) {
                              ws_ex = std::current_exception();
                        }
                        #pragma omp for schedule(static)
                        for (int chunk_index = 0; chunk_index < prep.n_full_chunks; chunk_index++) {
                              if (ws_ex) {
                                    exception_in_chunk[chunk_index] = ws_ex;
                                    continue;
                              }
                              try {
                                    beta_4_per_chunk[chunk_index].assign(
                                          n_class,
                                          Eigen::Matrix<double, -1, -1>::Zero(
                                                K_max,
                                                n_tests
                                          )
                                    );
                                    U_Omega_per_chunk[chunk_index] = vec_of_mats<double>(
                                          n_tests,
                                          n_tests,
                                          n_class
                                    );
                                    cut_per_chunk[chunk_index] = vec_of_mats<double>(
                                          std::max(1, prep.n_cutpoints_max),
                                          std::max(1, prep.n_ordinal_tests),
                                          n_class
                                    );
                                    fn_lp_grad_CLC_MVOP_NoLog_process_chunk_T<vec>(
                                          out_mat,
                                          theta_us_vec_ref,
                                          y_ref,
                                          grad_option,
                                          chunk_index,
                                          prep.normal_chunk_size,
                                          prep,
                                          log_jac_u_per_chunk[chunk_index],
                                          beta_4_per_chunk[chunk_index],
                                          U_Omega_per_chunk[chunk_index],
                                          cut_per_chunk[chunk_index],
                                          d_lp_d_log_prev_subject,
                                          ws_cur,
                                          Model_args_as_cpp_struct
                                    );
                              } catch (...) {
                                    exception_in_chunk[chunk_index] = std::current_exception();
                              }
                        }
                  }
                  for (int chunk_index = 0; chunk_index < prep.n_full_chunks; chunk_index++) {
                        if (exception_in_chunk[chunk_index]) {
                              std::rethrow_exception(exception_in_chunk[chunk_index]);
                        }
                  }
                  for (int chunk_index = 0; chunk_index < prep.n_full_chunks; chunk_index++) {
                        for (int class_index = 0; class_index < n_class; class_index++) {
                              beta_4_grad_lik[class_index] += beta_4_per_chunk[chunk_index][class_index];
                              U_Omega_grad_array[class_index] += U_Omega_per_chunk[chunk_index][class_index];
                              cutpoint_grad_4[class_index] += cut_per_chunk[chunk_index][class_index];
                        }
                        log_jac_u += log_jac_u_per_chunk[chunk_index];
                  }
                  if ((prep.n_full_chunks < prep.n_total_chunks) && (prep.last_chunk_size > 0)) {
                        LC_MVP_workspace_struct &ws_main = LC_MVP_ws_structs[0];
                        fn_lp_grad_CLC_MVOP_NoLog_process_chunk_T<vec>(
                              out_mat,
                              theta_us_vec_ref,
                              y_ref,
                              grad_option,
                              prep.n_full_chunks,
                              prep.last_chunk_size,
                              prep,
                              log_jac_u,
                              beta_4_grad_lik,
                              U_Omega_grad_array,
                              cutpoint_grad_4,
                              d_lp_d_log_prev_subject,
                              ws_main,
                              Model_args_as_cpp_struct
                        );
                  }
            }
      }

      fn_CLC_MVOP_assemble(
            out_mat,
            theta_main_vec_ref,
            prep,
            Model_args_as_cpp_struct,
            log_jac_u,
            beta_4_grad_lik,
            U_Omega_grad_array,
            cutpoint_grad_4,
            d_lp_d_log_prev_subject
      );
}


//// -------------------------------------------------------------------------------------
//// 3. Shared PartialLog (log-scale tail fallback) for CLC-MVOP
////    Compiled at -O1 to match MVOP_lp_grad_PartialLog_MD_AD_fns.hpp
//// -------------------------------------------------------------------------------------
#pragma GCC push_options
#pragma GCC optimize ("O1")
template <Vec vec>
inline void fn_lp_grad_CLC_MVOP_PartialLog_process_chunk_T(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const std::string &grad_option,
      const int chunk_counter,
      const int chunk_size,
      const CLC_MVOP_prep &prep,
      double &log_jac_u,
      std::vector<Eigen::Matrix<double, -1, -1>> &beta_4_grad_lik,
      std::vector<Eigen::Matrix<double, -1, -1>> &U_Omega_grad_array,
      std::vector<Eigen::Matrix<double, -1, -1>> &cutpoint_grad_4,
      Eigen::Matrix<double, -1, -1> &d_lp_d_log_prev_subject,
      const Model_fn_args_struct &Model_args_as_cpp_struct
) {
      const double sqrt_2_pi_recip = 1.0 / std::sqrt(2.0 * M_PI);
      const double pos_inf = std::numeric_limits<double>::infinity();
      const int n_params = prep.n_params;
      const int n_tests = prep.n_tests;
      const int n_class = prep.n_class;
      const int row_start = prep.chunk_size_orig * chunk_counter;
      const KernelChoice &kernel_choice = prep.kchoice;
      const bool use_scalar = (vec == Vec::Scalar);
      const Eigen::Matrix<int, -1, 1> &ord_idx_of_test = prep.ord_idx_of_test;
      const Eigen::Matrix<int, -1, 1> &n_cat_per_ord_test = *prep.n_cat_per_ord_test;
      const Eigen::Matrix<int, -1, 1> &n_thr_per_ord_test = *prep.n_thr_per_ord_test;
      const std::vector<Eigen::Matrix<double, -1, -1>> &cutpoint_mats = prep.C_4;
      const std::vector<Eigen::Matrix<double, -1, -1>> &L_Omega_mats = prep.L_Omega_double;
      const std::vector<Eigen::Matrix<double, -1, -1>> &L_Omega_recip_mats = prep.L_Omega_recip_double;

      const bool do_us = (grad_option == "us_only") || (grad_option == "all");
      const bool do_coeff = (grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only");
      const bool do_corr = (grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only");
      const bool do_cut = (grad_option == "main_only") || (grad_option == "all") || (grad_option == "cutpoints_only");
      const bool do_prev = (grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only");

      std::vector<Eigen::Matrix<double, -1, -1>> Z_std_norm = vec_of_mats<double>(
            chunk_size,
            n_tests,
            n_class
      );
      std::vector<Eigen::Matrix<double, -1, -1>> Bound_Z = Z_std_norm;
      std::vector<Eigen::Matrix<double, -1, -1>> Upper_Bound_Z = Z_std_norm;
      std::vector<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z = Z_std_norm;
      std::vector<Eigen::Matrix<double, -1, -1>> prob = Z_std_norm;
      std::vector<Eigen::Matrix<double, -1, -1>> Phi_Z = Z_std_norm;
      std::vector<Eigen::Matrix<double, -1, -1>> y1_log_prob = Z_std_norm;
      std::vector<Eigen::Matrix<double, -1, -1>> log_Z_std_norm = Z_std_norm;
      std::vector<Eigen::Matrix<double, -1, -1>> log_phi_Bound_Z = Z_std_norm;
      std::vector<Eigen::Matrix<double, -1, -1>> log_phi_Z_recip = Z_std_norm;

      Eigen::Matrix<double, -1, -1> phi_Z_recip = Eigen::Matrix<double, -1, -1>::Zero(
            chunk_size,
            n_tests
      );
      Eigen::Matrix<double, -1, -1> phi_Bound_Z = phi_Z_recip;
      Eigen::Matrix<double, -1, -1> phi_Upper_Bound_Z = phi_Z_recip;
      Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk = phi_Z_recip;
      Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk_block = phi_Z_recip;
      Eigen::Matrix<double, -1, -1> common_grad_term_1 = phi_Z_recip;
      Eigen::Matrix<double, -1, -1> prob_rowwise_prod_temp = phi_Z_recip;
      Eigen::Matrix<double, -1, -1> prob_recip_rowwise_prod_temp = phi_Z_recip;
      Eigen::Matrix<double, -1, -1> prob_recip = phi_Z_recip;
      Eigen::Matrix<double, -1, -1> grad_prob = phi_Z_recip;
      Eigen::Matrix<double, -1, -1> z_grad_term = phi_Z_recip;
      Eigen::Matrix<double, -1, -1> dphi_over_L = phi_Z_recip;
      Eigen::Matrix<double, -1, -1> dZ_dmu_neg = phi_Z_recip;
      Eigen::Matrix<double, -1, -1> dphi_times_bz = phi_Z_recip;
      Eigen::Matrix<double, -1, -1> dZ_times_bz = phi_Z_recip;
      Eigen::Matrix<double, -1, 1> prod_container_or_inc_array = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
      Eigen::Matrix<double, -1, 1> derivs_chain_container_vec = prod_container_or_inc_array;
      Eigen::Matrix<double, -1, 1> prob_rowwise_prod_temp_all = prod_container_or_inc_array;
      Eigen::Matrix<double, -1, 1> dphi_direct_Cj = prod_container_or_inc_array;
      Eigen::Matrix<double, -1, 1> dZ_direct_Cj = prod_container_or_inc_array;
      Eigen::Matrix<double, -1, 1> prob_n = prod_container_or_inc_array;
      Eigen::Matrix<double, -1, 1> prob_n_recip = prod_container_or_inc_array;
      Eigen::Matrix<double, -1, 1> log_prob_n_recip = prod_container_or_inc_array;
      Eigen::Matrix<double, -1, 1> log_sum_result = prod_container_or_inc_array;
      Eigen::Matrix<double, -1, 1> container_max_logs = prod_container_or_inc_array;
      Eigen::Matrix<double, -1, 1> rowwise_prod = prod_container_or_inc_array;
      Eigen::Matrix<double, -1, 1> rowwise_sum = prod_container_or_inc_array;
      Eigen::Matrix<double, -1, 1> log_lik_chunk = prod_container_or_inc_array;
      Eigen::Matrix<double, -1, 1> prev_per_obs_given_c = prod_container_or_inc_array;
      Eigen::Matrix<double, -1, 1> log_prev_per_obs_given_c = prod_container_or_inc_array;
      Eigen::Matrix<double, -1, -1> lp_array = Eigen::Matrix<double, -1, -1>::Zero(
            chunk_size,
            n_class
      );

      Eigen::Matrix<double, -1, -1> y_chunk = y_ref.middleRows(
            row_start,
            chunk_size
      ).cast<double>();
      Eigen::Matrix<double, -1, -1> y_upper_chunk;
      if (prep.has_y_upper) {
            y_upper_chunk = prep.y_upper_mat->middleRows(
                  row_start,
                  chunk_size
            ).cast<double>();
      }

      Eigen::Matrix<double, -1, 1> u_unc_vec_chunk = theta_us_vec_ref.segment(
            row_start * n_tests,
            chunk_size * n_tests
      );
      Eigen::Matrix<double, -1, 1> u_vec_chunk(chunk_size * n_tests);
      Eigen::Matrix<double, -1, 1> du_wrt_duu_chunk(chunk_size * n_tests);
      Eigen::Matrix<double, -1, 1> d_J_wrt_duu_chunk(chunk_size * n_tests);
      fn_MVP_compute_nuisance_T<vec>(
            u_vec_chunk,
            u_unc_vec_chunk,
            kernel_choice
      );
      log_jac_u += fn_MVP_compute_nuisance_log_jac_u_T<vec>(
            u_vec_chunk,
            u_unc_vec_chunk,
            d_J_wrt_duu_chunk,
            kernel_choice
      );
      Eigen::Matrix<double, -1, -1> u_array = u_vec_chunk.reshaped(
            chunk_size,
            n_tests
      );

      const bool chunk_has_missing_results = fn_MVP_rows_of_y_have_a_missing_result(
            y_ref,
            row_start,
            chunk_size
      );
      Eigen::Matrix<double, -1, -1> is_missing_result;
      Eigen::Matrix<double, -1, -1> Z_untruncated;
      if (chunk_has_missing_results) {
            fn_MVP_missing_result_indicator_and_recode_missing_results_to_y_equals_1(
                  y_chunk,
                  is_missing_result
            );
            if (prep.has_y_upper) {
                  for (int test_index = 0; test_index < n_tests; test_index++) {
                        const int ord_index = ord_idx_of_test(test_index);
                        const double fill_hi = (ord_index >= 0)
                                               ? static_cast<double>(n_cat_per_ord_test(ord_index))
                                               : 1.0;
                        y_upper_chunk.col(test_index).array() =
                              (is_missing_result.col(test_index).array() > 0.5).select(
                                    fill_hi,
                                    y_upper_chunk.col(test_index).array()
                              );
                  }
            }
            fn_MVP_compute_untruncated_GHK_Z_of_missing_results_T<vec>(
                  Z_untruncated,
                  u_array,
                  is_missing_result,
                  kernel_choice
            );
      }
      Eigen::Matrix<double, -1, -1> y_sign = (2.0 * y_chunk.array() - 1.0).matrix();
      Eigen::Matrix<double, -1, -1> y_m_y_sign_x_u = (y_chunk.array() - y_sign.array() * u_array.array()).matrix();

      std::vector<Eigen::Matrix<double, -1, -1>> Xbeta_chunk(
            n_class,
            Eigen::Matrix<double, -1, -1>::Zero(
                  chunk_size,
                  n_tests
            )
      );
      for (int class_index = 0; class_index < n_class; class_index++) {
            for (int test_index = 0; test_index < n_tests; test_index++) {
                  if ((*prep.is_perfect_ref)(test_index) == 1) {
                        Xbeta_chunk[class_index].col(test_index).setConstant(
                              prep.beta_4[class_index](0, test_index)
                        );
                  } else {
                        const int n_covs_ct = (*prep.n_covs_per_outcome)(class_index, test_index);
                        if (n_covs_ct > 0) {
                              Xbeta_chunk[class_index].col(test_index).noalias() =
                                    (*prep.X_class_test)[prep.class_index_of_X_to_read(class_index)][prep.test_index_of_X_to_read(test_index)].block(
                                          row_start,
                                          0,
                                          chunk_size,
                                          n_covs_ct
                                    )
                                    * prep.beta_4[class_index].col(test_index).head(n_covs_ct);
                        }
                  }
            }
      }

      std::vector<std::vector<int>> row_is_problem(
            n_class,
            std::vector<int>(
                  chunk_size,
                  0
            )
      );
      Eigen::Matrix<double, -1, 1> mu_vec(chunk_size);
      for (int class_index = 0; class_index < n_class; class_index++) {
            log_prev_per_obs_given_c = prep.log_prev_subject.col(class_index).segment(
                  row_start,
                  chunk_size
            );
            prev_per_obs_given_c = prep.prev_subject.col(class_index).segment(
                  row_start,
                  chunk_size
            );
            prod_container_or_inc_array.setZero();
            for (int test_index = 0; test_index < n_tests; test_index++) {
                  const int ord_index = ord_idx_of_test(test_index);
                  const double L_tt_inv = L_Omega_recip_mats[class_index](test_index, test_index);
                  mu_vec.array() = Xbeta_chunk[class_index].col(test_index).array()
                                   + prod_container_or_inc_array.array();
                  Eigen::Matrix<int, -1, 1> bin_ovf = Eigen::Matrix<int, -1, 1>::Zero(chunk_size);
                  Eigen::Matrix<int, -1, 1> bin_unf = Eigen::Matrix<int, -1, 1>::Zero(chunk_size);
                  std::vector<int> ord_problem_index;
                  if (ord_index < 0) {
                        Bound_Z[class_index].col(test_index).array() = -L_tt_inv * mu_vec.array();
                        bin_ovf = ((Bound_Z[class_index].col(test_index).array() > prep.overflow_threshold)
                                   && (y_chunk.col(test_index).array() == 1.0)).cast<int>();
                        bin_unf = ((Bound_Z[class_index].col(test_index).array() < prep.underflow_threshold)
                                   && (y_chunk.col(test_index).array() == 0.0)).cast<int>();
                        if (chunk_has_missing_results) {
                              fn_MVP_remove_missing_results_from_tail_mask(
                                    test_index,
                                    is_missing_result,
                                    bin_ovf
                              );
                              fn_MVP_remove_missing_results_from_tail_mask(
                                    test_index,
                                    is_missing_result,
                                    bin_unf
                              );
                        }
                        for (int row_index = 0; row_index < chunk_size; row_index++) {
                              if (bin_ovf(row_index) || bin_unf(row_index)) {
                                    row_is_problem[class_index][row_index] = 1;
                              }
                        }
                  } else {
                        const int n_cat_test = n_cat_per_ord_test(ord_index);
                        for (int row_index = 0; row_index < chunk_size; row_index++) {
                              const int y_lo = static_cast<int>(y_chunk(row_index, test_index));
                              const int y_hi = prep.has_y_upper
                                               ? static_cast<int>(y_upper_chunk(row_index, test_index))
                                               : y_lo;
                              Bound_Z[class_index](row_index, test_index) =
                                    (y_lo == 1)
                                    ? -pos_inf
                                    : (cutpoint_mats[class_index](y_lo - 2, ord_index) - mu_vec(row_index)) * L_tt_inv;
                              Upper_Bound_Z[class_index](row_index, test_index) =
                                    (y_hi == n_cat_test)
                                    ? pos_inf
                                    : (cutpoint_mats[class_index](y_hi - 1, ord_index) - mu_vec(row_index)) * L_tt_inv;
                        }
                        if (chunk_has_missing_results) {
                              fn_MVOP_set_bounds_of_missing_results_to_minus_and_plus_infinity(
                                    test_index,
                                    is_missing_result,
                                    Bound_Z[class_index],
                                    Upper_Bound_Z[class_index]
                              );
                        }
                        for (int row_index = 0; row_index < chunk_size; row_index++) {
                              if (chunk_has_missing_results && (is_missing_result(row_index, test_index) > 0.5)) {
                                    continue;
                              }
                              const double lower_bound = Bound_Z[class_index](row_index, test_index);
                              const double upper_bound = Upper_Bound_Z[class_index](row_index, test_index);
                              if ((upper_bound < prep.underflow_threshold)
                                  || (lower_bound > prep.overflow_threshold)) {
                                    ord_problem_index.push_back(row_index);
                                    row_is_problem[class_index][row_index] = 1;
                              }
                        }
                  }
                  fn_MVOP_compute_lp_GHK_cols_T<vec>(
                        test_index,
                        Bound_U_Phi_Bound_Z[class_index],
                        Phi_Z[class_index],
                        Z_std_norm[class_index],
                        prob[class_index],
                        y1_log_prob[class_index],
                        Bound_Z[class_index],
                        y_chunk,
                        u_array,
                        Upper_Bound_Z[class_index],
                        (ord_index < 0),
                        kernel_choice
                  );
                  if (chunk_has_missing_results) {
                        fn_MVP_set_GHK_col_of_missing_results_to_the_untruncated_draw(
                              test_index,
                              is_missing_result,
                              u_array,
                              Z_untruncated,
                              Bound_U_Phi_Bound_Z[class_index],
                              Phi_Z[class_index],
                              Z_std_norm[class_index],
                              prob[class_index],
                              y1_log_prob[class_index]
                        );
                  }
                  if (ord_index < 0) {
                        std::vector<int> idx_ovf;
                        std::vector<int> idx_unf;
                        for (int row_index = 0; row_index < chunk_size; row_index++) {
                              if (bin_ovf(row_index)) {
                                    idx_ovf.push_back(row_index);
                              }
                              if (bin_unf(row_index)) {
                                    idx_unf.push_back(row_index);
                              }
                        }
                        if (!idx_ovf.empty()) {
                              fn_MVP_compute_lp_GHK_cols_log_scale_overflow_T<vec>(
                                    test_index,
                                    static_cast<int>(idx_ovf.size()),
                                    idx_ovf,
                                    Bound_U_Phi_Bound_Z[class_index],
                                    Phi_Z[class_index],
                                    Z_std_norm[class_index],
                                    log_Z_std_norm[class_index],
                                    prob[class_index],
                                    y1_log_prob[class_index],
                                    log_phi_Bound_Z[class_index],
                                    log_phi_Z_recip[class_index],
                                    Bound_Z[class_index],
                                    u_array,
                                    kernel_choice
                              );
                        }
                        if (!idx_unf.empty()) {
                              fn_MVP_compute_lp_GHK_cols_log_scale_underflow_T<vec>(
                                    test_index,
                                    idx_unf,
                                    Bound_U_Phi_Bound_Z[class_index],
                                    Phi_Z[class_index],
                                    Z_std_norm[class_index],
                                    log_Z_std_norm[class_index],
                                    prob[class_index],
                                    y1_log_prob[class_index],
                                    log_phi_Bound_Z[class_index],
                                    log_phi_Z_recip[class_index],
                                    Bound_Z[class_index],
                                    u_array,
                                    kernel_choice
                              );
                        }
                  } else if (!ord_problem_index.empty()) {
                        fn_MVOP_compute_lp_GHK_cols_log_scale_ordinal(
                              test_index,
                              ord_problem_index,
                              Bound_U_Phi_Bound_Z[class_index],
                              Phi_Z[class_index],
                              Z_std_norm[class_index],
                              prob[class_index],
                              y1_log_prob[class_index],
                              Bound_Z[class_index],
                              Upper_Bound_Z[class_index],
                              u_array,
                              use_scalar,
                              kernel_choice
                        );
                  }
                  if (test_index < n_tests - 1) {
                        prod_container_or_inc_array =
                              Z_std_norm[class_index].leftCols(test_index + 1)
                              * L_Omega_mats[class_index].row(test_index + 1).head(test_index + 1).transpose();
                  }
            }
            rowwise_sum = y1_log_prob[class_index].rowwise().sum();
            rowwise_sum.array() += log_prev_per_obs_given_c.array();
            lp_array.col(class_index) = rowwise_sum;
      }

      const int index_start = 1 + n_params + row_start;
      log_sum_exp_general_T<vec>(
            lp_array,
            log_sum_result,
            container_max_logs
      );
      out_mat.segment(
            index_start,
            chunk_size
      ) = log_sum_result;
      log_lik_chunk = log_sum_result;
      prob_n = log_lik_chunk;
      apply_inplace<vec, Fn::exp>(prob_n);
      prob_n_recip = stan::math::inv(prob_n);
      log_prob_n_recip = -log_lik_chunk;
      for (int row_index = 0; row_index < chunk_size; row_index++) {
            if (log_lik_chunk(row_index) < -690.0) {
                  prob_n_recip(row_index) = 0.0;
                  for (int other_class = 0; other_class < n_class; other_class++) {
                        row_is_problem[other_class][row_index] = 1;
                  }
            }
      }

      for (int class_index = 0; class_index < n_class; class_index++) {
            prev_per_obs_given_c = prep.prev_subject.col(class_index).segment(
                  row_start,
                  chunk_size
            );
            log_prev_per_obs_given_c = prep.log_prev_subject.col(class_index).segment(
                  row_start,
                  chunk_size
            );
            prob_recip = stan::math::inv(prob[class_index]);
            for (int test_index = 0; test_index < n_tests; test_index++) {
                  fn_MVP_compute_phi_Z_recip_cols_T<vec>(
                        test_index,
                        phi_Z_recip,
                        Phi_Z[class_index],
                        Z_std_norm[class_index],
                        kernel_choice
                  );
                  if (ord_idx_of_test(test_index) < 0) {
                        fn_MVP_compute_phi_Bound_Z_cols_T<vec>(
                              test_index,
                              phi_Bound_Z,
                              Bound_U_Phi_Bound_Z[class_index],
                              Bound_Z[class_index],
                              kernel_choice
                        );
                        if (chunk_has_missing_results) {
                              fn_MVP_set_phi_Bound_Z_col_of_missing_results_to_zero(
                                    test_index,
                                    is_missing_result,
                                    phi_Bound_Z
                              );
                        }
                  } else {
                        if (kernel_choice.Phi_approx) {
                              phi_Bound_Z.col(test_index) = Bound_Z[class_index].col(test_index);
                              Phi_col_inplace<vec>(
                                    phi_Bound_Z,
                                    test_index,
                                    kernel_choice
                              );
                              phi_Upper_Bound_Z.col(test_index) = Upper_Bound_Z[class_index].col(test_index);
                              Phi_col_inplace<vec>(
                                    phi_Upper_Bound_Z,
                                    test_index,
                                    kernel_choice
                              );
                              Eigen::Matrix<double, -1, -1> neg_bounds(
                                    Bound_Z[class_index].rows(),
                                    2
                              );
                              neg_bounds.col(0) = -Bound_Z[class_index].col(test_index);
                              neg_bounds.col(1) = -Upper_Bound_Z[class_index].col(test_index);
                              Phi_col_inplace<vec>(
                                    neg_bounds,
                                    0,
                                    kernel_choice
                              );
                              Phi_col_inplace<vec>(
                                    neg_bounds,
                                    1,
                                    kernel_choice
                              );
                              phi_Bound_Z.col(test_index).array() =
                                    (3.0 * 0.07056 * Bound_Z[class_index].col(test_index).array().square() + 1.5976)
                                    * phi_Bound_Z.col(test_index).array()
                                    * neg_bounds.col(0).array();
                              phi_Upper_Bound_Z.col(test_index).array() =
                                    (3.0 * 0.07056 * Upper_Bound_Z[class_index].col(test_index).array().square() + 1.5976)
                                    * phi_Upper_Bound_Z.col(test_index).array()
                                    * neg_bounds.col(1).array();
                        } else {
                              phi_Bound_Z.col(test_index).array() =
                                    -0.5 * Bound_Z[class_index].col(test_index).array().square();
                              apply_col_inplace<vec, Fn::exp>(
                                    phi_Bound_Z,
                                    test_index
                              );
                              phi_Bound_Z.col(test_index).array() *= sqrt_2_pi_recip;
                              phi_Upper_Bound_Z.col(test_index).array() =
                                    -0.5 * Upper_Bound_Z[class_index].col(test_index).array().square();
                              apply_col_inplace<vec, Fn::exp>(
                                    phi_Upper_Bound_Z,
                                    test_index
                              );
                              phi_Upper_Bound_Z.col(test_index).array() *= sqrt_2_pi_recip;
                        }
                        phi_Bound_Z.col(test_index).array() =
                              Bound_Z[class_index].col(test_index).array().isInf().select(
                                    0.0,
                                    phi_Bound_Z.col(test_index).array()
                              );
                        phi_Upper_Bound_Z.col(test_index).array() =
                              Upper_Bound_Z[class_index].col(test_index).array().isInf().select(
                                    0.0,
                                    phi_Upper_Bound_Z.col(test_index).array()
                              );
                  }
            }
            std::vector<int> problem_rows_class;
            if (grad_option != "none") {
                  problem_rows_class.reserve(chunk_size);
                  for (int row_index = 0; row_index < chunk_size; row_index++) {
                        if (row_is_problem[class_index][row_index] == 1) {
                              problem_rows_class.push_back(row_index);
                        }
                  }
                  fn_MVOP_grad_prep(
                        prob[class_index],
                        y_sign,
                        y_m_y_sign_x_u,
                        u_array,
                        L_Omega_recip_mats[class_index],
                        prev_per_obs_given_c,
                        prob_n_recip,
                        phi_Z_recip,
                        phi_Bound_Z,
                        phi_Upper_Bound_Z,
                        Bound_Z[class_index],
                        Upper_Bound_Z[class_index],
                        prob_recip,
                        prob_rowwise_prod_temp,
                        prob_recip_rowwise_prod_temp,
                        prob_rowwise_prod_temp_all,
                        common_grad_term_1,
                        dphi_over_L,
                        dZ_dmu_neg,
                        dphi_times_bz,
                        dZ_times_bz,
                        ord_idx_of_test,
                        Model_args_as_cpp_struct
                  );
                  for (std::size_t problem_pos = 0; problem_pos < problem_rows_class.size(); problem_pos++) {
                        const int row_index = problem_rows_class[problem_pos];
                        common_grad_term_1.row(row_index).setZero();
                        prob_recip.row(row_index).setZero();
                        prob_rowwise_prod_temp.row(row_index).setZero();
                        prob_recip_rowwise_prod_temp.row(row_index).setZero();
                        prob_rowwise_prod_temp_all(row_index) = 0.0;
                        dphi_over_L.row(row_index).setZero();
                        dZ_dmu_neg.row(row_index).setZero();
                        dphi_times_bz.row(row_index).setZero();
                        dZ_times_bz.row(row_index).setZero();
                        phi_Z_recip.row(row_index).setZero();
                        prob[class_index].row(row_index).setZero();
                  }
            }
            if (do_us) {
                  u_grad_array_CM_chunk_block = u_grad_array_CM_chunk.block(
                        0,
                        0,
                        chunk_size,
                        n_tests
                  );
                  fn_MVP_compute_nuisance_grad_v2(
                        u_grad_array_CM_chunk_block,
                        phi_Z_recip,
                        common_grad_term_1,
                        L_Omega_mats[class_index],
                        prob[class_index],
                        prob_recip,
                        prob_rowwise_prod_temp,
                        dphi_over_L,
                        dZ_dmu_neg,
                        z_grad_term,
                        grad_prob,
                        prod_container_or_inc_array,
                        derivs_chain_container_vec,
                        Model_args_as_cpp_struct
                  );
                  u_grad_array_CM_chunk.block(
                        0,
                        0,
                        chunk_size,
                        n_tests
                  ).array() += u_grad_array_CM_chunk_block.array();
            }
            const Eigen::Matrix<int, -1, 1> n_cov_vec_c = prep.n_covs_per_outcome->row(class_index).transpose();
            if (do_coeff) {
                  fn_MVP_compute_coefficients_grad_v3(
                        class_index,
                        beta_4_grad_lik[class_index],
                        (*prep.X_class_test)[prep.class_index_of_X_to_read(class_index)],
                        n_cov_vec_c,
                        row_start,
                        prep.n_covariates_max,
                        common_grad_term_1,
                        L_Omega_mats[class_index],
                        prob[class_index],
                        prob_recip,
                        prob_rowwise_prod_temp,
                        dphi_over_L,
                        dZ_dmu_neg,
                        z_grad_term,
                        grad_prob,
                        prod_container_or_inc_array,
                        derivs_chain_container_vec,
                        true,
                        Model_args_as_cpp_struct
                  );
            }
            if (do_corr) {
                  fn_MOVP_compute_L_Omega_grad_v3(
                        U_Omega_grad_array[class_index],
                        common_grad_term_1,
                        L_Omega_mats[class_index],
                        prob[class_index],
                        prob_recip,
                        Z_std_norm[class_index],
                        prob_rowwise_prod_temp,
                        dphi_over_L,
                        dZ_dmu_neg,
                        dphi_times_bz,
                        dZ_times_bz,
                        z_grad_term,
                        grad_prob,
                        prod_container_or_inc_array,
                        derivs_chain_container_vec,
                        true,
                        Model_args_as_cpp_struct
                  );
            }
            if (do_cut) {
                  for (int test_index = 0; test_index < n_tests; test_index++) {
                        const int ord_index = ord_idx_of_test(test_index);
                        if (ord_index < 0) {
                              continue;
                        }
                        fn_MVOP_compute_cutpoint_grad_from_one_unit_dZ_seed_propagation_and_category_scatter(
                              class_index,
                              cutpoint_grad_4[class_index],
                              test_index,
                              ord_index,
                              n_thr_per_ord_test(ord_index) + 1,
                              common_grad_term_1,
                              L_Omega_mats[class_index],
                              prob[class_index],
                              prob_recip,
                              prob_rowwise_prod_temp,
                              phi_Bound_Z,
                              phi_Upper_Bound_Z,
                              phi_Z_recip,
                              u_array,
                              y_chunk,
                              dphi_over_L,
                              dZ_dmu_neg,
                              z_grad_term,
                              grad_prob,
                              prod_container_or_inc_array,
                              derivs_chain_container_vec,
                              dphi_direct_Cj,
                              dZ_direct_Cj,
                              true,
                              Model_args_as_cpp_struct
                        );
                  }
            }
            if ((grad_option != "none")
                && !problem_rows_class.empty()
                && (do_us || do_coeff || do_corr || do_cut)) {
                  for (std::size_t problem_pos = 0; problem_pos < problem_rows_class.size(); problem_pos++) {
                        const int row_index = problem_rows_class[problem_pos];
                        fn_MVOP_row_grads_log_scale(
                              row_index,
                              row_start + row_index,
                              class_index,
                              do_us,
                              do_coeff,
                              do_corr,
                              do_cut,
                              n_tests,
                              ord_idx_of_test,
                              n_class,
                              prep.n_covariates_max,
                              n_cov_vec_c,
                              n_cat_per_ord_test,
                              (*prep.X_class_test)[prep.class_index_of_X_to_read(class_index)],
                              y_chunk,
                              u_array,
                              Bound_Z[class_index],
                              Upper_Bound_Z[class_index],
                              Z_std_norm[class_index],
                              y1_log_prob[class_index],
                              L_Omega_mats[class_index],
                              log_prev_per_obs_given_c(row_index),
                              log_prob_n_recip(row_index),
                              u_grad_array_CM_chunk,
                              beta_4_grad_lik[class_index],
                              U_Omega_grad_array[class_index],
                              cutpoint_grad_4[class_index],
                              use_scalar,
                              kernel_choice.Phi_approx,
                              kernel_choice.inv_Phi_approx,
                              chunk_has_missing_results ? &is_missing_result : nullptr
                        );
                  }
            }
            if (do_coeff) {
                  for (int test_index = 0; test_index < n_tests; test_index++) {
                        if ((*prep.is_perfect_ref)(test_index) == 1) {
                              beta_4_grad_lik[class_index].col(test_index).setZero();
                        }
                  }
            }
            if (do_prev) {
                  rowwise_prod = prob[class_index].rowwise().prod();
                  d_lp_d_log_prev_subject.col(class_index).segment(
                        row_start,
                        chunk_size
                  ).array() = prev_per_obs_given_c.array()
                              * rowwise_prod.array()
                              * prob_n_recip.array();
                  for (std::size_t problem_pos = 0; problem_pos < problem_rows_class.size(); problem_pos++) {
                        const int row_index = problem_rows_class[problem_pos];
                        d_lp_d_log_prev_subject(row_start + row_index, class_index) = std::exp(
                              log_prev_per_obs_given_c(row_index)
                              + y1_log_prob[class_index].row(row_index).sum()
                              + log_prob_n_recip(row_index)
                        );
                  }
            }
            if (do_us && (class_index == n_class - 1)) {
                  const int start_index = 1 + row_start * n_tests;
                  const int block_length = chunk_size * n_tests;
                  out_mat.segment(
                        start_index,
                        block_length
                  ) = u_grad_array_CM_chunk.reshaped();
                  fn_MVP_nuisance_first_deriv_T<vec>(
                        du_wrt_duu_chunk,
                        u_vec_chunk,
                        u_unc_vec_chunk,
                        kernel_choice
                  );
                  fn_MVP_nuisance_deriv_of_log_det_J_T<vec>(
                        d_J_wrt_duu_chunk,
                        u_vec_chunk,
                        u_unc_vec_chunk,
                        du_wrt_duu_chunk,
                        kernel_choice
                  );
                  out_mat.segment(
                        start_index,
                        block_length
                  ).array() *= du_wrt_duu_chunk.array();
                  out_mat.segment(
                        start_index,
                        block_length
                  ).array() += d_J_wrt_duu_chunk.array();
            }
      }
}


template <Vec vec>
inline void fn_lp_grad_CLC_MVOP_PartialLog_Inplace_process_serial_T(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const std::string &grad_option,
      const Model_fn_args_struct &Model_args_as_cpp_struct,
      const int n_chunks_multiplier
) {
      CLC_MVOP_prep prep = fn_CLC_MVOP_prep(
            out_mat,
            theta_main_vec_ref,
            theta_us_vec_ref,
            y_ref,
            Model_args_as_cpp_struct,
            n_chunks_multiplier
      );
      if (prep.label_constraint_violated) {
            out_mat.setZero();
            out_mat(0) = -std::numeric_limits<double>::infinity();
            return;
      }
      const int n_obs = prep.n_obs;
      const int n_tests = prep.n_tests;
      const int n_class = prep.n_class;
      const int K_max = std::max(1, prep.n_covariates_max);

      std::vector<Eigen::Matrix<double, -1, -1>> beta_4_grad_lik(
            n_class,
            Eigen::Matrix<double, -1, -1>::Zero(
                  K_max,
                  n_tests
            )
      );
      std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array = vec_of_mats<double>(
            n_tests,
            n_tests,
            n_class
      );
      std::vector<Eigen::Matrix<double, -1, -1>> cutpoint_grad_4 = vec_of_mats<double>(
            std::max(1, prep.n_cutpoints_max),
            std::max(1, prep.n_ordinal_tests),
            n_class
      );
      Eigen::Matrix<double, -1, -1> d_lp_d_log_prev_subject = Eigen::Matrix<double, -1, -1>::Zero(
            n_obs,
            n_class
      );
      double log_jac_u = 0.0;

      if (prep.prior_only) {
            fn_CLC_MVOP_prior_only_nuisance_anchor_T<vec>(
                  out_mat,
                  theta_us_vec_ref,
                  grad_option,
                  prep,
                  log_jac_u
            );
      } else {
            for (int chunk_index = 0; chunk_index < prep.n_total_chunks; chunk_index++) {
                  const int chunk_size_cur = (chunk_index == prep.n_full_chunks)
                                             ? prep.last_chunk_size
                                             : prep.normal_chunk_size;
                  fn_lp_grad_CLC_MVOP_PartialLog_process_chunk_T<vec>(
                        out_mat,
                        theta_us_vec_ref,
                        y_ref,
                        grad_option,
                        chunk_index,
                        chunk_size_cur,
                        prep,
                        log_jac_u,
                        beta_4_grad_lik,
                        U_Omega_grad_array,
                        cutpoint_grad_4,
                        d_lp_d_log_prev_subject,
                        Model_args_as_cpp_struct
                  );
            }
      }

      fn_CLC_MVOP_assemble(
            out_mat,
            theta_main_vec_ref,
            prep,
            Model_args_as_cpp_struct,
            log_jac_u,
            beta_4_grad_lik,
            U_Omega_grad_array,
            cutpoint_grad_4,
            d_lp_d_log_prev_subject
      );
}
#pragma GCC pop_options


//// -------------------------------------------------------------------------------------
//// 4. Public dispatchers (NoLog, PartialLog, multi-attempts)
//// -------------------------------------------------------------------------------------
void fn_lp_grad_CLC_MVOP_NoLog_MD_and_AD_Inplace_process(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const std::string &grad_option,
      const Model_fn_args_struct &Model_args_as_cpp_struct,
      std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
      const int n_threads_WCP
) {
      if (Model_args_as_cpp_struct.Model_args_bools(1) || (theta_us_vec_ref.rows() <= 1)) {
            DISPATCH_VEC(vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0)),
                  fn_lp_grad_CLC_MVOP_CI_Inplace_process,
                  out_mat,
                  theta_main_vec_ref,
                  theta_us_vec_ref,
                  y_ref,
                  grad_option,
                  Model_args_as_cpp_struct
            );
            return;
      }
      const Vec vec = vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0));
      DISPATCH_VEC(
            vec,
            fn_lp_grad_CLC_MVOP_NoLog_Inplace_process_T,
            out_mat,
            theta_main_vec_ref,
            theta_us_vec_ref,
            y_ref,
            grad_option,
            Model_args_as_cpp_struct,
            LC_MVP_ws_structs,
            n_threads_WCP
      );
}


void fn_lp_grad_CLC_MVOP_PartialLog_MD_and_AD_Inplace_process(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const std::string &grad_option,
      const Model_fn_args_struct &Model_args_as_cpp_struct,
      std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
      const int n_threads_WCP,
      const int n_chunks_multiplier
) {
      (void)LC_MVP_ws_structs;
      (void)n_threads_WCP;
      if (Model_args_as_cpp_struct.Model_args_bools(1) || (theta_us_vec_ref.rows() <= 1)) {
            DISPATCH_VEC(vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0)),
                  fn_lp_grad_CLC_MVOP_CI_Inplace_process,
                  out_mat,
                  theta_main_vec_ref,
                  theta_us_vec_ref,
                  y_ref,
                  grad_option,
                  Model_args_as_cpp_struct
            );
            return;
      }
      const int n_chunks_multiplier_used = (n_chunks_multiplier > 0)
                                           ? n_chunks_multiplier
                                           : Model_args_as_cpp_struct.PartialLog_n_chunks_multiplier;
      const Vec vec = vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0));
      fn_PartialLog_evaluation_in_its_own_chunk_layout(
            out_mat,
            theta_us_vec_ref,
            static_cast<int>(y_ref.rows()),
            static_cast<int>(y_ref.cols()),
            Model_args_as_cpp_struct.Model_args_ints(3),
            fn_vec_size_of_vect_type(Model_args_as_cpp_struct.Model_args_strings(0)),
            n_chunks_multiplier_used,
            [&](
                  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_eval,
                  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> us_eval
            ) {
                  DISPATCH_VEC(
                        vec,
                        fn_lp_grad_CLC_MVOP_PartialLog_Inplace_process_serial_T,
                        out_eval,
                        theta_main_vec_ref,
                        us_eval,
                        y_ref,
                        grad_option,
                        Model_args_as_cpp_struct,
                        n_chunks_multiplier_used
                  );
            }
      );
}


void fn_lp_grad_CLC_MVOP_multi_attempts_InPlace_process(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const std::string &grad_option,
      const Model_fn_args_struct &Model_args_as_cpp_struct,
      std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
      const int n_threads_WCP,
      const bool force_autodiff,
      const bool force_PartialLog
) {
      int NaN_or_Inf_indicator = 0;
      Eigen::Matrix<double, -1, 1> out_mat_orig = out_mat;

      fn_check_settings_strings_before_multi_attempts(
            Model_args_as_cpp_struct,
            false,
            "fn_lp_grad_CLC_MVOP_multi_attempts_InPlace_process"
      );

      const bool is_CI_or_zero_nuisance =
            Model_args_as_cpp_struct.Model_args_bools(1) || (theta_us_vec_ref.rows() <= 1);

      //// 1st attempt: NoLog manual-gradient SIMD
      if ((force_autodiff == false) && (force_PartialLog == false)) {
            NaN_or_Inf_indicator = 0;
            try {
                  if (is_CI_or_zero_nuisance) {
                        DISPATCH_VEC(vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0)),
                              fn_lp_grad_CLC_MVOP_CI_Inplace_process,
                              out_mat,
                              theta_main_vec_ref,
                              theta_us_vec_ref,
                              y_ref,
                              grad_option,
                              Model_args_as_cpp_struct
                        );
                  } else {
                        fn_lp_grad_CLC_MVOP_NoLog_MD_and_AD_Inplace_process(
                              out_mat,
                              theta_main_vec_ref,
                              theta_us_vec_ref,
                              y_ref,
                              grad_option,
                              Model_args_as_cpp_struct,
                              LC_MVP_ws_structs,
                              n_threads_WCP
                        );
                  }
                  if (out_mat(0) == -std::numeric_limits<double>::infinity()) {
                        return;
                  }
                  if (is_NaN_or_Inf_Eigen(out_mat)) {
                        NaN_or_Inf_indicator = 1;
                  }
            } catch (const std::exception &) {
                  NaN_or_Inf_indicator = 1;
            }
      }

      //// 2nd attempt: log-scale manual-gradient PartialLog
      if (((NaN_or_Inf_indicator == 1) && (is_CI_or_zero_nuisance == false))
          || ((force_autodiff == false) && (force_PartialLog == true))) {
            NaN_or_Inf_indicator = 0;
            out_mat = out_mat_orig;
            try {
                  if (is_CI_or_zero_nuisance) {
                        DISPATCH_VEC(vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0)),
                              fn_lp_grad_CLC_MVOP_CI_Inplace_process,
                              out_mat,
                              theta_main_vec_ref,
                              theta_us_vec_ref,
                              y_ref,
                              grad_option,
                              Model_args_as_cpp_struct
                        );
                  } else {
                        const int n_chunks_multiplier = Model_args_as_cpp_struct.PartialLog_n_chunks_multiplier;
                        fn_lp_grad_CLC_MVOP_PartialLog_MD_and_AD_Inplace_process(
                              out_mat,
                              theta_main_vec_ref,
                              theta_us_vec_ref,
                              y_ref,
                              grad_option,
                              Model_args_as_cpp_struct,
                              LC_MVP_ws_structs,
                              n_threads_WCP,
                              n_chunks_multiplier
                        );
                  }
                  if (out_mat(0) == -std::numeric_limits<double>::infinity()) {
                        return;
                  }
                  if (is_NaN_or_Inf_Eigen(out_mat)) {
                        NaN_or_Inf_indicator = 1;
                  }
            } catch (const std::exception &) {
                  NaN_or_Inf_indicator = 1;
            }
      }

      //// 3rd attempt: whole-model full autodiff on the log scale
      if (force_autodiff == true) {
            out_mat = out_mat_orig;
            try {
                  fn_lp_and_grad_CLC_MVOP_Pinkney_AD_log_scale_InPlace_process(
                        out_mat,
                        theta_main_vec_ref,
                        theta_us_vec_ref,
                        y_ref,
                        grad_option,
                        Model_args_as_cpp_struct,
                        LC_MVP_ws_structs,
                        n_threads_WCP
                  );
                  if (out_mat(0) == -std::numeric_limits<double>::infinity()) {
                        return;
                  }
                  NaN_or_Inf_indicator = is_NaN_or_Inf_Eigen(out_mat) ? 1 : 0;
            } catch (const std::exception &) {
                  NaN_or_Inf_indicator = 1;
            }
      }

      if (NaN_or_Inf_indicator == 1) {
            out_mat = out_mat_orig;
            out_mat(0) = std::numeric_limits<double>::quiet_NaN();
      }
}
