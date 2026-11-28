//// -------------------------------------------------------------------------------------
//// CLC_MVOP_lp_grad_AD_fns.hpp
////
//// Whole-model log-scale Stan Math autodiff driver for Model_type = "CLC_MVOP" (alias
//// "CLC-MVOP"). Attempt 3 in fn_lp_grad_CLC_MVOP_multi_attempts_InPlace_process and
//// optional force_autodiff = TRUE path.
//// -------------------------------------------------------------------------------------

#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <stan/math.hpp>

#include "CLC_MVOP_lp_grad_common_T.hpp"


//// -------------------------------------------------------------------------------------
//// Helper: log Phi(x) in stan::math::var respecting Phi_type (exact or Phi_approx)
//// -------------------------------------------------------------------------------------
inline stan::math::var fn_CLC_MVOP_AD_log_Phi_var(
      const stan::math::var &arg_val,
      const bool use_Phi_approx
) {
      if (!use_Phi_approx) {
            return log_Phi_exact_var(arg_val);
      }
      const stan::math::var cubic_arg =
            0.07056 * stan::math::square(arg_val) * arg_val + 1.5976 * arg_val;
      return stan::math::log_inv_logit(cubic_arg);
}


//// -------------------------------------------------------------------------------------
//// Whole-model log-scale Stan Math autodiff log-density and gradient for CLC_MVOP
//// -------------------------------------------------------------------------------------
void fn_lp_and_grad_CLC_MVOP_Pinkney_AD_log_scale_InPlace_process(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const std::string &grad_option,
      const Model_fn_args_struct &Model_args_as_cpp_struct,
      std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
      const int n_threads_WCP
) {
      (void)LC_MVP_ws_structs;
      (void)n_threads_WCP;

      out_mat.setZero();

      const int n_obs = static_cast<int>(y_ref.rows());
      const int n_tests = static_cast<int>(y_ref.cols());
      const int n_us = static_cast<int>(theta_us_vec_ref.rows());
      const int n_params_main = static_cast<int>(theta_main_vec_ref.rows());
      const int n_params = n_params_main + n_us;

      //// Boolean flags
      const bool exclude_priors = Model_args_as_cpp_struct.Model_args_bools(0);
      const bool CI = Model_args_as_cpp_struct.Model_args_bools(1);
      const bool corr_force_positive = Model_args_as_cpp_struct.Model_args_bools(2);
      const bool prior_only = (Model_args_as_cpp_struct.Model_args_bools.rows() > 15)
                              ? Model_args_as_cpp_struct.Model_args_bools(15)
                              : false;
      const bool use_pop_Se_Sp_prior = (Model_args_as_cpp_struct.Model_args_bools.rows() > 16)
                                       ? Model_args_as_cpp_struct.Model_args_bools(16)
                                       : false;
      const bool has_y_upper = (Model_args_as_cpp_struct.Model_args_bools.rows() > 17)
                               ? Model_args_as_cpp_struct.Model_args_bools(17)
                               : false;

      //// Integer metadata
      const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
      const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
      const int n_pops = Model_args_as_cpp_struct.Model_args_ints(6);
      const int use_Se_Sp_prior = Model_args_as_cpp_struct.Model_args_ints(7);
      const int n_covariates_max = Model_args_as_cpp_struct.Model_args_ints(9);
      const int n_prevalence_covariates = Model_args_as_cpp_struct.Model_args_ints(10);
      const int pool_correlations_across_classes = Model_args_as_cpp_struct.Model_args_ints(11);
      const int use_pop_prev_prior = Model_args_as_cpp_struct.Model_args_ints(12);
      const int n_standardisation = Model_args_as_cpp_struct.Model_args_ints(13);
      const int use_label_constraint = Model_args_as_cpp_struct.Model_args_ints(14);

      if (use_Se_Sp_prior == 1) {
            throw std::invalid_argument("CLC-MVOP AD: use_Se_Sp_prior = 1 (the prior on the marginal Se / Sp) is in the manual-gradient path only.");
      }
      if (use_pop_prev_prior >= 2) {
            throw std::invalid_argument("CLC-MVOP AD: use_pop_prev_prior = 2 or 3 (priors on Pr(B | not A), and on Pr(A | B) for 3) is in the manual-gradient path only.");
      }
      if (n_class != 4) {
            throw std::invalid_argument("CLC-MVOP AD: requires n_class == 4.");
      }

      const double overflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(0);
      const double underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(1);
      const double C_raw_lower = Model_args_as_cpp_struct.Model_args_doubles(2);
      const double C_raw_upper = Model_args_as_cpp_struct.Model_args_doubles(3);
      const double perfect_ref_M = Model_args_as_cpp_struct.Model_args_doubles(4);

      const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
      const std::string &Phi_type = Model_args_as_cpp_struct.Model_args_strings(1);
      const std::string &inv_Phi_type = Model_args_as_cpp_struct.Model_args_strings(2);
      const std::string &nuisance_transformation = Model_args_as_cpp_struct.Model_args_strings(12);
      const AD_Phi_setting_struct ad_phi_setting = fn_AD_Phi_setting_from_strings(
            Phi_type,
            inv_Phi_type,
            "fn_lp_and_grad_CLC_MVOP_Pinkney_AD_log_scale_InPlace_process"
      );
      const bool use_Phi_approx = ad_phi_setting.use_Phi_approx;
      const bool use_inv_Phi_approx = ad_phi_setting.use_inv_Phi_approx;

      const Eigen::Matrix<int, -1, 1> &n_cat_per_ord_test = Model_args_as_cpp_struct.Model_args_col_vecs_int[0];
      const Eigen::Matrix<int, -1, 1> &n_thr_per_ord_test = Model_args_as_cpp_struct.Model_args_col_vecs_int[1];
      const Eigen::Matrix<int, -1, 1> &pop_ind = Model_args_as_cpp_struct.Model_args_col_vecs_int[2];
      const Eigen::Matrix<int, -1, 1> &n_cat_per_test = Model_args_as_cpp_struct.Model_args_col_vecs_int[3];
      const Eigen::Matrix<int, -1, 1> &test_condition = Model_args_as_cpp_struct.Model_args_col_vecs_int[4];
      const Eigen::Matrix<int, -1, 1> &is_perfect_ref = Model_args_as_cpp_struct.Model_args_col_vecs_int[5];
      const Eigen::Matrix<int, -1, 1> &std_index = Model_args_as_cpp_struct.Model_args_col_vecs_int[6];
      const Eigen::Matrix<int, -1, 1> &pop_std = Model_args_as_cpp_struct.Model_args_col_vecs_int[7];
      const Eigen::Matrix<int, -1, 1> &reference_test_fitted = Model_args_as_cpp_struct.Model_args_col_vecs_int[8];
      const Eigen::Matrix<int, -1, 1> &cutoff_category = Model_args_as_cpp_struct.Model_args_col_vecs_int[9];
      const Eigen::Matrix<int, -1, -1> &n_covs_per_outcome = Model_args_as_cpp_struct.Model_args_mats_int[0];
      const Eigen::Matrix<int, -1, -1> &class_D = Model_args_as_cpp_struct.Model_args_mats_int[1];
      const Eigen::Matrix<int, -1, -1> &y_upper_mat = Model_args_as_cpp_struct.Model_args_mats_int[2];
      const Eigen::Matrix<double, -1, -1> &X_prev = Model_args_as_cpp_struct.Model_args_mats_double[0];
      const Eigen::Matrix<double, -1, -1> &X_prev_std = Model_args_as_cpp_struct.Model_args_mats_double[1];
      const Eigen::Matrix<double, -1, 1> &std_w = Model_args_as_cpp_struct.Model_args_col_vecs_double[5];
      const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> &X_class_test =
            Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0];

      const std::vector<Eigen::Matrix<double, -1, -1>> &prior_beta_mean =
            Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0];
      const std::vector<Eigen::Matrix<double, -1, -1>> &prior_beta_sd =
            Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1];
      const std::vector<Eigen::Matrix<double, -1, -1>> &ub_corr =
            Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3];
      const std::vector<Eigen::Matrix<double, -1, -1>> &known_val =
            Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4];
      const std::vector<Eigen::Matrix<double, -1, -1>> &prior_dir_alpha =
            Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5];
      const std::vector<Eigen::Matrix<double, -1, -1>> &baseline_case =
            Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[11];
      const std::vector<Eigen::Matrix<int, -1, -1>> &known_ind =
            Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];

      //// Slot maps
      Eigen::Matrix<int, -1, 1> ord_idx_of_test(n_tests);
      Eigen::Matrix<int, -1, 1> fitted_test_of_ord_slot(std::max(1, n_tests));
      int n_ordinal_tests = 0;
      for (int test_index = 0; test_index < n_tests; test_index++) {
            if (n_cat_per_test(test_index) > 2) {
                  ord_idx_of_test(test_index) = n_ordinal_tests;
                  fitted_test_of_ord_slot(n_ordinal_tests) = test_index;
                  n_ordinal_tests++;
            } else {
                  ord_idx_of_test(test_index) = -1;
            }
      }
      Eigen::Matrix<int, -1, 1> ord_start_index(std::max(1, n_ordinal_tests));
      int n_total_C_per_class = 0;
      for (int ord_index = 0; ord_index < n_ordinal_tests; ord_index++) {
            ord_start_index(ord_index) = n_total_C_per_class;
            n_total_C_per_class += n_thr_per_ord_test(ord_index);
      }
      const int n_cutpoints_max = (n_ordinal_tests > 0)
                                  ? n_thr_per_ord_test.head(n_ordinal_tests).maxCoeff()
                                  : 0;

      int class_with_neither = 0;
      int class_with_only[2] = { 1, 2 };
      for (int class_index = 0; class_index < n_class; class_index++) {
            const int d0 = class_D(class_index, 0);
            const int d1 = class_D(class_index, 1);
            if ((d0 == 0) && (d1 == 0)) {
                  class_with_neither = class_index;
            } else if ((d0 == 1) && (d1 == 0)) {
                  class_with_only[0] = class_index;
            } else if ((d0 == 0) && (d1 == 1)) {
                  class_with_only[1] = class_index;
            }
      }

      std::vector<Eigen::Matrix<double, -1, -1>> lb_corr_actual =
            Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2];
      if (corr_force_positive) {
            for (int class_index = 0; class_index < n_class; class_index++) {
                  for (int row_index = 1; row_index < n_tests; row_index++) {
                        for (int col_index = 0; col_index < row_index; col_index++) {
                              lb_corr_actual[class_index](row_index, col_index) = 0.0;
                        }
                  }
            }
      }

      Eigen::Matrix<int, -1, 1> n_corrs_per_class = Eigen::Matrix<int, -1, 1>::Zero(n_class);
      Eigen::Matrix<int, -1, 1> corr_start_index(n_class);
      int n_corrs_total = 0;
      for (int class_index = 0; class_index < n_class; class_index++) {
            corr_start_index(class_index) = n_corrs_total;
            for (int row_index = 1; row_index < n_tests; row_index++) {
                  for (int col_index = 0; col_index < row_index; col_index++) {
                        if (known_ind[class_index](row_index, col_index) == 0) {
                              n_corrs_per_class(class_index) += 1;
                        }
                  }
            }
            n_corrs_total += n_corrs_per_class(class_index);
      }
      const int n_corrs_global = (pool_correlations_across_classes == 1) ? n_corrs_per_class(0) : 0;
      const int n_lambda = (pool_correlations_across_classes == 1) ? n_class : 0;

      int n_covariates_total = 0;
      for (int class_index = 0; class_index < n_class; class_index++) {
            for (int test_index = 0; test_index < n_tests; test_index++) {
                  n_covariates_total += n_covs_per_outcome(class_index, test_index);
            }
      }

      //// Exact theta_main offsets
      int offset_cur = 0;
      const int off_Omega = offset_cur;
      offset_cur += n_corrs_total;
      const int off_Omega_global = offset_cur;
      offset_cur += n_corrs_global;
      const int off_lambda_class = offset_cur;
      offset_cur += n_lambda;
      const int off_beta_vec = offset_cur;
      offset_cur += n_covariates_total;
      const int off_beta_prev_A = offset_cur;
      offset_cur += n_prevalence_covariates;
      const int off_beta_prev_B = offset_cur;
      offset_cur += n_prevalence_covariates;
      const int off_z_logit_A = offset_cur;
      offset_cur += n_pops;
      const int off_z_logit_B = offset_cur;
      offset_cur += n_pops;
      const int off_mu_logit_A = offset_cur;
      offset_cur += 1;
      const int off_mu_logit_B = offset_cur;
      offset_cur += 1;
      const int off_sigma_logit_A = offset_cur;
      offset_cur += 1;
      const int off_sigma_logit_B = offset_cur;
      offset_cur += 1;
      const int off_log_OR = offset_cur;
      offset_cur += 1;
      const int off_C_unc = offset_cur;
      offset_cur += n_class * n_total_C_per_class;
      if (offset_cur != n_params_main) {
            throw std::invalid_argument(
                  "CLC-MVOP AD: theta_main length mismatch (expected "
                  + std::to_string(offset_cur)
                  + ", got "
                  + std::to_string(n_params_main)
                  + ")."
            );
      }

      //// Chunking layout matching NoLog
      int vec_size = 1;
      if (vect_type == "AVX512") {
            vec_size = 8;
      } else if (vect_type == "AVX2") {
            vec_size = 4;
      } else if (vect_type == "AVX") {
            vec_size = 2;
      }
      const ChunkSizeInfo chunk_info = calculate_chunk_sizes(
            n_obs,
            vec_size,
            n_chunks
      );

      //// -------------------------------------------------------------------------------------
      //// Single nested Stan Math reverse-mode AD tape over ALL parameters (us + main)
      //// -------------------------------------------------------------------------------------
      stan::math::nested_rev_autodiff nested_autodiff_scope;

      Eigen::Matrix<stan::math::var, -1, 1> theta_var(n_params);
      for (int elem_index = 0; elem_index < n_us; elem_index++) {
            theta_var(elem_index) = stan::math::to_var(theta_us_vec_ref(elem_index));
      }
      for (int elem_index = 0; elem_index < n_params_main; elem_index++) {
            theta_var(n_us + elem_index) = stan::math::to_var(theta_main_vec_ref(elem_index));
      }
      const Eigen::Matrix<stan::math::var, -1, 1> u_unc_var = theta_var.head(n_us);
      const Eigen::Matrix<stan::math::var, -1, 1> theta_main_var = theta_var.tail(n_params_main);

      stan::math::var target_AD = 0.0;

      //// -------------------------------------------------------------------------------------
      //// 1. Correlations
      //// -------------------------------------------------------------------------------------
      const Eigen::Matrix<double, -1, 1> &prior_LKJ = Model_args_as_cpp_struct.Model_args_col_vecs_double[0];
      Eigen::Matrix<stan::math::var, -1, 1> lambda_var(n_lambda);
      stan::math::var lp_corr_jac = 0.0;
      stan::math::var lp_corr_prior = 0.0;

      for (int class_index = 0; class_index < n_lambda; class_index++) {
            const stan::math::var unc_lam = theta_main_var(off_lambda_class + class_index);
            lambda_var(class_index) = stan::math::inv_logit(unc_lam);
            lp_corr_jac += stan::math::log_inv_logit(unc_lam)
                           + stan::math::log_inv_logit(-unc_lam);
      }

      Eigen::Matrix<stan::math::var, -1, -1> L_global_var = Eigen::Matrix<stan::math::var, -1, -1>::Identity(
            n_tests,
            n_tests
      );
      Eigen::Matrix<stan::math::var, -1, -1> Omega_global_var = Eigen::Matrix<stan::math::var, -1, -1>::Identity(
            n_tests,
            n_tests
      );
      if (pool_correlations_across_classes == 1) {
            const Eigen::Matrix<stan::math::var, -1, 1> omega_glob_unc_var = theta_main_var.segment(
                  off_Omega_global,
                  n_corrs_global
            );
            Eigen::Matrix<stan::math::var, -1, -1> omega_glob_mat =
                  Eigen::Matrix<stan::math::var, -1, -1>::Zero(n_tests, n_tests);
            Eigen::Matrix<stan::math::var, -1, -1> lb_glob_var = lb_corr_actual[0].cast<stan::math::var>();
            Eigen::Matrix<stan::math::var, -1, -1> ub_glob_var = ub_corr[0].cast<stan::math::var>();
            int free_pos_g = 0;
            int n_known_g = 0;
            for (int r_i = 1; r_i < n_tests; r_i++) {
                  for (int c_i = 0; c_i < r_i; c_i++) {
                        if (known_ind[0](r_i, c_i) == 0) {
                              omega_glob_mat(r_i, c_i) = omega_glob_unc_var(free_pos_g);
                              free_pos_g++;
                        } else {
                              n_known_g++;
                        }
                  }
            }
            const Eigen::Matrix<stan::math::var, -1, -1> pinkney_g_out = Pinkney_LDL_bounds_opt(
                  n_tests,
                  lb_glob_var,
                  ub_glob_var,
                  omega_glob_mat,
                  known_ind[0],
                  known_val[0]
            );
            lp_corr_jac += pinkney_g_out(0, 0) + static_cast<double>(n_known_g) * clc_mvop_detail::CLC_HALF_LOG_2PI;
            L_global_var = pinkney_g_out.block(1, 0, n_tests, n_tests);
            Omega_global_var = stan::math::multiply_lower_tri_self_transpose(L_global_var);
            lp_corr_prior += stan::math::lkj_corr_cholesky_lpdf(
                  L_global_var,
                  Model_args_as_cpp_struct.Model_args_doubles(5)
            );
      }

      std::vector<Eigen::Matrix<stan::math::var, -1, -1>> L_Omega_var(n_class);
      const double prior_LKJ_dep = Model_args_as_cpp_struct.Model_args_doubles(6);
      const double lambda_alpha = Model_args_as_cpp_struct.Model_args_doubles(7);
      const double lambda_beta = Model_args_as_cpp_struct.Model_args_doubles(8);
      const double log_beta_lambda = std::lgamma(lambda_alpha)
                                     + std::lgamma(lambda_beta)
                                     - std::lgamma(lambda_alpha + lambda_beta);

      for (int class_index = 0; class_index < n_class; class_index++) {
            const int n_corrs_class = n_corrs_per_class(class_index);
            const Eigen::Matrix<stan::math::var, -1, 1> raw_class_vec = theta_main_var.segment(
                  off_Omega + corr_start_index(class_index),
                  n_corrs_class
            );
            Eigen::Matrix<stan::math::var, -1, -1> omega_cls_mat =
                  Eigen::Matrix<stan::math::var, -1, -1>::Zero(n_tests, n_tests);
            Eigen::Matrix<stan::math::var, -1, -1> lb_cls_var = lb_corr_actual[class_index].cast<stan::math::var>();
            Eigen::Matrix<stan::math::var, -1, -1> ub_cls_var = ub_corr[class_index].cast<stan::math::var>();
            int free_pos_c = 0;
            int n_known_c = 0;
            for (int r_i = 1; r_i < n_tests; r_i++) {
                  for (int c_i = 0; c_i < r_i; c_i++) {
                        if (known_ind[class_index](r_i, c_i) == 0) {
                              omega_cls_mat(r_i, c_i) = raw_class_vec(free_pos_c);
                              free_pos_c++;
                        } else {
                              n_known_c++;
                        }
                  }
            }
            const Eigen::Matrix<stan::math::var, -1, -1> pinkney_c_out = Pinkney_LDL_bounds_opt(
                  n_tests,
                  lb_cls_var,
                  ub_cls_var,
                  omega_cls_mat,
                  known_ind[class_index],
                  known_val[class_index]
            );
            lp_corr_jac += pinkney_c_out(0, 0) + static_cast<double>(n_known_c) * clc_mvop_detail::CLC_HALF_LOG_2PI;
            Eigen::Matrix<stan::math::var, -1, -1> L_dep = pinkney_c_out.block(1, 0, n_tests, n_tests);
            if (pool_correlations_across_classes == 1) {
                  Eigen::Matrix<stan::math::var, -1, -1> Omega_class =
                        (1.0 - lambda_var(class_index)) * Omega_global_var
                        + lambda_var(class_index) * stan::math::multiply_lower_tri_self_transpose(L_dep);
                  L_Omega_var[class_index] = stan::math::cholesky_decompose(Omega_class);
                  lp_corr_prior += stan::math::lkj_corr_cholesky_lpdf(
                        L_dep,
                        prior_LKJ_dep
                  );
                  lp_corr_prior += (lambda_alpha - 1.0) * stan::math::log(lambda_var(class_index))
                                   + (lambda_beta - 1.0) * stan::math::log1m(lambda_var(class_index))
                                   - log_beta_lambda;
            } else {
                  L_Omega_var[class_index] = L_dep;
                  lp_corr_prior += stan::math::lkj_corr_cholesky_lpdf(
                        L_Omega_var[class_index],
                        prior_LKJ(class_index)
                  );
            }
      }
      target_AD += lp_corr_jac;
      if (!exclude_priors) {
            target_AD += lp_corr_prior;
      }

      //// -------------------------------------------------------------------------------------
      //// 2. Per-class regression coefficients beta[c, k, t]
      //// -------------------------------------------------------------------------------------
      const int K_alloc = std::max(1, n_covariates_max);
      std::vector<Eigen::Matrix<stan::math::var, -1, -1>> beta_4_var(
            n_class,
            Eigen::Matrix<stan::math::var, -1, -1>::Zero(
                  K_alloc,
                  n_tests
            )
      );
      int beta_cursor = 0;
      for (int class_index = 0; class_index < n_class; class_index++) {
            for (int test_index = 0; test_index < n_tests; test_index++) {
                  const int n_covs_ct = n_covs_per_outcome(class_index, test_index);
                  const bool is_perf = (is_perfect_ref(test_index) == 1);
                  for (int cov_index = 0; cov_index < n_covs_ct; cov_index++) {
                        const stan::math::var val_cur = theta_main_var(off_beta_vec + beta_cursor);
                        if (is_perf) {
                              if (!exclude_priors) {
                                    target_AD += stan::math::std_normal_lpdf(val_cur);
                              }
                        } else {
                              beta_4_var[class_index](cov_index, test_index) = val_cur;
                              if (!exclude_priors) {
                                    target_AD += stan::math::normal_lpdf(
                                          val_cur,
                                          prior_beta_mean[class_index](cov_index, test_index),
                                          prior_beta_sd[class_index](cov_index, test_index)
                                    );
                              }
                        }
                        beta_cursor++;
                  }
                  if (is_perf) {
                        const int cond_index = test_condition(test_index);
                        beta_4_var[class_index](0, test_index) =
                              (class_D(class_index, cond_index) == 1) ? perfect_ref_M : (-perfect_ref_M);
                        for (int cov_index = 1; cov_index < K_alloc; cov_index++) {
                              beta_4_var[class_index](cov_index, test_index) = 0.0;
                        }
                  }
            }
      }

      //// -------------------------------------------------------------------------------------
      //// 3. Per-class ordered cutpoints (tanh lb_ub + log-difference)
      //// -------------------------------------------------------------------------------------
      std::vector<Eigen::Matrix<stan::math::var, -1, -1>> cutpoint_4_var(
            n_class,
            Eigen::Matrix<stan::math::var, -1, -1>::Zero(
                  std::max(1, n_cutpoints_max),
                  std::max(1, n_ordinal_tests)
            )
      );
      if (n_ordinal_tests > 0) {
            stan::math::var lp_cut_jac = 0.0;
            for (int class_index = 0; class_index < n_class; class_index++) {
                  const int class_base = off_C_unc + class_index * n_total_C_per_class;
                  for (int ord_index = 0; ord_index < n_ordinal_tests; ord_index++) {
                        const int n_thr_test = n_thr_per_ord_test(ord_index);
                        const int ord_base = class_base + ord_start_index(ord_index);
                        for (int thr_index = 0; thr_index < n_thr_test; thr_index++) {
                              const Eigen::Matrix<stan::math::var, -1, 1> lb_ub_cut_out = lb_ub_lp(
                                    theta_main_var(ord_base + thr_index),
                                    stan::math::var(C_raw_lower),
                                    stan::math::var(C_raw_upper)
                              );
                              lp_cut_jac += lb_ub_cut_out(0);
                              const stan::math::var C_raw_cur = lb_ub_cut_out(1);
                              if (thr_index == 0) {
                                    cutpoint_4_var[class_index](0, ord_index) = C_raw_cur;
                              } else {
                                    cutpoint_4_var[class_index](thr_index, ord_index) =
                                          cutpoint_4_var[class_index](thr_index - 1, ord_index)
                                          + stan::math::exp(C_raw_cur);
                                    lp_cut_jac += C_raw_cur;
                              }
                        }
                  }
            }
            target_AD += lp_cut_jac;
      }

      //// -------------------------------------------------------------------------------------
      //// 4. Optional label-ordering truncation constraint
      //// -------------------------------------------------------------------------------------
      if (use_label_constraint == 1) {
            for (int cond_index = 0; cond_index < 2; cond_index++) {
                  const int test_index = reference_test_fitted(cond_index);
                  const int class_pair[2] = { class_with_only[cond_index], class_with_neither };
                  double metric_val[2] = { 0.0, 0.0 };
                  for (int pair_pos = 0; pair_pos < 2; pair_pos++) {
                        const int class_index = class_pair[pair_pos];
                        const int n_covs_ct = n_covs_per_outcome(class_index, test_index);
                        double xb_base = 0.0;
                        for (int cov_index = 0; cov_index < n_covs_ct; cov_index++) {
                              xb_base += baseline_case[class_index](cov_index, test_index)
                                         * beta_4_var[class_index](cov_index, test_index).val();
                        }
                        const int ord_index = ord_idx_of_test(test_index);
                        if (ord_index >= 0) {
                              const int cut_idx = cutoff_category(test_index) - 2;
                              xb_base -= cutpoint_4_var[class_index](cut_idx, ord_index).val();
                        }
                        metric_val[pair_pos] = xb_base;
                  }
                  if (metric_val[0] < metric_val[1]) {
                        out_mat.setZero();
                        out_mat(0) = -std::numeric_limits<double>::infinity();
                        return;
                  }
            }
      }

      //// -------------------------------------------------------------------------------------
      //// 5. Prevalence (2 x 2 conditional factorial + population standardisation priors)
      //// -------------------------------------------------------------------------------------
      Eigen::Matrix<stan::math::var, -1, 1> beta_prev_A_var(n_prevalence_covariates);
      Eigen::Matrix<stan::math::var, -1, 1> beta_prev_B_var(n_prevalence_covariates);
      for (int cov_index = 0; cov_index < n_prevalence_covariates; cov_index++) {
            beta_prev_A_var(cov_index) = theta_main_var(off_beta_prev_A + cov_index);
            beta_prev_B_var(cov_index) = theta_main_var(off_beta_prev_B + cov_index);
            if (!exclude_priors) {
                  target_AD += stan::math::normal_lpdf(
                        beta_prev_A_var(cov_index),
                        Model_args_as_cpp_struct.Model_args_col_vecs_double[1](cov_index),
                        Model_args_as_cpp_struct.Model_args_col_vecs_double[2](cov_index)
                  );
                  target_AD += stan::math::normal_lpdf(
                        beta_prev_B_var(cov_index),
                        Model_args_as_cpp_struct.Model_args_col_vecs_double[3](cov_index),
                        Model_args_as_cpp_struct.Model_args_col_vecs_double[4](cov_index)
                  );
            }
      }

      const Eigen::Matrix<stan::math::var, -1, 1> z_A_var = theta_main_var.segment(
            off_z_logit_A,
            n_pops
      );
      const Eigen::Matrix<stan::math::var, -1, 1> z_B_var = theta_main_var.segment(
            off_z_logit_B,
            n_pops
      );
      const stan::math::var mu_A_var = theta_main_var(off_mu_logit_A);
      const stan::math::var mu_B_var = theta_main_var(off_mu_logit_B);
      const stan::math::var log_sig_A_var = theta_main_var(off_sigma_logit_A);
      const stan::math::var log_sig_B_var = theta_main_var(off_sigma_logit_B);
      const stan::math::var sigma_A_var = stan::math::exp(log_sig_A_var);
      const stan::math::var sigma_B_var = stan::math::exp(log_sig_B_var);
      const stan::math::var log_OR_var = theta_main_var(off_log_OR);
      target_AD += log_sig_A_var + log_sig_B_var;

      if (!exclude_priors) {
            for (int pop_index = 0; pop_index < n_pops; pop_index++) {
                  target_AD += stan::math::std_normal_lpdf(z_A_var(pop_index));
                  target_AD += stan::math::std_normal_lpdf(z_B_var(pop_index));
            }
            target_AD += stan::math::normal_lpdf(
                  sigma_A_var,
                  0.0,
                  Model_args_as_cpp_struct.Model_args_doubles(13)
            );
            target_AD += stan::math::normal_lpdf(
                  sigma_B_var,
                  0.0,
                  Model_args_as_cpp_struct.Model_args_doubles(14)
            );
            if (use_pop_prev_prior == 0) {
                  target_AD += stan::math::normal_lpdf(
                        mu_A_var,
                        Model_args_as_cpp_struct.Model_args_doubles(9),
                        Model_args_as_cpp_struct.Model_args_doubles(10)
                  );
                  target_AD += stan::math::normal_lpdf(
                        mu_B_var,
                        Model_args_as_cpp_struct.Model_args_doubles(11),
                        Model_args_as_cpp_struct.Model_args_doubles(12)
                  );
            }
            Eigen::Matrix<stan::math::var, -1, 1> log_w_a_terms(n_standardisation);
            Eigen::Matrix<stan::math::var, -1, 1> log_w_1ma_terms(n_standardisation);
            Eigen::Matrix<stan::math::var, -1, 1> log_w_aq_terms(n_standardisation);
            Eigen::Matrix<stan::math::var, -1, 1> log_w_a1mq_terms(n_standardisation);
            Eigen::Matrix<stan::math::var, -1, 1> log_w_jac_Q_terms(n_standardisation);
            Eigen::Matrix<stan::math::var, -1, 1> log_w_jac_A_terms(n_standardisation);
            Eigen::Matrix<stan::math::var, -1, 1> log_w_1mab_terms(n_standardisation);
            Eigen::Matrix<stan::math::var, -1, 1> log_w_1ma1mb_terms(n_standardisation);
            Eigen::Matrix<stan::math::var, -1, 1> log_w_jac_B_terms(n_standardisation);
            for (int prof_index = 0; prof_index < n_standardisation; prof_index++) {
                  const int pop_index = pop_std(prof_index);
                  stan::math::var eta_a_cur = mu_A_var + sigma_A_var * z_A_var(pop_index);
                  stan::math::var eta_b_cur = mu_B_var + sigma_B_var * z_B_var(pop_index);
                  for (int cov_index = 0; cov_index < n_prevalence_covariates; cov_index++) {
                        eta_a_cur += X_prev_std(prof_index, cov_index) * beta_prev_A_var(cov_index);
                        eta_b_cur += X_prev_std(prof_index, cov_index) * beta_prev_B_var(cov_index);
                  }
                  const double log_w_cur = std::log(std_w(prof_index));
                  const stan::math::var log_pa = stan::math::log_inv_logit(eta_a_cur);
                  const stan::math::var log_1m_pa = stan::math::log_inv_logit(-eta_a_cur);
                  const stan::math::var eta_q_cur = eta_b_cur + log_OR_var;
                  const stan::math::var log_pq = stan::math::log_inv_logit(eta_q_cur);
                  const stan::math::var log_1m_pq = stan::math::log_inv_logit(-eta_q_cur);
                  log_w_a_terms(prof_index) = log_w_cur + log_pa;
                  log_w_aq_terms(prof_index) = log_w_cur + log_pa + log_pq;
                  log_w_a1mq_terms(prof_index) = log_w_cur + log_pa + log_1m_pq;
                  log_w_jac_Q_terms(prof_index) = log_w_cur + log_pa + log_pq + log_1m_pq;
                  if (use_pop_prev_prior == 1) {
                        const stan::math::var log_pb = stan::math::log_inv_logit(eta_b_cur);
                        const stan::math::var log_1m_pb = stan::math::log_inv_logit(-eta_b_cur);
                        log_w_1ma_terms(prof_index) = log_w_cur + log_1m_pa;
                        log_w_jac_A_terms(prof_index) = log_w_cur + log_pa + log_1m_pa;
                        log_w_1mab_terms(prof_index) = log_w_cur + log_1m_pa + log_pb;
                        log_w_1ma1mb_terms(prof_index) = log_w_cur + log_1m_pa + log_1m_pb;
                        log_w_jac_B_terms(prof_index) = log_w_cur + log_1m_pa + log_pb + log_1m_pb;
                  }
            }
            const stan::math::var log_sum_a = stan::math::log_sum_exp(log_w_a_terms);
            const stan::math::var log_sum_aq = stan::math::log_sum_exp(log_w_aq_terms);
            const stan::math::var log_sum_a_1mq = stan::math::log_sum_exp(log_w_a1mq_terms);
            const stan::math::var log_sum_jac_Q = stan::math::log_sum_exp(log_w_jac_Q_terms);
            const stan::math::var logit_Q = log_sum_aq - log_sum_a_1mq;
            target_AD += stan::math::normal_lpdf(
                  logit_Q,
                  Model_args_as_cpp_struct.Model_args_doubles(15),
                  Model_args_as_cpp_struct.Model_args_doubles(16)
            ) - log_sum_aq - log_sum_a_1mq + log_sum_a + log_sum_jac_Q;

            if (use_pop_prev_prior == 1) {
                  const stan::math::var log_sum_1m_a = stan::math::log_sum_exp(log_w_1ma_terms);
                  const stan::math::var log_sum_jac_A = stan::math::log_sum_exp(log_w_jac_A_terms);
                  const stan::math::var log_sum_1ma_b = stan::math::log_sum_exp(log_w_1mab_terms);
                  const stan::math::var log_sum_1ma_1mb = stan::math::log_sum_exp(log_w_1ma1mb_terms);
                  const stan::math::var log_sum_jac_B = stan::math::log_sum_exp(log_w_jac_B_terms);
                  const stan::math::var log_sum_B = stan::math::log_sum_exp(
                        log_sum_1ma_b,
                        log_sum_aq
                  );
                  const stan::math::var log_sum_1m_B = stan::math::log_sum_exp(
                        log_sum_1ma_1mb,
                        log_sum_a_1mq
                  );
                  const stan::math::var logit_A = log_sum_a - log_sum_1m_a;
                  const stan::math::var logit_B = log_sum_B - log_sum_1m_B;
                  target_AD += stan::math::normal_lpdf(
                        logit_A,
                        Model_args_as_cpp_struct.Model_args_doubles(9),
                        Model_args_as_cpp_struct.Model_args_doubles(10)
                  ) - log_sum_a - log_sum_1m_a + log_sum_jac_A;
                  target_AD += stan::math::normal_lpdf(
                        logit_B,
                        Model_args_as_cpp_struct.Model_args_doubles(11),
                        Model_args_as_cpp_struct.Model_args_doubles(12)
                  ) - log_sum_B - log_sum_1m_B + log_sum_jac_B;
            }
      }

      Eigen::Matrix<stan::math::var, -1, -1> log_prev_subject_var(
            n_obs,
            n_class
      );
      for (int row_index = 0; row_index < n_obs; row_index++) {
            const int pop_index = pop_ind(row_index);
            stan::math::var eta_a_cur = mu_A_var + sigma_A_var * z_A_var(pop_index);
            stan::math::var eta_b_cur = mu_B_var + sigma_B_var * z_B_var(pop_index);
            for (int cov_index = 0; cov_index < n_prevalence_covariates; cov_index++) {
                  eta_a_cur += X_prev(row_index, cov_index) * beta_prev_A_var(cov_index);
                  eta_b_cur += X_prev(row_index, cov_index) * beta_prev_B_var(cov_index);
            }
            const stan::math::var eta_q_cur = eta_b_cur + log_OR_var;
            const stan::math::var log_pA = stan::math::log_inv_logit(eta_a_cur);
            const stan::math::var log_1m_pA = stan::math::log_inv_logit(-eta_a_cur);
            const stan::math::var log_pB = stan::math::log_inv_logit(eta_b_cur);
            const stan::math::var log_1m_pB = stan::math::log_inv_logit(-eta_b_cur);
            const stan::math::var log_pQ = stan::math::log_inv_logit(eta_q_cur);
            const stan::math::var log_1m_pQ = stan::math::log_inv_logit(-eta_q_cur);
            log_prev_subject_var(row_index, 0) = log_1m_pA + log_1m_pB;
            log_prev_subject_var(row_index, 1) = log_pA + log_1m_pQ;
            log_prev_subject_var(row_index, 2) = log_1m_pA + log_pB;
            log_prev_subject_var(row_index, 3) = log_pA + log_pQ;
      }

      //// -------------------------------------------------------------------------------------
      //// 6. Cutpoint priors (baseline OR population-averaged across ALL cutpoints)
      //// -------------------------------------------------------------------------------------
      if (!exclude_priors && (n_ordinal_tests > 0)) {
            if (use_pop_Se_Sp_prior) {
                  for (int class_index = 0; class_index < n_class; class_index++) {
                        Eigen::Matrix<stan::math::var, -1, 1> log_w_c(n_standardisation);
                        for (int prof_index = 0; prof_index < n_standardisation; prof_index++) {
                              log_w_c(prof_index) = std::log(std_w(prof_index))
                                                    + log_prev_subject_var(std_index(prof_index), class_index);
                        }
                        const stan::math::var log_sum_w_c = stan::math::log_sum_exp(log_w_c);
                        for (int prof_index = 0; prof_index < n_standardisation; prof_index++) {
                              log_w_c(prof_index) -= log_sum_w_c;
                        }
                        for (int ord_index = 0; ord_index < n_ordinal_tests; ord_index++) {
                              const int test_index = fitted_test_of_ord_slot(ord_index);
                              const int n_thr_test = n_thr_per_ord_test(ord_index);
                              const int n_cat_test = n_thr_test + 1;
                              const int n_covs_ct = n_covs_per_outcome(class_index, test_index);
                              const Eigen::Matrix<double, -1, 1> alpha_ct =
                                    prior_dir_alpha[class_index].col(ord_index).head(n_cat_test);
                              Eigen::Matrix<stan::math::var, -1, 1> Xbeta_std(n_standardisation);
                              for (int prof_index = 0; prof_index < n_standardisation; prof_index++) {
                                    const int subj_idx = std_index(prof_index);
                                    stan::math::var sum_val = 0.0;
                                    for (int cov_index = 0; cov_index < n_covs_ct; cov_index++) {
                                          sum_val += X_class_test[(X_class_test.size() == 1) ? 0 : class_index][(X_class_test[(X_class_test.size() == 1) ? 0 : class_index].size() == 1) ? 0 : test_index](subj_idx, cov_index)
                                                     * beta_4_var[class_index](cov_index, test_index);
                                    }
                                    Xbeta_std(prof_index) = sum_val;
                              }
                              Eigen::Matrix<stan::math::var, -1, 1> z_pop_raw(n_thr_test);
                              Eigen::Matrix<stan::math::var, -1, 1> log_p_terms(n_standardisation);
                              Eigen::Matrix<stan::math::var, -1, 1> log_q_terms(n_standardisation);
                              Eigen::Matrix<stan::math::var, -1, 1> log_J_terms(n_standardisation);
                              for (int thr_index = 0; thr_index < n_thr_test; thr_index++) {
                                    const stan::math::var C_k = cutpoint_4_var[class_index](thr_index, ord_index);
                                    for (int prof_index = 0; prof_index < n_standardisation; prof_index++) {
                                          const stan::math::var margin_ki = C_k - Xbeta_std(prof_index);
                                          log_p_terms(prof_index) = log_w_c(prof_index) + stan::math::std_normal_lcdf(margin_ki);
                                          log_q_terms(prof_index) = log_w_c(prof_index) + stan::math::std_normal_lcdf(-margin_ki);
                                          log_J_terms(prof_index) = log_w_c(prof_index) - 0.5 * stan::math::square(margin_ki);
                                    }
                                    const stan::math::var log_p_k = stan::math::log_sum_exp(log_p_terms);
                                    const stan::math::var log_q_k = stan::math::log_sum_exp(log_q_terms);
                                    z_pop_raw(thr_index) = fn_AD_inv_Phi_from_log_probs_var(
                                          log_p_k,
                                          log_q_k,
                                          false
                                    );
                                    target_AD += stan::math::log_sum_exp(log_J_terms)
                                                 + 0.5 * stan::math::square(z_pop_raw(thr_index));
                              }
                              const double x0_val = baseline_case[class_index](0, test_index);
                              const stan::math::var x0_shift = x0_val * (
                                    beta_4_var[class_index](0, test_index)
                                    - prior_beta_mean[class_index](0, test_index)
                              );
                              Eigen::Matrix<stan::math::var, -1, 1> C_eff_var(n_thr_test);
                              for (int thr_index = 0; thr_index < n_thr_test; thr_index++) {
                                    C_eff_var(thr_index) = z_pop_raw(thr_index) + x0_shift;
                              }
                              {
                                    Eigen::Matrix<stan::math::var, -1, 1> cumul_eff_var(n_thr_test);
                                    for (int thr_index = 0; thr_index < n_thr_test; thr_index++) {
                                          cumul_eff_var(thr_index) = stan::math::Phi(C_eff_var(thr_index));
                                    }
                                    const Eigen::Matrix<stan::math::var, -1, 1> p_ord_eff_var =
                                          cumul_probs_to_ord_probs_var(cumul_eff_var);
                                    target_AD += induced_dirichlet_given_C_lpdf_var(
                                          p_ord_eff_var,
                                          C_eff_var,
                                          alpha_ct.cast<stan::math::var>(),
                                          true
                                    );
                              }
                        }
                  }
            } else {
                  for (int class_index = 0; class_index < n_class; class_index++) {
                        for (int ord_index = 0; ord_index < n_ordinal_tests; ord_index++) {
                              const int test_index = fitted_test_of_ord_slot(ord_index);
                              const int n_thr_test = n_thr_per_ord_test(ord_index);
                              const int n_cat_test = n_thr_test + 1;
                              const int n_covs_ct = n_covs_per_outcome(class_index, test_index);
                              const Eigen::Matrix<double, -1, 1> alpha_ct =
                                    prior_dir_alpha[class_index].col(ord_index).head(n_cat_test);
                              double baseline_loc_prior = 0.0;
                              double v0_val = 0.0;
                              for (int cov_index = 0; cov_index < n_covs_ct; cov_index++) {
                                    const double b_val = baseline_case[class_index](cov_index, test_index);
                                    baseline_loc_prior += b_val * prior_beta_mean[class_index](cov_index, test_index);
                                    const double sd_term = b_val * prior_beta_sd[class_index](cov_index, test_index);
                                    v0_val += sd_term * sd_term;
                              }
                              Eigen::Matrix<stan::math::var, -1, 1> C_centred_var(n_thr_test);
                              for (int thr_index = 0; thr_index < n_thr_test; thr_index++) {
                                    C_centred_var(thr_index) =
                                          cutpoint_4_var[class_index](thr_index, ord_index) - baseline_loc_prior;
                              }
                              {
                                    Eigen::Matrix<stan::math::var, -1, 1> cumul_cen_var(n_thr_test);
                                    for (int thr_index = 0; thr_index < n_thr_test; thr_index++) {
                                          cumul_cen_var(thr_index) = stan::math::Phi(C_centred_var(thr_index));
                                    }
                                    const Eigen::Matrix<stan::math::var, -1, 1> p_ord_cen_var =
                                          cumul_probs_to_ord_probs_var(cumul_cen_var);
                                    target_AD += induced_dirichlet_given_C_lpdf_var(
                                          p_ord_cen_var,
                                          C_centred_var,
                                          alpha_ct.cast<stan::math::var>(),
                                          true
                                    );
                              }
                        }
                  }
            }
      }

      //// -------------------------------------------------------------------------------------
      //// 7. Nuisance transform + Log-scale GHK / CI likelihood
      //// -------------------------------------------------------------------------------------
      const bool is_ci_mode = (CI || (n_us == 0));
      stan::math::var log_jac_u_var = 0.0;
      stan::math::var target_likelihood_var = 0.0;

      Eigen::Matrix<stan::math::var, -1, -1> u_array_var;
      if (!is_ci_mode) {
            Eigen::Matrix<stan::math::var, -1, 1> u_vec_var(n_us);
            const double coeff_a_times_3 = 3.0 * 0.07056;
            const double coeff_b = 1.5976;
            const double half_log_2_pi = 0.5 * std::log(2.0 * M_PI);
            const double log_two = std::log(2.0);
            const double log_rough_scale = std::log(1.702);

            if (nuisance_transformation == "Phi") {
                  for (int elem_index = 0; elem_index < n_us; elem_index++) {
                        const stan::math::var unc_cur = u_unc_var(elem_index);
                        u_vec_var(elem_index) = stan::math::Phi(unc_cur);
                        log_jac_u_var += -half_log_2_pi - 0.5 * stan::math::square(unc_cur);
                  }
            } else if (nuisance_transformation == "Phi_approx") {
                  for (int elem_index = 0; elem_index < n_us; elem_index++) {
                        const stan::math::var unc_cur = u_unc_var(elem_index);
                        const stan::math::var prob_u = stan::math::Phi_approx(unc_cur);
                        u_vec_var(elem_index) = prob_u;
                        log_jac_u_var += stan::math::log(
                              coeff_a_times_3 * stan::math::square(unc_cur) + coeff_b
                        );
                        log_jac_u_var += stan::math::log(prob_u) + stan::math::log1m(prob_u);
                  }
            } else if (nuisance_transformation == "Phi_approx_rough") {
                  for (int elem_index = 0; elem_index < n_us; elem_index++) {
                        const stan::math::var unc_cur = u_unc_var(elem_index);
                        const stan::math::var prob_u = stan::math::inv_logit(1.702 * unc_cur);
                        u_vec_var(elem_index) = prob_u;
                        log_jac_u_var += log_rough_scale
                                         + stan::math::log(prob_u)
                                         + stan::math::log1m(prob_u);
                  }
            } else if (nuisance_transformation == "tanh") {
                  for (int elem_index = 0; elem_index < n_us; elem_index++) {
                        const stan::math::var unc_cur = u_unc_var(elem_index);
                        const stan::math::var tanh_cur = stan::math::tanh(unc_cur);
                        const stan::math::var prob_u = 0.5 * (tanh_cur + 1.0);
                        u_vec_var(elem_index) = prob_u;
                        log_jac_u_var += log_two
                                         + stan::math::log(prob_u)
                                         + stan::math::log1m(prob_u);
                  }
            } else if (nuisance_transformation == "inv_logit") {
                  for (int elem_index = 0; elem_index < n_us; elem_index++) {
                        const stan::math::var unc_cur = u_unc_var(elem_index);
                        const stan::math::var prob_u = stan::math::inv_logit(unc_cur);
                        u_vec_var(elem_index) = prob_u;
                        log_jac_u_var += stan::math::log(prob_u) + stan::math::log1m(prob_u);
                  }
            } else {
                  throw std::invalid_argument(
                        "fn_lp_and_grad_CLC_MVOP_Pinkney_AD_log_scale_InPlace_process: unknown nuisance_transformation '"
                        + nuisance_transformation
                        + "'."
                  );
            }

            if (!prior_only) {
                  u_array_var = Eigen::Matrix<stan::math::var, -1, -1>::Zero(
                        n_obs,
                        n_tests
                  );
                  int u_cursor = 0;
                  for (int chunk_index = 0; chunk_index < chunk_info.n_total_chunks; chunk_index++) {
                        int current_chunk_size = chunk_info.chunk_size_orig;
                        if ((chunk_info.n_total_chunks != chunk_info.n_full_chunks)
                            && (chunk_index == chunk_info.n_full_chunks)) {
                              current_chunk_size = chunk_info.last_chunk_size;
                        }
                        for (int test_index = 0; test_index < n_tests; test_index++) {
                              for (int row_in_chunk = 0; row_in_chunk < current_chunk_size; row_in_chunk++) {
                                    const int row_index = chunk_index * chunk_info.chunk_size_orig + row_in_chunk;
                                    if ((row_index < n_obs) && (u_cursor < n_us)) {
                                          u_array_var(row_index, test_index) = u_vec_var(u_cursor);
                                          u_cursor++;
                                    }
                              }
                        }
                  }
            }
      }

      if (!prior_only) {
            std::vector<Eigen::Matrix<stan::math::var, -1, -1>> Xbeta_lik_var(
                  n_class,
                  Eigen::Matrix<stan::math::var, -1, -1>::Zero(
                        n_obs,
                        n_tests
                  )
            );
            for (int class_index = 0; class_index < n_class; class_index++) {
                  for (int test_index = 0; test_index < n_tests; test_index++) {
                        if (is_perfect_ref(test_index) == 1) {
                              for (int row_index = 0; row_index < n_obs; row_index++) {
                                    Xbeta_lik_var[class_index](row_index, test_index) =
                                          beta_4_var[class_index](0, test_index);
                              }
                        } else {
                              const int n_covs_ct = n_covs_per_outcome(class_index, test_index);
                              if (n_covs_ct > 0) {
                                    const Eigen::Matrix<double, -1, -1> &X_mat = X_class_test[(X_class_test.size() == 1) ? 0 : class_index][(X_class_test[(X_class_test.size() == 1) ? 0 : class_index].size() == 1) ? 0 : test_index];
                                    for (int row_index = 0; row_index < n_obs; row_index++) {
                                          stan::math::var dot_val = 0.0;
                                          for (int cov_index = 0; cov_index < n_covs_ct; cov_index++) {
                                                dot_val += X_mat(row_index, cov_index)
                                                           * beta_4_var[class_index](cov_index, test_index);
                                          }
                                          Xbeta_lik_var[class_index](row_index, test_index) = dot_val;
                                    }
                              }
                        }
                  }
            }

            const bool data_has_missing_results = (y_ref.array() < 0).any();
            const double C_sentinel = 1000.0;
            Eigen::Matrix<stan::math::var, -1, 1> y1_var(n_tests);
            Eigen::Matrix<stan::math::var, -1, 1> lp_class_var(n_class);
            Eigen::Matrix<stan::math::var, -1, 1> Z_std_norm_var(n_tests);

            for (int row_index = 0; row_index < n_obs; row_index++) {
                  for (int class_index = 0; class_index < n_class; class_index++) {
                        if (is_ci_mode) {
                              for (int test_index = 0; test_index < n_tests; test_index++) {
                                    const int y_lo = y_ref(row_index, test_index);
                                    if (data_has_missing_results && (y_lo < 0)) {
                                          y1_var(test_index) = 0.0;
                                          continue;
                                    }
                                    const int ord_index = ord_idx_of_test(test_index);
                                    const stan::math::var &mu_val = Xbeta_lik_var[class_index](row_index, test_index);
                                    if (ord_index < 0) {
                                          const double sign_val = (y_lo == 1) ? 1.0 : -1.0;
                                          y1_var(test_index) = fn_CLC_MVOP_AD_log_Phi_var(
                                                sign_val * mu_val,
                                                use_Phi_approx
                                          );
                                    } else {
                                          const int n_cat_test = n_cat_per_ord_test(ord_index);
                                          const int y_hi = has_y_upper ? y_upper_mat(row_index, test_index) : y_lo;
                                          const bool open_lo = (y_lo == 1);
                                          const bool open_hi = (y_hi == n_cat_test);
                                          if (open_lo && open_hi) {
                                                y1_var(test_index) = 0.0;
                                          } else if (open_lo) {
                                                const stan::math::var upper_bound =
                                                      cutpoint_4_var[class_index](y_hi - 1, ord_index) - mu_val;
                                                y1_var(test_index) = fn_CLC_MVOP_AD_log_Phi_var(
                                                      upper_bound,
                                                      use_Phi_approx
                                                );
                                          } else if (open_hi) {
                                                const stan::math::var lower_bound =
                                                      cutpoint_4_var[class_index](y_lo - 2, ord_index) - mu_val;
                                                y1_var(test_index) = fn_CLC_MVOP_AD_log_Phi_var(
                                                      -lower_bound,
                                                      use_Phi_approx
                                                );
                                          } else {
                                                const stan::math::var lower_bound =
                                                      cutpoint_4_var[class_index](y_lo - 2, ord_index) - mu_val;
                                                const stan::math::var upper_bound =
                                                      cutpoint_4_var[class_index](y_hi - 1, ord_index) - mu_val;
                                                if (lower_bound <= 0.0) {
                                                      const stan::math::var log_phi_lb = fn_CLC_MVOP_AD_log_Phi_var(
                                                            lower_bound,
                                                            use_Phi_approx
                                                      );
                                                      const stan::math::var log_phi_ub = fn_CLC_MVOP_AD_log_Phi_var(
                                                            upper_bound,
                                                            use_Phi_approx
                                                      );
                                                      y1_var(test_index) =
                                                            log_phi_ub + stan::math::log1m_exp(log_phi_lb - log_phi_ub);
                                                } else {
                                                      const stan::math::var log_surv_lb = fn_CLC_MVOP_AD_log_Phi_var(
                                                            -lower_bound,
                                                            use_Phi_approx
                                                      );
                                                      const stan::math::var log_surv_ub = fn_CLC_MVOP_AD_log_Phi_var(
                                                            -upper_bound,
                                                            use_Phi_approx
                                                      );
                                                      y1_var(test_index) =
                                                            log_surv_lb + stan::math::log1m_exp(log_surv_ub - log_surv_lb);
                                                }
                                          }
                                    }
                              }
                        } else {
                              stan::math::var inc_val = 0.0;
                              for (int test_index = 0; test_index < n_tests; test_index++) {
                                    const int y_lo = y_ref(row_index, test_index);
                                    const stan::math::var &u_cur = u_array_var(row_index, test_index);
                                    if (data_has_missing_results && (y_lo < 0)) {
                                          Z_std_norm_var(test_index) = fn_AD_inv_Phi_var(
                                                u_cur,
                                                use_inv_Phi_approx
                                          );
                                          y1_var(test_index) = 0.0;
                                    } else {
                                          const int ord_index = ord_idx_of_test(test_index);
                                          const stan::math::var mu_val =
                                                Xbeta_lik_var[class_index](row_index, test_index) + inc_val;
                                          const stan::math::var L_tt = L_Omega_var[class_index](test_index, test_index);
                                          if (ord_index < 0) {
                                                const stan::math::var Bound_Z_cur = -mu_val / L_tt;
                                                const bool is_overflow =
                                                      (Bound_Z_cur.val() > overflow_threshold) && (y_lo == 1);
                                                const bool is_underflow =
                                                      (Bound_Z_cur.val() < underflow_threshold) && (y_lo == 0);
                                                if (is_overflow) {
                                                      const stan::math::var log_surv = fn_CLC_MVOP_AD_log_Phi_var(
                                                            -Bound_Z_cur,
                                                            use_Phi_approx
                                                      );
                                                      const stan::math::var log_1m_Phi_Z =
                                                            log_surv + stan::math::log1m(u_cur);
                                                      const stan::math::var log_Phi_Z =
                                                            stan::math::log1m_exp(log_1m_Phi_Z);
                                                      Z_std_norm_var(test_index) = fn_AD_inv_Phi_from_log_probs_var(
                                                            log_Phi_Z,
                                                            log_1m_Phi_Z,
                                                            use_inv_Phi_approx
                                                      );
                                                      y1_var(test_index) = log_surv;
                                                } else if (is_underflow) {
                                                      const stan::math::var log_cdf = fn_CLC_MVOP_AD_log_Phi_var(
                                                            Bound_Z_cur,
                                                            use_Phi_approx
                                                      );
                                                      const stan::math::var log_Phi_Z = log_cdf + stan::math::log(u_cur);
                                                      const stan::math::var log_1m_Phi_Z =
                                                            stan::math::log1m_exp(log_Phi_Z);
                                                      Z_std_norm_var(test_index) = fn_AD_inv_Phi_from_log_probs_var(
                                                            log_Phi_Z,
                                                            log_1m_Phi_Z,
                                                            use_inv_Phi_approx
                                                      );
                                                      y1_var(test_index) = log_cdf;
                                                } else {
                                                      const stan::math::var Phi_bound = use_Phi_approx
                                                            ? stan::math::Phi_approx(Bound_Z_cur)
                                                            : stan::math::Phi(Bound_Z_cur);
                                                      if (y_lo == 1) {
                                                            const stan::math::var prob_cur = 1.0 - Phi_bound;
                                                            const stan::math::var Phi_Z_cur = Phi_bound + prob_cur * u_cur;
                                                            Z_std_norm_var(test_index) = fn_AD_inv_Phi_var(
                                                                  Phi_Z_cur,
                                                                  use_inv_Phi_approx
                                                            );
                                                            y1_var(test_index) = stan::math::log(prob_cur);
                                                      } else {
                                                            const stan::math::var prob_cur = Phi_bound;
                                                            const stan::math::var Phi_Z_cur = prob_cur * u_cur;
                                                            Z_std_norm_var(test_index) = fn_AD_inv_Phi_var(
                                                                  Phi_Z_cur,
                                                                  use_inv_Phi_approx
                                                            );
                                                            y1_var(test_index) = stan::math::log(prob_cur);
                                                      }
                                                }
                                          } else {
                                                const int n_cat_test = n_cat_per_ord_test(ord_index);
                                                const int y_hi = has_y_upper ? y_upper_mat(row_index, test_index) : y_lo;
                                                const bool open_lo = (y_lo == 1);
                                                const bool open_hi = (y_hi == n_cat_test);
                                                if (open_lo && open_hi) {
                                                      Z_std_norm_var(test_index) = fn_AD_inv_Phi_var(
                                                            u_cur,
                                                            use_inv_Phi_approx
                                                      );
                                                      y1_var(test_index) = 0.0;
                                                } else {
                                                      const stan::math::var lower_bound = open_lo
                                                            ? stan::math::var(-C_sentinel)
                                                            : ((cutpoint_4_var[class_index](y_lo - 2, ord_index) - mu_val)
                                                               / L_tt);
                                                      const stan::math::var upper_bound = open_hi
                                                            ? stan::math::var(C_sentinel)
                                                            : ((cutpoint_4_var[class_index](y_hi - 1, ord_index) - mu_val)
                                                               / L_tt);
                                                      if (!open_lo && (lower_bound.val() > overflow_threshold)) {
                                                            const stan::math::var log_surv_lb = fn_CLC_MVOP_AD_log_Phi_var(
                                                                  -lower_bound,
                                                                  use_Phi_approx
                                                            );
                                                            const stan::math::var log_surv_ub = open_hi
                                                                  ? stan::math::var(-std::numeric_limits<double>::infinity())
                                                                  : fn_CLC_MVOP_AD_log_Phi_var(
                                                                          -upper_bound,
                                                                          use_Phi_approx
                                                                    );
                                                            const stan::math::var log_prob_cur = open_hi
                                                                  ? log_surv_lb
                                                                  : (log_surv_lb
                                                                     + stan::math::log1m_exp(log_surv_ub - log_surv_lb));
                                                            const stan::math::var log_1m_Phi_Z = open_hi
                                                                  ? (log_surv_lb + stan::math::log1m(u_cur))
                                                                  : stan::math::log_sum_exp(
                                                                          log_surv_ub,
                                                                          log_prob_cur + stan::math::log1m(u_cur)
                                                                    );
                                                            const stan::math::var log_Phi_Z =
                                                                  stan::math::log1m_exp(log_1m_Phi_Z);
                                                            Z_std_norm_var(test_index) = fn_AD_inv_Phi_from_log_probs_var(
                                                                  log_Phi_Z,
                                                                  log_1m_Phi_Z,
                                                                  use_inv_Phi_approx
                                                            );
                                                            y1_var(test_index) = log_prob_cur;
                                                      } else if (!open_hi && (upper_bound.val() < underflow_threshold)) {
                                                            const stan::math::var log_cdf_ub = fn_CLC_MVOP_AD_log_Phi_var(
                                                                  upper_bound,
                                                                  use_Phi_approx
                                                            );
                                                            const stan::math::var log_cdf_lb = open_lo
                                                                  ? stan::math::var(-std::numeric_limits<double>::infinity())
                                                                  : fn_CLC_MVOP_AD_log_Phi_var(
                                                                          lower_bound,
                                                                          use_Phi_approx
                                                                    );
                                                            const stan::math::var log_prob_cur = open_lo
                                                                  ? log_cdf_ub
                                                                  : (log_cdf_ub
                                                                     + stan::math::log1m_exp(log_cdf_lb - log_cdf_ub));
                                                            const stan::math::var log_Phi_Z = open_lo
                                                                  ? (log_cdf_ub + stan::math::log(u_cur))
                                                                  : stan::math::log_sum_exp(
                                                                          log_cdf_lb,
                                                                          log_prob_cur + stan::math::log(u_cur)
                                                                    );
                                                            const stan::math::var log_1m_Phi_Z =
                                                                  stan::math::log1m_exp(log_Phi_Z);
                                                            Z_std_norm_var(test_index) = fn_AD_inv_Phi_from_log_probs_var(
                                                                  log_Phi_Z,
                                                                  log_1m_Phi_Z,
                                                                  use_inv_Phi_approx
                                                            );
                                                            y1_var(test_index) = log_prob_cur;
                                                      } else {
                                                            const stan::math::var Phi_lb = open_lo
                                                                  ? stan::math::var(0.0)
                                                                  : (use_Phi_approx
                                                                     ? stan::math::Phi_approx(lower_bound)
                                                                     : stan::math::Phi(lower_bound));
                                                            const stan::math::var Phi_ub = open_hi
                                                                  ? stan::math::var(1.0)
                                                                  : (use_Phi_approx
                                                                     ? stan::math::Phi_approx(upper_bound)
                                                                     : stan::math::Phi(upper_bound));
                                                            const stan::math::var prob_cur = Phi_ub - Phi_lb;
                                                            const stan::math::var Phi_Z_cur = Phi_lb + prob_cur * u_cur;
                                                            Z_std_norm_var(test_index) = fn_AD_inv_Phi_var(
                                                                  Phi_Z_cur,
                                                                  use_inv_Phi_approx
                                                            );
                                                            y1_var(test_index) = stan::math::log(prob_cur);
                                                      }
                                                }
                                          }
                                    }
                                    if (test_index < n_tests - 1) {
                                          inc_val = 0.0;
                                          for (int prev_index = 0; prev_index <= test_index; prev_index++) {
                                                inc_val += L_Omega_var[class_index](test_index + 1, prev_index)
                                                           * Z_std_norm_var(prev_index);
                                          }
                                    }
                              }
                        }
                        stan::math::var sum_y1 = log_prev_subject_var(row_index, class_index);
                        for (int test_index = 0; test_index < n_tests; test_index++) {
                              sum_y1 += y1_var(test_index);
                        }
                        lp_class_var(class_index) = sum_y1;
                  }
                  const stan::math::var log_lik_row = stan::math::log_sum_exp(lp_class_var);
                  out_mat(1 + n_params + row_index) = log_lik_row.val();
                  target_likelihood_var += log_lik_row;
            }
      }

      target_AD += target_likelihood_var + log_jac_u_var;
      out_mat(0) = target_AD.val();

      if (grad_option != "none") {
            stan::math::grad(target_AD.vi_);
            const bool do_us = (grad_option == "us_only") || (grad_option == "all");
            const bool do_main = (grad_option != "us_only");
            if (do_us) {
                  for (int elem_index = 0; elem_index < n_us; elem_index++) {
                        out_mat(1 + elem_index) = theta_var(elem_index).adj();
                  }
            }
            if (do_main) {
                  for (int elem_index = 0; elem_index < n_params_main; elem_index++) {
                        out_mat(1 + n_us + elem_index) = theta_var(n_us + elem_index).adj();
                  }
            }
      }
}
