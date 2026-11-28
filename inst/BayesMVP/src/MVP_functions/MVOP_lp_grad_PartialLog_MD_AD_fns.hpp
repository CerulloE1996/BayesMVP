

//// MVOP_lp_grad_PartialLog_MD_AD_fns.hpp
////
//// Log-scale ("PartialLog") lp/grad for the LC-MVOP / MVOP model - the MVOP equivalent
//// of fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process.
////
//// DESIGN (differs from the LC-MVP PartialLog internals - see helper header for rationale):
////   1. LIKELIHOOD: standard-scale pass everywhere via the usual GHK helpers, then per-test
////      "problem" masks (binary: same Bound_Z/y masks as the MVP PartialLog; ordinal: both
////      truncation bounds deep in the SAME tail, i.e. lb > overflow_threshold or
////      ub < underflow_threshold). Problem entries are recomputed EXACTLY on the log scale:
////      binary -> the existing fn_MVP_compute_lp_GHK_cols_log_scale_underflow/_overflow;
////      ordinal -> fn_MVOP_compute_lp_GHK_cols_log_scale_ordinal (new, in the helper header).
////   2. GRADIENTS: after fn_MVOP_grad_prep, ALL standard-scale gradient inputs are ZEROED on
////      the problem rows (union over tests, plus rows whose TOTAL log-lik underflowed), so the
////      standard-scale grad fns contribute exactly 0 for those rows. The exact contributions of
////      the problem rows are then ADDED by the scalar per-row signed-log engine
////      (fn_MVOP_row_grads_log_scale) - no double counting, no Inf/NaN contamination.
////   3. CHUNKING: ONE shared per-chunk routine. Full chunks receive the SIMD-configured
////      Model_args struct; the remainder chunk receives a "Stan" (scalar) copy
////      ("Model_args_last_chunk") - i.e. the remainder is genuinely scalar end-to-end,
////      matching the fixed NoLog files.
////
//// REQUIREMENTS:
////   - MVOP_lp_grad_PartialLog_helper_fns.hpp  (SignedLog utils + ordinal log-scale lp handler
////     + fn_MVOP_row_grads_log_scale)  - include BEFORE this file.
////   - The MVP log-scale lp handlers (fn_MVP_compute_lp_GHK_cols_log_scale_underflow/_overflow).
////   - stan-math >= 4.3 (std_normal_log_qf, used by the ordinal log-scale handler).
////
//// NOTE: serial only for now - the WCP (within-chunk-parallel) PartialLog variant is TODO.
//// The LC_MVP_ws_structs argument is kept for interface parity but is UNUSED (this fn uses
//// fully local containers, like the MVP PartialLog).

#pragma once


#include <Eigen/Dense>

#include <unsupported/Eigen/SpecialFunctions>

#include "PartialLog_nuisance_chunk_relayout.hpp"
#include "MVP_and_MVOP_missing_results_GHK_fns.hpp"   //// missing test results (y = -1) by the GHK method



//////////////////////////////////////////////////------------------------------------------------------------------------------------------------------------------------
//// Internal per-chunk routine (lp + grad for ONE chunk). Called once per full chunk with the
//// SIMD struct, and once for the remainder chunk with the "Stan" (scalar) struct copy.



//// MVOP_lp_grad_PartialLog_MD_AD_fns_T.hpp - replaces the process_chunk + _process_serial in
//// MVOP_lp_grad_PartialLog_MD_AD_fns.hpp
////
//// Keep in the file: fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_Inplace_process (the dispatcher)
//// and the InPlace wrappers. They call the non-template ..._Inplace_process_serial at the bottom.
////
//// Requires: MVOP_lp_grad_common_T.hpp, MVP_log_scale_grad_calc_fns_T.hpp (binary log-scale fix-ups _T),
//// migration_MVOP_helpers_and_MVP_driver.hpp (fn_MVOP_compute_lp_GHK_cols_T), and the unchanged
//// MVOP_PartialLog_helper_fns.hpp (fn_MVOP_compute_lp_GHK_cols_log_scale_ordinal, fn_MVOP_row_grads_log_scale,
//// both of which take the scalar flag S; here S = (vec == Vec::Scalar)).

// #pragma once
// #include "MVOP_lp_grad_common_T.hpp"
// #include "MVP_log_scale_grad_calc_fns_T.hpp"

template <Vec vec>
inline void fn_lp_grad_MVOP_LC_Pinkney_PartialLog_process_chunk_T(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                    const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                    const std::string &grad_option,
                                                                    const int chunk_counter,
                                                                    const int chunk_size,
                                                                    const MVOP_prep &P,
                                                                    double &log_jac_u,
                                                                    std::vector<Eigen::Matrix<double, -1, -1>> &beta_grad_array,
                                                                    std::vector<Eigen::Matrix<double, -1, -1>> &U_Omega_grad_array,
                                                                    std::vector<Eigen::Matrix<double, -1, -1>> &cutpoint_grad_array,
                                                                    Eigen::Matrix<double, -1, -1> &prev_grad_mat,
                                                                    const Model_fn_args_struct &Model_args_as_cpp_struct
) {

        const double sqrt_2_pi_recip = 1.0 / std::sqrt(2.0 * M_PI);
        const double Inf = std::numeric_limits<double>::infinity();
        const bool S = (vec == Vec::Scalar);   //// scalar-Stan flag for the ordinal log-scale helpers
        const int N = P.N;
        const int n_params = P.n_params;
        const int n_tests = P.n_tests;
        const int n_class = P.n_class;
        const int n_pops = P.n_pops;
        const int n_covariates_max = P.n_covariates_max;
        const int row_start = P.chunk_size_orig * chunk_counter;
        const double overflow_threshold = P.overflow_threshold;
        const double underflow_threshold = P.underflow_threshold;
        const KernelChoice &k = P.kchoice;
        const Eigen::Matrix<int, -1, 1> &ord_idx_of_test = P.ord_idx_of_test;
        const Eigen::Matrix<int, -1, -1> &ncov = *P.n_covariates_per_outcome_vec;
        const Eigen::Matrix<int, -1, 1> &n_cat_per_ord_test = *P.n_cat_per_ord_test;
        const Eigen::Matrix<int, -1, 1> &n_thr_per_ord_test = *P.n_thr_per_ord_test;
        const Eigen::Matrix<int, -1, 1> &pop_ind = *P.pop_ind;
        const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> &X = *P.X;
        const std::vector<Eigen::Matrix<double, -1, -1>> &beta = P.beta_double_array;
        const std::vector<Eigen::Matrix<double, -1, -1>> &C = P.C;
        const std::vector<Eigen::Matrix<double, -1, -1>> &L = P.L_Omega_double;
        const std::vector<Eigen::Matrix<double, -1, -1>> &Lr = P.L_Omega_recip_double;
        const bool do_us = (grad_option == "us_only") || (grad_option == "all");
        const bool do_coeff = (grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only");
        const bool do_corr  = (grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only");
        const bool do_cut   = (grad_option == "main_only") || (grad_option == "all") || (grad_option == "cutpoints_only");
        const bool do_prev  = (grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only");

        //// ---- local containers ----
        std::vector<Eigen::Matrix<double, -1, -1>> Z_std_norm = vec_of_mats<double>(chunk_size, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> Bound_Z = Z_std_norm;
        std::vector<Eigen::Matrix<double, -1, -1>> Upper_Bound_Z = Z_std_norm;
        std::vector<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z = Z_std_norm;
        std::vector<Eigen::Matrix<double, -1, -1>> Phi_Z = Z_std_norm;
        std::vector<Eigen::Matrix<double, -1, -1>> prob = Z_std_norm;
        std::vector<Eigen::Matrix<double, -1, -1>> y1_log_prob = Z_std_norm;
        std::vector<Eigen::Matrix<double, -1, -1>> log_Z_std_norm = Z_std_norm;
        std::vector<Eigen::Matrix<double, -1, -1>> log_phi_Bound_Z = Z_std_norm;
        std::vector<Eigen::Matrix<double, -1, -1>> log_phi_Z_recip = Z_std_norm;
        Eigen::Matrix<double, -1, -1> y_chunk = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
        Eigen::Matrix<double, -1, -1> u_array = y_chunk;
        Eigen::Matrix<double, -1, -1> y_sign = y_chunk;
        Eigen::Matrix<double, -1, -1> y_m_y_sign_x_u = y_chunk;
        Eigen::Matrix<double, -1, -1> phi_Z_recip = y_chunk;
        Eigen::Matrix<double, -1, -1> phi_Bound_Z = y_chunk;
        Eigen::Matrix<double, -1, -1> phi_Upper_Bound_Z = y_chunk;
        Eigen::Matrix<double, -1, -1> common_grad_term_1 = y_chunk;
        Eigen::Matrix<double, -1, -1> prob_rowwise_prod_temp = y_chunk;
        Eigen::Matrix<double, -1, -1> prob_recip_rowwise_prod_temp = y_chunk;
        Eigen::Matrix<double, -1, 1> prob_rowwise_prod_temp_all = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
        Eigen::Matrix<double, -1, -1> dphi_over_L = y_chunk;
        Eigen::Matrix<double, -1, -1> dZ_dmu_neg = y_chunk;
        Eigen::Matrix<double, -1, -1> dphi_times_bz = y_chunk;
        Eigen::Matrix<double, -1, -1> dZ_times_bz = y_chunk;
        Eigen::Matrix<double, -1, -1> grad_prob = y_chunk;
        Eigen::Matrix<double, -1, -1> z_grad_term = y_chunk;
        Eigen::Matrix<double, -1, -1> prob_recip = y_chunk;
        Eigen::Matrix<double, -1, 1> prod_container_or_inc_array = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
        Eigen::Matrix<double, -1, 1> derivs_chain_container_vec = prod_container_or_inc_array;
        Eigen::Matrix<double, -1, 1> dphi_direct_Cj = prod_container_or_inc_array;
        Eigen::Matrix<double, -1, 1> dZ_direct_Cj = prod_container_or_inc_array;
        Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk = y_chunk;
        Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk_block = y_chunk;
        Eigen::Matrix<double, -1, 1> u_unc_vec_chunk = Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
        Eigen::Matrix<double, -1, 1> u_vec_chunk = u_unc_vec_chunk;
        Eigen::Matrix<double, -1, 1> du_wrt_duu_chunk = u_unc_vec_chunk;
        Eigen::Matrix<double, -1, 1> d_J_wrt_duu_chunk = u_unc_vec_chunk;
        Eigen::Matrix<double, -1, -1> lp_array = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_class);
        Eigen::Matrix<double, -1, 1> log_sum_result = Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -INFINITY);
        Eigen::Matrix<double, -1, 1> container_max_logs = log_sum_result;
        Eigen::Matrix<double, -1, 1> rowwise_sum = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
        Eigen::Matrix<double, -1, 1> rowwise_prod = rowwise_sum;
        Eigen::Matrix<double, -1, 1> log_lik_chunk = rowwise_sum;
        Eigen::Matrix<double, -1, 1> prob_n = rowwise_sum;
        Eigen::Matrix<double, -1, 1> prob_n_recip = rowwise_sum;
        Eigen::Matrix<double, -1, 1> log_prob_n_recip = rowwise_sum;
        Eigen::Matrix<double, -1, 1> log_prev_per_obs_given_c = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
        Eigen::Matrix<double, -1, 1> prev_per_obs_given_c = log_prev_per_obs_given_c;
        std::vector<std::vector<char>> row_is_problem(n_class, std::vector<char>(chunk_size, 0));

        //// ---- nuisance ----
        y_chunk = y_ref.middleRows(row_start, chunk_size).cast<double>();
        u_unc_vec_chunk = theta_us_vec_ref.segment(row_start * n_tests, chunk_size * n_tests);
        fn_MVP_compute_nuisance_T<vec>( u_vec_chunk,
                                        u_unc_vec_chunk,
                                        k);
        log_jac_u += fn_MVP_compute_nuisance_log_jac_u_T<vec>( u_vec_chunk,
                                                               u_unc_vec_chunk,
                                                               d_J_wrt_duu_chunk,
                                                               k);
        u_array = u_vec_chunk.reshaped(chunk_size, n_tests);
        //// ---- missing test results (y = -1), by the GHK method (MVP_and_MVOP_missing_results_GHK_fns.hpp),
        ////      only when the chunk has one: data without missing results run the code below unchanged.
        const bool chunk_has_missing_results = fn_MVP_rows_of_y_have_a_missing_result( y_ref,
                                                                                       row_start,
                                                                                       chunk_size);
        Eigen::Matrix<double, -1, -1> is_missing_result;
        Eigen::Matrix<double, -1, -1> Z_untruncated;   //// empty unless the chunk has one
        if (chunk_has_missing_results) {
          fn_MVP_missing_result_indicator_and_recode_missing_results_to_y_equals_1( y_chunk,
                                                                                    is_missing_result);
          fn_MVP_compute_untruncated_GHK_Z_of_missing_results_T<vec>( Z_untruncated,
                                                                      u_array,
                                                                      is_missing_result,
                                                                      k);
        }
        y_sign.array() = 2.0 * y_chunk.array() - 1.0;
        y_m_y_sign_x_u.array() = y_chunk.array() - (y_sign.array() * u_array.array());

        //// ---- likelihood ----
        for (int c = 0; c < n_class; c++) {
          if (n_class > 1) for (int n = 0; n < chunk_size; n++) { const int g = pop_ind(row_start + n); log_prev_per_obs_given_c(n) = P.log_prev_mat_small(g, c); prev_per_obs_given_c(n) = P.prev_mat(g, c); }
          prod_container_or_inc_array.setZero();
          for (int t = 0; t < n_tests; t++) {
            const int t_ord = ord_idx_of_test(t);
            const double L_tt_inv = Lr[c](t, t);
            Eigen::Matrix<double, -1, 1> mu(chunk_size);
            //// one matrix in the list means the same covariates for every class / test
            const int c_X = (X.size() == 1) ? 0 : c;
            const int t_X = (X[c_X].size() == 1) ? 0 : t;
            if (n_covariates_max > 1) mu = X[c_X][t_X].block(row_start,
                0,
                chunk_size,
                ncov(c, t)).cast<double>() * beta[c].col(t).head(ncov(c, t)) + prod_container_or_inc_array;
            else                      mu.array() = beta[c](0, t) + prod_container_or_inc_array.array();
            std::vector<int> over_index;
            std::vector<int> under_index;
            std::vector<int> ord_problem_index;
            if (t_ord < 0) {
              Bound_Z[c].col(t).array() = -L_tt_inv * mu.array();
              if (chunk_has_missing_results) {
                //// the loop below, with a missing result never a tail entry (its prob is exactly 1):
                for (int n = 0; n < chunk_size; n++) {
                  if (is_missing_result(n, t) > 0.5) continue;
                  const double BZ = Bound_Z[c](n, t);
                  const double yn = y_chunk(n, t);
                  if ((BZ > overflow_threshold) && (yn == 1.0)) {
                    over_index.push_back(n);  row_is_problem[c][n] = 1;
                  } else if ((BZ < underflow_threshold) && (yn == 0.0)) {
                    under_index.push_back(n); row_is_problem[c][n] = 1;
                  }
                }
              } else {
              for (int n = 0; n < chunk_size; n++) {
                const double BZ = Bound_Z[c](n, t);
                const double yn = y_chunk(n, t);
                if      ((BZ > overflow_threshold)  && (yn == 1.0)) { over_index.push_back(n);  row_is_problem[c][n] = 1; }
                else if ((BZ < underflow_threshold) && (yn == 0.0)) { under_index.push_back(n); row_is_problem[c][n] = 1; }
              }
              }
            } else {
              const int K_t = n_cat_per_ord_test(t_ord);
              if (chunk_has_missing_results) {
                //// the loop below, with both bounds of a missing result infinite (the untruncated draw), so that
                //// it is never a tail entry:
                for (int n = 0; n < chunk_size; n++) {
                  if (is_missing_result(n, t) > 0.5) {
                    Bound_Z[c](n, t) = -Inf;  Upper_Bound_Z[c](n, t) = Inf;
                    continue;
                  }
                  const int y_n = static_cast<int>(y_chunk(n, t));
                  const double lb = (y_n == 1)   ? -Inf : (C[c](y_n - 2, t_ord) - mu(n)) * L_tt_inv;
                  const double ub = (y_n == K_t) ?  Inf : (C[c](y_n - 1, t_ord) - mu(n)) * L_tt_inv;
                  Bound_Z[c](n, t) = lb; Upper_Bound_Z[c](n, t) = ub;
                  if ((lb > overflow_threshold) || (ub < underflow_threshold)) {
                    ord_problem_index.push_back(n); row_is_problem[c][n] = 1;
                  }
                }
              } else {
              for (int n = 0; n < chunk_size; n++) {
                const int y_n = static_cast<int>(y_chunk(n, t));
                const double lb = (y_n == 1)   ? -Inf : (C[c](y_n - 2, t_ord) - mu(n)) * L_tt_inv;
                const double ub = (y_n == K_t) ?  Inf : (C[c](y_n - 1, t_ord) - mu(n)) * L_tt_inv;
                Bound_Z[c](n, t) = lb; Upper_Bound_Z[c](n, t) = ub;
                if ((lb > overflow_threshold) || (ub < underflow_threshold)) { ord_problem_index.push_back(n); row_is_problem[c][n] = 1; }
              }
              }
            }
            //// standard-scale pass, all rows
            if (t_ord < 0) fn_MVP_compute_lp_GHK_cols_T<vec>(t,
                Bound_U_Phi_Bound_Z[c],
                Phi_Z[c],
                Z_std_norm[c],
                prob[c],
                y1_log_prob[c],
                Bound_Z[c],
                y_chunk,
                u_array,
                k);
            else           fn_MVOP_compute_lp_GHK_cols_T<vec>(t,
                                                              Bound_U_Phi_Bound_Z[c],
                                                              Phi_Z[c],
                                                              Z_std_norm[c],
                                                              prob[c],
                                                              y1_log_prob[c],
                                                              Bound_Z[c],
                                                              y_chunk,
                                                              u_array,
                                                              Upper_Bound_Z[c],
                                                              false,
                                                              k);
            if (chunk_has_missing_results) {
              fn_MVP_set_GHK_col_of_missing_results_to_the_untruncated_draw( t,
                                                                             is_missing_result,
                                                                             u_array,
                                                                             Z_untruncated,
                                                                             Bound_U_Phi_Bound_Z[c],
                                                                             Phi_Z[c],
                                                                             Z_std_norm[c],
                                                                             prob[c],
                                                                             y1_log_prob[c]);
            }
            //// log-scale fix-ups of the problem entries
            if (t_ord < 0) {
              if (!under_index.empty())
                fn_MVP_compute_lp_GHK_cols_log_scale_underflow_T<vec>( t,
                                                                       under_index,
                                                                       Bound_U_Phi_Bound_Z[c],
                                                                       Phi_Z[c],
                                                                       Z_std_norm[c],
                                                                       log_Z_std_norm[c],
                                                                       prob[c],
                                                                       y1_log_prob[c],
                                                                       log_phi_Bound_Z[c],
                                                                       log_phi_Z_recip[c],
                                                                       Bound_Z[c],
                                                                       u_array,
                                                                       k);   //// exact tails when Phi_type = "Phi"
              if (!over_index.empty())
                fn_MVP_compute_lp_GHK_cols_log_scale_overflow_T<vec>( t,
                                                                      (int)over_index.size(),
                                                                      over_index,
                                                                      Bound_U_Phi_Bound_Z[c],
                                                                      Phi_Z[c],
                                                                      Z_std_norm[c],
                                                                      log_Z_std_norm[c],
                                                                      prob[c],
                                                                      y1_log_prob[c],
                                                                      log_phi_Bound_Z[c],
                                                                      log_phi_Z_recip[c],
                                                                      Bound_Z[c],
                                                                      u_array,
                                                                      k);   //// exact tails when Phi_type = "Phi"
            } else if (!ord_problem_index.empty()) {
              fn_MVOP_compute_lp_GHK_cols_log_scale_ordinal(t,
                                                            ord_problem_index,
                                                            Bound_U_Phi_Bound_Z[c],
                                                            Phi_Z[c],
                                                            Z_std_norm[c],
                                                            prob[c],
                                                            y1_log_prob[c],
                                                            Bound_Z[c],
                                                            Upper_Bound_Z[c],
                                                            u_array,
                                                            S,
                                                            
                                                            k);   //// exact tails when Phi_type = "Phi"
            }
            if (t < n_tests - 1) prod_container_or_inc_array = Z_std_norm[c].leftCols(t + 1) * L[c].row(t + 1).head(t + 1).transpose();
          }
          rowwise_sum = y1_log_prob[c].rowwise().sum();
          if (n_class > 1) { rowwise_sum.array() += log_prev_per_obs_given_c.array(); lp_array.col(c) = rowwise_sum; } else lp_array.col(0) = rowwise_sum;
        }
        const int index_start = 1 + n_params + row_start;
        if (n_class > 1) { log_sum_exp_general_T<vec>(lp_array, log_sum_result, container_max_logs); out_mat.segment(index_start, chunk_size) = log_sum_result; }
        else               out_mat.segment(index_start, chunk_size) = lp_array.col(0);
        log_lik_chunk = out_mat.segment(index_start, chunk_size);
        prob_n = log_lik_chunk; apply_inplace<vec, Fn::exp>(prob_n);
        prob_n_recip = stan::math::inv(prob_n);
        log_prob_n_recip = -log_lik_chunk;
        for (int n = 0; n < chunk_size; n++) if (log_lik_chunk(n) < -690.0) { prob_n_recip(n) = 0.0; for (int c2 = 0; c2 < n_class; c2++) row_is_problem[c2][n] = 1; }

        //// ---- gradients ----
        for (int c = 0; c < n_class; c++) {
          if (n_class > 1) for (int n = 0; n < chunk_size; n++) { const int g = pop_ind(row_start + n); prev_per_obs_given_c(n) = P.prev_mat(g, c); log_prev_per_obs_given_c(n) = P.log_prev_mat_small(g, c); }
          Eigen::Matrix<int, -1, 1> n_cov_vec_c = ncov.row(c).transpose();
          prob_recip = stan::math::inv(prob[c]);
          for (int t = 0; t < n_tests; t++) {
            fn_MVP_compute_phi_Z_recip_cols_T<vec>( t,
                                                    phi_Z_recip,
                                                    Phi_Z[c],
                                                    Z_std_norm[c],
                                                    k);
            if (ord_idx_of_test(t) < 0) {
              fn_MVP_compute_phi_Bound_Z_cols_T<vec>( t,
                                                      phi_Bound_Z,
                                                      Bound_U_Phi_Bound_Z[c],
                                                      Bound_Z[c],
                                                      k);
              if (chunk_has_missing_results) {
                fn_MVP_set_phi_Bound_Z_col_of_missing_results_to_zero( t,
                                                                       is_missing_result,
                                                                       phi_Bound_Z);
              }
            } else {
              if (k.Phi_approx) {
                //// d/dx Phi_approx(x) = (3a x^2 + b) s (1 - s), with s = Phi_approx(x), a = 0.07056, b = 1.5976:
                phi_Bound_Z.col(t) = Bound_Z[c].col(t);             Phi_col_inplace<vec>(phi_Bound_Z, t, k);
                // phi_Bound_Z.col(t).array() = (3.0 * 0.07056 * Bound_Z[c].col(t).array().square() + 1.5976) * phi_Bound_Z.col(t).array() * (1.0 - phi_Bound_Z.col(t).array());
                phi_Upper_Bound_Z.col(t) = Upper_Bound_Z[c].col(t); Phi_col_inplace<vec>(phi_Upper_Bound_Z, t, k);
                // phi_Upper_Bound_Z.col(t).array() = (3.0 * 0.07056 * Upper_Bound_Z[c].col(t).array().square() + 1.5976) * phi_Upper_Bound_Z.col(t).array() * (1.0 - phi_Upper_Bound_Z.col(t).array());
                //// (with 1 - s formed directly, as s(-x) = Phi_approx(-x): 1 - s is exactly 0 once x > ~7.1,
                ////  where s rounds to 1)
                Eigen::Matrix<double, -1, -1> Phi_approx_of_minus_the_bounds(Bound_Z[c].rows(), 2);
                Phi_approx_of_minus_the_bounds.col(0) = -Bound_Z[c].col(t);
                Phi_approx_of_minus_the_bounds.col(1) = -Upper_Bound_Z[c].col(t);
                Phi_col_inplace<vec>(Phi_approx_of_minus_the_bounds, 0, k);
                Phi_col_inplace<vec>(Phi_approx_of_minus_the_bounds, 1, k);
                phi_Bound_Z.col(t).array() = (3.0 * 0.07056 * Bound_Z[c].col(t).array().square() + 1.5976)
                                             * phi_Bound_Z.col(t).array()
                                             * Phi_approx_of_minus_the_bounds.col(0).array();
                phi_Upper_Bound_Z.col(t).array() = (3.0 * 0.07056 * Upper_Bound_Z[c].col(t).array().square()
                                                    + 1.5976)
                                                   * phi_Upper_Bound_Z.col(t).array()
                                                   * Phi_approx_of_minus_the_bounds.col(1).array();
              } else {
                phi_Bound_Z.col(t).array() = -0.5 * Bound_Z[c].col(t).array().square();             apply_col_inplace<vec, Fn::exp>(phi_Bound_Z, t);       phi_Bound_Z.col(t).array() *= sqrt_2_pi_recip;
                phi_Upper_Bound_Z.col(t).array() = -0.5 * Upper_Bound_Z[c].col(t).array().square(); apply_col_inplace<vec, Fn::exp>(phi_Upper_Bound_Z, t); phi_Upper_Bound_Z.col(t).array() *= sqrt_2_pi_recip;
              }
           // for (int i = 0; i < chunk_size; ++i) { if (std::isinf(Bound_Z[c](i, t))) phi_Bound_Z(i, t) = 0.0; if (std::isinf(Upper_Bound_Z[c](i, t))) phi_Upper_Bound_Z(i, t) = 0.0; }
              //// the loop above (commented out) as column-wise selects: phi(bound) = 0 for an infinite bound:
              phi_Bound_Z.col(t).array() = Bound_Z[c].col(t).array().isInf()
                                           .select(0.0, phi_Bound_Z.col(t).array());
              phi_Upper_Bound_Z.col(t).array() = Upper_Bound_Z[c].col(t).array().isInf()
                                                 .select(0.0, phi_Upper_Bound_Z.col(t).array());
            }
          }
          std::vector<int> P_c;
          if (grad_option != "none") {
            P_c.reserve(chunk_size);
            for (int n = 0; n < chunk_size; n++) if (row_is_problem[c][n] == 1) P_c.push_back(n);
            fn_MVOP_grad_prep( prob[c],
                               y_sign,
                               y_m_y_sign_x_u,
                               u_array,
                               Lr[c],
                               prev_per_obs_given_c,
                               prob_n_recip,
                               phi_Z_recip,
                               phi_Bound_Z,
                               phi_Upper_Bound_Z,
                               Bound_Z[c],
                               Upper_Bound_Z[c],
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
                               Model_args_as_cpp_struct);
            for (std::size_t pi = 0; pi < P_c.size(); pi++) {
              const int n = P_c[pi];
              common_grad_term_1.row(n).setZero(); prob_recip.row(n).setZero(); prob_rowwise_prod_temp.row(n).setZero(); prob_recip_rowwise_prod_temp.row(n).setZero();
              prob_rowwise_prod_temp_all(n) = 0.0; dphi_over_L.row(n).setZero(); dZ_dmu_neg.row(n).setZero(); dphi_times_bz.row(n).setZero(); dZ_times_bz.row(n).setZero();
              phi_Z_recip.row(n).setZero(); prob[c].row(n).setZero();
            }
          }
          //// ---- (debug output of the non-finite gradient terms with Phi_approx, commented out: begin)
          // if (k.Phi_approx) {
                // static int n_debug_prints_1 = 0;
                // for (int n = 0; (n < chunk_size) && (n_debug_prints_1 < 3); ++n) {
                      // if (row_is_problem[c][n] == 1) continue;
                      // bool all_finite = true;
                      // for (int t = 0; t < n_tests; ++t) {
                            // all_finite = all_finite && std::isfinite(common_grad_term_1(n, t))
                                         // && std::isfinite(prob_recip(n, t))
                                         // && std::isfinite(phi_Z_recip(n, t))
                                         // && std::isfinite(dphi_over_L(n, t))
                                         // && std::isfinite(dZ_dmu_neg(n, t))
                                         // && std::isfinite(dphi_times_bz(n, t))
                                         // && std::isfinite(dZ_times_bz(n, t))
                                         // && std::isfinite(prob[c](n, t));
                      // }
                      // if (all_finite) continue;
                      // n_debug_prints_1++;
                      // std::cout << "DEBUG standard terms, class " << c << " row " << n
                                // << " prob_n_recip " << prob_n_recip(n) << std::endl;
                      // for (int t = 0; t < n_tests; ++t) {
                            // std::cout << "   t " << t << " y " << y_chunk(n, t)
                                      // << " lb " << Bound_Z[c](n, t)
                                      // << " ub " << Upper_Bound_Z[c](n, t) << " prob " << prob[c](n, t)
                                      // << " y1_log_prob " << y1_log_prob[c](n, t)
                                      // << " Z " << Z_std_norm[c](n, t)
                                      // << " phi_Z_recip " << phi_Z_recip(n, t)
                                      // << " phi_lb " << phi_Bound_Z(n, t)
                                      // << " phi_ub " << phi_Upper_Bound_Z(n, t)
                                      // << " common " << common_grad_term_1(n, t)
                                      // << " prob_recip " << prob_recip(n, t)
                                      // << " dphi_over_L " << dphi_over_L(n, t)
                                      // << " dZ_dmu_neg " << dZ_dmu_neg(n, t)
                                      // << " dphi_times_bz " << dphi_times_bz(n, t)
                                      // << " dZ_times_bz " << dZ_times_bz(n, t)
                                      // << " u " << u_array(n, t) << std::endl;
                      // }
                // }
          // }
          //// ---- (debug output: end)
          if (do_us) {
            u_grad_array_CM_chunk_block = u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests);
            fn_MVP_compute_nuisance_grad_v2( u_grad_array_CM_chunk_block,
                                             phi_Z_recip,
                                             common_grad_term_1,
                                             L[c],
                                             prob[c],
                                             prob_recip,
                                             prob_rowwise_prod_temp,
                                             dphi_over_L,
                                             dZ_dmu_neg,
                                             z_grad_term,
                                             grad_prob,
                                             prod_container_or_inc_array,
                                             derivs_chain_container_vec,
                                             Model_args_as_cpp_struct);
            u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests).array() += u_grad_array_CM_chunk_block.array();
          }
          //// ---- (debug output of the non-finite gradient terms with Phi_approx, commented out: begin)
          // if (k.Phi_approx && do_us) {
                // static int n_debug_prints_2 = 0;
                // for (int n = 0; (n < chunk_size) && (n_debug_prints_2 < 3); ++n) {
                      // if (row_is_problem[c][n] == 1) continue;
                      // bool all_finite = true;
                      // for (int t = 0; t < n_tests; ++t) {
                            // all_finite = all_finite && std::isfinite(u_grad_array_CM_chunk(n, t));
                      // }
                      // if (all_finite) continue;
                      // n_debug_prints_2++;
                      // std::cout << "DEBUG u gradient not finite after the standard terms, class " << c
                                // << " row " << n << ":";
                      // for (int t = 0; t < n_tests; ++t) std::cout << " " << u_grad_array_CM_chunk(n, t);
                      // std::cout << std::endl;
                // }
          // }
          //// ---- (debug output: end)
          if (do_coeff) {
            fn_MVP_compute_coefficients_grad_v3( c,
                                                 beta_grad_array[c],
                                                 X[(X.size() == 1) ? 0 : c],
                                                 n_cov_vec_c,
                                                 row_start,
                                                 n_covariates_max,
                                                 common_grad_term_1,
                                                 L[c],
                                                 prob[c],
                                                 prob_recip,
                                                 prob_rowwise_prod_temp,
                                                 dphi_over_L,
                                                 dZ_dmu_neg,
                                                 z_grad_term,
                                                 grad_prob,
                                                 prod_container_or_inc_array,
                                                 derivs_chain_container_vec,
                                                 true,
                                                 Model_args_as_cpp_struct);
          }
          if (do_corr) {
            fn_MOVP_compute_L_Omega_grad_v3( U_Omega_grad_array[c],
                                             common_grad_term_1,
                                             L[c],
                                             prob[c],
                                             prob_recip,
                                             Z_std_norm[c],
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
                                             Model_args_as_cpp_struct);
          }
          if (do_cut) {
            for (int t = 0; t < n_tests; t++) {
              const int t_ord = ord_idx_of_test(t); if (t_ord < 0) continue;
           // fn_MVOP_compute_cutpoint_grad(c, cutpoint_grad_array[c], t, t_ord, n_thr_per_ord_test(t_ord) + 1, common_grad_term_1, L[c], prob[c], prob_recip, prob_rowwise_prod_temp,
                                         // phi_Bound_Z, phi_Upper_Bound_Z, phi_Z_recip, u_array, y_chunk, dphi_over_L, dZ_dmu_neg, z_grad_term, grad_prob, prod_container_or_inc_array,
                                         // derivs_chain_container_vec, dphi_direct_Cj, dZ_direct_Cj, true, Model_args_as_cpp_struct);
              //// one propagation per test instead of one per cutpoint (the call above, commented out):
              fn_MVOP_compute_cutpoint_grad_from_one_unit_dZ_seed_propagation_and_category_scatter( c,
                                                                                                    cutpoint_grad_array[c],
                                                                                                    t,
                                                                                                    t_ord,
                                                                                                    n_thr_per_ord_test(t_ord) + 1,
                                                                                                    common_grad_term_1,
                                                                                                    L[c],
                                                                                                    prob[c],
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
                                                                                                    Model_args_as_cpp_struct);
            }
          }
          //// exact contributions of the problem rows (their std-scale contributions above were zero)
          if ((grad_option != "none") && !P_c.empty() && (do_us || do_coeff || do_corr || do_cut)) {
            for (std::size_t pi = 0; pi < P_c.size(); pi++) {
              const int n = P_c[pi];
              const double log_prev_nc = (n_class > 1) ? log_prev_per_obs_given_c(n) : 0.0;
              fn_MVOP_row_grads_log_scale( n,
                                           row_start + n,
                                           c,
                                           do_us,
                                           do_coeff,
                                           do_corr,
                                           do_cut,
                                           n_tests,
                                           ord_idx_of_test,
                                           n_class,
                                           n_covariates_max,
                                           n_cov_vec_c,
                                           n_cat_per_ord_test,
                                           X[(X.size() == 1) ? 0 : c],
                                           y_chunk,
                                           u_array,
                                           Bound_Z[c],
                                           Upper_Bound_Z[c],
                                           Z_std_norm[c],
                                           y1_log_prob[c],
                                           L[c],
                                           log_prev_nc,
                                           log_prob_n_recip(n),
                                           u_grad_array_CM_chunk,
                                           beta_grad_array[c],
                                           U_Omega_grad_array[c],
                                           cutpoint_grad_array[c],
                                           S,
                                           k.Phi_approx,
                                           k.inv_Phi_approx,
                                           chunk_has_missing_results ? &is_missing_result : nullptr);
              //// ---- (debug output of the non-finite gradient terms with Phi_approx, commented out: begin)
              // if (k.Phi_approx) {
                    // static int n_debug_prints_3 = 0;
                    // bool all_finite = true;
                    // for (int t = 0; t < n_tests; ++t) all_finite = all_finite
                                                               // && std::isfinite(u_grad_array_CM_chunk(n, t));
                    // if ((all_finite == false) && (n_debug_prints_3 < 3)) {
                          // n_debug_prints_3++;
                          // std::cout << "DEBUG problem row u gradient not finite, class " << c << " row " << n
                                    // << " log_prob_n_recip " << log_prob_n_recip(n) << " log_prev " << log_prev_nc
                                    // << std::endl;
                          // for (int t = 0; t < n_tests; ++t) {
                                // std::cout << "   t " << t << " y " << y_chunk(n, t)
                                          // << " lb " << Bound_Z[c](n, t)
                                          // << " ub " << Upper_Bound_Z[c](n, t)
                                          // << " y1_log_prob " << y1_log_prob[c](n, t)
                                          // << " Z " << Z_std_norm[c](n, t) << " u " << u_array(n, t)
                                          // << " u gradient " << u_grad_array_CM_chunk(n, t) << std::endl;
                          // }
                    // }
              // }
              //// ---- (debug output: end)
            }
          }
          if ((n_class > 1) && do_prev) {
            fn_MVP_prev_multi_pop_accumulate_grad(prob[c],
                                                  prob_n_recip,
                                                  pop_ind,
                                                  row_start,
                                                  chunk_size,
                                                  n_pops,
                                                  c,
                                                  prev_grad_mat,
                                                  rowwise_prod);
            for (std::size_t pi = 0; pi < P_c.size(); pi++) { const int n = P_c[pi]; prev_grad_mat(pop_ind(row_start + n), c) += std::exp(y1_log_prob[c].row(n).sum() + log_prob_n_recip(n)); }
          }
          if (do_us && (c == n_class - 1)) {
            const int start_index = 1 + row_start * n_tests;
            const int length = chunk_size * n_tests;
            out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();
            fn_MVP_nuisance_first_deriv_T<vec>( du_wrt_duu_chunk,
                                                u_vec_chunk,
                                                u_unc_vec_chunk,
                                                k);
            fn_MVP_nuisance_deriv_of_log_det_J_T<vec>(d_J_wrt_duu_chunk,
                                                      u_vec_chunk,
                                                      u_unc_vec_chunk,
                                                      du_wrt_duu_chunk,
                                                      k);
            out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
            out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();
          }
        }

}




//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_Inplace_process_serial_T(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                       const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                       const std::string &grad_option,
                                                                                       const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                       const int n_chunks_multiplier
) {

        const MVOP_prep P = fn_MVOP_prep(out_mat,
                                         theta_main_vec_ref,
                                         theta_us_vec_ref,
                                         y_ref,
                                         Model_args_as_cpp_struct,
                                         n_chunks_multiplier);
        std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array = vec_of_mats<double>(P.n_covariates_max,
                                                                                         P.n_tests,
                                                                                         P.n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array = vec_of_mats<double>(P.n_tests,
                                                                                            P.n_tests,
                                                                                            P.n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> cutpoint_grad_array = vec_of_mats<double>(P.n_cutpoints_max,
                                                                                             P.n_ordinal_tests,
                                                                                             P.n_class);
        Eigen::Matrix<double, -1, -1> prev_grad_mat = Eigen::Matrix<double, -1, -1>::Zero(P.n_pops, P.n_class);
        double log_jac_u = 0.0;
        for (int nc = 0; nc < P.n_total_chunks; nc++) {
          const int cs = (nc == P.n_full_chunks) ? P.last_chunk_size : P.normal_chunk_size;
          fn_lp_grad_MVOP_LC_Pinkney_PartialLog_process_chunk_T<vec>( out_mat,
                                                                      theta_us_vec_ref,
                                                                      y_ref,
                                                                      grad_option,
                                                                      nc,
                                                                      cs,
                                                                      P,
                                                                      log_jac_u,
                                                                      beta_grad_array,
                                                                      U_Omega_grad_array,
                                                                      cutpoint_grad_array,
                                                                      prev_grad_mat,
                                                                      Model_args_as_cpp_struct);
        }
        fn_MVOP_assemble(out_mat,
                         P,
                         log_jac_u,
                         beta_grad_array,
                         U_Omega_grad_array,
                         cutpoint_grad_array,
                         prev_grad_mat);

}




//// ---- original name / signature, called by the unchanged _Inplace_process dispatcher ----
inline void fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_Inplace_process_serial(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                     const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                     const std::string &grad_option,
                                                                                     const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                     std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                     const int n_chunks_multiplier
) {

        (void)LC_MVP_ws_structs;
        const Vec vec = vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0));
        DISPATCH_VEC(vec, fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_Inplace_process_serial_T,
                     out_mat, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct, n_chunks_multiplier);

}




//////////////////////////////////////////////////------------------------------------------------------------------------------------------------------------------------
//// Dispatcher - mirrors the NoLog naming. NOTE: the WCP (within-chunk-parallel) PartialLog
//// variant is NOT implemented yet, so this ALWAYS runs the serial version (n_threads_WCP ignored):
inline void fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_Inplace_process(    Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat ,
                                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                const std::string &grad_option,
                                                                                const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                const int n_threads_WCP,
                                                                              // const int n_chunks_multiplier = 1
                                                                                const int n_chunks_multiplier
) {
  
      (void) n_threads_WCP; //// WCP PartialLog variant: TODO
      
      // fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_Inplace_process_serial(   out_mat,
      //                                                                           theta_main_vec_ref,
      //                                                                           theta_us_vec_ref,
      //                                                                           y_ref,
      //                                                                           grad_option,
      //                                                                           Model_args_as_cpp_struct,
      //                                                                           LC_MVP_ws_structs,
      //                                                                           n_chunks_multiplier);
      ////
      //// n_chunks_multiplier: the value set in R (sample() argument
      //// n_chunks_multiplier_for_PartialLog_log_scale_evaluation, default 3, carried in
      //// Model_args_as_cpp_struct.PartialLog_n_chunks_multiplier) unless a caller passes a positive value. The
      //// evaluation reads the nuisance values in the chunk layout of its own number of chunks and the nuisance
      //// gradient is laid back into the stored layout (PartialLog_nuisance_chunk_relayout.hpp):
      ////
      const int n_chunks_multiplier_used = (n_chunks_multiplier > 0) ? n_chunks_multiplier :
                                           Model_args_as_cpp_struct.PartialLog_n_chunks_multiplier;
      fn_PartialLog_evaluation_in_its_own_chunk_layout(
            out_mat,
            theta_us_vec_ref,
            static_cast<int>(y_ref.rows()),
            static_cast<int>(y_ref.cols()),
            Model_args_as_cpp_struct.Model_args_ints(3),
            fn_vec_size_of_vect_type(Model_args_as_cpp_struct.Model_args_strings(0)),
            n_chunks_multiplier_used,
            [&]( Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat_of_evaluation,
                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_of_evaluation) {

                  fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_Inplace_process_serial(
                        out_mat_of_evaluation,
                        theta_main_vec_ref,
                        theta_us_of_evaluation,
                        y_ref,
                        grad_option,
                        Model_args_as_cpp_struct,
                        LC_MVP_ws_structs,
                        n_chunks_multiplier_used);

            });
  
}




//// Internal function using Eigen::Ref as inputs for matrices
inline void     fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_InPlace(   Eigen::Matrix<double, -1, 1> &&out_mat_R_val,
                                                                           const Eigen::Matrix<double, -1, 1> &&theta_main_vec_R_val,
                                                                           const Eigen::Matrix<double, -1, 1> &&theta_us_vec_R_val,
                                                                           const Eigen::Matrix<int, -1, -1> &&y_R_val,
                                                                           const std::string &grad_option,
                                                                           const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                           std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                           const int n_threads_WCP
) {
  
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat_ref(out_mat_R_val);
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref(theta_main_vec_R_val);  // create Eigen::Ref from R-value
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref(theta_us_vec_R_val);  // create Eigen::Ref from R-value
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref(y_R_val);  // create Eigen::Ref from R-value
      
      fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_Inplace_process(   out_mat_ref,
                                                                         theta_main_vec_ref,
                                                                         theta_us_vec_ref,
                                                                         y_ref,
                                                                         grad_option,
                                                                         Model_args_as_cpp_struct,
                                                                         LC_MVP_ws_structs,
                                                                         n_threads_WCP,
                                                                         -1);
  
}




//// Internal function using Eigen::Ref as inputs for matrices
inline void     fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_InPlace(   Eigen::Matrix<double, -1, 1> &out_mat_ref,
                                                                           const Eigen::Matrix<double, -1, 1> &theta_main_vec_ref,
                                                                           const Eigen::Matrix<double, -1, 1> &theta_us_vec_ref,
                                                                           const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                           const std::string &grad_option,
                                                                           const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                           std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                           const int n_threads_WCP
) {
  
      fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_Inplace_process(   out_mat_ref,
                                                                         theta_main_vec_ref,
                                                                         theta_us_vec_ref,
                                                                         y_ref,
                                                                         grad_option,
                                                                         Model_args_as_cpp_struct,
                                                                         LC_MVP_ws_structs,
                                                                         n_threads_WCP,
                                                                         -1);
  
}




//// Internal function using Eigen::Ref as inputs for matrices
template <typename MatrixType>
inline void     fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_InPlace(   Eigen::Ref<Eigen::Block<MatrixType, -1, 1>>  &out_mat_ref,
                                                                           const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_main_vec_ref,
                                                                           const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_us_vec_ref,
                                                                           const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                           const std::string &grad_option,
                                                                           const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                           std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                           const int n_threads_WCP
) {
  
      fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_Inplace_process(   out_mat_ref,
                                                                         theta_main_vec_ref,
                                                                         theta_us_vec_ref,
                                                                         y_ref,
                                                                         grad_option,
                                                                         Model_args_as_cpp_struct,
                                                                         LC_MVP_ws_structs,
                                                                         n_threads_WCP,
                                                                         -1);
  
}




//// Internal function using Eigen::Ref as inputs for matrices
inline Eigen::Matrix<double, -1, 1>    fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD(  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                         const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                         const std::string &grad_option,
                                                                                         const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                         std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                         const int n_threads_WCP
) {
  
      int n_params_main = theta_main_vec_ref.rows();
      int n_us = theta_us_vec_ref.rows();
      int n_params = n_us + n_params_main;
      int N = y_ref.rows();
      
      Eigen::Matrix<double, -1, 1> out_mat = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
      
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat_ref(out_mat);
      
      fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_Inplace_process(   out_mat_ref,
                                                                         theta_main_vec_ref,
                                                                         theta_us_vec_ref,
                                                                         y_ref,
                                                                         grad_option,
                                                                         Model_args_as_cpp_struct,
                                                                         LC_MVP_ws_structs,
                                                                         n_threads_WCP,
                                                                         -1);
      
      return out_mat;
  
}






















