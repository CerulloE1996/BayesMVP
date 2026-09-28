
//// MVP_lp_grad_MD_AD_fns_WCP.hpp

#pragma once


 
 


#include <Eigen/Dense>
#include <optional>
 
#include <unsupported/Eigen/SpecialFunctions>




//////////////////////////////////////////////////------------------------------------------------------------------------------------------------------------------------------------
//// MVP_lp_grad_MD_AD_fns_WCP_T.hpp — drop-in replacement for fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_WCP_impl
////
//// Keep everything else in the MVP_lp_grad_MD_AD_fns_WCP.hpp (the _serial / _WCP dispatchers,
//// _Inplace_process, the InPlace wrappers): they call the NON-template
//// fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_WCP_impl at the bottom of this file,
//// which has the same signature as before. Delete only the old _WCP_impl body.
////
//// Requires: MVP_lp_grad_MD_AD_fns_migrated.hpp (for resize_MVP_workspace and the helpers it pulls in).
////
//// Structure: the preamble (params, Omega, AD block, priors) is identical to the serial driver.
//// The chunk loop is an OpenMP parallel-for over the FULL chunks with thread-local workspaces and
//// thread-local gradient accumulators; the remainder chunk (if any) is processed serially after the
//// parallel region on ws[0], using the SAME <vec> instantiation (masked tails handle any length).


//// -------------------------------------------------------------------------------------
//// One chunk of the MVP NoLog lp+grad. Shared by the parallel loop and the remainder.
//// Accumulators (log_jac_u, beta/U_Omega/prev grads) are whichever the caller passes.
//// prev_grad_terms_by_row (optional, used by the serial-order evaluation below): when given, the prevalence gradient
//// terms of this chunk are stored one per row (row n of the chunk, column c) instead of being added to prev_grad_mat.
//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_MVP_NoLog_process_chunk_T(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                           const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                           const std::string &grad_option,
                                           const int chunk_counter, const int chunk_size, const int chunk_size_orig,
                                           const int N, const int n_params, const int n_tests, const int n_class,
                                           const int n_covariates_max, const int n_pops,
                                           const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec,
                                           const Eigen::Matrix<int, -1, 1> &pop_ind,
                                           const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> &X,
                                           const std::vector<Eigen::Matrix<double, -1, -1>> &beta_double_array,
                                           const std::vector<Eigen::Matrix<double, -1, -1>> &L_Omega_double,
                                           const std::vector<Eigen::Matrix<double, -1, -1>> &L_Omega_recip_double,
                                           const Eigen::Matrix<double, -1, -1> &prev_mat,
                                           const Eigen::Matrix<double, -1, -1> &log_prev_mat_small,
                                           double &log_jac_u,
                                           std::vector<Eigen::Matrix<double, -1, -1>> &beta_grad_array,
                                           std::vector<Eigen::Matrix<double, -1, -1>> &U_Omega_grad_array,
                                           Eigen::Matrix<double, -1, -1> &prev_grad_mat,
                                           LC_MVP_workspace_struct &ws,
                                           const KernelChoice &kchoice,
                                           const Model_fn_args_struct &Model_args_as_cpp_struct,
                                           Eigen::Matrix<double, -1, -1> *prev_grad_terms_by_row = nullptr
) {
  
        const int row_start = chunk_size_orig * chunk_counter;
        if (ws.y_chunk.rows() != chunk_size) resize_MVP_workspace(ws, chunk_size, n_tests, n_class);

        auto &Z_std_norm = ws.Z_std_norm; auto &Bound_Z = ws.Bound_Z; auto &Bound_U_Phi_Bound_Z = ws.Bound_U_Phi_Bound_Z;
        auto &prob = ws.prob; auto &Phi_Z = ws.Phi_Z;
        auto &y1_log_prob = ws.y1_log_prob; auto &phi_Z_recip = ws.phi_Z_recip; auto &phi_Bound_Z = ws.phi_Bound_Z;
        auto &u_grad_array_CM_chunk = ws.u_grad_array_CM_chunk; auto &common_grad_term_1 = ws.common_grad_term_1;
        auto &ys = ws.y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip;
        auto &ym = ws.y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip;
        auto &prob_rowwise_prod_temp = ws.prob_rowwise_prod_temp; auto &prob_recip_rowwise_prod_temp = ws.prob_recip_rowwise_prod_temp;
        auto &prod_container_or_inc_array = ws.prod_container_or_inc_array; auto &derivs_chain_container_vec = ws.derivs_chain_container_vec;
        auto &prob_rowwise_prod_temp_all = ws.prob_rowwise_prod_temp_all; auto &grad_prob = ws.grad_prob; auto &z_grad_term = ws.z_grad_term;
        auto &y_chunk = ws.y_chunk; auto &u_array = ws.u_array; auto &y_sign = ws.y_sign; auto &y_m_y_sign_x_u = ws.y_m_y_sign_x_u;
        auto &u_grad_array_CM_chunk_block = ws.u_grad_array_CM_chunk_block;
        auto &u_unc_vec_chunk = ws.u_unc_vec_chunk; auto &u_vec_chunk = ws.u_vec_chunk;
        auto &du_wrt_duu_chunk = ws.du_wrt_duu_chunk; auto &d_J_wrt_duu_chunk = ws.d_J_wrt_duu_chunk;
        auto &lp_array = ws.lp_array; auto &prob_n = ws.prob_n; auto &prob_n_recip = ws.prob_n_recip;
        auto &log_sum_result = ws.log_sum_result; auto &container_max_logs = ws.container_max_logs;
        auto &rowwise_prod = ws.rowwise_prod; auto &rowwise_sum = ws.rowwise_sum; auto &log_lik_chunk = ws.log_lik_chunk;
        auto &prob_recip = ws.prob_recip; auto &log_prev_per_obs_given_c = ws.log_prev_per_obs_given_c; auto &prev_per_obs_given_c = ws.prev_per_obs_given_c;

        u_grad_array_CM_chunk.setZero();
        y_chunk = y_ref.middleRows(row_start, chunk_size).cast<double>();
        u_unc_vec_chunk = theta_us_vec_ref.segment(row_start * n_tests, chunk_size * n_tests);
        fn_MVP_compute_nuisance_T<vec>(u_vec_chunk, u_unc_vec_chunk, kchoice);
        log_jac_u += fn_MVP_compute_nuisance_log_jac_u_T<vec>(u_vec_chunk, u_unc_vec_chunk, d_J_wrt_duu_chunk, kchoice);
        u_array = u_vec_chunk.reshaped(chunk_size, n_tests);
        y_sign.array() = 2.0 * y_chunk.array() - 1.0;
        y_m_y_sign_x_u.array() = y_chunk.array() - (y_sign.array() * u_array.array());

        for (int c = 0; c < n_class; c++) {
          if (n_class > 1) {
            for (int n = 0; n < chunk_size; ++n) { const int g = pop_ind(row_start + n); log_prev_per_obs_given_c(n) = log_prev_mat_small(g, c); prev_per_obs_given_c(n) = prev_mat(g, c); }
          }
          prod_container_or_inc_array.setZero();
          for (int t = 0; t < n_tests; t++) {
            if (n_covariates_max > 1) {
              Eigen::Matrix<double, -1, 1> Xbeta = X[c][t].block(row_start, 0, chunk_size, n_covariates_per_outcome_vec(c, t)).cast<double>()
                                                   * beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t));
              Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * (-1.0 * (Xbeta.array() + prod_container_or_inc_array.array()));
            } else {
              Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * (-1.0 * (beta_double_array[c](0, t) + prod_container_or_inc_array.array()));
            }
            fn_MVP_compute_lp_GHK_cols_T<vec>(t, Bound_U_Phi_Bound_Z[c], Phi_Z[c], Z_std_norm[c], prob[c], y1_log_prob, Bound_Z[c], y_chunk, u_array, kchoice);
            if (t < n_tests - 1) prod_container_or_inc_array = Z_std_norm[c].leftCols(t + 1) * L_Omega_double[c].row(t + 1).head(t + 1).transpose();
          }
          rowwise_sum = y1_log_prob.rowwise().sum();
          if (n_class > 1) { rowwise_sum.array() += log_prev_per_obs_given_c.array(); lp_array.col(c) = rowwise_sum; }
          else               lp_array.col(0) = rowwise_sum;
        }

        if (n_class > 1) { log_sum_exp_general_T<vec>(lp_array, log_sum_result, container_max_logs); out_mat.segment(1 + n_params + row_start, chunk_size) = log_sum_result; }
        else               out_mat.tail(N).segment(row_start, chunk_size) = lp_array.col(0);

        log_lik_chunk = out_mat.tail(N).segment(row_start, chunk_size);
        prob_n = log_lik_chunk; apply_inplace<vec, Fn::exp>(prob_n);
        prob_n_recip = stan::math::inv(prob_n);

        for (int c = 0; c < n_class; c++) {
          if (n_class > 1) for (int n = 0; n < chunk_size; ++n) prev_per_obs_given_c(n) = prev_mat(pop_ind(row_start + n), c);
          prob_recip = stan::math::inv(prob[c]);
          for (int t = 0; t < n_tests; t++) {
            fn_MVP_compute_phi_Z_recip_cols_T<vec>(t, phi_Z_recip, Phi_Z[c], Z_std_norm[c], kchoice);
            fn_MVP_compute_phi_Bound_Z_cols_T<vec>(t, phi_Bound_Z, Bound_U_Phi_Bound_Z[c], Bound_Z[c], kchoice);
          }
          if (grad_option != "none") {
            fn_MVP_grad_prep(prob[c], y_sign, y_m_y_sign_x_u, L_Omega_recip_double[c], prev_per_obs_given_c, prob_n_recip, phi_Z_recip, phi_Bound_Z, prob_recip,
                             prob_rowwise_prod_temp, prob_recip_rowwise_prod_temp, prob_rowwise_prod_temp_all, common_grad_term_1, ys, ym, Model_args_as_cpp_struct);
          }
          if ((grad_option == "us_only") || (grad_option == "all")) {
            u_grad_array_CM_chunk_block = u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests);
            fn_MVP_compute_nuisance_grad_v2(u_grad_array_CM_chunk_block, phi_Z_recip, common_grad_term_1, L_Omega_double[c], prob[c], prob_recip, prob_rowwise_prod_temp,
                                            ys, ym, z_grad_term, grad_prob, prod_container_or_inc_array, derivs_chain_container_vec, Model_args_as_cpp_struct);
            u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests).array() += u_grad_array_CM_chunk_block.array();
            if (c == n_class - 1) {
              const int start_index = 1 + row_start * n_tests, length = chunk_size * n_tests;
              out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();
              fn_MVP_nuisance_first_deriv_T<vec>(du_wrt_duu_chunk, u_vec_chunk, u_unc_vec_chunk, kchoice);
              fn_MVP_nuisance_deriv_of_log_det_J_T<vec>(d_J_wrt_duu_chunk, u_vec_chunk, u_unc_vec_chunk, du_wrt_duu_chunk, kchoice);
              out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
              out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();
            }
          }
          if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only")) {
            Eigen::Matrix<int, -1, 1> n_cov_vec_c = n_covariates_per_outcome_vec.row(c).transpose();
            fn_MVP_compute_coefficients_grad_v3(c, beta_grad_array[c], X[c], n_cov_vec_c, row_start, n_covariates_max, common_grad_term_1, L_Omega_double[c], prob[c], prob_recip,
                                                prob_rowwise_prod_temp, ys, ym, z_grad_term, grad_prob, prod_container_or_inc_array, derivs_chain_container_vec, true, Model_args_as_cpp_struct);
          }
          if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only")) {
            fn_MVP_compute_L_Omega_grad_v3(U_Omega_grad_array[c], common_grad_term_1, L_Omega_double[c], prob[c], prob_recip, Bound_Z[c], Z_std_norm[c], prob_rowwise_prod_temp,
                                           ys, ym, z_grad_term, grad_prob, prod_container_or_inc_array, derivs_chain_container_vec, true, Model_args_as_cpp_struct);
          }
          if ((n_class > 1) && ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only"))) {
            if (prev_grad_terms_by_row != nullptr) {
              //// the same helper and inputs, with row n of the chunk mapped to cell (n, c) of this chunk's own matrix
              //// (identity row index, offset 0), so each cell receives exactly the term the serial loop adds for row n
              Eigen::Matrix<int, -1, 1> row_index_in_chunk(chunk_size);
              for (int n = 0; n < chunk_size; ++n) row_index_in_chunk(n) = n;
              fn_MVP_prev_multi_pop_accumulate_grad(prob[c], prob_n_recip, row_index_in_chunk, 0, chunk_size, chunk_size, c, *prev_grad_terms_by_row, rowwise_prod);
            } else {
            fn_MVP_prev_multi_pop_accumulate_grad(prob[c], prob_n_recip, pop_ind, row_start, chunk_size, n_pops, c, prev_grad_mat, rowwise_prod);
            }
          }
        }
}




//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_WCP_impl_T(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                   const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                   const std::string &grad_option,
                                                                                   const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                   std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                   const int n_threads_WCP
) {
        out_mat.setZero();
        const int N = y_ref.rows(), n_tests = y_ref.cols(), n_us = theta_us_vec_ref.rows();
        const int n_params_main = theta_main_vec_ref.rows(), n_params = n_params_main + n_us;
        const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> &X = Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0];
        const bool exclude_priors = Model_args_as_cpp_struct.Model_args_bools(0), corr_force_positive = Model_args_as_cpp_struct.Model_args_bools(2);
        const bool corr_prior_beta = Model_args_as_cpp_struct.Model_args_bools(3), corr_prior_norm = Model_args_as_cpp_struct.Model_args_bools(4);
        const int n_class = Model_args_as_cpp_struct.Model_args_ints(1), n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
        const KernelChoice kchoice = kernel_choice_from_args(Model_args_as_cpp_struct);
        const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
        const std::string &J_grad_option = Model_args_as_cpp_struct.Model_args_strings(11);
        const Eigen::Matrix<double, -1, 1> &lkj_cholesky_eta = Model_args_as_cpp_struct.Model_args_col_vecs_double[0];
        const Eigen::Matrix<double, -1, 1> &prev_prior_a = Model_args_as_cpp_struct.Model_args_col_vecs_double[1];
        const Eigen::Matrix<double, -1, 1> &prev_prior_b = Model_args_as_cpp_struct.Model_args_col_vecs_double[2];
        const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_mean = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_sd   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_for_corr_a  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_for_corr_b  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3];
        std::vector<Eigen::Matrix<double, -1, -1>>        lb_corr           = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4];
        const std::vector<Eigen::Matrix<double, -1, -1>> &ub_corr           = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5];
        const std::vector<Eigen::Matrix<double, -1, -1>> &known_values      = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6];
        const std::vector<Eigen::Matrix<int, -1, -1>>    &known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];
        if (corr_force_positive) for (int c = 0; c < n_class; ++c) for (int i = 1; i < n_tests; ++i) for (int j = 0; j < i; ++j) lb_corr[c](i, j) = 0.0;
        const int n_pops = Model_args_as_cpp_struct.Model_args_ints(6);
        const Eigen::Matrix<int, -1, 1> &pop_ind = Model_args_as_cpp_struct.Model_args_col_vecs_int[2];
        const int n_corrs = n_class * n_tests * (n_tests - 1) / 2;
        int n_covariates_total, n_covariates_max;
        if (n_class > 1) { n_covariates_total = n_covariates_per_outcome_vec.row(0).sum() + n_covariates_per_outcome_vec.row(1).sum();
                           n_covariates_max = std::max(n_covariates_per_outcome_vec.row(0).maxCoeff(), n_covariates_per_outcome_vec.row(1).maxCoeff()); }
        else             { n_covariates_total = n_covariates_per_outcome_vec.sum(); n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff(); }

        int vec_size; if (vect_type == "AVX512") vec_size = 8; else if (vect_type == "AVX2") vec_size = 4; else if (vect_type == "AVX") vec_size = 2; else vec_size = 1;
        ChunkSizeInfo csi = calculate_chunk_sizes(N, vec_size, n_chunks);
        const int chunk_size_orig = csi.chunk_size_orig, normal_chunk_size = csi.normal_chunk_size, last_chunk_size = csi.last_chunk_size;
        const int n_total_chunks = csi.n_total_chunks, n_full_chunks = csi.n_full_chunks;

        const Eigen::Matrix<double, -1, 1> Omega_raw_vec_double = theta_main_vec_ref.head(n_corrs);
        std::vector<Eigen::Matrix<double, -1, -1>> beta_double_array = vec_of_mats(n_covariates_max, n_tests, n_class);
        { int i = n_corrs; for (int c = 0; c < n_class; ++c) for (int t = 0; t < n_tests; ++t) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); ++k) beta_double_array[c](k, t) = theta_main_vec_ref(i++); }
        Eigen::Matrix<double, -1, 1> u_prev_raw(n_pops);
        if (n_class > 1) for (int g = 0; g < n_pops; ++g) u_prev_raw(g) = theta_main_vec_ref(n_corrs + n_covariates_total + g);

        double prior_densities_L_Omega_double = 0.0, log_det_J_L_Omega_double = 0.0;
        Eigen::Matrix<double, -1, 1> grad_Omega_raw_priors_and_log_det_J(n_corrs);
        const int dim_choose_2 = n_tests * (n_tests - 1) / 2;
        std::vector<Eigen::Matrix<double, -1, -1>> deriv_L_wrt_unc_full = vec_of_mats_double(dim_choose_2 + n_tests, dim_choose_2, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_double = vec_of_mats_double(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_recip_double = vec_of_mats_double(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> Omega_unconstrained_double = fn_convert_std_vec_of_corrs_to_3d_array_double(Eigen_vec_to_std_vec(Omega_raw_vec_double), n_tests, n_class);
        for (int c = 0; c < n_class; ++c) {
          auto out = Pinkney_corr_master_dbl(n_tests, lb_corr[c], ub_corr[c], Omega_unconstrained_double[c], known_values_indicator[c], known_values[c]);
          L_Omega_double[c] = out.block(1, 0, n_tests, n_tests);
          for (int i = 1; i < n_tests; ++i) for (int j = 0; j < i; ++j) L_Omega_recip_double[c](i, j) = (std::abs(L_Omega_double[c](i, j)) > 1e-10) ? 1.0 / L_Omega_double[c](i, j) : 0.0;
        }
        if (J_grad_option == "num_diff") {
          for (int c = 0; c < n_class; ++c) {
            auto get_L_orig = [&](const Eigen::Matrix<double, -1, -1> &Ou) { return Pinkney_corr_master_dbl(n_tests, lb_corr[c], ub_corr[c], Ou, known_values_indicator[c], known_values[c]).block(1, 0, n_tests, n_tests).eval(); };
            int cnt_2 = 0;
            for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++) {
              double eps = std::max(1e-8, 1e-6 * std::abs(Omega_unconstrained_double[c](i, j))); if (std::abs(Omega_unconstrained_double[c](i, j)) > 2.0) eps = 1e-4;
              Omega_unconstrained_double[c](i, j) += eps; Eigen::Matrix<double, -1, -1> Lp = get_L_orig(Omega_unconstrained_double[c]); Omega_unconstrained_double[c](i, j) -= eps;
              int cnt_1 = 0;
              for (int k = 0; k < n_tests; k++) for (int l = 0; l <= k; l++) { const double d = (Lp(k, l) - L_Omega_double[c](k, l)) / eps; deriv_L_wrt_unc_full[c](cnt_1++, cnt_2) = std::isfinite(d) ? d : 0.0; }
              cnt_2++;
            }
          }
        }
        Eigen::Matrix<double, -1, 1> grad_prev_raw(n_pops);
        double prior_densities_prev_double = 0.0, log_det_J_prev_from_AD = 0.0;
        {
          std::optional<stan::math::nested_rev_autodiff> nested_guard(std::in_place);
          stan::math::var target_AD = 0.0;
          Eigen::Matrix<stan::math::var, -1, 1> Omega_raw_vec_var = stan::math::to_var(Omega_raw_vec_double);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> Omega_unconstrained_var = fn_convert_std_vec_of_corrs_to_3d_array_var(Eigen_vec_to_std_vec_var(Omega_raw_vec_var), n_tests, n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> L_Omega_var = vec_of_mats_var(n_tests, n_tests, n_class), Omega_var = vec_of_mats_var(n_tests, n_tests, n_class);
          {
            stan::math::var log_det_J_L_Omega = 0.0;
            for (int c = 0; c < n_class; ++c) {
              Eigen::Matrix<stan::math::var, -1, -1> CS = Pinkney_corr_master(n_tests, lb_corr[c], ub_corr[c], Omega_unconstrained_var[c], known_values_indicator[c], known_values[c]);
              L_Omega_var[c] = CS.block(1, 0, n_tests, n_tests); target_AD += CS(0, 0); log_det_J_L_Omega += CS(0, 0);
              Omega_var[c] = L_Omega_var[c] * L_Omega_var[c].transpose();
              for (int i = 1; i < n_tests; ++i) for (int j = 0; j < i; ++j) if (known_values_indicator[c](i, j) == 1) {
                stan::math::var kv = stan::math::normal_lpdf(Omega_var[c](i, j), 0.0, 10.0); target_AD += kv; prior_densities_L_Omega_double += kv.val(); }
            }
            log_det_J_L_Omega_double += log_det_J_L_Omega.val();
          }
          {
            stan::math::var pd = 0.0;
            for (int c = 0; c < n_class; ++c) {
              if (!corr_prior_beta && !corr_prior_norm) pd += stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c], lkj_cholesky_eta(c));
              else {
                for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++)
                  pd += corr_prior_beta ? stan::math::beta_lpdf((Omega_var[c](i, j) + 1) / 2, prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j))
                                        : stan::math::normal_lpdf(Omega_var[c](i, j), prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
                Eigen::Matrix<stan::math::var, -1, 1> jd(n_tests);
                for (int i = 0; i < n_tests; ++i) jd(i) = (n_tests + 1 - (i + 1)) * stan::math::log(L_Omega_var[c](i, i));
                pd += (n_tests * stan::math::log(2) + jd.sum());
              }
            }
            target_AD += pd; prior_densities_L_Omega_double += pd.val();
            target_AD.grad();
            grad_Omega_raw_priors_and_log_det_J = Omega_raw_vec_var.adj();
            out_mat.segment(1 + n_us, n_corrs) = grad_Omega_raw_priors_and_log_det_J;
            stan::math::set_zero_all_adjoints_nested();
          }
          if (n_class > 1) {
            fn_MVP_prev_multi_pop_AD(u_prev_raw, prev_prior_a, prev_prior_b, n_pops, prior_densities_prev_double, log_det_J_prev_from_AD, grad_prev_raw);
            out_mat.segment(1 + n_us + n_corrs + n_covariates_total, n_pops) = grad_prev_raw;
          }
          if (J_grad_option == "autodiff") {
            for (int c = 0; c < n_class; ++c) { int cnt_1 = 0;
              for (int k = 0; k < n_tests; k++) for (int l = 0; l < k + 1; l++) {
                (L_Omega_var[c](k, l)).grad(); int cnt_2 = 0;
                for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++) deriv_L_wrt_unc_full[c](cnt_1, cnt_2++) = Omega_unconstrained_var[c](i, j).adj();
                stan::math::set_zero_all_adjoints_nested(); cnt_1++;
              }
            }
          }
          for (int c = 0; c < n_class; ++c) for (int t2 = 0; t2 < n_tests; ++t2) for (int t1 = 0; t1 < n_tests; ++t1) {
            L_Omega_double[c](t1, t2) = L_Omega_var[c](t1, t2).val(); L_Omega_recip_double[c](t1, t2) = 1.0 / L_Omega_double[c](t1, t2); }
          nested_guard.reset();
        }

        Eigen::Matrix<double, -1, -1> prev_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class), log_prev_mat_small = prev_mat;
        Eigen::Matrix<double, -1, 1> tanh_u_prev_vec(n_pops), deriv_p_wrt_u_vec(n_pops);
        double log_det_J_prev_double_total = 0.0;
        if (n_class > 1) {
          for (int g = 0; g < n_pops; ++g) {
            tanh_u_prev_vec(g) = stan::math::tanh(u_prev_raw(g)); const double pg = 0.5 * (tanh_u_prev_vec(g) + 1.0);
            prev_mat(g, 1) = pg; prev_mat(g, 0) = 1.0 - pg;
            deriv_p_wrt_u_vec(g) = 0.5 * (1.0 - tanh_u_prev_vec(g) * tanh_u_prev_vec(g));
            log_det_J_prev_double_total += stan::math::log(deriv_p_wrt_u_vec(g));
          }
          log_prev_mat_small = stan::math::log(prev_mat);
        }
        double prior_densities = 0.0;
        if (!exclude_priors) {
          for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++)
            prior_densities += stan::math::normal_lpdf(beta_double_array[c](k, t), prior_coeffs_mean[c](k, t), prior_coeffs_sd[c](k, t));
          prior_densities += prior_densities_L_Omega_double + prior_densities_prev_double;
        }
        const double log_det_J_main = log_det_J_prev_double_total + log_det_J_L_Omega_double;

        std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array = vec_of_mats<double>(n_covariates_max, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array = vec_of_mats<double>(n_tests, n_tests, n_class);
        Eigen::Matrix<double, -1, -1> prev_grad_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        double log_jac_u = 0.0;

        //// ================= PARALLEL over full chunks =================
        //// Each chunk accumulates into its OWN zeroed slot; the slots are then added up in CHUNK ORDER after the
        //// parallel region. This makes lp/grad bit-reproducible run to run AND independent of n_threads_WCP.
        //// (Previously each thread's partial sums were added inside an 'omp critical' block in whatever order the
        //// threads finished, so the last bits changed from run to run and MCMC amplified the difference.)
        std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> beta_grad_per_chunk(n_full_chunks);
        std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> U_Omega_grad_per_chunk(n_full_chunks);
        std::vector<Eigen::Matrix<double, -1, -1>>              prev_grad_per_chunk(n_full_chunks);
        std::vector<double>                                     log_jac_u_per_chunk(n_full_chunks, 0.0);
        
        #pragma omp parallel num_threads(n_threads_WCP) if (n_threads_WCP > 1)
        {
          const int tid = omp_get_thread_num();
          LC_MVP_workspace_struct &ws = LC_MVP_ws_structs[tid];
          ws.reset_sizes();

          #pragma omp for schedule(static)
          for (int nc = 0; nc < n_full_chunks; nc++) {
            beta_grad_per_chunk[nc]    = vec_of_mats<double>(n_covariates_max, n_tests, n_class);
            U_Omega_grad_per_chunk[nc] = vec_of_mats<double>(n_tests, n_tests, n_class);
            prev_grad_per_chunk[nc]    = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
            fn_MVP_NoLog_process_chunk_T<vec>(out_mat, theta_us_vec_ref, y_ref, grad_option, nc, normal_chunk_size, chunk_size_orig,
                                              N, n_params, n_tests, n_class, n_covariates_max, n_pops, n_covariates_per_outcome_vec, pop_ind, X,
                                              beta_double_array, L_Omega_double, L_Omega_recip_double, prev_mat, log_prev_mat_small,
                                              log_jac_u_per_chunk[nc], beta_grad_per_chunk[nc], U_Omega_grad_per_chunk[nc], prev_grad_per_chunk[nc],
                                              ws, kchoice, Model_args_as_cpp_struct);
          }
        }
        //// ---- deterministic reduction, in chunk order:
        for (int nc = 0; nc < n_full_chunks; nc++) {
          for (int c = 0; c < n_class; c++) { beta_grad_array[c].array() += beta_grad_per_chunk[nc][c].array(); U_Omega_grad_array[c].array() += U_Omega_grad_per_chunk[nc][c].array(); }
          prev_grad_mat.array() += prev_grad_per_chunk[nc].array();
          log_jac_u += log_jac_u_per_chunk[nc];
        }

        //// ================= remainder chunk: serial, same instantiation, ws[0] resized =================
        if ((n_full_chunks < n_total_chunks) && (last_chunk_size > 0)) {
          LC_MVP_workspace_struct &ws0 = LC_MVP_ws_structs[0];
          resize_MVP_workspace(ws0, last_chunk_size, n_tests, n_class);
          fn_MVP_NoLog_process_chunk_T<vec>(out_mat, theta_us_vec_ref, y_ref, grad_option, n_full_chunks, last_chunk_size, chunk_size_orig,
                                            N, n_params, n_tests, n_class, n_covariates_max, n_pops, n_covariates_per_outcome_vec, pop_ind, X,
                                            beta_double_array, L_Omega_double, L_Omega_recip_double, prev_mat, log_prev_mat_small,
                                            log_jac_u, beta_grad_array, U_Omega_grad_array, prev_grad_mat, ws0, kchoice, Model_args_as_cpp_struct);
        }

        //// ================= post-loop =================
        Eigen::Matrix<double, -1, 1> prev_unc_grad = Eigen::Matrix<double, -1, 1>::Zero(n_pops);
        if (n_class > 1) for (int g = 0; g < n_pops; ++g) prev_unc_grad(g) = (prev_grad_mat(g, 1) - prev_grad_mat(g, 0)) * deriv_p_wrt_u_vec(g) - 2.0 * tanh_u_prev_vec(g);
        double log_prob_out = out_mat.tail(N).sum() + log_jac_u + log_det_J_main;
        if (!exclude_priors) log_prob_out += prior_densities;
        Eigen::Matrix<double, -1, 1> beta_grad_vec = Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);
        { int i = 0; for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++) beta_grad_vec(i++) = beta_grad_array[c](k, t); }
        Eigen::Matrix<double, -1, 1> L_Omega_grad_vec(n_corrs + n_class * n_tests), U_Omega_grad_vec(n_corrs);
        { int i = 0; for (int c = 0; c < n_class; c++) for (int t1 = 0; t1 < n_tests; t1++) for (int t2 = 0; t2 < t1 + 1; t2++) L_Omega_grad_vec(i++) = U_Omega_grad_array[c](t1, t2); }
        if (n_class > 1) {
          U_Omega_grad_vec.head(dim_choose_2) = (L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests).transpose() * deriv_L_wrt_unc_full[0]).transpose();
          U_Omega_grad_vec.segment(dim_choose_2, dim_choose_2) = (L_Omega_grad_vec.segment(dim_choose_2 + n_tests, dim_choose_2 + n_tests).transpose() * deriv_L_wrt_unc_full[1]).transpose();
        } else {
          U_Omega_grad_vec.head(dim_choose_2) = (L_Omega_grad_vec.head(dim_choose_2 + n_tests).transpose() * deriv_L_wrt_unc_full[0]).transpose();
        }
        out_mat(0) = log_prob_out;
        out_mat.segment(1 + n_us, n_corrs) += U_Omega_grad_vec;
        out_mat.segment(1 + n_us + n_corrs, n_covariates_total) += beta_grad_vec;
        if (n_class > 1) out_mat.segment(1 + n_us + n_corrs + n_covariates_total, n_pops) += prev_unc_grad;
        if (!exclude_priors) {
          int i = n_us + n_corrs + 1;
          for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++)
            out_mat(i++) += -((beta_double_array[c](k, t) - prior_coeffs_mean[c](k, t)) / prior_coeffs_sd[c](k, t)) * (1.0 / prior_coeffs_sd[c](k, t));
        }
        
}




//// Non-template entry point with the ORIGINAL name/signature — the _WCP dispatcher calls this unchanged.
inline void fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_WCP_impl(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                 const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                 const std::string &grad_option,
                                                                                 const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                 std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                 const int n_threads_WCP
) {
  
        const Vec vec = vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0));
        DISPATCH_VEC(vec, fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_WCP_impl_T,
                     out_mat, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct, LC_MVP_ws_structs, n_threads_WCP);
        
}




//// -------------------------------------------------------------------------------------
//// Serial-order parallel NoLog lp+grad (the step-size search, fn_find_initial_eps_main_and_us with n_threads > 1).
//// Everything outside the chunk loop is fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl_T
//// (MVP_lp_grad_MD_AD_fns.hpp) line for line. The chunk loop is replaced by:
////   - the FULL chunks on n_threads_serial_order threads, each on its thread's workspace, through
////     fn_MVP_NoLog_process_chunk_T (statement for statement the loop body of the serial evaluation), each chunk writing
////     its contributions to its own slots: log_jac_u and the beta / L_Omega gradient cells, which the serial loop
////     increments exactly once per chunk, and the per-row prevalence terms, which the serial loop adds one row at a time;
////   - the remainder chunk on workspace 0 after the parallel region, resized as in the serial loop;
////   - the contributions added to the accumulators in chunk order, rows in row order.
//// These are the additions of the serial loop, in the same order, so lp and the gradient are bitwise those of the serial
//// evaluation for any thread count. The slots start at +0.0, which is exact here: an accumulator that starts at +0.0
//// never becomes -0.0, so adding +0.0 + x instead of x gives the same result.
//// If any chunk throws, the serial evaluation is run from the start instead (and throws where it did before).
//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_order_parallel_impl_T(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                                    const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                                    const std::string &grad_option,
                                                                                                    const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                                    std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                                    const int n_threads_serial_order
) {

        out_mat.setZero();
        ////
        const int N = y_ref.rows();
        const int n_tests = y_ref.cols();
        const int n_us = theta_us_vec_ref.rows();
        const int n_params_main = theta_main_vec_ref.rows();
        const int n_params = n_params_main + n_us;
        ////
        const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>  &X =  Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0];
        ////
        const bool exclude_priors      = Model_args_as_cpp_struct.Model_args_bools(0);
        const bool corr_force_positive = Model_args_as_cpp_struct.Model_args_bools(2);
        const bool corr_prior_beta     = Model_args_as_cpp_struct.Model_args_bools(3);
        const bool corr_prior_norm     = Model_args_as_cpp_struct.Model_args_bools(4);
        ////
        const int n_class  = Model_args_as_cpp_struct.Model_args_ints(1);
        const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
        ////
        const KernelChoice kchoice = kernel_choice_from_args(Model_args_as_cpp_struct);   //// the ONLY runtime function choices, resolved once
        ////
        const std::string &vect_type     = Model_args_as_cpp_struct.Model_args_strings(0);   //// still used for vec_size below
        const std::string &J_grad_option = Model_args_as_cpp_struct.Model_args_strings(11);
        ////
        const Eigen::Matrix<double, -1, 1> &lkj_cholesky_eta = Model_args_as_cpp_struct.Model_args_col_vecs_double[0];
        const Eigen::Matrix<double, -1, 1> &prev_prior_a     = Model_args_as_cpp_struct.Model_args_col_vecs_double[1];
        const Eigen::Matrix<double, -1, 1> &prev_prior_b     = Model_args_as_cpp_struct.Model_args_col_vecs_double[2];
        ////
        const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
        ////
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_mean = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_sd   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_for_corr_a  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_for_corr_b  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3];
        std::vector<Eigen::Matrix<double, -1, -1>>        lb_corr           = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4];
        const std::vector<Eigen::Matrix<double, -1, -1>> &ub_corr           = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5];
        const std::vector<Eigen::Matrix<double, -1, -1>> &known_values      = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6];
        const std::vector<Eigen::Matrix<int, -1, -1>>    &known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];

        if (corr_force_positive == true) {
          for (int c = 0; c < n_class; ++c)
            for (int i = 1; i < n_tests; ++i)
              for (int j = 0; j < i; ++j)
                lb_corr[c](i, j) = 0.0;
        }

        //// ---- Other prev params:
        const int n_pops = Model_args_as_cpp_struct.Model_args_ints(6);
        const Eigen::Matrix<int, -1, 1> &pop_ind = Model_args_as_cpp_struct.Model_args_col_vecs_int[2];

        const int n_corrs = n_class * n_tests * (n_tests - 1) / 2;

        int n_covariates_total, n_covariates_max;
        if (n_class > 1) {
              const int n_covariates_total_nd = n_covariates_per_outcome_vec.row(0).sum();
              const int n_covariates_total_d  = n_covariates_per_outcome_vec.row(1).sum();
              n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
              n_covariates_max = std::max(n_covariates_per_outcome_vec.row(0).maxCoeff(), n_covariates_per_outcome_vec.row(1).maxCoeff());
        } else {
              n_covariates_total = n_covariates_per_outcome_vec.sum();
              n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();
        }

        //// ---- chunk sizes ---------------------------------------------------------------
        int vec_size;
        if (vect_type == "AVX512")      vec_size = 8;
        else if (vect_type == "AVX2")   vec_size = 4;
        else if (vect_type == "AVX")    vec_size = 2;
        else                            vec_size = 1;

        ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, n_chunks);
        const int chunk_size_orig   = chunk_size_info.chunk_size_orig;
        const int normal_chunk_size = chunk_size_info.normal_chunk_size;
        const int last_chunk_size   = chunk_size_info.last_chunk_size;
        const int n_total_chunks    = chunk_size_info.n_total_chunks;
        const int n_full_chunks     = chunk_size_info.n_full_chunks;

        //////////////  ----------------------------------------------------------------------------------------------------------------------------------------------
        //// ---- Corrs (doubles):
        const Eigen::Matrix<double, -1, 1> Omega_raw_vec_double = theta_main_vec_ref.head(n_corrs);
        //// ---- Coeffs (doubles):
        std::vector<Eigen::Matrix<double, -1, -1>> beta_double_array = vec_of_mats(n_covariates_max, n_tests, n_class);
        {
          int i = n_corrs;
          for (int c = 0; c < n_class; ++c)
            for (int t = 0; t < n_tests; ++t)
              for (int k = 0; k < n_covariates_per_outcome_vec(c, t); ++k)
                beta_double_array[c](k, t) = theta_main_vec_ref(i++);
        }
        //// ---- Prev (doubles):
        Eigen::Matrix<double, -1, 1> u_prev_raw(n_pops);
        if (n_class > 1) {
          for (int g = 0; g < n_pops; ++g) u_prev_raw(g) = theta_main_vec_ref(n_corrs + n_covariates_total + g);
        }
        //// ---- Omega:
        double prior_densities_L_Omega_double = 0.0;
        double log_det_J_L_Omega_double = 0.0;
        Eigen::Matrix<double, -1, 1> grad_Omega_raw_priors_and_log_det_J(n_corrs);

        const int dim_choose_2 = n_tests * (n_tests - 1) / 2;
        std::vector<Eigen::Matrix<double, -1, -1>> deriv_L_wrt_unc_full = vec_of_mats_double(dim_choose_2 + n_tests, dim_choose_2, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_double       = vec_of_mats_double(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_recip_double = vec_of_mats_double(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> Omega_double         = vec_of_mats_double(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> Omega_unconstrained_double =
            fn_convert_std_vec_of_corrs_to_3d_array_double(Eigen_vec_to_std_vec(Omega_raw_vec_double), n_tests, n_class);

        for (int c = 0; c < n_class; ++c) {
              auto out = Pinkney_corr_master_dbl(n_tests, lb_corr[c], ub_corr[c], Omega_unconstrained_double[c], known_values_indicator[c], known_values[c]);
              L_Omega_double[c] = out.block(1, 0, n_tests, n_tests);
              Omega_double[c] = L_Omega_double[c] * L_Omega_double[c].transpose();
              for (int i = 1; i < n_tests; ++i)
                for (int j = 0; j < i; ++j)
                  L_Omega_recip_double[c](i, j) = (std::abs(L_Omega_double[c](i, j)) > 1e-10) ? 1.0 / L_Omega_double[c](i, j) : 0.0;
        }

        if (J_grad_option == "num_diff") {
              for (int c = 0; c < n_class; ++c) {
                      auto get_L_orig = [&](const Eigen::Matrix<double, -1, -1> &Omega_unc) {
                            auto out = Pinkney_corr_master_dbl(n_tests, lb_corr[c], ub_corr[c], Omega_unc, known_values_indicator[c], known_values[c]);
                            return out.block(1, 0, n_tests, n_tests).eval();
                      };
                      int cnt_2 = 0;
                      for (int i = 1; i < n_tests; i++) {
                            for (int j = 0; j < i; j++) {
                                  double epsilon = std::max(1e-8, 1e-6 * std::abs(Omega_unconstrained_double[c](i, j)));
                                  if (std::abs(Omega_unconstrained_double[c](i, j)) > 2.0) epsilon = 1e-4;
                                  Omega_unconstrained_double[c](i, j) += epsilon;
                                  Eigen::Matrix<double, -1, -1> L_perturbed = get_L_orig(Omega_unconstrained_double[c]);
                                  Omega_unconstrained_double[c](i, j) -= epsilon;
                                  int cnt_1 = 0;
                                  for (int k = 0; k < n_tests; k++) {
                                    for (int l = 0; l <= k; l++) {
                                      double deriv = (L_perturbed(k, l) - L_Omega_double[c](k, l)) / epsilon;
                                      deriv_L_wrt_unc_full[c](cnt_1, cnt_2) = std::isfinite(deriv) ? deriv : 0.0;
                                      cnt_1++;
                                    }
                                  }
                                  cnt_2++;
                            }
                      }
              }
        }

        Eigen::Matrix<double, -1, 1> grad_prev_raw(n_pops);
        double prior_densities_prev_double = 0.0;
        double log_det_J_prev_from_AD = 0.0;

        {    ///////////   -------------------  AD block  ----------------------------------------------------------------------------------------------------------
          std::optional<stan::math::nested_rev_autodiff> nested_guard(std::in_place);
          stan::math::var target_AD = 0.0;

          Eigen::Matrix<stan::math::var, -1, 1> Omega_raw_vec_var = stan::math::to_var(Omega_raw_vec_double);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> Omega_unconstrained_var =
              fn_convert_std_vec_of_corrs_to_3d_array_var(Eigen_vec_to_std_vec_var(Omega_raw_vec_var), n_tests, n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> L_Omega_var = vec_of_mats_var(n_tests, n_tests, n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> Omega_var   = vec_of_mats_var(n_tests, n_tests, n_class);

          {
              stan::math::var log_det_J_L_Omega = 0.0;
              for (int c = 0; c < n_class; ++c) {
                    Eigen::Matrix<stan::math::var, -1, -1> Chol_Schur_outs = Pinkney_corr_master(n_tests, lb_corr[c], ub_corr[c], Omega_unconstrained_var[c], known_values_indicator[c], known_values[c]);
                    L_Omega_var[c] = Chol_Schur_outs.block(1, 0, n_tests, n_tests);
                    target_AD         += Chol_Schur_outs(0, 0);
                    log_det_J_L_Omega += Chol_Schur_outs(0, 0);
                    Omega_var[c] = L_Omega_var[c] * L_Omega_var[c].transpose();
                    for (int i = 1; i < n_tests; ++i) {
                      for (int j = 0; j < i; ++j) {
                        if (known_values_indicator[c](i, j) == 1) {
                          stan::math::var known_val_prior_ij = stan::math::normal_lpdf(Omega_var[c](i, j), 0.0, 10.0);
                          target_AD += known_val_prior_ij;
                          prior_densities_L_Omega_double += known_val_prior_ij.val();
                        }
                      }
                    }
              }
              log_det_J_L_Omega_double += log_det_J_L_Omega.val();
          }

          {
              stan::math::var prior_densities_L_Omega = 0.0;
              for (int c = 0; c < n_class; ++c) {
                    if ((corr_prior_beta == false) && (corr_prior_norm == false)) {
                      prior_densities_L_Omega += stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c], lkj_cholesky_eta(c));
                    } else if ((corr_prior_beta == true) && (corr_prior_norm == false)) {
                      for (int i = 1; i < n_tests; i++)
                        for (int j = 0; j < i; j++)
                          prior_densities_L_Omega += stan::math::beta_lpdf((Omega_var[c](i, j) + 1) / 2, prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
                      Eigen::Matrix<stan::math::var, -1, 1> jacobian_diag_elements(n_tests);
                      for (int i = 0; i < n_tests; ++i) jacobian_diag_elements(i) = (n_tests + 1 - (i + 1)) * stan::math::log(L_Omega_var[c](i, i));
                      prior_densities_L_Omega += (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());
                    } else if ((corr_prior_beta == false) && (corr_prior_norm == true)) {
                      for (int i = 1; i < n_tests; i++)
                        for (int j = 0; j < i; j++)
                          prior_densities_L_Omega += stan::math::normal_lpdf(Omega_var[c](i, j), prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
                      Eigen::Matrix<stan::math::var, -1, 1> jacobian_diag_elements(n_tests);
                      for (int i = 0; i < n_tests; ++i) jacobian_diag_elements(i) = (n_tests + 1 - (i + 1)) * stan::math::log(L_Omega_var[c](i, i));
                      prior_densities_L_Omega += (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());
                    }
              }
              target_AD += prior_densities_L_Omega;
              prior_densities_L_Omega_double += prior_densities_L_Omega.val();

              target_AD.grad();
              grad_Omega_raw_priors_and_log_det_J = Omega_raw_vec_var.adj();
              out_mat.segment(1 + n_us, n_corrs) = grad_Omega_raw_priors_and_log_det_J;
              stan::math::set_zero_all_adjoints_nested();
          }

          if (n_class > 1) {
              fn_MVP_prev_multi_pop_AD(u_prev_raw, prev_prior_a, prev_prior_b, n_pops, prior_densities_prev_double, log_det_J_prev_from_AD, grad_prev_raw);
              const int prev_start = 1 + n_us + n_corrs + n_covariates_total;
              out_mat.segment(prev_start, n_pops) = grad_prev_raw;
          }

          if (J_grad_option == "autodiff") {
            for (int c = 0; c < n_class; ++c) {
              int cnt_1 = 0;
              for (int k = 0; k < n_tests; k++) {
                for (int l = 0; l < k + 1; l++) {
                  (L_Omega_var[c](k, l)).grad();
                  int cnt_2 = 0;
                  for (int i = 1; i < n_tests; i++)
                    for (int j = 0; j < i; j++)
                      deriv_L_wrt_unc_full[c](cnt_1, cnt_2++) = Omega_unconstrained_var[c](i, j).adj();
                  stan::math::set_zero_all_adjoints_nested();
                  cnt_1 += 1;
                }
              }
            }
          }

          for (int c = 0; c < n_class; ++c) {
            for (int t2 = 0; t2 < n_tests; ++t2) {
              for (int t1 = 0; t1 < n_tests; ++t1) {
                L_Omega_double[c](t1, t2)       = L_Omega_var[c](t1, t2).val();
                L_Omega_recip_double[c](t1, t2) = 1.0 / L_Omega_double[c](t1, t2);
              }
            }
          }

          nested_guard.reset();
        }    //////////////////////////  end of AD block ------------------------------------------------------------------------------------------------------------

        /////////////  prev (multi-pop)
        Eigen::Matrix<double, -1, -1> prev_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        Eigen::Matrix<double, -1, -1> log_prev_mat_small = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        Eigen::Matrix<double, -1, 1>  tanh_u_prev_vec(n_pops);
        Eigen::Matrix<double, -1, 1>  deriv_p_wrt_u_vec(n_pops);
        double log_det_J_prev_double_total = 0.0;
        if (n_class > 1) {
            for (int g = 0; g < n_pops; ++g) {
                tanh_u_prev_vec(g) = stan::math::tanh(u_prev_raw(g));
                const double prev_g = 0.5 * (tanh_u_prev_vec(g) + 1.0);
                prev_mat(g, 1) = prev_g;
                prev_mat(g, 0) = 1.0 - prev_g;
                deriv_p_wrt_u_vec(g) = 0.5 * (1.0 - tanh_u_prev_vec(g) * tanh_u_prev_vec(g));
                log_det_J_prev_double_total += stan::math::log(deriv_p_wrt_u_vec(g));
            }
            log_prev_mat_small = stan::math::log(prev_mat);
        }

        ///////////////////////////////////////////////////////////////////////// prior densities
        double prior_densities = 0.0;
        if (exclude_priors == false) {
              double prior_densities_coeffs_double = 0.0;
              for (int c = 0; c < n_class; c++)
                for (int t = 0; t < n_tests; t++)
                  for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++)
                    prior_densities_coeffs_double += stan::math::normal_lpdf(beta_double_array[c](k, t), prior_coeffs_mean[c](k, t), prior_coeffs_sd[c](k, t));
              prior_densities += prior_densities_coeffs_double;
              prior_densities += prior_densities_L_Omega_double;
              prior_densities += prior_densities_prev_double;
        }

        ////////  ------- likelihood  ---------------------------------------------------------------------------------------------------------------------------------
        double log_prob_out = 0.0;
        const double log_det_J_main = log_det_J_prev_double_total + log_det_J_L_Omega_double;

        std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array    = vec_of_mats<double>(n_covariates_max, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array = vec_of_mats<double>(n_tests, n_tests, n_class);
        Eigen::Matrix<double, -1, -1> prev_grad_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        double log_jac_u = 0.0;

        // ============================================================================
        // CHUNKS: full chunks in parallel, then the remainder; contributions added in the serial order.
        // ============================================================================
        {
          const int n_threads_used = std::max(1, std::min(std::min(n_threads_serial_order, n_full_chunks),
                                                          static_cast<int>(LC_MVP_ws_structs.size())));
          const bool prev_grad_needed = (n_class > 1) && ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only"));
          ////
          std::vector<double>                                     log_jac_u_per_chunk(n_total_chunks, 0.0);
          std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> beta_grad_per_chunk(n_total_chunks);
          std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> U_Omega_grad_per_chunk(n_total_chunks);
          std::vector<Eigen::Matrix<double, -1, -1>>              prev_grad_terms_per_chunk(n_total_chunks);
          std::vector<int>                                        chunk_threw(n_total_chunks, 0);

          //// ---- full chunks:
          #pragma omp parallel num_threads(n_threads_used) if (n_threads_used > 1)
          {
            const int tid = omp_get_thread_num();
            LC_MVP_workspace_struct &ws = LC_MVP_ws_structs[tid];
            ws.reset_sizes();
            Eigen::Matrix<double, -1, -1> prev_grad_mat_not_used = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);

            #pragma omp for schedule(static)
            for (int nc = 0; nc < n_full_chunks; nc++) {
              try {
                    beta_grad_per_chunk[nc]       = vec_of_mats<double>(n_covariates_max, n_tests, n_class);
                    U_Omega_grad_per_chunk[nc]    = vec_of_mats<double>(n_tests, n_tests, n_class);
                    prev_grad_terms_per_chunk[nc] = Eigen::Matrix<double, -1, -1>::Zero(normal_chunk_size, n_class);
                    fn_MVP_NoLog_process_chunk_T<vec>(out_mat, theta_us_vec_ref, y_ref, grad_option, nc, normal_chunk_size, chunk_size_orig,
                                                      N, n_params, n_tests, n_class, n_covariates_max, n_pops, n_covariates_per_outcome_vec, pop_ind, X,
                                                      beta_double_array, L_Omega_double, L_Omega_recip_double, prev_mat, log_prev_mat_small,
                                                      log_jac_u_per_chunk[nc], beta_grad_per_chunk[nc], U_Omega_grad_per_chunk[nc], prev_grad_mat_not_used,
                                                      ws, kchoice, Model_args_as_cpp_struct, &prev_grad_terms_per_chunk[nc]);
              } catch (...) {
                    chunk_threw[nc] = 1;
              }
            }
          }

          //// ---- remainder chunk (the last iteration of the serial loop, on workspace 0 resized to last_chunk_size):
          if (n_full_chunks < n_total_chunks) {
            const int nc = n_full_chunks;
            try {
                  LC_MVP_workspace_struct &ws0 = LC_MVP_ws_structs[0];
                  resize_MVP_workspace(ws0, last_chunk_size, n_tests, n_class);
                  Eigen::Matrix<double, -1, -1> prev_grad_mat_not_used = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
                  beta_grad_per_chunk[nc]       = vec_of_mats<double>(n_covariates_max, n_tests, n_class);
                  U_Omega_grad_per_chunk[nc]    = vec_of_mats<double>(n_tests, n_tests, n_class);
                  prev_grad_terms_per_chunk[nc] = Eigen::Matrix<double, -1, -1>::Zero(last_chunk_size, n_class);
                  fn_MVP_NoLog_process_chunk_T<vec>(out_mat, theta_us_vec_ref, y_ref, grad_option, nc, last_chunk_size, chunk_size_orig,
                                                    N, n_params, n_tests, n_class, n_covariates_max, n_pops, n_covariates_per_outcome_vec, pop_ind, X,
                                                    beta_double_array, L_Omega_double, L_Omega_recip_double, prev_mat, log_prev_mat_small,
                                                    log_jac_u_per_chunk[nc], beta_grad_per_chunk[nc], U_Omega_grad_per_chunk[nc], prev_grad_mat_not_used,
                                                    ws0, kchoice, Model_args_as_cpp_struct, &prev_grad_terms_per_chunk[nc]);
            } catch (...) {
                  chunk_threw[nc] = 1;
            }
          }

          //// ---- a chunk threw: the serial evaluation from the start instead
          for (int nc = 0; nc < n_total_chunks; nc++) {
            if (chunk_threw[nc] != 0) {
                  fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl( out_mat,
                                                                                          theta_main_vec_ref,
                                                                                          theta_us_vec_ref,
                                                                                          y_ref,
                                                                                          grad_option,
                                                                                          Model_args_as_cpp_struct,
                                                                                          LC_MVP_ws_structs);
                  return;
            }
          }

          //// ---- contributions added to the accumulators in the serial order: chunk by chunk, rows in row order
          for (int nc = 0; nc < n_total_chunks; nc++) {
            const int row_start  = chunk_size_orig * nc;
            const int chunk_size = (nc == n_full_chunks) ? last_chunk_size : normal_chunk_size;
            log_jac_u += log_jac_u_per_chunk[nc];
            for (int c = 0; c < n_class; c++) {
              beta_grad_array[c].array()    += beta_grad_per_chunk[nc][c].array();
              U_Omega_grad_array[c].array() += U_Omega_grad_per_chunk[nc][c].array();
            }
            if (prev_grad_needed) {
              for (int c = 0; c < n_class; c++)
                for (int n = 0; n < chunk_size; ++n)
                  prev_grad_mat(pop_ind(row_start + n), c) += prev_grad_terms_per_chunk[nc](n, c);
            }
          }

        }

        // ============================================================================
        // Post-loop: prevalence gradient, log_prob, output assembly
        // ============================================================================
        Eigen::Matrix<double, -1, 1> prev_unconstrained_grad_vec_out = Eigen::Matrix<double, -1, 1>::Zero(n_pops);
        if (n_class > 1) {
          for (int g = 0; g < n_pops; ++g) {
            const double lik_grad = (prev_grad_mat(g, 1) - prev_grad_mat(g, 0)) * deriv_p_wrt_u_vec(g);
            const double jac_grad = -2.0 * tanh_u_prev_vec(g);
            prev_unconstrained_grad_vec_out(g) = lik_grad + jac_grad;
          }
        }

        log_prob_out += out_mat.tail(N).sum();
        log_prob_out += log_jac_u;
        if (exclude_priors == false) log_prob_out += prior_densities;
        log_prob_out += log_det_J_main;

        Eigen::Matrix<double, -1, 1> beta_grad_vec = Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);
        {
          int i = 0;
          for (int c = 0; c < n_class; c++)
            for (int t = 0; t < n_tests; t++)
              for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++)
                beta_grad_vec(i++) = beta_grad_array[c](k, t);
        }

        Eigen::Matrix<double, -1, 1> L_Omega_grad_vec(n_corrs + (n_class * n_tests));
        Eigen::Matrix<double, -1, 1> U_Omega_grad_vec(n_corrs);
        {
          int i = 0;
          for (int c = 0; c < n_class; c++)
            for (int t1 = 0; t1 < n_tests; t1++)
              for (int t2 = 0; t2 < t1 + 1; t2++)
                L_Omega_grad_vec(i++) = U_Omega_grad_array[c](t1, t2);
        }

        if (n_class > 1) {
              const Eigen::Matrix<double, -1, 1> grad_wrt_L_Omega_nd = L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
              const Eigen::Matrix<double, -1, 1> grad_wrt_L_Omega_d  = L_Omega_grad_vec.segment(dim_choose_2 + n_tests, dim_choose_2 + n_tests);
              U_Omega_grad_vec.head(dim_choose_2)                    = (grad_wrt_L_Omega_nd.transpose() * deriv_L_wrt_unc_full[0]).transpose();
              U_Omega_grad_vec.segment(dim_choose_2, dim_choose_2)   = (grad_wrt_L_Omega_d.transpose()  * deriv_L_wrt_unc_full[1]).transpose();
        } else {
              const Eigen::Matrix<double, -1, 1> grad_wrt_L_Omega_nd = L_Omega_grad_vec.head(dim_choose_2 + n_tests);
              U_Omega_grad_vec.head(dim_choose_2) = (grad_wrt_L_Omega_nd.transpose() * deriv_L_wrt_unc_full[0]).transpose();
        }

        //// ---- Final output assembly ----
        out_mat(0) = log_prob_out;
        out_mat.segment(1 + n_us, n_corrs) += U_Omega_grad_vec;
        out_mat.segment(1 + n_us + n_corrs, n_covariates_total) += beta_grad_vec;
        if (n_class > 1) {
          out_mat.segment(1 + n_us + n_corrs + n_covariates_total, n_pops) += prev_unconstrained_grad_vec_out;
        }

        //// ---- Prior gradient contribution to coefficients ----
        if (exclude_priors == false) {
          int i = n_us + n_corrs + 1;
          for (int c = 0; c < n_class; c++)
            for (int t = 0; t < n_tests; t++)
              for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++)
                out_mat(i++) += -((beta_double_array[c](k, t) - prior_coeffs_mean[c](k, t)) / prior_coeffs_sd[c](k, t)) * (1.0 / prior_coeffs_sd[c](k, t));
        }

}




//// Non-template entry point of the serial-order parallel evaluation (called by the NoLog dispatcher below).
inline void fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_order_parallel_impl(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                                  const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                                  const std::string &grad_option,
                                                                                                  const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                                  std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                                  const int n_threads_serial_order
) {
  
        const Vec vec = vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0));
        DISPATCH_VEC(vec, fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_order_parallel_impl_T,
                     out_mat, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct, LC_MVP_ws_structs,
                     n_threads_serial_order);
        
}












// ============================================================================
// Serial dispatcher
// ============================================================================
void fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial( Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                              const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                              const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                              const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                              const std::string &grad_option,
                                                                              const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                              std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs
) {
        
        const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
        
        // if (n_class == 1) {
        //   
        //   fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl<1>( out_mat, 
        //                                                                             theta_main_vec_ref, 
        //                                                                             theta_us_vec_ref,
        //                                                                             y_ref,  
        //                                                                             grad_option,
        //                                                                             Model_args_as_cpp_struct,
        //                                                                             LC_MVP_ws_structs);
        //   
        // } else {
        //   
        //   fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl<2>( out_mat,
        //                                                                             theta_main_vec_ref,
        //                                                                             theta_us_vec_ref, 
        //                                                                             y_ref, 
        //                                                                             grad_option,
        //                                                                             Model_args_as_cpp_struct, 
        //                                                                             LC_MVP_ws_structs);
        //   
        // }
        
        fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl( out_mat,
                                                                                  theta_main_vec_ref,
                                                                                  theta_us_vec_ref, 
                                                                                  y_ref, 
                                                                                  grad_option,
                                                                                  Model_args_as_cpp_struct, 
                                                                                  LC_MVP_ws_structs);
  
}





// ============================================================================
// WCP dispatcher
// ============================================================================
 void fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_WCP(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                            const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                            const std::string &grad_option,
                                                                            const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                            std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                            const int n_threads_WCP
) {
  
        const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
  
        // try {
          
              // if (n_class == 1) {
              //   
              //   fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_WCP_impl<1>( out_mat,
              //                                                                          theta_main_vec_ref,
              //                                                                          theta_us_vec_ref,
              //                                                                          y_ref,
              //                                                                          grad_option,
              //                                                                          Model_args_as_cpp_struct,
              //                                                                          LC_MVP_ws_structs,
              //                                                                          n_threads_WCP);
              //   
              // } else {
              //   
              //   fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_WCP_impl<2>( out_mat,
              //                                                                          theta_main_vec_ref, 
              //                                                                          theta_us_vec_ref,
              //                                                                          y_ref,
              //                                                                          grad_option,
              //                                                                          Model_args_as_cpp_struct,
              //                                                                          LC_MVP_ws_structs,
              //                                                                          n_threads_WCP);
              //   
              // }
              
              fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_WCP_impl( out_mat,
                                                                                     theta_main_vec_ref, 
                                                                                     theta_us_vec_ref,
                                                                                     y_ref,
                                                                                     grad_option,
                                                                                     Model_args_as_cpp_struct,
                                                                                     LC_MVP_ws_structs,
                                                                                     n_threads_WCP);
          
        // } catch (const std::exception& e) {
        //   
        //       std::cerr << "EXCEPTION: " << e.what() << std::endl << std::flush;
        //       // don't throw — just return so R survives
        //       return;
        //       
        // } catch (...) {
        //   
        //       std::cerr << "UNKNOWN EXCEPTION" << std::endl << std::flush;
        //       return;
        //       
        // }
        
}




inline void fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                        const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                        const std::string &grad_option,
                                                                        const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                        std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                        const int n_threads_WCP
) {
  
      if (n_threads_WCP == 1) {
            
            // fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial( out_mat,
            //                                                                   theta_main_vec_ref,
            //                                                                   theta_us_vec_ref,
            //                                                                   y_ref,
            //                                                                   grad_option,
            //                                                                   Model_args_as_cpp_struct,
            //                                                                   LC_MVP_ws_structs[0]);
            
            if ((g_n_threads_serial_order_lp_grad > 1) && (LC_MVP_ws_structs.size() > 1)) {

                  //// inside a step-size search with n_threads > 1 (g_n_threads_serial_order_lp_grad, native_api.cpp.in): the
                  //// serial evaluation with its full chunks spread over threads, bitwise the same lp and gradient
                  fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_order_parallel_impl( out_mat,
                                                                                                         theta_main_vec_ref,
                                                                                                         theta_us_vec_ref,
                                                                                                         y_ref,
                                                                                                         grad_option,
                                                                                                         Model_args_as_cpp_struct,
                                                                                                         LC_MVP_ws_structs,
                                                                                                         g_n_threads_serial_order_lp_grad);

            } else {

            fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial(  out_mat,
                                                                               theta_main_vec_ref,
                                                                               theta_us_vec_ref,
                                                                               y_ref,
                                                                               grad_option,
                                                                               Model_args_as_cpp_struct,
                                                                               LC_MVP_ws_structs);
        
            }

      } else {
        
            fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_WCP( out_mat,
                                                                           theta_main_vec_ref,
                                                                           theta_us_vec_ref,
                                                                           y_ref,
                                                                           grad_option,
                                                                           Model_args_as_cpp_struct,
                                                                           LC_MVP_ws_structs,
                                                                           n_threads_WCP);
        
      }
  
} 


 






// Internal function using Eigen::Ref as inputs for matrices
inline void     fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_InPlace(    Eigen::Matrix<double, -1, 1> &&out_mat_R_val,
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
    
      fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(  out_mat_ref,
                                                                  theta_main_vec_ref,
                                                                  theta_us_vec_ref,
                                                                  y_ref,
                                                                  grad_option,
                                                                  Model_args_as_cpp_struct,
                                                                  LC_MVP_ws_structs,
                                                                  n_threads_WCP);
  
}








// Internal function using Eigen::Ref as inputs for matrices
inline void     fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_InPlace(    Eigen::Matrix<double, -1, 1> &out_mat_ref,
                                                               const Eigen::Matrix<double, -1, 1> &theta_main_vec_ref,
                                                               const Eigen::Matrix<double, -1, 1> &theta_us_vec_ref,
                                                               const Eigen::Matrix<int, -1, -1> &y_ref,
                                                               const std::string &grad_option,
                                                               const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                               std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                               const int n_threads_WCP




) {

      fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(  out_mat_ref,
                                                                  theta_main_vec_ref,
                                                                  theta_us_vec_ref,
                                                                  y_ref,
                                                                  grad_option,
                                                                  Model_args_as_cpp_struct, 
                                                                  LC_MVP_ws_structs,
                                                                  n_threads_WCP);
  
}





// Internal function using Eigen::Ref as inputs for matrices
template <typename MatrixType>
inline void     fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_InPlace(    Eigen::Ref<Eigen::Block<MatrixType, -1, 1>>  &out_mat_ref,
                                                               const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_main_vec_ref,
                                                               const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_us_vec_ref,
                                                               const Eigen::Matrix<int, -1, -1> &y_ref,
                                                               const std::string &grad_option,
                                                               const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                               std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                               const int n_threads_WCP




) {

      fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(  out_mat_ref,
                                                                  theta_main_vec_ref,
                                                                  theta_us_vec_ref,
                                                                  y_ref,
                                                                  grad_option,
                                                                  Model_args_as_cpp_struct,
                                                                  LC_MVP_ws_structs,
                                                                  n_threads_WCP);
  
}














// Internal function using Eigen::Ref as inputs for matrices
inline Eigen::Matrix<double, -1, 1>    fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD(   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
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
    
      fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_InPlace(      out_mat,
                                                              theta_main_vec_ref,
                                                              theta_us_vec_ref,
                                                              y_ref,
                                                              grad_option,
                                                              Model_args_as_cpp_struct,
                                                              LC_MVP_ws_structs,
                                                              n_threads_WCP);
    
      return out_mat;

}


 








  










