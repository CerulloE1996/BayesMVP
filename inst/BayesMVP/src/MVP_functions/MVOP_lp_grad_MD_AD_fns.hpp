
//// MVOP_lp_grad_MD_AD_fns.hpp

#pragma once




#include <Eigen/Dense>

#include <unsupported/Eigen/SpecialFunctions>









//// MVOP_lp_grad_MD_AD_fns_T.hpp — replaces MVOP_lp_grad_MD_AD_fns.hpp AND the _WCP impl in MVOP_lp_grad_MD_AD_fns_WCP.hpp
////
//// Keep in your files: fn_MVOP_resize_ws_for_chunk (reused here), fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process
//// and the InPlace wrappers. They call the two non-template functions at the bottom of this file
//// (original names/signatures): ..._Inplace_process_serial and ..._Inplace_process_WCP.
//// Delete your old fn_lp_grad_MVOP_LC_Pinkney_NoLog_process_chunk, ..._process_serial and ..._process_WCP bodies.
////
//// Requires: MVOP_lp_grad_common_T.hpp, migration_MVOP_helpers_and_MVP_driver.hpp (fn_MVOP_compute_lp_GHK_cols_T),
//// and the unchanged fn_MVOP_grad_prep / fn_MVOP_compute_cutpoint_grad / fn_MOVP_compute_L_Omega_grad_v3.













//////////////////////////////////////////////////------------------------------------------------------------------------------------------------------------------------
//// Resize the workspace matrices for a given chunk size (only called when the current ws
//// dimensions differ - i.e. once, when switching from the full-chunk size to the remainder size):
inline void fn_MVOP_resize_ws_for_chunk(   LC_MVP_workspace_struct &ws,
                                                  const int chunk_size,
                                                  const int n_tests,
                                                  const int n_class
) {
      
      for (int c = 0; c < 2; c++) {
          ws.Z_std_norm[c].resize(chunk_size, n_tests);
          ws.Bound_Z[c].resize(chunk_size, n_tests);
          ws.Bound_U_Phi_Bound_Z[c].resize(chunk_size, n_tests);
          ws.prob[c].resize(chunk_size, n_tests);
          ws.Phi_Z[c].resize(chunk_size, n_tests);
      }
      ///////////////////////////////////////////////
      ws.y1_log_prob.resize(chunk_size, n_tests);
      ws.phi_Z_recip.resize(chunk_size, n_tests);
      ws.phi_Bound_Z.resize(chunk_size, n_tests);
      ws.u_grad_array_CM_chunk.resize(chunk_size, n_tests);
      ws.common_grad_term_1.resize(chunk_size, n_tests);
      ws.y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(chunk_size, n_tests);
      ws.y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(chunk_size, n_tests);
      ws.prob_rowwise_prod_temp.resize(chunk_size, n_tests);
      ws.prob_recip_rowwise_prod_temp.resize(chunk_size, n_tests);
      ///////////////////////////////////////////////
      ws.prod_container_or_inc_array.resize(chunk_size);
      ws.derivs_chain_container_vec.resize(chunk_size);
      ws.prob_rowwise_prod_temp_all.resize(chunk_size);
      ///////////////////////////////////////////////
      ws.dphi_direct_Cj.resize(chunk_size); //// ordinal-only
      ws.dZ_direct_Cj.resize(chunk_size); //// ordinal-only
      ///////////////////////////////////////////////
      ws.grad_prob.resize(chunk_size, n_tests);
      ws.z_grad_term.resize(chunk_size, n_tests);
      ws.y_chunk.resize(chunk_size, n_tests);
      ws.u_array.resize(chunk_size, n_tests);
      ws.y_sign.resize(chunk_size, n_tests);
      ws.y_m_y_sign_x_u.resize(chunk_size, n_tests);
      ws.u_grad_array_CM_chunk_block.resize(chunk_size, n_tests);
      ws.u_unc_vec_chunk.resize(chunk_size * n_tests);
      ws.u_vec_chunk.resize(chunk_size * n_tests);
      ws.du_wrt_duu_chunk.resize(chunk_size * n_tests);
      ws.d_J_wrt_duu_chunk.resize(chunk_size * n_tests);
      ///////////////////////////////////////////////
      ws.lp_array.resize(chunk_size, n_class);
      ///////////////////////////////////////////////
      ws.prob_n.resize(chunk_size);
      ws.prob_n_recip.resize(chunk_size);
      ws.log_sum_result.resize(chunk_size);
      ws.container_max_logs.resize(chunk_size);
      ws.rowwise_log_sum.resize(chunk_size);
      ws.rowwise_prod.resize(chunk_size);
      ws.rowwise_sum.resize(chunk_size);
      ws.log_lik_chunk.resize(chunk_size);
      ws.prob_recip.resize(chunk_size, n_tests);
      ///////////////////////////////////////////////
      for (int c = 0; c < n_class; c++) {
        ws.Upper_Bound_Z[c].resize(chunk_size, n_tests); //// ordinal-only
      }
      ///////////////////////////////////////////////
      ws.phi_Upper_Bound_Z.resize(chunk_size, n_tests); //// ordinal-only
      ws.dphi_over_L.resize(chunk_size, n_tests); //// ordinal-only
      ws.dZ_dmu_neg.resize(chunk_size, n_tests); //// ordinal-only
      ///////////////////////////////////////////////
      ws.dphi_times_bz.resize(chunk_size, n_tests); //// ordinal-only
      ws.dZ_times_bz.resize(chunk_size, n_tests); //// ordinal-only
      ///////////////////////////////////////////////
      ws.log_prev_per_obs_given_c.resize(chunk_size); 
      ws.prev_per_obs_given_c.resize(chunk_size);
  
}




//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_lp_grad_MVOP_LC_Pinkney_NoLog_process_chunk_T(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                               const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                               const std::string &grad_option,
                                                               const int chunk_counter, const int chunk_size,
                                                               const MVOP_prep &P,
                                                               double &log_jac_u,
                                                               std::vector<Eigen::Matrix<double, -1, -1>> &beta_grad_array,
                                                               std::vector<Eigen::Matrix<double, -1, -1>> &U_Omega_grad_array,
                                                               std::vector<Eigen::Matrix<double, -1, -1>> &cutpoint_grad_array,
                                                               Eigen::Matrix<double, -1, -1> &prev_grad_mat,
                                                               LC_MVP_workspace_struct &ws,
                                                               const Model_fn_args_struct &Model_args_as_cpp_struct
) {
        const double sqrt_2_pi_recip = 1.0 / std::sqrt(2.0 * M_PI);
        const double Inf = std::numeric_limits<double>::infinity();
        const int N = P.N, n_params = P.n_params, n_tests = P.n_tests, n_class = P.n_class, n_pops = P.n_pops, n_covariates_max = P.n_covariates_max;
        const int row_start = P.chunk_size_orig * chunk_counter;
        const KernelChoice &k = P.kchoice;
        const Eigen::Matrix<int, -1, 1> &ord_idx_of_test = P.ord_idx_of_test;
        const Eigen::Matrix<int, -1, -1> &ncov = *P.n_covariates_per_outcome_vec;
        const Eigen::Matrix<int, -1, 1> &n_cat_per_ord_test = *P.n_cat_per_ord_test, &n_thr_per_ord_test = *P.n_thr_per_ord_test, &pop_ind = *P.pop_ind;
        const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> &X = *P.X;
        const std::vector<Eigen::Matrix<double, -1, -1>> &beta = P.beta_double_array, &C = P.C, &L = P.L_Omega_double, &Lr = P.L_Omega_recip_double;

        if ((ws.y_chunk.rows() != chunk_size) || (ws.y_chunk.cols() != n_tests)) fn_MVOP_resize_ws_for_chunk(ws, chunk_size, n_tests, n_class);

        auto &Z_std_norm = ws.Z_std_norm; auto &Bound_Z = ws.Bound_Z; auto &Bound_U_Phi_Bound_Z = ws.Bound_U_Phi_Bound_Z; auto &prob = ws.prob; auto &Phi_Z = ws.Phi_Z;
        auto &y1_log_prob = ws.y1_log_prob; auto &phi_Z_recip = ws.phi_Z_recip; auto &phi_Bound_Z = ws.phi_Bound_Z;
        auto &u_grad_array_CM_chunk = ws.u_grad_array_CM_chunk; auto &common_grad_term_1 = ws.common_grad_term_1;
        auto &prob_rowwise_prod_temp = ws.prob_rowwise_prod_temp; auto &prob_recip_rowwise_prod_temp = ws.prob_recip_rowwise_prod_temp;
        auto &prod_container_or_inc_array = ws.prod_container_or_inc_array; auto &derivs_chain_container_vec = ws.derivs_chain_container_vec;
        auto &prob_rowwise_prod_temp_all = ws.prob_rowwise_prod_temp_all; auto &dphi_direct_Cj = ws.dphi_direct_Cj; auto &dZ_direct_Cj = ws.dZ_direct_Cj;
        auto &grad_prob = ws.grad_prob; auto &z_grad_term = ws.z_grad_term;
        auto &y_chunk = ws.y_chunk; auto &u_array = ws.u_array; auto &y_sign = ws.y_sign; auto &y_m_y_sign_x_u = ws.y_m_y_sign_x_u;
        auto &u_grad_array_CM_chunk_block = ws.u_grad_array_CM_chunk_block;
        auto &u_unc_vec_chunk = ws.u_unc_vec_chunk; auto &u_vec_chunk = ws.u_vec_chunk; auto &du_wrt_duu_chunk = ws.du_wrt_duu_chunk; auto &d_J_wrt_duu_chunk = ws.d_J_wrt_duu_chunk;
        auto &lp_array = ws.lp_array; auto &prob_n = ws.prob_n; auto &prob_n_recip = ws.prob_n_recip; auto &log_sum_result = ws.log_sum_result; auto &container_max_logs = ws.container_max_logs;
        auto &rowwise_prod = ws.rowwise_prod; auto &rowwise_sum = ws.rowwise_sum; auto &log_lik_chunk = ws.log_lik_chunk; auto &prob_recip = ws.prob_recip;
        auto &Upper_Bound_Z = ws.Upper_Bound_Z; auto &phi_Upper_Bound_Z = ws.phi_Upper_Bound_Z;
        auto &dphi_over_L = ws.dphi_over_L; auto &dZ_dmu_neg = ws.dZ_dmu_neg; auto &dphi_times_bz = ws.dphi_times_bz; auto &dZ_times_bz = ws.dZ_times_bz;
        auto &log_prev_per_obs_given_c = ws.log_prev_per_obs_given_c; auto &prev_per_obs_given_c = ws.prev_per_obs_given_c;

        u_grad_array_CM_chunk.setZero();
        y_chunk = y_ref.middleRows(row_start, chunk_size).cast<double>();
        u_unc_vec_chunk = theta_us_vec_ref.segment(row_start * n_tests, chunk_size * n_tests);
        fn_MVP_compute_nuisance_T<vec>(u_vec_chunk, u_unc_vec_chunk, k);
        log_jac_u += fn_MVP_compute_nuisance_log_jac_u_T<vec>(u_vec_chunk, u_unc_vec_chunk, d_J_wrt_duu_chunk, k);
        u_array = u_vec_chunk.reshaped(chunk_size, n_tests);
        y_sign.array() = 2.0 * y_chunk.array() - 1.0;
        y_m_y_sign_x_u.array() = y_chunk.array() - (y_sign.array() * u_array.array());

        //// ---- likelihood ----
        for (int c = 0; c < n_class; c++) {
          if (n_class > 1) for (int n = 0; n < chunk_size; ++n) { const int g = pop_ind(row_start + n); log_prev_per_obs_given_c(n) = P.log_prev_mat_small(g, c); prev_per_obs_given_c(n) = P.prev_mat(g, c); }
          prod_container_or_inc_array.setZero();
          for (int t = 0; t < n_tests; t++) {
            const int t_ord = ord_idx_of_test(t);
            const double L_tt_inv = Lr[c](t, t);
            //// mu_t (with or without covariates) into a single expression:
            Eigen::Matrix<double, -1, 1> mu(chunk_size);
            if (n_covariates_max > 1) mu = X[c][t].block(row_start, 0, chunk_size, ncov(c, t)).cast<double>() * beta[c].col(t).head(ncov(c, t)) + prod_container_or_inc_array;
            else                      mu.array() = beta[c](0, t) + prod_container_or_inc_array.array();
            if (t_ord < 0) {
              Bound_Z[c].col(t).array() = -L_tt_inv * mu.array();
            } else {
              const int K_t = n_cat_per_ord_test(t_ord);
              for (int n = 0; n < chunk_size; ++n) {
                const int y_n = static_cast<int>(y_chunk(n, t));
                Bound_Z[c](n, t)       = (y_n == 1)   ? -Inf : (C[c](y_n - 2, t_ord) - mu(n)) * L_tt_inv;
                Upper_Bound_Z[c](n, t) = (y_n == K_t) ?  Inf : (C[c](y_n - 1, t_ord) - mu(n)) * L_tt_inv;
              }
            }
            fn_MVOP_compute_lp_GHK_cols_T<vec>(t, Bound_U_Phi_Bound_Z[c], Phi_Z[c], Z_std_norm[c], prob[c], y1_log_prob, Bound_Z[c], y_chunk, u_array, Upper_Bound_Z[c], (t_ord < 0), k);
            if (t < n_tests - 1) prod_container_or_inc_array = Z_std_norm[c].leftCols(t + 1) * L[c].row(t + 1).head(t + 1).transpose();
          }
          rowwise_sum = y1_log_prob.rowwise().sum();
          if (n_class > 1) { rowwise_sum.array() += log_prev_per_obs_given_c.array(); lp_array.col(c) = rowwise_sum; } else lp_array.col(0) = rowwise_sum;
        }
        if (n_class > 1) { log_sum_exp_general_T<vec>(lp_array, log_sum_result, container_max_logs); out_mat.segment(1 + n_params + row_start, chunk_size) = log_sum_result; }
        else               out_mat.tail(N).segment(row_start, chunk_size) = lp_array.col(0);
        log_lik_chunk = out_mat.tail(N).segment(row_start, chunk_size);
        prob_n = log_lik_chunk; apply_inplace<vec, Fn::exp>(prob_n);
        prob_n_recip = stan::math::inv(prob_n);

        //// ---- gradients ----
        for (int c = 0; c < n_class; c++) {
          if (n_class > 1) for (int n = 0; n < chunk_size; ++n) prev_per_obs_given_c(n) = P.prev_mat(pop_ind(row_start + n), c);
          prob_recip = stan::math::inv(prob[c]);
          for (int t = 0; t < n_tests; t++) {
            fn_MVP_compute_phi_Z_recip_cols_T<vec>(t, phi_Z_recip, Phi_Z[c], Z_std_norm[c], k);
            if (ord_idx_of_test(t) < 0) {
              fn_MVP_compute_phi_Bound_Z_cols_T<vec>(t, phi_Bound_Z, Bound_U_Phi_Bound_Z[c], Bound_Z[c], k);
            } else {
              phi_Bound_Z.col(t).array() = -0.5 * Bound_Z[c].col(t).array().square();       apply_col_inplace<vec, Fn::exp>(phi_Bound_Z, t);       phi_Bound_Z.col(t).array() *= sqrt_2_pi_recip;
              phi_Upper_Bound_Z.col(t).array() = -0.5 * Upper_Bound_Z[c].col(t).array().square(); apply_col_inplace<vec, Fn::exp>(phi_Upper_Bound_Z, t); phi_Upper_Bound_Z.col(t).array() *= sqrt_2_pi_recip;
              for (int i = 0; i < chunk_size; ++i) { if (std::isinf(Bound_Z[c](i, t))) phi_Bound_Z(i, t) = 0.0; if (std::isinf(Upper_Bound_Z[c](i, t))) phi_Upper_Bound_Z(i, t) = 0.0; }
            }
          }
          if (grad_option != "none") {
            fn_MVOP_grad_prep(prob[c], y_sign, y_m_y_sign_x_u, u_array, Lr[c], prev_per_obs_given_c, prob_n_recip, phi_Z_recip, phi_Bound_Z, phi_Upper_Bound_Z,
                              Bound_Z[c], Upper_Bound_Z[c], prob_recip, prob_rowwise_prod_temp, prob_recip_rowwise_prod_temp, prob_rowwise_prod_temp_all, common_grad_term_1,
                              dphi_over_L, dZ_dmu_neg, dphi_times_bz, dZ_times_bz, ord_idx_of_test, Model_args_as_cpp_struct);
          }
          if ((grad_option == "us_only") || (grad_option == "all")) {
            u_grad_array_CM_chunk_block = u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests);
            fn_MVP_compute_nuisance_grad_v2(u_grad_array_CM_chunk_block, phi_Z_recip, common_grad_term_1, L[c], prob[c], prob_recip, prob_rowwise_prod_temp, dphi_over_L, dZ_dmu_neg,
                                            z_grad_term, grad_prob, prod_container_or_inc_array, derivs_chain_container_vec, Model_args_as_cpp_struct);
            u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests).array() += u_grad_array_CM_chunk_block.array();
            if (c == n_class - 1) {
              const int start_index = 1 + row_start * n_tests, length = chunk_size * n_tests;
              out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();
              fn_MVP_nuisance_first_deriv_T<vec>(du_wrt_duu_chunk, u_vec_chunk, u_unc_vec_chunk, k);
              fn_MVP_nuisance_deriv_of_log_det_J_T<vec>(d_J_wrt_duu_chunk, u_vec_chunk, u_unc_vec_chunk, du_wrt_duu_chunk, k);
              out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
              out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();
            }
          }
          if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only")) {
            Eigen::Matrix<int, -1, 1> n_cov_vec_c = ncov.row(c).transpose();
            fn_MVP_compute_coefficients_grad_v3(c, beta_grad_array[c], X[c], n_cov_vec_c, row_start, n_covariates_max, common_grad_term_1, L[c], prob[c], prob_recip, prob_rowwise_prod_temp,
                                                dphi_over_L, dZ_dmu_neg, z_grad_term, grad_prob, prod_container_or_inc_array, derivs_chain_container_vec, true, Model_args_as_cpp_struct);
          }
          if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only")) {
            fn_MOVP_compute_L_Omega_grad_v3(U_Omega_grad_array[c], common_grad_term_1, L[c], prob[c], prob_recip, Z_std_norm[c], prob_rowwise_prod_temp,
                                            dphi_over_L, dZ_dmu_neg, dphi_times_bz, dZ_times_bz, z_grad_term, grad_prob, prod_container_or_inc_array, derivs_chain_container_vec, true, Model_args_as_cpp_struct);
          }
          if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "cutpoints_only")) {
            for (int t = 0; t < n_tests; ++t) {
              const int t_ord = ord_idx_of_test(t); if (t_ord < 0) continue;
              fn_MVOP_compute_cutpoint_grad(c, cutpoint_grad_array[c], t, t_ord, n_thr_per_ord_test(t_ord) + 1, common_grad_term_1, L[c], prob[c], prob_recip, prob_rowwise_prod_temp,
                                            phi_Bound_Z, phi_Upper_Bound_Z, phi_Z_recip, u_array, y_chunk, dphi_over_L, dZ_dmu_neg, z_grad_term, grad_prob, prod_container_or_inc_array,
                                            derivs_chain_container_vec, dphi_direct_Cj, dZ_direct_Cj, true, Model_args_as_cpp_struct);
            }
          }
          if ((n_class > 1) && ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only"))) {
            fn_MVP_prev_multi_pop_accumulate_grad(prob[c], prob_n_recip, pop_ind, row_start, chunk_size, n_pops, c, prev_grad_mat, rowwise_prod);
          }
        }
}




//// -------------------------------------------------------------------------------------
//// One outer template for BOTH serial and WCP: n_threads_WCP == 1 => serial loop.
//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_T(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                           const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                           const std::string &grad_option,
                                                                           const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                           std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                           const int n_threads_WCP
) {
        const MVOP_prep P = fn_MVOP_prep(out_mat, theta_main_vec_ref, theta_us_vec_ref, y_ref, Model_args_as_cpp_struct);
        const int n_tests = P.n_tests, n_class = P.n_class, n_pops = P.n_pops, n_covariates_max = P.n_covariates_max;
        std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array = vec_of_mats<double>(n_covariates_max, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array = vec_of_mats<double>(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> cutpoint_grad_array = vec_of_mats<double>(P.n_cutpoints_max, P.n_ordinal_tests, n_class);
        Eigen::Matrix<double, -1, -1> prev_grad_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        double log_jac_u = 0.0;

        if (n_threads_WCP <= 1) {
          LC_MVP_workspace_struct &ws = LC_MVP_ws_structs[0]; ws.reset_sizes();
          for (int nc = 0; nc < P.n_total_chunks; nc++) {
            const int cs = (nc == P.n_full_chunks) ? P.last_chunk_size : P.normal_chunk_size;
            fn_lp_grad_MVOP_LC_Pinkney_NoLog_process_chunk_T<vec>(out_mat, theta_us_vec_ref, y_ref, grad_option, nc, cs, P, log_jac_u,
                                                                  beta_grad_array, U_Omega_grad_array, cutpoint_grad_array, prev_grad_mat, ws, Model_args_as_cpp_struct);
          }
        } else {
          //// Each chunk accumulates into its OWN zeroed slot; the slots are added up in CHUNK ORDER after the parallel
          //// region, so lp/grad is bit-reproducible run to run and independent of n_threads_WCP (previously the
          //// per-thread partials were added in an 'omp critical' block in whatever order the threads finished).
          std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> beta_grad_per_chunk(P.n_full_chunks);
          std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> U_Omega_grad_per_chunk(P.n_full_chunks);
          std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> cutpoint_grad_per_chunk(P.n_full_chunks);
          std::vector<Eigen::Matrix<double, -1, -1>>              prev_grad_per_chunk(P.n_full_chunks);
          std::vector<double>                                     log_jac_u_per_chunk(P.n_full_chunks, 0.0);
          #pragma omp parallel num_threads(n_threads_WCP)
          {
            LC_MVP_workspace_struct &ws = LC_MVP_ws_structs[omp_get_thread_num()]; ws.reset_sizes();
            #pragma omp for schedule(static)
            for (int nc = 0; nc < P.n_full_chunks; nc++) {
              beta_grad_per_chunk[nc]     = vec_of_mats<double>(n_covariates_max, n_tests, n_class);
              U_Omega_grad_per_chunk[nc]  = vec_of_mats<double>(n_tests, n_tests, n_class);
              cutpoint_grad_per_chunk[nc] = vec_of_mats<double>(P.n_cutpoints_max, P.n_ordinal_tests, n_class);
              prev_grad_per_chunk[nc]     = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
              fn_lp_grad_MVOP_LC_Pinkney_NoLog_process_chunk_T<vec>(out_mat, theta_us_vec_ref, y_ref, grad_option, nc, P.normal_chunk_size, P,
                                                                    log_jac_u_per_chunk[nc], beta_grad_per_chunk[nc], U_Omega_grad_per_chunk[nc],
                                                                    cutpoint_grad_per_chunk[nc], prev_grad_per_chunk[nc], ws, Model_args_as_cpp_struct);
            }
          }
          //// ---- deterministic reduction, in chunk order:
          for (int nc = 0; nc < P.n_full_chunks; nc++) {
            for (int c = 0; c < n_class; ++c) { beta_grad_array[c] += beta_grad_per_chunk[nc][c]; U_Omega_grad_array[c] += U_Omega_grad_per_chunk[nc][c]; cutpoint_grad_array[c] += cutpoint_grad_per_chunk[nc][c]; }
            prev_grad_mat.array() += prev_grad_per_chunk[nc].array(); log_jac_u += log_jac_u_per_chunk[nc];
          }
          if ((P.n_full_chunks < P.n_total_chunks) && (P.last_chunk_size > 0)) {
            LC_MVP_workspace_struct &ws0 = LC_MVP_ws_structs[0]; ws0.reset_sizes();
            fn_lp_grad_MVOP_LC_Pinkney_NoLog_process_chunk_T<vec>(out_mat, theta_us_vec_ref, y_ref, grad_option, P.n_full_chunks, P.last_chunk_size, P, log_jac_u,
                                                                  beta_grad_array, U_Omega_grad_array, cutpoint_grad_array, prev_grad_mat, ws0, Model_args_as_cpp_struct);
          }
        }
        fn_MVOP_assemble(out_mat, P, log_jac_u, beta_grad_array, U_Omega_grad_array, cutpoint_grad_array, prev_grad_mat);
}




//// ---- original names / signatures, called by your unchanged _Inplace_process dispatcher ----
inline void fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                const std::string &grad_option,
                                                                                const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs
) {
        const Vec vec = vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0));
        DISPATCH_VEC(vec, fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_T,
                     out_mat, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct, LC_MVP_ws_structs, 1);
}



inline void fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_WCP(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                             const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                             const std::string &grad_option,
                                                                             const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                             std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                             const int n_threads_WCP
) {
        const Vec vec = vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0));
        DISPATCH_VEC(vec, fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_T,
                     out_mat, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct, LC_MVP_ws_structs, n_threads_WCP);
}







