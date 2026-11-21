
//// MVP_lp_grad_log_scale_MD_AD_fns_T.hpp  - drop-in replacement for MVP_lp_grad_log_scale_MD_AD_fns.hpp
////
//// Keeps the SAME entry-point name and signature that the InPlace wrappers call:
////   fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process(...)
//// so the three fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace(...) overloads and
//// fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD(...) at the bottom of the file stay exactly as they are.
//// Delete the old ..._process_chunk and ..._InPlace_process, keep the wrappers, include this file.
////
//// Requires: MVP_log_scale_grad_calc_fns_T.hpp (which pulls in MVP_helpers_migrated.hpp and
//// fn_dispatch_templated.hpp), the fn_MVP_compute_lp_GHK_cols_T<vec>, and the unchanged
//// fn_MVP_compute_nuisance_grad_v2 / fn_MVP_compute_coefficients_grad_v2 / fn_MVP_compute_L_Omega_grad_v2.

#pragma once



#include <Eigen/Dense>
#include <unsupported/Eigen/SpecialFunctions>

#include "PartialLog_nuisance_chunk_relayout.hpp"
#include "MVP_and_MVOP_missing_results_GHK_fns.hpp"   //// missing test results (y = -1) by the GHK method

// #include "MVP_log_scale_grad_calc_fns_T.hpp"




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <Vec vec>
inline void fn_lp_grad_MVP_LC_Pinkney_PartialLog_process_chunk_T(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec,
                                                                   const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                   const std::string &grad_option,
                                                                   const int chunk_counter,
                                                                   const int chunk_size,
                                                                   const int chunk_size_orig,
                                                                   const int N,
                                                                   const int n_params,
                                                                   const int n_tests,
                                                                   const int n_class,
                                                                   const int n_covariates_max,
                                                                   const int n_pops,
                                                                   const double overflow_threshold,
                                                                   const double underflow_threshold,
                                                                   const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec,
                                                                   const Eigen::Matrix<int, -1, 1> &pop_ind,
                                                                   const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> &X,
                                                                   const std::vector<Eigen::Matrix<double, -1, -1>> &beta_double_array,
                                                                   const std::vector<Eigen::Matrix<double, -1, -1>> &L_Omega_double,
                                                                   const std::vector<Eigen::Matrix<double, -1, -1>> &L_Omega_recip_double,
                                                                   const std::vector<Eigen::Matrix<double, -1, -1>> &log_abs_L_Omega_double,
                                                                   const Eigen::Matrix<double, -1, -1> &prev_mat,
                                                                   const Eigen::Matrix<double, -1, -1> &log_prev_mat_small,
                                                                   double &log_jac_u,
                                                                   std::vector<Eigen::Matrix<double, -1, -1>> &beta_grad_array,
                                                                   std::vector<Eigen::Matrix<double, -1, -1>> &U_Omega_grad_array,
                                                                   Eigen::Matrix<double, -1, -1> &prev_grad_mat,
                                                                   const KernelChoice &kchoice,
                                                                   const Model_fn_args_struct &Model_args_as_cpp_struct,   //// only for the unchanged v2 grad fns (n_class etc.)
                                                                   //// optional (serial-order evaluation below):
                                                                   //// when given, this chunk's prevalence
                                                                   //// gradient terms are stored one per row (row
                                                                   //// n of the chunk, column c) instead of being
                                                                   //// added to prev_grad_mat
                                                                   Eigen::Matrix<double, -1, -1> *prev_grad_terms_by_row = nullptr
) {
  
        typedef Eigen::Matrix<double, -1, -1> M;
        typedef Eigen::Matrix<double, -1, 1>  V;
        const int row_start = chunk_size_orig * chunk_counter;

        //// ---- per-chunk containers ----
        std::vector<M> Z_std_norm = vec_of_mats<double>(chunk_size, n_tests, n_class);
        std::vector<M> Bound_Z = Z_std_norm, Bound_U_Phi_Bound_Z = Z_std_norm, Phi_Z = Z_std_norm, phi_Bound_Z = Z_std_norm,
                       y1_log_prob = Z_std_norm, prob = Z_std_norm, phi_Z_recip = Z_std_norm;
        std::vector<M> log_phi_Z_recip = Z_std_norm, log_phi_Bound_Z = Z_std_norm, log_Z_std_norm = Z_std_norm,
                       log_abs_Bound_Z = Z_std_norm, sign_Bound_Z = Z_std_norm;

        M y_chunk = M::Zero(chunk_size, n_tests), u_array = M::Zero(chunk_size, n_tests),
          y_sign_chunk = M::Zero(chunk_size, n_tests), y_m_y_sign_x_u = M::Zero(chunk_size, n_tests);
        V inc_array = V::Zero(chunk_size);
        M u_grad_array_CM_chunk = M::Zero(chunk_size, n_tests);
        M log_abs_u_grad_array_CM_chunk = M::Constant(chunk_size, n_tests, -INFINITY);

        V log_sum_result = V::Constant(chunk_size, -INFINITY), sign_result = V::Ones(chunk_size),
          container_max_logs = V::Constant(chunk_size, -INFINITY), container_sum_exp_signed = V::Zero(chunk_size);
        V u_unc_vec_chunk = V::Zero(chunk_size * n_tests), u_vec_chunk = V::Zero(chunk_size * n_tests),
          du_wrt_duu_chunk = V::Zero(chunk_size * n_tests), d_J_wrt_duu_chunk = V::Zero(chunk_size * n_tests);

        M log_common_grad_term_1 = M::Constant(chunk_size, n_tests, -INFINITY);
        M log_abs_ys = M::Constant(chunk_size,
                                   n_tests,
                                   -INFINITY), log_abs_ym = M::Constant(chunk_size,
                                   n_tests,
                                   -INFINITY);
        M sign_ys = M::Ones(chunk_size, n_tests), sign_ym = M::Ones(chunk_size, n_tests);
        M log_prob_rowwise_prod_temp = M::Constant(chunk_size, n_tests, -INFINITY), log_prob_recip_rowwise_prod_temp = M::Constant(chunk_size, n_tests, -INFINITY);
        M log_abs_grad_prob = M::Constant(chunk_size,
                                          n_tests,
                                          -INFINITY), log_abs_z_grad_term = M::Constant(chunk_size,
                                          n_tests,
                                          -INFINITY);
        M sign_grad_prob = M::Ones(chunk_size, n_tests), sign_z_grad_term = M::Ones(chunk_size, n_tests);
        M log_abs_prod_comp = M::Constant(chunk_size,
                                          n_tests,
                                          -INFINITY), sign_prod_comp = M::Ones(chunk_size,
                                          n_tests);
        M log_abs_dcc = M::Constant(chunk_size, n_tests, -INFINITY), sign_dcc = M::Ones(chunk_size, n_tests);
        V log_abs_prod = V::Constant(chunk_size, -INFINITY), sign_prod = V::Ones(chunk_size);
        V prod_container = V::Zero(chunk_size), derivs_chain_container_vec = V::Zero(chunk_size);
        V log_prob_rowwise_prod_temp_all = V::Constant(chunk_size, -INFINITY);
        V log_abs_a = V::Constant(chunk_size,
                                  -INFINITY), sign_a = V::Ones(chunk_size), log_abs_b = V::Constant(chunk_size,
                                  -INFINITY), sign_b = V::Ones(chunk_size);
        V sign_sum_result = V::Ones(chunk_size);
        M log_terms = M::Constant(chunk_size, n_tests, -INFINITY), sign_terms = M::Ones(chunk_size, n_tests);

        Eigen::Matrix<int, -1, 1> overflow_mask(chunk_size), underflow_mask(chunk_size), OK_mask(chunk_size);
        std::vector<std::vector<std::vector<int>>> problem_index_array(n_class);
        std::vector<std::vector<int>> n_problem_array(n_class);
        for (int c = 0; c < n_class; c++) { problem_index_array[c].resize(n_tests); n_problem_array[c].assign(n_tests, 0); }

        std::vector<M> Omega_grad_array_for_each_n = vec_of_mats<double>(chunk_size, n_tests, n_tests);
        std::vector<M> sign_Omega_grad_array_for_each_n = Omega_grad_array_for_each_n, log_abs_Omega_grad_array_for_each_n = Omega_grad_array_for_each_n;
        std::vector<M> beta_grad_array_for_each_n = vec_of_mats<double>(chunk_size, n_tests, n_covariates_max);
        std::vector<M> sign_beta_grad_array_for_each_n = beta_grad_array_for_each_n, log_abs_beta_grad_array_for_each_n = beta_grad_array_for_each_n;

        V rowwise_prod = V::Zero(chunk_size), rowwise_sum = V::Zero(chunk_size);
        V log_prev_per_obs_given_c = V::Zero(chunk_size), prev_per_obs_given_c = V::Zero(chunk_size);
        M lp_array = M::Zero(chunk_size, n_class);

        //// ---- nuisance ----
        y_chunk = y_ref.middleRows(row_start, chunk_size).cast<double>();
        u_unc_vec_chunk = u_unc_vec.segment(row_start * n_tests, chunk_size * n_tests);
        fn_MVP_compute_nuisance_T<vec>(u_vec_chunk, u_unc_vec_chunk, kchoice);
        log_jac_u += fn_MVP_compute_nuisance_log_jac_u_T<vec>(u_vec_chunk,
                                                              u_unc_vec_chunk,
                                                              d_J_wrt_duu_chunk /*scratch*/,
                                                              kchoice);
        u_array = u_vec_chunk.reshaped(chunk_size, n_tests);
        //// ---- missing test results (y = -1), by the GHK method (MVP_and_MVOP_missing_results_GHK_fns.hpp),
        ////      only when the chunk has one: data without missing results run the code below unchanged.
        const bool chunk_has_missing_results = fn_MVP_rows_of_y_have_a_missing_result(
                y_ref, row_start, chunk_size);
        Eigen::Matrix<double, -1, -1> is_missing_result, Z_untruncated;   //// empty unless the chunk has one
        if (chunk_has_missing_results) {
          fn_MVP_missing_result_indicator_and_recode_missing_results_to_y_equals_1(
                y_chunk, is_missing_result);
          fn_MVP_compute_untruncated_GHK_Z_of_missing_results_T<vec>(
                Z_untruncated, u_array, is_missing_result, kchoice);
        }
        y_sign_chunk.array()   = 2.0 * y_chunk.array() - 1.0;
        y_m_y_sign_x_u.array() = y_chunk.array() - (y_sign_chunk.array() * u_array.array());

        //// ---- likelihood ----
        for (int c = 0; c < n_class; c++) {

          inc_array.setZero();
          if (n_class > 1) {
            for (int n = 0; n < chunk_size; ++n) {
              const int g = pop_ind(row_start + n);
              log_prev_per_obs_given_c(n) = log_prev_mat_small(g, c);
              prev_per_obs_given_c(n)     = prev_mat(g, c);
            }
          }

          for (int t = 0; t < n_tests; t++) {

            if (n_covariates_max > 1) {
              const V Xbeta = X[c][t].block(row_start,
                                0,
                                chunk_size,
                                n_covariates_per_outcome_vec(c, t)).cast<double>()
                              * beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t));
              Bound_Z[c].col(t) = L_Omega_recip_double[c](t, t) * (-1.0 * (Xbeta + inc_array));
            } else {
              Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t,
                      t) * (-1.0 * (beta_double_array[c](0, t) + inc_array.array()));
            }
            sign_Bound_Z[c].col(t)    = Bound_Z[c].col(t).array().sign();
            log_abs_Bound_Z[c].col(t) = Bound_Z[c].col(t).cwiseAbs();
            apply_col_inplace<vec, Fn::log>(log_abs_Bound_Z[c], t);

            //// masks
            overflow_mask.array()  = ((Bound_Z[c].col(t).array() > overflow_threshold)  && (y_chunk.col(t).array() == 1.0)).template cast<int>();
            underflow_mask.array() = ((Bound_Z[c].col(t).array() < underflow_threshold) && (y_chunk.col(t).array() == 0.0)).template cast<int>();
            if (chunk_has_missing_results) {
              //// a missing result is never a tail entry (its prob is exactly 1):
              fn_MVP_remove_missing_results_from_tail_mask(t, is_missing_result, overflow_mask);
              fn_MVP_remove_missing_results_from_tail_mask(t, is_missing_result, underflow_mask);
            }
            OK_mask.array() = ((overflow_mask.array() == 0) && (underflow_mask.array() == 0)).template cast<int>();
            const int n_overflows = overflow_mask.sum(), n_underflows = underflow_mask.sum();
            const int n_problem = n_overflows + n_underflows, n_OK = OK_mask.sum();

            std::vector<int> over_index; over_index.reserve(n_overflows);
            std::vector<int> under_index; under_index.reserve(n_underflows);
            for (int n = 0; n < chunk_size; ++n) {
              if (overflow_mask(n) == 1) over_index.push_back(n);
              else if (underflow_mask(n) == 1) under_index.push_back(n);
            }
            std::vector<int> problem_index = over_index;
            problem_index.insert(problem_index.end(), under_index.begin(), under_index.end());
            n_problem_array[c][t] = n_problem;
            problem_index_array[c][t] = problem_index;

            //// standard-scale GHK
            fn_MVP_compute_lp_GHK_cols_T<vec>(t,
                                              Bound_U_Phi_Bound_Z[c],
                                              Phi_Z[c],
                                              Z_std_norm[c],
                                              prob[c],
                                              y1_log_prob[c],
                                              
                                            Bound_Z[c], y_chunk, u_array, kchoice);
            if (chunk_has_missing_results) {
              fn_MVP_set_GHK_col_of_missing_results_to_the_untruncated_draw(
                    t, is_missing_result, u_array, Z_untruncated,
                    Bound_U_Phi_Bound_Z[c], Phi_Z[c], Z_std_norm[c], prob[c], y1_log_prob[c]);
            }
            fn_MVP_compute_phi_Z_recip_cols_T<vec>(t, phi_Z_recip[c], Phi_Z[c], Z_std_norm[c], kchoice);
            fn_MVP_compute_phi_Bound_Z_cols_T<vec>(t,
                                                   phi_Bound_Z[c],
                                                   Bound_U_Phi_Bound_Z[c],
                                                   Bound_Z[c],
                                                   kchoice);
            if (chunk_has_missing_results) {
              fn_MVP_set_phi_Bound_Z_col_of_missing_results_to_zero(
                    t, is_missing_result, phi_Bound_Z[c]);
            }

            //// log-scale quantities (grad)
            log_phi_Bound_Z[c].col(t) = phi_Bound_Z[c].col(t).cwiseAbs();  apply_col_inplace<vec, Fn::log>(log_phi_Bound_Z[c], t);
            log_phi_Z_recip[c].col(t) = phi_Z_recip[c].col(t).cwiseAbs();  apply_col_inplace<vec, Fn::log>(log_phi_Z_recip[c], t);
            log_Z_std_norm[c].col(t)  = Z_std_norm[c].col(t).cwiseAbs();   apply_col_inplace<vec, Fn::log>(log_Z_std_norm[c], t);

            if (n_OK < chunk_size) {
              if (n_underflows > 0) {
                fn_MVP_compute_lp_GHK_cols_log_scale_underflow_T<vec>(t,
                                                                      under_index,
                                                                      Bound_U_Phi_Bound_Z[c],
                                                                      Phi_Z[c],
                                                                      Z_std_norm[c],
                                                                      log_Z_std_norm[c],
                                                                      
                                                                      prob[c], y1_log_prob[c], log_phi_Bound_Z[c], log_phi_Z_recip[c], Bound_Z[c], u_array,
                                                                      kchoice);   //// exact tails when Phi_type = "Phi"
                V tmp = log_phi_Bound_Z[c](under_index,
                                        t); apply_inplace<vec, Fn::exp>(tmp); phi_Bound_Z[c](under_index,
                                        t) = tmp;
                tmp   = log_phi_Z_recip[c](under_index,
                                        t); apply_inplace<vec, Fn::exp>(tmp); phi_Z_recip[c](under_index,
                                        t) = tmp;
              }
              if (n_overflows > 0) {
                fn_MVP_compute_lp_GHK_cols_log_scale_overflow_T<vec>(t,
                                                                     n_overflows,
                                                                     over_index,
                                                                     Bound_U_Phi_Bound_Z[c],
                                                                     Phi_Z[c],
                                                                     Z_std_norm[c],
                                                                     log_Z_std_norm[c],
                                                                     
                                                                     prob[c], y1_log_prob[c], log_phi_Bound_Z[c], log_phi_Z_recip[c], Bound_Z[c], u_array,
                                                                     kchoice);   //// exact tails when Phi_type = "Phi"
                V tmp = log_phi_Bound_Z[c](over_index,
                                        t); apply_inplace<vec, Fn::exp>(tmp); phi_Bound_Z[c](over_index,
                                        t) = tmp;
                tmp   = log_phi_Z_recip[c](over_index,
                                        t); apply_inplace<vec, Fn::exp>(tmp); phi_Z_recip[c](over_index,
                                        t) = tmp;
              }
            }

            if (t < n_tests - 1) {
              inc_array = Z_std_norm[c].leftCols(t + 1) * L_Omega_double[c].row(t + 1).head(t + 1).transpose();
            }
          }

          rowwise_sum = y1_log_prob[c].rowwise().sum();
          if (n_class > 1) { rowwise_sum.array() += log_prev_per_obs_given_c.array(); lp_array.col(c) = rowwise_sum; }
          else             { lp_array.col(0) = rowwise_sum; }
        }

        const int index_start = 1 + n_params + row_start;
        if (n_class > 1) {
          log_sum_exp_general_T<vec>(lp_array, log_sum_result, container_max_logs);
          out_mat.segment(index_start, chunk_size) = log_sum_result;
        } else {
          out_mat.segment(index_start, chunk_size) = lp_array.col(0);
        }

        V log_prob_n_recip(chunk_size), prob_n_recip(chunk_size);
        {
          V prob_n = out_mat.segment(index_start, chunk_size);
          apply_inplace<vec, Fn::exp>(prob_n);
          prob_n_recip = stan::math::inv(prob_n);
          log_prob_n_recip = -out_mat.segment(index_start, chunk_size);      //// exact log(1/prob_n)
        }

        ////
        //// ---- problem sets for the GRADIENT replacement functions (per test, plus the "severe" rows):
        ////
        //// The log-scale gradient functions below (fn_MVP_compute_nuisance_grad_log_scale_T,
        //// ..._coefficients_..., ..._L_Omega_...) overwrite, for the rows they are given, the natural-scale
        //// per-row gradients with exact log-scale ones; the functions of test t are given the rows in which test
        //// t is a problem row (problem_index_array[c][t]), and the natural-scale values are kept everywhere
        //// else. That is exact as long as every natural-scale factor of the row is a normal double: the gradient
        //// column(s) of test t use prob, phi(Bound_Z), 1 / prob and 1 / phi(Z) of tests t - 1, ..., n_tests - 1,
        //// and for n_class > 1 also exp(log_common_grad_term_1) and prob_rowwise_prod_temp. Once an entry is
        //// deep in the tail (log prob below ~ -700, or log(1 / phi(Z)) above ~700) or the row's log-likelihood
        //// is below ~ -700 (exp(log_common_grad_term_1) overflows whilst prob_rowwise_prod_temp underflows), the
        //// natural-scale value of every OTHER test's column in that row is 0 * Inf = NaN or has no correct
        //// digits - e.g. the 5-test fixture at intercept +/- 30 (NaN) and any |Bound_Z| > ~38.5 (where the
        //// multi_attempts fallback sends the evaluation to this path). Such a row is "severe" and is added to
        //// the problem set of EVERY test of its class, so all of its gradient columns come from the exact
        //// log-scale formulas (the same row-level idea as the MVOP / LC_MVOP PartialLog path, row_is_problem in
        //// MVOP_lp_grad_PartialLog_MD_AD_fns.hpp). The limit 500 leaves > 200 nats of headroom below the double
        //// overflow at ~709 (1 / phi(Z) can exceed 1 / prob by -log(u)); rows below it keep the per-test sets,
        //// so chunks with only moderate tails cost no more. The likelihood fix-ups above use the per-test sets
        //// (they are per-entry by construction).
        ////
        const double severe_row_log_scale_limit = 500.0;
        std::vector<std::vector<std::vector<int>>> gradient_problem_index_array(n_class);
        std::vector<std::vector<int>> n_gradient_problem_array(n_class);
        std::vector<int> n_severe_rows_per_class(n_class, 0);
        {
          std::vector<char> row_is_severe(chunk_size, 0), row_is_in_gradient_problem_set(chunk_size, 0);
          for (int c = 0; c < n_class; c++) {
            std::fill(row_is_severe.begin(), row_is_severe.end(), 0);
            for (int n = 0; n < chunk_size; ++n) if (out_mat(index_start + n) < -severe_row_log_scale_limit) row_is_severe[n] = 1;
            for (int t = 0; t < n_tests; t++) {
              for (int n = 0; n < chunk_size; ++n) {
                if ((y1_log_prob[c](n, t) < -severe_row_log_scale_limit) || (log_phi_Z_recip[c](n, t) > severe_row_log_scale_limit)) row_is_severe[n] = 1;
              }
            }
            for (int n = 0; n < chunk_size; ++n) n_severe_rows_per_class[c] += row_is_severe[n];
            gradient_problem_index_array[c].resize(n_tests);
            n_gradient_problem_array[c].assign(n_tests, 0);
            for (int t = 0; t < n_tests; t++) {
              row_is_in_gradient_problem_set = row_is_severe;
              for (const int n : problem_index_array[c][t]) row_is_in_gradient_problem_set[n] = 1;
              std::vector<int> &gradient_problem_rows = gradient_problem_index_array[c][t];
              gradient_problem_rows.reserve(n_problem_array[c][t] + n_severe_rows_per_class[c]);
              for (int n = 0; n < chunk_size; ++n) if (row_is_in_gradient_problem_set[n] == 1) gradient_problem_rows.push_back(n);
              n_gradient_problem_array[c][t] = static_cast<int>(gradient_problem_rows.size());
            }
          }
        }

        //// ---- gradients ----
        for (int c = 0; c < n_class; c++) {

          //// Any problem row (per-test or severe) in this class and chunk
          const bool chunk_has_gradient_problem_rows = std::any_of(n_gradient_problem_array[c].begin(),
                                                                   n_gradient_problem_array[c].end(),
                                                                   
                                                                   [](const int n_rows) { return n_rows > 0; });

          if (n_class > 1) {
            for (int n = 0; n < chunk_size; ++n) log_prev_per_obs_given_c(n) = log_prev_mat_small(pop_ind(row_start + n), c);
          }
          for (auto &m : beta_grad_array_for_each_n)          m.setZero();
          for (auto &m : sign_beta_grad_array_for_each_n)     m.setOnes();
          for (auto &m : log_abs_beta_grad_array_for_each_n)  m.setConstant(-INFINITY);
          for (auto &m : Omega_grad_array_for_each_n)         m.setZero();
          for (auto &m : sign_Omega_grad_array_for_each_n)    m.setOnes();
          for (auto &m : log_abs_Omega_grad_array_for_each_n) m.setConstant(-INFINITY);

          const M y1_log_prob_recip = -y1_log_prob[c];
          const M sign_Z_std_norm   = Z_std_norm[c].array().sign();
          const M prob_recip        = stan::math::inv(prob[c]);
          M common_grad_term_1 = M::Zero(chunk_size,
                                         n_tests), prob_rowwise_prod_temp = M::Zero(chunk_size,
                                         n_tests);
          M prob_recip_rowwise_prod_temp = M::Zero(chunk_size, n_tests);
          M z_grad_term = M::Zero(chunk_size, n_tests), grad_prob = M::Zero(chunk_size, n_tests);
          V prob_rowwise_prod_temp_all = V::Zero(chunk_size);
          M ys = M::Zero(chunk_size,
                         n_tests), ym = M::Zero(chunk_size,
                         n_tests); //// y_sign_chunk_times_... / y_m_ysign_x_u_array_times_...

          if (grad_option != "none") {
            //// the log-scale preparation also runs for n_class == 1 whenever the chunk has a problem row: the
            //// problem-row replacement functions below (fn_MVP_compute_nuisance_grad_log_scale_T,
            //// fn_MVP_compute_L_Omega_grad_log_scale_T and, for intercept-only models,
            //// fn_MVP_compute_coefficients_grad_log_scale_T) are called for n_class == 1 as well, and they read
            //// log_common_grad_term_1, log_prob_rowwise_prod_temp, log_abs_ys, log_abs_ym, sign_ys and sign_ym.
            //// With the natural-scale fn_MVP_grad_prep only, all of those would stay at their -700 / +1 initial
            //// values, and every problem row (|Bound_Z| beyond the over/underflow thresholds on the "tail" side)
            //// of tests 2..n_tests would have its nuisance and L_Omega gradient overwritten with
            //// exp(-700 + ...) ~ 0 (test-1 problem rows are never replaced for those parameters, so this only
            //// shows once a LATER test's bound crosses a threshold: intercept ~8.7 in the validation fixture).
            //// With no problem row in the chunk nothing reads these terms, so ordinary points skip the extra
            //// work.
            const bool need_log_scale_preparation = (n_class > 1) || chunk_has_gradient_problem_rows;
            if (need_log_scale_preparation) {
              M log_abs_L_Omega_recip_double = M::Zero(n_tests, n_tests);
              M sign_L_Omega_recip_double = L_Omega_recip_double[c].array().sign();
              for (int t = 0; t < n_tests; t++) log_abs_L_Omega_recip_double(t,
                   t) = std::log(std::abs(L_Omega_recip_double[c](t, t)));

              fn_MVP_grad_prep_log_scale_T<vec>(log_prob_rowwise_prod_temp,
                                                log_prob_recip_rowwise_prod_temp,
                                                log_prob_rowwise_prod_temp_all,
                                                
                                                log_common_grad_term_1, log_abs_ys, log_abs_ym, sign_ys, sign_ym,
                                                y1_log_prob[c], y1_log_prob_recip, log_prob_n_recip, log_prev_per_obs_given_c,
                                                log_phi_Bound_Z[c], log_phi_Z_recip[c], log_abs_L_Omega_recip_double, sign_L_Omega_recip_double,
                                                y_sign_chunk, y_m_y_sign_x_u, n_class);
            }

            if (n_class == 1) {
              //// Standard MVP: the NON-problem rows keep the natural-scale single-class formulas (prob_recip
              //// based, common_grad_term_1 = 1).
              fn_MVP_grad_prep(prob[c], y_sign_chunk, y_m_y_sign_x_u,
                               L_Omega_recip_double[c], prev_per_obs_given_c,
                               prob_n_recip, phi_Z_recip[c], phi_Bound_Z[c], prob_recip,
                               prob_rowwise_prod_temp, prob_recip_rowwise_prod_temp,
                               prob_rowwise_prod_temp_all, common_grad_term_1,
                               ys, ym,
                               Model_args_as_cpp_struct);
            } else {
              common_grad_term_1 = log_common_grad_term_1;       apply_inplace<vec, Fn::exp>(common_grad_term_1);
              prob_rowwise_prod_temp = log_prob_rowwise_prod_temp; apply_inplace<vec, Fn::exp>(prob_rowwise_prod_temp);
              ys = log_abs_ys; apply_inplace<vec, Fn::exp>(ys); ys.array() *= sign_ys.array();
              ym = log_abs_ym; apply_inplace<vec, Fn::exp>(ym); ym.array() *= sign_ym.array();
            }
          }

          auto reset_log_scale_scratch = [&]() {

            sign_z_grad_term.setOnes(); sign_grad_prob.setOnes(); sign_prod.setOnes(); sign_sum_result.setOnes(); sign_terms.setOnes();
            sign_a.setOnes(); sign_b.setOnes(); sign_prod_comp.setOnes(); sign_dcc.setOnes();
            log_abs_z_grad_term.setConstant(-INFINITY); log_abs_grad_prob.setConstant(-INFINITY); log_abs_prod.setConstant(-INFINITY);
            log_sum_result.setConstant(-INFINITY); log_terms.setConstant(-INFINITY); log_abs_a.setConstant(-INFINITY); log_abs_b.setConstant(-INFINITY);
            container_max_logs.setConstant(-INFINITY); log_abs_prod_comp.setConstant(-INFINITY); log_abs_dcc.setConstant(-INFINITY);

          };

          //// ---- nuisance ----
          if ((grad_option == "us_only") || (grad_option == "all")) {
            M u_grad_array_CM_chunk_block = u_grad_array_CM_chunk;
            fn_MVP_compute_nuisance_grad_v2(u_grad_array_CM_chunk_block,
                                            phi_Z_recip[c],
                                            common_grad_term_1,
                                            L_Omega_double[c],
                                            prob[c],
                                            prob_recip,
                                            
                                            prob_rowwise_prod_temp, ys, ym, log_abs_z_grad_term, log_abs_grad_prob, log_abs_prod, sign_prod, Model_args_as_cpp_struct);
            fn_MVP_compute_nuisance_grad_log_scale_T<vec>(n_gradient_problem_array[c],
                                                          gradient_problem_index_array[c],
                                                          log_abs_u_grad_array_CM_chunk,
                                                          u_grad_array_CM_chunk_block,
                                                          
                                                          L_Omega_double[c], log_abs_L_Omega_double[c], log_phi_Z_recip[c], y1_log_prob[c], y1_log_prob_recip,
                                                          log_prob_rowwise_prod_temp, log_abs_ys, sign_ys, log_abs_ym, sign_ym, log_common_grad_term_1,
                                                          log_abs_z_grad_term, sign_z_grad_term, log_abs_grad_prob, sign_grad_prob, log_abs_prod, sign_prod,
                                                          log_sum_result, sign_sum_result, log_terms, sign_terms, log_abs_a, log_abs_b, sign_a, sign_b,
                                                          container_max_logs, container_sum_exp_signed);
            u_grad_array_CM_chunk.array() += u_grad_array_CM_chunk_block.array();

            if (c == n_class - 1) {
              const int start_index = 1 + row_start * n_tests;
              const int length = chunk_size * n_tests;
              out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();
              fn_MVP_nuisance_first_deriv_T<vec>(du_wrt_duu_chunk, u_vec_chunk, u_unc_vec_chunk, kchoice);
              fn_MVP_nuisance_deriv_of_log_det_J_T<vec>(d_J_wrt_duu_chunk,
                                                        u_vec_chunk,
                                                        u_unc_vec_chunk,
                                                        du_wrt_duu_chunk,
                                                        kchoice);
              out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
              out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();
            }
            reset_log_scale_scratch();
          }

          //// ---- coefficients ----
          if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only")) {
            Eigen::Matrix<int, -1, 1> n_cov_vec_c = n_covariates_per_outcome_vec.row(c).transpose();
            //// a single-class, intercept-only chunk WITH SEVERE rows (see above) takes the same per-row +
            //// log-scale-replacement route as the latent-class intercept-only models (the branch below), instead
            //// of the natural-scale row sums of fn_MVP_compute_coefficients_grad_v3, whose
            //// prob_recip * phi_Bound_Z terms are 0 * Inf = NaN once a |Bound_Z| passes ~38.5 (exactly the
            //// region the multi_attempts fallback sends here). v3 is exact for moderate tails, so the other
            //// chunks use v3. Single-class models WITH covariates use v3 for every row: there is no log-scale
            //// covariate function (as for the latent-class covariate path).
            const bool use_single_class_log_scale_intercepts = (n_class == 1) && (n_covariates_max == 1) && (n_severe_rows_per_class[c] > 0);
            if ((n_class == 1) && !use_single_class_log_scale_intercepts) {
              fn_MVP_compute_coefficients_grad_v3(c, beta_grad_array[c], X[c], n_cov_vec_c, row_start,
                                                  n_covariates_max, common_grad_term_1, L_Omega_double[c],
                                                  prob[c], prob_recip, prob_rowwise_prod_temp,
                                                  ys, ym,
                                                  z_grad_term, grad_prob, prod_container, derivs_chain_container_vec,
                                                  true, Model_args_as_cpp_struct);
            } else if (n_covariates_max > 1) {
              fn_MVP_compute_coefficients_grad_v2(c,
                                                  beta_grad_array[c],
                                                  X[c],
                                                  n_cov_vec_c,
                                                  beta_grad_array_for_each_n,
                                                  chunk_counter,
                                                  n_covariates_max,
                                                  
                                                  common_grad_term_1, L_Omega_double[c], prob[c], prob_recip, prob_rowwise_prod_temp, ys, ym,
                                                  log_abs_grad_prob, sign_grad_prob, log_abs_prod, sign_prod, true, Model_args_as_cpp_struct);
            } else {
              if (use_single_class_log_scale_intercepts) {
                //// The intercept branch of fn_MVP_compute_coefficients_grad_v2 below uses the latent-class form
                //// common_grad_term_1(t) * sum_ii grad_prob(ii) * prob_rowwise_prod_temp(t) / prob(t + ii). With
                //// one class log_common_grad_term_1(t) = sum_{s >= t} log(1 / prob_s)
                //// (fn_MVP_grad_prep_log_scale_T), so this equals the single-class
                //// sum_ii grad_prob(ii) / prob(t + ii) on the non-problem rows; the problem rows are replaced on
                //// the log scale. (Only the coefficient function reads these two matrices when n_class == 1.)
                common_grad_term_1 = log_common_grad_term_1;          apply_inplace<vec, Fn::exp>(common_grad_term_1);
                prob_rowwise_prod_temp = log_prob_rowwise_prod_temp;  apply_inplace<vec, Fn::exp>(prob_rowwise_prod_temp);
              }
              log_abs_grad_prob.setZero(); sign_grad_prob.setZero(); log_abs_prod.setZero(); sign_prod.setZero();
              //// compute_final_scalar_grad = FALSE here: this call only fills the per-row gradients
              //// (beta_grad_array_for_each_n); the problem rows are then replaced on the log scale and ALL rows
              //// are summed into beta_grad_array[c] below. TRUE would add the row sums a second time,
              //// double-counting the intercept gradient on every PartialLog / fallback evaluation.
              fn_MVP_compute_coefficients_grad_v2(c,
                                                  beta_grad_array[c],
                                                  X[c],
                                                  n_cov_vec_c,
                                                  beta_grad_array_for_each_n,
                                                  chunk_counter,
                                                  n_covariates_max,
                                                  
                                                  common_grad_term_1, L_Omega_double[c], prob[c], prob_recip, prob_rowwise_prod_temp, ys, ym,
                                                  log_abs_grad_prob, sign_grad_prob, log_abs_prod, sign_prod, false, Model_args_as_cpp_struct);

              for (size_t i = 0; i < beta_grad_array_for_each_n.size(); i++) {
                sign_beta_grad_array_for_each_n[i]    = beta_grad_array_for_each_n[i].array().sign();
                log_abs_beta_grad_array_for_each_n[i] = beta_grad_array_for_each_n[i].cwiseAbs();
                apply_inplace<vec, Fn::log>(log_abs_beta_grad_array_for_each_n[i]);
                for (int t = 0; t < n_tests; t++) {
                  sign_beta_grad_array_for_each_n[i].col(t)(gradient_problem_index_array[c][t]).setOnes();
                  log_abs_beta_grad_array_for_each_n[i].col(t)(gradient_problem_index_array[c][t]).setConstant(-INFINITY);
                }
              }
              fn_MVP_compute_coefficients_grad_log_scale_T<vec>(n_gradient_problem_array[c],
                                                                gradient_problem_index_array[c],
                                                                beta_grad_array[c],
                                                                
                                                                sign_beta_grad_array_for_each_n, log_abs_beta_grad_array_for_each_n,
                                                                L_Omega_double[c], log_abs_L_Omega_double[c], log_phi_Z_recip[c], y1_log_prob[c],
                                                                log_prob_rowwise_prod_temp, log_abs_ys, sign_ys, log_abs_ym, sign_ym, log_common_grad_term_1,
                                                                log_abs_z_grad_term, sign_z_grad_term, log_abs_grad_prob, sign_grad_prob, log_abs_prod, sign_prod,
                                                                log_abs_prod_comp, sign_prod_comp, log_sum_result, sign_sum_result, log_terms, sign_terms,
                                                                container_max_logs, container_sum_exp_signed);
              for (int t = 0; t < n_tests; t++) {
                for (size_t k = 0; k < beta_grad_array_for_each_n.size(); k++) {
                  const LogSumSignedResult_T r = log_sum_vec_signed_T<vec>(log_abs_beta_grad_array_for_each_n[k].col(t), sign_beta_grad_array_for_each_n[k].col(t));
                  beta_grad_array[c](k, t) += std::exp(r.log_sum) * r.sign;
                }
              }
            }
            reset_log_scale_scratch();
          }

          //// ---- L_Omega ----
          if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only")) {
            log_abs_z_grad_term.setZero(); log_abs_grad_prob.setZero(); log_abs_prod.setZero(); sign_prod.setZero();
            fn_MVP_compute_L_Omega_grad_v2(U_Omega_grad_array[c],
                                           Omega_grad_array_for_each_n,
                                           common_grad_term_1,
                                           L_Omega_double[c],
                                           prob[c],
                                           prob_recip,
                                           
                                           Bound_Z[c], Z_std_norm[c], prob_rowwise_prod_temp, ys, ym,
                                           log_abs_z_grad_term, log_abs_grad_prob, log_abs_prod, sign_prod, false, Model_args_as_cpp_struct);

            for (int t1 = 0; t1 < n_tests; t1++) {
              sign_Omega_grad_array_for_each_n[t1]    = Omega_grad_array_for_each_n[t1].array().sign();
              log_abs_Omega_grad_array_for_each_n[t1] = Omega_grad_array_for_each_n[t1].cwiseAbs();
              apply_inplace<vec, Fn::log>(log_abs_Omega_grad_array_for_each_n[t1]);
              for (int t2 = 0; t2 < t1 + 1; t2++) {
                sign_Omega_grad_array_for_each_n[t1].col(t2)(gradient_problem_index_array[c][t1]).setOnes();
                log_abs_Omega_grad_array_for_each_n[t1].col(t2)(gradient_problem_index_array[c][t1]).setConstant(-INFINITY);
              }
            }
            fn_MVP_compute_L_Omega_grad_log_scale_T<vec>(n_gradient_problem_array[c],
                                                         gradient_problem_index_array[c],
                                                         U_Omega_grad_array[c],
                                                         
                                                         sign_Omega_grad_array_for_each_n, log_abs_Omega_grad_array_for_each_n,
                                                         log_abs_Bound_Z[c], sign_Bound_Z[c], log_Z_std_norm[c], sign_Z_std_norm,
                                                         L_Omega_double[c], log_abs_L_Omega_double[c], log_phi_Z_recip[c], y1_log_prob[c],
                                                         log_prob_rowwise_prod_temp, log_abs_ys, sign_ys, log_abs_ym, sign_ym, log_common_grad_term_1,
                                                         log_abs_z_grad_term, sign_z_grad_term, log_abs_grad_prob, sign_grad_prob, log_abs_prod, sign_prod,
                                                         log_abs_prod_comp, sign_prod_comp, log_abs_dcc, sign_dcc, log_sum_result, sign_sum_result,
                                                         log_terms, sign_terms, log_abs_a, log_abs_b, sign_a, sign_b, container_max_logs, container_sum_exp_signed);
            for (int t1 = 0; t1 < n_tests; t1++) {
              for (int t2 = 0; t2 < t1 + 1; t2++) {
                const LogSumSignedResult_T r = log_sum_vec_signed_T<vec>(log_abs_Omega_grad_array_for_each_n[t1].col(t2), sign_Omega_grad_array_for_each_n[t1].col(t2));
                U_Omega_grad_array[c](t1, t2) += std::exp(r.log_sum) * r.sign;
              }
            }
            reset_log_scale_scratch();
          }

          //// ---- prevalence ----
          if ((n_class > 1) && ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only"))) {
            //// the per-row weight prod_t prob_ct / prob_n (<= 1 / prev_c) is formed on the log scale,
            //// exp(sum_t log prob_ct - lp_n): the natural-scale form in
            //// fn_MVP_prev_multi_pop_accumulate_grad(prob[c], prob_n_recip, ...),
            //// prob[c].rowwise().prod() * prob_n_recip, is 0 * Inf = NaN once a row's log-likelihood is below
            //// ~ -745 (LC_MVP at intercept +/- 30 in the validation fixture). Same weight, same accumulation;
            //// the NoLog path keeps the natural-scale helper.
            V prev_gradient_weight_vec = y1_log_prob[c].rowwise().sum();
            prev_gradient_weight_vec.array() += log_prob_n_recip.array();
            apply_inplace<vec, Fn::exp>(prev_gradient_weight_vec);
            if (prev_grad_terms_by_row != nullptr) {
              //// row n's term to cell (n, c); the caller adds the terms to prev_grad_mat in row order
              prev_grad_terms_by_row->col(c).head(chunk_size) = prev_gradient_weight_vec;
            } else {
            for (int n = 0; n < chunk_size; ++n) prev_grad_mat(pop_ind(row_start + n),
                 c) += prev_gradient_weight_vec(n);
            }
          }
        }

}




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <Vec vec>
inline void fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process_T(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                const std::string &grad_option,
                                                                                const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                const int n_chunks_multiplier = 1
) {
  
        out_mat.setZero();
        const int N = y_ref.rows();
        const int n_tests = y_ref.cols();
        const int n_us = theta_us_vec_ref.rows();
        const int n_params_main = theta_main_vec_ref.rows();
        const int n_params = n_params_main + n_us;

        const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> &X = Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0];
        const bool exclude_priors      = Model_args_as_cpp_struct.Model_args_bools(0);
        const bool corr_force_positive = Model_args_as_cpp_struct.Model_args_bools(2);
        const bool corr_prior_beta     = Model_args_as_cpp_struct.Model_args_bools(3);
        const bool corr_prior_norm     = Model_args_as_cpp_struct.Model_args_bools(4);
        fn_stop_if_corr_prior_beta_and_corr_prior_norm_are_both_true(corr_prior_beta, corr_prior_norm,
                                                                 "MVP_lp_grad_log_scale_MD_AD_fns.hpp");
        const int n_class  = Model_args_as_cpp_struct.Model_args_ints(1);
        const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
        const double overflow_threshold  = Model_args_as_cpp_struct.Model_args_doubles(0);
        const double underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(1);
        const std::string &vect_type     = Model_args_as_cpp_struct.Model_args_strings(0);
        const std::string &J_grad_option = Model_args_as_cpp_struct.Model_args_strings(11);
        const KernelChoice kchoice = kernel_choice_from_args(Model_args_as_cpp_struct);

        const Eigen::Matrix<double, -1, 1> &lkj_cholesky_eta = Model_args_as_cpp_struct.Model_args_col_vecs_double[0];
        const Eigen::Matrix<double, -1, 1> &prev_prior_a     = Model_args_as_cpp_struct.Model_args_col_vecs_double[1];
        const Eigen::Matrix<double, -1, 1> &prev_prior_b     = Model_args_as_cpp_struct.Model_args_col_vecs_double[2];
        const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_mean = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_sd   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_for_corr_a  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_for_corr_b  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3];
        std::vector<Eigen::Matrix<double, -1, -1>>        lb_corr           = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4];
        const std::vector<Eigen::Matrix<double, -1, -1>> &ub_corr           = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5];
        const std::vector<Eigen::Matrix<double, -1, -1>> &known_values      = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6];
        const std::vector<Eigen::Matrix<int, -1, -1>>    &known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];

        if (corr_force_positive) {
          for (int c = 0; c < n_class; ++c) for (int i = 1; i < n_tests; ++i) for (int j = 0; j < i; ++j) lb_corr[c](i, j) = 0.0;
        }

        const int n_pops = Model_args_as_cpp_struct.Model_args_ints(6);
        const Eigen::Matrix<int, -1, 1> &pop_ind = Model_args_as_cpp_struct.Model_args_col_vecs_int[2];
        const int n_corrs = n_class * n_tests * (n_tests - 1) / 2;

        int n_covariates_total, n_covariates_max;
        if (n_class > 1) {
          n_covariates_total = n_covariates_per_outcome_vec.row(0).sum() + n_covariates_per_outcome_vec.row(1).sum();
          n_covariates_max = std::max(n_covariates_per_outcome_vec.row(0).maxCoeff(),
                                      n_covariates_per_outcome_vec.row(1).maxCoeff());
        } else {
          n_covariates_total = n_covariates_per_outcome_vec.sum();
          n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();
        }

        int vec_size;
        if (vect_type == "AVX512")      vec_size = 8;
        else if (vect_type == "AVX2")   vec_size = 4;
        else if (vect_type == "AVX")    vec_size = 2;
        else                            vec_size = 1;
        //// n_chunks_multiplier is 1 except in the PartialLog evaluations (the fallback, attempt 2 in
        //// MVP_lp_grad_multi_attempts.hpp, and the forced ones), which use the sample() argument
        //// n_chunks_multiplier_for_PartialLog_log_scale_evaluation (default 3) so that each
        //// (larger-per-individual) PartialLog chunk stays about the same size in memory as a NoLog chunk;
        //// the nuisance vector and its gradient are re-laid out between the stored chunk layout and this
        //// one (PartialLog_nuisance_chunk_relayout.hpp).
        //// Requested chunks are not capped; calculate_chunk_sizes preserves the existing SIMD packet width.
        const int n_chunks_used = n_chunks * std::max(1, n_chunks_multiplier);
        ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, n_chunks_used);
        const int chunk_size_orig   = chunk_size_info.chunk_size_orig;
        const int normal_chunk_size = chunk_size_info.normal_chunk_size;
        const int last_chunk_size   = chunk_size_info.last_chunk_size;
        const int n_total_chunks    = chunk_size_info.n_total_chunks;
        const int n_full_chunks     = chunk_size_info.n_full_chunks;

        //// ---- corrs / coeffs / prev unpack ----
        const Eigen::Matrix<double, -1, 1> Omega_raw_vec_double = theta_main_vec_ref.head(n_corrs);
        std::vector<Eigen::Matrix<double, -1, -1>> beta_double_array = vec_of_mats_double(n_covariates_max,
                                                                                          n_tests,
                                                                                          n_class);
        {
          int i = n_corrs;
          for (int c = 0; c < n_class; ++c) for (int t = 0; t < n_tests; ++t) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); ++k)
            beta_double_array[c](k, t) = theta_main_vec_ref(i++);
        }
        Eigen::Matrix<double, -1, 1> u_prev_raw(n_pops);
        if (n_class > 1) for (int g = 0; g < n_pops; ++g) u_prev_raw(g) = theta_main_vec_ref(n_corrs + n_covariates_total + g);

        //// ---- Omega ----
        double prior_densities_L_Omega_double = 0.0, log_det_J_L_Omega_double = 0.0;
        Eigen::Matrix<double, -1, 1> grad_Omega_raw_priors_and_log_det_J(n_corrs);
        const int dim_choose_2 = n_tests * (n_tests - 1) / 2;
        std::vector<Eigen::Matrix<double, -1, -1>> deriv_L_wrt_unc_full = vec_of_mats_double(dim_choose_2 + n_tests, dim_choose_2, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_double       = vec_of_mats_double(n_tests,
                                                                                             n_tests,
                                                                                             n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_recip_double = vec_of_mats_double(n_tests,
                                                                                             n_tests,
                                                                                             n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> Omega_double         = vec_of_mats_double(n_tests,
                                                                                             n_tests,
                                                                                             n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> Omega_unconstrained_double =
            fn_convert_std_vec_of_corrs_to_3d_array_double(Eigen_vec_to_std_vec(Omega_raw_vec_double),
                                                           n_tests,
                                                           n_class);

        for (int c = 0; c < n_class; ++c) {
          auto out = Pinkney_corr_master_dbl(n_tests,
                                             lb_corr[c],
                                             ub_corr[c],
                                             Omega_unconstrained_double[c],
                                             known_values_indicator[c],
                                             known_values[c]);
          L_Omega_double[c] = out.block(1, 0, n_tests, n_tests);
          Omega_double[c] = L_Omega_double[c] * L_Omega_double[c].transpose();
          for (int i = 1; i < n_tests; ++i) for (int j = 0; j < i; ++j)
            L_Omega_recip_double[c](i,
                                 j) = (std::abs(L_Omega_double[c](i, j)) > 1e-10) ? 1.0 / L_Omega_double[c](i,
                                 j) : 0.0;
        }

        if (J_grad_option == "num_diff") {
          for (int c = 0; c < n_class; ++c) {
            auto get_L_orig = [&](const Eigen::Matrix<double, -1, -1> &Omega_unc) {

              auto out = Pinkney_corr_master_dbl(n_tests,
                                                 lb_corr[c],
                                                 ub_corr[c],
                                                 Omega_unc,
                                                 known_values_indicator[c],
                                                 known_values[c]);
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
                for (int k = 0; k < n_tests; k++) for (int l = 0; l <= k; l++) {
               // const double deriv = (L_perturbed(k, l) - L_Omega_double[c](k, l)) / epsilon;
               // deriv_L_wrt_unc_full[c](cnt_1++, cnt_2) = std::isfinite(deriv) ? deriv : 0.0;
                  //// a non-finite derivative is stored as NaN, not as 0: the Omega gradient is then non-finite,
                  //// so the failed evaluation is flagged and the multi-attempt fallback runs:
                  double deriv = (L_perturbed(k, l) - L_Omega_double[c](k, l)) / epsilon;
                  if (!std::isfinite(deriv)) deriv = std::numeric_limits<double>::quiet_NaN();
                  deriv_L_wrt_unc_full[c](cnt_1++, cnt_2) = deriv;
                }
                cnt_2++;
              }
            }
          }
        }

        std::vector<Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_double = L_Omega_double;
        Eigen::Matrix<double, -1, 1> grad_prev_raw(n_pops);
        double prior_densities_prev_double = 0.0, log_det_J_prev_from_AD = 0.0;

        {   //// ---- AD block ----
          stan::math::start_nested();
          stan::math::var target_AD = 0.0;
          Eigen::Matrix<stan::math::var, -1, 1> Omega_raw_vec_var = stan::math::to_var(Omega_raw_vec_double);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> Omega_unconstrained_var =
              fn_convert_std_vec_of_corrs_to_3d_array_var(Eigen_vec_to_std_vec_var(Omega_raw_vec_var),
                                                          n_tests,
                                                          n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> L_Omega_var = vec_of_mats_var(n_tests,
                                                                                            n_tests,
                                                                                            n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> Omega_var   = vec_of_mats_var(n_tests,
                                                                                            n_tests,
                                                                                            n_class);
          {
            stan::math::var log_det_J_L_Omega = 0.0;
            for (int c = 0; c < n_class; ++c) {
              Eigen::Matrix<stan::math::var, -1, -1> Chol_Schur_outs = Pinkney_corr_master(n_tests, lb_corr[c], ub_corr[c], Omega_unconstrained_var[c], known_values_indicator[c], known_values[c]);
              L_Omega_var[c] = Chol_Schur_outs.block(1, 0, n_tests, n_tests);
              target_AD += Chol_Schur_outs(0, 0);
              log_det_J_L_Omega += Chol_Schur_outs(0, 0);
              Omega_var[c] = L_Omega_var[c] * L_Omega_var[c].transpose();
              for (int i = 1; i < n_tests; ++i) for (int j = 0; j < i; ++j) {
                if (known_values_indicator[c](i, j) == 1) {
                  stan::math::var kv = stan::math::normal_lpdf(Omega_var[c](i, j), 0.0, 10.0);
                  //// target_AD += kv;
                  //// ---- a prior: left out of the gradient with exclude_priors, as it is left out of the log
                  ////      density:
                  if (exclude_priors == false) target_AD += kv;
                  prior_densities_L_Omega_double += kv.val();
                }
              }
            }
            log_det_J_L_Omega_double += log_det_J_L_Omega.val();
          }
          {
            stan::math::var prior_densities_L_Omega = 0.0;
            for (int c = 0; c < n_class; ++c) {
              if (!corr_prior_beta && !corr_prior_norm) {
                prior_densities_L_Omega += stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c],
                                                                              lkj_cholesky_eta(c));
                //// ---- with known (fixed) correlations, the terms of the known pairs in the
                ////      Jacobian that lkj_corr_cholesky_lpdf contains are removed (the prior is
                ////      on the free correlations):
                fn_subtract_jacobian_terms_of_known_correlations_from_lkj_corr_cholesky_density(
                      prior_densities_L_Omega, L_Omega_var[c], known_values_indicator[c]);
              } else if (corr_prior_beta && !corr_prior_norm) {
                for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++)
                  prior_densities_L_Omega += stan::math::beta_lpdf((Omega_var[c](i, j) + 1) / 2,
                                                                   prior_for_corr_a[c](i, j),
                                                                   prior_for_corr_b[c](i, j));
                Eigen::Matrix<stan::math::var, -1, 1> jd(n_tests);
                //// ---- log Jacobian of the map from the free entries of L_Omega to the off-diagonal
                ////      entries of Omega: sum over i of (n_tests - 1 - i) * log(L_Omega(i, i)) for
                ////      0-indexed i, i.e. prod_r L_rr^(K - r) for 1-indexed r. The exponent
                ////      n_tests + 1 - (i + 1) of the commented-out line below has one power of L_Omega(i, i) too
                ////      many, i.e. an extra det(Omega)^(1/2) factor that pulls the correlations towards 0.
                //// for (int i = 0; i < n_tests; ++i) jd(i) = (n_tests + 1 - (i + 1)) *
                ////       stan::math::log(L_Omega_var[c](i, i));
                //// ---- with known (fixed) correlations the map is to the FREE off-diagonal
                ////      correlations only: the power of L_Omega(i, i) is the number of free
                ////      correlations below the diagonal in column i (n_tests - 1 - i when no
                ////      correlation is known):
                //// for (int i = 0; i < n_tests; ++i) {
                ////       jd(i) = (n_tests - 1 - i) *
                ////             stan::math::log(L_Omega_var[c](i, i));
                //// }
                for (int i = 0; i < n_tests; ++i) {
                      const int number_of_free_correlations_below_diagonal_in_column_i =
                            fn_number_of_free_correlations_below_diagonal_in_column(
                                  known_values_indicator[c], i);
                      jd(i) = number_of_free_correlations_below_diagonal_in_column_i *
                             stan::math::log(L_Omega_var[c](i, i));
                }
                //// ---- constant: (rho + 1) / 2 ~ Beta(a, b) gives rho the density
                ////      Beta((rho + 1) / 2 | a, b) / 2, i.e. -log(2) per correlation and
                ////      n_tests * (n_tests - 1) / 2 of them per class (the commented-out line below adds
                ////      + n_tests * log(2) instead; a constant, so the draws are the same):
                //// prior_densities_L_Omega += (n_tests * stan::math::log(2) + jd.sum());
                prior_densities_L_Omega += (- 0.5 * n_tests * (n_tests - 1) * stan::math::log(2.0) + jd.sum());
              } else if (!corr_prior_beta && corr_prior_norm) {
                for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++)
                  prior_densities_L_Omega += stan::math::normal_lpdf(Omega_var[c](i, j),
                                                                     prior_for_corr_a[c](i, j),
                                                                     prior_for_corr_b[c](i, j));
                Eigen::Matrix<stan::math::var, -1, 1> jd(n_tests);
                //// ---- log Jacobian of the map from the free entries of L_Omega to the off-diagonal
                ////      entries of Omega, as in the corr_prior_beta branch above (the commented-out exponent
                ////      n_tests + 1 - (i + 1) below has one power of L_Omega(i, i) too many):
                //// for (int i = 0; i < n_tests; ++i) jd(i) = (n_tests + 1 - (i + 1)) *
                ////       stan::math::log(L_Omega_var[c](i, i));
                //// ---- with known (fixed) correlations the map is to the FREE off-diagonal
                ////      correlations only: the power of L_Omega(i, i) is the number of free
                ////      correlations below the diagonal in column i (n_tests - 1 - i when no
                ////      correlation is known):
                //// for (int i = 0; i < n_tests; ++i) {
                ////       jd(i) = (n_tests - 1 - i) *
                ////             stan::math::log(L_Omega_var[c](i, i));
                //// }
                for (int i = 0; i < n_tests; ++i) {
                      const int number_of_free_correlations_below_diagonal_in_column_i =
                            fn_number_of_free_correlations_below_diagonal_in_column(
                                  known_values_indicator[c], i);
                      jd(i) = number_of_free_correlations_below_diagonal_in_column_i *
                             stan::math::log(L_Omega_var[c](i, i));
                }
                //// ---- constant: the normal prior is on rho itself, so there is no log(2) term
                ////      (the commented-out line below adds + n_tests * log(2); a constant, so the draws are the
                ////      same):
                //// prior_densities_L_Omega += (n_tests * stan::math::log(2) + jd.sum());
                prior_densities_L_Omega += jd.sum();
              }
            }
            //// target_AD += prior_densities_L_Omega;
            //// ---- the correlation prior: left out of the gradient with exclude_priors, as it is left out
            ////      of the log density (prior_densities):
            if (exclude_priors == false) target_AD += prior_densities_L_Omega;
            prior_densities_L_Omega_double += prior_densities_L_Omega.val();
            target_AD.grad();
            grad_Omega_raw_priors_and_log_det_J = Omega_raw_vec_var.adj();
            out_mat.segment(1 + n_us, n_corrs) = grad_Omega_raw_priors_and_log_det_J;
            stan::math::set_zero_all_adjoints_nested();
          }
          if (n_class > 1) {
            fn_MVP_prev_multi_pop_AD(u_prev_raw,
                                     prev_prior_a,
                                     prev_prior_b,
                                     n_pops,
                                     prior_densities_prev_double,
                                     log_det_J_prev_from_AD,
                                     grad_prev_raw);
            //// ---- the prevalence prior is a prior: with exclude_priors its gradient is left out, as its
            ////      density is left out of the log density (the Jacobian gradient is added below):
            if (exclude_priors) grad_prev_raw.setZero();
            out_mat.segment(1 + n_us + n_corrs + n_covariates_total, n_pops) = grad_prev_raw;
          }
          if (J_grad_option == "autodiff") {
            for (int c = 0; c < n_class; ++c) {
              int cnt_1 = 0;
              for (int k = 0; k < n_tests; k++) for (int l = 0; l < k + 1; l++) {
                (L_Omega_var[c](k, l)).grad();
                int cnt_2 = 0;
                for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++) deriv_L_wrt_unc_full[c](cnt_1,
                     cnt_2++) = Omega_unconstrained_var[c](i,
                     j).adj();
                stan::math::set_zero_all_adjoints_nested();
                cnt_1++;
              }
            }
          }
          for (int c = 0; c < n_class; ++c) for (int t1 = 0; t1 < n_tests; ++t1) for (int t2 = 0; t2 < n_tests; ++t2) {
            L_Omega_double[c](t1, t2)         = L_Omega_var[c](t1, t2).val();
            log_abs_L_Omega_double[c](t1, t2) = std::log(std::fabs(L_Omega_double[c](t1, t2)));
            L_Omega_recip_double[c](t1, t2)   = 1.0 / L_Omega_double[c](t1, t2);
          }
          stan::math::recover_memory_nested();
        }

        //// ---- prev ----
        Eigen::Matrix<double, -1, -1> prev_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        Eigen::Matrix<double, -1, -1> log_prev_mat_small = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        Eigen::Matrix<double, -1, 1>  tanh_u_prev_vec(n_pops), deriv_p_wrt_u_vec(n_pops);
        double log_det_J_prev_double_total = 0.0;
        if (n_class > 1) {
          for (int g = 0; g < n_pops; ++g) {
            tanh_u_prev_vec(g) = stan::math::tanh(u_prev_raw(g));
            const double prev_g = 0.5 * (tanh_u_prev_vec(g) + 1.0);
            prev_mat(g, 1) = prev_g; prev_mat(g, 0) = 1.0 - prev_g;
            deriv_p_wrt_u_vec(g) = 0.5 * (1.0 - tanh_u_prev_vec(g) * tanh_u_prev_vec(g));
            log_det_J_prev_double_total += stan::math::log(deriv_p_wrt_u_vec(g));
          }
          log_prev_mat_small = stan::math::log(prev_mat);
        }

        //// ---- priors ----
        double prior_densities = 0.0;
        if (!exclude_priors) {
          for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++)
            prior_densities += stan::math::normal_lpdf(beta_double_array[c](k, t),
                                                       prior_coeffs_mean[c](k, t),
                                                       prior_coeffs_sd[c](k, t));
          prior_densities += prior_densities_L_Omega_double + prior_densities_prev_double;
        }

        //// ---- likelihood: chunk loop (remainder is just the last iteration) ----
        double log_prob_out = 0.0;
        const double log_det_J_main = log_det_J_prev_double_total + log_det_J_L_Omega_double;
        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec = theta_us_vec_ref;
        std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array    = vec_of_mats<double>(n_covariates_max,
                                                                                            n_tests,
                                                                                            n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array = vec_of_mats<double>(n_tests,
                                                                                            n_tests,
                                                                                            n_class);
        Eigen::Matrix<double, -1, -1> prev_grad_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        double log_jac_u = 0.0;

        for (int nc = 0; nc < n_total_chunks; nc++) {
          const int chunk_size = (nc == n_full_chunks) ? last_chunk_size : normal_chunk_size;
          fn_lp_grad_MVP_LC_Pinkney_PartialLog_process_chunk_T<vec>(out_mat,
                                                                    u_unc_vec,
                                                                    y_ref,
                                                                    grad_option,
                                                                    nc,
                                                                    chunk_size,
                                                                    chunk_size_orig,
                                                                    
                                                                    N, n_params, n_tests, n_class, n_covariates_max, n_pops,
                                                                    overflow_threshold, underflow_threshold, n_covariates_per_outcome_vec, pop_ind, X,
                                                                    beta_double_array, L_Omega_double, L_Omega_recip_double, log_abs_L_Omega_double,
                                                                    prev_mat, log_prev_mat_small, log_jac_u, beta_grad_array, U_Omega_grad_array, prev_grad_mat,
                                                                    kchoice, Model_args_as_cpp_struct);
        }

        //// ---- post-loop ----
        Eigen::Matrix<double, -1, 1> prev_unconstrained_grad_vec_out = Eigen::Matrix<double, -1, 1>::Zero(n_pops);
        if (n_class > 1) {
          for (int g = 0; g < n_pops; ++g)
            prev_unconstrained_grad_vec_out(g) = (prev_grad_mat(g, 1) - prev_grad_mat(g, 0)) * deriv_p_wrt_u_vec(g) - 2.0 * tanh_u_prev_vec(g);
        }
        log_prob_out += out_mat.tail(N).sum() + log_jac_u + log_det_J_main;
        if (!exclude_priors) log_prob_out += prior_densities;

        Eigen::Matrix<double, -1, 1> beta_grad_vec = Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);
        { int i = 0; for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++) beta_grad_vec(i++) = beta_grad_array[c](k, t); }

        Eigen::Matrix<double, -1, 1> L_Omega_grad_vec(n_corrs + (n_class * n_tests));
        Eigen::Matrix<double, -1, 1> U_Omega_grad_vec(n_corrs);
        { int i = 0; for (int c = 0; c < n_class; c++) for (int t1 = 0; t1 < n_tests; t1++) for (int t2 = 0; t2 < t1 + 1; t2++) L_Omega_grad_vec(i++) = U_Omega_grad_array[c](t1, t2); }
        if (n_class > 1) {
          const Eigen::Matrix<double, -1, 1> g_nd = L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
          const Eigen::Matrix<double, -1, 1> g_d  = L_Omega_grad_vec.segment(dim_choose_2 + n_tests,
                                                                             dim_choose_2 + n_tests);
          U_Omega_grad_vec.head(dim_choose_2)                  = (g_nd.transpose() * deriv_L_wrt_unc_full[0]).transpose();
          U_Omega_grad_vec.segment(dim_choose_2,
                                   dim_choose_2) = (g_d.transpose()  * deriv_L_wrt_unc_full[1]).transpose();
        } else {
          const Eigen::Matrix<double, -1, 1> g_nd = L_Omega_grad_vec.head(dim_choose_2 + n_tests);
          U_Omega_grad_vec.head(dim_choose_2) = (g_nd.transpose() * deriv_L_wrt_unc_full[0]).transpose();
        }

        out_mat(0) = log_prob_out;
        out_mat.segment(1 + n_us, n_corrs).array() += U_Omega_grad_vec.array();
        out_mat.segment(1 + n_us + n_corrs, n_covariates_total) = beta_grad_vec;
        if (n_class > 1) out_mat.segment(1 + n_us + n_corrs + n_covariates_total,
            n_pops) += prev_unconstrained_grad_vec_out;

        if (!exclude_priors) {
          int i = n_us + n_corrs + 1;
          for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++)
            out_mat(i++) += -((beta_double_array[c](k, t) - prior_coeffs_mean[c](k, t)) / prior_coeffs_sd[c](k, t)) * (1.0 / prior_coeffs_sd[c](k, t));
        }

}




//// -------------------------------------------------------------------------------------
//// Serial-order parallel partial-log lp+grad (the step-size search, fn_find_initial_eps_main_and_us with
//// n_threads > 1). Everything outside the chunk loop is
//// fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process_T above, line for line. The chunk loop is
//// replaced by: every chunk (the remainder too; each chunk allocates its own containers) on
//// n_threads_serial_order threads through fn_lp_grad_MVP_LC_Pinkney_PartialLog_process_chunk_T, each writing its
//// contributions to its own slots (log_jac_u and the beta / L_Omega gradient cells, which the serial loop
//// increments exactly once per chunk, and the per-row prevalence terms, which it adds one row at a time); then
//// the contributions are added to the accumulators in chunk order, rows in row order. These are the additions of
//// the serial loop in the same order, so lp and the gradient are bitwise those of the serial evaluation for any
//// thread count (slots start at +0.0, which is exact because an accumulator that starts at +0.0 never becomes
//// -0.0). If any chunk throws, the serial evaluation is run from the start instead (and throws where the serial
//// evaluation throws).
//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process_serial_order_parallel_T(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                                    const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                                    const std::string &grad_option,
                                                                                                    const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                                    const int n_threads_serial_order,
                                                                                                    const int n_chunks_multiplier = 1
) {
  
        out_mat.setZero();
        const int N = y_ref.rows();
        const int n_tests = y_ref.cols();
        const int n_us = theta_us_vec_ref.rows();
        const int n_params_main = theta_main_vec_ref.rows();
        const int n_params = n_params_main + n_us;

        const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> &X = Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0];
        const bool exclude_priors      = Model_args_as_cpp_struct.Model_args_bools(0);
        const bool corr_force_positive = Model_args_as_cpp_struct.Model_args_bools(2);
        const bool corr_prior_beta     = Model_args_as_cpp_struct.Model_args_bools(3);
        const bool corr_prior_norm     = Model_args_as_cpp_struct.Model_args_bools(4);
        fn_stop_if_corr_prior_beta_and_corr_prior_norm_are_both_true(corr_prior_beta, corr_prior_norm,
                                                                 "MVP_lp_grad_log_scale_MD_AD_fns.hpp");
        const int n_class  = Model_args_as_cpp_struct.Model_args_ints(1);
        const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
        const double overflow_threshold  = Model_args_as_cpp_struct.Model_args_doubles(0);
        const double underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(1);
        const std::string &vect_type     = Model_args_as_cpp_struct.Model_args_strings(0);
        const std::string &J_grad_option = Model_args_as_cpp_struct.Model_args_strings(11);
        const KernelChoice kchoice = kernel_choice_from_args(Model_args_as_cpp_struct);

        const Eigen::Matrix<double, -1, 1> &lkj_cholesky_eta = Model_args_as_cpp_struct.Model_args_col_vecs_double[0];
        const Eigen::Matrix<double, -1, 1> &prev_prior_a     = Model_args_as_cpp_struct.Model_args_col_vecs_double[1];
        const Eigen::Matrix<double, -1, 1> &prev_prior_b     = Model_args_as_cpp_struct.Model_args_col_vecs_double[2];
        const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_mean = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_sd   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_for_corr_a  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2];
        const std::vector<Eigen::Matrix<double, -1, -1>> &prior_for_corr_b  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3];
        std::vector<Eigen::Matrix<double, -1, -1>>        lb_corr           = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4];
        const std::vector<Eigen::Matrix<double, -1, -1>> &ub_corr           = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5];
        const std::vector<Eigen::Matrix<double, -1, -1>> &known_values      = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6];
        const std::vector<Eigen::Matrix<int, -1, -1>>    &known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];

        if (corr_force_positive) {
          for (int c = 0; c < n_class; ++c) for (int i = 1; i < n_tests; ++i) for (int j = 0; j < i; ++j) lb_corr[c](i, j) = 0.0;
        }

        const int n_pops = Model_args_as_cpp_struct.Model_args_ints(6);
        const Eigen::Matrix<int, -1, 1> &pop_ind = Model_args_as_cpp_struct.Model_args_col_vecs_int[2];
        const int n_corrs = n_class * n_tests * (n_tests - 1) / 2;

        int n_covariates_total, n_covariates_max;
        if (n_class > 1) {
          n_covariates_total = n_covariates_per_outcome_vec.row(0).sum() + n_covariates_per_outcome_vec.row(1).sum();
          n_covariates_max = std::max(n_covariates_per_outcome_vec.row(0).maxCoeff(),
                                      n_covariates_per_outcome_vec.row(1).maxCoeff());
        } else {
          n_covariates_total = n_covariates_per_outcome_vec.sum();
          n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();
        }

        int vec_size;
        if (vect_type == "AVX512")      vec_size = 8;
        else if (vect_type == "AVX2")   vec_size = 4;
        else if (vect_type == "AVX")    vec_size = 2;
        else                            vec_size = 1;
        //// n_chunks_multiplier is 1 except in the PartialLog evaluations (the fallback, attempt 2 in
        //// MVP_lp_grad_multi_attempts.hpp, and the forced ones), which use the sample() argument
        //// n_chunks_multiplier_for_PartialLog_log_scale_evaluation (default 3) so that each
        //// (larger-per-individual) PartialLog chunk stays about the same size in memory as a NoLog chunk;
        //// the nuisance vector and its gradient are re-laid out between the stored chunk layout and this
        //// one (PartialLog_nuisance_chunk_relayout.hpp).
        //// Requested chunks are not capped; calculate_chunk_sizes preserves the existing SIMD packet width.
        const int n_chunks_used = n_chunks * std::max(1, n_chunks_multiplier);
        ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, n_chunks_used);
        const int chunk_size_orig   = chunk_size_info.chunk_size_orig;
        const int normal_chunk_size = chunk_size_info.normal_chunk_size;
        const int last_chunk_size   = chunk_size_info.last_chunk_size;
        const int n_total_chunks    = chunk_size_info.n_total_chunks;
        const int n_full_chunks     = chunk_size_info.n_full_chunks;

        //// ---- corrs / coeffs / prev unpack ----
        const Eigen::Matrix<double, -1, 1> Omega_raw_vec_double = theta_main_vec_ref.head(n_corrs);
        std::vector<Eigen::Matrix<double, -1, -1>> beta_double_array = vec_of_mats_double(n_covariates_max,
                                                                                          n_tests,
                                                                                          n_class);
        {
          int i = n_corrs;
          for (int c = 0; c < n_class; ++c) for (int t = 0; t < n_tests; ++t) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); ++k)
            beta_double_array[c](k, t) = theta_main_vec_ref(i++);
        }
        Eigen::Matrix<double, -1, 1> u_prev_raw(n_pops);
        if (n_class > 1) for (int g = 0; g < n_pops; ++g) u_prev_raw(g) = theta_main_vec_ref(n_corrs + n_covariates_total + g);

        //// ---- Omega ----
        double prior_densities_L_Omega_double = 0.0, log_det_J_L_Omega_double = 0.0;
        Eigen::Matrix<double, -1, 1> grad_Omega_raw_priors_and_log_det_J(n_corrs);
        const int dim_choose_2 = n_tests * (n_tests - 1) / 2;
        std::vector<Eigen::Matrix<double, -1, -1>> deriv_L_wrt_unc_full = vec_of_mats_double(dim_choose_2 + n_tests, dim_choose_2, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_double       = vec_of_mats_double(n_tests,
                                                                                             n_tests,
                                                                                             n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_recip_double = vec_of_mats_double(n_tests,
                                                                                             n_tests,
                                                                                             n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> Omega_double         = vec_of_mats_double(n_tests,
                                                                                             n_tests,
                                                                                             n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> Omega_unconstrained_double =
            fn_convert_std_vec_of_corrs_to_3d_array_double(Eigen_vec_to_std_vec(Omega_raw_vec_double),
                                                           n_tests,
                                                           n_class);

        for (int c = 0; c < n_class; ++c) {
          auto out = Pinkney_corr_master_dbl(n_tests,
                                             lb_corr[c],
                                             ub_corr[c],
                                             Omega_unconstrained_double[c],
                                             known_values_indicator[c],
                                             known_values[c]);
          L_Omega_double[c] = out.block(1, 0, n_tests, n_tests);
          Omega_double[c] = L_Omega_double[c] * L_Omega_double[c].transpose();
          for (int i = 1; i < n_tests; ++i) for (int j = 0; j < i; ++j)
            L_Omega_recip_double[c](i,
                                 j) = (std::abs(L_Omega_double[c](i, j)) > 1e-10) ? 1.0 / L_Omega_double[c](i,
                                 j) : 0.0;
        }

        if (J_grad_option == "num_diff") {
          for (int c = 0; c < n_class; ++c) {
            auto get_L_orig = [&](const Eigen::Matrix<double, -1, -1> &Omega_unc) {

              auto out = Pinkney_corr_master_dbl(n_tests,
                                                 lb_corr[c],
                                                 ub_corr[c],
                                                 Omega_unc,
                                                 known_values_indicator[c],
                                                 known_values[c]);
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
                for (int k = 0; k < n_tests; k++) for (int l = 0; l <= k; l++) {
               // const double deriv = (L_perturbed(k, l) - L_Omega_double[c](k, l)) / epsilon;
               // deriv_L_wrt_unc_full[c](cnt_1++, cnt_2) = std::isfinite(deriv) ? deriv : 0.0;
                  //// a non-finite derivative is stored as NaN, not as 0: the Omega gradient is then non-finite,
                  //// so the failed evaluation is flagged and the multi-attempt fallback runs:
                  double deriv = (L_perturbed(k, l) - L_Omega_double[c](k, l)) / epsilon;
                  if (!std::isfinite(deriv)) deriv = std::numeric_limits<double>::quiet_NaN();
                  deriv_L_wrt_unc_full[c](cnt_1++, cnt_2) = deriv;
                }
                cnt_2++;
              }
            }
          }
        }

        std::vector<Eigen::Matrix<double, -1, -1>> log_abs_L_Omega_double = L_Omega_double;
        Eigen::Matrix<double, -1, 1> grad_prev_raw(n_pops);
        double prior_densities_prev_double = 0.0, log_det_J_prev_from_AD = 0.0;

        {   //// ---- AD block ----
          stan::math::start_nested();
          stan::math::var target_AD = 0.0;
          Eigen::Matrix<stan::math::var, -1, 1> Omega_raw_vec_var = stan::math::to_var(Omega_raw_vec_double);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> Omega_unconstrained_var =
              fn_convert_std_vec_of_corrs_to_3d_array_var(Eigen_vec_to_std_vec_var(Omega_raw_vec_var),
                                                          n_tests,
                                                          n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> L_Omega_var = vec_of_mats_var(n_tests,
                                                                                            n_tests,
                                                                                            n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> Omega_var   = vec_of_mats_var(n_tests,
                                                                                            n_tests,
                                                                                            n_class);
          {
            stan::math::var log_det_J_L_Omega = 0.0;
            for (int c = 0; c < n_class; ++c) {
              Eigen::Matrix<stan::math::var, -1, -1> Chol_Schur_outs = Pinkney_corr_master(n_tests, lb_corr[c], ub_corr[c], Omega_unconstrained_var[c], known_values_indicator[c], known_values[c]);
              L_Omega_var[c] = Chol_Schur_outs.block(1, 0, n_tests, n_tests);
              target_AD += Chol_Schur_outs(0, 0);
              log_det_J_L_Omega += Chol_Schur_outs(0, 0);
              Omega_var[c] = L_Omega_var[c] * L_Omega_var[c].transpose();
              for (int i = 1; i < n_tests; ++i) for (int j = 0; j < i; ++j) {
                if (known_values_indicator[c](i, j) == 1) {
                  stan::math::var kv = stan::math::normal_lpdf(Omega_var[c](i, j), 0.0, 10.0);
                  //// target_AD += kv;
                  //// ---- a prior: left out of the gradient with exclude_priors, as it is left out of the log
                  ////      density:
                  if (exclude_priors == false) target_AD += kv;
                  prior_densities_L_Omega_double += kv.val();
                }
              }
            }
            log_det_J_L_Omega_double += log_det_J_L_Omega.val();
          }
          {
            stan::math::var prior_densities_L_Omega = 0.0;
            for (int c = 0; c < n_class; ++c) {
              if (!corr_prior_beta && !corr_prior_norm) {
                prior_densities_L_Omega += stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c],
                                                                              lkj_cholesky_eta(c));
                //// ---- with known (fixed) correlations, the terms of the known pairs in the
                ////      Jacobian that lkj_corr_cholesky_lpdf contains are removed (the prior is
                ////      on the free correlations):
                fn_subtract_jacobian_terms_of_known_correlations_from_lkj_corr_cholesky_density(
                      prior_densities_L_Omega, L_Omega_var[c], known_values_indicator[c]);
              } else if (corr_prior_beta && !corr_prior_norm) {
                for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++)
                  prior_densities_L_Omega += stan::math::beta_lpdf((Omega_var[c](i, j) + 1) / 2,
                                                                   prior_for_corr_a[c](i, j),
                                                                   prior_for_corr_b[c](i, j));
                Eigen::Matrix<stan::math::var, -1, 1> jd(n_tests);
                //// ---- log Jacobian of the map from the free entries of L_Omega to the off-diagonal
                ////      entries of Omega: sum over i of (n_tests - 1 - i) * log(L_Omega(i, i)) for
                ////      0-indexed i, i.e. prod_r L_rr^(K - r) for 1-indexed r. The exponent
                ////      n_tests + 1 - (i + 1) of the commented-out line below has one power of L_Omega(i, i) too
                ////      many, i.e. an extra det(Omega)^(1/2) factor that pulls the correlations towards 0.
                //// for (int i = 0; i < n_tests; ++i) jd(i) = (n_tests + 1 - (i + 1)) *
                ////       stan::math::log(L_Omega_var[c](i, i));
                //// ---- with known (fixed) correlations the map is to the FREE off-diagonal
                ////      correlations only: the power of L_Omega(i, i) is the number of free
                ////      correlations below the diagonal in column i (n_tests - 1 - i when no
                ////      correlation is known):
                //// for (int i = 0; i < n_tests; ++i) {
                ////       jd(i) = (n_tests - 1 - i) *
                ////             stan::math::log(L_Omega_var[c](i, i));
                //// }
                for (int i = 0; i < n_tests; ++i) {
                      const int number_of_free_correlations_below_diagonal_in_column_i =
                            fn_number_of_free_correlations_below_diagonal_in_column(
                                  known_values_indicator[c], i);
                      jd(i) = number_of_free_correlations_below_diagonal_in_column_i *
                             stan::math::log(L_Omega_var[c](i, i));
                }
                //// ---- constant: (rho + 1) / 2 ~ Beta(a, b) gives rho the density
                ////      Beta((rho + 1) / 2 | a, b) / 2, i.e. -log(2) per correlation and
                ////      n_tests * (n_tests - 1) / 2 of them per class (the commented-out line below adds
                ////      + n_tests * log(2) instead; a constant, so the draws are the same):
                //// prior_densities_L_Omega += (n_tests * stan::math::log(2) + jd.sum());
                prior_densities_L_Omega += (- 0.5 * n_tests * (n_tests - 1) * stan::math::log(2.0) + jd.sum());
              } else if (!corr_prior_beta && corr_prior_norm) {
                for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++)
                  prior_densities_L_Omega += stan::math::normal_lpdf(Omega_var[c](i, j),
                                                                     prior_for_corr_a[c](i, j),
                                                                     prior_for_corr_b[c](i, j));
                Eigen::Matrix<stan::math::var, -1, 1> jd(n_tests);
                //// ---- log Jacobian of the map from the free entries of L_Omega to the off-diagonal
                ////      entries of Omega, as in the corr_prior_beta branch above (the commented-out exponent
                ////      n_tests + 1 - (i + 1) below has one power of L_Omega(i, i) too many):
                //// for (int i = 0; i < n_tests; ++i) jd(i) = (n_tests + 1 - (i + 1)) *
                ////       stan::math::log(L_Omega_var[c](i, i));
                //// ---- with known (fixed) correlations the map is to the FREE off-diagonal
                ////      correlations only: the power of L_Omega(i, i) is the number of free
                ////      correlations below the diagonal in column i (n_tests - 1 - i when no
                ////      correlation is known):
                //// for (int i = 0; i < n_tests; ++i) {
                ////       jd(i) = (n_tests - 1 - i) *
                ////             stan::math::log(L_Omega_var[c](i, i));
                //// }
                for (int i = 0; i < n_tests; ++i) {
                      const int number_of_free_correlations_below_diagonal_in_column_i =
                            fn_number_of_free_correlations_below_diagonal_in_column(
                                  known_values_indicator[c], i);
                      jd(i) = number_of_free_correlations_below_diagonal_in_column_i *
                             stan::math::log(L_Omega_var[c](i, i));
                }
                //// ---- constant: the normal prior is on rho itself, so there is no log(2) term
                ////      (the commented-out line below adds + n_tests * log(2); a constant, so the draws are the
                ////      same):
                //// prior_densities_L_Omega += (n_tests * stan::math::log(2) + jd.sum());
                prior_densities_L_Omega += jd.sum();
              }
            }
            //// target_AD += prior_densities_L_Omega;
            //// ---- the correlation prior: left out of the gradient with exclude_priors, as it is left out
            ////      of the log density (prior_densities):
            if (exclude_priors == false) target_AD += prior_densities_L_Omega;
            prior_densities_L_Omega_double += prior_densities_L_Omega.val();
            target_AD.grad();
            grad_Omega_raw_priors_and_log_det_J = Omega_raw_vec_var.adj();
            out_mat.segment(1 + n_us, n_corrs) = grad_Omega_raw_priors_and_log_det_J;
            stan::math::set_zero_all_adjoints_nested();
          }
          if (n_class > 1) {
            fn_MVP_prev_multi_pop_AD(u_prev_raw,
                                     prev_prior_a,
                                     prev_prior_b,
                                     n_pops,
                                     prior_densities_prev_double,
                                     log_det_J_prev_from_AD,
                                     grad_prev_raw);
            //// ---- the prevalence prior is a prior: with exclude_priors its gradient is left out, as its
            ////      density is left out of the log density (the Jacobian gradient is added below):
            if (exclude_priors) grad_prev_raw.setZero();
            out_mat.segment(1 + n_us + n_corrs + n_covariates_total, n_pops) = grad_prev_raw;
          }
          if (J_grad_option == "autodiff") {
            for (int c = 0; c < n_class; ++c) {
              int cnt_1 = 0;
              for (int k = 0; k < n_tests; k++) for (int l = 0; l < k + 1; l++) {
                (L_Omega_var[c](k, l)).grad();
                int cnt_2 = 0;
                for (int i = 1; i < n_tests; i++) for (int j = 0; j < i; j++) deriv_L_wrt_unc_full[c](cnt_1,
                     cnt_2++) = Omega_unconstrained_var[c](i,
                     j).adj();
                stan::math::set_zero_all_adjoints_nested();
                cnt_1++;
              }
            }
          }
          for (int c = 0; c < n_class; ++c) for (int t1 = 0; t1 < n_tests; ++t1) for (int t2 = 0; t2 < n_tests; ++t2) {
            L_Omega_double[c](t1, t2)         = L_Omega_var[c](t1, t2).val();
            log_abs_L_Omega_double[c](t1, t2) = std::log(std::fabs(L_Omega_double[c](t1, t2)));
            L_Omega_recip_double[c](t1, t2)   = 1.0 / L_Omega_double[c](t1, t2);
          }
          stan::math::recover_memory_nested();
        }

        //// ---- prev ----
        Eigen::Matrix<double, -1, -1> prev_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        Eigen::Matrix<double, -1, -1> log_prev_mat_small = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        Eigen::Matrix<double, -1, 1>  tanh_u_prev_vec(n_pops), deriv_p_wrt_u_vec(n_pops);
        double log_det_J_prev_double_total = 0.0;
        if (n_class > 1) {
          for (int g = 0; g < n_pops; ++g) {
            tanh_u_prev_vec(g) = stan::math::tanh(u_prev_raw(g));
            const double prev_g = 0.5 * (tanh_u_prev_vec(g) + 1.0);
            prev_mat(g, 1) = prev_g; prev_mat(g, 0) = 1.0 - prev_g;
            deriv_p_wrt_u_vec(g) = 0.5 * (1.0 - tanh_u_prev_vec(g) * tanh_u_prev_vec(g));
            log_det_J_prev_double_total += stan::math::log(deriv_p_wrt_u_vec(g));
          }
          log_prev_mat_small = stan::math::log(prev_mat);
        }

        //// ---- priors ----
        double prior_densities = 0.0;
        if (!exclude_priors) {
          for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++)
            prior_densities += stan::math::normal_lpdf(beta_double_array[c](k, t),
                                                       prior_coeffs_mean[c](k, t),
                                                       prior_coeffs_sd[c](k, t));
          prior_densities += prior_densities_L_Omega_double + prior_densities_prev_double;
        }

        //// ---- likelihood: chunk loop (remainder is just the last iteration) ----
        double log_prob_out = 0.0;
        const double log_det_J_main = log_det_J_prev_double_total + log_det_J_L_Omega_double;
        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec = theta_us_vec_ref;
        std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array    = vec_of_mats<double>(n_covariates_max,
                                                                                            n_tests,
                                                                                            n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array = vec_of_mats<double>(n_tests,
                                                                                            n_tests,
                                                                                            n_class);
        Eigen::Matrix<double, -1, -1> prev_grad_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        double log_jac_u = 0.0;

        //// ---- chunks in parallel; contributions added in the serial order ----
        {
          const int n_threads_used = std::max(1, std::min(n_threads_serial_order, n_total_chunks));
          const bool prev_grad_needed = (n_class > 1) && ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only"));
          ////
          std::vector<double>                                     log_jac_u_per_chunk(n_total_chunks, 0.0);
          std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> beta_grad_per_chunk(n_total_chunks);
          std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> U_Omega_grad_per_chunk(n_total_chunks);
          std::vector<Eigen::Matrix<double, -1, -1>>              prev_grad_terms_per_chunk(n_total_chunks);
          std::vector<int>                                        chunk_threw(n_total_chunks, 0);

          #pragma omp parallel for num_threads(n_threads_used) schedule(static) if (n_threads_used > 1)
          for (int nc = 0; nc < n_total_chunks; nc++) {
            try {
                  const int chunk_size = (nc == n_full_chunks) ? last_chunk_size : normal_chunk_size;
                  Eigen::Matrix<double, -1, -1> prev_grad_mat_not_used = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
                  beta_grad_per_chunk[nc]       = vec_of_mats<double>(n_covariates_max, n_tests, n_class);
                  U_Omega_grad_per_chunk[nc]    = vec_of_mats<double>(n_tests, n_tests, n_class);
                  prev_grad_terms_per_chunk[nc] = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_class);
                  fn_lp_grad_MVP_LC_Pinkney_PartialLog_process_chunk_T<vec>(out_mat,
                                                                            u_unc_vec,
                                                                            y_ref,
                                                                            grad_option,
                                                                            nc,
                                                                            chunk_size,
                                                                            chunk_size_orig,
                                                                            
                                                                            N, n_params, n_tests, n_class, n_covariates_max, n_pops,
                                                                            overflow_threshold, underflow_threshold, n_covariates_per_outcome_vec, pop_ind, X,
                                                                            beta_double_array, L_Omega_double, L_Omega_recip_double, log_abs_L_Omega_double,
                                                                            prev_mat, log_prev_mat_small, log_jac_u_per_chunk[nc],
                                                                            beta_grad_per_chunk[nc], U_Omega_grad_per_chunk[nc], prev_grad_mat_not_used,
                                                                            kchoice, Model_args_as_cpp_struct, &prev_grad_terms_per_chunk[nc]);
            } catch (...) {
                  chunk_threw[nc] = 1;
            }
          }

          //// ---- a chunk threw: the serial evaluation from the start instead
          for (int nc = 0; nc < n_total_chunks; nc++) {
            if (chunk_threw[nc] != 0) {
                  fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process_T<vec>(out_mat,
                                                                                        theta_main_vec_ref,
                                                                                        theta_us_vec_ref,
                                                                                        y_ref,
                                                                                        
                                                                                        grad_option, Model_args_as_cpp_struct, n_chunks_multiplier);
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

        //// ---- post-loop ----
        Eigen::Matrix<double, -1, 1> prev_unconstrained_grad_vec_out = Eigen::Matrix<double, -1, 1>::Zero(n_pops);
        if (n_class > 1) {
          for (int g = 0; g < n_pops; ++g)
            prev_unconstrained_grad_vec_out(g) = (prev_grad_mat(g, 1) - prev_grad_mat(g, 0)) * deriv_p_wrt_u_vec(g) - 2.0 * tanh_u_prev_vec(g);
        }
        log_prob_out += out_mat.tail(N).sum() + log_jac_u + log_det_J_main;
        if (!exclude_priors) log_prob_out += prior_densities;

        Eigen::Matrix<double, -1, 1> beta_grad_vec = Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);
        { int i = 0; for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++) beta_grad_vec(i++) = beta_grad_array[c](k, t); }

        Eigen::Matrix<double, -1, 1> L_Omega_grad_vec(n_corrs + (n_class * n_tests));
        Eigen::Matrix<double, -1, 1> U_Omega_grad_vec(n_corrs);
        { int i = 0; for (int c = 0; c < n_class; c++) for (int t1 = 0; t1 < n_tests; t1++) for (int t2 = 0; t2 < t1 + 1; t2++) L_Omega_grad_vec(i++) = U_Omega_grad_array[c](t1, t2); }
        if (n_class > 1) {
          const Eigen::Matrix<double, -1, 1> g_nd = L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
          const Eigen::Matrix<double, -1, 1> g_d  = L_Omega_grad_vec.segment(dim_choose_2 + n_tests,
                                                                             dim_choose_2 + n_tests);
          U_Omega_grad_vec.head(dim_choose_2)                  = (g_nd.transpose() * deriv_L_wrt_unc_full[0]).transpose();
          U_Omega_grad_vec.segment(dim_choose_2,
                                   dim_choose_2) = (g_d.transpose()  * deriv_L_wrt_unc_full[1]).transpose();
        } else {
          const Eigen::Matrix<double, -1, 1> g_nd = L_Omega_grad_vec.head(dim_choose_2 + n_tests);
          U_Omega_grad_vec.head(dim_choose_2) = (g_nd.transpose() * deriv_L_wrt_unc_full[0]).transpose();
        }

        out_mat(0) = log_prob_out;
        out_mat.segment(1 + n_us, n_corrs).array() += U_Omega_grad_vec.array();
        out_mat.segment(1 + n_us + n_corrs, n_covariates_total) = beta_grad_vec;
        if (n_class > 1) out_mat.segment(1 + n_us + n_corrs + n_covariates_total,
            n_pops) += prev_unconstrained_grad_vec_out;

        if (!exclude_priors) {
          int i = n_us + n_corrs + 1;
          for (int c = 0; c < n_class; c++) for (int t = 0; t < n_tests; t++) for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++)
            out_mat(i++) += -((beta_double_array[c](k, t) - prior_coeffs_mean[c](k, t)) / prior_coeffs_sd[c](k, t)) * (1.0 / prior_coeffs_sd[c](k, t));
        }

}




//// -------------------------------------------------------------------------------------
//// Entry point with the ORIGINAL name/signature: the InPlace wrappers call this unchanged.
//// -------------------------------------------------------------------------------------
inline void fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process(   Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                              const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                              const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                              const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                              const std::string grad_option,
                                                                              const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                              std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                              const int n_threads_WCP,
                                                                            // const int n_chunks_multiplier = 1
                                                                              const int n_chunks_multiplier = -1
) {

        (void)LC_MVP_ws_structs; (void)n_threads_WCP;
        const Vec vec = vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0));
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

#ifdef NICOSTAN_SERIAL_ORDER_LP_GRAD
        //// inside a step-size search with n_threads > 1 (g_n_threads_serial_order_lp_grad, declared in
        //// native_api.cpp.in, which also defines NICOSTAN_SERIAL_ORDER_LP_GRAD): the serial-order parallel
        //// evaluation below, bitwise the same result
        // if (g_n_threads_serial_order_lp_grad > 1) {
        //       DISPATCH_VEC(vec,
        //             fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process_serial_order_parallel_T,
        //                    out_mat, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option,
        //                    Model_args_as_cpp_struct,
        //                    g_n_threads_serial_order_lp_grad, n_chunks_multiplier);
        //       return;
        // }
                    if (g_n_threads_serial_order_lp_grad > 1) {
                          DISPATCH_VEC( vec,
                    fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process_serial_order_parallel_T,
                                        out_mat_of_evaluation, theta_main_vec_ref, theta_us_of_evaluation, y_ref,
                                        grad_option, Model_args_as_cpp_struct, g_n_threads_serial_order_lp_grad,
                                        n_chunks_multiplier_used);
                          return;
                    }
#endif
        // DISPATCH_VEC(vec, fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process_T,
        //              out_mat, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option,
        //              Model_args_as_cpp_struct, n_chunks_multiplier);
                    DISPATCH_VEC( vec,
                                  fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process_T,
                                  out_mat_of_evaluation, theta_main_vec_ref, theta_us_of_evaluation, y_ref,
                                  grad_option, Model_args_as_cpp_struct, n_chunks_multiplier_used);

              });

}




// Internal function using Eigen::Ref as inputs for matrices
inline void     fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace(   Eigen::Matrix<double, -1, 1> &&out_mat_R_val,
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
      
      
      fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process( out_mat_ref,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref,
                                                                      y_ref,
                                                                      grad_option,
                                                                      Model_args_as_cpp_struct, 
                                                                      LC_MVP_ws_structs,
                                                                      n_threads_WCP);
      
}




// Internal function using Eigen::Ref as inputs for matrices
inline void     fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace(   Eigen::Matrix<double, -1, 1> &out_mat_ref,
                                                                   const Eigen::Matrix<double, -1, 1> &theta_main_vec_ref,
                                                                   const Eigen::Matrix<double, -1, 1> &theta_us_vec_ref,
                                                                   const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                   const std::string &grad_option,
                                                                   const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                   std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                   const int n_threads_WCP
) {
  
      fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process( out_mat_ref,
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
inline void     fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace(   Eigen::Ref<Eigen::Block<MatrixType, -1, 1>>  &out_mat_ref,
                                                                   const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_main_vec_ref,
                                                                   const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_us_vec_ref,
                                                                   const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                   const std::string &grad_option,
                                                                   const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                   std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                   const int n_threads_WCP
) {
      
      fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process( out_mat_ref,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref,
                                                                      y_ref,
                                                                      grad_option,
                                                                      Model_args_as_cpp_struct, 
                                                                      LC_MVP_ws_structs,
                                                                      n_threads_WCP);
  
}




// Internal function using Eigen::Ref as inputs for matrices
inline Eigen::Matrix<double, -1, 1>    fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD(  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
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
      
      fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace( out_mat,
                                                              theta_main_vec_ref,
                                                              theta_us_vec_ref,
                                                              y_ref,
                                                              grad_option,
                                                              Model_args_as_cpp_struct, 
                                                              LC_MVP_ws_structs,
                                                              n_threads_WCP);
      
      return out_mat;
  
}






















