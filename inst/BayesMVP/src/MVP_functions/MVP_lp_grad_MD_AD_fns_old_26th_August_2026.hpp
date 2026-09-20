
//// MVP_lp_grad_MD_AD_fns.hpp

#pragma once


 
 


#include <Eigen/Dense>
 
#include <unsupported/Eigen/SpecialFunctions>


 
 



 
 
//////////////////////////////////////////////////------------------------------------------------------------------------------------------------------------------------------
inline void fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                      const std::string &grad_option,
                                                                                      const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                      std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs
) {

        out_mat.setZero();
        ////
        const int N = y_ref.rows();
        const int n_tests = y_ref.cols();
        const int n_us = theta_us_vec_ref.rows();
        const int n_params_main = theta_main_vec_ref.rows();
        const int n_params = n_params_main + n_us;
        ////
        //////////////  access elements from struct and read
        const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>  &X =  Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0];
        ////
        const bool exclude_priors = Model_args_as_cpp_struct.Model_args_bools(0);
        const bool CI =             Model_args_as_cpp_struct.Model_args_bools(1);
        const bool corr_force_positive = Model_args_as_cpp_struct.Model_args_bools(2);
        const bool corr_prior_beta = Model_args_as_cpp_struct.Model_args_bools(3);
        const bool corr_prior_norm = Model_args_as_cpp_struct.Model_args_bools(4);
        const bool handle_numerical_issues = Model_args_as_cpp_struct.Model_args_bools(5);
        const bool skip_checks_exp =   Model_args_as_cpp_struct.Model_args_bools(6);
        const bool skip_checks_log =   Model_args_as_cpp_struct.Model_args_bools(7);
        const bool skip_checks_lse =   Model_args_as_cpp_struct.Model_args_bools(8);
        const bool skip_checks_tanh =  Model_args_as_cpp_struct.Model_args_bools(9);
        const bool skip_checks_Phi =  Model_args_as_cpp_struct.Model_args_bools(10);
        const bool skip_checks_log_Phi = Model_args_as_cpp_struct.Model_args_bools(11);
        const bool skip_checks_inv_Phi = Model_args_as_cpp_struct.Model_args_bools(12);
        const bool skip_checks_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_bools(13);
        const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
        ////
        const int n_cores = Model_args_as_cpp_struct.Model_args_ints(0);
        const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
        const int ub_threshold_phi_approx = Model_args_as_cpp_struct.Model_args_ints(2);
        const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);

       // const KernelChoice kchoice = kernel_choice_from_args(Model_args_as_cpp_struct); // ----

        ////
        // const double prev_prior_a = Model_args_as_cpp_struct.Model_args_doubles(0);
        // const double prev_prior_b = Model_args_as_cpp_struct.Model_args_doubles(1);
        const double overflow_threshold  = Model_args_as_cpp_struct.Model_args_doubles(0);
        const double underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(1);
        ////
        //// NOTE: these are non-const because they get overridden for the last chunk
        //// For WCP, each thread needs its own copies (handled below)
        ////
        std::string vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
        const std::string &Phi_type = Model_args_as_cpp_struct.Model_args_strings(1);
        const std::string &inv_Phi_type = Model_args_as_cpp_struct.Model_args_strings(2);
        std::string vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);
        std::string vect_type_log = Model_args_as_cpp_struct.Model_args_strings(4);
        std::string vect_type_lse = Model_args_as_cpp_struct.Model_args_strings(5);
        std::string vect_type_tanh = Model_args_as_cpp_struct.Model_args_strings(6);
        std::string vect_type_Phi = Model_args_as_cpp_struct.Model_args_strings(7);
        std::string vect_type_log_Phi = Model_args_as_cpp_struct.Model_args_strings(8);
        std::string vect_type_inv_Phi = Model_args_as_cpp_struct.Model_args_strings(9);
        std::string vect_type_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_strings(10);
        const std::string J_grad_option =  Model_args_as_cpp_struct.Model_args_strings(11);
        const std::string nuisance_transformation =   Model_args_as_cpp_struct.Model_args_strings(12);
        ////
        const Eigen::Matrix<double, -1, 1> &lkj_cholesky_eta = Model_args_as_cpp_struct.Model_args_col_vecs_double[0];
        const Eigen::Matrix<double, -1, 1> &prev_prior_a     = Model_args_as_cpp_struct.Model_args_col_vecs_double[1]; //// ---- for mult-pops
        const Eigen::Matrix<double, -1, 1> &prev_prior_b     = Model_args_as_cpp_struct.Model_args_col_vecs_double[2]; //// ---- for mult-pops
        ////
        const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
        ////
        // const Eigen::Matrix<double, -1, -1> &LT_b_priors_shape  = Model_args_as_cpp_struct.Model_args_mats_double[0];
        // const Eigen::Matrix<double, -1, -1> &LT_b_priors_scale  = Model_args_as_cpp_struct.Model_args_mats_double[1];
        // const Eigen::Matrix<double, -1, -1> &LT_known_bs_indicator = Model_args_as_cpp_struct.Model_args_mats_double[2];
        // const Eigen::Matrix<double, -1, -1> &LT_known_bs_values = Model_args_as_cpp_struct.Model_args_mats_double[3];
        ////
        std::vector<Eigen::Matrix<double, -1, -1>>   prior_coeffs_mean  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0];
        std::vector<Eigen::Matrix<double, -1, -1>>   prior_coeffs_sd   =  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1];
        const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_for_corr_a   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2];
        const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_for_corr_b   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3];
        std::vector<Eigen::Matrix<double, -1, -1>>   lb_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4];
        const std::vector<Eigen::Matrix<double, -1, -1>>   &ub_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5];
        const std::vector<Eigen::Matrix<double, -1, -1>>   &known_values    = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6];
        ////
        const std::vector<Eigen::Matrix<int, -1, -1 >> &known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];

        // const std::vector<Eigen::Matrix<int, -1, 1 >> &perm     = Model_args_as_cpp_struct.Model_args_vecs_of_col_vecs_int[0];
        // const std::vector<Eigen::Matrix<int, -1, 1 >> &inv_perm = Model_args_as_cpp_struct.Model_args_vecs_of_col_vecs_int[1];

        //// Override lb_corr if corr_force_positive is TRUE:
        if (corr_force_positive == true) {
          for (int c = 0; c < n_class; ++c) {
            for (int i = 1; i < n_tests; ++i) {
              for (int j = 0; j < i; ++j) {
                lb_corr[c](i, j) = 0.0;
              }
            }
          }
        }

        //// ---- Other prev params:
        const int n_pops = Model_args_as_cpp_struct.Model_args_ints(6);  //// length = N ---- multi-pops
        const Eigen::Matrix<int, -1, 1> &pop_ind = Model_args_as_cpp_struct.Model_args_col_vecs_int[2];  // 0-indexed, length N //// ---- multi-pop

        //////////////
        const int n_corrs =  n_class * n_tests * (n_tests - 1) * 0.5;

        int n_covariates_total, n_covariates_max;
        int n_covariates_total_nd, n_covariates_total_d;
        int n_covariates_max_nd, n_covariates_max_d;

        if (n_class > 1) {

              n_covariates_total_nd = n_covariates_per_outcome_vec.row(0).sum();
              n_covariates_total_d = n_covariates_per_outcome_vec.row(1).sum();
              n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
              n_covariates_max_nd = n_covariates_per_outcome_vec.row(0).maxCoeff();
              n_covariates_max_d = n_covariates_per_outcome_vec.row(1).maxCoeff();
              n_covariates_max = std::max(n_covariates_max_nd, n_covariates_max_d);

        } else {

              n_covariates_total = n_covariates_per_outcome_vec.sum();
              n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();

        }

        const double sqrt_2_pi_recip = 1.0 / sqrt(2.0 * M_PI);
        const double sqrt_2_recip = 1.0 / stan::math::sqrt(2.0);
        const double minus_sqrt_2_recip = -sqrt_2_recip;
        const double a = 0.07056;
        const double b = 1.5976;
        const double a_times_3 = 3.0 * 0.07056;
        const double s = 1.0 / 1.702;
        const double Inf = std::numeric_limits<double>::infinity();

        //// ---- determine chunk size --------------------------------------------------
        const int desired_n_chunks = n_chunks;

        int vec_size;
        if (vect_type == "AVX512")      vec_size = 8;
        else if (vect_type == "AVX2")   vec_size = 4;
        else if (vect_type == "AVX")    vec_size = 2;
        else                            vec_size = 1;

        ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, desired_n_chunks);

        const int chunk_size_orig = chunk_size_info.chunk_size_orig;
        const int normal_chunk_size = chunk_size_info.normal_chunk_size;
        const int last_chunk_size = chunk_size_info.last_chunk_size;
        const int n_total_chunks = chunk_size_info.n_total_chunks;
        const int n_full_chunks = chunk_size_info.n_full_chunks;

        //////////////  ----------------------------------------------------------------------------------------------------------------------------------------------
        ////
        //// ---- Corrs (doubles):
        ////
        const Eigen::Matrix<double, -1, 1> Omega_raw_vec_double = theta_main_vec_ref.head(n_corrs);
        ////
        //// ---- Coeffs (doubles):
        ////
        std::vector<Eigen::Matrix<double, -1, -1>> beta_double_array = vec_of_mats(n_covariates_max, n_tests, n_class);
        {
          int i = n_corrs;
          for (int c = 0; c < n_class; ++c) {
            for (int t = 0; t < n_tests; ++t) {
              for (int k = 0; k < n_covariates_per_outcome_vec(c, t); ++k) {
                beta_double_array[c](k, t) = theta_main_vec_ref(i);
                i += 1;
              }
            }
          }
        }
        // //// ---- Identification constraint (label-switch guard): D+ must have the HIGHER intercept on
        // ////      the (first) binary test. Outside this half-space the posterior is -Inf. Implemented as
        // ////      a hard rejection rather than a reparameterisation, so the priors on the two intercepts
        // ////      are exactly as specified, restricted to beta_d > beta_nd (cf. Stan: target += -Inf).
        // ////      The gradient is irrelevant at a rejected point; NaN-free zeros are fine.
        // ////
        // if (n_class > 1) {
        //       const double beta_bin_nd = beta_double_array[0](0, 0);
        //       const double beta_bin_d  = beta_double_array[1](0, 0);
        //       if (!(beta_bin_d > beta_bin_nd)) {
        //             out_mat.setZero();
        //             out_mat(0) = -std::numeric_limits<double>::infinity();
        //             // stan::math::recover_memory_nested();
        //             return;
        //       }
        // }
        ////
        //// ---- Prev (doubles):
        ////
        Eigen::Matrix<double, -1, 1> u_prev_raw(n_pops);
        if (n_class > 1) {
          for (int g = 0; g < n_pops; ++g) {
            u_prev_raw(g) = theta_main_vec_ref(n_corrs + n_covariates_total + g);
          }
        }
        // double u_prev_diseased = 0.00001;
        // if (n_class > 1) {
        //   u_prev_diseased = theta_main_vec_ref(n_params_main - 1);
        // }
        ////
        //// ---- Omega:
        ////
        double prior_densities_L_Omega_double = 0.0;
        double log_det_J_L_Omega_double = 0.0;
        Eigen::Matrix<double, -1, 1> grad_Omega_raw_priors_and_log_det_J(n_corrs);

        int dim_choose_2 = n_tests * (n_tests - 1) * 0.5;
        std::vector<Eigen::Matrix<double, -1, -1>> deriv_L_wrt_unc_full = vec_of_mats_double(dim_choose_2 + n_tests, dim_choose_2, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_double = vec_of_mats_double(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_recip_double = vec_of_mats_double(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> Omega_double = vec_of_mats_double(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> Omega_unconstrained_double = fn_convert_std_vec_of_corrs_to_3d_array_double( Eigen_vec_to_std_vec(Omega_raw_vec_double),
                                                                                                                                n_tests,
                                                                                                                                n_class);
        // ============================================================================
        // Cholesky factorisation + numeric differentiation / AD for L_Omega
        // ============================================================================
        for (int c = 0; c < n_class; ++c) {   // Pinkney_cvine_bounds_opt_dbl   Pinkney_LDL_bounds_opt_dbl

              auto out = Pinkney_corr_master_dbl(  n_tests,
                                                   lb_corr[c],
                                                   ub_corr[c],
                                                   Omega_unconstrained_double[c],
                                                   known_values_indicator[c],
                                                   known_values[c]);

              L_Omega_double[c] = out.block(1, 0, n_tests, n_tests);  // use L_perm directly
              Omega_double[c] = L_Omega_double[c] * L_Omega_double[c].transpose();

              ////
              for (int i = 1; i < n_tests; ++i) {
                for (int j = 0; j < i; ++j) {  // Only lower triangular
                  if (std::abs(L_Omega_double[c](i, j)) > 1e-10) {  // Avoid near-zero
                    L_Omega_recip_double[c](i, j) = 1.0 / L_Omega_double[c](i, j);
                  } else {
                    L_Omega_recip_double[c](i, j) = 0.0;  // or handle differently
                  }
                }
              }

        }

        //// Compute L_Omega derivatives gradient using finite differences
        if (J_grad_option == "num_diff") {

              int global_idx = 0;  // Cumulative counter for all classes

              for (int c = 0; c < n_class; ++c) {

                      //// Helper lambda: unconstrained -> un-permuted L
                      auto get_L_orig = [&](const Eigen::Matrix<double, -1, -1> &Omega_unc) {

                            auto out = Pinkney_corr_master_dbl(   n_tests,
                                                                  lb_corr[c],
                                                                  ub_corr[c],
                                                                  Omega_unc,
                                                                  known_values_indicator[c],
                                                                  known_values[c]);
                            return out.block(1, 0, n_tests, n_tests).eval();

                      };

                      // // Base
                      // L_Omega_double[c] = get_L_orig( Omega_unconstrained_double[c]);

                      // Finite differences
                      int cnt_2 = 0;
                      for (int i = 1; i < n_tests; i++) {

                            for (int j = 0; j < i; j++) {

                                  double epsilon = std::max(1e-8, 1e-6 * std::abs( Omega_unconstrained_double[c](i, j)));
                                  if (std::abs( Omega_unconstrained_double[c](i, j)) > 2.0) epsilon = 1e-4;

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
                                  global_idx++;

                            }

                      }

              }

        } else if (J_grad_option == "analytical") {

              // for (int c = 0; c < n_class; ++c) {
              //   deriv_L_wrt_unc_full[c] = compute_L_Omega_derivatives_analytical( Omega_unconstrained_double[c],
              //                                                                     L_Omega_double[c],
              //                                                                     lb_corr[c],
              //                                                                     ub_corr[c],
              //                                                                     known_values_indicator[c],
              //                                                                     known_values[c],
              //                                                                     n_tests);
              // }

        }

        Eigen::Matrix<double, -1, 1> grad_prev_raw(n_pops);
        double prior_densities_prev_double = 0.0;
        // double log_det_J_prev_double = 0.0; //// ---- multi-pops
        // double grad_prev_raw_priors_and_log_det_J = 0.0; //// ---- multi-pops
        double log_det_J_prev_from_AD = 0.0;

        {    ///////////   -------------------  start of AD block  ------------------------------------------------------------------------------------------------------------------

          stan::math::start_nested();  ////////////////////////
          ////
          stan::math::var target_AD = 0.0;

          // L_Omega /  correlation params  ----------------------------------------------------------------------------------------------------------------------
          Eigen::Matrix<stan::math::var, -1, 1>  Omega_raw_vec_var =  stan::math::to_var(Omega_raw_vec_double) ;
          std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_unconstrained_var = fn_convert_std_vec_of_corrs_to_3d_array_var(Eigen_vec_to_std_vec_var(Omega_raw_vec_var),
                                                                                                                                       n_tests,
                                                                                                                                       n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  L_Omega_var = vec_of_mats_var(n_tests, n_tests, n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_var   = vec_of_mats_var(n_tests, n_tests, n_class);
          // Eigen::Matrix<stan::math::var, -1, 1> log_det_J_Omega(2);

          {

              stan::math::var log_det_J_L_Omega = 0.0;

              for (int c = 0; c < n_class; ++c) { // Pinkney_LDL_bounds_opt  Pinkney_cvine_bounds_opt

                    Eigen::Matrix<stan::math::var, -1, -1> Chol_Schur_outs = Pinkney_corr_master(  n_tests,
                                                                                                   lb_corr[c],
                                                                                                   ub_corr[c],
                                                                                                   Omega_unconstrained_var[c],
                                                                                                   known_values_indicator[c],
                                                                                                   known_values[c]);
                    L_Omega_var[c] = Chol_Schur_outs.block(1, 0, n_tests, n_tests);
                    target_AD         += Chol_Schur_outs(0, 0);
                    log_det_J_L_Omega += Chol_Schur_outs(0, 0);
                    Omega_var[c] = L_Omega_var[c] * L_Omega_var[c].transpose();

                    for (int i = 1; i < n_tests; ++i) {    ////   for (i in 2:n_tests) {
                      for (int j = 0; j < i; ++j) {   ////   for (j in 1:(i - 1)) {
                        if (known_values_indicator[c](i, j) == 1) {
                          stan::math::var known_val_prior_ij = stan::math::normal_lpdf( Omega_var[c](i, j), 0.0, 10.0 ); //// to ensure any corr's we aren't estimating dont cause divergences
                          target_AD += known_val_prior_ij;
                          prior_densities_L_Omega_double += known_val_prior_ij.val();   //// value now reaches lp (was grad-only)
                        }
                      }
                    }

              }

              // for (int i = 0; i < (Omega_raw_vec_var.size()); ++i) {
              //   target_AD += stan::math::normal_lpdf(Omega_raw_vec_var(i), 0.0, 10.0);
              // }

              log_det_J_L_Omega_double += log_det_J_L_Omega.val();

          }

          {

              stan::math::var prior_densities_L_Omega = 0.0;

              for (int c = 0; c < n_class; ++c) {

                    if ( (corr_prior_beta == false)   &&  (corr_prior_norm == false) ) {
                      prior_densities_L_Omega +=  stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c], lkj_cholesky_eta(c)); // for "Pinkney_LDL_bounds_opt"
                      // prior_densities_L_Omega += (lkj_cholesky_eta(c) - 1.0) * log_det_J_Omega(c);  // (0, 1) holds log_det_C // for "Pinkney_cvine_bounds_opt"
                    } else if ( (corr_prior_beta == true)   &&  (corr_prior_norm == false) ) {
                      for (int i = 1; i < n_tests; i++) {
                        for (int j = 0; j < i; j++) {
                          prior_densities_L_Omega +=  stan::math::beta_lpdf(  (Omega_var[c](i, j) + 1)/2, prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
                        }
                      }
                      //  Jacobian for  Omega -> L_Omega transformation for prior log-densities (since both LKJ and truncated normal prior densities are in terms of Omega, not L_Omega)
                      Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
                      for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i + 1) ) * stan::math::log(L_Omega_var[c](i, i));
                      prior_densities_L_Omega  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
                    } else if  ( (corr_prior_beta == false)   &&  (corr_prior_norm == true) ) {
                      for (int i = 1; i < n_tests; i++) {
                        for (int j = 0; j < i; j++) {
                          prior_densities_L_Omega +=  stan::math::normal_lpdf(  Omega_var[c](i, j), prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
                        }
                      }
                      Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
                      for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i + 1) ) * stan::math::log(L_Omega_var[c](i, i));
                      prior_densities_L_Omega  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
                    }

              }

              target_AD += prior_densities_L_Omega;
              prior_densities_L_Omega_double += prior_densities_L_Omega.val();

              ////////////////////////////////////////////////////////////
              target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
              grad_Omega_raw_priors_and_log_det_J = Omega_raw_vec_var.adj();    // differentiating WRT this
              // if (J_grad_option != "autodiff") {
              //   grad_Omega_raw_priors_and_log_det_J.array() += grad_log_det_J.array();
              // }
              out_mat.segment(1 + n_us, n_corrs) = grad_Omega_raw_priors_and_log_det_J ;   //// add grad constribution to output
              stan::math::set_zero_all_adjoints_nested();
              ////////////////////////////////////////////////////////////

          }

          /////////////  prev stuff  ---- vars
          {
              if (n_class > 1) {  //// if latent class

                  fn_MVP_prev_multi_pop_AD(  u_prev_raw,
                                             prev_prior_a,
                                             prev_prior_b,
                                             n_pops,
                                             prior_densities_prev_double,
                                             log_det_J_prev_from_AD,
                                             grad_prev_raw);

                  // Write to output (n_pops positions instead of 1)
                  int prev_start = 1 + n_us + n_corrs + n_covariates_total;
                  out_mat.segment(prev_start, n_pops) = grad_prev_raw;

              }
          }
          ////////////////////////////////////////////////////////////
          if (J_grad_option == "autodiff") {
            for (int c = 0; c < n_class; ++c) {
              int cnt_1 = 0;
              for (int k = 0; k < n_tests; k++) {
                for (int l = 0; l < k + 1; l++) {
                  (  L_Omega_var[c](k, l)).grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
                  int cnt_2 = 0;
                  for (int i = 1; i < n_tests; i++) {
                    for (int j = 0; j < i; j++) {
                      deriv_L_wrt_unc_full[c](cnt_1, cnt_2)  =   Omega_unconstrained_var[c](i, j).adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
                      cnt_2 += 1;
                    }
                  }
                  stan::math::set_zero_all_adjoints_nested();
                  cnt_1 += 1;
                }
              }
            }
          }

          ///////////////// get cholesky factor's (lower-triangular) of corr matrices
          // convert to 3d var array
          for (int c = 0; c < n_class; ++c) {
            for (int t2 = 0; t2 < n_tests; ++t2) { //// col-major storage
              for (int t1 = 0; t1 < n_tests; ++t1) {
                L_Omega_double[c](t1, t2) =   L_Omega_var[c](t1, t2).val()  ;
                L_Omega_recip_double[c](t1, t2) =   1.0 / L_Omega_double[c](t1, t2) ;
              }
            }
          }

          stan::math::recover_memory_nested();  //////////////////////////////////////////

        }    //////////////////////////  end of local AD block --------------------------------------------------------------------------------------------------------------------------

        /////////////  prev stuff (multi-pop)
        Eigen::Matrix<double, -1, -1> prev_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        Eigen::Matrix<double, -1, -1> log_prev_mat_small = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        Eigen::Matrix<double, -1, 1>  tanh_u_prev_vec(n_pops);
        Eigen::Matrix<double, -1, 1>  deriv_p_wrt_u_vec(n_pops);
        double log_det_J_prev_double_total = 0.0;
        ////
        if (n_class > 1) {

            for (int g = 0; g < n_pops; ++g) {
                tanh_u_prev_vec(g) = stan::math::tanh(u_prev_raw(g));
                double prev_g = 0.5 * (tanh_u_prev_vec(g) + 1.0);
                prev_mat(g, 1) = prev_g;
                prev_mat(g, 0) = 1.0 - prev_g;
                deriv_p_wrt_u_vec(g) = 0.5 * (1.0 - tanh_u_prev_vec(g) * tanh_u_prev_vec(g));
                log_det_J_prev_double_total +=  stan::math::log(deriv_p_wrt_u_vec(g));
            }
            ////
            log_prev_mat_small = stan::math::log(prev_mat);

        }

        ///////////////////////////////////////////////////////////////////////// prior densities
        double prior_densities = 0.0;

        if (exclude_priors == false) {

              double prior_densities_coeffs_double = 0.0;
              for (int c = 0; c < n_class; c++) {
                for (int t = 0; t < n_tests; t++) {
                  for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++) {
                    prior_densities_coeffs_double += stan::math::normal_lpdf(beta_double_array[c](k, t), prior_coeffs_mean[c](k, t), prior_coeffs_sd[c](k, t));
                  }
                }
              }

              prior_densities += prior_densities_coeffs_double;
              prior_densities += prior_densities_L_Omega_double;
              prior_densities += prior_densities_prev_double;

        }

        ////////  ------- likelihood function  ---------------------------------------------------------------------------------------------------------------------------------
        double log_prob_out = 0.0;

        // Jacobian adjustments (none needed for coeffs as unconstrained - so only for L_Omega -> Omega and u_prev -> prev, and the one for u's is computed in the likelihood)
        const double log_det_J_main = log_det_J_prev_double_total + log_det_J_L_Omega_double;

        //// Global (shared) gradient accumulators — will be written to directly (no thread-local needed)
        std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array = vec_of_mats<double>(n_covariates_max, n_tests, n_class);
        std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array = vec_of_mats<double>(n_tests, n_tests, n_class);
        Eigen::Matrix<double, -1, -1> prev_grad_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
        ////
        double log_jac_u = 0.0;
        ////
        // ============================================================================
        // SERIAL: CHUNK LOOP (full-sized chunks only)
        // ============================================================================
        // Process n_full_chunks serially. Last chunk (remainder) handled separately
        // after because it requires workspace resize + SIMD fallback.
        // ============================================================================

        {
          ////////////////////////////////////////////////
          // ---- Workspace ----
          LC_MVP_workspace_struct &ws = LC_MVP_ws_structs[0];
          ws.reset_sizes();
          ////////////////////////////////////////////////
          // ---- Workspace references (normal chunk size) ----
          ///////////////////////////////////////////////
          std::vector<Eigen::Matrix<double, -1, -1>> &Z_std_norm = ws.Z_std_norm;
          std::vector<Eigen::Matrix<double, -1, -1>> &Bound_Z = ws.Bound_Z;
          std::vector<Eigen::Matrix<double, -1, -1>> &Bound_U_Phi_Bound_Z = ws.Bound_U_Phi_Bound_Z;
          std::vector<Eigen::Matrix<double, -1, -1>> &prob = ws.prob;
          std::vector<Eigen::Matrix<double, -1, -1>> &Phi_Z = ws.Phi_Z;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &y1_log_prob = ws.y1_log_prob;
          Eigen::Matrix<double, -1, -1> &phi_Z_recip = ws.phi_Z_recip;
          Eigen::Matrix<double, -1, -1> &phi_Bound_Z = ws.phi_Bound_Z;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &u_grad_array_CM_chunk = ws.u_grad_array_CM_chunk;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &common_grad_term_1 = ws.common_grad_term_1;
          Eigen::Matrix<double, -1, -1> &y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = ws.y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip;
          Eigen::Matrix<double, -1, -1> &y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = ws.y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip;
          Eigen::Matrix<double, -1, -1> &prob_rowwise_prod_temp = ws.prob_rowwise_prod_temp;
          Eigen::Matrix<double, -1, -1> &prob_recip_rowwise_prod_temp = ws.prob_recip_rowwise_prod_temp;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, 1> &prod_container_or_inc_array = ws.prod_container_or_inc_array;
          Eigen::Matrix<double, -1, 1> &derivs_chain_container_vec = ws.derivs_chain_container_vec;
          Eigen::Matrix<double, -1, 1> &prob_rowwise_prod_temp_all = ws.prob_rowwise_prod_temp_all;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &grad_prob = ws.grad_prob;
          Eigen::Matrix<double, -1, -1> &z_grad_term = ws.z_grad_term;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &y_chunk = ws.y_chunk;
          Eigen::Matrix<double, -1, -1> &u_array = ws.u_array;
          Eigen::Matrix<double, -1, -1> &y_sign = ws.y_sign;
          Eigen::Matrix<double, -1, -1> &y_m_y_sign_x_u = ws.y_m_y_sign_x_u;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &u_grad_array_CM_chunk_block = ws.u_grad_array_CM_chunk_block;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, 1> &u_unc_vec_chunk = ws.u_unc_vec_chunk;
          Eigen::Matrix<double, -1, 1> &u_vec_chunk = ws.u_vec_chunk;
          Eigen::Matrix<double, -1, 1> &du_wrt_duu_chunk = ws.du_wrt_duu_chunk;
          Eigen::Matrix<double, -1, 1> &d_J_wrt_duu_chunk = ws.d_J_wrt_duu_chunk;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &lp_array = ws.lp_array;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, 1> &prob_n = ws.prob_n;
          Eigen::Matrix<double, -1, 1> &prob_n_recip = ws.prob_n_recip;
          Eigen::Matrix<double, -1, 1> &log_sum_result = ws.log_sum_result;
          Eigen::Matrix<double, -1, 1> &container_max_logs = ws.container_max_logs;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, 1> &rowwise_log_sum = ws.rowwise_log_sum;
          Eigen::Matrix<double, -1, 1> &rowwise_prod = ws.rowwise_prod;
          Eigen::Matrix<double, -1, 1> &rowwise_sum = ws.rowwise_sum;
          Eigen::Matrix<double, -1, 1> &log_lik_chunk = ws.log_lik_chunk;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &prob_recip = ws.prob_recip;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, 1> &log_prev_per_obs_given_c = ws.log_prev_per_obs_given_c;
          Eigen::Matrix<double, -1, 1> &prev_per_obs_given_c     = ws.prev_per_obs_given_c;
          ///////////////////////////////////////////////
          // ---- SIMD types (full SIMD for full chunks) ----
          std::string vt       = vect_type;
          std::string vt_exp   = vect_type_exp;
          std::string vt_log   = vect_type_log;
          std::string vt_lse   = vect_type_lse;
          std::string vt_tanh  = vect_type_tanh;
          std::string vt_Phi   = vect_type_Phi;
          std::string vt_log_Phi = vect_type_log_Phi;
          std::string vt_inv_Phi = vect_type_inv_Phi;
          std::string vt_inv_Phi_approx = vect_type_inv_Phi_approx_from_logit_prob;

          // ====================================================================
          // Serial for over full-sized chunks
          // ====================================================================
          for (int nc = 0; nc < n_full_chunks; nc++) {

            const int chunk_counter = nc;
            const int chunk_size = normal_chunk_size;  // All full chunks are same size

            u_grad_array_CM_chunk.setZero();

            y_chunk = y_ref.middleRows(chunk_size_orig * chunk_counter, chunk_size).cast<double>();

            // Nuisance parameter transformation
            u_unc_vec_chunk = theta_us_vec_ref.segment(chunk_size_orig * n_tests * chunk_counter, chunk_size * n_tests);
            fn_MVP_compute_nuisance(u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);
            log_jac_u += fn_MVP_compute_nuisance_log_jac_u(u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);

            u_array = u_vec_chunk.reshaped(chunk_size, n_tests);
            y_sign.array() = y_chunk.array() + (y_chunk.array() - 1.0);
            y_m_y_sign_x_u.array() = y_chunk.array() - (y_sign.array() * u_array.array());

            // ---- Class loop (identical to original) ----
            for (int c = 0; c < n_class; c++) {

                  ////-----------------------------------------------
                  if (n_class > 1) {

                    for (int n = 0; n < chunk_size; ++n) {
                      int n_global = chunk_size_orig * chunk_counter + n;
                      int g = pop_ind(n_global);                          // was missing `int`
                      log_prev_per_obs_given_c(n) = log_prev_mat_small(g, c);
                      prev_per_obs_given_c(n) = prev_mat(g, c);           // was `prev`, now `prev_mat`
                    }

                  }
                  ////-------------------------------------------

                  prod_container_or_inc_array.setZero();

                  for (int t = 0; t < n_tests; t++) {

                        ////-----------------------------------------------
                        if (n_covariates_max > 1) {
                              // for (int n = 0; n < chunk_size; n++) {
                              //     int n_global = chunk_size_orig * chunk_counter + n;
                              //     double Xbeta_n = ( X[c][t].row(n_global).head(n_covariates_per_outcome_vec(c, t)).cast<double>() * beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t)) ).eval()(0, 0);
                              //     Bound_Z[c](n, t) = L_Omega_recip_double[c](t, t) * (-1.0 * (Xbeta_n + prod_container_or_inc_array(n)));
                              // }
                              Eigen::Matrix<double, -1, 1> Xbeta_given_class_c_col_t = X[c][t].block(chunk_size_orig * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(c, t)).cast<double>()
                                                                                       * beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t));
                              Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * (-1.0 * (Xbeta_given_class_c_col_t.array() + prod_container_or_inc_array.array()));
                        } else {
                              Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * (-1.0 * (beta_double_array[c](0, t) + prod_container_or_inc_array.array()));
                        }
                        ////-----------------------------------------------
                        ////-----------------------------------------------
                        fn_MVP_compute_lp_GHK_cols(   t,
                                                      Bound_U_Phi_Bound_Z[c], // computing this
                                                      Phi_Z[c], // computing this
                                                      Z_std_norm[c], // computing this
                                                      prob[c], // computing this
                                                      y1_log_prob, // computing this
                                                      Bound_Z[c],
                                                      y_chunk,
                                                      u_array,
                                                      Model_args_as_cpp_struct);
                        // ////
                        // if (g_debug_cutpoint_grads == true) {
                        //     if (chunk_counter == 0 && c == 0) {
                        //       std::cout << "MD n=0 c=0 t=" << t
                        //                 << " Bound_Z=" << Bound_Z[c](0, t)
                        //                 << " Bound_U_Phi_BZ=" << Bound_U_Phi_Bound_Z[c](0, t)
                        //                 << " Phi_Z=" << Phi_Z[c](0, t)
                        //                 << " Z_std_norm=" << Z_std_norm[c](0, t)
                        //                 << " prob=" << prob[c](0, t)
                        //                 << " y1_log_prob=" << y1_log_prob(0, t)
                        //                 << std::endl;
                        //     }
                        // }
                        ////-----------------------------------------------
                        ////-----------------------------------------------
                        if (t < n_tests - 1) {
                          auto L_Omega_row = L_Omega_double[c].row(t + 1);
                          prod_container_or_inc_array = Z_std_norm[c].leftCols(t + 1) * L_Omega_row.head(t + 1).transpose();
                        }
                        ////-----------------------------------------------

                  } //// end of t loop

                  ////-----------------------------------------------
                  if (n_class > 1) { // if latent class
                        rowwise_sum = y1_log_prob.rowwise().sum();
                        rowwise_sum.array() += log_prev_per_obs_given_c.array();
                        lp_array.col(c) = rowwise_sum;
                  } else {
                        rowwise_sum = y1_log_prob.rowwise().sum();
                        lp_array.col(0) = rowwise_sum;
                  }
                  ////-----------------------------------------------

            }  // end of c loop

            // ---- Log-sum-exp and write log-lik to out_mat ----
            // NOTE: out_mat writes are to NON-OVERLAPPING segments per chunk — safe without synchronisation
            ////-----------------------------------------------
            if (n_class > 1) { // if latent class

                  log_sum_exp_general(lp_array,
                                      vt_exp,
                                      vt_log,
                                      log_sum_result,
                                      container_max_logs);
                  const int index_start = 1 + n_params + chunk_size_orig * chunk_counter;
                  out_mat.segment(index_start, chunk_size) = log_sum_result;

            } else {

                  out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size) = lp_array.col(0);

            }
            ////-----------------------------------------------

            ////-----------------------------------------------
            ////-----------------------------------------------
            log_lik_chunk = out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size);
            ////
            prob_n = fn_EIGEN_double(log_lik_chunk, "exp", vt_exp);
            // prob_n = log_lik_chunk;
            // apply_inplace<vec, Fn::exp>(prob_n);
            ////
            prob_n_recip = stan::math::inv(prob_n); // this CANNOT be a temporary otherwise it created a "dangling reference", since prob_n is a temporary!!
            ////-----------------------------------------------

            /////////////////  ------------------------- compute grad  ---------------------------------------------------------------------------------
            for (int c = 0; c < n_class; c++) {

                  //// ---- re-fill prev for this class (needed for fn_MVOP_grad_prep) ----
                  if (n_class > 1) {
                    for (int n = 0; n < chunk_size; ++n) {
                      int n_global = chunk_size_orig * chunk_counter + n;
                      int g = pop_ind(n_global);
                      prev_per_obs_given_c(n) = prev_mat(g, c);
                    }
                  }

                  ////-----------------------------------------------
                  prob_recip = stan::math::inv(prob[c]);
                  ////-----------------------------------------------

                  ////-----------------------------------------------
                  for (int t = 0; t < n_tests; t++) {

                        fn_MVP_compute_phi_Z_recip_cols(t,
                                                        phi_Z_recip, // computing this
                                                        Phi_Z[c], Z_std_norm[c], Model_args_as_cpp_struct);

                        fn_MVP_compute_phi_Bound_Z_cols(t,
                                                        phi_Bound_Z, // computing this
                                                        Bound_U_Phi_Bound_Z[c], Bound_Z[c], Model_args_as_cpp_struct);

                  }
                  ////-----------------------------------------------
                  if (grad_option != "none") {

                        fn_MVP_grad_prep( prob[c],
                                          y_sign,
                                          y_m_y_sign_x_u,
                                          L_Omega_recip_double[c],
                                          prev_per_obs_given_c, //// prev(0, c),
                                          prob_n_recip,
                                          phi_Z_recip,
                                          phi_Bound_Z,
                                          prob_recip,
                                          prob_rowwise_prod_temp, // computing this
                                          prob_recip_rowwise_prod_temp, // computing this
                                          prob_rowwise_prod_temp_all, // computing this
                                          common_grad_term_1, // computing this
                                          y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip, // computing this
                                          y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip, // computing this
                                          Model_args_as_cpp_struct);

                        // if (g_debug_cutpoint_grads == true) {
                        //     if (chunk_counter == 0 && c == 0) {
                        //       for (int n = 0; n < std::min(3, chunk_size); ++n) {
                        //         for (int t = 0; t < n_tests; ++t) {
                        //           std::cout << "MD c=0 n=" << n << " t=" << t
                        //                     << " phi_Z_recip=" << phi_Z_recip(n, t)
                        //                     << " phi_Bound_Z=" << phi_Bound_Z(n, t)
                        //                     << " y_sign=" << y_sign(n, t)
                        //                     << " y_m_ysign_xu=" << y_m_y_sign_x_u(n, t) // correct
                        //                     << " ys_phiBZ_Lrecip=" << y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip(n, t) // wrong!!!!!!!!!!
                        //                     << " ymysxu_phiZr_phiBZ_Lr=" << y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip(n, t)
                        //                     // << " dphi_over_L=" << y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip(n, t)
                        //                     // << " dZ_dmu_neg=" << y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip(n, t)
                        //                     << " cgt1=" << common_grad_term_1(n, t)
                        //                     << " prob=" << prob[c](n, t)
                        //                     << " prob_recip=" << prob_recip(n, t)
                        //                     << " prwpt=" << prob_rowwise_prod_temp(n, t)
                        //                     << std::endl;
                        //         }
                        //       }
                        //     }
                        // }

                  }
                  // ---- Nuisance gradients (write to non-overlapping out_mat segment) ----
                  if ((grad_option == "us_only") || (grad_option == "all")) {

                        u_grad_array_CM_chunk_block = u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests);

                        fn_MVP_compute_nuisance_grad_v2( u_grad_array_CM_chunk_block,
                                                         phi_Z_recip,
                                                         common_grad_term_1,
                                                         L_Omega_double[c],
                                                         prob[c],
                                                         prob_recip,
                                                         prob_rowwise_prod_temp,
                                                         y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                         y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                         z_grad_term,
                                                         grad_prob,
                                                         prod_container_or_inc_array,
                                                         derivs_chain_container_vec,
                                                         Model_args_as_cpp_struct);

                        u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests).array() += u_grad_array_CM_chunk_block.array();

                        const int start_index = 1 + (chunk_size_orig * n_tests * chunk_counter);
                        const int length = chunk_size * n_tests;

                        if (c == n_class - 1) {

                          //// Write nuisance grads to out_mat — NON-OVERLAPPING, no sync needed:

                          //// update output vector once all u_grad computations are done:
                          out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();

                          //// account for unconstrained -> constrained transformations and Jacobian adjustments
                          fn_MVP_nuisance_first_deriv(du_wrt_duu_chunk,
                                                      u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);

                          fn_MVP_nuisance_deriv_of_log_det_J(d_J_wrt_duu_chunk,
                                                             u_vec_chunk, u_unc_vec_chunk, du_wrt_duu_chunk, Model_args_as_cpp_struct);

                          out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
                          out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();

                        }

                  }
                  // ---- Coefficient gradients (accumulate directly to shared) ----
                  if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only")) {

                        // if (g_debug_cutpoint_grads == true) {
                        //     if (chunk_counter == 0 && c == 0) {
                        //       int t_last = n_tests - 1;
                        //       for (int n = 0; n < std::min(5, chunk_size); ++n) {
                        //         std::cout << "MD c=0 n=" << n << " t=" << t_last
                        //                   << " cgt1=" << common_grad_term_1(n, t_last)
                        //                   // << " dphi=" << dphi_over_L(n, t_last)
                        //                   << " prob=" << prob[c](n, t_last)
                        //                   << " prob_n_recip=" << prob_n_recip(n)
                        //                   << " prev=" << prev_per_obs_given_c(n)
                        //                   << std::endl;
                        //       }
                        //     }
                        // }

                        Eigen::Matrix<int, -1, 1> n_cov_vec_c = n_covariates_per_outcome_vec.row(c).transpose();

                        fn_MVP_compute_coefficients_grad_v3(  c,
                                                              beta_grad_array[c],   // <<< direct to shared accumulator
                                                              X[c],
                                                              n_cov_vec_c,
                                                              // chunk_counter,
                                                              chunk_size_orig * chunk_counter,
                                                              n_covariates_max,
                                                              common_grad_term_1,
                                                              L_Omega_double[c],
                                                              prob[c],
                                                              prob_recip,
                                                              prob_rowwise_prod_temp,
                                                              y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                              y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                              z_grad_term,
                                                              grad_prob,
                                                              prod_container_or_inc_array,
                                                              derivs_chain_container_vec,
                                                              true,
                                                              Model_args_as_cpp_struct);

                  }
                  // ---- L_Omega gradients (accumulate directly to shared) ----
                  if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only")) {

                        fn_MVP_compute_L_Omega_grad_v3(  U_Omega_grad_array[c],   // <<< direct to shared accumulator
                                                         common_grad_term_1,
                                                         L_Omega_double[c],
                                                         prob[c],
                                                         prob_recip,
                                                         Bound_Z[c],
                                                         Z_std_norm[c],
                                                         prob_rowwise_prod_temp,
                                                         y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                         y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                         z_grad_term,
                                                         grad_prob,
                                                         prod_container_or_inc_array,
                                                         derivs_chain_container_vec,
                                                         true,
                                                         Model_args_as_cpp_struct);

                  }
                  // ---- Prevalence gradient (accumulate directly to shared) ----
                  if (n_class > 1) {

                    if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only")) {

                          // rowwise_prod = prob[c].rowwise().prod();
                          // rowwise_prod.head(chunk_size).array() = prob_n_recip.head(chunk_size).array() * rowwise_prod.head(chunk_size).array();
                          // double prev_grad = rowwise_prod.head(chunk_size).sum();
                          // prev_grad_vec(c) += prev_grad;   // <<< direct to shared accumulator

                          fn_MVP_prev_multi_pop_accumulate_grad( prob[c],
                                                                 prob_n_recip,
                                                                 pop_ind,
                                                                 chunk_size_orig * chunk_counter,
                                                                 chunk_size,
                                                                 n_pops,
                                                                 c,
                                                                 prev_grad_mat,
                                                                 rowwise_prod);

                    }

                  }

            }  // end of c loop (gradient)

          }  // end of serial for over chunks

        }  // end of block 1 scope


        // ============================================================================
        // LAST CHUNK (remainder) — processed serially with fallback SIMD
        // ============================================================================
        if ((n_full_chunks < n_total_chunks) && (last_chunk_size > 0)) {

          // Make a mutable copy of the struct for scalar fallback
          Model_fn_args_struct Model_args_last_chunk = Model_args_as_cpp_struct;
          Model_args_last_chunk.Model_args_strings(0)  = "Stan";  // vect_type
          Model_args_last_chunk.Model_args_strings(3)  = "Stan";  // vect_type_exp
          Model_args_last_chunk.Model_args_strings(4)  = "Stan";  // vect_type_log
          Model_args_last_chunk.Model_args_strings(5)  = "Stan";  // vect_type_lse
          Model_args_last_chunk.Model_args_strings(6)  = "Stan";  // vect_type_tanh
          Model_args_last_chunk.Model_args_strings(7)  = "Stan";  // vect_type_Phi
          Model_args_last_chunk.Model_args_strings(8)  = "Stan";  // vect_type_log_Phi
          Model_args_last_chunk.Model_args_strings(9)  = "Stan";  // vect_type_inv_Phi
          Model_args_last_chunk.Model_args_strings(10) = "Stan";  // vect_type_inv_Phi_approx

          ///////////////////////////////////////////////
          const int nc = n_full_chunks;
          const int chunk_counter = nc;
          const int chunk_size = last_chunk_size;
          ///////////////////////////////////////////////
          // Use thread 0's workspace, resized for last chunk
          LC_MVP_workspace_struct &ws = LC_MVP_ws_structs[0];
          ws.reset_sizes();
          ///////////////////////////////////////////////
          // Override SIMD to Stan (scalar) for remainder
          std::string vt_last       = "Stan";
          std::string vt_exp_last   = "Stan";
          std::string vt_log_last   = "Stan";
          std::string vt_lse_last   = "Stan";
          std::string vt_tanh_last  = "Stan";
          std::string vt_Phi_last   = "Stan";
          std::string vt_log_Phi_last = "Stan";
          std::string vt_inv_Phi_last = "Stan";
          std::string vt_inv_Phi_approx_last = "Stan";
          ///////////////////////////////////////////////
          // Resize workspace for last chunk
          for (int c = 0; c < 2; c++) {
            ws.Z_std_norm[c].resize(last_chunk_size, n_tests);
            ws.Bound_Z[c].resize(last_chunk_size, n_tests);
            ws.Bound_U_Phi_Bound_Z[c].resize(last_chunk_size, n_tests);
            ws.prob[c].resize(last_chunk_size, n_tests);
            ws.Phi_Z[c].resize(last_chunk_size, n_tests);
          }
          ///////////////////////////////////////////////
          ws.y1_log_prob.resize(last_chunk_size, n_tests);
          ws.phi_Z_recip.resize(last_chunk_size, n_tests);
          ws.phi_Bound_Z.resize(last_chunk_size, n_tests);
          ws.u_grad_array_CM_chunk.resize(last_chunk_size, n_tests);
          ws.common_grad_term_1.resize(last_chunk_size, n_tests);
          ws.y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
          ws.y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
          ws.prob_rowwise_prod_temp.resize(last_chunk_size, n_tests);
          ws.prob_recip_rowwise_prod_temp.resize(last_chunk_size, n_tests);
          ws.prod_container_or_inc_array.resize(last_chunk_size);
          ws.derivs_chain_container_vec.resize(last_chunk_size);
          ws.prob_rowwise_prod_temp_all.resize(last_chunk_size);
          ws.grad_prob.resize(last_chunk_size, n_tests);
          ws.z_grad_term.resize(last_chunk_size, n_tests);
          ws.y_chunk.resize(last_chunk_size, n_tests);
          ws.u_array.resize(last_chunk_size, n_tests);
          ws.y_sign.resize(last_chunk_size, n_tests);
          ws.y_m_y_sign_x_u.resize(last_chunk_size, n_tests);
          ws.u_grad_array_CM_chunk_block.resize(last_chunk_size, n_tests);
          ws.u_unc_vec_chunk.resize(last_chunk_size * n_tests);
          ws.u_vec_chunk.resize(last_chunk_size * n_tests);
          ws.du_wrt_duu_chunk.resize(last_chunk_size * n_tests);
          ws.d_J_wrt_duu_chunk.resize(last_chunk_size * n_tests);
          ///////////////////////////////////////////////
          ws.lp_array.resize(last_chunk_size, n_class);
          ///////////////////////////////////////////////
          ws.prob_n.resize(last_chunk_size);
          ws.prob_n_recip.resize(last_chunk_size);
          ws.log_sum_result.resize(last_chunk_size);
          ws.container_max_logs.resize(last_chunk_size);
          ws.rowwise_log_sum.resize(last_chunk_size);
          ws.rowwise_prod.resize(last_chunk_size);
          ws.rowwise_sum.resize(last_chunk_size);
          ws.log_lik_chunk.resize(last_chunk_size);
          ws.prob_recip.resize(last_chunk_size, n_tests);
          ///////////////////////////////////////////////
          ws.log_prev_per_obs_given_c.resize(last_chunk_size);
          ws.prev_per_obs_given_c.resize(last_chunk_size);
          ///////////////////////////////////////////////
          ////
          //// Re-bind workspace refs for last chunk
          ////
          ///////////////////////////////////////////////
          std::vector<Eigen::Matrix<double, -1, -1>> &Z_std_norm = ws.Z_std_norm;
          std::vector<Eigen::Matrix<double, -1, -1>> &Bound_Z = ws.Bound_Z;
          std::vector<Eigen::Matrix<double, -1, -1>> &Bound_U_Phi_Bound_Z = ws.Bound_U_Phi_Bound_Z;
          std::vector<Eigen::Matrix<double, -1, -1>> &prob = ws.prob;
          std::vector<Eigen::Matrix<double, -1, -1>> &Phi_Z = ws.Phi_Z;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &y1_log_prob = ws.y1_log_prob;
          Eigen::Matrix<double, -1, -1> &phi_Z_recip = ws.phi_Z_recip;
          Eigen::Matrix<double, -1, -1> &phi_Bound_Z = ws.phi_Bound_Z;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &u_grad_array_CM_chunk = ws.u_grad_array_CM_chunk;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &common_grad_term_1 = ws.common_grad_term_1;
          Eigen::Matrix<double, -1, -1> &y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = ws.y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip;
          Eigen::Matrix<double, -1, -1> &y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = ws.y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip;
          Eigen::Matrix<double, -1, -1> &prob_rowwise_prod_temp = ws.prob_rowwise_prod_temp;
          Eigen::Matrix<double, -1, -1> &prob_recip_rowwise_prod_temp = ws.prob_recip_rowwise_prod_temp;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, 1> &prod_container_or_inc_array = ws.prod_container_or_inc_array;
          Eigen::Matrix<double, -1, 1> &derivs_chain_container_vec = ws.derivs_chain_container_vec;
          Eigen::Matrix<double, -1, 1> &prob_rowwise_prod_temp_all = ws.prob_rowwise_prod_temp_all;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &grad_prob = ws.grad_prob;
          Eigen::Matrix<double, -1, -1> &z_grad_term = ws.z_grad_term;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &y_chunk = ws.y_chunk;
          Eigen::Matrix<double, -1, -1> &u_array = ws.u_array;
          Eigen::Matrix<double, -1, -1> &y_sign = ws.y_sign;
          Eigen::Matrix<double, -1, -1> &y_m_y_sign_x_u = ws.y_m_y_sign_x_u;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &u_grad_array_CM_chunk_block = ws.u_grad_array_CM_chunk_block;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, 1> &u_unc_vec_chunk = ws.u_unc_vec_chunk;
          Eigen::Matrix<double, -1, 1> &u_vec_chunk = ws.u_vec_chunk;
          Eigen::Matrix<double, -1, 1> &du_wrt_duu_chunk = ws.du_wrt_duu_chunk;
          Eigen::Matrix<double, -1, 1> &d_J_wrt_duu_chunk = ws.d_J_wrt_duu_chunk;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &lp_array = ws.lp_array;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, 1> &prob_n = ws.prob_n;
          Eigen::Matrix<double, -1, 1> &prob_n_recip = ws.prob_n_recip;
          Eigen::Matrix<double, -1, 1> &log_sum_result = ws.log_sum_result;
          Eigen::Matrix<double, -1, 1> &container_max_logs = ws.container_max_logs;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, 1> &rowwise_log_sum = ws.rowwise_log_sum;
          Eigen::Matrix<double, -1, 1> &rowwise_prod = ws.rowwise_prod;
          Eigen::Matrix<double, -1, 1> &rowwise_sum = ws.rowwise_sum;
          Eigen::Matrix<double, -1, 1> &log_lik_chunk = ws.log_lik_chunk;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, -1> &prob_recip = ws.prob_recip;
          ///////////////////////////////////////////////
          Eigen::Matrix<double, -1, 1> &log_prev_per_obs_given_c = ws.log_prev_per_obs_given_c;
          Eigen::Matrix<double, -1, 1> &prev_per_obs_given_c = ws.prev_per_obs_given_c;
          ///////////////////////////////////////////////
          std::string vt       = vect_type;
          std::string vt_exp   = vect_type_exp;
          std::string vt_log   = vect_type_log;
          std::string vt_lse   = vect_type_lse;
          std::string vt_tanh  = vect_type_tanh;
          std::string vt_Phi   = vect_type_Phi;
          std::string vt_log_Phi = vect_type_log_Phi;
          std::string vt_inv_Phi = vect_type_inv_Phi;
          std::string vt_inv_Phi_approx = vect_type_inv_Phi_approx_from_logit_prob;
          ///////////////////////////////////////////////
          ////-----------------------------------------------
          u_grad_array_CM_chunk.setZero();

          y_chunk = y_ref.middleRows(chunk_size_orig * chunk_counter, chunk_size).cast<double>();

          u_unc_vec_chunk = theta_us_vec_ref.segment(chunk_size_orig * n_tests * chunk_counter, chunk_size * n_tests);
          fn_MVP_compute_nuisance(u_vec_chunk, u_unc_vec_chunk, Model_args_last_chunk);
          log_jac_u += fn_MVP_compute_nuisance_log_jac_u(u_vec_chunk, u_unc_vec_chunk, Model_args_last_chunk);

          u_array = u_vec_chunk.reshaped(chunk_size, n_tests);
          y_sign.array() = y_chunk.array() + (y_chunk.array() - 1.0);
          y_m_y_sign_x_u.array() = y_chunk.array() - (y_sign.array() * u_array.array());
          ////-----------------------------------------------

          ////-----------------------------------------------
          for (int c = 0; c < n_class; c++) {

            ////----------------------------------------------
            if (n_class > 1) {

                for (int n = 0; n < chunk_size; ++n) {
                  int n_global = chunk_size_orig * chunk_counter + n;
                  int g = pop_ind(n_global);                          // was missing `int`
                  log_prev_per_obs_given_c(n) = log_prev_mat_small(g, c);
                  prev_per_obs_given_c(n) = prev_mat(g, c);           // was `prev`, now `prev_mat`
                }

            }
            ////---------------------------------------------

            prod_container_or_inc_array.setZero();

            ////-----------------------------------------------
            for (int t = 0; t < n_tests; t++) {

                if (n_covariates_max > 1) {

                      // for (int n = 0; n < chunk_size; n++) {
                      //   int n_global = chunk_size_orig * chunk_counter + n;
                      //   double Xbeta_n = ( X[c][t].row(n_global).head(n_covariates_per_outcome_vec(c, t)).cast<double>() * beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t)) ).eval()(0, 0);
                      //   Bound_Z[c](n, t) = L_Omega_recip_double[c](t, t) * (-1.0 * (Xbeta_n + prod_container_or_inc_array(n)));
                      // }
                      Eigen::Matrix<double, -1, 1> Xbeta_given_class_c_col_t = X[c][t].block(chunk_size_orig * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(c, t)).cast<double>()
                                                                               * beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t));
                      Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * (-1.0 * (Xbeta_given_class_c_col_t.array() + prod_container_or_inc_array.array()));

                } else {

                      Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * (-1.0 * (beta_double_array[c](0, t) + prod_container_or_inc_array.array()));

                }

                ////-----------------------------------------------
                fn_MVP_compute_lp_GHK_cols( t,
                                            Bound_U_Phi_Bound_Z[c],
                                            Phi_Z[c],
                                            Z_std_norm[c],
                                            prob[c],
                                            y1_log_prob,
                                            Bound_Z[c],
                                            y_chunk,
                                            u_array,
                                            Model_args_last_chunk);
                ////-----------------------------------------------

                ////-----------------------------------------------
                if (t < n_tests - 1) {
                      auto L_Omega_row = L_Omega_double[c].row(t + 1);
                      prod_container_or_inc_array = Z_std_norm[c].leftCols(t + 1) * L_Omega_row.head(t + 1).transpose();
                }
                ////-----------------------------------------------

            }

            ////-----------------------------------------------
            if (n_class > 1) {
                  rowwise_sum = y1_log_prob.rowwise().sum();
                  rowwise_sum.array() += log_prev_per_obs_given_c.array();
                  lp_array.col(c) = rowwise_sum;
            } else {
                  rowwise_sum = y1_log_prob.rowwise().sum();
                  lp_array.col(0) = rowwise_sum;
            }
            ////-----------------------------------------------

          } //// end of c loop

          ////-----------------------------------------------
          if (n_class > 1) {

                log_sum_exp_general(lp_array,
                                    vt_exp_last,
                                    vt_log_last,
                                    log_sum_result,
                                    container_max_logs);
                const int index_start = 1 + n_params + chunk_size_orig * chunk_counter;
                out_mat.segment(index_start, chunk_size) = log_sum_result;

          } else {

                out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size) = lp_array.col(0);

          }
          ////-----------------------------------------------

          ////-----------------------------------------------
          log_lik_chunk = out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size);
          prob_n = fn_EIGEN_double(log_lik_chunk, "exp", vt_exp_last);
          prob_n_recip = stan::math::inv(prob_n);
          ////-----------------------------------------------

          /////////////////  ------------------------- compute grad  ---------------------------------------------------------------------------------
          for (int c = 0; c < n_class; c++) {

            //// ---- re-fill prev for this class (needed for fn_MVOP_grad_prep) ----
            if (n_class > 1) {
              for (int n = 0; n < chunk_size; ++n) {
                int n_global = chunk_size_orig * chunk_counter + n;
                int g = pop_ind(n_global);
                prev_per_obs_given_c(n) = prev_mat(g, c);
              }
            }

            ////-----------------------------------------------
            prob_recip = stan::math::inv(prob[c]);
            ////-----------------------------------------------

            ////-----------------------------------------------
            //// compute/update important log-lik quantities for GHK-MVP
            for (int t = 0; t < n_tests; t++) {

                  fn_MVP_compute_phi_Z_recip_cols(t,
                                                  phi_Z_recip,
                                                  Phi_Z[c], Z_std_norm[c], Model_args_last_chunk);

                  fn_MVP_compute_phi_Bound_Z_cols(t,
                                                  phi_Bound_Z,
                                                  Bound_U_Phi_Bound_Z[c], Bound_Z[c], Model_args_last_chunk);

            }
            ////-----------------------------------------------

            ////-----------------------------------------------
            if (grad_option != "none") {

                  fn_MVP_grad_prep( prob[c],
                                    y_sign,
                                    y_m_y_sign_x_u,
                                    L_Omega_recip_double[c],
                                    prev_per_obs_given_c, //// prev(0, c),
                                    prob_n_recip,
                                    phi_Z_recip,
                                    phi_Bound_Z,
                                    prob_recip,
                                    prob_rowwise_prod_temp,
                                    prob_recip_rowwise_prod_temp,
                                    prob_rowwise_prod_temp_all,
                                    common_grad_term_1,
                                    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                    y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                    Model_args_last_chunk);
            }
            ////////////////////////////////////////////////////////////////////////////////////  Grad of nuisance parameters / u's (manual)
            if ((grad_option == "us_only") || (grad_option == "all")) {

                  u_grad_array_CM_chunk_block = u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests);

                  fn_MVP_compute_nuisance_grad_v2( u_grad_array_CM_chunk_block,
                                                   phi_Z_recip,
                                                   common_grad_term_1,
                                                   L_Omega_double[c],
                                                   prob[c],
                                                   prob_recip,
                                                   prob_rowwise_prod_temp,
                                                   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                   y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                   z_grad_term, grad_prob,
                                                   prod_container_or_inc_array,
                                                   derivs_chain_container_vec,
                                                   Model_args_last_chunk);

                  u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests).array() += u_grad_array_CM_chunk_block.array();

                  const int start_index = 1 + (chunk_size_orig * n_tests * chunk_counter);
                  const int length = chunk_size * n_tests;

                  if (c == n_class - 1) {

                        out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();

                        fn_MVP_nuisance_first_deriv(du_wrt_duu_chunk,
                                                    u_vec_chunk, u_unc_vec_chunk, Model_args_last_chunk);

                        fn_MVP_nuisance_deriv_of_log_det_J(d_J_wrt_duu_chunk,
                                                           u_vec_chunk, u_unc_vec_chunk, du_wrt_duu_chunk, Model_args_last_chunk);

                        out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
                        out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();

                  }

            }
            /////////////////////////////////////////////////////////////////////////// Grad of intercepts / coefficients (beta's)
            if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only")) {

                  Eigen::Matrix<int, -1, 1> n_cov_vec_c = n_covariates_per_outcome_vec.row(c).transpose();

                  fn_MVP_compute_coefficients_grad_v3(  c,
                                                        beta_grad_array[c],  // direct to shared (serial)
                                                        X[c],
                                                        n_cov_vec_c,
                                                        // chunk_counter,
                                                        chunk_size_orig * chunk_counter,
                                                        n_covariates_max,
                                                        common_grad_term_1,
                                                        L_Omega_double[c],
                                                        prob[c],
                                                        prob_recip,
                                                        prob_rowwise_prod_temp,
                                                        y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                        y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                        z_grad_term,
                                                        grad_prob,
                                                        prod_container_or_inc_array,
                                                        derivs_chain_container_vec,
                                                        true,
                                                        Model_args_last_chunk);

            }
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of L_Omega ('s)
            if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only")) {

                  fn_MVP_compute_L_Omega_grad_v3( U_Omega_grad_array[c],  // direct to shared (serial)
                                                  common_grad_term_1,
                                                  L_Omega_double[c],
                                                  prob[c],
                                                  prob_recip,
                                                  Bound_Z[c],
                                                  Z_std_norm[c],
                                                  prob_rowwise_prod_temp,
                                                  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                  y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                  z_grad_term,
                                                  grad_prob,
                                                  prod_container_or_inc_array,
                                                  derivs_chain_container_vec,
                                                  true,
                                                  Model_args_last_chunk);

            }
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of cutpoints
            if (n_class > 1) {

                if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only")) {

                      // rowwise_prod = prob[c].rowwise().prod();
                      // rowwise_prod.head(chunk_size).array() = prob_n_recip.head(chunk_size).array() * rowwise_prod.head(chunk_size).array();
                      // prev_grad_vec(c) += rowwise_prod.head(chunk_size).sum();

                      fn_MVP_prev_multi_pop_accumulate_grad( prob[c],
                                                             prob_n_recip,
                                                             pop_ind,
                                                             chunk_size_orig * chunk_counter,
                                                             chunk_size,
                                                             n_pops,
                                                             c,
                                                             prev_grad_mat,
                                                             rowwise_prod);

                }

            }

          }

        }  // end of last chunk loop

        // ============================================================================
        // Post-loop: prevalence gradient, log_prob, output assembly
        // ============================================================================
        // Eigen::Matrix<double, -1, 1> prev_unconstrained_grad_vec = Eigen::Matrix<double, -1, 1>::Zero(n_class);
        Eigen::Matrix<double, -1, 1> prev_unconstrained_grad_vec_out = Eigen::Matrix<double, -1, 1>::Zero(n_pops);

        ////////////////////////  --------------------------------------------------------------------------
        // if (n_class > 1) {
        //   for (int c = 0; c < n_class; c++) {
        //     prev_unconstrained_grad_vec(c) = prev_grad_vec(c) * deriv_p_wrt_pu_double;
        //   }
        //   prev_unconstrained_grad_vec(0) = prev_unconstrained_grad_vec(1) - prev_unconstrained_grad_vec(0) - 2 * tanh_u_prev[1];
        //   prev_unconstrained_grad_vec_out(0) = prev_unconstrained_grad_vec(0);
        // }
        if (n_class > 1) {
          for (int g = 0; g < n_pops; ++g) {
            double lik_grad = (prev_grad_mat(g, 1) - prev_grad_mat(g, 0)) * deriv_p_wrt_u_vec(g);
            double jac_grad = -2.0 * tanh_u_prev_vec(g);
            prev_unconstrained_grad_vec_out(g) = lik_grad + jac_grad;
          }
        }

        ////////////////////////  --------------------------------------------------------------------------
        log_prob_out += out_mat.tail(N).sum(); // log_prob_out += log_lik.sum();
        log_prob_out += log_jac_u;
        if (exclude_priors == false) log_prob_out += prior_densities;
        log_prob_out += log_det_J_main;

        // Pack beta gradients into vector
        Eigen::Matrix<double, -1, 1> beta_grad_vec = Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);
        {
          int i = 0;
          for (int c = 0; c < n_class; c++) {
            for (int t = 0; t < n_tests; t++) {
              for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++) {
                beta_grad_vec(i) = beta_grad_array[c](k, t);
                i += 1;
              }
            }
          }
        }

        //// Pack L_Omega gradients and chain-rule through deriv_L_wrt_unc_full
        Eigen::Matrix<double, -1, 1> L_Omega_grad_vec(n_corrs + (n_class * n_tests));
        Eigen::Matrix<double, -1, 1> U_Omega_grad_vec(n_corrs);
        {
          int i = 0;
          for (int c = 0; c < n_class; c++) {
            for (int t1 = 0; t1 < n_tests; t1++) {
              for (int t2 = 0; t2 < t1 + 1; t2++) {
                L_Omega_grad_vec(i) = U_Omega_grad_array[c](t1, t2);
                i += 1;
              }
            }
          }
        }

        Eigen::Matrix<double, -1, 1> grad_wrt_L_Omega_nd(dim_choose_2 + n_tests);
        Eigen::Matrix<double, -1, 1> grad_wrt_L_Omega_d(dim_choose_2 + n_tests);

        if (n_class > 1) {
              grad_wrt_L_Omega_nd = L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
              grad_wrt_L_Omega_d = L_Omega_grad_vec.segment(dim_choose_2 + n_tests, dim_choose_2 + n_tests);
              U_Omega_grad_vec.head(dim_choose_2) = (grad_wrt_L_Omega_nd.transpose() * deriv_L_wrt_unc_full[0]).transpose();
              U_Omega_grad_vec.segment(dim_choose_2, dim_choose_2) = (grad_wrt_L_Omega_d.transpose() * deriv_L_wrt_unc_full[1]).transpose();
        } else {
              grad_wrt_L_Omega_nd = L_Omega_grad_vec.head(dim_choose_2 + n_tests);
              U_Omega_grad_vec.head(dim_choose_2) = (grad_wrt_L_Omega_nd.transpose() * deriv_L_wrt_unc_full[0]).transpose();
        }

        //// ---- Final output assembly ----
        out_mat(0) = log_prob_out;
        out_mat.segment(1 + n_us, n_corrs) += U_Omega_grad_vec;
        out_mat.segment(1 + n_us + n_corrs, n_covariates_total) += beta_grad_vec;
        ////
        if (n_class > 1) {
          const int prev_start_final = 1 + n_us + n_corrs + n_covariates_total;
          out_mat.segment(prev_start_final, n_pops) += prev_unconstrained_grad_vec_out;
        }

        //// ---- Prior gradient contribution to coefficients ----
        {
          int i = n_us + n_corrs + 1;
          for (int c = 0; c < n_class; c++) {
            for (int t = 0; t < n_tests; t++) {
              for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++) {
                if (exclude_priors == false) {
                  out_mat(i) += -((beta_double_array[c](k, t) - prior_coeffs_mean[c](k, t)) / prior_coeffs_sd[c](k, t)) * (1.0 / prior_coeffs_sd[c](k, t));
                  i += 1;
                }
              }
            }
          }
        }

}








 

//// MVP_lp_grad_MD_AD_fns.hpp  — MIGRATED (complete replacement for the serial NoLog driver)
////
//// Requires: fn_dispatch_templated.hpp, MVP_helpers_migrated.hpp, and fn_MVP_compute_lp_GHK_cols<vec>
//// (the version you already have in MVP_manual_grad_calc_fns.hpp).
////
//// What changed vs your version:
////   - one chunk loop over n_total_chunks; the remainder is just the last iteration with a smaller
////     chunk_size and a workspace resize. The 500-line duplicated "LAST CHUNK" block is gone.
////   - no Model_args_last_chunk, no "Stan" overrides, no vt_* strings: every element-wise call is
////     apply_*<vec, Fn::...>, and the masked tail handles any chunk length.
////   - workspace references are bound ONCE before the loop. resize() on a member does not
////     invalidate a reference to that member, so no re-binding for the last chunk.
////   - fn_MVP_compute_nuisance_log_jac_u_T<vec> needs a scratch vector of length chunk_size*n_tests:
////     d_J_wrt_duu_chunk is used (it is recomputed later in the same iteration, so this is safe).


//// -------------------------------------------------------------------------------------
//// Resize every per-chunk workspace member to `rows`. Used only for the remainder chunk.
//// -------------------------------------------------------------------------------------
inline void resize_MVP_workspace( LC_MVP_workspace_struct &ws,
                                  const int rows,
                                  const int n_tests,
                                  const int n_class) {

        for (int c = 0; c < 2; c++) {
          ws.Z_std_norm[c].resize(rows, n_tests);
          ws.Bound_Z[c].resize(rows, n_tests);
          ws.Bound_U_Phi_Bound_Z[c].resize(rows, n_tests);
          ws.prob[c].resize(rows, n_tests);
          ws.Phi_Z[c].resize(rows, n_tests);
        }
        ws.y1_log_prob.resize(rows, n_tests);
        ws.phi_Z_recip.resize(rows, n_tests);
        ws.phi_Bound_Z.resize(rows, n_tests);
        ws.u_grad_array_CM_chunk.resize(rows, n_tests);
        ws.common_grad_term_1.resize(rows, n_tests);
        ws.y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(rows, n_tests);
        ws.y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(rows, n_tests);
        ws.prob_rowwise_prod_temp.resize(rows, n_tests);
        ws.prob_recip_rowwise_prod_temp.resize(rows, n_tests);
        ws.prod_container_or_inc_array.resize(rows);
        ws.derivs_chain_container_vec.resize(rows);
        ws.prob_rowwise_prod_temp_all.resize(rows);
        ws.grad_prob.resize(rows, n_tests);
        ws.z_grad_term.resize(rows, n_tests);
        ws.y_chunk.resize(rows, n_tests);
        ws.u_array.resize(rows, n_tests);
        ws.y_sign.resize(rows, n_tests);
        ws.y_m_y_sign_x_u.resize(rows, n_tests);
        ws.u_grad_array_CM_chunk_block.resize(rows, n_tests);
        ws.u_unc_vec_chunk.resize(rows * n_tests);
        ws.u_vec_chunk.resize(rows * n_tests);
        ws.du_wrt_duu_chunk.resize(rows * n_tests);
        ws.d_J_wrt_duu_chunk.resize(rows * n_tests);
        ws.lp_array.resize(rows, n_class);
        ws.prob_n.resize(rows);
        ws.prob_n_recip.resize(rows);
        ws.log_sum_result.resize(rows);
        ws.container_max_logs.resize(rows);
        ws.rowwise_log_sum.resize(rows);
        ws.rowwise_prod.resize(rows);
        ws.rowwise_sum.resize(rows);
        ws.log_lik_chunk.resize(rows);
        ws.prob_recip.resize(rows, n_tests);
        ws.log_prev_per_obs_given_c.resize(rows);
        ws.prev_per_obs_given_c.resize(rows);

}




//////////////////////////////////////////////////------------------------------------------------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl_T(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                      const std::string &grad_option,
                                                                                      const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                      std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs
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
          stan::math::start_nested();
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

          stan::math::recover_memory_nested();
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
        // CHUNK LOOP — full chunks AND the remainder in one loop.
        // ============================================================================
        {
          LC_MVP_workspace_struct &ws = LC_MVP_ws_structs[0];
          ws.reset_sizes();

          //// ---- Workspace references, bound ONCE. Resizing a member does not invalidate these. ----
          std::vector<Eigen::Matrix<double, -1, -1>> &Z_std_norm          = ws.Z_std_norm;
          std::vector<Eigen::Matrix<double, -1, -1>> &Bound_Z             = ws.Bound_Z;
          std::vector<Eigen::Matrix<double, -1, -1>> &Bound_U_Phi_Bound_Z = ws.Bound_U_Phi_Bound_Z;
          std::vector<Eigen::Matrix<double, -1, -1>> &prob                = ws.prob;
          std::vector<Eigen::Matrix<double, -1, -1>> &Phi_Z               = ws.Phi_Z;
          Eigen::Matrix<double, -1, -1> &y1_log_prob = ws.y1_log_prob;
          Eigen::Matrix<double, -1, -1> &phi_Z_recip = ws.phi_Z_recip;
          Eigen::Matrix<double, -1, -1> &phi_Bound_Z = ws.phi_Bound_Z;
          Eigen::Matrix<double, -1, -1> &u_grad_array_CM_chunk = ws.u_grad_array_CM_chunk;
          Eigen::Matrix<double, -1, -1> &common_grad_term_1 = ws.common_grad_term_1;
          Eigen::Matrix<double, -1, -1> &y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = ws.y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip;
          Eigen::Matrix<double, -1, -1> &y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = ws.y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip;
          Eigen::Matrix<double, -1, -1> &prob_rowwise_prod_temp       = ws.prob_rowwise_prod_temp;
          Eigen::Matrix<double, -1, -1> &prob_recip_rowwise_prod_temp = ws.prob_recip_rowwise_prod_temp;
          Eigen::Matrix<double, -1, 1>  &prod_container_or_inc_array  = ws.prod_container_or_inc_array;
          Eigen::Matrix<double, -1, 1>  &derivs_chain_container_vec   = ws.derivs_chain_container_vec;
          Eigen::Matrix<double, -1, 1>  &prob_rowwise_prod_temp_all   = ws.prob_rowwise_prod_temp_all;
          Eigen::Matrix<double, -1, -1> &grad_prob   = ws.grad_prob;
          Eigen::Matrix<double, -1, -1> &z_grad_term = ws.z_grad_term;
          Eigen::Matrix<double, -1, -1> &y_chunk        = ws.y_chunk;
          Eigen::Matrix<double, -1, -1> &u_array        = ws.u_array;
          Eigen::Matrix<double, -1, -1> &y_sign         = ws.y_sign;
          Eigen::Matrix<double, -1, -1> &y_m_y_sign_x_u = ws.y_m_y_sign_x_u;
          Eigen::Matrix<double, -1, -1> &u_grad_array_CM_chunk_block = ws.u_grad_array_CM_chunk_block;
          Eigen::Matrix<double, -1, 1>  &u_unc_vec_chunk  = ws.u_unc_vec_chunk;
          Eigen::Matrix<double, -1, 1>  &u_vec_chunk      = ws.u_vec_chunk;
          Eigen::Matrix<double, -1, 1>  &du_wrt_duu_chunk = ws.du_wrt_duu_chunk;
          Eigen::Matrix<double, -1, 1>  &d_J_wrt_duu_chunk = ws.d_J_wrt_duu_chunk;
          Eigen::Matrix<double, -1, -1> &lp_array = ws.lp_array;
          Eigen::Matrix<double, -1, 1>  &prob_n             = ws.prob_n;
          Eigen::Matrix<double, -1, 1>  &prob_n_recip       = ws.prob_n_recip;
          Eigen::Matrix<double, -1, 1>  &log_sum_result     = ws.log_sum_result;
          Eigen::Matrix<double, -1, 1>  &container_max_logs = ws.container_max_logs;
          Eigen::Matrix<double, -1, 1>  &rowwise_prod  = ws.rowwise_prod;
          Eigen::Matrix<double, -1, 1>  &rowwise_sum   = ws.rowwise_sum;
          Eigen::Matrix<double, -1, 1>  &log_lik_chunk = ws.log_lik_chunk;
          Eigen::Matrix<double, -1, -1> &prob_recip    = ws.prob_recip;
          Eigen::Matrix<double, -1, 1>  &log_prev_per_obs_given_c = ws.log_prev_per_obs_given_c;
          Eigen::Matrix<double, -1, 1>  &prev_per_obs_given_c     = ws.prev_per_obs_given_c;

          for (int nc = 0; nc < n_total_chunks; nc++) {

            const int  chunk_counter = nc;
            const bool is_last       = (nc == n_full_chunks);                     //// only true when a remainder exists
            const int  chunk_size    = is_last ? last_chunk_size : normal_chunk_size;
            const int  row_start     = chunk_size_orig * chunk_counter;

            if (is_last) resize_MVP_workspace(ws, last_chunk_size, n_tests, n_class);

            u_grad_array_CM_chunk.setZero();

            y_chunk = y_ref.middleRows(row_start, chunk_size).cast<double>();

            //// ---- nuisance transform ----
            u_unc_vec_chunk = theta_us_vec_ref.segment(row_start * n_tests, chunk_size * n_tests);
            fn_MVP_compute_nuisance_T<vec>(u_vec_chunk, u_unc_vec_chunk, kchoice);
            log_jac_u += fn_MVP_compute_nuisance_log_jac_u_T<vec>(u_vec_chunk, u_unc_vec_chunk, d_J_wrt_duu_chunk /*scratch*/, kchoice);

            u_array = u_vec_chunk.reshaped(chunk_size, n_tests);
            y_sign.array() = 2.0 * y_chunk.array() - 1.0;
            y_m_y_sign_x_u.array() = y_chunk.array() - (y_sign.array() * u_array.array());

            //// ---- likelihood: class loop ----
            for (int c = 0; c < n_class; c++) {

                  if (n_class > 1) {
                    for (int n = 0; n < chunk_size; ++n) {
                      const int g = pop_ind(row_start + n);
                      log_prev_per_obs_given_c(n) = log_prev_mat_small(g, c);
                      prev_per_obs_given_c(n)     = prev_mat(g, c);
                    }
                  }

                  prod_container_or_inc_array.setZero();

                  for (int t = 0; t < n_tests; t++) {

                        if (n_covariates_max > 1) {
                              Eigen::Matrix<double, -1, 1> Xbeta_given_class_c_col_t =
                                  X[c][t].block(row_start, 0, chunk_size, n_covariates_per_outcome_vec(c, t)).cast<double>()
                                  * beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t));
                              Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * (-1.0 * (Xbeta_given_class_c_col_t.array() + prod_container_or_inc_array.array()));
                        } else {
                              Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * (-1.0 * (beta_double_array[c](0, t) + prod_container_or_inc_array.array()));
                        }

                        fn_MVP_compute_lp_GHK_cols_T<vec>(t, Bound_U_Phi_Bound_Z[c], Phi_Z[c], Z_std_norm[c], prob[c], y1_log_prob,
                                                        Bound_Z[c], y_chunk, u_array, kchoice);

                        if (t < n_tests - 1) {
                          prod_container_or_inc_array = Z_std_norm[c].leftCols(t + 1) * L_Omega_double[c].row(t + 1).head(t + 1).transpose();
                        }
                  }

                  rowwise_sum = y1_log_prob.rowwise().sum();
                  if (n_class > 1) {
                        rowwise_sum.array() += log_prev_per_obs_given_c.array();
                        lp_array.col(c) = rowwise_sum;
                  } else {
                        lp_array.col(0) = rowwise_sum;
                  }
            }

            //// ---- log-lik for this chunk ----
            if (n_class > 1) {
                  log_sum_exp_general_T<vec>(lp_array, log_sum_result, container_max_logs);
                  out_mat.segment(1 + n_params + row_start, chunk_size) = log_sum_result;
            } else {
                  out_mat.tail(N).segment(row_start, chunk_size) = lp_array.col(0);
            }

            log_lik_chunk = out_mat.tail(N).segment(row_start, chunk_size);
            prob_n = log_lik_chunk;
            apply_inplace<vec, Fn::exp>(prob_n);
            prob_n_recip = stan::math::inv(prob_n);

            //// ---- gradients: class loop ----
            for (int c = 0; c < n_class; c++) {

                  if (n_class > 1) {
                    for (int n = 0; n < chunk_size; ++n) prev_per_obs_given_c(n) = prev_mat(pop_ind(row_start + n), c);
                  }

                  prob_recip = stan::math::inv(prob[c]);

                  for (int t = 0; t < n_tests; t++) {
                        fn_MVP_compute_phi_Z_recip_cols_T<vec>(t, phi_Z_recip, Phi_Z[c], Z_std_norm[c], kchoice);
                        fn_MVP_compute_phi_Bound_Z_cols_T<vec>(t, phi_Bound_Z, Bound_U_Phi_Bound_Z[c], Bound_Z[c], kchoice);
                  }

                  if (grad_option != "none") {
                        fn_MVP_grad_prep( prob[c], y_sign, y_m_y_sign_x_u, L_Omega_recip_double[c],
                                          prev_per_obs_given_c, prob_n_recip, phi_Z_recip, phi_Bound_Z, prob_recip,
                                          prob_rowwise_prod_temp, prob_recip_rowwise_prod_temp, prob_rowwise_prod_temp_all,
                                          common_grad_term_1,
                                          y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                          y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                          Model_args_as_cpp_struct);
                  }

                  //// nuisance grads
                  if ((grad_option == "us_only") || (grad_option == "all")) {
                        u_grad_array_CM_chunk_block = u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests);
                        fn_MVP_compute_nuisance_grad_v2( u_grad_array_CM_chunk_block, phi_Z_recip, common_grad_term_1, L_Omega_double[c],
                                                         prob[c], prob_recip, prob_rowwise_prod_temp,
                                                         y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                         y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                         z_grad_term, grad_prob, prod_container_or_inc_array, derivs_chain_container_vec,
                                                         Model_args_as_cpp_struct);
                        u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests).array() += u_grad_array_CM_chunk_block.array();

                        if (c == n_class - 1) {
                          const int start_index = 1 + row_start * n_tests;
                          const int length = chunk_size * n_tests;
                          out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();
                          fn_MVP_nuisance_first_deriv_T<vec>(du_wrt_duu_chunk, u_vec_chunk, u_unc_vec_chunk, kchoice);
                          fn_MVP_nuisance_deriv_of_log_det_J_T<vec>(d_J_wrt_duu_chunk, u_vec_chunk, u_unc_vec_chunk, du_wrt_duu_chunk, kchoice);
                          out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
                          out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();
                        }
                  }

                  //// coefficient grads
                  if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only")) {
                        Eigen::Matrix<int, -1, 1> n_cov_vec_c = n_covariates_per_outcome_vec.row(c).transpose();
                        fn_MVP_compute_coefficients_grad_v3( c, beta_grad_array[c], X[c], n_cov_vec_c, row_start, n_covariates_max,
                                                             common_grad_term_1, L_Omega_double[c], prob[c], prob_recip, prob_rowwise_prod_temp,
                                                             y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                             y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                             z_grad_term, grad_prob, prod_container_or_inc_array, derivs_chain_container_vec,
                                                             true, Model_args_as_cpp_struct);
                  }

                  //// L_Omega grads
                  if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only")) {
                        fn_MVP_compute_L_Omega_grad_v3( U_Omega_grad_array[c], common_grad_term_1, L_Omega_double[c], prob[c], prob_recip,
                                                        Bound_Z[c], Z_std_norm[c], prob_rowwise_prod_temp,
                                                        y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                        y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                        z_grad_term, grad_prob, prod_container_or_inc_array, derivs_chain_container_vec,
                                                        true, Model_args_as_cpp_struct);
                  }

                  //// prevalence grads
                  if ((n_class > 1) && ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only"))) {
                        fn_MVP_prev_multi_pop_accumulate_grad(prob[c], prob_n_recip, pop_ind, row_start, chunk_size, n_pops, c, prev_grad_mat, rowwise_prod);
                  }

            }  // end of c loop (gradient)

          }  // end of chunk loop

        }  // end of workspace scope

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





//// -------------------------------------------------------------------------------------
//// Non-template entry point: the ONE runtime dispatch. Every existing caller stays as-is.
//// -------------------------------------------------------------------------------------
inline void fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                    const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                    const std::string &grad_option,
                                                                                    const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                    std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs
) {

        const Vec vec = vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0));

        if (vec == Vec::AVX512) {
              fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl_T<Vec::AVX512>(out_mat,
                                                                                                   theta_main_vec_ref,
                                                                                                   theta_us_vec_ref,
                                                                                                   y_ref,
                                                                                                   grad_option,
                                                                                                   Model_args_as_cpp_struct,
                                                                                                   LC_MVP_ws_structs);
        } else {
              fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl_T<Vec::Scalar>(out_mat,
                                                                                                   theta_main_vec_ref,
                                                                                                   theta_us_vec_ref,
                                                                                                   y_ref,
                                                                                                   grad_option,
                                                                                                   Model_args_as_cpp_struct,
                                                                                                   LC_MVP_ws_structs);
        }

}





  