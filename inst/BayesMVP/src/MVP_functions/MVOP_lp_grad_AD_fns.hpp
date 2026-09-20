



#pragma once






#include <Eigen/Dense>
 
#include <unsupported/Eigen/SpecialFunctions>












inline void  fn_lp_and_grad_MVOP_Pinkney_AD_log_scale_InPlace_process(   Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                         const Eigen::Matrix<double, -1, 1> theta_main_vec_ref,
                                                                         const Eigen::Matrix<double, -1, 1> theta_us_vec_ref,
                                                                         const Eigen::Matrix<int, -1, -1> y_ref,
                                                                         const std::string grad_option,
                                                                         const Model_fn_args_struct Model_args_as_cpp_struct,
                                                                         std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                         const int n_threads_WCP
) {
   
        //// important params
        const int N = y_ref.rows();
        const int n_tests = y_ref.cols();
        const int n_us = theta_us_vec_ref.rows()  ; 
        const int n_params_main =  theta_main_vec_ref.rows()  ; 
        const int n_params = n_params_main + n_us;
        
        //////////////  access elements from struct and read 
        const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>  &X =  Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0]; 
        
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
        
        const int n_cores = Model_args_as_cpp_struct.Model_args_ints(0);
        const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
        const int ub_threshold_phi_approx = Model_args_as_cpp_struct.Model_args_ints(2);
        const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
        
        const int n_pops = Model_args_as_cpp_struct.Model_args_ints(6); //// ----
        const Eigen::Matrix<int, -1, 1> &pop_ind = Model_args_as_cpp_struct.Model_args_col_vecs_int[2]; //// ----
        
        const int n_binary_tests  = Model_args_as_cpp_struct.Model_args_ints(4);  //// ordinal
        const int n_ordinal_tests = Model_args_as_cpp_struct.Model_args_ints(5);  //// ordinal
        
        const Eigen::Matrix<int, -1, 1> &n_cat_per_ord_test = Model_args_as_cpp_struct.Model_args_col_vecs_int[0];  //// ordinal
        const Eigen::Matrix<int, -1, 1> &n_thr_per_ord_test = Model_args_as_cpp_struct.Model_args_col_vecs_int[1];  //// ordinal
        // const Eigen::Matrix<double, -1, -1> &prior_dirichlet_alpha = Model_args_as_cpp_struct.Model_args_mats_double[4]; //// ordinal
        const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_dirichlet_alpha = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[7]; //// ---- ordinal-only
        
        // const double prev_prior_a = Model_args_as_cpp_struct.Model_args_doubles(0);
        // const double prev_prior_b = Model_args_as_cpp_struct.Model_args_doubles(1);
        const double overflow_threshold  = Model_args_as_cpp_struct.Model_args_doubles(0);
        const double underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(1);
        const double C_raw_lower = Model_args_as_cpp_struct.Model_args_doubles(2); //// ---- ordinal-only
        const double C_raw_upper = Model_args_as_cpp_struct.Model_args_doubles(3); //// ---- ordinal-only
        
        const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
        const std::string &Phi_type = Model_args_as_cpp_struct.Model_args_strings(1);
        const std::string &inv_Phi_type = Model_args_as_cpp_struct.Model_args_strings(2);
        const std::string &vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);
        const std::string &vect_type_log = Model_args_as_cpp_struct.Model_args_strings(4);
        const std::string &vect_type_lse = Model_args_as_cpp_struct.Model_args_strings(5);
        const std::string &vect_type_tanh = Model_args_as_cpp_struct.Model_args_strings(6);
        const std::string &vect_type_Phi = Model_args_as_cpp_struct.Model_args_strings(7);
        const std::string &vect_type_log_Phi = Model_args_as_cpp_struct.Model_args_strings(8);
        const std::string &vect_type_inv_Phi = Model_args_as_cpp_struct.Model_args_strings(9);
        const std::string &vect_type_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_strings(10);
        const std::string &nuisance_transformation =   Model_args_as_cpp_struct.Model_args_strings(12);
        
        const Eigen::Matrix<double, -1, 1> &lkj_cholesky_eta =   Model_args_as_cpp_struct.Model_args_col_vecs_double[0];
        const Eigen::Matrix<double, -1, 1> &prev_prior_a = Model_args_as_cpp_struct.Model_args_col_vecs_double[1]; //// ----
        const Eigen::Matrix<double, -1, 1> &prev_prior_b = Model_args_as_cpp_struct.Model_args_col_vecs_double[2]; //// ----
        
        const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
        
        const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_coeffs_mean  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0]; 
        const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_coeffs_sd   =  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1]; 
        const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_for_corr_a   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2]; 
        const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_for_corr_b   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3]; 
        const std::vector<Eigen::Matrix<double, -1, -1>>   &lb_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4]; 
        const std::vector<Eigen::Matrix<double, -1, -1>>   &ub_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5]; 
        const std::vector<Eigen::Matrix<double, -1, -1>>   &known_values    = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6]; 
        
        const std::vector<Eigen::Matrix<int, -1, -1>> &known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];
         
        //////////////
        const int n_corrs =  n_class * n_tests * (n_tests - 1) * 0.5;
        
        int n_covariates_total_nd, n_covariates_total_d, n_covariates_total;
        int n_covariates_max_nd, n_covariates_max_d, n_covariates_max;
        
        if (n_class > 1)  {
              
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
        
        const double a_times_3 = 3.0 * 0.07056;
        const double b = 1.5976;
        const double Inf = std::numeric_limits<double>::infinity();
        
        //// ---- determine chunk size --------------------------------------------------
        const int desired_n_chunks = n_chunks;
        int vec_size;
        if (vect_type == "AVX512")      vec_size = 8;
        else if (vect_type == "AVX2")   vec_size = 4;
        else if (vect_type == "AVX")    vec_size = 2;
        else                            vec_size = 1;
        
        ChunkSizeInfo chunk_size_info = calculate_chunk_sizes( N, 
                                                               vec_size,
                                                               desired_n_chunks);
        int chunk_size = chunk_size_info.chunk_size;
        int chunk_size_orig = chunk_size_info.chunk_size_orig;
        int normal_chunk_size = chunk_size_info.normal_chunk_size;
        int last_chunk_size = chunk_size_info.last_chunk_size;
        int n_total_chunks = chunk_size_info.n_total_chunks;
        int n_full_chunks = chunk_size_info.n_full_chunks;
        
        /////////////////  ------------------------------------------------------------ 
        using namespace stan::math;
         
        stan::math::nested_rev_autodiff nested_autodiff_scope;
        
        Eigen::Matrix<stan::math::var, -1, 1> theta_var(n_params);
        {
          Eigen::Matrix<double, -1, 1> theta(n_params);
          theta.head(n_us) = theta_us_vec_ref;
          theta.tail(n_params_main) = theta_main_vec_ref;
          theta_var = stan::math::to_var(theta);
        } 
        
        Eigen::Matrix<stan::math::var, -1, 1> u_unconstrained_vec_var = theta_var.head(n_us);
        
        //////////////  corrs  
        Eigen::Matrix<stan::math::var, -1, 1> theta_corrs_var = theta_var.segment(n_us, n_corrs);
        std::vector<stan::math::var> Omega_unconstrained_vec_var(n_corrs, 0.0);
        Omega_unconstrained_vec_var = Eigen_vec_to_std_vec_var(theta_corrs_var);
         
        //// coeffs
        std::vector<Eigen::Matrix<stan::math::var, -1, -1>> beta_all_tests_class_var = vec_of_mats_var(n_covariates_max, n_tests, n_class);
        {
          int i = n_us + n_corrs;
          for (int c = 0; c < n_class; ++c) {
            for (int t = 0; t < n_tests; ++t) {
              for (int k = 0; k < n_covariates_per_outcome_vec(c, t); ++k) {
                beta_all_tests_class_var[c](k, t) = theta_var(i);
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
        // if (n_class > 1 && n_binary_tests > 0) {
        //       const double beta_bin_nd = beta_all_tests_class_var[0](0, 0).val();
        //       const double beta_bin_d  = beta_all_tests_class_var[1](0, 0).val();
        //       if (!(beta_bin_d > beta_bin_nd)) {
        //             out_mat.setZero();
        //             out_mat(0) = -std::numeric_limits<double>::infinity();
        //             stan::math::recover_memory_nested();
        //             return;
        //       }
        // }
        //// prev  -- in MVOP, prev is at n_us + n_corrs + n_covariates_total (NOT n_params - 1, since cutpoints come after)
        // stan::math::var u_prev_diseased = 0.0;
        // if (n_class > 1)  u_prev_diseased = theta_var(n_us + n_corrs + n_covariates_total);
        Eigen::Matrix<stan::math::var, -1, 1> u_prev_raw_var(n_pops); //// ----
        if (n_class > 1) { //// ----
          for (int g = 0; g < n_pops; ++g) {
            u_prev_raw_var(g) = theta_var(n_us + n_corrs + n_covariates_total + g);
          }
        } 
        
        stan::math::var target_AD = 0.0;
        stan::math::var target_likelihood = 0.0;
        
        // ====================================================================
        // ORDINAL ADDITION #1: Unpack C_raw_var, transform to C, prior + Jacobian
        // ====================================================================
        const int n_cutpoints_max = (n_ordinal_tests > 0) ? n_thr_per_ord_test.maxCoeff() : 0;
        ////
        const double C_raw_range = C_raw_upper - C_raw_lower;
        const double log_half_range = std::log(0.5 * C_raw_range);
        ////
        std::vector<Eigen::Matrix<stan::math::var, -1, -1>> C_raw_var       = vec_of_mats_var(n_cutpoints_max, n_ordinal_tests, n_class);
        std::vector<Eigen::Matrix<stan::math::var, -1, -1>> C_var           = vec_of_mats_var(n_cutpoints_max, n_ordinal_tests, n_class);
        // std::vector<Eigen::Matrix<stan::math::var, -1, -1>> dC_raw_dunc     = vec_of_mats_var(n_cutpoints_max, n_ordinal_tests, n_class);
        ////
        // stan::math::var log_det_J_unc_to_C_raw = 0.0;
        {
          int i = n_us + n_corrs + n_covariates_total + n_pops;
          for (int c = 0; c < n_class; ++c) {
            for (int t_ord = 0; t_ord < n_ordinal_tests; ++t_ord) {
              for (int k = 0; k < n_thr_per_ord_test(t_ord); ++k) {
                
                    // stan::math::var C_unc_k      = theta_var(i);
                    stan::math::var C_unc_vec_var = theta_var(i);
                    ////
                    // C_raw_var[c](k, t_ord)       = C_raw_lower + C_raw_range * 0.5 * (1.0 + tanh_C_unc_k);
                    stan::math::var tanh_C_unc_var = stan::math::tanh(C_unc_vec_var);
                    C_raw_var[c](k, t_ord) = C_raw_lower + C_raw_range * 0.5 * (1.0 + tanh_C_unc_var);
                    // dC_raw_dunc[c](k, t_ord) = C_raw_range * 0.5 * (1.0 - tanh_C_unc_k * tanh_C_unc_k);
                    ////
                    //// log|dC_raw/dtheta| = log(0.5*range) + log(1 - tanh^2)
                    //// log(1 - tanh^2(x)) = 2*(log2 - |x| - log1p(exp(-2|x|)))  <- stable for large |x|
                    ////
                    // stan::math::var log_1m_tanh_sq = stan::math::log1m(stan::math::square(tanh_C_unc_var));
                    // stan::math::var abs_C_unc_k = stan::math::fabs(C_unc_k);
                    // stan::math::var log_1m_tanh_sq = 2.0 * (0.6931471805599453 - abs_C_unc_k - stan::math::log1p_exp(-2.0 * abs_C_unc_k));
                    // stan::math::var abs_C_unc_k = stan::math::fabs(C_unc_k);
                    // stan::math::var log_1m_tanh_sq = 2.0 * (0.6931471805599453 - abs_C_unc_k - stan::math::log1p(stan::math::exp(-2.0 * abs_C_unc_k)));
                    target_AD += stan::math::log(0.5 * C_raw_range) + stan::math::log1m(stan::math::square(tanh_C_unc_var));
                    // log_det_J_unc_to_C_raw += log_half_range + log_1m_tanh_sq;
                    ////
                    i += 1;
                
              }
            } 
          }
          // target_AD += log_det_J_unc_to_C_raw;
        }
        
        //// C_raw -> C transform + Jacobian + induced Dirichlet prior
        for (int c = 0; c < n_class; ++c) {
              
              for (int t_ord = 0; t_ord < n_ordinal_tests; ++t_ord) {
                
                    int n_thr_t = n_thr_per_ord_test(t_ord);
                    int n_cat_t = n_thr_t + 1;
                    
                    //// ordered cutpoints: C[0] = C_raw[0], C[k] = C[k-1] + exp(C_raw[k])
                    C_var[c](0, t_ord) = C_raw_var[c](0, t_ord);
                    for (int k = 1; k < n_thr_t; ++k) {
                      C_var[c](k, t_ord) = C_var[c](k - 1, t_ord) + stan::math::exp(C_raw_var[c](k, t_ord));
                    } 
                    
                    //// Jacobian for C_raw -> C: log|det(J)| = sum_{k>=1} C_raw[k]
                    for (int k = 1; k < n_thr_t; ++k) {
                      target_AD += C_raw_var[c](k, t_ord);
                    } 
                    
                    //// Induced Dirichlet: p_k = Phi(C_k) - Phi(C_{k-1}), then Dirichlet(alpha)
                    int t = n_binary_tests + t_ord;
                    stan::math::var anchor = stan::math::to_var(prior_coeffs_mean[c](0, t)); // prior of intercepts as anchor
                    ////
                    Eigen::Matrix<stan::math::var, -1, 1> cumul_probs(n_thr_t);
                    for (int k = 0; k < n_thr_t; ++k) {
                      cumul_probs(k) = stan::math::Phi(C_var[c](k, t_ord) - anchor);
                    } 
                    
                    Eigen::Matrix<stan::math::var, -1, 1> p_ord(n_cat_t);
                    p_ord(0) = cumul_probs(0);
                    for (int k = 1; k < n_thr_t; ++k) {
                      p_ord(k) = cumul_probs(k) - cumul_probs(k - 1);
                    }
                    p_ord(n_cat_t - 1) = 1.0 - cumul_probs(n_thr_t - 1);
                    
                    //// Dirichlet log-density: sum (alpha_k - 1) * log(p_k) + lgamma(sum(alpha)) - sum(lgamma(alpha_k))
                    Eigen::Matrix<double, -1, 1> alpha_t = prior_dirichlet_alpha[c].col(t_ord).head(n_cat_t);
                    double alpha_sum = alpha_t.sum();
                    target_AD += stan::math::lgamma(alpha_sum);
                    ////
                    if (prior_dirichlet_alpha[c].col(t_ord).head(n_cat_t).isOnes() == false) {
                      for (int k = 0; k < n_cat_t; ++k) {
                        target_AD -= stan::math::lgamma(alpha_t(k));
                        double coeff = alpha_t(k) - 1.0;
                        if (std::abs(coeff) > 1e-15) {
                          target_AD += coeff * stan::math::log(p_ord(k));
                        }
                      }
                    } 
                    ////
                    //// Jacobian for C -> p (induced Dirichlet): sum log(phi(C_k))  i.e. std_normal_lpdf
                    ////
                    for (int k = 0; k < n_thr_t; ++k) {
                      target_AD += stan::math::std_normal_lpdf(C_var[c](k, t_ord) - anchor);
                    }
                
              }
          
        }
        // ====================================================================
        // ==== END ORDINAL ADDITION #1 ====
        // ====================================================================
        
        /////////////  prev stuff  -------------------------------------------------------------
        Eigen::Matrix<stan::math::var, -1, -1> prev_var = Eigen::Matrix<stan::math::var, -1, -1>::Zero(n_pops, n_class);
        
        if (n_class > 1) {
          
          for (int g = 0; g < n_pops; ++g) {
            
            stan::math::var tanh_u = stan::math::tanh(u_prev_raw_var(g));
            stan::math::var prev_g = 0.5 * (tanh_u + 1.0);
            prev_var(g, 1) = prev_g;
            prev_var(g, 0) = 1.0 - prev_g;
            
            // Jacobian: d(prev)/d(u_raw) = 0.5 * (1 - tanh^2)
            stan::math::var deriv_p = 0.5 * (1.0 - tanh_u * tanh_u);
            target_AD += stan::math::log(deriv_p);
            
            // Prior on prev for this population
            target_AD += stan::math::beta_lpdf(prev_var(g, 1), prev_prior_a(g), prev_prior_b(g));
            
          }
          
        } 
        
        ////////////////// u 
        Eigen::Matrix<stan::math::var, -1, 1>  u_vec(n_us);
        stan::math::var log_jac_u = 0.0;
        
        if (nuisance_transformation == "Phi") {
          u_vec.array() =   Phi(u_unconstrained_vec_var).array();
          log_jac_u +=    - 0.5 * log(2 * M_PI) -  0.5 * sum(square(u_unconstrained_vec_var)) ;
        } else if (nuisance_transformation == "Phi_approx") {
          u_vec.array() =   Phi_approx(u_unconstrained_vec_var).array();
          log_jac_u   +=    (a_times_3 * u_unconstrained_vec_var.array().square() +  b).array().log().sum();
          log_jac_u   +=    sum(log(u_vec));
          log_jac_u   +=    sum(log1m(u_vec));
        } else if (nuisance_transformation == "Phi_approx_rough") {
          u_vec.array() =   inv_logit(1.702 * u_unconstrained_vec_var).array();
          log_jac_u   +=    log(1.702) ;
          log_jac_u   +=    sum(log(u_vec));
          log_jac_u   +=    sum(log1m(u_vec));
        } else if (nuisance_transformation == "tanh") {
          Eigen::Matrix<stan::math::var, -1, 1> tanh_u_unc = tanh(u_unconstrained_vec_var);
          u_vec.array() =     0.5 * (  tanh_u_unc.array() + 1.0).array() ;
          log_jac_u  +=   - log(2.0) ;
          log_jac_u   +=    sum(log(u_vec));
          log_jac_u   +=    sum(log1m(u_vec));
        }
        
        ///////////////// get cholesky factor's of corr matrices
        std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > Omega_unconstrained_var = fn_convert_std_vec_of_corrs_to_3d_array_var(Omega_unconstrained_vec_var, 
                                                                                                                                    n_tests, 
                                                                                                                                    n_class);
        std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > L_Omega_var = vec_of_mats_var(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_var  = vec_of_mats_var(n_tests, n_tests, n_class);
        
        for (int c = 0; c < n_class; ++c) {
          
              Eigen::Matrix<stan::math::var, -1, -1  >  Chol_Schur_outs =  Pinkney_corr_master(  n_tests,
                                                                                                 lb_corr[c], 
                                                                                                 ub_corr[c], 
                                                                                                 Omega_unconstrained_var[c], 
                                                                                                 known_values_indicator[c], 
                                                                                                 known_values[c]);
              L_Omega_var[c]   =  Chol_Schur_outs.block(1, 0, n_tests, n_tests);
              target_AD +=   Chol_Schur_outs(0, 0);
              Omega_var[c] =   L_Omega_var[c] * L_Omega_var[c].transpose() ;
              
              for (int i = 1; i < n_tests; ++i) {    ////   for (i in 2:n_tests) {
                for (int j = 0; j < i; ++j) {   ////   for (j in 1:(i - 1)) {
                  if (known_values_indicator[c](i, j) == 1) {
                    stan::math::var known_val_prior_ij = stan::math::normal_lpdf( Omega_var[c](i, j), 0.0, 10.0 ); //// to ensure any corr's we aren't estimating dont cause divergences
                    target_AD += known_val_prior_ij;
                  }
                }
              }
          
        }
        
        // {
        //   // double raw_prior_val_double = 0.0;
        //   for (int i = 0; i < (Omega_unconstrained_vec_var.size()); ++i) {
        //     stan::math::var raw_prior_i = stan::math::normal_lpdf(Omega_unconstrained_vec_var[i], 0.0, 10.0);
        //     target_AD += raw_prior_i;
        //     // raw_prior_val_double += raw_prior_i.val();
        //   }
        // }
        
        ///////////////////////////////////////////////////////////////////////// prior densities
        for (int c = 0; c < n_class; c++) {
          for (int t = 0; t < n_tests; t++) {
            for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++) {
              target_AD  += stan::math::normal_lpdf(beta_all_tests_class_var[c](k, t), prior_coeffs_mean[c](k, t), prior_coeffs_sd[c](k, t));
            }
          }
          target_AD += stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c], lkj_cholesky_eta(c));
        }
        
        /////////////////////////////////////////////////////////////////////////////////////////////////////////// likelihood
        Eigen::Matrix<stan::math::var, -1, 1> y1 = Eigen::Matrix<stan::math::var, -1, 1>::Zero(n_tests);
        Eigen::Matrix<stan::math::var, -1, 1> lp = Eigen::Matrix<stan::math::var, -1, 1>::Zero(n_class);
        Eigen::Matrix<stan::math::var, -1, 1> Z_std_norm = Eigen::Matrix<stan::math::var, -1, 1>::Zero(n_tests);
        Eigen::Matrix<stan::math::var, -1, -1> u_array = Eigen::Matrix<stan::math::var, -1, -1>::Zero(N, n_tests);
        stan::math::var Xbeta_n = 0.0;
        
        //// ---- Finite "pseudo-infinity" cutpoint sentinel for the unbounded bottom/top categories
        ////      (and for the open side of a BINARY test). Mirrors LC_MVOP_PartialLog_v2.stan.
        ////      Do NOT use +/-inf: a saturated Phi() contributes an exactly-zero adjoint, and
        ////      0 * inf = NaN in reverse-mode AD. With 1000 the partial is finite (~2e5), so
        ////      0 * finite = 0 and the tape stays clean.
        const double C_sentinel = 1000.0;
        
        if (g_debug_cutpoint_grads) {
          
          std::cout << "N=" << N << " n_us=" << n_us << " n_params_main=" << n_params_main 
                    << " n_params=" << n_params << " theta_main_vec_ref.size()=" << theta_main_vec_ref.rows()
                    << " theta_us_vec_ref.size()=" << theta_us_vec_ref.rows() << std::endl;
          std::cout << "chunk_size_orig=" << chunk_size_orig << " n_total_chunks=" << n_total_chunks 
                    << " n_full_chunks=" << n_full_chunks << " last_chunk_size=" << last_chunk_size << std::endl;
          
        }
        
        int i = 0;
        for (int nc = 0; nc < n_total_chunks; nc++) {
              
              int current_chunk_size;
          
              if (n_total_chunks != n_full_chunks) {
                current_chunk_size = (nc == n_full_chunks) ? last_chunk_size : chunk_size_orig;
              } else {
                current_chunk_size = chunk_size_orig;
              }
              
              for (int t = 0; t < n_tests; t++) {
                for (int n = 0; n < current_chunk_size; n++) {
                  int n_index = nc * chunk_size_orig + n;
                  if (n_index < N) {
                    u_array(n_index, t) = u_vec(t * N + n_index);
                  }
                }
              }
          
        }
        
        Eigen::Matrix<stan::math::var, -1, -1> log_prev = Eigen::Matrix<stan::math::var, -1, -1>::Zero(n_pops, n_class);
        if (n_class > 1) {
          log_prev = stan::math::log(prev_var);
        }
        
        for (int n = 0; n < N; n++ ) {
          
          for (int c = 0; c < n_class; ++c) {
                
                stan::math::var inc  = 0.0;
                
                for (int t = 0; t < n_tests; t++) {
                  
                      if (n_covariates_max > 1) {
                        Xbeta_n = ( X[c][t].row(n).head(n_covariates_per_outcome_vec(c, t)).cast<double>() * beta_all_tests_class_var[c].col(t).head(n_covariates_per_outcome_vec(c, t))  ).eval()(0, 0) ;
                      } else {
                        Xbeta_n = beta_all_tests_class_var[c](0, t);
                      }
                      
                      const stan::math::var mu_t        = Xbeta_n + inc;
                      const stan::math::var L_tt_recip  = 1.0 / L_Omega_var[c](t, t);
                      
                      //// ============================================================================
                      //// UNIFIED INTERVAL: slot t contributes  P( C_lo < Z*_t <= C_hi ).
                      ////   BINARY  y=0 -> (-sentinel, 0]        y=1 -> (0, +sentinel]
                      ////   ORDINAL k   -> (C[k-1], C[k]], with sentinels for k = 1 and k = K.
                      //// For K = 2 the ordinal formulas below reduce EXACTLY to the binary
                      //// underflow/overflow formulas in the MVP AD file.
                      //// ============================================================================
                      stan::math::var C_lo;
                      stan::math::var C_hi;
                      
                      if (t < n_binary_tests) {
                            
                            if (y_ref(n, t) == 1) { C_lo =  0.0;          C_hi =  C_sentinel; }
                            else                  { C_lo = -C_sentinel;   C_hi =  0.0;        }
                        
                      } else {
                            
                            const int t_ord = t - n_binary_tests;
                            const int K_t   = n_cat_per_ord_test(t_ord);
                            const int y_val = y_ref(n, t);   //// 1-indexed category
                            ////
                            if (y_val == 1)   C_lo = -C_sentinel;  else  C_lo = C_var[c](y_val - 2, t_ord);
                            if (y_val == K_t) C_hi =  C_sentinel;  else  C_hi = C_var[c](y_val - 1, t_ord);
                        
                      }
                      
                      const stan::math::var Bound_Z_lo = (C_lo - mu_t) * L_tt_recip;
                      const stan::math::var Bound_Z_hi = (C_hi - mu_t) * L_tt_recip;
                      
                      //// ---- Classify. Note Bound_Z_lo <= Bound_Z_hi ALWAYS (cutpoints strictly
                      ////      increasing, 1/L_tt > 0), so the three regimes are disjoint:
                      ////        left  <=> BOTH bounds in the left tail  <=> Bound_Z_hi < UF
                      ////        right <=> BOTH bounds in the right tail <=> Bound_Z_lo > OF
                      ////        else  -> standard scale (incl. straddling / one-extreme-one-moderate,
                      ////                 which do not cancel).
                      ////
                      if (Bound_Z_hi < underflow_threshold) {
                        
                            //// ================= LEFT TAIL (log scale) =================
                            //// log Phi(x) ~=~ log_inv_logit(0.07056*x^3 + 1.5976*x)
                            ////
                            const stan::math::var log_Phi_lo = stan::math::log_inv_logit( 0.07056 * stan::math::square(Bound_Z_lo) * Bound_Z_lo + 1.5976 * Bound_Z_lo );
                            const stan::math::var log_Phi_hi = stan::math::log_inv_logit( 0.07056 * stan::math::square(Bound_Z_hi) * Bound_Z_hi + 1.5976 * Bound_Z_hi );
                            ////
                            //// y1 = log(Phi_hi - Phi_lo) = log_diff_exp(log_Phi_hi, log_Phi_lo).
                            //// Bottom category: log_Phi_lo ~ -7e7 -> exp() underflows to exact 0
                            //// -> y1 = log_Phi_hi, identical to the binary "underflow & y==0" form:
                            ////
                            y1(t) = log_Phi_hi + stan::math::log1m_exp(log_Phi_lo - log_Phi_hi);
                            ////
                            //// Phi_Z = Phi_lo*(1 - u) + Phi_hi*u:
                            ////
                            const stan::math::var log_Phi_Z    = stan::math::log_sum_exp( log_Phi_lo + stan::math::log1m(u_array(n, t)),
                                                                                          log_Phi_hi + stan::math::log(u_array(n, t)) );
                            const stan::math::var log_1m_Phi_Z = stan::math::log1m_exp(log_Phi_Z);
                            const stan::math::var logit_Phi_Z  = log_Phi_Z - log_1m_Phi_Z;
                            ////
                            Z_std_norm(t) = inv_Phi_approx_from_logit_prob_var(logit_Phi_Z);
                        
                      } else if (Bound_Z_lo > overflow_threshold) {
                        
                            //// ================= RIGHT TAIL (log scale) =================
                            //// log(1 - Phi(x)) ~=~ log_inv_logit( -(0.07056*x^3 + 1.5976*x) )
                            ////
                            const stan::math::var log_1m_Phi_lo = stan::math::log_inv_logit( - 0.07056 * stan::math::square(Bound_Z_lo) * Bound_Z_lo - 1.5976 * Bound_Z_lo );
                            const stan::math::var log_1m_Phi_hi = stan::math::log_inv_logit( - 0.07056 * stan::math::square(Bound_Z_hi) * Bound_Z_hi - 1.5976 * Bound_Z_hi );
                            ////
                            //// y1 = log((1 - Phi_lo) - (1 - Phi_hi)) = log_diff_exp(log_1m_Phi_lo, log_1m_Phi_hi).
                            //// Top category: log_1m_Phi_hi ~ -7e7 -> y1 = log_1m_Phi_lo, identical to
                            //// the binary "overflow & y==1" form:
                            ////
                            y1(t) = log_1m_Phi_lo + stan::math::log1m_exp(log_1m_Phi_hi - log_1m_Phi_lo);
                            ////
                            //// 1 - Phi_Z = (1 - Phi_lo)*(1 - u) + (1 - Phi_hi)*u:
                            ////
                            const stan::math::var log_1m_Phi_Z = stan::math::log_sum_exp( log_1m_Phi_lo + stan::math::log1m(u_array(n, t)),
                                                                                          log_1m_Phi_hi + stan::math::log(u_array(n, t)) );
                            const stan::math::var log_Phi_Z    = stan::math::log1m_exp(log_1m_Phi_Z);
                            const stan::math::var logit_Phi_Z  = log_Phi_Z - log_1m_Phi_Z;
                            ////
                            Z_std_norm(t) = inv_Phi_approx_from_logit_prob_var(logit_Phi_Z);
                        
                      } else {
                        
                            //// ================= STANDARD SCALE =================
                            //// Guaranteed here: Bound_Z_hi >= UF and Bound_Z_lo <= OF, so
                            //// prob_t >= ~Phi(UF) > 0 -- no log(0), bounded cancellation.
                            ////
                            const stan::math::var Phi_lo = stan::math::Phi(Bound_Z_lo);
                            const stan::math::var Phi_hi = stan::math::Phi(Bound_Z_hi);
                            const stan::math::var prob_t = Phi_hi - Phi_lo;
                            ////
                            y1(t) = stan::math::log(prob_t);
                            ////
                            const stan::math::var Phi_Z = Phi_lo + prob_t * u_array(n, t);
                            Z_std_norm(t) = stan::math::inv_Phi(Phi_Z);
                        
                      }
                      
                      if (t < n_tests - 1)    inc  = (L_Omega_var[c].row(t+1).head(t+1) * Z_std_norm.head(t+1)).eval()(0, 0);
                  
                } // end of t loop
                
                if (n_class > 1) {
                  int g = pop_ind(n);
                  lp(c) = log_prev(g, c) + y1.sum();
                } else {
                  lp(0) = y1.sum();
                }
            
          } ///// end of c loop
          
          if (n_class > 1) {
            stan::math::var log_posterior = stan::math::log_sum_exp(lp);
            target_likelihood += log_posterior;
            if (out_mat.size() >= 1 + n_params + N) out_mat(1 + n_params + n) = log_posterior.val();
          } else {
            stan::math::var log_posterior = lp(0);
            target_likelihood += log_posterior;
            if (out_mat.size() >= 1 + n_params + N) out_mat(1 + n_params + n) = log_posterior.val();
          }
          
        } // end of n loop
        
        // int i = 0;
        // for (int nc = 0; nc < n_total_chunks; nc++) {
        //   int current_chunk_size;
        //   if (n_total_chunks != n_full_chunks) {
        //     current_chunk_size = (nc == n_full_chunks) ? last_chunk_size : chunk_size_orig;
        //   } else {
        //     current_chunk_size = chunk_size_orig;
        //   }
        // //   for (int t = 0; t < n_tests; t++) {
        // //     for (int n = 0; n < current_chunk_size; n++) {
        // //       int n_index = nc * chunk_size_orig + n;
        // //       if (n_index < N && i < n_us) {
        // //         u_array(n_index, t) = u_vec(i);
        // //         i++;
        // //       }
        // //     }
        // //   }
        // // }
        // 
        //   for (int t = 0; t < n_tests; t++) {
        //     for (int n = 0; n < current_chunk_size; n++) {
        //       
        //       // if (n % 50 == 0) std::cout << "AD n=" << n << "/" << N << std::flush << std::endl;
        //       
        //       int n_index = nc * chunk_size_orig + n;
        //       if (n_index < N) {
        //         u_array(n_index, t) = u_vec(t * N + n_index);
        //       }
        //     }
        //   }
        //   
        //   // for (int t = 0; t < n_tests; t++) {
        //   //   for (int n = 0; n < current_chunk_size; n++) {
        //   //     int n_index = nc * chunk_size_orig + n;
        //   //     if (n_index < N && i < n_us) {
        //   //       u_array(n_index, t) = u_vec(i);
        //   //     }
        //   //     i++;  // always advance, even for padding positions
        //   //   }
        //   // }
        // }

        target_AD += target_likelihood;
        target_AD += log_jac_u;
        double log_prob = target_AD.val();
        // std::cout << "AD log_prob=" << log_prob << std::endl;
        
        out_mat(0) = log_prob;
        // std::cout << "AD before grad" << std::endl;
        
        //////////////////// calculate gradients
        // out_mat(0) = log_prob;
        std::vector<stan::math::var> theta_grad;
        
        if (grad_option == "all") {
          
              for (int i = 0; i < n_params; i++) {
                theta_grad.push_back(theta_var(i));
              }
              std::vector<double> gradient_std_vec(n_params, 0.0);
              Eigen::Matrix<double, -1, 1>  gradient_vec(n_params);
              target_AD.grad(theta_grad, gradient_std_vec);
              gradient_vec = std_vec_to_Eigen_vec(gradient_std_vec);
              out_mat.segment(1, n_params) = gradient_vec;
          
        } else if (grad_option == "us_only") {
          
              for (int i = 0; i < n_us; i++) {
                theta_grad.push_back(theta_var(i));
              }
              std::vector<double> gradient_std_vec(n_us, 0.0);
              Eigen::Matrix<double, -1, 1>  gradient_vec(n_us);
              target_AD.grad(theta_grad, gradient_std_vec);
              gradient_vec = std_vec_to_Eigen_vec(gradient_std_vec);
              out_mat.segment(1, n_us) = gradient_vec;
          
        } else if (grad_option == "main_only") {
          
              for (int i = n_us; i < n_params; i++) {
                theta_grad.push_back(theta_var(i));
              }
              std::vector<double> gradient_std_vec(n_params_main, 0.0);
              Eigen::Matrix<double, -1, 1>  gradient_vec(n_params_main);
              target_AD.grad(theta_grad, gradient_std_vec);
              gradient_vec = std_vec_to_Eigen_vec(gradient_std_vec);
              out_mat.segment(1 + n_us, n_params_main) = gradient_vec;
          
        } else if (grad_option == "L_diag") {
          
              std::vector<stan::math::var> L_vars;
              for (int c = 0; c < n_class; c++) {
                for (int i = 0; i < n_tests; i++) {
                  for (int j = 0; j <= i; j++) {
                    L_vars.push_back(L_Omega_var[c](i, j));
                  }
                }
              }
               
              std::vector<double> L_grads(L_vars.size(), 0.0);
              target_likelihood.grad(L_vars, L_grads);  // likelihood only, no priors
              
              int idx = 0;
              for (int c = 0; c < n_class; c++) {
                std::cout << "AD dLikelihood/dL[" << c << "]:\n";
                for (int i = 0; i < n_tests; i++) {
                  for (int j = 0; j <= i; j++) {
                    std::cout << "  (" << i << "," << j << ") = " << L_grads[idx++] << "\n";
                  }
                }
              } 
          
        }
        
        
        // std::cout << "AD after grad" << std::endl;
        
        //// nested_autodiff_scope releases the tape, including on exceptions.
        // std::cout << "AD after recover" << std::endl;
  
}
















// Internal function using Eigen::Ref as inputs for matrices
inline void     fn_lp_and_grad_MVOP_Pinkney_AD_log_scale_InPlace(   Eigen::Matrix<double, -1, 1> &&out_mat_R_val, 
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
  
  fn_lp_and_grad_MVOP_Pinkney_AD_log_scale_InPlace_process(  out_mat_ref,
                                                            theta_main_vec_ref,
                                                            theta_us_vec_ref,
                                                            y_ref,
                                                            grad_option,
                                                            Model_args_as_cpp_struct, 
                                                            LC_MVP_ws_structs, 
                                                            n_threads_WCP);
  
}   








// Internal function using Eigen::Ref as inputs for matrices
inline void     fn_lp_and_grad_MVOP_Pinkney_AD_log_scale_InPlace(   Eigen::Matrix<double, -1, 1> &out_mat_ref, 
                                                            const Eigen::Matrix<double, -1, 1> &theta_main_vec_ref,
                                                            const Eigen::Matrix<double, -1, 1> &theta_us_vec_ref,
                                                            const Eigen::Matrix<int, -1, -1> &y_ref,
                                                            const std::string &grad_option,
                                                            const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                            std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                            const int n_threads_WCP 
                                                            
                                                            
                                                            
                                                            
) {
  
  fn_lp_and_grad_MVOP_Pinkney_AD_log_scale_InPlace_process( out_mat_ref,
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
inline void     fn_lp_and_grad_MVOP_Pinkney_AD_log_scale_InPlace(   Eigen::Ref<Eigen::Block<MatrixType, -1, 1>>  &out_mat_ref, 
                                                            const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_main_vec_ref,
                                                            const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_us_vec_ref,
                                                            const Eigen::Matrix<int, -1, -1> &y_ref,
                                                            const std::string &grad_option,
                                                            const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                            std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                            const int n_threads_WCP
                                                            
                                                             
                                                            
                                                            
) { 
  
  fn_lp_and_grad_MVOP_Pinkney_AD_log_scale_InPlace_process( out_mat_ref,
                                                           theta_main_vec_ref,
                                                           theta_us_vec_ref,
                                                           y_ref,
                                                           grad_option, 
                                                           Model_args_as_cpp_struct, 
                                                           LC_MVP_ws_structs,
                                                           n_threads_WCP);
  
}      







 






// Internal function using Eigen::Ref as inputs for matrices
inline Eigen::Matrix<double, -1, 1>    fn_lp_and_grad_MVOP_Pinkney_AD_log_scale(  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
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
  
  fn_lp_and_grad_MVOP_Pinkney_AD_log_scale_InPlace(  out_mat,
                                                    theta_main_vec_ref,
                                                    theta_us_vec_ref, 
                                                    y_ref,
                                                    grad_option,
                                                    Model_args_as_cpp_struct, 
                                                    LC_MVP_ws_structs,
                                                    n_threads_WCP);
  
  return out_mat;
   
}  
























