
#pragma once
 
 
 
 
 

 
#include <Eigen/Dense>
 


 
 

 
 
 
  





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




inline  void                             fn_lp_and_grad_std_MVP_Pinkney_AD_log_scale_InPlace_process(    Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
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
  const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>  X =  Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0]; 
  
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
  const int n_class = 1; ///// Model_args_as_cpp_struct.Model_args_ints(1);
  const int ub_threshold_phi_approx = Model_args_as_cpp_struct.Model_args_ints(2);
  const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
  
  // const int n_pops = Model_args_as_cpp_struct.Model_args_ints(6); //// ----
  // const Eigen::Matrix<int, -1, 1> &pop_ind = Model_args_as_cpp_struct.Model_args_col_vecs_int[2]; //// ----
  
  // const double prev_prior_a = Model_args_as_cpp_struct.Model_args_doubles(0);
  // const double prev_prior_b = Model_args_as_cpp_struct.Model_args_doubles(1);
  const double overflow_threshold  = Model_args_as_cpp_struct.Model_args_doubles(0);
  const double underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(1);
  
  const std::string vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
  const std::string Phi_type = Model_args_as_cpp_struct.Model_args_strings(1);
  const std::string inv_Phi_type = Model_args_as_cpp_struct.Model_args_strings(2);
  const std::string vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);
  const std::string vect_type_log = Model_args_as_cpp_struct.Model_args_strings(4);
  const std::string vect_type_lse = Model_args_as_cpp_struct.Model_args_strings(5);
  const std::string vect_type_tanh = Model_args_as_cpp_struct.Model_args_strings(6);
  const std::string vect_type_Phi = Model_args_as_cpp_struct.Model_args_strings(7);
  const std::string vect_type_log_Phi = Model_args_as_cpp_struct.Model_args_strings(8);
  const std::string vect_type_inv_Phi = Model_args_as_cpp_struct.Model_args_strings(9);
  const std::string vect_type_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_strings(10);
  ////// const std::string grad_option =  Model_args_as_cpp_struct.Model_args_strings(11);
  const std::string nuisance_transformation =   Model_args_as_cpp_struct.Model_args_strings(12);
  
  const Eigen::Matrix<double, -1, 1> &lkj_cholesky_eta =   Model_args_as_cpp_struct.Model_args_col_vecs_double[0];
  // const Eigen::Matrix<double, -1, 1> &prev_prior_a = Model_args_as_cpp_struct.Model_args_col_vecs_double[1]; //// ----
  // const Eigen::Matrix<double, -1, 1> &prev_prior_b = Model_args_as_cpp_struct.Model_args_col_vecs_double[2]; //// ----
  
  const Eigen::Matrix<int, -1, -1> n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
  
  const std::vector<Eigen::Matrix<double, -1, -1>>   prior_coeffs_mean  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0]; 
  const std::vector<Eigen::Matrix<double, -1, -1>>   prior_coeffs_sd   =  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1]; 
  const std::vector<Eigen::Matrix<double, -1, -1>>   prior_for_corr_a   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2]; 
  const std::vector<Eigen::Matrix<double, -1, -1>>   prior_for_corr_b   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3]; 
  const std::vector<Eigen::Matrix<double, -1, -1>>   lb_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4]; 
  const std::vector<Eigen::Matrix<double, -1, -1>>   ub_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5]; 
  const std::vector<Eigen::Matrix<double, -1, -1>>   known_values    = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6]; 
  
  const std::vector<Eigen::Matrix<int, -1, -1>> known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];
   
  //////////////
  const int n_corrs =  1 * n_tests * (n_tests - 1) * 0.5;
  
  int n_covariates_total_nd, n_covariates_total_d, n_covariates_total;
  int n_covariates_max_nd, n_covariates_max_d, n_covariates_max;
  
    
  n_covariates_total = n_covariates_per_outcome_vec.sum();
  n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();
  
  
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
  if (vect_type == "AVX512") {
    vec_size = 8;
  } else  if (vect_type == "AVX2") { 
    vec_size = 4;
  } else  if (vect_type == "AVX") { 
    vec_size = 2;
  } else { 
    vec_size = 1;
  }
   
  ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, desired_n_chunks); 
  
  int chunk_size = chunk_size_info.chunk_size;
  int chunk_size_orig = chunk_size_info.chunk_size_orig;
  int normal_chunk_size = chunk_size_info.normal_chunk_size;
  int last_chunk_size = chunk_size_info.last_chunk_size;
  int n_total_chunks = chunk_size_info.n_total_chunks;
  int n_full_chunks = chunk_size_info.n_full_chunks;
  
  /////////////////  ------------------------------------------------------------ 
  using namespace stan::math;
 
  stan::math::nested_rev_autodiff nested_autodiff_scope;
  
  Eigen::Matrix<stan::math::var, -1, 1  >  theta_var(n_params);

  {
    Eigen::Matrix<double, -1, 1> theta(n_params);
    theta.head(n_us) = theta_us_vec_ref;
    theta.tail(n_params_main) = theta_main_vec_ref;
    theta_var = stan::math::to_var(theta);
  }

  Eigen::Matrix<stan::math::var, -1, 1>    u_unconstrained_vec_var = theta_var.head(n_us);   // u's
  
  //////////////
  //// corrs  
  Eigen::Matrix<stan::math::var, -1, 1> theta_corrs_var = theta_var.segment(n_us, n_corrs);  // corrs
  std::vector<stan::math::var>  Omega_unconstrained_vec_var(n_corrs, 0.0);
  Omega_unconstrained_vec_var = Eigen_vec_to_std_vec_var(theta_corrs_var);

  //// coeffs
  std::vector<Eigen::Matrix<stan::math::var, -1, -1>> beta_all_tests_class_var = vec_of_mats_var(n_covariates_max, n_tests,  1); // coeffs

  {
    int i = n_us + n_corrs;
      for (int t = 0; t < n_tests; ++t) {
        for (int k = 0; k < n_covariates_per_outcome_vec(0, t); ++k) {
          beta_all_tests_class_var[0](k, t) = theta_var(i);
          i += 1;
        }
      }
  }
  
  stan::math::var target = 0.0;

  ////////////////// u (double / manual diff)
  Eigen::Matrix<stan::math::var, -1, 1>  u_vec(n_us);
  stan::math::var log_jac_u = 0.0;

  if (nuisance_transformation == "Phi") { // correct
    u_vec.array() =   Phi(u_unconstrained_vec_var).array(); // correct
    log_jac_u +=    - 0.5 * log(2 * M_PI) -  0.5 * sum(square(u_unconstrained_vec_var)) ;   // correct
  } else if (nuisance_transformation == "Phi_approx") {  // correct
    u_vec.array() =   Phi_approx(u_unconstrained_vec_var).array();
    log_jac_u   +=    (a_times_3 * u_unconstrained_vec_var.array().square() +  b).array().log().sum();  // correct
    log_jac_u   +=    sum(log(u_vec));  // correct
    log_jac_u   +=    sum(log1m(u_vec));    // correct
  } else if (nuisance_transformation == "Phi_approx_rough") {
    u_vec.array() =   inv_logit(1.702 * u_unconstrained_vec_var).array();  // correct
    log_jac_u   +=    log(1.702) ;  // correct
    log_jac_u   +=    sum(log(u_vec));  // correct
    log_jac_u   +=    sum(log1m(u_vec));  // correct
  } else if (nuisance_transformation == "tanh") {
    Eigen::Matrix<stan::math::var, -1, 1> tanh_u_unc = tanh(u_unconstrained_vec_var);   // correct
    u_vec.array() =     0.5 * (  tanh_u_unc.array() + 1.0).array() ;   // correct
    log_jac_u  +=   - log(2.0) ;   // correct
    log_jac_u   +=    sum(log(u_vec));  // correct
    log_jac_u   +=    sum(log1m(u_vec));  // correct
  }

  ///////////////// get cholesky factor's (lower-triangular) of corr matrices
  ////// first need to convert Omega_unconstrained to var   // then convert to 3d var array
  std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > Omega_unconstrained_var = fn_convert_std_vec_of_corrs_to_3d_array_var(Omega_unconstrained_vec_var, n_tests, 1);
  std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > L_Omega_var = vec_of_mats_var(n_tests, n_tests, 1);
  std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > Omega_var  = vec_of_mats_var(n_tests, n_tests, 1);
    
  Eigen::Matrix<stan::math::var, -1, -1 >  ub = stan::math::to_var(ub_corr[0]);
  Eigen::Matrix<stan::math::var, -1, -1 >  lb = stan::math::to_var(lb_corr[0]);
  Eigen::Matrix<stan::math::var, -1, -1  >  Chol_Schur_outs =  Pinkney_LDL_bounds_opt(n_tests, lb, ub, Omega_unconstrained_var[0], known_values_indicator[0], known_values[0]) ;
  L_Omega_var[0]   =  Chol_Schur_outs.block(1, 0, n_tests, n_tests);  // stan::math::cholesky_decompose( Omega_var[0]) ;
  target +=   Chol_Schur_outs(0, 0); // now can set prior directly on Omega
  Omega_var[0] =   L_Omega_var[0] * L_Omega_var[0].transpose() ;


  ///////////////////////////////////////////////////////////////////////// prior densities
  for (int t = 0; t < n_tests; t++) {
      for (int k = 0; k < n_covariates_per_outcome_vec(0, t); k++) {
        target  += stan::math::normal_lpdf(beta_all_tests_class_var[0](k, t), prior_coeffs_mean[0](k, t), prior_coeffs_sd[0](k, t));
      }
  }
  target +=  stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[0], lkj_cholesky_eta(0)) ;
    
  /////////////////////////////////////////////////////////////////////////////////////////////////////////// likelihood
  Eigen::Matrix<stan::math::var, -1, 1>	   y1(n_tests);
  Eigen::Matrix<stan::math::var, -1, 1>	   lp(1);
  Eigen::Matrix<stan::math::var, -1, 1>	   Z_std_norm(n_tests);
  Eigen::Matrix<stan::math::var, -1, -1>	 u_array(N, n_tests);
  stan::math::var Xbeta_n = 0.0;
  
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
            if (n_index < N && i < n_us) {
              u_array(n_index, t) = u_vec(i);
              i++;
            }
          }
        }
    
  }

    for (int n = 0; n < N; n++ ) {

      {

        stan::math::var inc  = 0.0;

        for (int t = 0; t < n_tests; t++) {

          if (n_covariates_max > 1) {
             Xbeta_n = ( X[0][t].row(n).head(n_covariates_per_outcome_vec(0, t)).cast<double>() * beta_all_tests_class_var[0].col(t).head(n_covariates_per_outcome_vec(0, t))  ).eval()(0, 0) ;
          } else {
             Xbeta_n = beta_all_tests_class_var[0](0, t);
          }
          
          stan::math::var  Bound_Z =    (  - ( Xbeta_n     +   inc   )  )   / L_Omega_var[0](t, t)  ;
          stan::math::var  Phi_Z  = 0.0;

          if ( (Bound_Z > overflow_threshold) &&  (y_ref(n, t) == 1) )   {
            
                  using namespace stan::math;
                  stan::math::var  log_Bound_U_Phi_Bound_Z_1m = log_inv_logit( - 0.07056 * square(Bound_Z) * Bound_Z  - 1.5976 * Bound_Z );
                  stan::math::var  Bound_U_Phi_Bound_Z_1m = exp(log_Bound_U_Phi_Bound_Z_1m);
                  stan::math::var  Bound_U_Phi_Bound_Z = 1.0 - Bound_U_Phi_Bound_Z_1m;
                  stan::math::var  lse_term_1 =  log_Bound_U_Phi_Bound_Z_1m + stan::math::log(u_array(n, t));
                  stan::math::var  log_Bound_U_Phi_Bound_Z =  log1m(Bound_U_Phi_Bound_Z_1m);
                  stan::math::var  lse_term_2 =  log_Bound_U_Phi_Bound_Z;
                  stan::math::var  log_Phi_Z = log_sum_exp(lse_term_1, lse_term_2);
                  stan::math::var  log_1m_Phi_Z  =   log1m(u_array(n, t))  + log_Bound_U_Phi_Bound_Z_1m;
                  stan::math::var  logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
                  Z_std_norm(t) =  inv_Phi_approx_from_logit_prob_var(logit_Phi_Z);
                  y1(t) =  log_Bound_U_Phi_Bound_Z_1m ;
                
          } else if  ( (Bound_Z < underflow_threshold) &&  (y_ref(n, t) == 0) ) { // y == 0
              
                  using namespace stan::math;
                  stan::math::var  log_Bound_U_Phi_Bound_Z =  log_inv_logit( 0.07056 * square(Bound_Z) * Bound_Z  + 1.5976 * Bound_Z );
                  stan::math::var  Bound_U_Phi_Bound_Z = exp(log_Bound_U_Phi_Bound_Z);
                  stan::math::var  log_Phi_Z = stan::math::log(u_array(n, t)) + log_Bound_U_Phi_Bound_Z;
                  stan::math::var  log_1m_Phi_Z =  log1m(u_array(n, t) * Bound_U_Phi_Bound_Z);
                  stan::math::var  logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
                  Z_std_norm(t) = inv_Phi_approx_from_logit_prob_var(logit_Phi_Z);
                  y1(t)  =  log_Bound_U_Phi_Bound_Z ;
                
          } else {

    
                if  (y_ref(n, t) == 1) {
                  stan::math::var  Bound_U_Phi_Bound_Z = stan::math::Phi( Bound_Z );
                  y1(t) = stan::math::log1m(Bound_U_Phi_Bound_Z);
                  Phi_Z  = Bound_U_Phi_Bound_Z + (1.0 - Bound_U_Phi_Bound_Z) * u_array(n, t);
                  Z_std_norm(t) =   stan::math::inv_Phi(Phi_Z) ;
                } else {
                  stan::math::var  Bound_U_Phi_Bound_Z = stan::math::Phi( Bound_Z );
                  y1(t) = stan::math::log(Bound_U_Phi_Bound_Z);
                  Phi_Z  =  Bound_U_Phi_Bound_Z * u_array(n, t);
                  Z_std_norm(t) =   stan::math::inv_Phi(Phi_Z) ;
                }

         }

          if (t < n_tests - 1)    inc  = (L_Omega_var[0].row(t+1).head(t+1) * Z_std_norm.head(t+1)).eval()(0, 0);

        } // end of t loop

           lp(0) = y1.sum();

      } ///// 

        stan::math::var log_posterior = lp(0);
        target += log_posterior;
        if (out_mat.size() >= 1 + n_params + N) out_mat(1 + n_params + n) = log_posterior.val();

    } // end of n loop

  target +=  log_jac_u;
  double log_prob = target.val();

  //////////////////// calculate gradients
  out_mat(0) = log_prob;
  std::vector<stan::math::var> theta_grad;//(n_params, 0.0);
  
  if (grad_option == "all") {

        for (int i = 0; i < n_params; i++) {
          theta_grad.push_back(theta_var(i));
        }

        std::vector<double> gradient_std_vec(n_params, 0.0);
        Eigen::Matrix<double, -1, 1>  gradient_vec(n_params);
    
        target.grad(theta_grad, gradient_std_vec);
        gradient_vec = std_vec_to_Eigen_vec(gradient_std_vec);

        out_mat.segment(1, n_params) = gradient_vec;

  } else if (grad_option == "us_only") {

        for (int i = 0; i < n_us; i++) {
          theta_grad.push_back(theta_var(i));
        }

        std::vector<double> gradient_std_vec(n_us, 0.0);
        Eigen::Matrix<double, -1, 1>  gradient_vec(n_us);

        target.grad(theta_grad, gradient_std_vec);
        gradient_vec = std_vec_to_Eigen_vec(gradient_std_vec);

        out_mat.segment(1, n_us) = gradient_vec;


  } else if (grad_option == "main_only") {

        for (int i = n_us; i < n_params; i++) {
          theta_grad.push_back(theta_var(i));
        }

        std::vector<double> gradient_std_vec(n_params_main, 0.0);
        Eigen::Matrix<double, -1, 1>  gradient_vec(n_params_main);
    
        target.grad(theta_grad, gradient_std_vec);
        gradient_vec = std_vec_to_Eigen_vec(gradient_std_vec);

        out_mat.segment(1 + n_us, n_params_main) = gradient_vec;


  }

  // Eigen::Matrix<double, -1, 1>  vec_grad_us = out_mat.segment(1, n_us);
  // out_mat.segment(1, n_us) = reorder_output_from_normal_to_chunk(vec_grad_us);
 // // theta_grad.reserve(n_params);
 //  for (int i = 0; i < n_params; i++) {
 //    theta_grad.push_back(theta_var(i));
 //  }
 //  
 //  std::vector<double> gradient_std_vec(n_params, 0.0);
 //  
 //  target.grad(theta_grad, gradient_std_vec);
 //  
 //  Eigen::Matrix<double, -1, 1>  gradient_vec(n_params);
 //  gradient_vec = std_vec_to_Eigen_vec(gradient_std_vec);
 //  
 //  out_mat.segment(1, n_params) = gradient_vec;
 //  
  
  
  //// nested_autodiff_scope releases the tape, including on exceptions.  
 
  //// return(out_mat);
 

}








// Internal function using Eigen::Ref as inputs for matrices
void     fn_lp_and_grad_std_MVP_Pinkney_AD_log_scale_InPlace(   Eigen::Matrix<double, -1, 1> &&out_mat_R_val, 
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
  
  
  fn_lp_and_grad_std_MVP_Pinkney_AD_log_scale_InPlace_process( out_mat_ref,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref,
                                                                      y_ref,
                                                                      grad_option,
                                                                      Model_args_as_cpp_struct,
                                                                      LC_MVP_ws_structs,
                                                                      n_threads_WCP); 
  
  
}  








// Internal function using Eigen::Ref as inputs for matrices
void     fn_lp_and_grad_std_MVP_Pinkney_AD_log_scale_InPlace(   Eigen::Matrix<double, -1, 1> &out_mat_ref, 
                                                                       const Eigen::Matrix<double, -1, 1> &theta_main_vec_ref,
                                                                       const Eigen::Matrix<double, -1, 1> &theta_us_vec_ref,
                                                                       const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                       const std::string &grad_option,
                                                                       const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                       std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                       const int n_threads_WCP
) {
  
  
  fn_lp_and_grad_std_MVP_Pinkney_AD_log_scale_InPlace_process( out_mat_ref,
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
void     fn_lp_and_grad_std_MVP_Pinkney_AD_log_scale_InPlace(   Eigen::Ref<Eigen::Block<MatrixType, -1, 1>>  &out_mat_ref, 
                                                                       const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_main_vec_ref,
                                                                       const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_us_vec_ref,
                                                                       const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                       const std::string &grad_option,
                                                                       const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                       std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                       const int n_threads_WCP
) { 
  
  
  fn_lp_and_grad_std_MVP_Pinkney_AD_log_scale_InPlace_process( out_mat_ref,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref,
                                                                      y_ref,
                                                                      grad_option,
                                                                      Model_args_as_cpp_struct, 
                                                                      LC_MVP_ws_structs,
                                                                      n_threads_WCP); 
  
  
}     







 






// Internal function using Eigen::Ref as inputs for matrices
Eigen::Matrix<double, -1, 1>    fn_lp_and_grad_std_MVP_Pinkney_AD_log_scale(  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
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
  
  fn_lp_and_grad_std_MVP_Pinkney_AD_log_scale_InPlace( out_mat,
                                                              theta_main_vec_ref,
                                                              theta_us_vec_ref,
                                                              y_ref,
                                                              grad_option,
                                                              Model_args_as_cpp_struct, 
                                                              LC_MVP_ws_structs,
                                                              n_threads_WCP); 
  
  return out_mat;
  
}  










 





















