
#pragma once
 
 
 
#include <Eigen/Dense> 
 
 
 
 
#include <stan/math/rev.hpp>


#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/prim/err/check_pos_definite.hpp>
#include <stan/math/prim/err/check_square.hpp>
#include <stan/math/prim/err/check_symmetric.hpp>
 

#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/sqrt.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/transpose.hpp>
#include <stan/math/prim/fun/dot_product.hpp>
#include <stan/math/prim/fun/norm2.hpp>
#include <stan/math/prim/fun/diagonal.hpp>
#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/eigenvalues_sym.hpp>
#include <stan/math/prim/fun/diag_post_multiply.hpp>




#include <stan/math/prim/prob/multi_normal_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/lkj_corr_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/weibull_lpdf.hpp>
#include <stan/math/prim/prob/gamma_lpdf.hpp>
#include <stan/math/prim/prob/beta_lpdf.hpp>


 
#include <Eigen/Dense>
 
 
 
 



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




void                             fn_lp_and_grad_LC_LT_AD_log_scale_InPlace_process(    Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                               const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                               const std::string &grad_option,
                                                                                               const Model_fn_args_struct &Model_args_as_cpp_struct
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
  const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
  const int ub_threshold_phi_approx = Model_args_as_cpp_struct.Model_args_ints(2);
  const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
  
  const double prev_prior_a = Model_args_as_cpp_struct.Model_args_doubles(0);
  const double prev_prior_b = Model_args_as_cpp_struct.Model_args_doubles(1);
  const double overflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(2);
  const double underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(3);
  
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
  
  // const Eigen::Matrix<double, -1, 1>  lkj_cholesky_eta =   Model_args_as_cpp_struct.Model_args_col_vecs_double[0];
  
  const Eigen::Matrix<double, -1, -1> LT_b_priors_shape  = Model_args_as_cpp_struct.Model_args_mats_double[0]; 
  const Eigen::Matrix<double, -1, -1> LT_b_priors_scale  = Model_args_as_cpp_struct.Model_args_mats_double[1]; 
  const Eigen::Matrix<double, -1, -1> LT_known_bs_indicator = Model_args_as_cpp_struct.Model_args_mats_double[2]; 
  const Eigen::Matrix<double, -1, -1> LT_known_bs_values = Model_args_as_cpp_struct.Model_args_mats_double[3]; 
  
  const Eigen::Matrix<int, -1, -1> n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
  
  const std::vector<Eigen::Matrix<double, -1, -1 > >   prior_coeffs_mean  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   prior_coeffs_sd   =  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1]; 
  // const std::vector<Eigen::Matrix<double, -1, -1 > >   prior_for_corr_a   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2]; 
  // const std::vector<Eigen::Matrix<double, -1, -1 > >   prior_for_corr_b   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3]; 
  // const std::vector<Eigen::Matrix<double, -1, -1 > >   lb_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4]; 
  // const std::vector<Eigen::Matrix<double, -1, -1 > >   ub_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5]; 
  // const std::vector<Eigen::Matrix<double, -1, -1 > >   known_values    = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6]; 
  // 
  // const std::vector<Eigen::Matrix<int, -1, -1 >> known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];
   
  //////////////
  //// const int n_corrs =  n_class * n_tests * (n_tests - 1) * 0.5; //// for LC-MVP
  
  int n_covariates_total_nd, n_covariates_total_d, n_covariates_total;
  int n_covariates_max_nd, n_covariates_max_d, n_covariates_max;
  
  n_covariates_total_nd = n_covariates_per_outcome_vec.row(0).sum();
  n_covariates_total_d = n_covariates_per_outcome_vec.row(1).sum();
  n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
  
  n_covariates_max_nd = n_covariates_per_outcome_vec.row(0).maxCoeff();
  n_covariates_max_d = n_covariates_per_outcome_vec.row(1).maxCoeff();
  n_covariates_max = std::max(n_covariates_max_nd, n_covariates_max_d);
  
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
  
  
  
  //////// latent-trait specific variables 
  const int n_bs_LT = n_class * n_tests;
  const int n_coeffs = n_bs_LT; //// latent-trait currently does not support covariates

     
   /////////////////  ------------------------------------------------------------   
   using namespace stan::math;
      
   stan::math::start_nested();
   
   Eigen::Matrix<stan::math::var, -1, 1  >  theta_var(n_params);
   
   {
     Eigen::Matrix<double, -1, 1> theta(n_params);
     theta.head(n_us) = theta_us_vec_ref;
     theta.tail(n_params_main) = theta_main_vec_ref;
     theta_var = stan::math::to_var(theta);
   }
   
   Eigen::Matrix<stan::math::var, -1, 1>    u_unconstrained_vec_var = theta_var.head(n_us);   // u's
  
   //////////////
   // corrs  -  b's (latent_trait only)
   Eigen::Matrix<stan::math::var, -1, 1>  bs_raw_vec_var =  theta_var.segment(n_us, n_bs_LT) ; // stan::math::to_var(bs_raw_vec_double) ;
   Eigen::Matrix<stan::math::var, -1, -1>  bs_mat =      Eigen::Matrix<stan::math::var, -1, -1 >::Zero(n_class, n_tests);
   Eigen::Matrix<stan::math::var, -1, -1> bs_raw_mat =   Eigen::Matrix<stan::math::var, -1, -1 >::Zero(n_class, n_tests);
   
   bs_raw_mat.row(0) =  bs_raw_vec_var.segment(0, n_tests).transpose();
   bs_raw_mat.row(1) =  bs_raw_vec_var.segment(n_tests, n_tests).transpose();
   
   bs_mat.row(0) = stan::math::exp( bs_raw_mat.row(0)) ;
   bs_mat.row(1) = stan::math::exp( bs_raw_mat.row(1)) ;
    
   stan::math::var known_bs_raw_sum = 0.0;
   
   Eigen::Matrix<stan::math::var, -1, 1 > bs_nd  =   bs_mat.row(0).transpose() ; //  bs_constrained_raw_vec_var.head(n_tests);
   Eigen::Matrix<stan::math::var, -1, 1 > bs_d   =   bs_mat.row(1).transpose() ; //  bs_constrained_raw_vec_var.segment(n_tests, n_tests);
   
   //// coeffs
   Eigen::Matrix<stan::math::var, -1, -1  > LT_theta(n_class, n_tests);
   Eigen::Matrix<stan::math::var, -1, -1  > LT_a(n_class, n_tests);
   
   Eigen::Matrix<stan::math::var, -1, 1> coeffs_vec_var = theta_var.segment(n_us + n_bs_LT, n_coeffs);
  
   {
     int i = 0 ; // 0 + n_bs_LT;
     for (int c = 0; c < n_class; ++c) {
       for (int t = 0; t < n_tests; ++t) {
         LT_a(c, t) = coeffs_vec_var(i);
         i = i + 1;
       }
     }
   }
 
   //// LT_theta as TRANSFORMED parameter (need Jacobian adj. if wish to put prior on theta!!!)
   for (int t = 0; t < n_tests; ++t) {
     LT_theta(1, t)   =    LT_a(1, t) /  stan::math::sqrt(1.0 + ( bs_d(t) * bs_d(t)));
     LT_theta(0, t)   =    LT_a(0, t) /  stan::math::sqrt(1.0 + ( bs_nd(t) * bs_nd(t)));
   }


  stan::math::var  u_prev_diseased = theta_var(n_params - 1);
 
  stan::math::var target = 0.0;
  

  /////////////  prev stuff  (only if latent class, otherwise just initialise and leave empty) -------------------------------------------------------------
  Eigen::Matrix<stan::math::var, -1, -1>	 prev_var = Eigen::Matrix<stan::math::var, -1, -1>::Zero(1, 2);
  if (n_class > 1)  { 
    
    std::vector<stan::math::var> 	 u_prev_var_vec_var(2, 0.0);
    std::vector<stan::math::var> 	 prev_var_vec_var(2, 0.0);
    std::vector<stan::math::var> 	 tanh_u_prev_var(2, 0.0);
    
    stan::math::var tanh_pu_deriv_var = 0.0; 
    stan::math::var deriv_p_wrt_pu_var = 0.0;
    stan::math::var tanh_pu_second_deriv_var = 0.0;
    stan::math::var log_jac_p_deriv_wrt_pu_var = 0.0;
    stan::math::var log_jac_p_var = 0.0;
    stan::math::var target_AD_prev = 0.0;
    
    u_prev_var_vec_var[1] =  stan::math::to_var(u_prev_diseased);
    tanh_u_prev_var[1] = ( stan::math::exp(2*u_prev_var_vec_var[1] ) - 1) / ( stan::math::exp(2*u_prev_var_vec_var[1] ) + 1) ; 
    u_prev_var_vec_var[0] =   0.5 *  log( (1.0 + ( (1.0 - 0.5 * ( tanh_u_prev_var[1] + 1))*2.0 - 1.0) ) / (1.0 - ( (1.0 - 0.5 * ( tanh_u_prev_var[1] + 1))*2 - 1) ) )  ;
    tanh_u_prev_var[0] = (stan::math::exp(2*u_prev_var_vec_var[0] ) - 1) / ( stan::math::exp(2*u_prev_var_vec_var[0] ) + 1) ;
    
    prev_var_vec_var[1] =  0.5 * ( tanh_u_prev_var[1] + 1);
    prev_var_vec_var[0] =  0.5 * ( tanh_u_prev_var[0] + 1); 
    prev_var(0, 1) =  prev_var_vec_var[1];
    prev_var(0, 0) =  prev_var_vec_var[0];
    
    tanh_pu_deriv_var = ( 1.0 - tanh_u_prev_var[1] * tanh_u_prev_var[1]  );
    deriv_p_wrt_pu_var = 0.5 *  tanh_pu_deriv_var;
    tanh_pu_second_deriv_var  = -2.0 * tanh_u_prev_var[1]  * tanh_pu_deriv_var;
    log_jac_p_deriv_wrt_pu_var  = ( 1.0 / deriv_p_wrt_pu_var) * 0.5 * tanh_pu_second_deriv_var; // for gradient of u's 
    log_jac_p_var =    stan::math::log( deriv_p_wrt_pu_var );
    
    target += beta_lpdf( prev_var(0, 1), prev_prior_a, prev_prior_b  ); // weakly informative prior - helps avoid boundaries with slight negative skew (for lower N)
    target += log_jac_p_var;
    
  }
  

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
    log_jac_u   +=   - log(2.0) ;   // correct
    log_jac_u   +=    sum(log(u_vec));  // correct
    log_jac_u   +=    sum(log1m(u_vec));  // correct
  }

  ///////////////// get cholesky factor's (lower-triangular) of corr matrices
  std::vector<Eigen::Matrix<stan::math::var, -1, -1>>  L_Omega_var = vec_of_mats<stan::math::var>(n_tests, n_tests, n_class);
  std::vector<Eigen::Matrix<stan::math::var, -1, -1>>  Omega_var   = vec_of_mats<stan::math::var>(n_tests, n_tests, n_class);
  Eigen::Matrix<stan::math::var, -1, -1 > identity_dim_T =     Eigen::Matrix<stan::math::var, -1, -1 > ::Zero(n_tests, n_tests) ; //  stan::math::diag_matrix(  stan::math::rep_vector(1, n_tests)  ) ;

  for (int i = 0; i < n_tests; ++i) {
    identity_dim_T(i, i) = 1.0;
  }

  Omega_var[0] = identity_dim_T +  bs_nd * bs_nd.transpose();
  Omega_var[1] = identity_dim_T +  bs_d * bs_d.transpose();

  for (int c = 0; c < n_class; ++c) {
    L_Omega_var[c]   = stan::math::cholesky_decompose(Omega_var[c]) ;
  }

  ///////////////////////////////////////////////////////////////////////// prior densities
  /// priors and Jacobians for coeffs
  for (int c = 0; c < n_class; ++c) {
    for (int t = 0; t < n_tests; ++t) {
      target += stan::math::normal_lpdf(LT_theta(c, t), prior_coeffs_mean[c](0, t), prior_coeffs_sd[c](0, t)); //// Prior on theta
      target +=  - 0.5 * stan::math::log(1.0 + stan::math::square(stan::math::abs(bs_mat(c, t) ))); // Jacobian for LT_theta -> LT_a
    }
  }

  ////////////////////// Weibull priors for  b's / corr
  for (int t = 0; t < n_tests; ++t) {
    target += stan::math::weibull_lpdf(  bs_nd(t) ,   LT_b_priors_shape(0, t), LT_b_priors_scale(0, t)  );
    target += stan::math::weibull_lpdf(  bs_d(t)  ,   LT_b_priors_shape(1, t), LT_b_priors_scale(1, t)  );
  }

  target +=  (bs_raw_mat).sum()  - known_bs_raw_sum ; // Jacobian b -> raw_b


  /////////////////////////////////////////////////////////////////////////////////////////////////////////// likelihood
  Eigen::Matrix<stan::math::var, -1, 1>	   y1(n_tests);
  Eigen::Matrix<stan::math::var, -1, 1>	   lp(2);
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
  

  Eigen::Matrix<stan::math::var, -1, -1 >  log_prev  =  Eigen::Matrix<stan::math::var, -1, -1>::Zero(1, 2);
  for (int c = 0; c < 2; c++) {
    log_prev(0, c) =  stan::math::log(prev_var(0, c));
  }

  int counter = 0;

  if (Phi_type == "Phi") {

    for (int n = 0; n < N; n++ ) {

      for (int c = 0; c < 2; ++c) {

        stan::math::var inc  = 0.0;

        for (int t = 0; t < n_tests; t++) {

          Xbeta_n = LT_a(c, t); /// currently intercept-only !!
          stan::math::var  Bound_Z =    (  - ( Xbeta_n     +   inc   )  )   / L_Omega_var[c](t, t)  ;
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

          if (t < n_tests - 1)    inc  = (L_Omega_var[c].row(t+1).head(t+1) * Z_std_norm.head(t+1)).eval()(0, 0);


          counter += 1;

        } // end of t loop

        lp(c) = log_prev(0, c) + y1.sum();

      }

      stan::math::var log_posterior = stan::math::log_sum_exp(lp);
      target += log_posterior;

    } // end of n loop

  } else if ( (Phi_type == "Phi_approx") || (Phi_type == "Phi_approx_2") ) {

    for (int n = 0; n < N; n++ ) {

      for (int c = 0; c < 2; ++c) {

        stan::math::var inc  = 0.0;

        for (int t = 0; t < n_tests; t++) {

          // if (n_covariates_max > 1)  Xbeta_n = ( X[c][t].row(n).head(n_covariates_per_outcome_vec[c](t)).cast<double>() * beta_all_tests_class_var[c].col(t).head(n_covariates_per_outcome_vec[c](t))  ).eval()(0, 0) ;
          // else   Xbeta_n = beta_all_tests_class_var[c](0, t);
          Xbeta_n = LT_a(c, t); /// currently intercept-only !!
          stan::math::var  Bound_Z =    (  - ( Xbeta_n     +   inc   )  )   / L_Omega_var[c](t, t)  ;
          stan::math::var  Phi_Z  = 0.0;

          if ( (Bound_Z > overflow_threshold) &&  (y_ref(n, t) == 1) )   {

                using namespace stan::math;
                stan::math::var  log_Bound_U_Phi_Bound_Z_1m = log_inv_logit( - 0.07056 * square(Bound_Z) * Bound_Z  - 1.5976 * Bound_Z );
                stan::math::var  Bound_U_Phi_Bound_Z_1m = exp(log_Bound_U_Phi_Bound_Z_1m);
                stan::math::var  Bound_U_Phi_Bound_Z = 1.0 - Bound_U_Phi_Bound_Z_1m;
                stan::math::var  lse_term_1 =  log_Bound_U_Phi_Bound_Z_1m + log(u_array(n, t));
                stan::math::var  log_Bound_U_Phi_Bound_Z =  log1m(Bound_U_Phi_Bound_Z_1m);
                stan::math::var  lse_term_2 =  log_Bound_U_Phi_Bound_Z;
                stan::math::var  log_Phi_Z = log_sum_exp(lse_term_1, lse_term_2);
                stan::math::var  log_1m_Phi_Z  =   log1m(u_array(n, t))  + log_Bound_U_Phi_Bound_Z_1m;
                stan::math::var  logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
                Z_std_norm(t) =  inv_Phi_approx_from_logit_prob_var(logit_Phi_Z);
                y1(t) =  log_Bound_U_Phi_Bound_Z_1m ;

          } else if  ( (Bound_Z < underflow_threshold) &&  (y_ref(n, t) == 0) ) { // y_ref == 0

                using namespace stan::math;
                stan::math::var  log_Bound_U_Phi_Bound_Z =  log_inv_logit( 0.07056 * square(Bound_Z) * Bound_Z  + 1.5976 * Bound_Z );
                stan::math::var  Bound_U_Phi_Bound_Z = exp(log_Bound_U_Phi_Bound_Z);
                stan::math::var  log_Phi_Z = log(u_array(n, t)) + log_Bound_U_Phi_Bound_Z;
                stan::math::var  log_1m_Phi_Z =  log1m(u_array(n, t) * Bound_U_Phi_Bound_Z);
                stan::math::var  logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
                Z_std_norm(t) = inv_Phi_approx_from_logit_prob_var(logit_Phi_Z);
                y1(t)  =  log_Bound_U_Phi_Bound_Z ;

          } else {

                if  (y_ref(n, t) == 1) {
                  stan::math::var  Bound_U_Phi_Bound_Z = stan::math::Phi_approx( Bound_Z );
                  y1(t) = stan::math::log1m(Bound_U_Phi_Bound_Z);
                  Phi_Z  = Bound_U_Phi_Bound_Z + (1.0 - Bound_U_Phi_Bound_Z) * u_array(n, t);
                  Z_std_norm(t) =   inv_Phi_approx_var(Phi_Z) ;
                } else {
                  stan::math::var  Bound_U_Phi_Bound_Z = stan::math::Phi_approx( Bound_Z );
                  y1(t) = stan::math::log(Bound_U_Phi_Bound_Z);
                  Phi_Z  =  Bound_U_Phi_Bound_Z * u_array(n, t);
                  Z_std_norm(t) =   inv_Phi_approx_var(Phi_Z) ;
                }

          }

          if (t < n_tests - 1)    inc  = (L_Omega_var[c].row(t+1).head(t+1) * Z_std_norm.head(t+1)).eval()(0, 0);


          counter += 1;

        } // end of t loop

        lp(c) = log_prev(0, c) + y1.sum();

      }

      stan::math::var log_posterior = stan::math::log_sum_exp(lp);
      target += log_posterior;
    } // end of n loop

  }

  target +=  log_jac_u;
  double log_prob = target.val();

  //////////////////// calculate gradients
  out_mat(0) = log_prob;
  std::vector<stan::math::var> theta_grad;//(n_params, 0.0);

  if (grad_option == "all") {

        //theta_grad.reserve(n_params);
        for (int i = 0; i < n_params; i++) {
          theta_grad.push_back(theta_var(i));
        }

         std::vector<double> gradient_std_vec(n_params, 0.0);

         target.grad(theta_grad, gradient_std_vec);

        Eigen::Matrix<double, -1, 1>  gradient_vec(n_params);
        gradient_vec = std_vec_to_Eigen_vec(gradient_std_vec);

        out_mat.segment(1, n_params) = gradient_vec;

  } else if (grad_option == "us_only") {

        //theta_grad.reserve(n_us);
        for (int i = 0; i < n_us; i++) {
          theta_grad.push_back(theta_var(i));
        }

        std::vector<double> gradient_std_vec(n_us, 0.0);

        target.grad(theta_grad, gradient_std_vec);

        Eigen::Matrix<double, -1, 1>  gradient_vec(n_us);
        gradient_vec = std_vec_to_Eigen_vec(gradient_std_vec);

        out_mat.segment(1, n_us) = gradient_vec;


  } else if (grad_option == "main_only") {

        //theta_grad.reserve(n_params_main);
        for (int i = n_us; i < n_params; i++) {
          theta_grad.push_back(theta_var(i));
        }

        std::vector<double> gradient_std_vec(n_params_main, 0.0);

        target.grad(theta_grad, gradient_std_vec);


        Eigen::Matrix<double, -1, 1>  gradient_vec(n_params_main);

        gradient_vec = std_vec_to_Eigen_vec(gradient_std_vec);

        out_mat.segment(1 + n_us, n_params_main) = gradient_vec;


  }
  
  
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

 
  
  stan::math::recover_memory_nested();  
 
  //// return(out_mat);
 

}








// Internal function using Eigen::Ref as inputs for matrices
void     fn_lp_and_grad_LC_LT_AD_log_scale_InPlace(   Eigen::Matrix<double, -1, 1> &&out_mat_R_val, 
                                                                       const Eigen::Matrix<double, -1, 1> &&theta_main_vec_R_val,
                                                                       const Eigen::Matrix<double, -1, 1> &&theta_us_vec_R_val,
                                                                       const Eigen::Matrix<int, -1, -1> &&y_R_val,
                                                                       const std::string &grad_option,
                                                                       const Model_fn_args_struct &Model_args_as_cpp_struct
                                                                         
                                                                         
                                                                         
                                                                         
) {
  
  
  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat_ref(out_mat_R_val);
  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref(theta_main_vec_R_val);  // create Eigen::Ref from R-value
  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref(theta_us_vec_R_val);  // create Eigen::Ref from R-value
  const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref(y_R_val);  // create Eigen::Ref from R-value
  
  
  fn_lp_and_grad_LC_LT_AD_log_scale_InPlace_process( out_mat_ref,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref,
                                                                      y_ref,
                                                                      grad_option,
                                                                      Model_args_as_cpp_struct); 
  
  
}  








// Internal function using Eigen::Ref as inputs for matrices
void     fn_lp_and_grad_LC_LT_AD_log_scale_InPlace(   Eigen::Matrix<double, -1, 1> &out_mat_ref, 
                                                                       const Eigen::Matrix<double, -1, 1> &theta_main_vec_ref,
                                                                       const Eigen::Matrix<double, -1, 1> &theta_us_vec_ref,
                                                                       const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                       const std::string &grad_option,
                                                                       const Model_fn_args_struct &Model_args_as_cpp_struct
                                                                         
                                                                         
                                                                         
                                                                         
) {
  
  
  fn_lp_and_grad_LC_LT_AD_log_scale_InPlace_process( out_mat_ref,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref,
                                                                      y_ref,
                                                                      grad_option,
                                                                      Model_args_as_cpp_struct); 
  
  
}    





// Internal function using Eigen::Ref as inputs for matrices
template <typename MatrixType>
void     fn_lp_and_grad_LC_LT_AD_log_scale_InPlace(   Eigen::Ref<Eigen::Block<MatrixType, -1, 1>>  &out_mat_ref, 
                                                                       const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_main_vec_ref,
                                                                       const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_us_vec_ref,
                                                                       const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                       const std::string &grad_option,
                                                                       const Model_fn_args_struct &Model_args_as_cpp_struct
                                                                         
                                                                         
                                                                         
                                                                         
) { 
  
  
  fn_lp_and_grad_LC_LT_AD_log_scale_InPlace_process( out_mat_ref,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref,
                                                                      y_ref,
                                                                      grad_option,
                                                                      Model_args_as_cpp_struct); 
  
  
}     







 






// Internal function using Eigen::Ref as inputs for matrices
Eigen::Matrix<double, -1, 1>    fn_lp_and_grad_LC_LT_AD_log_scale(  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                     const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                     const std::string &grad_option,
                                                                                     const Model_fn_args_struct &Model_args_as_cpp_struct
                                                                                       
                                                                                       
                                                                                       
                                                                                        
) {
  
  int n_params_main = theta_main_vec_ref.rows();
  int n_us = theta_us_vec_ref.rows();
  int n_params = n_us + n_params_main;
  int N = y_ref.rows();
  
  Eigen::Matrix<double, -1, 1> out_mat = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
  
  fn_lp_and_grad_LC_LT_AD_log_scale_InPlace(          out_mat,
                                                              theta_main_vec_ref,
                                                              theta_us_vec_ref,
                                                              y_ref,
                                                              grad_option,
                                                              Model_args_as_cpp_struct); 
  
  return out_mat;
  
}  










 






















