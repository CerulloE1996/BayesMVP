

#pragma once

 
 
#include <Eigen/Dense>
 
 

 


 
inline  void         fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace_process(    Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat ,
                                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                               const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                               const std::string &grad_option,
                                                                               const Model_fn_args_struct &Model_args_as_cpp_struct
) { 
  
  out_mat.setZero(); //// set log_prob and grad vec to zero at the start (only do this on outer fns, not inner/likelihood fns)
  
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
  const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
  
  const int n_cores = Model_args_as_cpp_struct.Model_args_ints(0);
  const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
  const int ub_threshold_phi_approx = Model_args_as_cpp_struct.Model_args_ints(2);
  const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
  
  const double prev_prior_a = Model_args_as_cpp_struct.Model_args_doubles(0);
  const double prev_prior_b = Model_args_as_cpp_struct.Model_args_doubles(1);
  const double overflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(2);
  const double underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(3);
  
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
  // const std::string grad_option =  Model_args_as_cpp_struct.Model_args_strings(11);
  const std::string &nuisance_transformation =   Model_args_as_cpp_struct.Model_args_strings(12);
  
  const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
  
  const Eigen::Matrix<double, -1, -1> &LT_b_priors_shape  = Model_args_as_cpp_struct.Model_args_mats_double[0]; 
  const Eigen::Matrix<double, -1, -1> &LT_b_priors_scale  = Model_args_as_cpp_struct.Model_args_mats_double[1]; 
  const Eigen::Matrix<double, -1, -1> &LT_known_bs_indicator = Model_args_as_cpp_struct.Model_args_mats_double[2]; 
  const Eigen::Matrix<double, -1, -1> &LT_known_bs_values = Model_args_as_cpp_struct.Model_args_mats_double[3]; 
  
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &prior_coeffs_mean  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &prior_coeffs_sd   =  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1]; 
  
  //////////////
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
  
  const double sqrt_2_pi_recip = 1.0 / sqrt(2.0 * M_PI);
  const double sqrt_2_recip = 1.0 / stan::math::sqrt(2.0);
  const double minus_sqrt_2_recip = -sqrt_2_recip;
  const double a = 0.07056;
  const double b = 1.5976;
  const double a_times_3 = 3.0 * 0.07056;
  const double s = 1.0 / 1.702;
  const double Inf = std::numeric_limits<double>::infinity();
  
  //// ---- determine chunk size -------------------------- 
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

  //////////////  -----------------------------------------------------------------------------------------------------------------------------------------------------------
  //////// latent-trait specific variables 
  const int n_bs_LT = n_class * n_tests;
  const int n_coeffs = n_bs_LT; //// latent-trait currently does not support covariates
  
  //// prev
  double u_prev_diseased = theta_main_vec_ref(n_params_main - 1);
   
  int n_choose_2 = n_tests * (n_tests - 1.0) * 0.5 ;
  std::vector< std::vector<Eigen::Matrix<double, -1, -1 > > > Jacobian_d_L_Sigma_wrt_b_3d_arrays_double =  vec_of_vec_of_mats<double>(n_tests, n_tests, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_double =  vec_of_mats<double>(n_tests, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_recip_double =  vec_of_mats<double>(n_tests, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1 > > log_abs_L_Omega_double =  vec_of_mats<double>(n_tests, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1 > > sign_L_Omega_double =  vec_of_mats<double>(n_tests, n_tests, n_class);
  Eigen::Matrix<double, -1, 1 >   bs_d_double = Eigen::Matrix<double, -1, 1 >::Zero(n_tests);
  Eigen::Matrix<double, -1, 1 >   bs_nd_double  = Eigen::Matrix<double, -1, 1 >::Zero(n_tests);
  
  std::vector< std::vector<Eigen::Matrix<double, -1, -1 > > > log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double =  vec_of_vec_of_mats<double>(n_tests, n_tests, n_tests, n_class);
  std::vector< std::vector<Eigen::Matrix<double, -1, -1 > > > sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double =  vec_of_vec_of_mats<double>(n_tests, n_tests, n_tests, n_class);

  double grad_prev_AD = 0.0;
  double log_jac_p_double = 0.0; 
  double prior_densities = 0.0; 
  Eigen::Matrix<double, -1, 1  >  bs_raw_vec_double = theta_main_vec_ref.segment(0, n_bs_LT) ;
  Eigen::Matrix<double, -1, -1 >  bs_mat_double =    Eigen::Matrix<double, -1, -1 >::Zero(n_class, n_tests);
  Eigen::Matrix<double, -1, 1  >  coeffs_vec_double  = Eigen::Matrix<double, -1, 1>::Zero(n_coeffs);
  
  ////  std::vector<Eigen::Matrix<double, -1, -1 > > LT_a_double = vec_of_mats_double(n_covariates_max, n_tests,  n_class); // currently intercept-only
  Eigen::Matrix<double, -1, -1 >  LT_a_double =    Eigen::Matrix<double, -1, -1 >::Zero(n_class, n_tests); // currently intercept-only
  
  {  ///////////   -------------------  start of AD block  -----------------------

        stan::math::start_nested();  //////////  ----------
    
        //////////////
        // corrs / b's
        Eigen::Matrix<stan::math::var, -1, 1  >  bs_raw_vec_var =  stan::math::to_var(bs_raw_vec_double) ;
        Eigen::Matrix<stan::math::var, -1, -1 >  bs_mat =      Eigen::Matrix<stan::math::var, -1, -1 >::Zero(n_class, n_tests);
        Eigen::Matrix<stan::math::var, -1, -1 >  bs_raw_mat =  Eigen::Matrix<stan::math::var, -1, -1 >::Zero(n_class, n_tests);

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


        Eigen::Matrix<stan::math::var, -1, 1  > coeffs_vec_var(n_coeffs);

        coeffs_vec_double = theta_main_vec_ref.segment(0 + n_bs_LT, n_coeffs);
        coeffs_vec_var = stan::math::to_var(coeffs_vec_double);

        {
          int i = 0 ; // 0 + n_bs_LT;
          for (int c = 0; c < n_class; ++c) {
            for (int t = 0; t < n_tests; ++t) {
              LT_a(c, t) = coeffs_vec_var(i);
              LT_a_double(c, t) = LT_a(c, t).val();
              bs_mat_double(c, t) = bs_mat(c, t).val();
              i = i + 1;
            }
          }
        }

        //// LT_theta as TRANSFORMED parameter (need Jacobian adj. if wish to put prior on theta!!!)
        for (int t = 0; t < n_tests; ++t) {
          LT_theta(1, t)   =    LT_a(1, t) /  stan::math::sqrt(1.0 + ( bs_d(t) * bs_d(t)));
          LT_theta(0, t)   =    LT_a(0, t) /  stan::math::sqrt(1.0 + ( bs_nd(t) * bs_nd(t)));
        }

        std::vector<Eigen::Matrix<stan::math::var, -1, -1>>  L_Omega_var = vec_of_mats<stan::math::var>(n_tests, n_tests, n_class);
        std::vector<Eigen::Matrix<stan::math::var, -1, -1>>  Omega_var   = vec_of_mats<stan::math::var>(n_tests, n_tests, n_class);
        Eigen::Matrix<stan::math::var, -1, -1 > identity_dim_T =     Eigen::Matrix<stan::math::var, -1, -1 > ::Zero(n_tests, n_tests) ; //  stan::math::diag_matrix(  stan::math::rep_vector(1, n_tests)  ) ;

        for (int i = 0; i < n_tests; ++i) {
          identity_dim_T(i, i) = 1.0;
          bs_d_double(i) = bs_d(i).val() ;
          bs_nd_double(i) = bs_nd(i).val() ;
        }

        Omega_var[0] = identity_dim_T +  bs_nd * bs_nd.transpose();
        Omega_var[1] = identity_dim_T +  bs_d * bs_d.transpose();

        stan::math::var target_AD = 0.0;

        for (int c = 0; c < n_class; ++c) {
          L_Omega_var[c]   = stan::math::cholesky_decompose(Omega_var[c]) ;
        }

        //////////////// Jacobian L_Sigma -> b's
        std::vector< std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > > Jacobian_d_L_Sigma_wrt_b_3d_arrays_var = vec_of_vec_of_mats<stan::math::var>(n_tests, n_tests, n_tests, n_class);

        for (int c = 0; c < n_class; ++c) {

          //  # -----------  wrt last b first
          int t = n_tests;
          stan::math::var sum_sq_1 = 0.0;
          for (int j = 1; j < t; ++j) {
            Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, j-1) = ( L_Omega_var[c](n_tests-1, j-1) / bs_mat(c, n_tests-1) ) ;//* bs_nd(n_tests-1) ;
            sum_sq_1 +=   bs_mat(c, j-1) * bs_mat(c, j-1) ;
          }
          stan::math::var big_denom_p1 =  1 + sum_sq_1;
          Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, n_tests-1) =   (1 / L_Omega_var[c](n_tests-1, n_tests-1) ) * ( bs_mat(c, n_tests-1) / big_denom_p1 ) ;//* bs_nd(n_tests-1) ;

          //  # -----------  wrt 2nd-to-last b
          t = n_tests - 1;
          sum_sq_1 = 0;
          stan::math::var  sum_sq_2 = 0.0;
          for (int j = 1; j < t + 1; ++j) {
            Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, j-1) = ( L_Omega_var[c](t-1, j-1) / bs_mat(c, t-1) );// * bs_nd(t-1) ;
            sum_sq_1 +=   bs_mat(c, j-1) * bs_mat(c, j-1) ;
            if (j < (t))   sum_sq_2 +=  bs_mat(c, j-1) * bs_mat(c, j-1) ;
          }
          big_denom_p1 =  1 + sum_sq_1;
          stan::math::var big_denom_p2 =  1 + sum_sq_2;
          stan::math::var  big_denom_part =  big_denom_p1 * big_denom_p2;
          Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, t-1) =   (1 / L_Omega_var[c](t-1, t-1)) * ( bs_mat(c, t-1) / big_denom_p2 );// * bs_nd(t-1) ;

          for (int j = t+1; j < n_tests + 1; ++j) {
            Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](j-1, t-1) =   ( 1/L_Omega_var[c](j-1, t-1) ) * (bs_mat(c, j-1) *  bs_mat(c, j-1)  ) * (   bs_mat(c, t-1)  / big_denom_part) * (1 - ( bs_mat(c, t-1) * bs_mat(c, t-1)  / big_denom_p1 ) );// * bs_nd(t-1)   ;
          }

          Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t, t)   =  - ( 1/L_Omega_var[c](t, t) ) * (bs_mat(c, t) * bs_mat(c, t)) * ( bs_mat(c, t-1)  / (big_denom_p1*big_denom_p1));//*  bs_nd(t-1) ;

          // # -----------  wrt rest of b's
          for (int t = 1; t < (n_tests - 2) + 1; ++t) {

            sum_sq_1  = 0;
            sum_sq_2  = 0;

            for (int j = 1; j < t + 1; ++j) {
              if (j < (t)) Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, j-1) = ( L_Omega_var[c](t-1, j-1) /  bs_mat(c, t-1) ) ;//* ;// bs_nd(t-1) ;
              sum_sq_1 +=   bs_mat(c, j-1) *   bs_mat(c, j-1) ;
              if (j < (t))   sum_sq_2 +=    bs_mat(c, j-1) *   bs_mat(c, j-1) ;
            }
            big_denom_p1 = 1 + sum_sq_1;
            big_denom_p2 = 1 + sum_sq_2;
            big_denom_part =  big_denom_p1 * big_denom_p2;

            Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](t-1, t-1) =   (1 / L_Omega_var[c](t-1, t-1) ) * (  bs_mat(c, t-1) / big_denom_p2 ) ;//*  bs_nd(t-1) ;

            for (int j = t + 1; j < n_tests + 1; ++j) {
              Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](j-1, t-1)  =   (1/L_Omega_var[c](j-1, t-1)) * (  bs_mat(c, j-1) *   bs_mat(c, j-1) ) * (   bs_mat(c, t-1) / big_denom_part) * (1 - ( ( bs_mat(c, t-1) *  bs_mat(c, t-1) ) / big_denom_p1 ) ) ;//*  bs_nd(t-1) ;
            }

            for (int j = t + 1; j < n_tests ; ++j) {
              Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](j-1, j-1) =  - (1/L_Omega_var[c](j-1, j-1)) * (  bs_mat(c, j-1) *   bs_mat(c, j-1) ) * ( bs_mat(c, t-1) / (big_denom_p1*big_denom_p1)) ;//*  bs_nd(t-1) ;
              big_denom_p1 = big_denom_p1 +   bs_mat(c, j-1) *   bs_mat(c, j-1) ;
              big_denom_p2 = big_denom_p2 + bs_mat(c, j-2) * bs_mat(c, j-2) ;
              big_denom_part =  big_denom_p1 * big_denom_p2 ;
              if (t < n_tests - 1) {
                for (int k = j + 1; k < n_tests + 1; ++k) {
                  Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](k-1, j-1) =   (-1 / L_Omega_var[c](k-1, j-1)) * (  bs_mat(c, k-1) *   bs_mat(c, k-1) ) * (  bs_mat(c, j-1) *   bs_mat(c, j-1) ) * (  bs_mat(c, t-1) / big_denom_part ) * ( ( 1 / big_denom_p2 )  +  ( 1 / big_denom_p1 ) ) ;//*  bs_nd(t-1) ;
                }
              }
            }

            Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t-1](n_tests-1, n_tests-1) =  - (1/L_Omega_var[c](n_tests-1, n_tests-1)) * (bs_mat(c, n_tests-1) * bs_mat(c, n_tests-1)) * ( bs_mat(c, t-1) / (big_denom_p1*big_denom_p1)) ;//*  bs_nd(t-1) ;

          }

          for (int t1 = 0; t1 < n_tests; ++t1) {
            for (int t3 = 0; t3 < n_tests; ++t3) { //// col-major storage
              for (int t2 = 0; t2 < n_tests; ++t2) {
                Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t2, t3)    =      Jacobian_d_L_Sigma_wrt_b_3d_arrays_var[c][t1](t2, t3).val();
                log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t2, t3) = stan::math::log(stan::math::abs(Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t2, t3)));
                sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t2, t3) = stan::math::sign(Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c][t1](t2, t3));
              }
            }
          }
        }

   
        ////////////////////// Weibull priors for  b's / corr
        for (int t = 0; t < n_tests; ++t) {
          target_AD += stan::math::weibull_lpdf(  bs_nd(t) ,   LT_b_priors_shape(0, t), LT_b_priors_scale(0, t)  );
          target_AD += stan::math::weibull_lpdf(  bs_d(t)  ,   LT_b_priors_shape(1, t), LT_b_priors_scale(1, t)  );
        }

        target_AD +=  (bs_raw_mat).sum()  - known_bs_raw_sum ; // Jacobian b -> raw_b

        /// priors and Jacobians for coeffs
        for (int c = 0; c < n_class; ++c) {
          for (int t = 0; t < n_tests; ++t) {
            target_AD += stan::math::normal_lpdf(LT_theta(c, t), prior_coeffs_mean[c](0, t), prior_coeffs_sd[c](0, t)); //// Prior on theta
            target_AD +=  - 0.5 * stan::math::log(1.0 + stan::math::square(stan::math::abs(bs_mat(c, t) ))); // Jacobian for LT_theta -> LT_a
          }
        }


        /////////////  prev stuff  ---- vars
        std::vector<stan::math::var> 	 u_prev_var_vec_var(n_class, 0.0);
        std::vector<stan::math::var> 	 prev_var_vec_var(n_class, 0.0);
        std::vector<stan::math::var> 	 tanh_u_prev_var(n_class, 0.0);
        Eigen::Matrix<stan::math::var, -1, -1>	 prev_var(1, n_class);

        u_prev_var_vec_var[1] =  stan::math::to_var(u_prev_diseased);
        tanh_u_prev_var[1] = ( exp(2*u_prev_var_vec_var[1] ) - 1) / ( exp(2*u_prev_var_vec_var[1] ) + 1) ;
        u_prev_var_vec_var[0] =   0.5 *  log( (1 + ( (1 - 0.5 * ( tanh_u_prev_var[1] + 1))*2 - 1) ) / (1 - ( (1 - 0.5 * ( tanh_u_prev_var[1] + 1))*2 - 1) ) )  ;
        tanh_u_prev_var[0] = (exp(2*u_prev_var_vec_var[0] ) - 1) / ( exp(2*u_prev_var_vec_var[0] ) + 1) ;

        prev_var_vec_var[1] = 0.5 * ( tanh_u_prev_var[1] + 1);
        prev_var_vec_var[0] =  0.5 * ( tanh_u_prev_var[0] + 1);
        prev_var(0,1) =  prev_var_vec_var[1];
        prev_var(0,0) =  prev_var_vec_var[0];

        stan::math::var tanh_pu_deriv_var = ( 1 - tanh_u_prev_var[1] * tanh_u_prev_var[1]  );
        stan::math::var deriv_p_wrt_pu_var = 0.5 *  tanh_pu_deriv_var;
        stan::math::var tanh_pu_second_deriv_var  = -2 * tanh_u_prev_var[1]  * tanh_pu_deriv_var;
        stan::math::var log_jac_p_deriv_wrt_pu_var  = ( 1 / deriv_p_wrt_pu_var) * 0.5 * tanh_pu_second_deriv_var; // for gradient of u's
        stan::math::var  log_jac_p_var =    stan::math::log( deriv_p_wrt_pu_var );

        target_AD += beta_lpdf(  prev_var(0, 1), prev_prior_a, prev_prior_b  ); // weakly informative prior - helps avoid boundaries with slight negative skew (for lower N)
        target_AD += log_jac_p_var;

        log_jac_p_double = log_jac_p_var.val();
        prior_densities = target_AD.val() ; // target_AD_coeffs.val() + target_AD_corrs.val();

        //  ///////////////////////
        // stan::math::set_zero_all_adjoints();
        target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
        out_mat.segment(1 + n_us, n_bs_LT) = bs_raw_vec_var.adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
        stan::math::set_zero_all_adjoints();
        //////////////////////////////////////////////////////////// end of AD part

        //  ///////////////////////
        stan::math::set_zero_all_adjoints();
        target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
        out_mat.segment(1 + n_us + n_bs_LT, n_coeffs)  = coeffs_vec_var.adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
        stan::math::set_zero_all_adjoints();
        //////////////////////////////////////////////////////////// end of AD part

        ///////////////////////
        stan::math::set_zero_all_adjoints();
        target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
        grad_prev_AD  =  u_prev_var_vec_var[1].adj() - u_prev_var_vec_var[0].adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
        stan::math::set_zero_all_adjoints();
        //////////////////////////////////////////////////////////// end of AD part


        for (int c = 0; c < n_class; ++c) {
          for (int t1 = 0; t1 < n_tests; ++t1) {
            for (int t2 = 0; t2 < n_tests; ++t2) {
              L_Omega_double[c](t1, t2) =   L_Omega_var[c](t1, t2).val();
              L_Omega_recip_double[c](t1, t2) =   1.0 / L_Omega_double[c](t1, t2) ;
              log_abs_L_Omega_double[c](t1, t2) = stan::math::log(stan::math::abs( L_Omega_double[c](t1, t2) ));
              sign_L_Omega_double[c](t1, t2) = stan::math::sign(L_Omega_double[c](t1, t2));
            }
          }
        }


        stan::math::recover_memory_nested(); //////////

  }  ///////////   -------------------  end of AD block   ------------------------------------

  
  /////////////  prev_double stuff
  std::vector<double> 	 u_prev_var_vec(n_class, 0.0);
  std::vector<double> 	 prev_var_vec(n_class, 0.0);
  std::vector<double> 	 tanh_u_prev(n_class, 0.0);
  Eigen::Matrix<double, -1, -1>	 prev_double(1, n_class);

  u_prev_var_vec[1] =  (double) u_prev_diseased ;
  tanh_u_prev[1] = ( exp(2.0*u_prev_var_vec[1] ) - 1.0) / ( exp(2.0*u_prev_var_vec[1] ) + 1.0) ;
  u_prev_var_vec[0] =   0.5 *  log( (1 + ( (1 - 0.5 * ( tanh_u_prev[1] + 1))*2.0 - 1.0) ) / (1.0 - ( (1.0 - 0.5 * ( tanh_u_prev[1] + 1.0))*2.0 - 1.0) ) )  ;
  tanh_u_prev[0] = (exp(2.0*u_prev_var_vec[0] ) - 1.0) / ( exp(2.0*u_prev_var_vec[0] ) + 1.0) ;

  prev_var_vec[1] =  0.5 * ( tanh_u_prev[1] + 1.0);
  prev_var_vec[0] =  0.5 * ( tanh_u_prev[0] + 1.0);
  prev_double(0,1) =  prev_var_vec[1];
  prev_double(0,0) =  prev_var_vec[0];

  double tanh_pu_deriv = ( 1.0 - tanh_u_prev[1] * tanh_u_prev[1]  );
  double deriv_p_wrt_pu_double = 0.5 *  tanh_pu_deriv;
  double tanh_pu_second_deriv  = -2.0 * tanh_u_prev[1]  * tanh_pu_deriv;
  double log_jac_p_deriv_wrt_pu  = ( 1.0 / deriv_p_wrt_pu_double) * 0.5 * tanh_pu_second_deriv; // for gradient of u's

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////// likelihood
  double log_prob_out = 0.0;
  
  Eigen::Matrix<double, -1, -1 >  log_prev = stan::math::log(prev_double);
  
  //// define unconstrained nuisance parameter vec 
  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec = theta_us_vec_ref; 
  
  ///////////////////////////////////////////////
  Eigen::Matrix<double , -1, 1>   beta_grad_vec   =  Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);  //
  std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array =  vec_of_mats<double>(n_covariates_max, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array =  vec_of_mats<double>(n_tests, n_tests, n_class);
  Eigen::Matrix<double, -1, 1>  prev_unconstrained_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(2); //
  Eigen::Matrix<double, -1, 1>  prev_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(2); //
  Eigen::Matrix<double,  1, 1>  prev_unconstrained_grad_vec_out =   Eigen::Matrix<double, 1, 1>::Zero(2 - 1); //
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1>   grad_pi_wrt_b_raw =  Eigen::Matrix<double, -1, -1>::Zero(2, n_tests) ;  //// for latent_trait
  //std::vector<Eigen::Matrix<double, -1, -1 > >  deriv_Bound_Z_x_L = vec_of_mats(chunk_size, n_tests*2, 2) ;  //// for latent_trait
  ///////////////////////////////////////////////
  
  {
    
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::Matrix<double, -1, -1>>   Z_std_norm =  vec_of_mats<double>(chunk_size, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>>   Bound_Z =  Z_std_norm ;
  std::vector<Eigen::Matrix<double, -1, -1>>   Bound_U_Phi_Bound_Z =  Z_std_norm ;
  std::vector<Eigen::Matrix<double, -1, -1>>   Phi_Z =  Z_std_norm ;
  std::vector<Eigen::Matrix<double, -1, -1>>   phi_Bound_Z =  Z_std_norm ;
  std::vector<Eigen::Matrix<double, -1, -1>>   y1_log_prob =  Z_std_norm ;
  std::vector<Eigen::Matrix<double, -1, -1>>   prob =  Z_std_norm ;
  std::vector<Eigen::Matrix<double, -1, -1>>   phi_Z_recip =  Z_std_norm ;
  ///////////////////////////////////////////////
  std::vector<Eigen::Matrix<double, -1, -1>>     log_phi_Z_recip =  vec_of_mats<double>(chunk_size, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>>     log_phi_Bound_Z =  vec_of_mats<double>(chunk_size, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>>     log_Z_std_norm =   vec_of_mats<double>(chunk_size, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>>     log_abs_Bound_Z =  vec_of_mats<double>(chunk_size, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>>     sign_Bound_Z =     vec_of_mats<double>(chunk_size, n_tests, n_class);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> y_chunk =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1> u_array =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1> y_sign_chunk =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1> y_m_y_sign_x_u = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1>  inc_array  =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk   =           Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1> log_abs_u_grad_array_CM_chunk   =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> log_sum_result = Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
  Eigen::Matrix<double, -1, 1> log_sum_abs_result = Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
  Eigen::Matrix<double, -1, 1> sign_result = Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
  Eigen::Matrix<double, -1, 1> container_max_logs = Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
  Eigen::Matrix<double, -1, 1> container_sum_exp_signed = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> u_unc_vec_chunk =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests); 
  Eigen::Matrix<double, -1, 1> u_vec_chunk =        Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests); 
  Eigen::Matrix<double, -1, 1> du_wrt_duu_chunk =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests); 
  Eigen::Matrix<double, -1, 1> d_J_wrt_duu_chunk =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests); 
  ///////////////////////////////////////////////
  double log_jac_u = 0.0;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1>     log_common_grad_term_1   =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0);
  Eigen::Matrix<double, -1, -1>     log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip   =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0) ;
  Eigen::Matrix<double, -1, -1>     log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip   =   Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0) ;
  Eigen::Matrix<double, -1, -1>     sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip   =  Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests) ;
  Eigen::Matrix<double, -1, -1>     sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip   =   Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests) ;
  Eigen::Matrix<double, -1, -1>     log_prob_rowwise_prod_temp   =   Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0) ;
  Eigen::Matrix<double, -1, -1>     log_prob_recip_rowwise_prod_temp   =   Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0) ;
  Eigen::Matrix<double, -1, -1>     log_abs_grad_prob =    Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0);
  Eigen::Matrix<double, -1, -1>     log_abs_z_grad_term =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests,  -700.0);
  Eigen::Matrix<double, -1, -1>     sign_grad_prob =    Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1>     sign_z_grad_term =  Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1>     log_abs_prod_container_or_inc_array_comp  =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests, -700.0);
  Eigen::Matrix<double, -1, -1>     sign_prod_container_or_inc_array_comp  =  Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
  Eigen::Matrix<double, -1, -1>     log_abs_derivs_chain_container_vec_comp =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests, -700.0);
  Eigen::Matrix<double, -1, -1>     sign_derivs_chain_container_vec_comp =  Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1>      log_abs_prod_container_or_inc_array  =  Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
  Eigen::Matrix<double, -1, 1>      sign_prod_container_or_inc_array  =  Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
  Eigen::Matrix<double, -1, 1>      log_abs_derivs_chain_container_vec  =  Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
  Eigen::Matrix<double, -1, 1>      sign_derivs_chain_container_vec  =  Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
  Eigen::Matrix<double, -1, 1>      log_prob_rowwise_prod_temp_all  =  Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1>  log_abs_prev_grad_array_col_for_each_n =  Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
  Eigen::Matrix<double, -1, 1>  sign_prev_grad_array_col_for_each_n =  Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1>  log_abs_a  =  Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
  Eigen::Matrix<double, -1, 1>  sign_a =  Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
  Eigen::Matrix<double, -1, 1>  log_abs_b =  Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
  Eigen::Matrix<double, -1, 1>  sign_b =  Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
  Eigen::Matrix<double, -1, 1>  sign_sum_result =  Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
  Eigen::Matrix<double, -1, -1> log_terms =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_tests, -700.0);
  Eigen::Matrix<double, -1, -1> sign_terms =  Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_tests);
  Eigen::Matrix<double, -1, 1>  final_log_sum =  Eigen::Matrix<double, -1, 1>::Constant(chunk_size, -700.0);
  Eigen::Matrix<double, -1, 1>  final_sign =  Eigen::Matrix<double, -1, 1>::Ones(chunk_size);
  ///////////////////////////////////////////////
  Eigen::VectorXi overflow_mask(chunk_size);
  Eigen::VectorXi underflow_mask(chunk_size);
  Eigen::VectorXi OK_mask(chunk_size);
  /////////////////////////////////////////////  ////-----------------------------------------------
  std::vector<std::vector<std::vector<int>>> problem_index_array(n_class);
  std::vector<std::vector<int>> n_problem_array(n_class);
  for (int c = 0; c < n_class; c++) {
    problem_index_array[c].resize(n_tests); // initialise 
    n_problem_array[c].resize(n_tests);  // initialise 
    for (int t = 0; t < n_tests; t++) { 
      n_problem_array[c][t] = 0; // initialise 
    }
  }
  /////////////////////////////////////////////  ////-----------------------------------------------
  std::vector<Eigen::Matrix<double, -1, -1>>   Omega_grad_array_for_each_n =     vec_of_mats<double>(chunk_size, n_tests, n_tests);
  std::vector<Eigen::Matrix<double, -1, -1>>   sign_Omega_grad_array_for_each_n =     Omega_grad_array_for_each_n;
  std::vector<Eigen::Matrix<double, -1, -1>>   log_abs_Omega_grad_array_for_each_n =  Omega_grad_array_for_each_n;
  std::vector<Eigen::Matrix<double, -1, -1>>   beta_grad_array_for_each_n =      vec_of_mats<double>(chunk_size, n_tests, n_covariates_max);
  std::vector<Eigen::Matrix<double, -1, -1>>   sign_beta_grad_array_for_each_n =      beta_grad_array_for_each_n;
  std::vector<Eigen::Matrix<double, -1, -1>>   log_abs_beta_grad_array_for_each_n =   beta_grad_array_for_each_n;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1>  rowwise_log_sum = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  Eigen::Matrix<double, -1, 1>  rowwise_prod =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  Eigen::Matrix<double, -1, 1>  rowwise_sum =     Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
  ///////////////////////////////////////////////
  ////   latent_trait specific:
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1>  log_abs_bs_grad_array_col_for_each_n  =  Eigen::Matrix<double, -1, 1>::Zero(1); 
  Eigen::Matrix<double, -1, 1>  sign_bs_grad_array_col_for_each_n  =  Eigen::Matrix<double, -1, 1>::Zero(1);
  Eigen::Matrix<double, -1, -1> log_abs_deriv_Bound_Z_x_L =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
  Eigen::Matrix<double, -1, -1> sign_deriv_Bound_Z_x_L =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
  Eigen::Matrix<double, -1, -1> log_abs_deriv_Bound_Z_x_L_comp =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
  Eigen::Matrix<double, -1, -1> sign_deriv_Bound_Z_x_L_comp =  Eigen::Matrix<double, -1, -1>::Zero(1, n_tests);
  ///////////////////////////////////////////////
  
  
   { // start of big local block

        Eigen::Matrix<double, -1, -1>    lp_array  =  Eigen::Matrix<double, -1, -1>::Constant(chunk_size, n_class, -700.0); 
        const Eigen::Matrix<double, -1, -1>    signs_Ones = Eigen::Matrix<double, -1, -1>::Ones(chunk_size, n_class);

    for (int nc = 0; nc < n_chunks; nc++) {

      int chunk_counter = nc;
      
      if ((chunk_counter == n_full_chunks) && (n_chunks > 1) && (last_chunk_size > 0)) { // Last chunk (remainder - don't use AVX / SIMD for this)
        
                      chunk_size = last_chunk_size;  //// update chunk_size 
                      
                      /// use either Loop (i.e. double fn's) or Stan's vectorisation for the remainder (i.e. last) chunk, regardless of input
                      vect_type = "Stan";
                      vect_type_exp = "Stan";
                      vect_type_log = "Stan";
                      vect_type_lse = "Stan";
                      vect_type_tanh = "Stan";
                      vect_type_Phi = "Stan";
                      vect_type_log_Phi = "Stan";
                      vect_type_inv_Phi = "Stan";
                      vect_type_inv_Phi_approx_from_logit_prob = "Stan";
 
                      // vectors
                      inc_array.resize(last_chunk_size);
                      log_sum_result.resize(last_chunk_size);
                      log_sum_abs_result.resize(last_chunk_size);
                      container_max_logs.resize(last_chunk_size);
                      container_sum_exp_signed.resize(last_chunk_size);
                      sign_result.resize(last_chunk_size);
                      
                      u_unc_vec_chunk.resize(last_chunk_size * n_tests);
                      u_vec_chunk.resize(last_chunk_size * n_tests);
                      du_wrt_duu_chunk.resize(last_chunk_size * n_tests);
                      d_J_wrt_duu_chunk.resize(last_chunk_size * n_tests);
                      
                      log_abs_a.resize(last_chunk_size);
                      sign_a.resize(last_chunk_size);
                      log_abs_b.resize(last_chunk_size);
                      sign_b.resize(last_chunk_size);
                      sign_sum_result.resize(last_chunk_size);
                      final_log_sum.resize(last_chunk_size);
                      final_sign.resize(last_chunk_size);
                      
                      log_abs_prev_grad_array_col_for_each_n.resize(last_chunk_size);
                      sign_prev_grad_array_col_for_each_n.resize(last_chunk_size);
                      
                      log_abs_prod_container_or_inc_array.resize(last_chunk_size);
                      sign_prod_container_or_inc_array.resize(last_chunk_size);
                      log_abs_derivs_chain_container_vec.resize(last_chunk_size);
                      sign_derivs_chain_container_vec.resize(last_chunk_size);
                      log_prob_rowwise_prod_temp_all.resize(last_chunk_size);
                      
                      overflow_mask.resize(last_chunk_size);
                      underflow_mask.resize(last_chunk_size);
                      OK_mask.resize(last_chunk_size);
                      
                      log_abs_bs_grad_array_col_for_each_n.resize(last_chunk_size);
                      sign_bs_grad_array_col_for_each_n.resize(last_chunk_size);
                      
                      // matrices
                      u_grad_array_CM_chunk.resize(last_chunk_size, n_tests); u_grad_array_CM_chunk.setZero();
                      log_abs_u_grad_array_CM_chunk.resize(last_chunk_size, n_tests); log_abs_u_grad_array_CM_chunk.setZero();
                      lp_array.resize(last_chunk_size, n_class); lp_array.setZero();
                      log_common_grad_term_1.resize(last_chunk_size, n_tests); log_common_grad_term_1.setConstant(-700.0);
                      
                      y_chunk.resize(last_chunk_size, n_tests); y_chunk.setZero();
                      u_array.resize(last_chunk_size, n_tests); u_array.setZero();
                      y_sign_chunk.resize(last_chunk_size, n_tests); y_sign_chunk.setZero();
                      y_m_y_sign_x_u.resize(last_chunk_size, n_tests); y_m_y_sign_x_u.setZero();
                      
                      log_terms.resize(last_chunk_size, n_tests);
                      sign_terms.resize(last_chunk_size, n_tests);
                      
                      log_abs_prod_container_or_inc_array_comp.resize(last_chunk_size, n_tests);
                      sign_prod_container_or_inc_array_comp.resize(last_chunk_size, n_tests);
                      log_abs_derivs_chain_container_vec_comp.resize(last_chunk_size, n_tests);
                      sign_derivs_chain_container_vec_comp.resize(last_chunk_size, n_tests);
                      
                      log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
                      log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
                      sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
                      sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
                      log_prob_recip_rowwise_prod_temp.resize(last_chunk_size, n_tests);
                      log_abs_grad_prob.resize(last_chunk_size, n_tests);
                      log_abs_z_grad_term.resize(last_chunk_size, n_tests);
                      sign_grad_prob.resize(last_chunk_size, n_tests);
                      sign_z_grad_term.resize(last_chunk_size, n_tests);
                      
                      // ///deriv_Bound_Z_x_L.resize(last_chunk_size, n_tests);
                      // grad_bound_z.resize(last_chunk_size, n_tests);
                      // grad_Phi_bound_z.resize(last_chunk_size, n_tests);
                      
                      // matrix arrays
                      for (int c = 0; c < n_class; c++) {
                            Z_std_norm[c].resize(last_chunk_size, n_tests);
                            log_Z_std_norm[c].resize(last_chunk_size, n_tests);
                            Bound_Z[c].resize(last_chunk_size, n_tests);
                            Bound_U_Phi_Bound_Z[c].resize(last_chunk_size, n_tests);
                            Phi_Z[c].resize(last_chunk_size, n_tests);
                            phi_Bound_Z[c].resize(last_chunk_size, n_tests);
                            y1_log_prob[c].resize(last_chunk_size, n_tests);
                            prob[c].resize(last_chunk_size, n_tests);
                            phi_Z_recip[c].resize(last_chunk_size, n_tests);
                            log_phi_Z_recip[c].resize(last_chunk_size, n_tests);
                            log_phi_Bound_Z[c].resize(last_chunk_size, n_tests);
                            log_abs_Bound_Z[c].resize(last_chunk_size, n_tests);
                            sign_Bound_Z[c].resize(last_chunk_size, n_tests);
                      }
                      
                      for (int i = 0; i < Omega_grad_array_for_each_n.size(); i++) {
                        Omega_grad_array_for_each_n[i].resize(last_chunk_size, n_tests);
                        sign_Omega_grad_array_for_each_n[i].resize(last_chunk_size, n_tests);
                        log_abs_Omega_grad_array_for_each_n[i].resize(last_chunk_size, n_tests);
                      }
                      for (int i = 0; i < beta_grad_array_for_each_n.size(); i++) {
                        beta_grad_array_for_each_n[i].resize(last_chunk_size, n_tests);
                        sign_beta_grad_array_for_each_n[i].resize(last_chunk_size, n_tests);
                        log_abs_beta_grad_array_for_each_n[i].resize(last_chunk_size, n_tests);
                      } 
        
      }
      
      
      { ////-----------------------------------------------
        u_grad_array_CM_chunk.setZero() ; //// reset between chunks as re-using same container
        log_abs_u_grad_array_CM_chunk.setConstant(-700);  // reset between chunks as re-using same container
        
        y_chunk = y_ref.middleRows( chunk_size_orig * chunk_counter, chunk_size).cast<double>();
        
        //// Nuisance parameter transformation step
        u_unc_vec_chunk = u_unc_vec.segment( chunk_size_orig * n_tests * chunk_counter, chunk_size * n_tests);
        
        fn_MVP_compute_nuisance( u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);
        log_jac_u +=    fn_MVP_compute_nuisance_log_jac_u(   u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);
        
        u_array  =  u_vec_chunk.reshaped(chunk_size, n_tests);
        y_sign_chunk.array() =      y_chunk.array() + (y_chunk.array() - 1.0) ;
        y_m_y_sign_x_u.array() =  y_chunk.array() - (y_sign_chunk.array() * u_array.array());
      } ////-----------------------------------------------
      

      {
        // START of c loop
        for (int c = 0; c < n_class; c++) {
            
            inc_array.setZero(); //   = 0.0; // needs to be reset to 0
            
            for (int t = 0; t < n_tests; t++) {   // start of t loop
              
              if (n_covariates_max > 1) {  //// at present latent_trait only works w/o covariates !!  
                
                // Eigen::Matrix<double, -1, 1>    Xbeta_given_class_c_col_t = X[c][t].block(chunk_size_orig * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(c, t)).cast<double>()  * beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t));
                // Bound_Z[c].col(t).array() =     L_Omega_recip_double[c](t, t) * (  - ( Xbeta_given_class_c_col_t.array()    +      inc_array.array()   )  ) ;
                // sign_Bound_Z[c].col(t) =   Bound_Z[c].col(t).array().sign();
                // log_abs_Bound_Z[c].col(t) =    (fn_EIGEN_double(Bound_Z[c].col(t).array().abs(), "log", vect_type_log));
                
              } else {  // intercept-only
                    
                     Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * ( -1.0*( LT_a_double(c, t) + inc_array.array() ) ) ;
                    
                    { ////-----------------------------------------------
                      Eigen::Matrix<double, -1, 1> Bound_Z_col_t = Bound_Z[c].col(t);
                      Eigen::Matrix<double, -1, 1> Bound_Z_col_t_sign = stan::math::sign(Bound_Z_col_t);
                      Eigen::Matrix<double, -1, 1> Bound_Z_col_t_abs =  stan::math::abs(Bound_Z_col_t);
                      ////
                      sign_Bound_Z[c].col(t) =   Bound_Z_col_t_sign;
                      log_abs_Bound_Z[c].col(t) =    fn_EIGEN_double( Bound_Z_col_t_abs, "log", vect_type_log);
                    } ////-----------------------------------------------
                
              }
              
              //// create masks
              { ////-----------------------------------------------
                Eigen::Matrix<double, -1, 1> y_chunk_col_t_dbl = y_chunk.col(t);
                Eigen::Matrix<double, -1, 1> Bound_Z_col_t = Bound_Z[c].col(t);
                overflow_mask.array() =   ( (Bound_Z_col_t.array() > overflow_threshold)  && (y_chunk_col_t_dbl.array() == 1.0) ).cast<int>()  ;
                underflow_mask.array() =  ( (Bound_Z_col_t.array() < underflow_threshold) && (y_chunk_col_t_dbl.array() == 0.0) ).cast<int>()  ;
                OK_mask.array() = ( (overflow_mask.array() == 0) && (underflow_mask.array() == 0) ).cast<int>() ;
              } ////-----------------------------------------------
              
            //// counts 
            const int n_overflows =   overflow_mask.sum();
            const int n_underflows =  underflow_mask.sum();
            const int n_problem = n_overflows + n_underflows;
            const int n_OK = OK_mask.sum();
            
            std::vector<int> over_index(n_overflows);
            std::vector<int> under_index(n_underflows);
            std::vector<int> problem_index(n_problem);
            
            int counter_over  = 0;
            int counter_under  = 0;
            
            for (int n = 0; n < chunk_size; ++n) {
              
              if  (overflow_mask(n) == 1) {
                over_index[counter_over] = n;
                counter_over += 1;
              } else if  (underflow_mask(n) == 1)   {
                under_index[counter_under] = n;
                counter_under += 1;
              } else { 
                // OK_index[counter_ok] = n;
                // counter_ok += 1;
              }
              
            }
            
            
            {
              
              int counter = 0;
              
              for (int i = 0; i < n_overflows; ++i) {
                problem_index[counter] = over_index[i];
                counter += 1;
              }
              
              for (int i = 0; i < n_underflows; ++i) {
                problem_index[counter] = under_index[i];
                counter += 1;
              }
              
            }
            
            /// fill the index arrays
            n_problem_array[c][t] = n_problem;
            problem_index_array[c][t].resize(n_problem);  
            problem_index_array[c][t] = problem_index;
            
            //// compute/update important log-lik quantities for GHK-MVP
            fn_MVP_compute_lp_GHK_cols(t,
                                       Bound_U_Phi_Bound_Z[c], // computing this
                                       Phi_Z[c], // computing this
                                       Z_std_norm[c], // computing this
                                       prob[c],        // computing this
                                       y1_log_prob[c], // computing this
                                       Bound_Z[c],
                                       y_chunk,
                                       u_array,
                                       Model_args_as_cpp_struct);
            
            //// compute/update important grad quantities for GHK-MVP
            fn_MVP_compute_phi_Z_recip_cols(    t,
                                                phi_Z_recip[c], // computing this
                                                Phi_Z[c], Z_std_norm[c], Model_args_as_cpp_struct);
            
            fn_MVP_compute_phi_Bound_Z_cols(     t,
                                                 phi_Bound_Z[c], // computing this
                                                 Bound_U_Phi_Bound_Z[c], Bound_Z[c], Model_args_as_cpp_struct);
            
             //// compute log-scale quantities (for grad)
             { ////-----------------------------------------------
                   Eigen::Matrix<double, -1, -1> temp_mat =       Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
                   Eigen::Matrix<double, -1, 1>  temp_col_t =     Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
                   Eigen::Matrix<double, -1, 1>  temp_col_t_abs = Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
                   
                   temp_mat = phi_Bound_Z[c];
                   temp_col_t = temp_mat.col(t);
                   temp_col_t_abs = stan::math::abs(temp_col_t);
                   log_phi_Bound_Z[c].col(t) =   fn_EIGEN_double( temp_col_t_abs,  "log", vect_type_log);
                   
                   temp_mat = phi_Z_recip[c];
                   temp_col_t = temp_mat.col(t);
                   temp_col_t_abs = stan::math::abs(temp_col_t);
                   log_phi_Z_recip[c].col(t) =   fn_EIGEN_double( temp_col_t_abs,  "log", vect_type_log);
                   
                   temp_mat = Z_std_norm[c];
                   temp_col_t = temp_mat.col(t);
                   temp_col_t_abs = stan::math::abs(temp_col_t);
                   log_Z_std_norm[c].col(t)  =   fn_EIGEN_double( temp_col_t_abs,  "log", vect_type_log);
             } ////-----------------------------------------------
            
            if   (n_OK == chunk_size)  { 
                          //// carry on as normal as (likely) no * problematic * overflows/underflows
            }  else if (n_OK < chunk_size)  {
                    
                    if (n_underflows > 0) { //// underflow (w/ y == 0)
                      
                          const std::vector<int> index = under_index;
                          const int index_size = index.size();
                          
                          fn_MVP_compute_lp_GHK_cols_log_scale_underflow(t, 
                                                                         index, 
                                                                         Bound_U_Phi_Bound_Z[c], // computing this
                                                                         Phi_Z[c],  // computing this
                                                                         Z_std_norm[c],  // computing this
                                                                         log_Z_std_norm[c],  // computing this
                                                                         prob[c],  // computing this
                                                                         y1_log_prob[c], // computing this
                                                                         log_phi_Bound_Z[c], // computing this
                                                                         log_phi_Z_recip[c], // computing this
                                                                         Bound_Z[c],
                                                                         u_array,
                                                                         Model_args_as_cpp_struct);
                          
                          phi_Bound_Z[c](index, t).array() =  fn_EIGEN_double(  log_phi_Bound_Z[c](index, t),  "exp", vect_type_exp); // not needed if comp. grad on log-scale
                          phi_Z_recip[c](index, t).array() =  fn_EIGEN_double(  log_phi_Z_recip[c](index, t),  "exp", vect_type_exp); // not needed if comp. grad on log-scale
                      
                    }
                    
                    if (n_overflows > 0) { //// overflow (w/ y == 1)
                      
                          const std::vector<int>   index = over_index;
                          const int index_size = index.size();
                          
                          fn_MVP_compute_lp_GHK_cols_log_scale_overflow( t, 
                                                                         n_overflows,
                                                                         index, 
                                                                         Bound_U_Phi_Bound_Z[c],  // computing this
                                                                         Phi_Z[c],  // computing this
                                                                         Z_std_norm[c],  // computing this
                                                                         log_Z_std_norm[c],  // computing this
                                                                         prob[c],  // computing this
                                                                         y1_log_prob[c],  // computing this
                                                                         log_phi_Bound_Z[c],  // computing this
                                                                         log_phi_Z_recip[c],  // computing this
                                                                         Bound_Z[c],
                                                                         u_array,
                                                                         Model_args_as_cpp_struct);
                          
                          phi_Bound_Z[c](index, t).array() =  fn_EIGEN_double(  log_phi_Bound_Z[c](index, t),  "exp", vect_type_exp);  // not needed if comp. grad on log-scale
                          phi_Z_recip[c](index, t).array() =  fn_EIGEN_double(  log_phi_Z_recip[c](index, t),  "exp", vect_type_exp);  // not needed if comp. grad on log-scale
                      
                    }
            
            }  ///// end of "if overflow or underflow" block
            
            if (t < n_tests - 1) { 
              
                    //// -----------------------------------------------
                    Eigen::Matrix<double, 1, -1> L_Omega_row = L_Omega_double[c].row(t + 1);
                    inc_array = Z_std_norm[c].leftCols(t + 1) * L_Omega_row.head(t + 1).transpose();
                    //// -----------------------------------------------
              
            }
                
        }      //// end of t loop
            
            //// -----------------------------------------------  
            if (n_class > 1) {  //// if latent class
              rowwise_sum = y1_log_prob[c].rowwise().sum();
              rowwise_sum.array() += log_prev(0, c);
              lp_array.col(c) = rowwise_sum;
            } else {
              rowwise_sum = y1_log_prob[c].rowwise().sum();
              lp_array.col(0) =     rowwise_sum;
            }
            //// -----------------------------------------------  
        
      }   //// end of c loop
      
    }  //// end of local block
      
      if (n_class > 1) {  /// if latent class 
        
              //// -----------------------------------------------   
              log_sum_exp_general(   lp_array,
                                     vect_type_exp,
                                     vect_type_log,
                                     log_sum_result,
                                     container_max_logs);
              const int index_start = 1 + n_params + chunk_size_orig * chunk_counter;
              out_mat.segment(index_start, chunk_size) = log_sum_result;
              //// -----------------------------------------------   
        
      } else {
              
              const int index_start = 1 + n_params + chunk_size_orig * chunk_counter;
              out_mat.segment(index_start, chunk_size) = lp_array.col(0);
        
      }
      
      Eigen::Matrix<double, -1, 1> log_prob_n_recip =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
      { //// -----------------------------------------------  
        Eigen::Matrix<double, -1, 1> log_lik = out_mat.tail(N);
        Eigen::Matrix<double, -1, 1> log_lik_segment = log_lik.segment(chunk_size_orig * chunk_counter, chunk_size);
        Eigen::Matrix<double, -1, 1> prob_n  =  fn_EIGEN_double( log_lik_segment, "exp",  vect_type_exp);
        Eigen::Matrix<double, -1, 1> prob_n_recip = stan::math::inv(prob_n);
        log_prob_n_recip = fn_EIGEN_double( prob_n_recip, "log", vect_type_log);
      } //// -----------------------------------------------  
      
      const bool compute_final_scalar_grad = false;
      
      /////////////////////////////////////////////////
      ///////////////// ------------------------- compute grad  -----------------------------------------------------------------------------------------------------------
      /////////////////////////////////////////////
      //// Needed for latent_trait (not LC_MVP):
      Eigen::Matrix<double, -1, -1> log_abs_y_sign_chunk =   (fn_EIGEN_double(y_sign_chunk.array().abs(), "log", vect_type_log)); 
      Eigen::Matrix<double, -1, -1> log_abs_y_m_y_sign_x_u =  (fn_EIGEN_double(y_m_y_sign_x_u.array().abs(), "log", vect_type_log));  
      Eigen::Matrix<double, -1, -1> sign_y_m_y_sign_x_u = y_m_y_sign_x_u.array().sign();

      for (int c = 0; c < n_class; c++) {
          
          ////-----------------------------------------------  
          for (int i = 0; i <  beta_grad_array_for_each_n.size();  i++) {
            beta_grad_array_for_each_n[i].setZero();
            sign_beta_grad_array_for_each_n[i].setOnes();
            log_abs_beta_grad_array_for_each_n[i].setConstant(-700.0);  
          }
          for (int i = 0; i <  Omega_grad_array_for_each_n.size();  i++) {
            Omega_grad_array_for_each_n[i].setZero();
            sign_Omega_grad_array_for_each_n[i].setOnes();
            log_abs_Omega_grad_array_for_each_n[i].setConstant(-700.0);  
          } 
          
          Eigen::Matrix<double, -1, -1> y1_log_prob_recip = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> sign_Z_std_norm =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> prob_recip =        Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> common_grad_term_1 =                                                         Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> prob_rowwise_prod_temp =                                                     Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip =                        Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          Eigen::Matrix<double, -1, -1> y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
          ////-----------------------------------------------  
          
          { ////----------------------------------------------- 
            
            Eigen::Matrix<double, -1, -1> temp_mat =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
            Eigen::Matrix<double, -1, -1> temp_mat_2 = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
            
            {
              temp_mat = y1_log_prob[c];
              temp_mat_2 = -1.0*temp_mat;
              y1_log_prob_recip = temp_mat_2;
            }
            {
              temp_mat = Z_std_norm[c];
              temp_mat_2 = stan::math::sign(temp_mat);
              sign_Z_std_norm = temp_mat_2;
            }
            {
              temp_mat = prob[c];
              temp_mat_2 = stan::math::inv(temp_mat);
              prob_recip = temp_mat_2;
            }
            
          } ////----------------------------------------------- 
          
          if ( (grad_option != "none") || (grad_option == "test") ) {
                    
                    ////-----------------------------------------------  
                    Eigen::Matrix<double, -1, -1> abs_L_Omega_recip_double =     Eigen::Matrix<double, -1, -1>::Zero(n_tests, n_tests);
                    Eigen::Matrix<double, -1, -1> log_abs_L_Omega_recip_double = Eigen::Matrix<double, -1, -1>::Zero(n_tests, n_tests);
                    Eigen::Matrix<double, -1, -1> sign_L_Omega_recip_double =    Eigen::Matrix<double, -1, -1>::Ones(n_tests, n_tests);
                    ////-----------------------------------------------   
                    
                    ////-----------------------------------------------  
                    abs_L_Omega_recip_double =  stan::math::abs(L_Omega_recip_double[c]);    //  std::cout << "After abs" << std::endl;
                    sign_L_Omega_recip_double = stan::math::sign(L_Omega_recip_double[c]);   //  std::cout << "After sign" << std::endl;
                    for (int t = 0; t < n_tests; t++) {
                      log_abs_L_Omega_recip_double(t, t) = stan::math::log(abs_L_Omega_recip_double(t, t));
                    }
                    ////-----------------------------------------------  
                     
                    fn_MVP_grad_prep_log_scale(       log_prob_rowwise_prod_temp,
                                                      log_prob_recip_rowwise_prod_temp,
                                                      log_prob_rowwise_prod_temp_all,
                                                      log_common_grad_term_1,
                                                      log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                      log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                      sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                      sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                      y1_log_prob[c],
                                                      y1_log_prob_recip,
                                                      log_prob_n_recip,
                                                      log_prev(0, c), /// if not latent class, this is a dummy variable
                                                      log_phi_Bound_Z[c],
                                                      log_phi_Z_recip[c],
                                                      log_abs_L_Omega_recip_double,
                                                      sign_L_Omega_recip_double,
                                                      y_sign_chunk,
                                                      y_m_y_sign_x_u,
                                                      Model_args_as_cpp_struct);
                    
                    ////-----------------------------------------------  
                    //// these should all be OK on windows  (not dangling reference)
                    common_grad_term_1 = fn_EIGEN_double(log_common_grad_term_1, "exp", vect_type_exp);
                    prob_rowwise_prod_temp = fn_EIGEN_double(log_prob_rowwise_prod_temp, "exp", vect_type_exp);
                    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip  =  fn_EIGEN_double(log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip, "exp", vect_type_exp).array() *
                                                                            sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.array();
                    y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = fn_EIGEN_double(log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip, "exp", vect_type_exp).array() *
                                                                                                 sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.array();
                    ////-----------------------------------------------  
            
          }

          //////////////////////////////////////////////////////////////////////////////// Grad of nuisance parameters / u's (manual) ----------------------------------------------------------------
          if ( (grad_option == "us_only") || (grad_option == "all") ) {
            
            Eigen::Matrix<double, -1, -1>   u_grad_array_CM_chunk_block =        u_grad_array_CM_chunk; 
            
            {
              
              fn_MVP_compute_nuisance_grad_v2(  u_grad_array_CM_chunk_block,
                                                phi_Z_recip[c],
                                                common_grad_term_1,
                                                L_Omega_double[c],
                                                prob[c],
                                                prob_recip,
                                                prob_rowwise_prod_temp,
                                                y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                log_abs_z_grad_term,
                                                log_abs_grad_prob,
                                                log_abs_prod_container_or_inc_array,
                                                sign_prod_container_or_inc_array,
                                                Model_args_as_cpp_struct);
              
              
            }
            
            
           { /// then compute gradients on the LOG-scale, but ONLY where we have underflow or overflow.
              
              fn_MVP_compute_nuisance_grad_log_scale( n_problem_array[c],
                                                      problem_index_array[c],
                                                      log_abs_u_grad_array_CM_chunk,
                                                      u_grad_array_CM_chunk_block,
                                                      L_Omega_double[c],
                                                      log_abs_L_Omega_double[c],
                                                      log_phi_Z_recip[c],
                                                      y1_log_prob[c],
                                                      y1_log_prob_recip,
                                                      log_prob_rowwise_prod_temp,
                                                      log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                      sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                      log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                      sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                      log_common_grad_term_1,
                                                      log_abs_z_grad_term,
                                                      sign_z_grad_term,
                                                      log_abs_grad_prob,
                                                      sign_grad_prob,
                                                      log_abs_prod_container_or_inc_array,
                                                      sign_prod_container_or_inc_array,
                                                      log_sum_result,
                                                      sign_sum_result,
                                                      log_terms,
                                                      sign_terms,
                                                      log_abs_a,
                                                      log_abs_b,
                                                      sign_a,
                                                      sign_b,
                                                      container_max_logs,
                                                      container_sum_exp_signed,
                                                      Model_args_as_cpp_struct);
              
            }
            
            //// update u_grad_array_CM_chunk once standard-scale and log-scale grad computations are done
            u_grad_array_CM_chunk.array() += u_grad_array_CM_chunk_block.array();
            
            const int start_index = 1 + (chunk_size_orig * n_tests * chunk_counter);
            const int length = chunk_size * n_tests;
            
            if (c == n_class - 1) {
              
                  //// update output vector once all u_grad computations are done
                  out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();
                  
                  //// account for unconstrained -> constrained transformations and Jacobian adjustments
                  fn_MVP_nuisance_first_deriv( du_wrt_duu_chunk,
                                               u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct); 
                  
                  fn_MVP_nuisance_deriv_of_log_det_J(    d_J_wrt_duu_chunk,
                                                         u_vec_chunk, u_unc_vec_chunk, du_wrt_duu_chunk, Model_args_as_cpp_struct);
                  
                  out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
                  out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();
              
            }
            
            
            sign_z_grad_term.setOnes();
            sign_grad_prob.setOnes();
            sign_prod_container_or_inc_array.setOnes();
            sign_sum_result.setOnes();
            sign_terms.setOnes();
            sign_a.setOnes();
            sign_b.setOnes();
            
            log_abs_z_grad_term.setConstant(-700.0);
            log_abs_grad_prob.setConstant(-700.0);
            log_abs_prod_container_or_inc_array.setConstant(-700.0);
            log_sum_result.setConstant(-700.0);
            log_terms.setConstant(-700.0);
            log_abs_a.setConstant(-700.0);
            log_abs_b.setConstant(-700.0);
            container_max_logs.setConstant(-700.0);
            
            
          }

          
          ///////////////////////////////////////////////////////////////////////////   -------------  Grad of intercepts / coefficients (beta's) ----------------------------------------------------------------
          if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only") ) {
            
                 if (n_covariates_max > 1) {  ///  not implemented yet for latent_trait (bookmark - for future)
                   
                          // fn_MVP_compute_coefficients_grad_v2(    c,
                          //                                         beta_grad_array[c],
                          //                                         beta_grad_array_for_each_n,
                          //                                         chunk_counter,
                          //                                         n_covariates_max,
                          //                                         common_grad_term_1,
                          //                                         L_Omega_double[c],
                          //                                         prob[c],
                          //                                         prob_recip,
                          //                                         prob_rowwise_prod_temp,
                          //                                         y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip, 
                          //                                         y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                          //                                         log_abs_grad_prob,
                          //                                         sign_grad_prob,
                          //                                         log_abs_prod_container_or_inc_array,
                          //                                         sign_prod_container_or_inc_array,
                          //                                         false, /// compute_final_scalar_grad
                          //                                         Model_args_as_cpp_struct);
                    
                   } else { 
                     
                     {   // compute (some or all) of grads on log-scale
                  
                     
                        { /// first compute gradients on the standard (non-log) scale. 
                          
                          
                          log_abs_grad_prob.setZero();
                          sign_grad_prob.setZero();
                          log_abs_prod_container_or_inc_array.setZero();
                          sign_prod_container_or_inc_array.setZero();
                          
                          fn_MVP_compute_coefficients_grad_v2(   c,
                                                                 beta_grad_array[c],
                                                                 beta_grad_array_for_each_n,
                                                                 chunk_counter,
                                                                 n_covariates_max,
                                                                 common_grad_term_1,
                                                                 L_Omega_double[c],
                                                                 prob[c],
                                                                 prob_recip,
                                                                 prob_rowwise_prod_temp,
                                                                 y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                                 y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                                 log_abs_grad_prob,
                                                                 sign_grad_prob,
                                                                 log_abs_prod_container_or_inc_array,
                                                                 sign_prod_container_or_inc_array,
                                                                 false, /// compute_final_scalar_grad
                                                                 Model_args_as_cpp_struct);
                          
                        }
                       
                        { // then compute gradients on the LOG-scale, but ONLY where we have underflow or overflow.
                    
                        //// extract previous signs and log_abs values
                        for (int i = 0; i <  beta_grad_array_for_each_n.size();  i++) {
                          auto temp_array = beta_grad_array_for_each_n[i];
                          auto temp_array_sign = stan::math::sign(temp_array);
                          sign_beta_grad_array_for_each_n[i] =  temp_array_sign;
                          auto temp_array_abs = stan::math::abs(temp_array);
                          log_abs_beta_grad_array_for_each_n[i] = fn_EIGEN_double( temp_array_abs, "log", vect_type_log);
                          for (int t = 0; t < n_tests; t++) {
                            sign_beta_grad_array_for_each_n[i].col(t)(problem_index_array[c][t]).setOnes();
                            log_abs_beta_grad_array_for_each_n[i].col(t)(problem_index_array[c][t]).setConstant(-700.0);
                          }
                        }
                     
                     fn_MVP_compute_coefficients_grad_log_scale( n_problem_array[c],
                                                                 problem_index_array[c],
                                                                 beta_grad_array[c],
                                                                 sign_beta_grad_array_for_each_n,
                                                                 log_abs_beta_grad_array_for_each_n,
                                                                 L_Omega_double[c],
                                                                 log_abs_L_Omega_double[c],
                                                                 log_phi_Z_recip[c],
                                                                 y1_log_prob[c],
                                                                 log_prob_rowwise_prod_temp,
                                                                 log_abs_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                                 sign_y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                                 log_abs_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                                 sign_y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                                 log_common_grad_term_1,
                                                                 log_abs_z_grad_term,
                                                                 sign_z_grad_term,
                                                                 log_abs_grad_prob,
                                                                 sign_grad_prob,
                                                                 log_abs_prod_container_or_inc_array,
                                                                 sign_prod_container_or_inc_array,
                                                                 log_abs_prod_container_or_inc_array_comp,
                                                                 sign_prod_container_or_inc_array_comp,
                                                                 log_sum_result,
                                                                 sign_sum_result,
                                                                 log_terms,
                                                                 sign_terms,
                                                                 container_max_logs,
                                                                 container_sum_exp_signed,
                                                                 Model_args_as_cpp_struct);
                    
                            for (int t = 0; t < n_tests; t++) {
                              for (int k = 0; k <  beta_grad_array_for_each_n.size(); k++) {
                                LogSumVecSingedResult log_sum_vec_signed_struct = log_sum_vec_signed_v1(log_abs_beta_grad_array_for_each_n[k].col(t),  // entire array as carrying fwd the non-log-scale grad previously computed
                                                                                                        sign_beta_grad_array_for_each_n[k].col(t), // entire array as carrying fwd the non-log-scale grad previously computed
                                                                                                        vect_type);
                                
                                beta_grad_array[c](k, t) +=   stan::math::exp(log_sum_vec_signed_struct.log_sum) * log_sum_vec_signed_struct.sign;
                              }
                            }
                
                        }
                       
                     }
                     
                   }
            
            sign_z_grad_term.setOnes();
            sign_grad_prob.setOnes();
            sign_prod_container_or_inc_array.setOnes();
            sign_sum_result.setOnes();
            sign_terms.setOnes();
            sign_a.setOnes();
            sign_b.setOnes();
            sign_prod_container_or_inc_array_comp.setOnes();
            
            log_abs_z_grad_term.setConstant(-700.0);
            log_abs_grad_prob.setConstant(-700.0);
            log_abs_prod_container_or_inc_array.setConstant(-700.0);
            log_sum_result.setConstant(-700.0);
            log_terms.setConstant(-700.0);
            log_abs_a.setConstant(-700.0);
            log_abs_b.setConstant(-700.0);
            container_max_logs.setConstant(-700.0);
            log_abs_prod_container_or_inc_array_comp.setConstant(-700.0);
            
          }
      
          //////////////////////////////////////////////////////////////////  ---------------- Grad of b's (corr parameters)  --------------------------------------------------------------------------------
          if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only") ) {
            
             {  /// entire b-grad computed on log-scale (partial-log-scale not yet implemented)
 
              /////////////////////// TEMP - TEST GRAD OF LT_b's on NON-LOG SCALE:
              Eigen::Matrix<double, -1, -1> prob_rowwise_prod_temp = stan::math::exp(log_prob_rowwise_prod_temp);
              Eigen::Matrix<double, -1, -1> common_grad_term_1 = stan::math::exp(log_common_grad_term_1);
              ///////////////////////////////////////////////daq
              Eigen::Matrix<double, -1, -1> grad_bound_z = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
              Eigen::Matrix<double, -1, -1> grad_Phi_bound_z = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
              ///////////////////////////////////////////////
              std::vector<Eigen::Matrix<double, -1, -1>>  deriv_Bound_Z_x_L = vec_of_mats(chunk_size, n_tests*n_class, n_class) ;   
              for (int c = 0; c < n_class; ++c) {
                  deriv_Bound_Z_x_L[c].array() = stan::math::exp(log_abs_deriv_Bound_Z_x_L).array() * sign_deriv_Bound_Z_x_L.array();
              }
              ///////////////////////////////////////////////
              
              fn_LC_LT_compute_bs_grad_v1(  grad_pi_wrt_b_raw,  //// ---- correct
                                            deriv_Bound_Z_x_L[c],   //// ?????????????????????
                                            c,   //// ---- correct
                                            Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[c], 
                                            common_grad_term_1,  //// ---- correct
                                            L_Omega_double[c],   //// ---- correct
                                            prob[c],    //// ---- correct
                                            prob_recip,   //// ---- correct
                                            Bound_Z[c],    //// ---- correct
                                            Z_std_norm[c],   //// ---- correct
                                            phi_Bound_Z[c],     //// ?????????????????????
                                            phi_Z_recip[c],    //// ---- correct
                                            y_sign_chunk,    //// ---- correct
                                            y_m_y_sign_x_u,      //// ---- correct
                                            prob_rowwise_prod_temp,    //// ---- correct
                                            grad_bound_z,   //// ---- correct
                                            grad_Phi_bound_z,   //// ---- correct
                                            sign_z_grad_term,  //// ---- correct
                                            sign_grad_prob,   //// ---- correct
                                            sign_prod_container_or_inc_array,  //// ---- correct
                                            sign_derivs_chain_container_vec,   //// ---- correct
                                            true,  ///   compute_final_scalar_grad,   //// ---- correct
                                            Model_args_as_cpp_struct);    //// ---- correct
              
            }
            
          }

          if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only" ) ) {
            
                {
                  
                    rowwise_log_sum = y1_log_prob[c].rowwise().sum();
                    log_abs_prev_grad_array_col_for_each_n   =    log_prob_n_recip + rowwise_log_sum ;
                    sign_prev_grad_array_col_for_each_n.setOnes(); //// just a vector of +1's since probs are always positive
                    
                    // final scalar grad using log-sum-exp
                    LogSumVecSingedResult log_sum_vec_signed_struct = log_sum_vec_signed_v1( log_abs_prev_grad_array_col_for_each_n,
                                                                                             sign_prev_grad_array_col_for_each_n,
                                                                                             vect_type);
                    
                    prev_grad_vec(c)  +=   stan::math::exp(log_sum_vec_signed_struct.log_sum) * log_sum_vec_signed_struct.sign;
                  
                }
            
          }
          
      }
      
    }
    
   }

    ////////////////////////  --------------------------------------------------------------------------
    if (n_class > 1) {
      for (int c = 0; c < n_class; c++) {
        prev_unconstrained_grad_vec(c)  =   prev_grad_vec(c) * deriv_p_wrt_pu_double ;
      }
      prev_unconstrained_grad_vec(0) = prev_unconstrained_grad_vec(1) - prev_unconstrained_grad_vec(0) - 2 * tanh_u_prev[1];
      prev_unconstrained_grad_vec_out(0) = prev_unconstrained_grad_vec(0);
    }
    
    log_prob_out += out_mat.segment(1 + n_params, N).sum();       // log_prob_out += log_lik.sum();
    if (exclude_priors == false)  log_prob_out += prior_densities;
    log_prob_out +=  log_jac_u;
    log_prob_out +=  log_jac_p_double;

    int i = 0; // probs_all_range.prod() cancels out
    for (int c = 0; c < n_class; c++) {
      for (int t = 0; t < n_tests; t++) {
        beta_grad_vec(i) = beta_grad_array[c](0, t);
        i += 1;
      }
    }

    Eigen::Matrix<double, -1, 1>  bs_grad_vec_nd =  (grad_pi_wrt_b_raw.row(0).transpose().array() * bs_nd_double.array()).matrix() ; //     ( deriv_log_pi_wrt_L_Omega[0].asDiagonal().diagonal().array() * bs_nd_double.array()  ).matrix()  ; //  Jacobian_d_L_Sigma_wrt_b_matrix[0].transpose() * deriv_log_pi_wrt_L_Omega_vec_nd;
    Eigen::Matrix<double, -1, 1>  bs_grad_vec_d =   (grad_pi_wrt_b_raw.row(1).transpose().array() * bs_d_double.array()).matrix() ; //    ( deriv_log_pi_wrt_L_Omega[1].asDiagonal().diagonal().array() * bs_d_double.array()  ).matrix()  ; //   Jacobian_d_L_Sigma_wrt_b_matrix[1].transpose()  * deriv_log_pi_wrt_L_Omega_vec_d;

    Eigen::Matrix<double, -1, 1>   bs_grad_vec(n_bs_LT);
    bs_grad_vec.head(n_tests)              = bs_grad_vec_nd ;
    bs_grad_vec.segment(n_tests, n_tests)  = bs_grad_vec_d;

    
    {   ////////////////////////////  outputs // add log grad and sign stuff';///////////////
      
      out_mat(0) =  log_prob_out;
      out_mat.segment(1 + n_us, n_bs_LT)  += bs_grad_vec ;
      out_mat.segment(1 + n_us + n_bs_LT, n_coeffs) += beta_grad_vec;
      out_mat(1 + n_us + n_bs_LT + n_coeffs)  =  ( grad_prev_AD +  prev_unconstrained_grad_vec_out(0) );
      
    }
    
    
  }


    
    // int LT_cnt_2 = 0;
    // for (int c = 0; c < n_class; ++c) {
    //   for (int t = 0; t < n_tests; ++t) {
    //     if (LT_known_bs_indicator(c, t) == 1) {
    //       out_mat(1 + n_us + LT_cnt_2) = 0;
    //     }
    //     LT_cnt_2 += 1;
    //   }
    // }
    // //


}
   
 
 
 
 
 










// Internal function using Eigen::Ref as inputs for matrices
void     fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace(        Eigen::Matrix<double, -1, 1> &&out_mat_R_val,
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


   fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace_process(  out_mat_ref,
                                                           theta_main_vec_ref,
                                                           theta_us_vec_ref,
                                                           y_ref,
                                                           grad_option,
                                                           Model_args_as_cpp_struct);


}








// Internal function using Eigen::Ref as inputs for matrices
void     fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace(       Eigen::Matrix<double, -1, 1> &out_mat_ref,
                                                              const Eigen::Matrix<double, -1, 1> &theta_main_vec_ref,
                                                              const Eigen::Matrix<double, -1, 1> &theta_us_vec_ref,
                                                              const Eigen::Matrix<int, -1, -1> &y_ref,
                                                              const std::string &grad_option,
                                                              const Model_fn_args_struct &Model_args_as_cpp_struct




) {


  fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace_process(   out_mat_ref,
                                                           theta_main_vec_ref,
                                                           theta_us_vec_ref,
                                                           y_ref,
                                                           grad_option,
                                                           Model_args_as_cpp_struct);


}



// Internal function using Eigen::Ref as inputs for matrices
template <typename MatrixType>
void     fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace(       Eigen::Ref<Eigen::Block<MatrixType, -1, 1>>  out_mat_ref,
                                                                  const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  theta_main_vec_ref,
                                                                  const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  theta_us_vec_ref,
                                                                  const Eigen::Matrix<int, -1, -1> y_ref,
                                                                  const std::string &grad_option,
                                                                  const Model_fn_args_struct &Model_args_as_cpp_struct




) {


  fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace_process(   out_mat_ref,
                                                           theta_main_vec_ref,
                                                           theta_us_vec_ref,
                                                           y_ref,
                                                           grad_option,
                                                           Model_args_as_cpp_struct);


}














// Internal function using Eigen::Ref as inputs for matrices
Eigen::Matrix<double, -1, 1>    fn_lp_grad_LT_LC_PartialLog_MD_and_AD(           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
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

   fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace(   out_mat,
                                                    theta_main_vec_ref,
                                                    theta_us_vec_ref,
                                                    y_ref,
                                                    grad_option,
                                                    Model_args_as_cpp_struct);

  return out_mat;

}















