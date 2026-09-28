
#pragma once
 
 
   
 
#include <Eigen/Dense> 
 
 
 
 
 
 
using namespace Eigen;
 
  
 
 
 
#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
 
 
 
 
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



 
 


#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
 
 

// [[Rcpp::plugins(cpp17)]]
 


 
using namespace Eigen;


 


#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
 

using std_vec_of_EigenVecs_dbl = std::vector<Eigen::Matrix<double, -1, 1>>;
using std_vec_of_EigenVecs_int = std::vector<Eigen::Matrix<int, -1, 1>>;

using std_vec_of_EigenMats_dbl = std::vector<Eigen::Matrix<double, -1, -1>>;
using std_vec_of_EigenMats_int = std::vector<Eigen::Matrix<int, -1, -1>>;

using two_layer_std_vec_of_EigenVecs_dbl =  std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>;
using two_layer_std_vec_of_EigenVecs_int = std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>;

using two_layer_std_vec_of_EigenMats_dbl = std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>;
using two_layer_std_vec_of_EigenMats_int = std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>;


using three_layer_std_vec_of_EigenVecs_dbl =  std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>>;
using three_layer_std_vec_of_EigenVecs_int =  std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>>;

using three_layer_std_vec_of_EigenMats_dbl = std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>>; 
using three_layer_std_vec_of_EigenMats_int = std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>>;
 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  
  

inline Eigen::Matrix<int, 3, -1>  comb2M_double( Eigen::Ref<Eigen::Matrix<int, -1, 1>>   x) {
  int xlen = x.rows();
  int counter = 0;
  int outsize = xlen * (xlen - 1) / 2 ; // Rf_choose(xlen, 2);
  Eigen::Matrix<int, 3, -1>  out(3, outsize);
  for (int a = 0; a < xlen; ++a) {
    for (int b = a+1; b < xlen; ++b) {
      out(0, counter) = x(a);
      out(1, counter) = x(b);
      if ( x(b) > x(a) )  {
        out(2, counter) = x(b) -  x(a);
      } else {
        out(2, counter) = x(a) -  x(b);
      }
      ++counter;
    }
  }
  return out;
}










Eigen::Matrix<double, -1, 1>     fn_diag_hessian_us_only_manual(      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                      const Model_fn_args_struct &Model_args_as_cpp_struct

) {

  
  
  //// important params
  const int N = y_ref.rows();
  const int n_tests = y_ref.cols();
  const int n_us = theta_us_vec_ref.rows()  ; 
  const int n_params_main =  theta_main_vec_ref.rows()  ; 
  const int n_params = n_params_main + n_us;
  
  //////////////  access elements from struct 
  const Eigen::Matrix<bool, -1, 1>   &Model_args_bools              = Model_args_as_cpp_struct.Model_args_bools;  
  const Eigen::Matrix<int, -1, 1>    &Model_args_ints               = Model_args_as_cpp_struct.Model_args_ints;  
  const Eigen::Matrix<double, -1, 1> &Model_args_doubles            = Model_args_as_cpp_struct.Model_args_doubles;   
  const Eigen::Matrix<std::string, -1, 1>  &Model_args_strings            = Model_args_as_cpp_struct.Model_args_strings;  
  
  const std_vec_of_EigenVecs_dbl  &Model_args_col_vecs_double     = Model_args_as_cpp_struct.Model_args_col_vecs_double;   
  const std_vec_of_EigenMats_dbl  &Model_args_mats_double     = Model_args_as_cpp_struct.Model_args_mats_double;  
  
  const two_layer_std_vec_of_EigenMats_dbl &Model_args_vecs_of_mats_double = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double;  
  const two_layer_std_vec_of_EigenMats_int &Model_args_vecs_of_mats_int    = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int;  
  const two_layer_std_vec_of_EigenVecs_int &Model_args_vecs_of_col_vecs_int= Model_args_as_cpp_struct.Model_args_vecs_of_col_vecs_int;  
  
  const three_layer_std_vec_of_EigenMats_dbl &Model_args_2_later_vecs_of_mats_double  = Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double;  
  
  /////////// read items from std::vectors
  const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>  &X = Model_args_2_later_vecs_of_mats_double[0]; 
  
  const bool &exclude_priors = Model_args_bools(0);
  const bool &CI =             Model_args_bools(1);
  const bool &corr_force_positive = Model_args_bools(2);
  const bool &corr_prior_beta = Model_args_bools(3);
  const bool &corr_prior_norm = Model_args_bools(4);
  const bool &handle_numerical_issues = Model_args_bools(5);
  const bool &skip_checks_exp =   Model_args_bools(6);
  const bool &skip_checks_log =   Model_args_bools(7);
  const bool &skip_checks_lse =   Model_args_bools(8);
  const bool &skip_checks_tanh =  Model_args_bools(9);
  const bool &skip_checks_Phi =  Model_args_bools(10);
  const bool &skip_checks_log_Phi = Model_args_bools(11);
  const bool &skip_checks_inv_Phi = Model_args_bools(12);
  const bool &skip_checks_inv_Phi_approx_from_logit_prob = Model_args_bools(13);
  
  const int &n_cores = Model_args_ints(0);
  const int &n_class = Model_args_ints(1);
  const int &ub_threshold_phi_approx = Model_args_ints(2);
  const int &n_chunks = Model_args_ints(3);
  
  const double &prev_prior_a = Model_args_doubles(0);
  const double &prev_prior_b = Model_args_doubles(1);
  const double &overflow_threshold = Model_args_doubles(2);
  const double &underflow_threshold = Model_args_doubles(3);
  
  const std::string &vect_type = Model_args_strings(0);
  const std::string &Phi_type = Model_args_strings(1);
  const std::string &inv_Phi_type = Model_args_strings(2);
  const std::string &vect_type_exp = Model_args_strings(3);
  const std::string &vect_type_log = Model_args_strings(4);
  const std::string &vect_type_lse = Model_args_strings(5);
  const std::string &vect_type_tanh = Model_args_strings(6);
  const std::string &vect_type_Phi = Model_args_strings(7);
  const std::string &vect_type_log_Phi = Model_args_strings(8);
  const std::string &vect_type_inv_Phi = Model_args_strings(9);
  const std::string &vect_type_inv_Phi_approx_from_logit_prob = Model_args_strings(10);
  // const std::string &grad_option =  Model_args_strings(11);
  const std::string &nuisance_transformation =   Model_args_strings(12);
  
  const Eigen::Matrix<double, -1, 1>  &lkj_cholesky_eta =  Model_args_col_vecs_double[0];
  
  const Eigen::Matrix<double, -1, -1> &LT_b_priors_shape  = Model_args_mats_double[0]; 
  const Eigen::Matrix<double, -1, -1> &LT_b_priors_scale  = Model_args_mats_double[1]; 
  const Eigen::Matrix<double, -1, -1> &LT_known_bs_indicator = Model_args_mats_double[2]; 
  const Eigen::Matrix<double, -1, -1> &LT_known_bs_values = Model_args_mats_double[3]; 
  
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &prior_coeffs_mean  = Model_args_vecs_of_mats_double[0]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &prior_coeffs_sd   =  Model_args_vecs_of_mats_double[1]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &prior_for_corr_a   = Model_args_vecs_of_mats_double[2]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &prior_for_corr_b   = Model_args_vecs_of_mats_double[3]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &lb_corr   = Model_args_vecs_of_mats_double[4]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &ub_corr   = Model_args_vecs_of_mats_double[5]; 
  const std::vector<Eigen::Matrix<double, -1, -1 > >   &known_values    = Model_args_vecs_of_mats_double[6]; 
  
  const std::vector<Eigen::Matrix<int, -1, -1 >> &known_values_indicator = Model_args_vecs_of_mats_int[0];
  
  const std::vector<Eigen::Matrix<int, -1, 1 >> &n_covariates_per_outcome_vec = Model_args_vecs_of_col_vecs_int[0];
   
  //////////////
  const int n_corrs =  n_class * n_tests * (n_tests - 1) * 0.5;
  
  const int n_covariates_total_nd = n_covariates_per_outcome_vec[0].sum();
  const int n_covariates_total_d = n_covariates_per_outcome_vec[1].sum();
  const int n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
  
  const int n_covariates_max_nd = n_covariates_per_outcome_vec[0].maxCoeff();
  const int n_covariates_max_d = n_covariates_per_outcome_vec[1].maxCoeff();
  const int n_covariates_max = std::max(n_covariates_max_nd, n_covariates_max_d);
  
  const double sqrt_2_pi_recip = 1.0 / sqrt(2.0 * M_PI);
  const double sqrt_2_recip = 1.0 / stan::math::sqrt(2.0);
  const double minus_sqrt_2_recip = -sqrt_2_recip;
  const double a = 0.07056;
  const double b = 1.5976;
  const double a_times_3 = 3.0 * 0.07056;
  const double s = 1.0 / 1.702;
  
  const int chunk_size = std::floor(N / n_chunks);
  const int N_divisible_by_chunk_size = std::floor(N / chunk_size) * chunk_size;
  

  const bool hessian = true;

 
  
  ////// Nuisance parameter transformation step
  const Eigen::Matrix<double, -1, 1> u_unc_vec = theta_us_vec_ref;
  const Eigen::Matrix<double, -1, 1> u_vec =  fn_MVP_compute_nuisance(   u_unc_vec,
                                                                         nuisance_transformation,
                                                                         vect_type_Phi,
                                                                         vect_type_log,
                                                                         vect_type_tanh);
  
  const double log_jac_u =    fn_MVP_compute_nuisance_log_jac_u(   u_vec,
                                                                   u_unc_vec,
                                                                   nuisance_transformation,
                                                                   vect_type_Phi,
                                                                   vect_type_log,
                                                                   vect_type_tanh,
                                                                   skip_checks_log);
  
  const Eigen::Matrix<double, -1, 1 >  du_wrt_duu =         fn_MVP_nuisance_first_deriv(   u_vec,
                                                                                           u_unc_vec,
                                                                                           nuisance_transformation,
                                                                                           vect_type_exp);
  
  const Eigen::Matrix<double, -1, 1 >  d_J_wrt_duu =    fn_MVP_nuisance_deriv_of_log_det_J(  u_vec,
                                                                                             u_unc_vec,
                                                                                             nuisance_transformation,
                                                                                             du_wrt_duu);

  //////////////
  // corrs
  Eigen::Matrix<double, -1, 1  >  Omega_raw_vec_double = theta_main_vec_ref.head(n_corrs); // .cast<double>();

  // coeffs
  std::vector<Eigen::Matrix<double, -1, -1 > > beta_double_array = vec_of_mats(n_covariates_max, n_tests,  n_class);

  {
    int i = n_corrs;
    for (int c = 0; c < n_class; ++c) {
      for (int t = 0; t < n_tests; ++t) {
        for (int k = 0; k < n_covariates_per_outcome_vec[c](t); ++k) {
          beta_double_array[c](k, t) = theta_main_vec_ref(i);
          i += 1;
        }
      }
    }
  }

  // prev
  double u_prev_diseased = theta_main_vec_ref(n_params_main - 1);

  Eigen::Matrix<double, -1, 1 >  target_AD_grad(n_corrs);
  stan::math::var target_AD = 0.0;
  double grad_prev_AD = 0.0;

  int dim_choose_2 = n_tests * (n_tests - 1) * 0.5 ;
  std::vector<Eigen::Matrix<double, -1, -1 > > deriv_L_wrt_unc_full = vec_of_mats(dim_choose_2 + n_tests, dim_choose_2, n_class);
  std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_double = vec_of_mats(n_tests, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1 > > L_Omega_recip_double = L_Omega_double;
  std::vector<Eigen::Matrix<double, -1, -1 > > log_abs_L_Omega_double = L_Omega_double;
  std::vector<Eigen::Matrix<double, -1, -1 > > sign_L_Omega_double = L_Omega_double;

  ///////////////// beginning of local autodiff block  ----------------------------------------------------------------------------------------------
  {  
    stan::math::start_nested();
    
  Eigen::Matrix<stan::math::var, -1, 1  >  Omega_raw_vec_var =  stan::math::to_var(Omega_raw_vec_double) ;
  Eigen::Matrix<stan::math::var, -1, 1  >  Omega_constrained_raw_vec_var =  Eigen::Matrix<stan::math::var, -1, 1  >::Zero(n_corrs) ;
  Omega_constrained_raw_vec_var = Omega_raw_vec_var ; // no transformation for Nump needed! done later on
  
  {
    
    
    std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > Omega_unconstrained_var = fn_convert_std_vec_of_corrs_to_3d_array_var(Eigen_vec_to_std_vec_var(Omega_constrained_raw_vec_var),  n_tests, n_class);
    std::vector<Eigen::Matrix<stan::math::var, -1, -1 > > L_Omega_var = vec_of_mats_var(n_tests, n_tests, n_class);
    std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_var   = vec_of_mats_var(n_tests, n_tests, n_class);

    for (int c = 0; c < n_class; ++c) {
      Eigen::Matrix<stan::math::var, -1, -1 >  ub = stan::math::to_var(ub_corr[c]);
      Eigen::Matrix<stan::math::var, -1, -1 >  lb = stan::math::to_var(lb_corr[c]);
      Eigen::Matrix<stan::math::var, -1, -1  >  Chol_Schur_outs =  Pinkney_LDL_bounds_opt(n_tests, lb, ub, Omega_unconstrained_var[c], known_values_indicator[c], known_values[c]) ;
      L_Omega_var[c]   =  Chol_Schur_outs.block(1, 0, n_tests, n_tests);
      Omega_var[c] =   L_Omega_var[c] * L_Omega_var[c].transpose() ;
      target_AD +=   Chol_Schur_outs(0, 0); // now can set prior directly on Omega
    }

    for (int c = 0; c < n_class; ++c) {
      if ( (corr_prior_beta == false)   &&  (corr_prior_norm == false) ) {
        target_AD +=  stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c], lkj_cholesky_eta(c)) ;
      } else if ( (corr_prior_beta == true)   &&  (corr_prior_norm == false) ) {
        for (int i = 1; i < n_tests; i++) {
          for (int j = 0; j < i; j++) {
            target_AD +=  stan::math::beta_lpdf(  (Omega_var[c](i, j) + 1)/2, prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
          }
        }
        //  Jacobian for  Omega -> L_Omega transformation for prior log-densities (since both LKJ and truncated normal prior densities are in terms of Omega, not L_Omega)
        Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
        for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i+1) ) * log(L_Omega_var[c](i, i));
        target_AD  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
      } else if  ( (corr_prior_beta == false)   &&  (corr_prior_norm == true) ) {
        for (int i = 1; i < n_tests; i++) {
          for (int j = 0; j < i; j++) {
            target_AD +=  stan::math::normal_lpdf(  Omega_var[c](i, j), prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
          }
        }
        Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
        for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i+1) ) * log(L_Omega_var[c](i, i));
        target_AD  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
      }
    }

    ///////////////////////
    stan::math::set_zero_all_adjoints();
    target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
    target_AD_grad =  Omega_raw_vec_var.adj();    // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
    stan::math::set_zero_all_adjoints();
    //////////////////////////////////////////////////////////// end of AD part

    /////////////  prev stuff  ---- vars
    std::vector<stan::math::var> 	 u_prev_var_vec_var(n_class, 0.0);
    std::vector<stan::math::var> 	 prev_var_vec_var(n_class, 0.0);
    std::vector<stan::math::var> 	 tanh_u_prev_var(n_class, 0.0);
    Eigen::Matrix<stan::math::var, -1, -1>	 prev_var(1, n_class);

    u_prev_var_vec_var[1] =  stan::math::to_var(u_prev_diseased);
    tanh_u_prev_var[1] = ( stan::math::exp(2.0*u_prev_var_vec_var[1] ) - 1.0) / ( stan::math::exp(2*u_prev_var_vec_var[1] ) + 1.0) ;
    u_prev_var_vec_var[0] =   0.5 *  stan::math::log( (1.0 + ( (1.0 - 0.5 * ( tanh_u_prev_var[1] + 1.0))*2.0 - 1.0) ) / (1.0 - ( (1.0 - 0.5 * ( tanh_u_prev_var[1] + 1.0))*2.0 - 1.0) ) )  ;
    tanh_u_prev_var[0] = (stan::math::exp(2.0*u_prev_var_vec_var[0] ) - 1.0) / ( stan::math::exp(2*u_prev_var_vec_var[0] ) + 1.0) ;

    prev_var_vec_var[1] =  0.5 * ( tanh_u_prev_var[1] + 1.0);
    prev_var_vec_var[0] =  0.5 * ( tanh_u_prev_var[0] + 1.0);
    prev_var(0,1) =  prev_var_vec_var[1];
    prev_var(0,0) =  prev_var_vec_var[0];

    stan::math::var tanh_pu_deriv_var = ( 1.0 - (tanh_u_prev_var[1] * tanh_u_prev_var[1])  );
    stan::math::var deriv_p_wrt_pu_var = 0.5 *  tanh_pu_deriv_var;
    stan::math::var tanh_pu_second_deriv_var  = -2.0 * tanh_u_prev_var[1]  * tanh_pu_deriv_var;
    stan::math::var log_jac_p_deriv_wrt_pu_var  = ( 1.0 / deriv_p_wrt_pu_var) * 0.5 * tanh_pu_second_deriv_var; // for gradient of u's
    stan::math::var log_jac_p_var =    stan::math::log( deriv_p_wrt_pu_var );

    stan::math::var  target_AD_prev = beta_lpdf(prev_var(0, 1), prev_prior_a, prev_prior_b)  ;// + log_jac_p_var ; // weakly informative prior - helps avoid boundaries with slight negative skew (for lower N)
    target_AD  +=  target_AD_prev;
    ///////////////////////
    target_AD_prev.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
    grad_prev_AD  =  u_prev_var_vec_var[1].adj() - u_prev_var_vec_var[0].adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
    stan::math::set_zero_all_adjoints();
    //////////////////////////////////////////////////////////// end of AD part

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
          stan::math::set_zero_all_adjoints();
          cnt_1 += 1;
        }
      }
    }
    ///////////////// get cholesky factor's (lower-triangular) of corr matrices
    // convert to 3d var array
    for (int c = 0; c < n_class; ++c) {
      for (int t1 = 0; t1 < n_tests; ++t1) {
        for (int t2 = 0; t2 < n_tests; ++t2) {
          L_Omega_double[c](t1, t2) =   L_Omega_var[c](t1, t2).val()  ;
          log_abs_L_Omega_double[c](t1, t2) =   stan::math::log(stan::math::fabs( L_Omega_double[c](t1, t2) ))  ;
          sign_L_Omega_double[c](t1, t2) = stan::math::sign( L_Omega_double[c](t1, t2) );
          L_Omega_recip_double[c](t1, t2) =   1.0 / L_Omega_double[c](t1, t2) ;
        }
      }
    }
    
  }
  
  //  stan::math::recover_memory();
  stan::math::recover_memory_nested();
  
  }

  /////////////  prev stuff
  std::vector<double> 	 u_prev_var_vec(n_class, 0.0);
  std::vector<double> 	 prev_var_vec(n_class, 0.0);
  std::vector<double> 	 tanh_u_prev(n_class, 0.0);
  Eigen::Matrix<double, -1, -1>	 prev(1, n_class);

  u_prev_var_vec[1] =  (double) u_prev_diseased ;
  tanh_u_prev[1] = ( exp(2.0*u_prev_var_vec[1] ) - 1.0) / ( exp(2.0*u_prev_var_vec[1] ) + 1.0) ;
  u_prev_var_vec[0] =   0.5 *  log( (1.0 + ( (1.0 - 0.5 * ( tanh_u_prev[1] + 1.0))*2.0 - 1.0) ) / (1.0 - ( (1.0 - 0.5 * ( tanh_u_prev[1] + 1.0))*2.0 - 1.0) ) )  ;
  tanh_u_prev[0] = (exp(2.0*u_prev_var_vec[0] ) - 1.0) / ( exp(2.0*u_prev_var_vec[0] ) + 1.0) ;

  prev_var_vec[1] =  0.5 * ( tanh_u_prev[1] + 1.0);
  prev_var_vec[0] =  0.5 * ( tanh_u_prev[0] + 1.0);
  prev(0,1) =  prev_var_vec[1];
  prev(0,0) =  prev_var_vec[0];

  double tanh_pu_deriv = ( 1.0 - (tanh_u_prev[1] * tanh_u_prev[1])  );
  double deriv_p_wrt_pu_double = 0.5 *  tanh_pu_deriv;
  double tanh_pu_second_deriv  = -2.0 * tanh_u_prev[1]  * tanh_pu_deriv;
  double log_jac_p_deriv_wrt_pu  = ( 1.0 / deriv_p_wrt_pu_double) * 0.5 * tanh_pu_second_deriv; // for gradient of u's
  double log_jac_p =    log( deriv_p_wrt_pu_double );

  ///////////////////////////////////////////////////////////////////////// prior densities
  double prior_densities = 0.0;

  if (exclude_priors == false) {
    ///////////////////// priors for coeffs
    double prior_densities_coeffs = 0.0;
    for (int c = 0; c < n_class; c++) {
      for (int t = 0; t < n_tests; t++) {
        for (int k = 0; k < n_covariates_per_outcome_vec[c](t); k++) {
          prior_densities_coeffs  += stan::math::normal_lpdf(beta_double_array[c](k, t), prior_coeffs_mean[c](k, t), prior_coeffs_sd[c](k, t));
        }
      }
    }
    double prior_densities_corrs = target_AD.val();
    prior_densities = prior_densities_coeffs  +      prior_densities_corrs ;     // total prior densities and Jacobian adjustments
  }







  /////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////// likelihood
  double log_prob_out = 0.0;
  double log_prob = 0.0;

  Eigen::Matrix<double, -1, -1 >  log_prev = prev;
  for (int c = 0; c < n_class; c++) {
    log_prev(0,c) =  log(prev(0,c));
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////// likelihood
  Eigen::Matrix<double, -1, -1 >  u_array(N, n_tests);
  Eigen::Matrix<double, -1, -1 >   y_sign(N, n_tests);

  std::vector<Eigen::Matrix<double, -1, -1 > >  prob = vec_of_mats(N, n_tests, 2);
  std::vector<Eigen::Matrix<double, -1, -1 > >  Z_std_norm =  prob;
  std::vector<Eigen::Matrix<double, -1, -1 > >  Phi_Z =  prob;
  std::vector<Eigen::Matrix<double, -1, -1 > >  Bound_Z =  prob;
  std::vector<Eigen::Matrix<double, -1, -1 > >  Bound_U_Phi_Bound_Z =  prob;

  Eigen::Matrix<double, -1, 1 >  prob_n(N);
  Eigen::Matrix<double, -1, 1>	 lp(n_class); // colvec container
  Eigen::Matrix<double, -1, 1>	 inc(n_class); // col vec  container
  Eigen::Matrix<double, -1, -1>	 y1(n_tests, n_class); // container
  Eigen::Matrix<double, -1, -1>	 Z_std_norm_var(n_tests, n_class); // container
  Eigen::Matrix<double, -1, 1 >  log_posterior = prob_n;


  int i = 0;
  for (int t = 0; t < n_tests; t++) {
    for (int n = 0; n < N; n++ ) {
      u_array(n, t) =  u_vec(i);
      i += 1;
    }
  }

  for (int n = 0; n < N; n++ ) {

    for (int c = 0; c < n_class; c++) {
      inc(c, 0) = 0;
    }


    for (int t = 0; t < n_tests; t++) {

      int index = y_ref(n, t);

      if (index == 1) {
        y_sign(n,t) = +1;
      } else {
        y_sign(n,t) = -1;
      }

      for (int c = 0; c < n_class; c++) {

        if (CI == true) {
          L_Omega_double[c](t,t) = 1;
          inc(c, 0)  = 0;
        }

        Bound_Z[c](n,t) =   ( ((  0 - ( beta_double_array[c](0, t) + inc(c, 0)   )  ) / L_Omega_double[c](t, t) )  );
        Bound_U_Phi_Bound_Z[c](n,t) =   stan::math::Phi_approx(   Bound_Z[c](n, t)  );

        if (index == 1) {
          Phi_Z[c](n,t) = Bound_U_Phi_Bound_Z[c](n,t) + (1 - Bound_U_Phi_Bound_Z[c](n,t)) * u_array(n, t);
          y1(t,c) =  log1p(-Bound_U_Phi_Bound_Z[c](n,t));  // log(1 - Bound_U_Phi_Bound_Z[c](n,t));
          prob[c](n,t) =  1 - Bound_U_Phi_Bound_Z[c](n,t);
          Z_std_norm_var(t, c) = stan::math::inv_Phi(Phi_Z[c](n,t)) ;
          Z_std_norm[c](n, t) =   Z_std_norm_var(t, c);
        } else { // y == 0
          Phi_Z[c](n,t) =  Bound_U_Phi_Bound_Z[c](n,t) * u_array(n, t);
          y1(t,c) =  log(Bound_U_Phi_Bound_Z[c](n,t));
          prob[c](n,t) =  Bound_U_Phi_Bound_Z[c](n,t);
          Z_std_norm_var(t, c) =   stan::math::inv_Phi(Phi_Z[c](n,t)) ;
          Z_std_norm[c](n, t) =   Z_std_norm_var(t, c);
        }


        if (CI == false) {
          if (t < n_tests - 1) {
            inc(c, 0)  = (L_Omega_double[c].row(t+1).head(t+1) * Z_std_norm_var.col(c).head(t+1)).eval()(0,0);
          }
        }


      } // end of c loop

    } // end of t loop


    for (int c = 0; c < n_class; c++) {
      lp(c) = y1.colwise().sum().eval()(0,c)  + log_prev(0,c);
    }

    log_posterior(n) =  stan::math::log_sum_exp(lp);


  } // end of n loop

  prob_n =    log_posterior.array().exp().matrix();

  log_prob_out = log_posterior.sum();


  log_prob_out += prior_densities;
  log_prob_out +=  log_jac_u;
  log_prob_out +=  log_jac_p;




  log_prob = log_prob_out;


  std::vector<Eigen::Matrix<double, -1, -1 > >  phi_Z =  prob;
  std::vector<Eigen::Matrix<double, -1, -1 > >  phi_Bound_Z = prob;
  std::vector<Eigen::Matrix<double, -1, -1 > >  deriv_phi_Z =  prob;
  std::vector<Eigen::Matrix<double, -1, -1 > >  deriv_phi_Bound_Z =  prob;
  std::vector<Eigen::Matrix<double, -1, -1 > >  y_m_y_sign_x_u =  prob;

  for (int c = 0; c < n_class; c++) {

    {
      for (int t = 0; t < n_tests; t++) {
        for (int n = 0; n < N; n++ ) {
          phi_Z[c](n, t)  =                       exp(stan::math::normal_lpdf(Z_std_norm[c](n, t) , 0, 1)) ;
          phi_Bound_Z[c](n, t)        =           exp(stan::math::normal_lpdf(Bound_Z[c](n, t) , 0, 1)) ;
          deriv_phi_Z[c](n, t)     =             -  phi_Z[c](n, t)   * Z_std_norm[c](n, t)  ;
          deriv_phi_Bound_Z[c](n, t)    =        -     phi_Bound_Z[c](n, t)  * Bound_Z[c](n, t)  ;
        }
      }
    }

    y_m_y_sign_x_u[c].array()    =           y_ref.array() - y_sign.array() * u_array.array();
  }

  /////////////////////////////////////////////////////////////////////////////////////// Manual gradients
  std::vector<Eigen::Matrix<double, -1, -1 > > common_grad_term_1 =     vec_of_mats(N, n_tests, 2);
  std::vector<Eigen::Matrix<double, -1, -1 > > common_grad_term_numerator =    common_grad_term_1;
  std::vector<Eigen::Matrix<double, -1, -1 > > common_grad_term_denominator  =    common_grad_term_1;




  for (int n = 0; n < N; n++ ) {

    for (int c = 0; c < n_class; c++) {

      double prev_div_prob = (prev(0,c)/ prob_n(n));

      for (int i = 0; i < n_tests; i++) { // i goes from 1 to 3

        int t = n_tests - (i + 1) ;
        common_grad_term_1[c](n, t) =   prev_div_prob * (prob[c].row(n).prod()/  prob[c].row(n).segment(t + 0, i + 1).prod()  );
        common_grad_term_numerator[c](n, t) =   prev(0,c) * ( prob[c].row(n).prod() /  prob[c].row(n).segment(t + 0, i + 1).prod()  );
        common_grad_term_denominator[c](n, t) =    prob_n(n);

      }

    }

  }


  ////////////////////////////////////////////////////////////////////////// /  Grad of nuisanc`e parameters / u's (manual)
  Eigen::Matrix<double, -1, -1 > u_grad_array  = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //  (N, n_tests); // initialise to zero's
  Eigen::Matrix<double, -1, -1 > u_grad_wrt_constrained_wo_jac_array   = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //  (N, n_tests); // initialise to zero's
  Eigen::Matrix<double, -1, -1 > u_grad_array_term_1 = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //  (N, n_tests); // initialise to zero's
  Eigen::Matrix<double, -1, -1 > u_grad_array_term_2 = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //  (N, n_tests); // initialise to zero's
  Eigen::Matrix<double , -1, 1>  u_grad  = Eigen::Matrix<double, -1, 1>::Zero(n_us);   //  (n_us);

  Eigen::Matrix<double , -1, -1> derivs_chain_container_vec_array  = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //  (N, n_tests);

  // terms for first-order grad
  Eigen::Matrix<double,  -1, -1> grad_prob  = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //  (N, n_tests);
  Eigen::Matrix<double , -1, -1> grad_Phi_bound_z = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //   =  grad_prob;
  Eigen::Matrix<double , -1, -1> grad_bound_z  = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //  = grad_prob;
  Eigen::Matrix<double , -1, -1> grad_z  = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //  = grad_prob;

  // terms for second-order grad
  Eigen::Matrix<double , -1, -1> grad_grad_Phi_bound_z  = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //   = grad_prob;
  Eigen::Matrix<double , -1, -1> grad_grad_z = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //   = grad_prob;
  Eigen::Matrix<double , -1, -1> grad_grad_bound_z = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //   = grad_prob;
  Eigen::Matrix<double , -1, -1> grad_grad_prob  = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //   = grad_prob;
  Eigen::Matrix<double , -1, -1> grad_phi_z  = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //   = grad_prob;
  Eigen::Matrix<double , -1, -1> grad_phi_z_recip  = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //   = grad_prob;
  Eigen::Matrix<double , -1, -1> grad_phi_bound_z  = Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //   = grad_prob;

  Eigen::Matrix< double , -1, 1>  term_A  = Eigen::Matrix<double, -1, 1>::Zero(N);   //  (N) ;
  Eigen::Matrix< double , -1, 1>  term_B = Eigen::Matrix<double, -1, 1>::Zero(N);   // = term_A;
  Eigen::Matrix< double , -1, 1>  term_C = Eigen::Matrix<double, -1, 1>::Zero(N);   // = term_A;
  Eigen::Matrix< double , -1, 1>  grad_term_A = Eigen::Matrix<double, -1, 1>::Zero(N);   // = term_A;
  Eigen::Matrix< double , -1, 1>  grad_term_B = Eigen::Matrix<double, -1, 1>::Zero(N);   // = term_A;

  Eigen::Matrix< double , -1, 1>  temp_L_Omega_x_grad_z_sum_1 = Eigen::Matrix<double, -1, 1>::Zero(N);   // = term_A;
  Eigen::Matrix< double , -1, 1>  temp_L_Omega_x_grad_grad_z_sum_1 = Eigen::Matrix<double, -1, 1>::Zero(N);   // = term_A;

  Eigen::Matrix<double, -1, -1> grad_term_A_components_with_grad_grad(N, n_tests*n_tests -   (n_tests * (n_tests - 1) / 2) );
  Eigen::Matrix<double, -1, -1> grad_term_A_components_without_grad_grad(N, n_tests*n_tests -   (n_tests * (n_tests - 1) / 2) );

  Eigen::Matrix<double, -1, -1 > u_hessian_diag_array_wrt_constrained_wo_jac =   Eigen::Matrix<double, -1, -1>::Zero(N, n_tests);   //  (N, n_tests);


  {

    for (int c = 0; c < n_class; c++) {

      ///// last term first (test T)
      {
        int t = n_tests - 1;

        for (int n = 0; n < N; n++ ) {
          u_grad_wrt_constrained_wo_jac_array(n,  n_tests - 1)   =   0.0 ;
          u_hessian_diag_array_wrt_constrained_wo_jac(n,  n_tests - 1)    = 0.0 ;
        }

      }

      ///// then second-to-last term (test T - 1)
      {
        int t = n_tests - 2;

        grad_z.col(0) =  ( 1 / phi_Z[c].col(t).array() ) *  prob[c].col(t).array() ;
        temp_L_Omega_x_grad_z_sum_1 =    L_Omega_double[c](t + 1, t)   * grad_z.col(0).array() ;
        grad_bound_z.col(0) =   (  1 / L_Omega_double[c](t + 1, t + 1)  ) *  ( -  temp_L_Omega_x_grad_z_sum_1.array()  ) ;
        grad_Phi_bound_z.col(0) =  phi_Bound_Z[c].col(t + 1).array() * grad_bound_z.col(0).array();
        grad_prob.col(0)   =   (  - y_sign.col(t + 1).array()  )  * grad_Phi_bound_z.col(0).array();

        term_A =     grad_prob.col(0) ;
        u_grad_wrt_constrained_wo_jac_array.col( n_tests - 2).array()  +=   (  common_grad_term_1[c].col(t + 1).array()   *    term_A.array()   ).array() ;

        if (hessian == true) {

          grad_phi_z.col(0) = grad_z.col(0).array()  * deriv_phi_Z[c].col(t).array();   // correct
          grad_phi_z_recip.col(0) =    - ( 1 / ( phi_Z[c].col(t).array() * phi_Z[c].col(t).array() ) ) * grad_phi_z.col(0).array() ;   // correct
          grad_grad_z.col(0) = grad_phi_z_recip.col(0).array()  *  (    prob[c].col(t).array()   )  ;  // correct

          temp_L_Omega_x_grad_grad_z_sum_1 =    L_Omega_double[c](t + 1, t)   * grad_grad_z.col(0).array() ;  // correct
          grad_grad_bound_z.col(0) =  (  1 / L_Omega_double[c](t + 1, t + 1)  ) * ( -  temp_L_Omega_x_grad_grad_z_sum_1.array()  ) ;   // correct
          grad_phi_bound_z.col(0) = deriv_phi_Bound_Z[c].col(t + 1);
          grad_grad_Phi_bound_z.col(0) = deriv_phi_Bound_Z[c].col(t + 1).array() *   grad_bound_z.col(0).array() * grad_bound_z.col(0).array()  + phi_Bound_Z[c].col(t + 1).array() *    grad_grad_bound_z.col(0).array() ;  // wrong
          grad_grad_prob.col(0) =   (  - y_sign.col(t + 1).array()  )  * grad_grad_Phi_bound_z.col(0).array();

          grad_term_A_components_with_grad_grad.col(0)  =  grad_grad_prob.col(0) ;

          grad_term_A.array() =  grad_term_A_components_with_grad_grad.col(0).array() ;
          term_B.array() =  1 / common_grad_term_denominator[c].col(t + 1).array() ;
          term_C.array() =  common_grad_term_numerator[c].col(t + 1).array();
          grad_term_B.array() =   - (  term_B.array() * term_B.array()  * term_C.array()  * term_A.array()   ).array() ;

          u_hessian_diag_array_wrt_constrained_wo_jac.col(n_tests - 2).array() += (term_C.array() * ( grad_term_B.array() * term_A.array()    +     term_B.array() * grad_term_A.array() ).array() ).array() ;

        }

      }

      ///// then third-to-last term
      {
        int t = n_tests - 3;

        // 1st set of grad_z and grad_prob terms
        grad_z.col(0) =  ( 1 / phi_Z[c].col(t).array() ) *  prob[c].col(t).array() ;
        temp_L_Omega_x_grad_z_sum_1 =    L_Omega_double[c](t + 1, t)   * grad_z.col(0).array() ;
        grad_bound_z.col(0) =   (  1 / L_Omega_double[c](t + 1, t + 1)  ) *  ( -  temp_L_Omega_x_grad_z_sum_1.array()  ) ;
        grad_Phi_bound_z.col(0) =  phi_Bound_Z[c].col(t + 1).array() * grad_bound_z.col(0).array();
        grad_prob.col(0)   =   (  - y_sign.col(t + 1).array()  )  * grad_Phi_bound_z.col(0).array();


        // 2nd set of grad_z and grad_prob terms
        grad_z.col(1) =  ( 1 / phi_Z[c].col(t + 1).array() ) *     y_m_y_sign_x_u[c].col(t + 1).array() *    grad_Phi_bound_z.col(0).array() ;
        temp_L_Omega_x_grad_z_sum_1 =   L_Omega_double[c](t + 2, t)   * grad_z.col(0).array()   +  L_Omega_double[c](t + 2, t + 1)   * grad_z.col(1).array()    ;
        grad_bound_z.col(1) =   (  1 / L_Omega_double[c](t + 2, t + 2)  ) *  ( -  temp_L_Omega_x_grad_z_sum_1.array()  ) ;
        grad_Phi_bound_z.col(1) =  phi_Bound_Z[c].col(t + 2).array() * grad_bound_z.col(1).array();
        grad_prob.col(1)   =   (  - y_sign.col(t + 2).array()  )  * grad_Phi_bound_z.col(1).array();


        term_A.array() =  (  grad_prob.col(1).array()  * prob[c].col(t + 1).array()  +      grad_prob.col(0).array()  *  prob[c].col(t + 2).array() )   ;
        u_grad_wrt_constrained_wo_jac_array.col(n_tests - 3).array()  +=       (common_grad_term_1[c].col(t + 1).array()   *  term_A.array() ).array() ;


        if (hessian == true)  {

          // 1st set of grad_grad_z and grad_grad_prob terms
          grad_phi_z.col(0) = grad_z.col(0).array()  * deriv_phi_Z[c].col(t).array();   // correct
          grad_phi_z_recip.col(0) =    - ( 1 / ( phi_Z[c].col(t).array() * phi_Z[c].col(t).array() ) ) * grad_phi_z.col(0).array() ;   // correct
          grad_grad_z.col(0) = grad_phi_z_recip.col(0).array()  *  (    prob[c].col(t).array()   )  ;  // correct

          temp_L_Omega_x_grad_grad_z_sum_1 =    L_Omega_double[c](t + 1, t)   * grad_grad_z.col(0).array() ;  // correct
          grad_grad_bound_z.col(0) =  (  1 / L_Omega_double[c](t + 1, t + 1)  ) * ( -  temp_L_Omega_x_grad_grad_z_sum_1.array()  ) ;   // correct
          grad_phi_bound_z.col(0) = deriv_phi_Bound_Z[c].col(t + 1);
          grad_grad_Phi_bound_z.col(0) = deriv_phi_Bound_Z[c].col(t + 1).array() *   grad_bound_z.col(0).array() * grad_bound_z.col(0).array()  + phi_Bound_Z[c].col(t + 1).array() *    grad_grad_bound_z.col(0).array() ;
          grad_grad_prob.col(0) =   (  - y_sign.col(t + 1).array()  )  * grad_grad_Phi_bound_z.col(0).array();

          // 2nd set of grad_grad_z and grad_grad_prob terms
          grad_phi_z.col(1) = grad_z.col(1).array()  * deriv_phi_Z[c].col(t + 1).array();
          grad_phi_z_recip.col(1) =    - ( 1 / ( phi_Z[c].col(t + 1).array() * phi_Z[c].col(t + 1).array() ) ) * grad_phi_z.col(1).array() ;

          grad_grad_z.col(1) =   y_m_y_sign_x_u[c].col(t + 1).array()  * (  grad_phi_z_recip.col(1).array() *    grad_Phi_bound_z.col(0).array()  +  ( 1 /   phi_Z[c].col(t + 1).array()   ) *  grad_grad_Phi_bound_z.col(0).array()    ) ; // new

          temp_L_Omega_x_grad_grad_z_sum_1 =    L_Omega_double[c](t + 2, t)   * grad_grad_z.col(0).array()  +    L_Omega_double[c](t + 2, t + 1)   * grad_grad_z.col(1).array()   ;
          grad_grad_bound_z.col(1) =  (  1 / L_Omega_double[c](t + 2, t + 2)  ) * ( -  temp_L_Omega_x_grad_grad_z_sum_1.array()  ) ;   // correct
          grad_phi_bound_z.col(1) = deriv_phi_Bound_Z[c].col(t + 2);
          grad_grad_Phi_bound_z.col(1) = deriv_phi_Bound_Z[c].col(t + 2).array() *   grad_bound_z.col(1).array() * grad_bound_z.col(1).array()  + phi_Bound_Z[c].col(t + 2).array() *    grad_grad_bound_z.col(1).array() ;  // wrong
          grad_grad_prob.col(1) =   (  - y_sign.col(t + 2).array()  )  * grad_grad_Phi_bound_z.col(1).array();



          int   grad_term_A_n_components_with_grad_grad = 2 ;  // ---------- checked
          int   grad_term_A_n_components_without_grad_grad =  1;   // ---------- checked

          grad_term_A_components_with_grad_grad.col(0).array()   = grad_grad_prob.col(0).array()  * prob[c].col(t + 2).array(); // ---------- checked
          grad_term_A_components_with_grad_grad.col(1).array()   = grad_grad_prob.col(1).array()  * prob[c].col(t + 1).array();  // ---------- checked

          grad_term_A_components_without_grad_grad.col(0).array()  = grad_prob.col(0).array()   * grad_prob.col(1).array() ; // ---------- checked

          grad_term_A =  grad_term_A_components_with_grad_grad.block(0, 0, N, 2).rowwise().sum() +   2 * grad_term_A_components_without_grad_grad.col(0) ;  // ---------- checked

          term_B =   (  1 / prob_n.array() ).matrix() ;   // ---------- checked
          term_C =  common_grad_term_numerator[c].col(t + 1); // ---------- checked
          grad_term_B =   - (  term_B.array() * term_B.array()  * term_C.array()  * term_A.array()   ).matrix() ; // ---------- checked

          u_hessian_diag_array_wrt_constrained_wo_jac.col(n_tests - 3).array() += (term_C.array() * ( grad_term_B.array() * term_A.array()    +     term_B.array() * grad_term_A.array() ).array() ).array().array() ;

        }


      }

      ///// then fourth-to-last term
      {

        for (int i = 1; i < n_tests - 2; i++ ) {

          int t = n_tests - (i + 3);

          // 1st set of grad_z and grad_prob terms
          grad_z.col(0) =  ( 1 / phi_Z[c].col(t).array() ) *  prob[c].col(t).array() ;
          temp_L_Omega_x_grad_z_sum_1 =    L_Omega_double[c](t + 1, t)   * grad_z.col(0).array() ;
          grad_bound_z.col(0) =   (  1 / L_Omega_double[c](t + 1, t + 1)  ) *  ( -  temp_L_Omega_x_grad_z_sum_1.array()  ) ;
          grad_Phi_bound_z.col(0) =  phi_Bound_Z[c].col(t + 1).array() * grad_bound_z.col(0).array();
          grad_prob.col(0)   =   (  - y_sign.col(t + 1).array()  )  * grad_Phi_bound_z.col(0).array();


          for (int ii = 1; ii < i + 2; ii++ ) {
            grad_z.col(ii) =  ( 1 / phi_Z[c].col(t + ii).array() ) *     y_m_y_sign_x_u[c].col(t + ii).array() *    grad_Phi_bound_z.col(ii - 1).array() ;
            temp_L_Omega_x_grad_z_sum_1 =  grad_z.block(0, 0, N, ii + 1)  *  L_Omega_double[c].row(t + ii + 1).segment(t, ii + 1).transpose()   ;
            grad_bound_z.col(ii) =   (  1 / L_Omega_double[c](t + ii + 1, t + ii + 1)  ) *  ( -  temp_L_Omega_x_grad_z_sum_1.array()  ) ;
            grad_Phi_bound_z.col(ii) =  phi_Bound_Z[c].col(t + ii + 1).array() * grad_bound_z.col(ii).array();
            grad_prob.col(ii)   =   (  - y_sign.col(t + ii + 1).array()  )  * grad_Phi_bound_z.col(ii).array();
          }


          // ///// attempt at vectorising  // bookmark
          for (int ii = 0; ii < i + 2; ii++) {
            derivs_chain_container_vec_array.col(ii)  =  ( grad_prob.col(ii).array()  * (    prob[c].block(0, t + 1, N, i + 2).rowwise().prod().array() /  prob[c].col(t + ii + 1).array()  ).array() ).matrix() ;
          }

          term_A = derivs_chain_container_vec_array.block(0, 0, N, i + 2).rowwise().sum();
          u_grad_wrt_constrained_wo_jac_array.col(n_tests - (i + 3)).array()   +=       ( common_grad_term_1[c].col(t + 1).array()   * term_A.array()  ).array()   ;




          if (hessian == true)  {

            // 1st set of grad_grad_z and grad_grad_prob terms
            grad_phi_z.col(0) = grad_z.col(0).array()  * deriv_phi_Z[c].col(t).array();   // correct
            grad_phi_z_recip.col(0) =    - ( 1 / ( phi_Z[c].col(t).array() * phi_Z[c].col(t).array() ) ) * grad_phi_z.col(0).array() ;   // correct
            grad_grad_z.col(0) = grad_phi_z_recip.col(0).array()  *  (    prob[c].col(t).array()   )  ;  // correct

            temp_L_Omega_x_grad_grad_z_sum_1 =    L_Omega_double[c](t + 1, t)   * grad_grad_z.col(0).array() ;  // correct
            grad_grad_bound_z.col(0) =  (  1 / L_Omega_double[c](t + 1, t + 1)  ) * ( -  temp_L_Omega_x_grad_grad_z_sum_1.array()  ) ;   // correct
            grad_phi_bound_z.col(0) = deriv_phi_Bound_Z[c].col(t + 1);
            grad_grad_Phi_bound_z.col(0) = deriv_phi_Bound_Z[c].col(t + 1).array() *   grad_bound_z.col(0).array() * grad_bound_z.col(0).array()  + phi_Bound_Z[c].col(t + 1).array() *    grad_grad_bound_z.col(0).array() ;
            grad_grad_prob.col(0) =   (  - y_sign.col(t + 1).array()  )  * grad_grad_Phi_bound_z.col(0).array();

            for (int ii = 1; ii < i + 2; ii++ ) {
              grad_phi_z.col(ii) = grad_z.col(ii).array()  * deriv_phi_Z[c].col(t + ii).array();     // CORRECT
              grad_phi_z_recip.col(ii) =    - ( 1 / ( phi_Z[c].col(t + ii).array() * phi_Z[c].col(t + ii).array() ) ) * grad_phi_z.col(ii).array() ;     // CORRECT

              grad_grad_z.col(ii) =   y_m_y_sign_x_u[c].col(t + ii).array()  * (  grad_phi_z_recip.col(ii).array() *    grad_Phi_bound_z.col(ii - 1).array()  +  ( 1 /   phi_Z[c].col(t + ii).array()   ) *  grad_grad_Phi_bound_z.col(ii - 1).array()    ) ;  // CORRECT
              temp_L_Omega_x_grad_grad_z_sum_1 =  grad_grad_z.block(0, 0, N, ii + 1)  *  L_Omega_double[c].row(t + ii + 1).segment(t, ii + 1).transpose()   ;    // CORRECT
              grad_grad_bound_z.col(ii) =  (  1 / L_Omega_double[c](t + ii + 1, t + ii + 1)  ) * ( -  temp_L_Omega_x_grad_grad_z_sum_1.array()  ) ;     // CORRECT
              grad_phi_bound_z.col(ii) = deriv_phi_Bound_Z[c].col(t + ii + 1);    // CORRECT
              grad_grad_Phi_bound_z.col(ii) = deriv_phi_Bound_Z[c].col(t + ii + 1).array() *   grad_bound_z.col(ii).array() * grad_bound_z.col(ii).array()  + phi_Bound_Z[c].col(t + ii + 1).array() *    grad_grad_bound_z.col(ii).array() ;     // CORRECT
              grad_grad_prob.col(ii) =   (  - y_sign.col(t + ii + 1).array()  )  * grad_grad_Phi_bound_z.col(ii).array();    // CORRECT
            }



            // create containers outside n loop
            int   n_elements  =  i + 2;
            int   grad_term_A_n_components_total =  (n_elements * n_elements) -   (n_elements * (n_elements - 1) / 2);
            int   grad_term_A_n_components_with_grad_grad = n_elements;
            int   grad_term_A_n_components_without_grad_grad =  grad_term_A_n_components_total - n_elements;

            int nn = i + 2;
            int g = 2;

            Eigen::Matrix<int, -1, 1>  nn_vec(nn);
            for (int ii = 0; ii < nn ; ii++) {
              nn_vec(ii) = ii + 1;
            }

            Eigen::Matrix<int, -1, -1>  m =  comb2M_double(nn_vec);
            Eigen::Matrix<int, -1, 1>  mm(g * m.cols());


            t = n_tests - (i + 3) ;


            int mm_counter = 0;
            for (int jj = 0; jj < m.cols() ; jj++) {
              for (int ii = 0; ii < 2 ; ii++) {
                mm(mm_counter) = m(ii, jj);
                mm_counter += 1;
              }
            }


            Eigen::Matrix<double, -1, -1>  mat( g * m.cols(), nn );
            int nrows_mat = mat.rows();
            int ncols_mat = mat.cols();

            Eigen::Matrix<int, -1, -1>  indexes(nrows_mat, 2);


            for (int ii = 0; ii < nrows_mat; ii++) {
              indexes(ii, 0) = ii + 1;
              indexes(ii, 1) = mm(ii);
            }

            for (int ii = 0; ii < nrows_mat; ii++) {
              mat(indexes(ii, 0) - 1, indexes(ii, 1) - 1) = 1;
            }

            Eigen::Matrix<int, -1, -1>  new_indexes = indexes;
            for (int ii = 0; ii < nrows_mat - 1; ii++) {
              new_indexes(ii, 1) =   new_indexes(ii + 1, 1);
            }

            Eigen::Matrix<double, -1, -1>  new_mat =  mat;
            for (int ii = 0; ii < nrows_mat; ii++) {
              new_mat(new_indexes(ii, 0) - 1, new_indexes(ii, 1) - 1) = 1;
            }

            Eigen::Matrix<double, -1, -1>  new_new_mat(nrows_mat/2, ncols_mat);
            for (int ii = 1; ii < new_new_mat.rows() + 1;  ii++) {
              new_new_mat.row(ii - 1) = new_mat.row(2*(ii - 1) + 1 - 1);
            }



            {

              for (int ii = 0; ii < i + 2; ii++) {
                grad_term_A_components_with_grad_grad.col(ii)  =  ( grad_grad_prob.col(ii).array()  * (    prob[c].block(0, t + 1, N, i + 2).rowwise().prod().array() /  prob[c].col(t + ii + 1).array()  ).array() ).matrix() ;
              }


              for (int n = 0; n < N; n++ ) {

                for (int ii = 0; ii < grad_term_A_n_components_without_grad_grad ; ii++) {

                  double     temp_grad_prob_prod =  1;
                  double     temp_prob_prod =  1;

                  for (int jj = 0; jj < new_new_mat.cols();  jj++) {
                    if (new_new_mat(ii, jj) == 1) {
                      temp_grad_prob_prod = temp_grad_prob_prod * grad_prob(n, jj);
                    } else {
                      temp_prob_prod = temp_prob_prod *  prob[c](n, t + jj + 1) ;
                    }
                  }

                  grad_term_A_components_without_grad_grad(n, ii) = temp_grad_prob_prod * temp_prob_prod;

                }

              }


              grad_term_A.array() = grad_term_A_components_with_grad_grad.block(0, 0, N, grad_term_A_n_components_with_grad_grad).rowwise().sum().array() + 2 * grad_term_A_components_without_grad_grad.block(0, 0, N, grad_term_A_n_components_without_grad_grad).rowwise().sum().array() ;
              term_B.array()   =  1 / common_grad_term_denominator[c].col(t + 1).array();
              term_C.array()  =  common_grad_term_numerator[c].col(t + 1).array();
              grad_term_B.array()  =   - (  term_B.array()  * term_B.array()   * term_C.array()   * term_A.array()    );
              u_hessian_diag_array_wrt_constrained_wo_jac.col(n_tests - (i + 3)).array() += (term_C.array() * ( grad_term_B.array() * term_A.array()    +     term_B.array() * grad_term_A.array() ).array() ).array().array() ;

            }


          }

        } // end of i loop


      }



    } // end of c loop






  } // end of if "grad_main == TRUE"





  Eigen::Matrix<double , -1, 1>  deriv_d_J_wrt_duu =   Eigen::Matrix<double, -1, 1>::Zero(n_us);
  Eigen::Matrix<double , -1, 1>  deriv_du_wrt_duu =    Eigen::Matrix<double, -1, 1>::Zero(n_us);
  Eigen::Matrix<double , -1, 1>  u_hessian_diag =      Eigen::Matrix<double, -1, 1>::Zero(n_us);


  i = 0; // probs_all_range.prod() cancels out
  for (int t = 0; t < n_tests; t++) {
    for (int n = 0; n < N; n++ ) {

      /////// grad
      u_grad_array(n, t)    = u_grad_array(n, t) * du_wrt_duu(i)  + d_J_wrt_duu(i) ;
      u_grad(i)    = u_grad_array(n, t) ;

      /////// Hessian diagonal
      if (nuisance_transformation == "Phi") { // checked
        deriv_du_wrt_duu(i) =  - u_unc_vec(i) * du_wrt_duu(i) ;  // checked
        deriv_d_J_wrt_duu(i) =  -1.0;    // checked
      } else if (nuisance_transformation == "Phi_approx") {

        double u_unc_sq = u_unc_vec(i) * u_unc_vec(i) ;  // checked
        deriv_du_wrt_duu(i) =    ( 6.0 * a * u_unc_vec(i) * (1.0 - u_vec(i)) ) ;  // checked
        deriv_du_wrt_duu(i) += (3.0*a*u_unc_sq + b)*(1.0 - 2.0*u_vec(i))*du_wrt_duu(i) ;   // checked

        double term_A = -36.0*a*a*u_unc_sq / (3.0*a*u_unc_sq + b) ;  // checked  (sort of - check again)
        double term_C = (-1.0 / (u_vec(i)*(1.0-u_vec(i))) ) * (1.0 + ( std::pow( (1.0-2.0*u_vec(i)), 2.0)/(u_vec(i)*(1.0-u_vec(i))) ) ) ;  // checked  (sort of - check again)
        double term_B =  deriv_du_wrt_duu(i) * ( (1.0 - 2.0*u_vec(i))/(u_vec(i)*(1.0-u_vec(i))) )   +   du_wrt_duu(i) * term_C ;  // checked  (sort of - check again)

        deriv_d_J_wrt_duu(i) = term_A + term_B ;  // checked  (sort of - check again)

      } else if (nuisance_transformation == "Phi_approx_rough") {    // checked
        deriv_du_wrt_duu(i) =   (1.702*1.702) * u_vec(i) * (1.0 - u_vec(i)) * (1.0 - 2.0*u_vec(i)) ;  // checked
        deriv_d_J_wrt_duu(i) =   -2.0 * (1.702*1.702) * u_vec(i) * (1.0 - u_vec(i)) ; // checked
      } else if (nuisance_transformation == "tanh") {    // checked
        deriv_du_wrt_duu(i) = 4.0 * u_vec(i) * (1.0 - u_vec(i)) * (1.0 - 2.0*u_vec(i)) ;   // checked
        deriv_d_J_wrt_duu(i) =  - 8.0 * u_vec(i) * (1.0 - u_vec(i)) ; // checked
      }

      // put it all together
      u_hessian_diag(i) =      u_hessian_diag_array_wrt_constrained_wo_jac(n, t)  * du_wrt_duu(i)    +  u_grad_wrt_constrained_wo_jac_array(n, t)  * deriv_du_wrt_duu(i)   ;
      u_hessian_diag(i) +=     deriv_d_J_wrt_duu(i)  ;

      i += 1;

    }
  }



  //////////////// output
  return(u_hessian_diag);



}




//
//
//
//
//
//
//














 
              
              
              
              



