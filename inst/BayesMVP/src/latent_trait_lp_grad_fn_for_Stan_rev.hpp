#pragma once

#ifndef LATENT_TRAIT_LP_GRAD_FN_FOR_STAN_REV_HPP
#define LATENT_TRAIT_LP_GRAD_FN_FOR_STAN_REV_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/core.hpp>
#include <stan/math/prim/fun/Eigen.hpp> 

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta.hpp>

#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/meta.hpp>
 
#include <typeinfo>
#include <type_traits>
#include <sstream>
#include <stdexcept>
#include <stan/math/prim/err/invalid_argument.hpp>

#include <stan/math/prim/meta/is_var_matrix.hpp> 
 
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

 
#include "lp_grad_fn_for_Stan_prim.hpp"

 
 
 
 
 
  

//////////////  ---------  LC-MVP manual-gradient lp_grad function  --------------------------------------------------------------------------------------------------------------------


 
namespace stan {
namespace math {

 
 
 
double                                         Stan_wrapper_lp_fn_latent_trait_var(    const int Model_type_int,
                                                                                       const int force_autodiff_int,
                                                                                       const int force_PartialLog_int,
                                                                                       const int multi_attempts_int,
                                                                                       const Eigen::Matrix<var_value<double>, -1, 1> &theta_main_vec,
                                                                                       const Eigen::Matrix<var_value<double>, -1, 1> &theta_us_vec,
                                                                                       const Eigen::Matrix<int, -1, -1>  &y,
                                                                                       const int n_chunks,
                                                                                       const double overflow_threshold,
                                                                                       const double underflow_threshold,
                                                                                       const double prev_prior_a,
                                                                                       const double prev_prior_b, 
                                                                                       const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec,
                                                                                       const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_mean,
                                                                                       const std::vector<Eigen::Matrix<double, -1, -1>> &prior_coeffs_sd,
                                                                                       const Eigen::Matrix<double, -1, -1> &LT_b_priors_shape,
                                                                                       const Eigen::Matrix<double, -1, -1> &LT_b_priors_scale,
                                                                                       const Eigen::Matrix<double, -1, -1> &LT_known_bs_indicator,
                                                                                       const Eigen::Matrix<double, -1, -1> &LT_known_bs_values,
                                                                                       std::ostream* pstream__ = nullptr) {
   
   const int N = y.rows();
   const int n_nuisance = theta_us_vec.rows()  ; 
   const int n_params_main =  theta_main_vec.rows();
   const int n_params = n_params_main + n_nuisance;
   
   const int n_tests = y.cols(); 
   
   const std::string Model_type = "latent_trait";
   
   bool multi_attempts = true;
   if (multi_attempts_int == 0) multi_attempts = false;
   
   const std::string grad_option = "all";
   
   bool force_PartialLog = false;
   bool force_autodiff = false;
   
   if (force_autodiff_int == 1) {
     force_autodiff = true;
     force_PartialLog = true;
   } else if ((force_PartialLog_int == 1) && (force_autodiff_int == 0)) {
     force_autodiff = false;
     force_PartialLog = true;
   }
    
   int n_class = 2;
   if (Model_type == "MVP") n_class = 1;
   
   /////  --------  create Model_fn_args_struct object -------------------------------------------
   Model_fn_args_struct Model_args_as_cpp_struct(N, n_nuisance, n_params_main, 
                                                 15, 4, 4, 13,
                                                 1, 1, 1, 1, 
                                                 1, 1, 7, 1, 
                                                 1, 1, 1, 1);
   
   const int desired_n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
   const int vec_size = 8;
   ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, desired_n_chunks);
   int chunk_size = chunk_size_info.chunk_size;
   
   Model_args_as_cpp_struct.n_nuisance = n_nuisance;
   Model_args_as_cpp_struct.n_params_main = n_params_main;
   
   Model_args_as_cpp_struct.Model_args_bools(0) =  false;  // exclude_priors
   Model_args_as_cpp_struct.Model_args_bools(1) =  false;  // CI
   Model_args_as_cpp_struct.Model_args_bools(2) =  false;  // corr_force_positive
   Model_args_as_cpp_struct.Model_args_bools(3) =  false;  // corr_prior_beta
   Model_args_as_cpp_struct.Model_args_bools(4) =  false;  // corr_prior_norm
   Model_args_as_cpp_struct.Model_args_bools(5) =  true;   // handle_numerical_issues 
   Model_args_as_cpp_struct.Model_args_bools(6) =  false;  // skip_checks_exp
   Model_args_as_cpp_struct.Model_args_bools(7) =  false;  // skip_checks_log
   Model_args_as_cpp_struct.Model_args_bools(8) =  false;  // skip_checks_lse
   Model_args_as_cpp_struct.Model_args_bools(9) =  false;  // skip_checks_tanh
   Model_args_as_cpp_struct.Model_args_bools(10) = false;  // skip_checks_Phi
   Model_args_as_cpp_struct.Model_args_bools(11) = false;  // skip_checks_log_Phi
   Model_args_as_cpp_struct.Model_args_bools(12) = false;  // skip_checks_inv_Phi
   Model_args_as_cpp_struct.Model_args_bools(13) = false;  // skip_checks_inv_Phi_approx_from_logit_prob
   Model_args_as_cpp_struct.Model_args_bools(14) = false;  // debug
   
   Model_args_as_cpp_struct.Model_args_ints(0) = 1;  // n_cores
   Model_args_as_cpp_struct.Model_args_ints(1) = n_class;  // n_class
   Model_args_as_cpp_struct.Model_args_ints(2) = 5;  // ub_threshold_phi_approx
   Model_args_as_cpp_struct.Model_args_ints(3) = n_chunks;  // n_chunks
   
   Model_args_as_cpp_struct.Model_args_doubles(0) = prev_prior_a;
   Model_args_as_cpp_struct.Model_args_doubles(1) = prev_prior_b;
   Model_args_as_cpp_struct.Model_args_doubles(2) = overflow_threshold;
   Model_args_as_cpp_struct.Model_args_doubles(3) = underflow_threshold;
    
   const std::string Phi_type = "Phi";
   const std::string inv_Phi_type = "inv_Phi";
   const std::string nuisance_transformation = "Phi";
   
   const std::string vect_type = "Stan";  ////////////////  TEMP
   
   Model_args_as_cpp_struct.Model_args_strings(0) = vect_type; // vect_type
   Model_args_as_cpp_struct.Model_args_strings(1) = Phi_type; // Phi_type
   Model_args_as_cpp_struct.Model_args_strings(2) = inv_Phi_type; // inv_Phi_type
   Model_args_as_cpp_struct.Model_args_strings(3) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(4) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(5) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(6) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(7) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(8) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(9) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(10) = vect_type;
   Model_args_as_cpp_struct.Model_args_strings(12) = nuisance_transformation;
   
   // Model_args_as_cpp_struct.Model_args_col_vecs_double[0] = lkj_cholesky_eta;
   
   // Eigen::Matrix<int, -1, -1> n_covariates_per_outcome_vec_int = n_covariates_per_outcome_vec.cast<int>();
   Model_args_as_cpp_struct.Model_args_mats_int[0] = n_covariates_per_outcome_vec;
   
   Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0] = prior_coeffs_mean;
   Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1] = prior_coeffs_sd;
   
   // Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2] = prior_for_corr_a; //// For MVP / LC_MVP
   // Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3] = prior_for_corr_b; //// For MVP / LC_MVP
   // Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4] = lb_corr; //// For MVP / LC_MVP
   // Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5] = ub_corr; //// For MVP / LC_MVP
   // Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6] = known_values; //// For MVP / LC_MVP
   
   // std::vector<Eigen::Matrix<int, -1, -1>> known_values_indicator_int = vec_of_mats<int>(n_tests, n_tests, n_class); //// For MVP / LC_MVP
   // for (int c = 0; c < n_class; c++) {
   //   known_values_indicator_int[c] = known_values_indicator[0].cast<int>();
   // } 
   // Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0] = known_values_indicator_int;
   
   // Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0] = X; //// Note: latent_trait doesnt support covariates yet
   
   //// For latent-trait only (if not using latent_trait then these are dummy variables): 
   Model_args_as_cpp_struct.Model_args_mats_double[0] = LT_b_priors_shape;
   Model_args_as_cpp_struct.Model_args_mats_double[1] = LT_b_priors_scale;
   Model_args_as_cpp_struct.Model_args_mats_double[2] = LT_known_bs_indicator;
   Model_args_as_cpp_struct.Model_args_mats_double[3] = LT_known_bs_values;
   
   /////  --------  call lp_grad function  --------------------------------
   // stan::arena_t<Eigen::Matrix<double, -1, 1>> theta_main_double = value_of(theta_main_vec);
   // stan::arena_t<Eigen::Matrix<double, -1, 1>> theta_us_double =   value_of(theta_us_vec);
   // Eigen::Matrix<double, -1, 1> theta_main_double = value_of(theta_main_vec);
   // Eigen::Matrix<double, -1, 1> theta_us_double =   value_of(theta_us_vec);
   
   //stan::arena_t<Eigen::Matrix<double, -1, 1>> lp_grad_outs = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
   Eigen::Matrix<double, -1, 1> lp_grad_outs = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
   //stan::math::start_nested();
   fn_lp_grad_InPlace(   lp_grad_outs,  Model_type, 
                         force_autodiff, force_PartialLog, multi_attempts, 
                         value_of(theta_main_vec), value_of(theta_us_vec),
                         y, 
                         grad_option,
                         Model_args_as_cpp_struct);


   // set adjoints for nuisance
   for(int i = 0; i < n_nuisance; ++i) {
     theta_us_vec(i).adj() = lp_grad_outs(1 + i);
   }
   
 //  stan::math::set_zero_all_adjoints();
   
   // set adjoints for main
   for(int i = 0; i < n_params_main; ++i) {
     theta_main_vec(i).adj() = lp_grad_outs(1 + n_nuisance + i);
   }
   
 //  stan::math::set_zero_all_adjoints();
   
  // stan::math::recover_memory_nested();
   
   // Eigen::Matrix<double, -1, 1> log_lik = lp_grad_outs.tail(N);
   double log_posterior = lp_grad_outs(0);
   
   // Eigen::Matrix<double, -1, 1> outs(1 + N);
   // outs(0) = log_posterior;
   // outs.tail(N) = log_lik;
   
   return log_posterior;
   
   
}
 
 
 
 
 
 
 
  
  



}  // namespace math
}  // namespace stan


 








 
#endif
    
     

 


















