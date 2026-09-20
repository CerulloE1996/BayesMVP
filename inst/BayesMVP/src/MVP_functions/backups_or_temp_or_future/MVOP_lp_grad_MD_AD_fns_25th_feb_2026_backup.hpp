
#pragma once


 
 


#include <Eigen/Dense>
 
#include <unsupported/Eigen/SpecialFunctions>


 
 
// #include <stan/math.hpp>
// #include <vector>
// #include <cmath>


 
// ============================================================================
// fn_MVOP_ordinal_helpers.hpp
// 
// C++ helper functions for ordinal model components.
// Includes: construct_C, cutpoint Jacobian, induced Dirichlet prior (AD),
//           cumul_probs_to_ord_probs
// ============================================================================




// ============================================================================
// construct_C: Convert unconstrained C_raw to ordered cutpoints
// 
// C[0] = C_raw[0]  (unconstrained)
// C[k] = C[k-1] + exp(C_raw[k])  for k >= 1
//
// This guarantees C[0] < C[1] < ... < C[K-2] (strict ordering)
//
// Input:  C_raw_vec of length n_thr (= K - 1 cutpoints)
// Output: C_vec of length n_thr (ordered cutpoints)
// ============================================================================
ALWAYS_INLINE Eigen::Matrix<double, -1, 1> construct_C_double( const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> C_raw_vec) {
   
       const int n_thr = C_raw_vec.size();
       Eigen::Matrix<double, -1, 1> C_vec(n_thr);
       
       C_vec(0) = C_raw_vec(0);
       for (int k = 1; k < n_thr; ++k) {
         C_vec(k) = C_vec(k - 1) + std::exp(C_raw_vec(k));
       }
       
       return C_vec;
   
}
 

// ============================================================================
// construct_C Jacobian: log|det(J)| for C_raw -> C transform
//
// The Jacobian is lower-triangular with:
//   J[0,0] = 1
//   J[k,k] = exp(C_raw[k]) for k >= 1
//   (off-diagonal entries also exist but det only needs diagonal for triangular)
//
// log|det(J)| = sum_{k=1}^{n_thr-1} C_raw[k]
// ============================================================================
ALWAYS_INLINE double construct_C_log_det_J( const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> C_raw_vec) {
   
       const int n_thr = C_raw_vec.size();
       double log_det_J = 0.0;
       
       for (int k = 1; k < n_thr; ++k) {
         log_det_J += C_raw_vec(k);  // = log(exp(C_raw[k]))
       }
       
       return log_det_J;
       
}


// ============================================================================
// Gradient of log|det(J)| w.r.t. C_raw
//
// ∂log|det(J)|/∂C_raw[0] = 0
// ∂log|det(J)|/∂C_raw[k] = 1 for k >= 1
// ============================================================================
ALWAYS_INLINE Eigen::Matrix<double, -1, 1> construct_C_log_det_J_grad( const int n_thr) {
   
       Eigen::Matrix<double, -1, 1> grad = Eigen::Matrix<double, -1, 1>::Ones(n_thr);
       grad(0) = 0.0;
       return grad;
   
}
 

// // ============================================================================
// // Gradient of C w.r.t. C_raw (needed for chain rule: ∂f/∂C_raw = ∂f/∂C * ∂C/∂C_raw)
// //
// // ∂C[j]/∂C_raw[k]:
// //   - If k == 0: ∂C[j]/∂C_raw[0] = 1 for all j >= 0  (shifting all cutpoints)
// //   - If k >= 1: ∂C[j]/∂C_raw[k] = exp(C_raw[k]) for j >= k, else 0
// //
// // This is a lower-triangular matrix. For efficiency, we don't form it explicitly
// // but provide a function to compute ∂f/∂C_raw given ∂f/∂C.
// //
// // ∂f/∂C_raw[0] = Σ_j ∂f/∂C[j]
// // ∂f/∂C_raw[k] = exp(C_raw[k]) * Σ_{j>=k} ∂f/∂C[j]    for k >= 1
// // ============================================================================
// ALWAYS_INLINE Eigen::Matrix<double, -1, 1> chain_rule_C_to_C_raw( const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> &grad_wrt_C,
//                                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> &C_raw_vec
// ) {
//    
//        const int n_thr = C_raw_vec.size();
//        Eigen::Matrix<double, -1, 1> grad_wrt_C_raw(n_thr);
//        
//        // Compute suffix sums: suffix_sum[k] = Σ_{j=k}^{n_thr-1} grad_wrt_C[j]
//        // Do this from right to left
//        double suffix_sum = 0.0;
//        
//        for (int k = n_thr - 1; k >= 1; --k) {
//          suffix_sum += grad_wrt_C(k);
//          grad_wrt_C_raw(k) = std::exp(C_raw_vec(k)) * suffix_sum;
//        }
//        
//        suffix_sum += grad_wrt_C(0);
//        grad_wrt_C_raw(0) = suffix_sum;  // ∂f/∂C_raw[0] = Σ_j ∂f/∂C[j]
//        
//        return grad_wrt_C_raw;
//    
//  }
//  
//  
// // ============================================================================
// // cumul_probs_to_ord_probs: Convert cumulative probs to ordinal category probs
// //
// // Input: cumul_probs of length n_thr (Φ(C_1), Φ(C_2), ..., Φ(C_{K-1}))
// // Output: ord_probs of length n_cat = n_thr + 1
// //
// // ord_probs[0] = cumul_probs[0]
// // ord_probs[k] = cumul_probs[k] - cumul_probs[k-1]  for 1 <= k < n_thr
// // ord_probs[n_cat-1] = 1 - cumul_probs[n_thr-1]
// // ============================================================================
// ALWAYS_INLINE Eigen::Matrix<double, -1, 1> cumul_probs_to_ord_probs( const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> cumul_probs) {
//    
//        const int n_thr = cumul_probs.size();
//        const int n_cat = n_thr + 1;
//        Eigen::Matrix<double, -1, 1> ord_probs(n_cat);
//        
//        ord_probs(0) = cumul_probs(0);
//        for (int k = 1; k < n_thr; ++k) {
//          ord_probs(k) = cumul_probs(k) - cumul_probs(k - 1);
//        }
//        ord_probs(n_cat - 1) = 1.0 - cumul_probs(n_thr - 1);
//        
//        return ord_probs;
//    
// }
//  
 
// // ============================================================================
// // induced_dirichlet_given_C_lpdf (double version, for prior density value)
// //
// // log p(p_ord | C, alpha, probit) = dirichlet_lpdf(p_ord | alpha) + Σ_k log(φ(C_k))
// //                                    (for probit link)
// //
// // where φ is std_normal_pdf for probit, or logistic PDF for logit.
// // We use probit link here (consistent with MVP probit model).
// // ============================================================================
// ALWAYS_INLINE double induced_dirichlet_given_C_lpdf_double( const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> C_vec,
//                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> alpha,  // length n_cat
//                                                             const bool use_probit_link
// ) {
//        
//        const int n_thr = C_vec.size();
//        const int n_cat = n_thr + 1;
//        
//        // Compute ordinal probabilities from cutpoints
//        Eigen::Matrix<double, -1, 1> cumul_probs(n_thr);
//        for (int k = 0; k < n_thr; ++k) {
//          if (use_probit_link) {
//            cumul_probs(k) = stan::math::Phi(C_vec(k));
//          } else {
//            cumul_probs(k) = stan::math::inv_logit(C_vec(k));
//          }
//        }
//        Eigen::Matrix<double, -1, 1> p_ord = cumul_probs_to_ord_probs(cumul_probs);
//        
//        // Dirichlet log-density
//        double log_prob = 0.0;
//        double alpha_sum = alpha.sum();
//        log_prob += std::lgamma(alpha_sum);
//        for (int k = 0; k < n_cat; ++k) {
//          log_prob -= std::lgamma(alpha(k));
//          log_prob += (alpha(k) - 1.0) * std::log(p_ord(k));
//        }
//        
//        // Jacobian: log|det(∂p/∂C)| = Σ_k log(φ_link(C_k))
//        for (int k = 0; k < n_thr; ++k) {
//          if (use_probit_link) {
//            log_prob += stan::math::std_normal_lpdf(C_vec(k));  // log(φ(C_k))
//          } else {
//            log_prob += stan::math::log_inv_logit(C_vec(k)) 
//            + stan::math::log1m_inv_logit(C_vec(k));
//          }
//        }
//        
//        return log_prob;
// 
// }


// // ============================================================================
// // induced_dirichlet_prior_AD: Compute prior density AND gradient w.r.t. C_raw
// // using Stan math autodiff.
// //
// // This wraps the induced Dirichlet + construct_C transform in a single AD block.
// // Not performance-critical (called once per iteration, not per observation).
// //
// // Returns: {log_prior_density, gradient w.r.t. C_raw}
// // ============================================================================
// ALWAYS_INLINE std::pair<double, Eigen::Matrix<double, -1, 1>> induced_dirichlet_prior_AD(  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> C_raw_vec_double,
//                                                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> alpha, // Dirichlet concentration (length n_cat)
//                                                                                            const bool use_probit_link
// ) {
//    
//        const int n_thr = C_raw_vec_double.size();
//        
//        stan::math::start_nested();
//        
//        // Convert to var
//        Eigen::Matrix<double, -1, 1> C_raw_concrete = C_raw_vec_double;  // copy from Ref
//        Eigen::Matrix<stan::math::var, -1, 1> C_raw_var = stan::math::to_var(C_raw_concrete);
//        // Eigen::Matrix<stan::math::var, -1, 1> C_raw_var = stan::math::to_var(C_raw_vec_double);
//        
//        // Construct ordered cutpoints
//        Eigen::Matrix<stan::math::var, -1, 1> C_var(n_thr);
//        C_var(0) = C_raw_var(0);
//        for (int k = 1; k < n_thr; ++k) {
//          C_var(k) = C_var(k - 1) + stan::math::exp(C_raw_var(k));
//        }
//        
//        // Jacobian for C_raw -> C transform
//        stan::math::var log_det_J_C = 0.0;
//        for (int k = 1; k < n_thr; ++k) {
//          log_det_J_C += C_raw_var(k);
//        }
//        
//        // Compute cumulative probs
//        const int n_cat = n_thr + 1;
//        Eigen::Matrix<stan::math::var, -1, 1> cumul_probs(n_thr);
//        for (int k = 0; k < n_thr; ++k) {
//          if (use_probit_link) {
//            cumul_probs(k) = stan::math::Phi(C_var(k));
//          } else {
//            cumul_probs(k) = stan::math::inv_logit(C_var(k));
//          }
//        }
//        
//        // Ordinal probs
//        Eigen::Matrix<stan::math::var, -1, 1> p_ord(n_cat);
//        p_ord(0) = cumul_probs(0);
//        for (int k = 1; k < n_thr; ++k) {
//          p_ord(k) = cumul_probs(k) - cumul_probs(k - 1);
//        }
//        p_ord(n_cat - 1) = 1.0 - cumul_probs(n_thr - 1);
//        
//        // Dirichlet log-density
//        stan::math::var target = 0.0;
//        stan::math::var alpha_sum_var = 0.0;
//        for (int k = 0; k < n_cat; ++k) alpha_sum_var += stan::math::to_var(alpha(k));
//        target += stan::math::lgamma(alpha_sum_var);
//        for (int k = 0; k < n_cat; ++k) {
//          target -= stan::math::lgamma(stan::math::to_var(alpha(k)));
//          target += (stan::math::to_var(alpha(k)) - 1.0) * stan::math::log(p_ord(k));
//        }
//        
//        // Jacobian for C -> p_ord (induced Dirichlet part)
//        for (int k = 0; k < n_thr; ++k) {
//          if (use_probit_link) {
//            target += stan::math::std_normal_lpdf(C_var(k));
//          } else {
//            target += stan::math::log_inv_logit(C_var(k))
//            + stan::math::log1m_inv_logit(C_var(k));
//          }
//        }
//        
//        // Add C_raw -> C Jacobian
//        target += log_det_J_C;
//        
//        double log_prior = target.val();
//        target.grad();
//        
//        Eigen::Matrix<double, -1, 1> grad(n_thr);
//        for (int k = 0; k < n_thr; ++k) {
//          grad(k) = C_raw_var(k).adj();
//        }
//        
//        stan::math::recover_memory_nested();
//        
//        return {log_prior, grad};
//    
// }
//  
 
// ============================================================================
// For each observation i and ordinal test t, compute the lower and upper 
// cutpoint values based on y_i,t.
//
// Given:
//   y_val = observed category (1-indexed: 1, 2, ..., K_t)
//   C_vec = ordered cutpoints for this test (length K_t - 1)
//
// Returns: {lower_cutpoint, upper_cutpoint}
//   where lower = -∞ (represented as -1e30) if y_val == 1
//   and   upper = +∞ (represented as +1e30) if y_val == K_t
// ============================================================================
ALWAYS_INLINE std::pair<double, double> get_ordinal_bounds(  int y_val, 
                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> C_vec,
                                                             int K_t
) {
       // y_val is 1-indexed
       double lower = (y_val == 1)   ? -1e30 : C_vec(y_val - 2);  // C_{k-1}, 0-indexed
       double upper = (y_val == K_t) ? +1e30 : C_vec(y_val - 1);  // C_k, 0-indexed
       return {lower, upper};
   
}
 
 
// ============================================================================
// Vectorised: For a chunk of observations, compute lower and upper bounds
// for an ordinal test.
//
// y_col: integer column of y values for this test (1-indexed categories)
// C_vec: ordered cutpoints (length K_t - 1)
// K_t: number of categories for this test
//
// Outputs: lower_bounds, upper_bounds (both length chunk_size)
// ============================================================================
ALWAYS_INLINE void compute_ordinal_bounds_chunk( Eigen::Ref<Eigen::Matrix<double, -1, 1>> lower_bounds,     // output
                                                 Eigen::Ref<Eigen::Matrix<double, -1, 1>> upper_bounds,      // output
                                                 const Eigen::Ref<const Eigen::Matrix<int, -1, 1>> y_col,  // integer y values
                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> C_vec,  // ordered cutpoints
                                                 const int K_t,
                                                 const int chunk_size
) {
   
       const double NEG_INF = -1e30;
       const double POS_INF = +1e30;
       
       for (int i = 0; i < chunk_size; ++i) {
         const int k = y_col(i);  // 1-indexed category
         lower_bounds(i) = (k == 1)   ? NEG_INF : C_vec(k - 2);
         upper_bounds(i) = (k == K_t) ? POS_INF : C_vec(k - 1);
       }
       
}
 

// ============================================================================
// Double-precision version using double y_col (for when y_chunk is cast to double)
// ============================================================================
ALWAYS_INLINE void compute_ordinal_bounds_chunk(   Eigen::Ref<Eigen::Matrix<double, -1, 1>> lower_bounds,
                                                   Eigen::Ref<Eigen::Matrix<double, -1, 1>> upper_bounds,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> y_col_double,  // y as double
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> C_vec,
                                                   const int K_t,
                                                   const int chunk_size
) {
  
       const double NEG_INF = -1e30;
       const double POS_INF = +1e30;
       
       for (int i = 0; i < chunk_size; ++i) {
         const int k = static_cast<int>(y_col_double(i));  // 1-indexed
         lower_bounds(i) = (k == 1)   ? NEG_INF : C_vec(k - 2);
         upper_bounds(i) = (k == K_t) ? POS_INF : C_vec(k - 1);
       }
       
}

 


//////////////////////////////////////////////////----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

inline void fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_WCP( Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat ,
                                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                            const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                            const std::string &grad_option,
                                                                            const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                            std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                            const int n_threads_WCP
) {
  
 ////  std::cout << "MVOP: entered function" << std::endl; std::cout.flush();
  
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
  const std::string J_grad_option =  Model_args_as_cpp_struct.Model_args_strings(11); // for "num_diff" or "analytical" or "autodiff" (for complex Jacobian)
  const std::string nuisance_transformation =   Model_args_as_cpp_struct.Model_args_strings(12);

  ///// load vectors / matrices
  const Eigen::Matrix<double, -1, 1>  &lkj_cholesky_eta =   Model_args_as_cpp_struct.Model_args_col_vecs_double[0];

  const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];

  // const Eigen::Matrix<double, -1, -1> &LT_b_priors_shape  = Model_args_as_cpp_struct.Model_args_mats_double[0];
  // const Eigen::Matrix<double, -1, -1> &LT_b_priors_scale  = Model_args_as_cpp_struct.Model_args_mats_double[1];
  // const Eigen::Matrix<double, -1, -1> &LT_known_bs_indicator = Model_args_as_cpp_struct.Model_args_mats_double[2];
  // const Eigen::Matrix<double, -1, -1> &LT_known_bs_values = Model_args_as_cpp_struct.Model_args_mats_double[3];

  const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_coeffs_mean  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0];
  const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_coeffs_sd   =  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1];
  ////
  const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_for_corr_a   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2];
  const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_for_corr_b   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3];
  std::vector<Eigen::Matrix<double, -1, -1>>   lb_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4];
  const std::vector<Eigen::Matrix<double, -1, -1>>   &ub_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5];
  const std::vector<Eigen::Matrix<double, -1, -1>>   &known_values    = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6];

  const std::vector<Eigen::Matrix<int, -1, -1 >> &known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];
  
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
  
  //// ---- Ordinal-only params:
  const int n_binary_tests       = Model_args_as_cpp_struct.Model_args_ints(4); //// ---- ordinal-only
  const int n_ordinal_tests      = Model_args_as_cpp_struct.Model_args_ints(5); //// ---- ordinal-only
  
 ////  std::cout << "MVOP: n_bin=" << n_binary_tests << " n_ord=" << n_ordinal_tests << std::endl; std::cout.flush();
  
  const Eigen::Matrix<int, -1, 1> &n_cat_per_ord_test   = Model_args_as_cpp_struct.Model_args_col_vecs_int[0]; //// ---- ordinal-only
  const Eigen::Matrix<int, -1, 1> &n_thr_per_ord_test   = Model_args_as_cpp_struct.Model_args_col_vecs_int[1]; //// ---- ordinal-only
  
 ////  std::cout << "MVOP: n_thr_per_ord_test read OK, size=" << n_thr_per_ord_test.size() << std::endl; std::cout.flush();
 ////  std::cout << "MVOP: n_cat_per_ord_test read OK" << std::endl; std::cout.flush();
  
  const int n_cutpoints_total = n_class * n_thr_per_ord_test.sum(); //// ---- ordinal-only
  
  //// Flattened Dirichlet concentration params (stacked: alpha for test 0, then test 1, etc):
  // const Eigen::Matrix<double, -1, 1> &prior_dirichlet_alpha_flat = Model_args_as_cpp_struct.Model_args_col_vecs_double[1]; 
  const Eigen::Matrix<double, -1, -1> &prior_dirichlet_alpha = Model_args_as_cpp_struct.Model_args_mats_double[4]; //// ---- ordinal-only
  
  // ... read prior_dirichlet_alpha
 ////  std::cout << "MVOP: prior_dirichlet_alpha read OK" << std::endl; std::cout.flush();
  
  //////////////
  const int n_corrs =  n_class * n_tests * (n_tests - 1) * 0.5;

  int n_covariates_total, n_covariates_max;
  int n_covariates_total_nd, n_covariates_total_d;
  int n_covariates_max_nd, n_covariates_max_d;

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
  
  //// ---- determine chunk size --------------------------------------------------
  const int desired_n_chunks = n_chunks;

  int vec_size;
  if (vect_type == "AVX512")      vec_size = 8;
  else if (vect_type == "AVX2")   vec_size = 4;
  else if (vect_type == "AVX")    vec_size = 2;
  else                            vec_size = 1;

  ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, desired_n_chunks);

  int chunk_size = chunk_size_info.chunk_size;
  const int chunk_size_orig = chunk_size_info.chunk_size_orig;
  const int normal_chunk_size = chunk_size_info.normal_chunk_size;
  const int last_chunk_size = chunk_size_info.last_chunk_size;
  const int n_total_chunks = chunk_size_info.n_total_chunks;
  const int n_full_chunks = chunk_size_info.n_full_chunks;

  //////////////  --------------------------------------------------------------------------------------------------------------------------------------
  ////
  //// ---- Corrs (doubles):
  ////
  const Eigen::Matrix<double, -1, 1>  Omega_raw_vec_double = theta_main_vec_ref.head(n_corrs); 
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
  ////
  //// ---- Prev (doubles):
  ////
  double u_prev_diseased = 0.00001;
  if (n_class > 1) u_prev_diseased = theta_main_vec_ref(n_params_main - 1);
  
  ////
  //// ---- Cutpoints (doubles):
  ////
  // const int n_binary_tests  = n_binary_tests;
  // const int n_ordinal_tests = n_ordinal_tests;
  const int n_cutpoints_max = n_thr_per_ord_test.maxCoeff();
  // Eigen::Matrix<int, -1, 1> n_cutpoints_per_outcome_vec(n_tests);
  // int n_ordinal_tests = 0;
  // ////
  // for (int t = 0; t < n_ordinal_tests; ++t) {
  //   if (stan::math::max(y_ref.col(t)) == 1) { 
  //     n_cutpoints_per_outcome_vec(t) = 1;
  //   } else { 
  //     int n_cat_t = stan::math::max(y_ref.col(t));
  //     n_cutpoints_per_outcome_vec(t) = n_cat_t - 1;
  //     n_ordinal_tests += 1;
  //   }
  // }
  // const int n_binary_tests = n_tests - n_ordinal_tests;
  // const int n_cutpoints_max = stan::math::max(n_cutpoints_per_outcome_vec);
  // const int n_cutpoints_total = 2*n_cutpoints_per_outcome_vec.sum();
  ////
  std::vector<Eigen::Matrix<double, -1, -1>> C_raw = vec_of_mats(n_cutpoints_max, n_ordinal_tests, n_class);
  Eigen::Matrix<double, -1, 1> C_raw_vec(n_cutpoints_total);
  {
    int i = n_us + n_corrs + n_covariates_total + 1;
    int j = 0;
    for (int c = 0; c < n_class; ++c) {
      for (int t = 0; t < n_ordinal_tests; ++t) {
        for (int k = 0; k < n_thr_per_ord_test(t); ++k) {
          C_raw[c](k, t) = theta_main_vec_ref(i);
          C_raw_vec(j) = theta_main_vec_ref(i);
          i += 1;
          j += 1;
        }
      } 
    }
  }
  
 ////  std::cout << "MVOP: Random checkpoint #1" << std::endl; std::cout.flush();
  
  const bool softplus = false;
  std::vector<Eigen::Matrix<double, -1, -1>> C = vec_of_mats(n_cutpoints_max, n_ordinal_tests, n_class);
  ////
  for (int c = 0; c < n_class; ++c) {
    for (int t = 0; t < n_ordinal_tests; ++t) {
      int n_thr_t = n_thr_per_ord_test(t);
      Eigen::Matrix<double, -1, 1> C_raw_t = C_raw[c].col(t).head(n_thr_t);
      C[c].col(t).head(n_thr_t) = construct_C(C_raw_t, softplus);
    }
  }
  
 ////  std::cout << "MVOP: Random checkpoint #2" << std::endl; std::cout.flush();
  
  ////
  //// ---- Cutpoints (priors + Jacobian) - doubles:
  ////
  ////
  //// ---- Jacobian for raw_C -> C transformation:
  ////
  double log_det_J_C_double = 0.0;
  {
    for (int c = 0; c < n_class; ++c) {
      for (int t = 0; t < n_ordinal_tests; ++t) {
        int n_thr_t = n_thr_per_ord_test(t);
        log_det_J_C_double += raw_C_to_C_log_det_J_lp(C_raw[c].col(t).head(n_thr_t), softplus);
      } 
    }
  }
  
 ////  std::cout << "MVOP: Random checkpoint #3" << std::endl; std::cout.flush();
  
  // target_AD += log_det_J_C_double;
  ////
  //// ---- Cutpoint priors (Induced-Dirichlet):
  ////
  double log_prior_density_C_double = 0.0;
  {
    for (int c = 0; c < n_class; ++c) {
      for (int t = 0; t < n_ordinal_tests; ++t) {
        
          int n_thr_t = n_thr_per_ord_test(t);
          int n_cat_t = n_thr_t + 1;
          Eigen::Matrix<double, -1, 1> ID_cumul_probs_t = stan::math::Phi(C[c].col(t).head(n_thr_t));
          Eigen::Matrix<double, -1, 1> ID_ord_probs_t   = cumul_probs_to_ord_probs(ID_cumul_probs_t);
          ////
          Eigen::Matrix<double, -1, 1> prior_dirichlet_alpha_t = prior_dirichlet_alpha.col(t).head(n_cat_t);
          ////
          log_prior_density_C_double += induced_dirichlet_given_C_lpdf( ID_ord_probs_t,
                                                                        C[c].col(t).head(n_thr_t), 
                                                                        prior_dirichlet_alpha_t,
                                                                        true);
        
      }
    }
    // target_AD += log_prior_density_C_double;
  }
  
 ////  std::cout << "MVOP: Random checkpoint #4" << std::endl; std::cout.flush();
  
  ////
  //// ---- Omega:
  ////
  double prior_densities_L_Omega_double = 0.0;
  double log_det_J_L_Omega_double = 0.0;
  Eigen::Matrix<double, -1, 1>  grad_Omega_raw_priors_and_log_det_J(n_corrs);

  double prior_densities_prev_double = 0.0;
  double log_det_J_prev_double = 0.0;
  double grad_prev_raw_priors_and_log_det_J = 0.0;

  int dim_choose_2 = n_tests * (n_tests - 1) * 0.5 ;
  std::vector<Eigen::Matrix<double, -1, -1>> deriv_L_wrt_unc_full = vec_of_mats_double(dim_choose_2 + n_tests, dim_choose_2, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_double = vec_of_mats_double(n_tests, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_recip_double = vec_of_mats_double(n_tests, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> Omega_double = vec_of_mats_double(n_tests, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> Omega_unconstrained_double = fn_convert_std_vec_of_corrs_to_3d_array_double(Eigen_vec_to_std_vec(Omega_raw_vec_double),
                                                                           n_tests,
                                                                           n_class);
  
  for (int c = 0; c < n_class; ++c) {
      Eigen::Matrix<double, -1, -1>  Chol_Schur_outs_dbl =  Pinkney_LDL_bounds_opt_dbl(n_tests,
                                                                                       lb_corr[c],
                                                                                       ub_corr[c],
                                                                                       Omega_unconstrained_double[c],
                                                                                       known_values_indicator[c],
                                                                                       known_values[c]);
      L_Omega_double[c] =  Chol_Schur_outs_dbl.block(1, 0, n_tests, n_tests);
      Omega_double[c]   = L_Omega_double[c] * L_Omega_double[c].transpose();
      ////
      L_Omega_recip_double[c] = Eigen::Matrix<double, -1, -1>::Zero(n_tests, n_tests);
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
  ////
  //// ---- Compute L_Omega derivatives gradient using finite differences:
  ////
  Eigen::VectorXd grad_log_det_J(n_corrs);
  grad_log_det_J.setZero();
  
  if (J_grad_option == "num_diff") {
    
    int global_idx = 0;  // Cumulative counter for all classes
    
    ////// First, calculate the expected dimensions
    int n_unconstrained = n_tests * (n_tests - 1) / 2;  // Strict lower triangle
    int n_L_elements = n_tests * (n_tests + 1) / 2;     // Full lower triangle with diagonal
    
    for (int c = 0; c < n_class; ++c) {
      
      double epsilon = 0.000001; 
      
      // Get base values (compute once)
      Eigen::Matrix<double, -1, -1> Chol_Schur_outs_base = Pinkney_LDL_bounds_opt_dbl( n_tests,
                                                                                       lb_corr[c], 
                                                                                       ub_corr[c],
                                                                                       Omega_unconstrained_double[c],
                                                                                       known_values_indicator[c],
                                                                                       known_values[c]);
      L_Omega_double[c] = Chol_Schur_outs_base.block(1, 0, n_tests, n_tests);
      double log_det_J_base = Chol_Schur_outs_base(0, 0);
      // log_det_J_L_Omega_double += log_det_J_base;
      
      // Finite difference for each parameter
      int cnt_2 = 0;
      for (int i = 1; i < n_tests; i++) {
        
            for (int j = 0; j < i; j++) {
                    
                    // Perturb parameter
                    double y = Omega_unconstrained_double[c](i, j);
                    double epsilon = std::max(1e-8, 1e-6 * std::abs(y));
                    if (std::abs(y) > 2.0) {
                      epsilon = 1e-4; // Larger epsilon when near tanh saturation
                    }
                    Omega_unconstrained_double[c](i, j) += epsilon;
                    
                    // ONE computation gets both L_Omega AND log_det_J
                    Eigen::Matrix<double, -1, -1> Chol_Schur_outs_perturbed = Pinkney_LDL_bounds_opt_dbl(  n_tests,
                                                                                                           lb_corr[c],
                                                                                                           ub_corr[c],
                                                                                                           Omega_unconstrained_double[c],
                                                                                                           known_values_indicator[c],
                                                                                                           known_values[c]);
                    Eigen::Matrix<double, -1, -1> L_perturbed = Chol_Schur_outs_perturbed.block(1, 0, n_tests, n_tests);
                    double log_det_J_perturbed = Chol_Schur_outs_perturbed(0, 0);
                    
                    // Compute L_Omega derivatives
                    int cnt_1 = 0;
                    for (int k = 0; k < n_tests; k++) {
                      for (int l = 0; l <= k; l++) {
                        
                        double deriv = (L_perturbed(k, l) - L_Omega_double[c](k, l)) / epsilon;
                        if (!std::isfinite(deriv)) {
                          deriv = 0.0;  // or handle differently
                        }
                        deriv_L_wrt_unc_full[c](cnt_1, cnt_2) = deriv;
                        cnt_1 += 1;
                        
                      }
                    } 
                    
                    // // Compute log_det_J derivative
                    // double deriv = (log_det_J_perturbed - log_det_J_base) / epsilon;
                    // if (!std::isfinite(deriv)) {
                    //   deriv = 0.0;
                    // }
                    // // grad_log_det_J(global_idx) = deriv;
                    
                    // Restore parameter
                    Omega_unconstrained_double[c](i, j) -= epsilon;
                    
                    cnt_2 += 1;
                    global_idx += 1;  // Increment global index after each (i,j) pair
              
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
  
  
 ////  std::cout << "MVOP: Random checkpoint #5" << std::endl; std::cout.flush();
  
  // double log_prior_density_C = 0.0;
  // double log_det_J_C = 0.0;
  Eigen::Matrix<double, -1, 1> grad_C_raw_priors_and_log_det_J_double(n_cutpoints_total);
  
 ////  std::cout << "MVOP: Random checkpoint #6" << std::endl; std::cout.flush();
  
  // {
 
  {    ///////////   -------------------  start of AD block  ------------------------------------------------------------------------------------------------------------------

          stan::math::start_nested();  ////////////////////////
          ////
          stan::math::var target_AD = 0.0;
          ////
          //// ---- Cutpoints (var's):
          ////
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> C_raw_var = vec_of_mats_var(n_cutpoints_max, n_ordinal_tests, n_class);
          Eigen::Matrix<stan::math::var, -1, 1> C_raw_vec_var(n_cutpoints_total);
          {
            int i = n_us + n_corrs + n_covariates_total + 1;
            int j = 0;
            for (int c = 0; c < n_class; ++c) {
              for (int t = 0; t < n_ordinal_tests; ++t) {
                for (int k = 0; k < n_thr_per_ord_test(t); ++k) {
                  C_raw_vec_var(j) = stan::math::to_var(theta_main_vec_ref(i));
                  C_raw_var[c](k, t) = C_raw_vec_var(j);
                  i += 1;
                  j += 1;
                }
              }  
            }
          }
          
         ////  std::cout << "MVOP: Random checkpoint #7" << std::endl; std::cout.flush();
          
          const bool softplus = false;
          std::vector<Eigen::Matrix<stan::math::var, -1, -1>> C_var = vec_of_mats_var(n_cutpoints_max, n_ordinal_tests, n_class);
          for (int c = 0; c < n_class; ++c) {
            for (int t = 0; t < n_ordinal_tests; ++t) {
                
                int n_thr_t = n_thr_per_ord_test(t);
                Eigen::Matrix<stan::math::var, -1, 1> C_raw_t_var = C_raw_var[c].col(t).head(n_thr_t);
                C_var[c].col(t).head(n_thr_t) = construct_C_var(C_raw_t_var, softplus);
              
            }
          }
          
         ////  std::cout << "MVOP: Random checkpoint #8" << std::endl; std::cout.flush();
          
          ////
          //// Cutpoints (priors + Jacobian) - var's: ----------------------------------------------------------------------------------------------------------------------
          ////
          stan::math::var log_det_J_C = 0.0;
          ////
          //// Jacobian for raw_C -> C transformation:
          ////
          for (int c = 0; c < n_class; ++c) {
            for (int t = 0; t < n_ordinal_tests; ++t) {
              
                int n_thr_t = n_thr_per_ord_test(t);
                log_det_J_C += raw_C_to_C_log_det_J_lp_var(C_raw_var[c].col(t).head(n_thr_t), softplus);
                
            }
          }
          target_AD += log_det_J_C;
          
         ////  std::cout << "MVOP: Random checkpoint #9" << std::endl; std::cout.flush();
          
          ////
          //// Cutpoint priors (Induced-Dirichlet):
          ////
          // prior_dirichlet_alpha = ....;
          {
            stan::math::var log_prior_density_C = 0.0;
            for (int c = 0; c < n_class; ++c) {
              for (int t = 0; t < n_ordinal_tests; ++t) {
                
                  int n_thr_t = n_thr_per_ord_test(t);
                  int n_cat_t = n_thr_t + 1;
                  
                  Eigen::Matrix<stan::math::var, -1, 1> ID_cumul_probs_t_var = stan::math::Phi(C_var[c].col(t).head(n_thr_t));
                  Eigen::Matrix<stan::math::var, -1, 1> ID_ord_probs_t_var   = cumul_probs_to_ord_probs_var(ID_cumul_probs_t_var);
                  ////
                  Eigen::Matrix<double, -1, 1> prior_dirichlet_alpha_t = prior_dirichlet_alpha.col(t).head(n_cat_t);
                  ////
                  log_prior_density_C += induced_dirichlet_given_C_lpdf_var( ID_ord_probs_t_var,
                                                                             C_var[c].col(t).head(n_thr_t), 
                                                                             prior_dirichlet_alpha_t,
                                                                             true);
                  
              }
              
            }
            target_AD += log_prior_density_C;
          }
          
         ////  std::cout << "MVOP: Random checkpoint #10" << std::endl; std::cout.flush();
          
          ////////////////////////////////////////////////////////////
          target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
          grad_C_raw_priors_and_log_det_J_double =  C_raw_vec_var.adj();    // differentiating WRT this
          out_mat.segment(1 + n_us + n_corrs + n_covariates_total, n_cutpoints_total) =  grad_C_raw_priors_and_log_det_J_double ;   //// add grad constribution to output
          stan::math::set_zero_all_adjoints_nested();
          ////////////////////////////////////////////////////////////
          
         ////  std::cout << "MVOP: Random checkpoint #11" << std::endl; std::cout.flush();
          
          ////
          //// L_Omega /  correlation params - var's:  ----------------------------------------------------------------------------------------------------------------------
          ////
          Eigen::Matrix<stan::math::var, -1, 1  >  Omega_raw_vec_var =  stan::math::to_var(Omega_raw_vec_double) ;
          std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_unconstrained_var = fn_convert_std_vec_of_corrs_to_3d_array_var(Eigen_vec_to_std_vec_var(Omega_raw_vec_var),
                                                                                                                                       n_tests,
                                                                                                                                       n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  L_Omega_var = vec_of_mats_var(n_tests, n_tests, n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_var   = vec_of_mats_var(n_tests, n_tests, n_class);
          {
                stan::math::var log_det_J_L_Omega = 0.0;

                for (int c = 0; c < n_class; ++c) {
                  
                      // for (int i = 1; i < n_tests; ++i) {    ////   for (i in 2:n_tests) {
                      //   for (int j = 0; j < i; ++j) {   ////   for (j in 1:(i - 1)) {
                      //     if (known_values_indicator[c](i, j) == 1) {
                      //       stan::math::var zero_var = 0.0;
                      //       Omega_unconstrained_var[c](i, j) = zero_var;
                      //     }
                      //   }
                      // }
                  
                      Eigen::Matrix<stan::math::var, -1, -1>  ub = stan::math::to_var(ub_corr[c]);
                      Eigen::Matrix<stan::math::var, -1, -1>  lb = stan::math::to_var(lb_corr[c]);
                      Eigen::Matrix<stan::math::var, -1, -1>  Chol_Schur_outs =  Pinkney_LDL_bounds_opt(n_tests,
                                                                                                        lb,
                                                                                                        ub,
                                                                                                        Omega_unconstrained_var[c],
                                                                                                        known_values_indicator[c],
                                                                                                        known_values[c]);
                      L_Omega_var[c]   =  Chol_Schur_outs.block(1, 0, n_tests, n_tests);
                      Omega_var[c] =   L_Omega_var[c] * L_Omega_var[c].transpose();
                      log_det_J_L_Omega +=   Chol_Schur_outs(0, 0); // now can set prior directly on Omega (as this is Jacobian adjustment)
                      
                      for (int i = 1; i < n_tests; ++i) {    ////   for (i in 2:n_tests) {
                        for (int j = 0; j < i; ++j) {   ////   for (j in 1:(i - 1)) {
                            if (known_values_indicator[c](i, j) == 1) {
                              target_AD += stan::math::normal_lpdf( Omega_var[c](i, j), 0.0, 10.0 ); //// to ensure any corr's we aren't estimating dont cause divergences
                            }
                        }
                      }
                }
                
                for (int i = 0; i < (Omega_raw_vec_var.size()); ++i) {
                    target_AD += stan::math::normal_lpdf(Omega_raw_vec_var(i), 0.0, 10.0); 
                }

                log_det_J_L_Omega_double += log_det_J_L_Omega.val();
                target_AD += log_det_J_L_Omega;

          }
          
         ////  std::cout << "MVOP: Random checkpoint #12" << std::endl; std::cout.flush();

          {
                    stan::math::var prior_densities_L_Omega = 0.0;

                    for (int c = 0; c < n_class; ++c) {

                          if ( (corr_prior_beta == false)   &&  (corr_prior_norm == false) ) {
                            prior_densities_L_Omega +=  stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c], lkj_cholesky_eta(c)) ;
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
                grad_Omega_raw_priors_and_log_det_J =  Omega_raw_vec_var.adj();    // differentiating WRT this
                // if (J_grad_option != "autodiff") {
                //   grad_Omega_raw_priors_and_log_det_J.array() += grad_log_det_J.array();
                // } 
                out_mat.segment(1 + n_us, n_corrs) =  grad_Omega_raw_priors_and_log_det_J ;   //// add grad constribution to output
                stan::math::set_zero_all_adjoints_nested();
                ////////////////////////////////////////////////////////////
            }
          
         ////  std::cout << "MVOP: Random checkpoint #13" << std::endl; std::cout.flush();

          /////////////  prev stuff  ---- vars
          {
              if (n_class > 1) {  //// if latent class
                
                  stan::math::var target_AD_prev = 0.0;
    
                  std::vector<stan::math::var> 	 u_prev_var_vec_var(n_class, 0.0);
                  std::vector<stan::math::var> 	 prev_var_vec_var(n_class, 0.0);
                  std::vector<stan::math::var> 	 tanh_u_prev_var(n_class, 0.0);
                  Eigen::Matrix<stan::math::var, -1, -1>	 prev_var = Eigen::Matrix<stan::math::var, -1, -1>::Zero(1, 2);
                  stan::math::var tanh_pu_deriv_var = 0.0;
                  stan::math::var deriv_p_wrt_pu_var = 0.0;
                  stan::math::var tanh_pu_second_deriv_var = 0.0;
                  stan::math::var log_jac_p_deriv_wrt_pu_var = 0.0;
                  stan::math::var log_det_J_prev_var = 0.0;
    
                  u_prev_var_vec_var[1] =  stan::math::to_var(u_prev_diseased);
                  tanh_u_prev_var[1] =  stan::math::tanh(u_prev_var_vec_var[1]); /// ( stan::math::exp(2.0*u_prev_var_vec_var[1] ) - 1.0) / ( stan::math::exp(2*u_prev_var_vec_var[1] ) + 1.0) ;
                  u_prev_var_vec_var[0] =   0.5 *  stan::math::log( (1.0 + ( (1.0 - 0.5 * ( tanh_u_prev_var[1] + 1.0))*2.0 - 1.0) ) / (1.0 - ( (1.0 - 0.5 * ( tanh_u_prev_var[1] + 1.0))*2.0 - 1.0) ) )  ;
                  tanh_u_prev_var[0] =  stan::math::tanh(u_prev_var_vec_var[0]); ///  (stan::math::exp(2.0*u_prev_var_vec_var[0] ) - 1.0) / ( stan::math::exp(2*u_prev_var_vec_var[0] ) + 1.0) ;
    
                  prev_var_vec_var[1] =  0.5 * ( tanh_u_prev_var[1] + 1.0);
                  prev_var_vec_var[0] =  0.5 * ( tanh_u_prev_var[0] + 1.0);
                  prev_var(0,1) =  prev_var_vec_var[1];
                  prev_var(0,0) =  prev_var_vec_var[0];
    
                  tanh_pu_deriv_var = ( 1.0 - (tanh_u_prev_var[1] * tanh_u_prev_var[1])  );
                  deriv_p_wrt_pu_var = 0.5 *  tanh_pu_deriv_var;
                  tanh_pu_second_deriv_var  = -2.0 * tanh_u_prev_var[1]  * tanh_pu_deriv_var;
                  log_jac_p_deriv_wrt_pu_var  = ( 1.0 / deriv_p_wrt_pu_var) * 0.5 * tanh_pu_second_deriv_var; // for gradient of u's
    
                  log_det_J_prev_var =    stan::math::log( deriv_p_wrt_pu_var );
                  log_det_J_prev_double =  log_det_J_prev_var.val() ;
    
                  stan::math::var prior_densities_prev = beta_lpdf(prev_var(0, 1), prev_prior_a, prev_prior_b)  ;  // weakly informative prior - helps avoid boundaries with slight negative skew (for lower N)
                  prior_densities_prev_double = prior_densities_prev.val();
    
                  //  target_AD_prev += log_det_J_prev_var ; /// done manually later
                  target_AD_prev += prior_densities_prev;
                  target_AD += target_AD_prev;
    
                  ////////////////////////////////////////////////////////////
                  target_AD_prev.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
                  grad_prev_raw_priors_and_log_det_J  =  u_prev_var_vec_var[1].adj() - u_prev_var_vec_var[0].adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
                  out_mat(1 + n_us + n_corrs + n_covariates_total) = grad_prev_raw_priors_and_log_det_J; //// add grad constribution to output vec
                  stan::math::set_zero_all_adjoints_nested();
                  ////////////////////////////////////////////////////////////
  
              }
          }
          
         ////  std::cout << "MVOP: Random checkpoint #14" << std::endl; std::cout.flush();

          
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
          
         ////  std::cout << "MVOP: Random checkpoint #15" << std::endl; std::cout.flush();

  }   //////////////////////////  end of local AD block
  
 ////  std::cout << "MVOP: end of local AD block" << std::endl; std::cout.flush();
    
  /////////////  prev stuff
  std::vector<double> 	 u_prev_var_vec(n_class, 0.0);
  std::vector<double> 	 prev_var_vec(n_class, 0.0);
  std::vector<double> 	 tanh_u_prev(n_class, 0.0);
  Eigen::Matrix<double, -1, -1>	 prev = Eigen::Matrix<double, -1, -1>::Zero(1, n_class);
  double tanh_pu_deriv = 0.0;
  double deriv_p_wrt_pu_double = 0.0;
  double tanh_pu_second_deriv = 0.0;
  double log_jac_p_deriv_wrt_pu = 0.0;
  double log_jac_p = 0.0;

  if (n_class > 1) {  //// if latent class

        u_prev_var_vec[1] =  (double) u_prev_diseased ;
        tanh_u_prev[1] = stan::math::tanh(u_prev_var_vec[1]);//  ( exp(2.0*u_prev_var_vec[1] ) - 1.0) / ( exp(2.0*u_prev_var_vec[1] ) + 1.0) ;
        u_prev_var_vec[0] =   0.5 *  stan::math::log( (1.0 + ( (1.0 - 0.5 * ( tanh_u_prev[1] + 1.0))*2.0 - 1.0) ) / (1.0 - ( (1.0 - 0.5 * ( tanh_u_prev[1] + 1.0))*2.0 - 1.0) ) )  ;
        tanh_u_prev[0] = stan::math::tanh(u_prev_var_vec[0]); //  (exp(2.0*u_prev_var_vec[0] ) - 1.0) / ( exp(2.0*u_prev_var_vec[0] ) + 1.0) ;

        prev_var_vec[1] =  0.5 * ( tanh_u_prev[1] + 1.0);
        prev_var_vec[0] =  0.5 * ( tanh_u_prev[0] + 1.0);
        prev(0,1) =  prev_var_vec[1];
        prev(0,0) =  prev_var_vec[0];

        tanh_pu_deriv = ( 1.0 - (tanh_u_prev[1] * tanh_u_prev[1])  );
        deriv_p_wrt_pu_double = 0.5 *  tanh_pu_deriv;
        tanh_pu_second_deriv  = -2.0 * tanh_u_prev[1]  * tanh_pu_deriv;
        log_jac_p_deriv_wrt_pu  = ( 1.0 / deriv_p_wrt_pu_double) * 0.5 * tanh_pu_second_deriv; // for gradient of u's
        log_jac_p =    stan::math::log( deriv_p_wrt_pu_double );

  }
  
 ////  std::cout << "MVOP: Random checkpoint #16" << std::endl; std::cout.flush();

  ///////////////////////////////////////////////////////////////////////// prior densities
  double prior_densities = 0.0;

  if (exclude_priors == false) {

        ///////////////////// priors for coeffs
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
        prior_densities += log_prior_density_C_double;

  }
  
 ////  std::cout << "MVOP: Random checkpoint #17" << std::endl; std::cout.flush();

  ////////  ------- likelihood function  ---------------------------------------------------------------------------------------------------------------------------------------------------
  double log_prob_out = 0.0;
  
  // Jacobian adjustments (none needed for coeffs as unconstrained - so only for L_Omega -> Omega and u_prev -> prev, and the one for u's is computed in the likelihood)
  const double log_det_J_main = log_det_J_prev_double + log_det_J_L_Omega_double + log_det_J_C_double;

  const Eigen::Matrix<double, -1, -1>  log_prev = stan::math::log(prev);
  
  //// Global (shared) gradient accumulators — will be written to by reduction
  std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array = vec_of_mats<double>(n_covariates_max, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array = vec_of_mats<double>(n_tests, n_tests, n_class);
  Eigen::Matrix<double, -1, 1> prev_grad_vec = Eigen::Matrix<double, -1, 1>::Zero(n_class);
  double log_jac_u = 0.0;
  ////
  std::vector<Eigen::Matrix<double, -1, -1>> cutpoint_grad_array = vec_of_mats<double>(n_cutpoints_max, n_ordinal_tests, n_class); //// ordinal-only
  
 ////  std::cout << "MVOP: Random checkpoint #18" << std::endl; std::cout.flush();
  
  // ============================================================================
  // WCP: PARALLEL CHUNK LOOP (full-sized chunks only)
  // ============================================================================
  // Process n_full_chunks in parallel. Last chunk (remainder) handled separately
  // after the parallel region because it requires workspace resize + SIMD fallback.
  // ============================================================================
  
  // Number of full chunks to process in parallel
  const int n_parallel_chunks = n_full_chunks;
  
  #pragma omp parallel num_threads(n_threads_WCP) if (n_threads_WCP > 1)
  {
    const int tid = omp_get_thread_num();
    
    // ---- Thread-local workspace ----
    LC_MVP_workspace_struct &ws = LC_MVP_ws_structs[tid];
    ws.reset_sizes();
    
    // ... allocate workspace matrices
   ////  std::cout << "MVOP: workspace allocated" << std::endl; std::cout.flush();
    
   ////  std::cout << "MVOP: Random checkpoint #19" << std::endl; std::cout.flush();
    
    // ---- Thread-local gradient accumulators (ZERO-INITIALISED) ----
    std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_local = vec_of_mats<double>(n_covariates_max, n_tests, n_class);
    std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_local = vec_of_mats<double>(n_tests, n_tests, n_class);
    Eigen::Matrix<double, -1, 1> prev_grad_local = Eigen::Matrix<double, -1, 1>::Zero(n_class);
    ////
    double log_jac_u_local = 0.0;
    ////
    std::vector<Eigen::Matrix<double, -1, -1>> cutpoint_grad_local = vec_of_mats<double>(n_cutpoints_max, n_ordinal_tests, n_class); //// ordinal-only
    
    // ---- Thread-local workspace references (normal chunk size) ----
    ////////////////////////////////////////////////
    std::vector<Eigen::Matrix<double, -1, -1>> &Z_std_norm = ws.Z_std_norm;
    std::vector<Eigen::Matrix<double, -1, -1>> &Bound_Z = ws.Bound_Z;
    std::vector<Eigen::Matrix<double, -1, -1>> &Bound_U_Phi_Bound_Z = ws.Bound_U_Phi_Bound_Z;
    std::vector<Eigen::Matrix<double, -1, -1>> &prob = ws.prob;
    std::vector<Eigen::Matrix<double, -1, -1>> &Phi_Z = ws.Phi_Z;
    ////////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> &y1_log_prob = ws.y1_log_prob;
    Eigen::Matrix<double, -1, -1> &phi_Z_recip = ws.phi_Z_recip;
    Eigen::Matrix<double, -1, -1> &phi_Bound_Z = ws.phi_Bound_Z;
    ////////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> &u_grad_array_CM_chunk = ws.u_grad_array_CM_chunk;
    ////////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> &common_grad_term_1 = ws.common_grad_term_1;
    Eigen::Matrix<double, -1, -1> &y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = ws.y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip;
    Eigen::Matrix<double, -1, -1> &y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = ws.y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip;
    Eigen::Matrix<double, -1, -1> &prob_rowwise_prod_temp = ws.prob_rowwise_prod_temp;
    Eigen::Matrix<double, -1, -1> &prob_recip_rowwise_prod_temp = ws.prob_recip_rowwise_prod_temp;
    ////////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1> &prod_container_or_inc_array = ws.prod_container_or_inc_array;
    Eigen::Matrix<double, -1, 1> &derivs_chain_container_vec = ws.derivs_chain_container_vec;
    Eigen::Matrix<double, -1, 1> &prob_rowwise_prod_temp_all = ws.prob_rowwise_prod_temp_all;
    ////////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> &grad_prob = ws.grad_prob;
    Eigen::Matrix<double, -1, -1> &z_grad_term = ws.z_grad_term;
    ////////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> &y_chunk = ws.y_chunk;
    Eigen::Matrix<double, -1, -1> &u_array = ws.u_array;
    Eigen::Matrix<double, -1, -1> &y_sign = ws.y_sign;
    Eigen::Matrix<double, -1, -1> &y_m_y_sign_x_u = ws.y_m_y_sign_x_u;
    ////////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> &u_grad_array_CM_chunk_block = ws.u_grad_array_CM_chunk_block;
    ////////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1> &u_unc_vec_chunk = ws.u_unc_vec_chunk;
    Eigen::Matrix<double, -1, 1> &u_vec_chunk = ws.u_vec_chunk;
    Eigen::Matrix<double, -1, 1> &du_wrt_duu_chunk = ws.du_wrt_duu_chunk;
    Eigen::Matrix<double, -1, 1> &d_J_wrt_duu_chunk = ws.d_J_wrt_duu_chunk;
    ////////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> &lp_array = ws.lp_array;
    ////////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1> &prob_n = ws.prob_n;
    Eigen::Matrix<double, -1, 1> &prob_n_recip = ws.prob_n_recip;
    Eigen::Matrix<double, -1, 1> &log_sum_result = ws.log_sum_result;
    Eigen::Matrix<double, -1, 1> &container_max_logs = ws.container_max_logs;
    ////////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1> &rowwise_log_sum = ws.rowwise_log_sum;
    Eigen::Matrix<double, -1, 1> &rowwise_prod = ws.rowwise_prod;
    Eigen::Matrix<double, -1, 1> &rowwise_sum = ws.rowwise_sum;
    Eigen::Matrix<double, -1, 1> &log_lik_chunk = ws.log_lik_chunk;
    ////////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> &prob_recip = ws.prob_recip;
    ////////////////////////////////////////////////
    std::vector<Eigen::Matrix<double, -1, -1>> &Upper_Bound_Z     = ws.Upper_Bound_Z; // ordinal-only
    ////////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> &phi_Upper_Bound_Z = ws.phi_Upper_Bound_Z; // ordinal-only
    Eigen::Matrix<double, -1, -1> &dphi_over_L       = ws.dphi_over_L; // ordinal-only
    Eigen::Matrix<double, -1, -1> &dZ_dmu_neg        = ws.dZ_dmu_neg; // ordinal-only
    ////////////////////////////////////////////////
    
   ////  std::cout << "MVOP: Random checkpoint #20" << std::endl; std::cout.flush();
    
    // ---- Thread-local SIMD types (each thread uses the original SIMD for full chunks) ----
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
    // PARALLEL FOR over full-sized chunks
    // schedule(static): predictable assignment, no overhead
    // Each thread gets contiguous range of chunks → good for pre-fetching
    // ====================================================================
    #pragma omp for schedule(static) nowait
      for (int nc = 0; nc < n_parallel_chunks; nc++) {
          
          const int chunk_counter = nc;
          const int chunk_size = normal_chunk_size;  // All full chunks are same size
  
          ////-----------------------------------------------
          u_grad_array_CM_chunk.setZero(); //// reset between chunks as re-using same container
  
          y_chunk = y_ref.middleRows( chunk_size_orig * chunk_counter, chunk_size).cast<double>();
  
          //// Nuisance parameter transformation step
          u_unc_vec_chunk = theta_us_vec_ref.segment( chunk_size_orig * n_tests * chunk_counter, chunk_size * n_tests);
          fn_MVP_compute_nuisance( u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);
          log_jac_u += fn_MVP_compute_nuisance_log_jac_u( u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);
          
          u_array  =  u_vec_chunk.reshaped(chunk_size, n_tests);
          y_sign.array() =  y_chunk.array() + (y_chunk.array() - 1.0) ;
          y_m_y_sign_x_u.array() =  y_chunk.array() - (y_sign.array() * u_array.array());
          ////-----------------------------------------------
          
         ////  std::cout << "MVOP: Random checkpoint #21" << std::endl; std::cout.flush();
        
          // ---- Class loop (identical to original) ----
          //// START of c loop
          for (int c = 0; c < n_class; c++) {

                  prod_container_or_inc_array.setZero(); //// reset to 0
      
                  //// start of t loop
                  for (int t = 0; t < n_tests; t++) {
      
                          ////-----------------------------------------------
                          if (n_covariates_max > 1) {
                            
                                  Eigen::Matrix<double, -1, 1> Xbeta_given_class_c_col_t = X[c][t].block(chunk_size_orig * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(c, t)).cast<double>()  *
                                                                                           beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t));
                            
                                  if (t < n_binary_tests) { // For binary: Bound_Z = L_recip * (-Xbeta - prod) 
                                  
                                      Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * (-1.0*( Xbeta_given_class_c_col_t.array() + prod_container_or_inc_array.array())) ;
                                  
                                  } else { // Store μ_t temporarily, compute bounds per-observation
                                  
                                      int t_ord = t - n_binary_tests;
                                      int n_thr_t = n_thr_per_ord_test(t_ord);
                                      
                                      for (int i = 0; i < chunk_size; ++i) {
                                        
                                            double mu_t = Xbeta_given_class_c_col_t(i) + prod_container_or_inc_array(i);
                                            int y_i = static_cast<int>(y_chunk(i, t));
                                            double L_tt = L_Omega_double[c](t, t);
                                            
                                            double lb = (y_i == 1) ? -Inf : (C[c](y_i - 2, t_ord) - mu_t) / L_tt; // Lower bound
                                             
                                            double ub = (y_i == n_cat_per_ord_test(t_ord)) ? Inf : (C[c](y_i - 1, t_ord) - mu_t) / L_tt; // Upper bound  
                                            
                                            // Store for GHK:
                                            Bound_Z[c](i, t) = lb;           // lower_bz (re-using Bound_Z container for GHK lower-bound)
                                            Upper_Bound_Z[c](i, t) = ub;     // upper_bz (using the "Upper_Bound_Z", ordinal-only workspace matrix)
                                        
                                      } // end of i loop
                                    
                                  }
                              
                          } else {  // intercept-only
                                
                                  if (t < n_binary_tests) { // For binary: Bound_Z = L_recip * (-Xbeta - prod) 
                                    
                                      Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * ( -1.0*( beta_double_array[c](0, t) + prod_container_or_inc_array.array() ) ) ;
                                        
                                  } else { 
                                    
                                      int t_ord = t - n_binary_tests;
                                      int n_thr_t = n_thr_per_ord_test(t_ord);
                                    
                                      for (int i = 0; i < chunk_size; ++i) {
                                      
                                            double mu_t = beta_double_array[c](0, t) + prod_container_or_inc_array(i);
                                            int y_i = static_cast<int>(y_chunk(i, t));
                                            double L_tt = L_Omega_double[c](t, t);
                                            
                                            double lb = (y_i == 1) ? -Inf : (C[c](y_i - 2, t_ord) - mu_t) / L_tt; // Lower bound
                                            
                                            double ub = (y_i == n_cat_per_ord_test(t_ord)) ? Inf : (C[c](y_i - 1, t_ord) - mu_t) / L_tt; // Upper bound  
                                            
                                            // Store for GHK:
                                            Bound_Z[c](i, t) = lb;           // lower_bz (re-using Bound_Z container for GHK lower-bound)
                                            Upper_Bound_Z[c](i, t) = ub;     // upper_bz (using the "Upper_Bound_Z", ordinal-only workspace matrix)
                                      
                                      } // end of i loop
                                    
                                  }
                              
                          }
                          
                         ////  std::cout << "MVOP: Random checkpoint #22" << std::endl; std::cout.flush();
                          
                          ////-----------------------------------------------
                          ////-----------------------------------------------
                          //// compute/update important log-lik quantities for GHK-MVP
                          if (t < n_binary_tests) {
                            
                              fn_MVP_compute_lp_GHK_cols(  t,
                                                           Bound_U_Phi_Bound_Z[c],  // computing this
                                                           Phi_Z[c],  // computing this
                                                           Z_std_norm[c],  // computing this
                                                           prob[c],  // computing this
                                                           y1_log_prob,  // computing this
                                                           Bound_Z[c],
                                                           y_chunk,
                                                           u_array,
                                                           Model_args_as_cpp_struct);
                          } else { 
                            
                              fn_MVOP_compute_lp_GHK_cols( t,
                                                           Bound_U_Phi_Bound_Z[c], 
                                                           Phi_Z[c],
                                                           Z_std_norm[c], 
                                                           prob[c],
                                                           y1_log_prob,
                                                           Bound_Z[c],   // lower_bz for ordinal
                                                           y_chunk,
                                                           u_array, 
                                                           Upper_Bound_Z[c], // ordinal-only - upper_bz (ordinal only, ignored for binary)
                                                           // Phi_Upper_Bound_Z,  // Φ(upper_bz) (ordinal only)
                                                           n_binary_tests, // ordinal-only
                                                           Model_args_as_cpp_struct);
                            
                          }
                          
                         ////  std::cout << "MVOP: Random checkpoint #23" << std::endl; std::cout.flush();
                          
                          ////-----------------------------------------------
                          
                          ////-----------------------------------------------
                          if (t < n_tests - 1) {
                               auto L_Omega_row = L_Omega_double[c].row(t + 1);
                               prod_container_or_inc_array = Z_std_norm[c].leftCols(t + 1)  *   L_Omega_row.head(t+1).transpose();
                          }
                          ////-----------------------------------------------
                          
                  }   //// end of t loop
                  
                  ////-----------------------------------------------
                  if (n_class > 1) { // if latent class
                      rowwise_sum = y1_log_prob.rowwise().sum();
                      rowwise_sum.array() += log_prev(0, c);
                      lp_array.col(c) = rowwise_sum;
                  } else {
                      rowwise_sum = y1_log_prob.rowwise().sum();
                      lp_array.col(0) =     rowwise_sum;
                  }
                  ////-----------------------------------------------

          }  //// end of c loop
          
         ////  std::cout << "MVOP: Random checkpoint #24" << std::endl; std::cout.flush();
          
        ////-----------------------------------------------  
        if (n_class > 1) {

                log_sum_exp_general( lp_array,
                                     vect_type_exp,
                                     vect_type_log,
                                     log_sum_result,
                                     container_max_logs);
                const int index_start = 1 + n_params + chunk_size_orig * chunk_counter;
                out_mat.segment(index_start, chunk_size) = log_sum_result;

        } else {

                out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size)  = lp_array.col(0);

        }
        
       ////  std::cout << "MVOP: Random checkpoint #25" << std::endl; std::cout.flush();
        
        ////-----------------------------------------------  
        
        ////-----------------------------------------------
        log_lik_chunk = out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size);
        prob_n  =  fn_EIGEN_double(log_lik_chunk, "exp",  vect_type_exp);
        prob_n_recip  =  stan::math::inv(prob_n); // this CANNOT be a temporary otherwise it created a "dangling reference", since prob_n is a temporary!!
        ////-----------------------------------------------
        
        
        /////////////////  ------------------------- compute grad  ---------------------------------------------------------------------------------
        for (int c = 0; c < n_class; c++) {
          
              ////-----------------------------------------------
              prob_recip = stan::math::inv(prob[c]);  
              ////-----------------------------------------------
              
              ////-----------------------------------------------
              //// compute/update important log-lik quantities for GHK-MVP
              for (int t = 0; t < n_tests; t++) {

                    fn_MVP_compute_phi_Z_recip_cols(   t,
                                                       phi_Z_recip, // computing this
                                                       Phi_Z[c], Z_std_norm[c], Model_args_as_cpp_struct);
                
                    if (t < n_binary_tests) {
                      
                          fn_MVP_compute_phi_Bound_Z_cols(t, 
                                                          phi_Bound_Z, // computing this
                                                          Bound_U_Phi_Bound_Z[c], Bound_Z[c], Model_args_as_cpp_struct);
                      
                    } else { // for ordinal tests
                          
                          // φ(lower_bz) — stored in phi_Bound_Z
                          phi_Bound_Z.col(t).array() = (-(Bound_Z[c].col(t).array().square()) * 0.5).exp() * sqrt_2_pi_recip;
                      
                          // φ(upper_bz) — stored in phi_Upper_Bound_Z
                          phi_Upper_Bound_Z.col(t).array() = (-(Upper_Bound_Z[c].col(t).array().square()) * 0.5).exp() * sqrt_2_pi_recip;
                          
                          // Handle ±∞ bounds → φ = 0
                          for (int i = 0; i < chunk_size; ++i) {
                            if (std::isinf(Bound_Z[c](i, t)))       phi_Bound_Z(i, t) = 0.0;
                            if (std::isinf(Upper_Bound_Z[c](i, t))) phi_Upper_Bound_Z(i, t) = 0.0;
                          }
                          
                    }
                
              }
              
             ////  std::cout << "MVOP: Random checkpoint #26" << std::endl; std::cout.flush();
              
              ////-----------------------------------------------
              if (grad_option != "none") {
    
                    fn_MVOP_grad_prep(      prob[c],
                                            y_sign,
                                            y_m_y_sign_x_u,
                                            u_array, //// for ordinal-only fn
                                            L_Omega_recip_double[c],
                                            prev(0, c),
                                            prob_n_recip,
                                            phi_Z_recip,
                                            phi_Bound_Z,
                                            phi_Upper_Bound_Z, //// for ordinal-only fn
                                            prob_recip,
                                            prob_rowwise_prod_temp,
                                            prob_recip_rowwise_prod_temp,
                                            prob_rowwise_prod_temp_all,
                                            common_grad_term_1,
                                            dphi_over_L, //// for ordinal-only fn
                                            dZ_dmu_neg, //// for ordinal-only fn
                                            n_binary_tests, //// for ordinal-only fn
                                            // y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                            // y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                            Model_args_as_cpp_struct) ;
    
              }
              
             ////  std::cout << "MVOP: Random checkpoint #27" << std::endl; std::cout.flush();
              
              ////////////////////////////////////////////////////////////////////////////////////  Grad of nuisance parameters / u's (manual)
              if ( (grad_option == "us_only") || (grad_option == "all") ) {
        
                    u_grad_array_CM_chunk_block =  u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests);
        
                    //// NOTE: this uses the Call BINARY gradient fn's with "dphi_over_L" and "dZ_dmu_neg"
                    ////    as the "y_sign_chunk_times..." and "y_m_ysign_x_u_array_times..." arguments:
                    fn_MVP_compute_nuisance_grad_v2(  u_grad_array_CM_chunk_block,
                                                      phi_Z_recip,
                                                      common_grad_term_1,
                                                      L_Omega_double[c],
                                                      prob[c],
                                                      prob_recip,
                                                      prob_rowwise_prod_temp,
                                                      dphi_over_L,
                                                      dZ_dmu_neg,
                                                      // y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                      // y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                      z_grad_term,
                                                      grad_prob,
                                                      prod_container_or_inc_array,
                                                      derivs_chain_container_vec,
                                                      Model_args_as_cpp_struct);
        
                    u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests).array() += u_grad_array_CM_chunk_block.array();
                    
                    const int start_index = 1 + (chunk_size_orig * n_tests * chunk_counter);
                    const int length = chunk_size * n_tests;
        
                    if (c == n_class - 1) {
                      
                        //// update output vector once all u_grad computations are done
                        out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();
            
                        //// account for unconstrained -> constrained transformations and Jacobian adjustments
                        fn_MVP_nuisance_first_deriv(  du_wrt_duu_chunk,
                                                      u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);
            
                        fn_MVP_nuisance_deriv_of_log_det_J(    d_J_wrt_duu_chunk,
                                                               u_vec_chunk, u_unc_vec_chunk, du_wrt_duu_chunk, Model_args_as_cpp_struct);
    
                        out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
                        out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();
                    
                    }
    
              }
              
             ////  std::cout << "MVOP: Random checkpoint #28" << std::endl; std::cout.flush();
              
              /////////////////////////////////////////////////////////////////////////// Grad of intercepts / coefficients (beta's)
              if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only") ) {
    
                    fn_MVP_compute_coefficients_grad_v3(     c,
                                                             beta_grad_local[c],  // <<< LOCAL accumulator
                                                             chunk_counter,
                                                             n_covariates_max,
                                                             common_grad_term_1,
                                                             L_Omega_double[c],
                                                             prob[c],
                                                             prob_recip,
                                                             prob_rowwise_prod_temp,
                                                             dphi_over_L,
                                                             dZ_dmu_neg,
                                                             // y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                             // y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                             z_grad_term,
                                                             grad_prob,
                                                             prod_container_or_inc_array,
                                                             derivs_chain_container_vec,
                                                             true,  ///   compute_final_scalar_grad,
                                                             Model_args_as_cpp_struct);
                
              }
              
             ////  std::cout << "MVOP: Random checkpoint #29" << std::endl; std::cout.flush();
              
              /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of L_Omega ('s)
              if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only") ) {
    
                    fn_MVP_compute_L_Omega_grad_v3(      U_Omega_grad_local[c],  // <<< LOCAL accumulator
                                                         common_grad_term_1,
                                                         L_Omega_double[c],
                                                         prob[c],
                                                         prob_recip,
                                                         Bound_Z[c],
                                                         Z_std_norm[c],
                                                         prob_rowwise_prod_temp,
                                                         dphi_over_L,
                                                         dZ_dmu_neg,
                                                         // y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                         // y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                         z_grad_term,
                                                         grad_prob,
                                                         prod_container_or_inc_array,
                                                         derivs_chain_container_vec,
                                                         true,  ///   compute_final_scalar_grad,
                                                         Model_args_as_cpp_struct);
    
              }
              
             ////  std::cout << "MVOP: Random checkpoint #30" << std::endl; std::cout.flush();
              
              /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of Cutpoints (MVOP/ordinal only!!)
              if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "cutpoints_only") ) {
                
                   for (int t_ord = 0; t_ord < n_ordinal_tests; ++t_ord) {
                     
                       int t = n_binary_tests + t_ord;
                       int K_t = n_thr_per_ord_test(t_ord) + 1;  // n_cat
                           
                       fn_MVOP_compute_cutpoint_grad(   c, 
                                                        cutpoint_grad_local[c],  // (n_cutpoints_max × n_ordinal_tests)  // <<< LOCAL accumulator
                                                        t, 
                                                        t_ord, 
                                                        K_t,
                                                        common_grad_term_1,
                                                        L_Omega_double[c],
                                                        prob[c],
                                                        prob_recip, 
                                                        prob_rowwise_prod_temp,
                                                        phi_Bound_Z,
                                                        phi_Upper_Bound_Z,
                                                        phi_Z_recip,
                                                        u_array,
                                                        y_chunk,
                                                        dphi_over_L,
                                                        dZ_dmu_neg,
                                                        // y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                        // y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                        z_grad_term,
                                                        grad_prob,
                                                        prod_container_or_inc_array,
                                                        derivs_chain_container_vec,
                                                        true,
                                                        Model_args_as_cpp_struct);
                   }
                
              }
              
             ////  std::cout << "MVOP: Random checkpoint #31" << std::endl; std::cout.flush();
              
              /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Prev grad (LC-MVP/LC-MVOP only)
              if (n_class > 1) { /// prevelance only estimated for latent class models
    
                    if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only" ) ) {
                        
                          rowwise_prod = prob[c].rowwise().prod();
                          rowwise_prod.array() = prob_n_recip.array() * rowwise_prod.array();
                          double prev_grad = rowwise_prod.sum();
                          prev_grad_local(c)  +=  prev_grad;  // <<< LOCAL accumulator
                          
                    }
    
              }
              
             ////  std::cout << "MVOP: Random checkpoint #32" << std::endl; std::cout.flush();
              
        } // end of c loop (gradient)
        
      } // end of nc / parallel_chunks loop
      
     ////  std::cout << "MVOP: Random checkpoint #33" << std::endl; std::cout.flush();
      
      // ====================================================================
      // REDUCTION: accumulate thread-local results into shared accumulators
      // ====================================================================
      #pragma omp critical
      {
        for (int c = 0; c < n_class; c++) {
          beta_grad_array[c].array() += beta_grad_local[c].array();
          U_Omega_grad_array[c].array() += U_Omega_grad_local[c].array();
          cutpoint_grad_array[c].array() += cutpoint_grad_local[c].array();  // ordinal-only
        }
        prev_grad_vec.array() += prev_grad_local.array();
        log_jac_u += log_jac_u_local;
      }
      
     ////  std::cout << "MVOP: Random checkpoint #34" << std::endl; std::cout.flush();
      
    } // end of omp parallel
    
   ////  std::cout << "MVOP: Random checkpoint #35" << std::endl; std::cout.flush();
    
    //// After the chunk loop, chain-rule cutpoint grads through C_raw → C:
    int out_idx = 1 + n_us + n_corrs + n_covariates_total;
    if (n_class > 1) out_idx += 1;  // skip prev slot
    ////
    for (int c = 0; c < n_class; ++c) {
      
        for (int t_ord = 0; t_ord < n_ordinal_tests; ++t_ord) {
              
            // int n_thr_t = n_cutpoints_per_outcome_vec(n_binary_tests + t_ord);
            int n_thr_t = n_thr_per_ord_test(t_ord);
          
            Eigen::Matrix<double, -1, 1> grad_wrt_C = cutpoint_grad_array[c].col(t_ord).head(n_thr_t);
            Eigen::Matrix<double, -1, 1> grad_wrt_C_raw = chain_rule_C_to_C_raw( grad_wrt_C, C_raw[c].col(t_ord).head(n_thr_t));
            out_mat.segment(out_idx, n_thr_t) += grad_wrt_C_raw;
            out_idx += n_thr_t;
          
        }
      
    }
    
   ////  std::cout << "MVOP: Random checkpoint #36" << std::endl; std::cout.flush();
    
    // ============================================================================
    // LAST CHUNK (remainder) — processed serially with fallback SIMD
    // ============================================================================
    if ((n_full_chunks < n_total_chunks) && (last_chunk_size > 0)) {
      
      const int nc = n_full_chunks;
      const int chunk_counter = nc;
      const int chunk_size = last_chunk_size;
      
      // Use thread 0's workspace, resized for last chunk
      LC_MVP_workspace_struct &ws = LC_MVP_ws_structs[0];
      
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
      
      // Resize workspace for last chunk
      for (int c = 0; c < n_class; c++) {
        ws.Z_std_norm[c].resize(last_chunk_size, n_tests);
        ws.Bound_Z[c].resize(last_chunk_size, n_tests);
        ws.Bound_U_Phi_Bound_Z[c].resize(last_chunk_size, n_tests);
        ws.prob[c].resize(last_chunk_size, n_tests);
        ws.Phi_Z[c].resize(last_chunk_size, n_tests);
      }
      
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
      ws.lp_array.resize(last_chunk_size, n_class);
      ws.prob_n.resize(last_chunk_size);
      ws.prob_n_recip.resize(last_chunk_size);
      ws.log_sum_result.resize(last_chunk_size);
      ws.container_max_logs.resize(last_chunk_size);
      ws.rowwise_log_sum.resize(last_chunk_size);
      ws.rowwise_prod.resize(last_chunk_size);
      ws.rowwise_sum.resize(last_chunk_size);
      ws.log_lik_chunk.resize(last_chunk_size);
      ws.prob_recip.resize(last_chunk_size, n_tests);
      ////
      for (int c = 0; c < n_class; c++) {
        ws.Upper_Bound_Z[c].resize(last_chunk_size, n_tests); //// ordinal-only
      }
      ////
      ws.phi_Upper_Bound_Z.resize(last_chunk_size, n_tests); //// ordinal-only
      ws.dphi_over_L.resize(last_chunk_size, n_tests); //// ordinal-only
      ws.dZ_dmu_neg.resize(last_chunk_size, n_tests); //// ordinal-only
      
      // Re-bind workspace refs for last chunk
      auto &Z_std_norm = ws.Z_std_norm;
      auto &Bound_Z = ws.Bound_Z;
      auto &Bound_U_Phi_Bound_Z = ws.Bound_U_Phi_Bound_Z;
      auto &prob = ws.prob;
      auto &Phi_Z = ws.Phi_Z;
      auto &y1_log_prob = ws.y1_log_prob;
      auto &phi_Z_recip = ws.phi_Z_recip;
      auto &phi_Bound_Z = ws.phi_Bound_Z;
      auto &u_grad_array_CM_chunk = ws.u_grad_array_CM_chunk;
      auto &common_grad_term_1 = ws.common_grad_term_1;
      auto &y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = ws.y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip;
      auto &y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = ws.y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip;
      auto &prob_rowwise_prod_temp = ws.prob_rowwise_prod_temp;
      auto &prob_recip_rowwise_prod_temp = ws.prob_recip_rowwise_prod_temp;
      auto &prod_container_or_inc_array = ws.prod_container_or_inc_array;
      auto &derivs_chain_container_vec = ws.derivs_chain_container_vec;
      auto &prob_rowwise_prod_temp_all = ws.prob_rowwise_prod_temp_all;
      auto &grad_prob = ws.grad_prob;
      auto &z_grad_term = ws.z_grad_term;
      auto &y_chunk = ws.y_chunk;
      auto &u_array = ws.u_array;
      auto &y_sign = ws.y_sign;
      auto &y_m_y_sign_x_u = ws.y_m_y_sign_x_u;
      auto &u_grad_array_CM_chunk_block = ws.u_grad_array_CM_chunk_block;
      auto &u_unc_vec_chunk = ws.u_unc_vec_chunk;
      auto &u_vec_chunk = ws.u_vec_chunk;
      auto &du_wrt_duu_chunk = ws.du_wrt_duu_chunk;
      auto &d_J_wrt_duu_chunk = ws.d_J_wrt_duu_chunk;
      auto &lp_array = ws.lp_array;
      auto &prob_n = ws.prob_n;
      auto &prob_n_recip = ws.prob_n_recip;
      auto &log_sum_result = ws.log_sum_result;
      auto &container_max_logs = ws.container_max_logs;
      auto &rowwise_sum = ws.rowwise_sum;
      auto &rowwise_prod = ws.rowwise_prod;
      auto &log_lik_chunk = ws.log_lik_chunk;
      auto &prob_recip = ws.prob_recip;
      ////
      auto &Upper_Bound_Z = ws.Upper_Bound_Z; //// ordinal-only
      ////
      auto &phi_Upper_Bound_Z = ws.phi_Upper_Bound_Z; //// ordinal-only
      auto &dphi_over_L = ws.dphi_over_L; //// ordinal-only
      auto &dZ_dmu_neg = ws.dZ_dmu_neg; //// ordinal-only
      
     ////  std::cout << "MVOP: Random checkpoint #37" << std::endl; std::cout.flush();
      
      // ---- Process last chunk (identical logic to inner loop, using vt_*_last) ----
      ////-----------------------------------------------
      u_grad_array_CM_chunk.setZero(); //// reset between chunks as re-using same container
      
      y_chunk = y_ref.middleRows(chunk_size_orig * chunk_counter, chunk_size).cast<double>();
      
      //// Nuisance parameter transformation step
      u_unc_vec_chunk = theta_us_vec_ref.segment(chunk_size_orig * n_tests * chunk_counter, chunk_size * n_tests);
      fn_MVP_compute_nuisance(u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);
      log_jac_u += fn_MVP_compute_nuisance_log_jac_u(u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);
      
      u_array = u_vec_chunk.reshaped(chunk_size, n_tests);
      y_sign.array() = y_chunk.array() + (y_chunk.array() - 1.0);
      y_m_y_sign_x_u.array() = y_chunk.array() - (y_sign.array() * u_array.array());
      ////-----------------------------------------------
      
     ////  std::cout << "MVOP: Random checkpoint #38" << std::endl; std::cout.flush();
      
      // ---- Class loop (identical to original) ----
      //// START of c loop
      for (int c = 0; c < n_class; c++) {
        
            prod_container_or_inc_array.setZero(); //// reset to 0
            
            for (int t = 0; t < n_tests; t++) {
              
                  ////-----------------------------------------------
                  if (n_covariates_max > 1) {
                    
                        Eigen::Matrix<double, -1, 1> Xbeta_given_class_c_col_t = X[c][t].block(chunk_size_orig * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(c, t)).cast<double>() *
                                                                                 beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t));
                    
                        if (t < n_binary_tests) { // For binary: Bound_Z = L_recip * (-Xbeta - prod) 
                        
                            Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * (-1.0 * (Xbeta_given_class_c_col_t.array() + prod_container_or_inc_array.array()));
                            
                        } else { // Store μ_t temporarily, compute bounds per-observation
                          
                            int t_ord = t - n_binary_tests;
                            int n_thr_t = n_thr_per_ord_test(t_ord);
                            
                            for (int i = 0; i < chunk_size; ++i) {
                              
                                  double mu_t = Xbeta_given_class_c_col_t(i) + prod_container_or_inc_array(i);
                                  int y_i = static_cast<int>(y_chunk(i, t));
                                  double L_tt = L_Omega_double[c](t, t);
                                  
                                  double lb = (y_i == 1) ? -Inf : (C[c](y_i - 2, t_ord) - mu_t) / L_tt; // Lower bound
                                  
                                  double ub = (y_i == n_cat_per_ord_test(t_ord)) ? Inf : (C[c](y_i - 1, t_ord) - mu_t) / L_tt; // Upper bound  
                                  
                                  // Store for GHK:
                                  Bound_Z[c](i, t) = lb;           // lower_bz (re-using Bound_Z container for GHK lower-bound)
                                  Upper_Bound_Z[c](i, t) = ub;     // upper_bz (using the "Upper_Bound_Z", ordinal-only workspace matrix)
                              
                            } // end of i loop
                          
                        }
                    
                  } else {
                    
                          if (t < n_binary_tests) { // For binary: Bound_Z = L_recip * (-Xbeta - prod) 
                     
                              Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * (-1.0 * (beta_double_array[c](0, t) + prod_container_or_inc_array.array()));
                          
                          } else { // Store μ_t temporarily, compute bounds per-observation
                              
                              int t_ord = t - n_binary_tests;
                              int n_thr_t = n_thr_per_ord_test(t_ord);
                              
                              for (int i = 0; i < chunk_size; ++i) {
                                    
                                    double mu_t = beta_double_array[c](0, t) + prod_container_or_inc_array(i);
                                    int y_i = static_cast<int>(y_chunk(i, t));
                                    double L_tt = L_Omega_double[c](t, t);
                                    
                                    double lb = (y_i == 1) ? -Inf : (C[c](y_i - 2, t_ord) - mu_t) / L_tt; // Lower bound
                                    
                                    double ub = (y_i == n_cat_per_ord_test(t_ord)) ? Inf : (C[c](y_i - 1, t_ord) - mu_t) / L_tt; // Upper bound  
                                    
                                    // Store for GHK:
                                    Bound_Z[c](i, t) = lb;           // lower_bz (re-using Bound_Z container for GHK lower-bound)
                                    Upper_Bound_Z[c](i, t) = ub;     // upper_bz (using the "Upper_Bound_Z", ordinal-only workspace matrix)
                                
                              } // end of i loop
                            
                          }
                     
                  }
                  ////-----------------------------------------------
                  ////-----------------------------------------------
                  //// compute/update important log-lik quantities for GHK-MVP
                  if (t < n_binary_tests) {
                          
                        fn_MVP_compute_lp_GHK_cols( t,
                                                    Bound_U_Phi_Bound_Z[c], 
                                                    Phi_Z[c],
                                                    Z_std_norm[c], 
                                                    prob[c],
                                                    y1_log_prob, 
                                                    Bound_Z[c], 
                                                    y_chunk,
                                                    u_array, 
                                                    Model_args_as_cpp_struct);
                    
                  } else { // if test t is ordinal
                    
                        fn_MVOP_compute_lp_GHK_cols( t,
                                                     Bound_U_Phi_Bound_Z[c], 
                                                     Phi_Z[c],
                                                     Z_std_norm[c], 
                                                     prob[c],
                                                     y1_log_prob,
                                                     Bound_Z[c],   // lower_bz for ordinal
                                                     y_chunk,
                                                     u_array, 
                                                     Upper_Bound_Z[c], // ordinal-only - upper_bz (ordinal only, ignored for binary)
                                                     // Phi_Upper_Bound_Z,  // Φ(upper_bz) (ordinal only)
                                                     n_binary_tests, // ordinal-only
                                                     Model_args_as_cpp_struct);
                    
                  }
                  ////----------------------------------------------
                  
                  ////----------------------------------------------
                  if (t < n_tests - 1) {
                        auto L_Omega_row = L_Omega_double[c].row(t + 1);
                        prod_container_or_inc_array = Z_std_norm[c].leftCols(t + 1) * L_Omega_row.head(t + 1).transpose();
                  }
                  ////-----------------------------------------------
              
            } //// end of t loop
            
            ////-----------------------------------------------
            if (n_class > 1) { // if latent class
                rowwise_sum = y1_log_prob.rowwise().sum();
                rowwise_sum.array() += log_prev(0, c);
                lp_array.col(c) = rowwise_sum;
            } else {
                rowwise_sum = y1_log_prob.rowwise().sum();
                lp_array.col(0) = rowwise_sum;
            }
            ////-----------------------------------------------
            
      } //// end of c loop
      
      ////-----------------------------------------------
      if (n_class > 1) {
            
              log_sum_exp_general( lp_array, 
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
      
      
     ////  std::cout << "MVOP: Random checkpoint #39" << std::endl; std::cout.flush();
      
      /////////////////  ------------------------- compute grad  ---------------------------------------------------------------------------------
      for (int c = 0; c < n_class; c++) {
        
        ////-----------------------------------------------
        prob_recip = stan::math::inv(prob[c]);
        ////-----------------------------------------------
        
        ////-----------------------------------------------
        //// compute/update important log-lik quantities for GHK-MVP
        for (int t = 0; t < n_tests; t++) {
          
              fn_MVP_compute_phi_Z_recip_cols( t,
                                               phi_Z_recip, // computing this
                                               Phi_Z[c], Z_std_norm[c], Model_args_as_cpp_struct);
              
              if (t < n_binary_tests) {
                
                    fn_MVP_compute_phi_Bound_Z_cols(t,
                                                    phi_Bound_Z, // computing this
                                                    Bound_U_Phi_Bound_Z[c], Bound_Z[c], Model_args_as_cpp_struct);
                
              } else { // for ordinal tests
                
                    // φ(lower_bz) — stored in phi_Bound_Z
                    phi_Bound_Z.col(t).array() = (-(Bound_Z[c].col(t).array().square()) * 0.5).exp() * sqrt_2_pi_recip;
                    
                    // φ(upper_bz) — stored in phi_Upper_Bound_Z
                    phi_Upper_Bound_Z.col(t).array() = (-(Upper_Bound_Z[c].col(t).array().square()) * 0.5).exp() * sqrt_2_pi_recip;
                    
                    // Handle ±∞ bounds → φ = 0
                    for (int i = 0; i < chunk_size; ++i) {
                      if (std::isinf(Bound_Z[c](i, t)))       phi_Bound_Z(i, t) = 0.0;
                      if (std::isinf(Upper_Bound_Z[c](i, t))) phi_Upper_Bound_Z(i, t) = 0.0;
                    }
                
              }
          
        }
        
       ////  std::cout << "MVOP: Random checkpoint #40" << std::endl; std::cout.flush();
        
        ////-----------------------------------------------
        if (grad_option != "none") {
          
             fn_MVOP_grad_prep( prob[c],
                                y_sign,
                                y_m_y_sign_x_u,
                                u_array,
                                L_Omega_recip_double[c],
                                prev(0, c), 
                                prob_n_recip, 
                                phi_Z_recip,
                                phi_Bound_Z, 
                                phi_Upper_Bound_Z, //// for ordinal-only fn
                                prob_recip,
                                prob_rowwise_prod_temp, 
                                prob_recip_rowwise_prod_temp,
                                prob_rowwise_prod_temp_all,
                                common_grad_term_1,
                                dphi_over_L, //// for ordinal-only fn
                                dZ_dmu_neg, //// for ordinal-only fn
                                n_binary_tests, //// for ordinal-only fn
                                // y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                // y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                Model_args_as_cpp_struct);
        }
        
       ////  std::cout << "MVOP: Random checkpoint #41" << std::endl; std::cout.flush();
        
        ////////////////////////////////////////////////////////////////////////////////////  Grad of nuisance parameters / u's (manual)
        // ---- Nuisance gradients (write to non-overlapping out_mat segment) ----
        if ((grad_option == "us_only") || (grad_option == "all")) {
          
              u_grad_array_CM_chunk_block = u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests);
          
              //// NOTE: this uses the Call BINARY gradient fn's with "dphi_over_L" and "dZ_dmu_neg"
              ////    as the "y_sign_chunk_times..." and "y_m_ysign_x_u_array_times..." arguments:
              fn_MVP_compute_nuisance_grad_v2( u_grad_array_CM_chunk_block,
                                               phi_Z_recip, 
                                               common_grad_term_1,
                                               L_Omega_double[c],
                                               prob[c],
                                               prob_recip, 
                                               prob_rowwise_prod_temp,
                                               dphi_over_L,
                                               dZ_dmu_neg,
                                               // y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                               // y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                               z_grad_term,
                                               grad_prob, 
                                               prod_container_or_inc_array,
                                               derivs_chain_container_vec,
                                               Model_args_as_cpp_struct);
              
              u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests).array() += u_grad_array_CM_chunk_block.array();
              
              const int start_index = 1 + (chunk_size_orig * n_tests * chunk_counter);
              const int length = chunk_size * n_tests;
              
              if (c == n_class - 1) {
                
                    //// update output vector once all u_grad computations are done
                    out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();
                
                    //// account for unconstrained -> constrained transformations and Jacobian adjustments
                    fn_MVP_nuisance_first_deriv( du_wrt_duu_chunk, 
                                                 u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);
                    
                    fn_MVP_nuisance_deriv_of_log_det_J( d_J_wrt_duu_chunk,
                                                        u_vec_chunk, u_unc_vec_chunk, du_wrt_duu_chunk, Model_args_as_cpp_struct);
                    
                    out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
                    out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();
                
              }
          
        }
        
       ////  std::cout << "MVOP: Random checkpoint #42" << std::endl; std::cout.flush();
        
        /////////////////////////////////////////////////////////////////////////// Grad of intercepts / coefficients (beta's)
        // ---- Coefficient gradients ----
        if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only")) {
              
              fn_MVP_compute_coefficients_grad_v3( c,
                                                   beta_grad_array[c],  // direct to shared (serial)
                                                   chunk_counter,
                                                   n_covariates_max,
                                                   common_grad_term_1,
                                                   L_Omega_double[c],
                                                   prob[c], 
                                                   prob_recip, 
                                                   prob_rowwise_prod_temp,
                                                   dphi_over_L,
                                                   dZ_dmu_neg,
                                                   // y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                   // y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                   z_grad_term,
                                                   grad_prob,
                                                   prod_container_or_inc_array,
                                                   derivs_chain_container_vec, 
                                                   true, 
                                                   Model_args_as_cpp_struct);
          
        }
        
       ////  std::cout << "MVOP: Random checkpoint #43" << std::endl; std::cout.flush();
        
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of L_Omega ('s)
        // ---- L_Omega gradients ----
        if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only")) {
          
              fn_MVP_compute_L_Omega_grad_v3( U_Omega_grad_array[c],  // direct to shared (serial)
                                              common_grad_term_1, 
                                              L_Omega_double[c],
                                              prob[c],
                                              prob_recip,
                                              Bound_Z[c], 
                                              Z_std_norm[c],
                                              prob_rowwise_prod_temp,
                                              dphi_over_L,
                                              dZ_dmu_neg,
                                              // y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                              // y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                              z_grad_term, 
                                              grad_prob, 
                                              prod_container_or_inc_array,
                                              derivs_chain_container_vec, 
                                              true,
                                              Model_args_as_cpp_struct);
          
        }
        
       ////  std::cout << "MVOP: Random checkpoint #44" << std::endl; std::cout.flush();
        
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of cutpoints
        // ---- Cutpoint gradients ----
        if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "cutpoints_only")) {
          
              for (int t_ord = 0; t_ord < n_ordinal_tests; ++t_ord) {
                
                int t = n_binary_tests + t_ord;
                int K_t = n_thr_per_ord_test(t_ord) + 1;  // n_cat
                
                fn_MVOP_compute_cutpoint_grad(   c, 
                                                 cutpoint_grad_array[c],  // (n_cutpoints_max × n_ordinal_tests)  // <<< LOCAL accumulator
                                                 t, 
                                                 t_ord, 
                                                 K_t,
                                                 common_grad_term_1,
                                                 L_Omega_double[c],
                                                 prob[c],
                                                 prob_recip, 
                                                 prob_rowwise_prod_temp,
                                                 phi_Bound_Z,
                                                 phi_Upper_Bound_Z,
                                                 phi_Z_recip,
                                                 u_array,
                                                 y_chunk,
                                                 dphi_over_L,
                                                 dZ_dmu_neg,
                                                 // y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                 // y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                 z_grad_term,
                                                 grad_prob,
                                                 prod_container_or_inc_array,
                                                 derivs_chain_container_vec,
                                                 true,
                                                 Model_args_as_cpp_struct);
              }
          
        }
        
       ////  std::cout << "MVOP: Random checkpoint #45" << std::endl; std::cout.flush();
        
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Prev grad (LC-MVP/LC-MVOP only)
        // ---- Prevalence gradient ----
        if (n_class > 1) {
            
            if ((grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only")) {
              
                  rowwise_prod = prob[c].rowwise().prod();
                  rowwise_prod.head(chunk_size).array() = prob_n_recip.head(chunk_size).array() * rowwise_prod.head(chunk_size).array();
                  prev_grad_vec(c) += rowwise_prod.head(chunk_size).sum();
              
            }
          
        }
        
       ////  std::cout << "MVOP: Random checkpoint #46" << std::endl; std::cout.flush();
        
      }
      
    }  // end of parallel for over chunks
    
   ////  std::cout << "MVOP: Random checkpoint #47" << std::endl; std::cout.flush();
    
    // ============================================================================
    // [UNCHANGED] Post-loop: prevalence gradient, log_prob, output assembly
    // ============================================================================
    Eigen::Matrix<double, -1, 1> prev_unconstrained_grad_vec = Eigen::Matrix<double, -1, 1>::Zero(n_class);
    Eigen::Matrix<double, -1, 1> prev_unconstrained_grad_vec_out = Eigen::Matrix<double, -1, 1>::Zero(n_class - 1);

    ////////////////////////  --------------------------------------------------------------------------
    if (n_class > 1) {
      for (int c = 0; c < n_class; c++) {
        prev_unconstrained_grad_vec(c)  =   prev_grad_vec(c) * deriv_p_wrt_pu_double ;
      }
      prev_unconstrained_grad_vec(0) = prev_unconstrained_grad_vec(1) - prev_unconstrained_grad_vec(0) - 2 * tanh_u_prev[1];
      prev_unconstrained_grad_vec_out(0) = prev_unconstrained_grad_vec(0);
    }

   ////  std::cout << "MVOP: Random checkpoint #48" << std::endl; std::cout.flush();
    
    ////////////////////////  --------------------------------------------------------------------------
    log_prob_out +=  out_mat.tail(N).sum();  ////  log_lik
    log_prob_out +=  log_jac_u;
    if (exclude_priors == false)  log_prob_out += prior_densities;
    log_prob_out +=  log_det_J_main ; // log_jac_p_double;
    
   ////  std::cout << "MVOP: Random checkpoint #49" << std::endl; std::cout.flush();
    
    // Pack beta gradients into vector
    Eigen::Matrix<double, -1, 1> beta_grad_vec = Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);
    {
      int i = 0;
      for (int c = 0; c < n_class; c++ ) {
        for (int t = 0; t < n_tests; t++) {
          for (int k = 0; k <  n_covariates_per_outcome_vec(c, t); k++) {
            beta_grad_vec(i) = beta_grad_array[c](k, t);
            i += 1;
          }
        }
      }
    }
    
   ////  std::cout << "MVOP: Random checkpoint #50" << std::endl; std::cout.flush();
    
    // Pack L_Omega gradients and chain-rule through deriv_L_wrt_unc_full
    Eigen::Matrix<double, -1, 1> L_Omega_grad_vec(n_corrs + (n_class * n_tests));
    Eigen::Matrix<double, -1, 1> U_Omega_grad_vec(n_corrs);
    {
      int i = 0;
      for (int c = 0; c < n_class; c++) {
        for (int t1 = 0; t1 < n_tests; t1++) {
          for (int t2 = 0; t2 <  t1 + 1; t2++) {
            L_Omega_grad_vec(i) = U_Omega_grad_array[c](t1,t2);
            i += 1;
          }
        }
      }
    }
    
   ////  std::cout << "MVOP: Random checkpoint #51" << std::endl; std::cout.flush();
    
    Eigen::Matrix<double, -1, 1>  grad_wrt_L_Omega_nd(dim_choose_2 + n_tests);
    Eigen::Matrix<double, -1, 1>  grad_wrt_L_Omega_d(dim_choose_2 + n_tests);

    if (n_class > 1) {
      grad_wrt_L_Omega_nd =   L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
      grad_wrt_L_Omega_d =   L_Omega_grad_vec.segment(dim_choose_2 + n_tests, dim_choose_2 + n_tests);
      U_Omega_grad_vec.head(dim_choose_2) =  ( grad_wrt_L_Omega_nd.transpose()  *  deriv_L_wrt_unc_full[0]  ).transpose() ;
      U_Omega_grad_vec.segment(dim_choose_2, dim_choose_2) =   ( grad_wrt_L_Omega_d.transpose()  *  deriv_L_wrt_unc_full[1] ).transpose()  ;
    } else {
      grad_wrt_L_Omega_nd =   L_Omega_grad_vec.head(dim_choose_2 + n_tests);
      U_Omega_grad_vec.head(dim_choose_2) =  ( grad_wrt_L_Omega_nd.transpose()  *  deriv_L_wrt_unc_full[0] ).transpose() ;
    }
    
   ////  std::cout << "MVOP: Random checkpoint #52" << std::endl; std::cout.flush();

    // ---- Final output assembly ----
    ////////////////////////////  outputs // add log grad and sign stuff';///////////////
    out_mat(0) =  log_prob_out;
    out_mat.segment(1 + n_us, n_corrs) += U_Omega_grad_vec ;
    out_mat.segment(1 + n_us + n_corrs, n_covariates_total) += beta_grad_vec ;  /// no Jacobian needed
    if (n_class > 1) {
       out_mat(1 + n_us + n_corrs + n_covariates_total) += prev_unconstrained_grad_vec_out(0) ;  
    }
    
    // add derivative of normal prior density to beta / coeffs gradient
    // ---- Prior gradient contribution to coefficients ----
    {
      int i = n_us + n_corrs + 1; /// + 1 because first element is the log_prob !!
      for (int c = 0; c < n_class; c++) {
        for (int t = 0; t < n_tests; t++) {
          for (int k = 0; k <  n_covariates_per_outcome_vec(c, t); k++) {
            if (exclude_priors == false) {
                out_mat(i) += - ((beta_double_array[c](k, t) - prior_coeffs_mean[c](k, t)) / prior_coeffs_sd[c](k, t) ) * (1.0 / prior_coeffs_sd[c](k, t) ) ;
                i += 1;
            }
          }
        }
      }
    }
    
   ////  std::cout << "MVOP: Random checkpoint #53" << std::endl; std::cout.flush();

}




inline void fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                      const std::string &grad_option,
                                                                      const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                      std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                      const int n_threads_WCP
) {

      // if (n_threads_WCP == 1) {
      //   
      //       fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial( out_mat,
      //                                                                         theta_main_vec_ref,
      //                                                                         theta_us_vec_ref,
      //                                                                         y_ref,
      //                                                                         grad_option,
      //                                                                         Model_args_as_cpp_struct,
      //                                                                         LC_MVP_ws_structs[0]);
      //   
      // } else {
            
            fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_WCP( out_mat,
                                                                           theta_main_vec_ref,
                                                                           theta_us_vec_ref,
                                                                           y_ref,
                                                                           grad_option,
                                                                           Model_args_as_cpp_struct,
                                                                           LC_MVP_ws_structs,
                                                                           n_threads_WCP);
        
      // } 

} 




//// Internal function using Eigen::Ref as inputs for matrices
inline void fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_InPlace( Eigen::Matrix<double, -1, 1> &&out_mat_R_val,
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
    
      fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(     out_mat_ref,
                                                                      theta_main_vec_ref,
                                                                      theta_us_vec_ref,
                                                                      y_ref,
                                                                      grad_option,
                                                                      Model_args_as_cpp_struct,
                                                                      LC_MVP_ws_structs,
                                                                      n_threads_WCP);

}




//// Internal function using Eigen::Ref as inputs for matrices
inline void fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_InPlace( Eigen::Matrix<double, -1, 1> &out_mat_ref,
                                                                const Eigen::Matrix<double, -1, 1> &theta_main_vec_ref,
                                                                const Eigen::Matrix<double, -1, 1> &theta_us_vec_ref,
                                                                const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                const std::string &grad_option,
                                                                const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                const int n_threads_WCP
) {
    
      fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process( out_mat_ref,
                                                                  theta_main_vec_ref,
                                                                  theta_us_vec_ref,
                                                                  y_ref,
                                                                  grad_option,
                                                                  Model_args_as_cpp_struct,
                                                                  LC_MVP_ws_structs,
                                                                  n_threads_WCP);
  
}




//// Internal function using Eigen::Ref as inputs for matrices
template <typename MatrixType>
inline void fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_InPlace( Eigen::Ref<Eigen::Block<MatrixType, -1, 1>>  &out_mat_ref,
                                                                const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_main_vec_ref,
                                                                const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_us_vec_ref,
                                                                const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                const std::string &grad_option,
                                                                const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                const int n_threads_WCP
) {

      fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process( out_mat_ref,
                                                                  theta_main_vec_ref,
                                                                  theta_us_vec_ref,
                                                                  y_ref,
                                                                  grad_option,
                                                                  Model_args_as_cpp_struct,
                                                                  LC_MVP_ws_structs,
                                                                  n_threads_WCP);
  
}




//// Internal function using Eigen::Ref as inputs for matrices
inline Eigen::Matrix<double, -1, 1> fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD( const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
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
    
      fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_InPlace( out_mat,
                                                          theta_main_vec_ref,
                                                          theta_us_vec_ref,
                                                          y_ref,
                                                          grad_option,
                                                          Model_args_as_cpp_struct,
                                                          LC_MVP_ws_structs,
                                                          n_threads_WCP);
      
      return out_mat;

}


 








  