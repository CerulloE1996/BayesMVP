
#pragma once 


#ifndef FN_WRAPPERS_LOG_SUM_EXP_DBL_HPP
#define FN_WRAPPERS_LOG_SUM_EXP_DBL_HPP


#include <stan/math/prim/fun/sqrt.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/prob/std_normal_log_qf.hpp>
#include <stan/math/prim/fun/Phi.hpp>
#include <stan/math/prim/fun/inv_Phi.hpp>
#include <stan/math/prim/fun/Phi_approx.hpp>
#include <stan/math/prim/fun/tanh.hpp>
#include <stan/math/prim/fun/log_inv_logit.hpp>


#include <Eigen/Dense>
#include <Eigen/Core>


#include <immintrin.h>




using namespace Eigen;




ALWAYS_INLINE  void log_sum_exp_general(      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_vals,  
                                              const std::string &vect_type_exp,
                                              const std::string &vect_type_log,
                                              Eigen::Ref<Eigen::Matrix<double, -1, 1>> log_sum_result,
                                              Eigen::Ref<Eigen::Matrix<double, -1, 1>> container_max_logs) {
    
    // find max for each row across all columns
    container_max_logs = log_vals.rowwise().maxCoeff();
    //// const Eigen::Matrix<double, -1, 1>  temp_container_max_logs = log_vals.rowwise().maxCoeff(); // this works on windows  //// BOOKMARK
    const Eigen::Matrix<double, -1, -1> temp = (log_vals.colwise() - container_max_logs);
    const Eigen::Matrix<double, -1, -1> temp_2 = fn_EIGEN_double(temp, "exp", vect_type_exp);
    const Eigen::Matrix<double, -1, 1>  sum_exp =  temp_2.rowwise().sum();
    const Eigen::Matrix<double, -1, 1>  sum_exp_abs = stan::math::abs(sum_exp);
    /// up to here OK on windows
    log_sum_result = fn_EIGEN_double(sum_exp_abs, "log", vect_type_log);
    log_sum_result.array() += container_max_logs.array();
   
}






//// with optional additional underflow protection (can be commented out easily)
ALWAYS_INLINE void log_abs_sum_exp_general_v2(   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_vals,
                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> signs,
                                                 const std::string &vect_type_exp,
                                                 const std::string &vect_type_log,
                                                 Eigen::Ref<Eigen::Matrix<double, -1, 1>> log_sum_abs_result,
                                                 Eigen::Ref<Eigen::Matrix<double, -1, 1>> sign_result,
                                                 Eigen::Ref<Eigen::Matrix<double, -1, 1>> container_max_logs,
                                                 Eigen::Ref<Eigen::Matrix<double, -1, 1>> container_sum_exp_signed) {
     
    // const double min_exp_neg = -700.0;
    // const double max_exp_arg =  700.0;
    // const double tiny = stan::math::exp(min_exp_neg);
     
    //container_max_logs = log_abs_vals.rowwise().maxCoeff();    //// Find max log_abs value for each row 
    //// const Eigen::Matrix<double, -1, 1>  temp_container_max_logs = log_abs_vals.rowwise().maxCoeff();  // this works on windows  //// BOOKMARK
    container_max_logs = log_abs_vals.rowwise().maxCoeff();
    
    // const Eigen::Matrix<double, -1, -1>  &shifted_logs = (log_abs_vals.colwise() - container_max_logs);   //// Compute shifted logs with underflow protection
    const Eigen::Matrix<double, -1, -1>  shifted_logs = (log_abs_vals.colwise() - container_max_logs);   //// Compute shifted logs with underflow protection
    //  Eigen::Matrix<double, -1, -1>  shifted_logs = (log_abs_vals.colwise() - container_max_logs);   //// Compute shifted logs with underflow protection
     
    //// Compute exp terms and sum over columns with signs 
    const Eigen::Matrix<double, -1, -1> temp = (log_abs_vals.colwise() - container_max_logs);
    const Eigen::Matrix<double, -1, -1> temp_2 = fn_EIGEN_double(temp, "exp", vect_type_exp);
    const Eigen::Matrix<double, -1, -1> temp_3 = (temp_2.array() * signs.array()).matrix();
    //// const Eigen::Matrix<double, -1, 1>  container_sum_exp_signed_temp = temp_3.rowwise().sum();  // this works on windows  //// BOOKMARK
    container_sum_exp_signed = temp_3.rowwise().sum();
    const Eigen::Matrix<double, -1, 1>  container_sum_exp_signed_temp_signed = stan::math::sign(container_sum_exp_signed);
    const Eigen::Matrix<double, -1, 1>  container_sum_exp_signed_temp_abs = stan::math::abs(container_sum_exp_signed);
    const Eigen::Matrix<double, -1, 1>  container_sum_exp_signed_temp_log_abs = fn_EIGEN_double( container_sum_exp_signed_temp_abs, "log", vect_type_log);
    
    //// Compute sign_result and log_sum_abs_result
    sign_result = container_sum_exp_signed_temp_signed;
    log_sum_abs_result = container_max_logs + container_sum_exp_signed_temp_log_abs;
    
    // sign_result(i) = std::copysign(1.0, sum_exp);
    // log_sum_abs_result(i) = container_max_logs(i) +  stan::math::log(stan::math::abs(sum_exp));
    
    // for (Eigen::Index i = 0; i < container_sum_exp_signed.rows(); ++i) {
    // 
    //       double sum_exp = container_sum_exp_signed(i);
    // 
    //       if (stan::math::abs(sum_exp) < tiny) {   //  if exp's cancel out or are too small
    // 
    //             sign_result(i) = 0.0;
    //             log_sum_abs_result(i) = min_exp_neg;
    // 
    //       } else {  // Normal case
    // 
    //             sign_result(i) = std::copysign(1.0, sum_exp); 
    //             log_sum_abs_result(i) = container_max_logs(i) +  stan::math::log(stan::math::abs(sum_exp));
    //       }
    // 
    // }
    
   
} 







 


struct LogSumVecSingedResult { 
  
     double log_sum;
     double sign;
 
};
 
 
 
 
 
ALWAYS_INLINE  LogSumVecSingedResult log_sum_vec_signed_v1(   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> log_abs_vec,
                                                              const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> signs,
                                                              const std::string &vect_type) {
   
             // const double huge_neg = -700.0;
             double max_log_abs = stan::math::max(log_abs_vec);  // find max 
             
             //// every term exactly zero (log_abs_vec = -Inf throughout). Without this, -Inf - (-Inf) = NaN
             //// made the sum NaN; same guard as log_sum_vec_signed_T (MVP_log_scale_grad_calc_fns_T.hpp). Callers here: the latent-trait
             //// log-scale helpers (LC_LT_*, whose PartialLog path is disabled in lp_grad_model_selector.hpp) and the uncalled original MVP ones.
             if (std::isinf(max_log_abs) && (max_log_abs < 0.0)) {
               return {-700.0, 1.0};
             }
           
             const Eigen::Matrix<double, -1, 1> shifted_logs = (log_abs_vec.array() - max_log_abs);   ///// Shift logs and clip
             // shifted_logs = (shifted_logs.array() < huge_neg).select(huge_neg, shifted_logs);   ///// additionally clip (can comment out for no clipping)
             
             // Compute sum with signs carefully
             const Eigen::Matrix<double, -1, 1> exp_terms = fn_EIGEN_double((log_abs_vec.array() - max_log_abs), "exp", vect_type);
             double sum_exp = (signs.array() * exp_terms.array()).sum();
             
             // // Handle near-zero sums (optional)
             // if (stan::math::abs(sum_exp) < stan::math::exp(huge_neg)) {
             //   return {huge_neg, 0.0};
             // }
             
             double log_abs_sum = max_log_abs + stan::math::log(stan::math::abs(sum_exp));   
             
             // // Clip final result if too large (optional)
             // if (log_abs_sum > 10.0) {  // exp(10) ≈ 22026, reasonable bound
             //   log_abs_sum = 10.0;
             // }
             
             return {log_abs_sum, sum_exp > 0 ? 1.0 : -1.0};
   
 }

 
 
 
 

 
 
 
 
 

 
ALWAYS_INLINE  void log_abs_matrix_vector_mult_v1(   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_matrix,
                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_matrix,
                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> log_abs_vector,
                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> sign_vector,
                                                     const std::string &vect_type_exp,
                                                     const std::string &vect_type_log,
                                                     Eigen::Ref<Eigen::Matrix<double, -1, 1>> log_abs_result,
                                                     Eigen::Ref<Eigen::Matrix<double, -1, 1>> sign_result) {

   int n_rows = log_abs_matrix.rows();
   int n_cols = log_abs_matrix.cols();

             // Initialize temp storage for max finding pass
             Eigen::Matrix<double, -1, 1> max_logs = Eigen::Matrix<double, -1, 1>::Constant(n_rows, -700.0);

             // First pass: find max_log for each row
             for (int j = 0; j < n_cols; j++) {
               double log_vec_j = log_abs_vector(j);
               for (int i = 0; i < n_rows; i++) {
                 max_logs(i) = std::max(max_logs(i), log_abs_matrix(i, j) + log_vec_j);
               }
             }

             // Second pass: compute sums using exp-trick
             Eigen::Matrix<double, -1, 1> sums = Eigen::Matrix<double, -1, 1>::Zero(n_rows);
             for (int j = 0; j < n_cols; j++) {
               
                   double log_vec_j = log_abs_vector(j);
                   double sign_vec_j = sign_vector(j);
    
                   for (int i = 0; i < n_rows; i++) {
                         double term = std::exp(log_abs_matrix(i, j) + log_vec_j - max_logs(i)) * sign_matrix(i, j) * sign_vec_j;
                         sums(i) += term;
                   }
                   
             }

             // Final pass: compute results
             for (int i = 0; i < n_rows; i++) {
                 sign_result(i) = (sums(i) >= 0) ? 1.0 : -1.0;
                 log_abs_result(i) = std::log(std::abs(sums(i))) + max_logs(i);
             }

 }


 
 
 
 
 
ALWAYS_INLINE void log_abs_matrix_vector_mult_v2(   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> log_abs_matrix,
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> sign_matrix,
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> log_abs_vector,
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> sign_vector,
                                             const std::string &vect_type_exp,
                                             const std::string &vect_type_log,
                                             Eigen::Ref<Eigen::Matrix<double, -1, 1>> log_abs_result,
                                             Eigen::Ref<Eigen::Matrix<double, -1, 1>> sign_result) {
   
   const int n_rows = log_abs_matrix.rows();
   const int n_cols = log_abs_matrix.cols();
   
   // Add broadcasted log_abs_vector to each row of log_abs_matrix
   Eigen::Matrix<double, -1, -1> combined_logs = (log_abs_matrix.rowwise() + log_abs_vector.transpose());
   
   // Find max along rows
   Eigen::Matrix<double, -1, 1> max_logs = combined_logs.rowwise().maxCoeff();
   
   // Compute exp(log_abs - max) * sign for all elements
   /// Eigen::Matrix<double, -1, -1> exp_terms = (combined_logs.colwise() - max_logs).array().exp();
   //// Eigen::Matrix<double, -1, -1> signed_terms = ( (combined_logs.colwise() - max_logs).array().exp().array() * sign_matrix.array() * sign_vector.transpose().array() ) ;
   
   // Sum rows
   Eigen::Matrix<double, -1, 1> sums = ( (combined_logs.colwise() - max_logs).array().exp().array() * sign_matrix.array() * sign_vector.transpose().array() ).rowwise().sum();
   
   // Compute final results
   sign_result = sums.array().sign().max(0.0); // Handles zero case
   log_abs_result = sums.array().abs().log() + max_logs.array();
   
}
 
 
 
 
 
 
 
 
 
 
 
 // ALWAYS_INLINE Eigen::Matrix<double, -1, 1  >   log_sum_exp_2d_Eigen_double( const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  x )  {
 //   
 //   int N = x.rows();
 //   Eigen::Matrix<double, -1, -1> rowwise_maxes_2d_array(N, 2);
 //   rowwise_maxes_2d_array.col(0) = x.array().rowwise().maxCoeff().matrix();
 //   rowwise_maxes_2d_array.col(1) = rowwise_maxes_2d_array.col(0);
 //   
 //   /// Eigen::Matrix<double, -1, 1>  rowwise_maxes_1d_vec = rowwise_maxes_2d_array.col(0);
 //   Eigen::Matrix<double, -1, 1>  sum_exp_vec =  (  (x.array()  -  rowwise_maxes_2d_array.array()).matrix() ).array().exp().matrix().rowwise().sum() ;
 //   
 //   std::string eigen_string = "Eigen";
 //   
 //   log_sum_exp_general(lp_array,
 //                       vect_type_exp,
 //                       vect_type_log,
 //                       log_sum_result,
 //                       container_max_logs);
 //   
 //   return     ( rowwise_maxes_2d_array.col(0).array()    +    sum_exp_vec.array().log() ).matrix() ;
 //   
 //   
 // }



 
 
ALWAYS_INLINE   Eigen::Matrix<double, -1, 1>   log_sum_exp_2d_Eigen_double( const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  x )  {
   
       int N = x.rows();
       
       std::string std_string = "Stan";
       
       Eigen::Matrix<double, -1, 1>  log_sum_result = Eigen::Matrix<double, -1, 1>::Zero(N);
       Eigen::Matrix<double, -1, 1>  container_max_logs = Eigen::Matrix<double, -1, 1>::Zero(N);
       
       log_sum_exp_general(x,
                           std_string,
                           std_string,
                           log_sum_result,
                           container_max_logs);
       
       return   log_sum_result;//   ( rowwise_maxes_2d_array.col(0).array()    +    sum_exp_vec.array().log() ).matrix() ;
   
} 
 
 
 
 
 
 

ALWAYS_INLINE   Eigen::Matrix<double, -1, 1>   log_sum_exp_2d_Stan_double( const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  x )  {
   
       int N = x.rows();
       
       std::string std_string = "Stan";
       
       Eigen::Matrix<double, -1, 1>  log_sum_result = Eigen::Matrix<double, -1, 1>::Zero(N);
       Eigen::Matrix<double, -1, 1>  container_max_logs = Eigen::Matrix<double, -1, 1>::Zero(N);
       
       log_sum_exp_general(x,
                           std_string,
                           std_string,
                           log_sum_result,
                           container_max_logs);
       
       return     log_sum_result;//  ( rowwise_maxes_2d_array.col(0).array()    +    sum_exp_vec.array().log() ).matrix() ;
   
} 

 


 
 
 
 
  
 





#endif

  
  
  
  
  