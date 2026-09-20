
#pragma once 
 
#ifndef FN_WRAPPERS_LOOP_HPP
#define FN_WRAPPERS_LOOP_HPP
 
 
 
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


 
 
template <typename T,  typename FuncDouble>
inline void fn_Loop_row_or_col_vector(      Eigen::Ref<T>  x, 
                                                   const FuncDouble &fn_double) {
  
   const int N = x.size();
   
     for (int i = 0; i < N; ++i) {
       x(i) = fn_double(x(i));
     }
   
}
 
 


 
 
template<typename T, typename FuncDouble>
inline void fn_Loop_matrix(  Eigen::Ref<T> x,
                                    const FuncDouble &fn_double) {
     
     const int rows = x.rows(); 
     const int cols = x.cols();
      
     for (int j = 0; j < cols; ++j) {
           for (int i = 0; i < rows; ++i) {
             x(i, j) = fn_double(x(i, j));
           }
     }
 
}

  
  
  
  
template <typename T, typename FuncDouble>
inline void fn_Loop_dbl_Eigen( Eigen::Ref<T> x, 
                                      const FuncDouble &fn_double) {
  
     constexpr int n_rows = T::RowsAtCompileTime;
     constexpr int n_cols = T::ColsAtCompileTime;
     
     if constexpr (n_rows == 1 && n_cols == -1) {   // Row vector case
       
       fn_Loop_row_or_col_vector(x, fn_double);
       
     } else if constexpr (n_rows == -1 && n_cols == 1) {  // Column vector case
       
       fn_Loop_row_or_col_vector(x, fn_double);
       
     } else {   // General matrix case
       
       fn_Loop_matrix(x, fn_double);
       
     }
 
}

 
  
 
  


template<typename FuncDouble, typename FuncDouble_wo_checks, typename T>
inline void    fn_process_double_Loop_sub_function(     Eigen::Ref<T> x,  
                                                               const FuncDouble &fn_fast_double_function,
                                                               const FuncDouble_wo_checks &fn_fast_double_function_wo_checks, 
                                                               const bool skip_checks) {
  
  if (skip_checks == false) {
    
    fn_Loop_dbl_Eigen(x, fn_fast_double_function);
    
  }   else  {
    
    fn_Loop_dbl_Eigen(x, fn_fast_double_function_wo_checks);
    
  }
  
}



 
 
 

template <typename T>
inline   void        fn_return_Loop(   Eigen::Ref<T> x,
                                       const std::string &fn,
                                       const bool &skip_checks) {
  
      if        (fn == "exp") {                              fn_process_double_Loop_sub_function(x, fast_exp_1, fast_exp_1_wo_checks, skip_checks) ;
      } else if (fn == "log") {                              fn_process_double_Loop_sub_function(x, fast_log_1, fast_log_1_wo_checks, skip_checks) ;
      } else if (fn == "log1p") {                            fn_process_double_Loop_sub_function(x, fast_log1p_1, fast_log1p_1_wo_checks, skip_checks) ;
      } else if (fn == "log1m") {                            fn_process_double_Loop_sub_function(x, fast_log1m_1, fast_log1m_1_wo_checks, skip_checks) ;
      } else if (fn == "tanh") {                             fn_process_double_Loop_sub_function(x, fast_tanh, fast_tanh_wo_checks, skip_checks) ;
      } else if (fn == "Phi_approx") {                       fn_process_double_Loop_sub_function(x, fast_Phi_approx, fast_Phi_approx_wo_checks, skip_checks) ;
      } else if (fn == "log_Phi_approx") {                   fn_process_double_Loop_sub_function(x, fast_log_Phi_approx, fast_log_Phi_approx_wo_checks, skip_checks) ;
      } else if (fn == "inv_Phi_approx") {                   fn_process_double_Loop_sub_function(x, fast_inv_Phi_approx, fast_inv_Phi_approx_wo_checks, skip_checks) ;
      } else if (fn == "inv_Phi_approx_from_logit_prob") {   fn_process_double_Loop_sub_function(x, fast_inv_Phi_approx_from_logit_prob, fast_inv_Phi_approx_from_logit_prob_wo_checks, skip_checks) ;
      } else if (fn == "Phi") {                              fn_process_double_Loop_sub_function(x, fast_Phi, fast_Phi_wo_checks, skip_checks) ;
      } else if (fn == "inv_Phi") {                          fn_process_double_Loop_sub_function(x, fast_inv_Phi_wo_checks, fast_inv_Phi_wo_checks, skip_checks) ;
      } else if (fn == "inv_logit") {                        fn_process_double_Loop_sub_function(x, fast_inv_logit, fast_inv_logit_wo_checks, skip_checks) ;
      } else if (fn == "log_inv_logit") {                    fn_process_double_Loop_sub_function(x, fast_log_inv_logit, fast_log_inv_logit_wo_checks, skip_checks) ;
      }
  
}





 
 
 




inline Eigen::Matrix<double, -1, 1  >   fast_log_sum_exp_2d_double(   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  x )  {
  
  int N = x.rows();
  Eigen::Matrix<double, -1, -1  > rowwise_maxes_2d_array(N, 2);
  rowwise_maxes_2d_array.col(0) = x.array().rowwise().maxCoeff().matrix();
  rowwise_maxes_2d_array.col(1) = rowwise_maxes_2d_array.col(0); 
  
  // std::function<double(double)>  exp_fn =  static_cast<double(*)(double const &)>(fast_exp_1);
  // std::function<double(double)>  log_fn =  static_cast<double(*)(double const &)>(fast_log_1);
  
  return      rowwise_maxes_2d_array.col(0) +
    fn_colvec_loop_dbl_Eigen(  fn_mat_loop_dbl_Eigen( (x  - rowwise_maxes_2d_array).matrix(),  static_cast<double(*)(double const)>(fast_exp_1)).array().rowwise().sum().abs().matrix(), static_cast<double(*)(double const)>(fast_log_1)   )   ;
  
  
}




 
  

#endif

  
  
  
  