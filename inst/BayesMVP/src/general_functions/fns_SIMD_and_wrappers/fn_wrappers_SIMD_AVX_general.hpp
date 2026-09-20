
#pragma once 
 
#ifndef FN_WRAPPERS_SIMD_AVX_GENERAL_HPP
#define FN_WRAPPERS_SIMD_AVX_GENERAL_HPP

 
 
#include <Eigen/Dense>
#include <Eigen/Core>
 
 
#if  (defined(USE_AVX2) || defined(USE_AVX_512)) // will only compile if AVX-512 (1st choice) or AVX2 (2nd choice) are available
 
 
#include <immintrin.h>


template <typename T>
ALWAYS_INLINE  void fn_AVX_row_or_col_vector(    Eigen::Ref<T>  x_Ref,
                                                 const FuncAVX &fn_AVX,
                                                 const FuncDouble &fn_double) {
  
        const int N = x_Ref.size();
  
        #if defined(USE_AVX_512)
          const int vect_size = 8;
        #elif defined(USE_AVX2)
          const int vect_size = 4;
        #endif
          
        const double vect_siz_dbl = static_cast<double>(vect_size);
        const double N_dbl = static_cast<double>(N);
        const int N_divisible_by_vect_size = std::floor(N_dbl / vect_siz_dbl) * vect_size;
        
       Eigen::Matrix<double, -1, 1> x_tail = Eigen::Matrix<double, -1, 1>::Zero(vect_size); // last vect_size elements
       {
           int counter = 0;
           for (int i = N - vect_size; i < N; ++i) {
             x_tail(counter) = x_Ref(i);
             counter += 1;
           } 
       }
       
       if (N >= vect_size) {
             
                     for (int i = 0; i + vect_size <= N_divisible_by_vect_size; i += vect_size) {
                       
                         #if defined(USE_AVX_512)
                             __m512d const AVX_array = _mm512_loadu_pd(&x_Ref(i));
                             __m512d const AVX_array_out = fn_AVX(AVX_array);
                             _mm512_storeu_pd(&x_Ref(i), AVX_array_out);
                         #elif defined(USE_AVX2)
                             __m256d const AVX_array = _mm256_loadu_pd(&x_Ref(i));
                             __m256d const AVX_array_out = fn_AVX(AVX_array);
                             _mm256_storeu_pd(&x_Ref(i), AVX_array_out);
                         #endif
                           
                     }
                     
                     if (N_divisible_by_vect_size != N) {    // Handle remainder
                       int counter = 0;
                       for (int i = N - vect_size; i < N; ++i) {
                         x_Ref(i) =  fn_double(x_tail(counter));
                         counter += 1;
                       }
                     }
         
       }  else {   // If N < vect_size, handle everything with scalar operations
         
             for (int i = 0; i < N; ++i) {
               x_Ref(i) = fn_double(x_Ref(i));
             }
         
       }
   
}
 
 
 
 
 
 
 

template <typename T>
ALWAYS_INLINE  void fn_AVX_matrix(    Eigen::Ref<T> x_Ref,
                                      const FuncAVX &fn_AVX, 
                                      const FuncDouble &fn_double) {
     
     const int n_rows = x_Ref.rows();
     const int n_cols = x_Ref.cols();
     
     if (n_rows > n_cols) { // if data in "long" format
       for (int j = 0; j < n_cols; ++j) {   
         Eigen::Matrix<double, -1, 1> x_col = x_Ref.col(j);
         using ColType = decltype(x_col);
         Eigen::Ref<Eigen::Matrix<double, -1, 1>> x_col_Ref(x_col); 
         fn_AVX_row_or_col_vector<ColType>(x_col_Ref, fn_AVX, fn_double);
         x_Ref.col(j) = x_col_Ref;
       }
     } else { 
       for (int j = 0; j < n_rows; ++j) {
         Eigen::Matrix<double, 1, -1> x_row = x_Ref.row(j);
         using RowType = decltype(x_row);
         Eigen::Ref<Eigen::Matrix<double, 1, -1>> x_row_Ref(x_row); 
         fn_AVX_row_or_col_vector<RowType>(x_row_Ref, fn_AVX, fn_double);
         x_Ref.row(j) = x_row_Ref;
       }
     }
   
} 
 
 
 
 
 
 
 

template <typename T>
ALWAYS_INLINE  void fn_AVX_dbl_Eigen(     Eigen::Ref<T> x_Ref, 
                                          const FuncAVX &fn_AVX, 
                                          const FuncDouble &fn_double) {
     
     constexpr int n_rows = T::RowsAtCompileTime;
     constexpr int n_cols = T::ColsAtCompileTime;
     
     if constexpr (n_rows == 1 && n_cols == -1) {   // Row vector case
        
           fn_AVX_row_or_col_vector(x_Ref, fn_AVX, fn_double);
       
     } else if constexpr (n_rows == -1 && n_cols == 1) {  // Column vector case
       
           fn_AVX_row_or_col_vector(x_Ref, fn_AVX, fn_double);
       
     } else {   // General matrix case
       
           fn_AVX_matrix(x_Ref, fn_AVX, fn_double);
       
     }
   
}
  
 
 

 
 
 
  

template <typename T>
ALWAYS_INLINE  void    fn_process_double_AVX_sub_function(    Eigen::Ref<T> x_Ref,  
                                                              const FuncAVX    &fn_fast_AVX_function,
                                                              const FuncDouble &fn_fast_double_function,
                                                              const FuncAVX    &fn_fast_AVX_function_wo_checks,
                                                              const FuncDouble &fn_fast_double_function_wo_checks, 
                                                              const bool skip_checks) {
      
      if (skip_checks == false) {
        
           fn_AVX_dbl_Eigen(x_Ref, fn_fast_AVX_function, fn_fast_double_function);
        
      } else {
        
           fn_AVX_dbl_Eigen(x_Ref, fn_fast_AVX_function_wo_checks, fn_fast_double_function_wo_checks);
        
      }
  
}

 


 

#if defined(USE_AVX_512)
//// #pragma message "About to define AVX-512 mplementation of fn_process_Ref_double_AVX"
 
template <typename T>
ALWAYS_INLINE  void       fn_process_Ref_double_AVX(       Eigen::Ref<T> x_Ref,
                                                           const std::string &fn,
                                                           const bool &skip_checks) {
  
    if        (fn == "test_simple") {    
          std::cout << "Calling test_simple function" << std::endl;
          try { 
            // fn_process_double_AVX_sub_function(x_Ref, 
            //                                     test_simple_AVX512,  test_simple_double,  
            //                                     test_simple_AVX512,  test_simple_double, skip_checks) ;
            fn_AVX_dbl_Eigen(x_Ref, test_simple_AVX512, test_simple_double);
            // fn_process_double_AVX_sub_function(x_Ref, test_simple_AVX512,  test_simple_double,   test_simple_AVX512, test_simple_double, skip_checks) ;
          } catch (const std::exception& e) { 
              std::cout << "Exception caught: " << e.what() << std::endl;
              throw;
          } catch (...) {
              std::cout << "Unknown exception caught" << std::endl;
              throw;
          }
    } else if (fn == "exp") {    
          fn_process_double_AVX_sub_function( x_Ref, 
                                              fast_exp_1_AVX512, mvp_std_exp, 
                                              fast_exp_1_wo_checks_AVX512, mvp_std_exp, skip_checks) ;
    } else if (fn == "log") {   
          fn_process_double_AVX_sub_function( x_Ref, 
                                              fast_log_1_AVX512, mvp_std_log, 
                                              fast_log_1_wo_checks_AVX512, mvp_std_log, skip_checks) ;
    } else if (fn == "log1p") {    
          fn_process_double_AVX_sub_function( x_Ref, 
                                              fast_log1p_1_AVX512, mvp_std_log1p, 
                                              fast_log1p_1_wo_checks_AVX512, mvp_std_log1p, skip_checks) ;
    } else if (fn == "log1m") {     
          fn_process_double_AVX_sub_function( x_Ref, 
                                              fast_log1m_1_AVX512, mvp_std_log1m, 
                                              fast_log1m_1_wo_checks_AVX512, mvp_std_log1m, skip_checks) ;
    } else if (fn == "logit") {        
          fn_process_double_AVX_sub_function( x_Ref, 
                                              fast_logit_AVX512, mvp_std_logit, 
                                              fast_logit_wo_checks_AVX512, mvp_std_logit, skip_checks) ;
    } else if (fn == "tanh") {  
          fn_process_double_AVX_sub_function( x_Ref, 
                                              fast_tanh_AVX512, mvp_std_tanh, 
                                              fast_tanh_wo_checks_AVX512, mvp_std_tanh, skip_checks) ;
    } else if (fn == "Phi_approx") {    
          fn_process_double_AVX_sub_function( x_Ref, 
                                              fast_Phi_approx_AVX512, mvp_std_Phi_approx, 
                                              fast_Phi_approx_wo_checks_AVX512, mvp_std_Phi_approx, skip_checks) ;
    } else if (fn == "log_Phi_approx") {      
          fn_process_double_AVX_sub_function( x_Ref, 
                                              fast_log_Phi_approx_AVX512, fast_log_Phi_approx, 
                                              fast_log_Phi_approx_wo_checks_AVX512, fast_log_Phi_approx_wo_checks, skip_checks) ;
    } else if (fn == "inv_Phi_approx") {      
          fn_process_double_AVX_sub_function( x_Ref, 
                                              fast_inv_Phi_approx_AVX512, fast_inv_Phi_approx, 
                                              fast_inv_Phi_approx_wo_checks_AVX512, fast_inv_Phi_approx_wo_checks, skip_checks) ;
    } else if (fn == "inv_Phi_approx_from_logit_prob") { 
          fn_process_double_AVX_sub_function( x_Ref, 
                                              fast_inv_Phi_approx_from_logit_prob_AVX512, fast_inv_Phi_approx_from_logit_prob, 
                                              fast_inv_Phi_approx_from_logit_prob_wo_checks_AVX512, fast_inv_Phi_approx_from_logit_prob_wo_checks, skip_checks) ;
    } else if (fn == "Phi") {             
          fn_process_double_AVX_sub_function( x_Ref, 
                                              fast_Phi_AVX512, mvp_std_Phi, 
                                              fast_Phi_wo_checks_AVX512, mvp_std_Phi, skip_checks) ;
    } else if (fn == "inv_Phi") {            
          fn_process_double_AVX_sub_function( x_Ref, 
                                              fast_inv_Phi_wo_checks_AVX512, mvp_std_inv_Phi, 
                                              fast_inv_Phi_wo_checks_AVX512, mvp_std_inv_Phi, skip_checks) ;
    } else if (fn == "inv_logit") {           
          fn_process_double_AVX_sub_function( x_Ref, 
                                              fast_inv_logit_AVX512, mvp_std_inv_logit, 
                                              fast_inv_logit_wo_checks_AVX512, mvp_std_inv_logit, skip_checks) ;
    } else if (fn == "log_inv_logit") {     
          fn_process_double_AVX_sub_function( x_Ref, 
                                              fast_log_inv_logit_AVX512, fast_log_inv_logit, 
                                              fast_log_inv_logit_wo_checks_AVX512, fast_log_inv_logit_wo_checks, skip_checks) ;
    }

}
 

#elif defined(USE_AVX2)
//// #pragma message "About to define AVX2 implementation of fn_process_Ref_double_AVX"
 
template <typename T>
ALWAYS_INLINE  void       fn_process_Ref_double_AVX(        Eigen::Ref<T> x_Ref,
                                                            const std::string &fn,
                                                            const bool &skip_checks) {
   
   if        (fn == "test_simple") {    
     std::cout << "Calling test_simple function" << std::endl;
     try { 
       // fn_process_double_AVX_sub_function(x_Ref, 
       //                                     test_simple_AVX2,  test_simple_double,  
       //                                     test_simple_AVX2,  test_simple_double, skip_checks) ;
       fn_AVX_dbl_Eigen(x_Ref, test_simple_AVX2, test_simple_double);
       // fn_process_double_AVX_sub_function(x_Ref, test_simple_AVX2,  test_simple_double,   test_simple_AVX2, test_simple_double, skip_checks) ;
     } catch (const std::exception& e) { 
       std::cout << "Exception caught: " << e.what() << std::endl;
       throw;
     } catch (...) {
       std::cout << "Unknown exception caught" << std::endl;
       throw;
     }
   } else if (fn == "exp") {    
     fn_process_double_AVX_sub_function( x_Ref, 
                                         fast_exp_1_AVX2, mvp_std_exp, 
                                         fast_exp_1_wo_checks_AVX2, mvp_std_exp, skip_checks) ;
   } else if (fn == "log") {   
     fn_process_double_AVX_sub_function( x_Ref, 
                                         fast_log_1_AVX2, mvp_std_log, 
                                         fast_log_1_wo_checks_AVX2, mvp_std_log, skip_checks) ;
   } else if (fn == "log1p") {    
     fn_process_double_AVX_sub_function( x_Ref, 
                                         fast_log1p_1_AVX2, mvp_std_log1p, 
                                         fast_log1p_1_wo_checks_AVX2, mvp_std_log1p, skip_checks) ;
   } else if (fn == "log1m") {     
     fn_process_double_AVX_sub_function( x_Ref, 
                                         fast_log1m_1_AVX2, mvp_std_log1m, 
                                         fast_log1m_1_wo_checks_AVX2, mvp_std_log1m, skip_checks) ;
   } else if (fn == "logit") {        
     fn_process_double_AVX_sub_function( x_Ref, 
                                         fast_logit_AVX2, mvp_std_logit, 
                                         fast_logit_wo_checks_AVX2, mvp_std_logit, skip_checks) ;
   } else if (fn == "tanh") {  
     fn_process_double_AVX_sub_function( x_Ref, 
                                         fast_tanh_AVX2, mvp_std_tanh, 
                                         fast_tanh_wo_checks_AVX2, mvp_std_tanh, skip_checks) ;
   } else if (fn == "Phi_approx") {    
     fn_process_double_AVX_sub_function( x_Ref, 
                                         fast_Phi_approx_AVX2, mvp_std_Phi_approx, 
                                         fast_Phi_approx_wo_checks_AVX2, mvp_std_Phi_approx, skip_checks) ;
   } else if (fn == "log_Phi_approx") {      
     fn_process_double_AVX_sub_function( x_Ref, 
                                         fast_log_Phi_approx_AVX2, fast_log_Phi_approx, 
                                         fast_log_Phi_approx_wo_checks_AVX2, fast_log_Phi_approx_wo_checks, skip_checks) ;
   } else if (fn == "inv_Phi_approx") {      
     fn_process_double_AVX_sub_function( x_Ref, 
                                         fast_inv_Phi_approx_AVX2, fast_inv_Phi_approx, 
                                         fast_inv_Phi_approx_wo_checks_AVX2, fast_inv_Phi_approx_wo_checks, skip_checks) ;
   } else if (fn == "inv_Phi_approx_from_logit_prob") { 
     fn_process_double_AVX_sub_function( x_Ref, 
                                         fast_inv_Phi_approx_from_logit_prob_AVX2, fast_inv_Phi_approx_from_logit_prob, 
                                         fast_inv_Phi_approx_from_logit_prob_wo_checks_AVX2, fast_inv_Phi_approx_from_logit_prob_wo_checks, skip_checks) ;
   } else if (fn == "Phi") {             
     fn_process_double_AVX_sub_function( x_Ref, 
                                         fast_Phi_AVX2, mvp_std_Phi, 
                                         fast_Phi_wo_checks_AVX2, mvp_std_Phi, skip_checks) ;
   } else if (fn == "inv_Phi") {            
     fn_process_double_AVX_sub_function( x_Ref, 
                                         fast_inv_Phi_wo_checks_AVX2, mvp_std_inv_Phi, 
                                         fast_inv_Phi_wo_checks_AVX2, mvp_std_inv_Phi, skip_checks) ;
   } else if (fn == "inv_logit") {           
     fn_process_double_AVX_sub_function( x_Ref, 
                                         fast_inv_logit_AVX2, mvp_std_inv_logit, 
                                         fast_inv_logit_wo_checks_AVX2, mvp_std_inv_logit, skip_checks) ;
   } else if (fn == "log_inv_logit") {     
     fn_process_double_AVX_sub_function( x_Ref, 
                                         fast_log_inv_logit_AVX2, fast_log_inv_logit, 
                                         fast_log_inv_logit_wo_checks_AVX2, fast_log_inv_logit_wo_checks, skip_checks) ;
   }
   
 }
 
#else 
//// #pragma message "Defining dummy fn_process_Ref_double_AVX - since neither AVX2 nor AVX-512 are available"
 
template <typename T>
ALWAYS_INLINE  void       fn_process_Ref_double_AVX(         Eigen::Ref<T> x_Ref,
                                                             const std::string &fn,
                                                             const bool &skip_checks) {
   

}
 
 
 
#endif
 
 

 
  
  


#endif



#endif

  
  