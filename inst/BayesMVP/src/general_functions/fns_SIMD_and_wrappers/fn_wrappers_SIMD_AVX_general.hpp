
#pragma once 
 
#ifndef FN_WRAPPERS_SIMD_AVX_GENERAL_HPP
#define FN_WRAPPERS_SIMD_AVX_GENERAL_HPP

 
 
#include <Eigen/Dense>
#include <Eigen/Core>
 
#include <string>
 
#include "fn_SIMD_level_resolver.hpp"
 
 
//// -------------------------------------------------------------------------------------------------------------------------------------------------------------
//// ---- 2026-09-22: BOTH SIMD levels are compiled on an AVX-512 build.
////      Until 2026-09-22 this file compiled ONE level (#if USE_AVX_512 ... #elif USE_AVX2, macros from NicoStan's SIMD_config.hpp):
////      fn_process_Ref_double_AVX ran the AVX-512 kernels on an AVX-512 build and the AVX2 kernels on an AVX2 build, and the helpers
////      below were typed on the single FuncAVX of SIMD_config.hpp. So vect_type = "AVX2" on an AVX-512 build could only ever run the
////      8-lane kernels. Now:
////        - fn_process_Ref_double_AVX512  (8 lanes, fast_*_AVX512 kernels) is compiled when BAYESMVP_COMPILED_AVX512_KERNELS,
////        - fn_process_Ref_double_AVX2    (4 lanes, fast_*_AVX2 kernels)   is compiled when BAYESMVP_COMPILED_AVX2_KERNELS
////          (i.e. on AVX2-only AND on AVX-512 builds),
////        - the Eigen helpers take the lane width (8 or 4) as a template argument, so the 8-lane instantiation is the same code as the
////          old AVX-512 one (lane width a compile-time constant, no per-element branching).
////      fn_EIGEN_Ref_double (fn_wrappers_overall.hpp) picks one via fn_BayesMVP_SIMD_lane_width_for_vect_type() (fn_SIMD_level_resolver.hpp).
////      fn_process_Ref_double_AVX (the name NicoStan's SIMD_config.hpp forward-declares) is kept and runs the highest compiled level, as before.
////
#if BAYESMVP_COMPILED_AVX512_KERNELS || BAYESMVP_COMPILED_AVX2_KERNELS
 
 
#include <immintrin.h>
 
 
//// Kernel function-pointer type for each lane width. The old helpers took "const FuncAVX &" (FuncAVX = the ONE level's pointer type
//// in SIMD_config.hpp); BayesMVP_SIMD_kernel_pointer<8 or 4>::type is exactly that type for each level, so nothing else changes.
template <int SIMD_lane_width> struct BayesMVP_SIMD_kernel_pointer;
#if BAYESMVP_COMPILED_AVX512_KERNELS
template <> struct BayesMVP_SIMD_kernel_pointer<8> { typedef __m512d (*type)(const __m512d); };   //// e.g. fast_exp_1_AVX512
#endif
#if BAYESMVP_COMPILED_AVX2_KERNELS
template <> struct BayesMVP_SIMD_kernel_pointer<4> { typedef __m256d (*type)(const __m256d); };   //// e.g. fast_exp_1_AVX2
#endif


//// SIMD_lane_width = 8 (AVX-512 kernels) or 4 (AVX2 kernels); a compile-time constant (was: #if USE_AVX_512 / #elif USE_AVX2).
template <int SIMD_lane_width, typename T>
ALWAYS_INLINE  void fn_AVX_row_or_col_vector(    Eigen::Ref<T>  x_Ref,
                                                 const typename BayesMVP_SIMD_kernel_pointer<SIMD_lane_width>::type &fn_AVX,
                                                 const FuncDouble &fn_double) {
  
        static_assert((SIMD_lane_width == 8) || (SIMD_lane_width == 4), "SIMD_lane_width must be 8 (AVX-512) or 4 (AVX2)");
  
        const int N = x_Ref.size();
  
        const int vect_size = SIMD_lane_width;
          
        const double vect_siz_dbl = static_cast<double>(vect_size);
        const double N_dbl = static_cast<double>(N);
        const int N_divisible_by_vect_size = std::floor(N_dbl / vect_siz_dbl) * vect_size;
        
       if (N >= vect_size) {
             
                     //// last vect_size elements, copied BEFORE the SIMD pass overwrites them (the scalar remainder pass below recomputes
                     //// them from their original values). 2026-09-22: moved inside "if (N >= vect_size)"; it used to run for every N and so
                     //// read x_Ref(N - vect_size), i.e. before the start of the vector, whenever N < vect_size (the copy was unused then).
                     Eigen::Matrix<double, -1, 1> x_tail = Eigen::Matrix<double, -1, 1>::Zero(vect_size); // last vect_size elements
                     {
                         int counter = 0;
                         for (int i = N - vect_size; i < N; ++i) {
                           x_tail(counter) = x_Ref(i);
                           counter += 1;
                         } 
                     }
                     
                     for (int i = 0; i + vect_size <= N_divisible_by_vect_size; i += vect_size) {
                       
                         if constexpr (SIMD_lane_width == 8) {
                           #if BAYESMVP_COMPILED_AVX512_KERNELS
                             __m512d const AVX_array = _mm512_loadu_pd(&x_Ref(i));
                             __m512d const AVX_array_out = fn_AVX(AVX_array);
                             _mm512_storeu_pd(&x_Ref(i), AVX_array_out);
                           #endif
                         } else {
                           #if BAYESMVP_COMPILED_AVX2_KERNELS
                             __m256d const AVX_array = _mm256_loadu_pd(&x_Ref(i));
                             __m256d const AVX_array_out = fn_AVX(AVX_array);
                             _mm256_storeu_pd(&x_Ref(i), AVX_array_out);
                           #endif
                         }
                           
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
 
 
 
 
 
 
 

template <int SIMD_lane_width, typename T>
ALWAYS_INLINE  void fn_AVX_matrix(    Eigen::Ref<T> x_Ref,
                                      const typename BayesMVP_SIMD_kernel_pointer<SIMD_lane_width>::type &fn_AVX, 
                                      const FuncDouble &fn_double) {
     
     const int n_rows = x_Ref.rows();
     const int n_cols = x_Ref.cols();
     
     if (n_rows > n_cols) { // if data in "long" format
       for (int j = 0; j < n_cols; ++j) {   
         Eigen::Matrix<double, -1, 1> x_col = x_Ref.col(j);
         using ColType = decltype(x_col);
         Eigen::Ref<Eigen::Matrix<double, -1, 1>> x_col_Ref(x_col); 
         fn_AVX_row_or_col_vector<SIMD_lane_width, ColType>(x_col_Ref, fn_AVX, fn_double);
         x_Ref.col(j) = x_col_Ref;
       }
     } else { 
       for (int j = 0; j < n_rows; ++j) {
         Eigen::Matrix<double, 1, -1> x_row = x_Ref.row(j);
         using RowType = decltype(x_row);
         Eigen::Ref<Eigen::Matrix<double, 1, -1>> x_row_Ref(x_row); 
         fn_AVX_row_or_col_vector<SIMD_lane_width, RowType>(x_row_Ref, fn_AVX, fn_double);
         x_Ref.row(j) = x_row_Ref;
       }
     }
   
} 
 
 
 
 
 
 
 

template <int SIMD_lane_width, typename T>
ALWAYS_INLINE  void fn_AVX_dbl_Eigen(     Eigen::Ref<T> x_Ref, 
                                          const typename BayesMVP_SIMD_kernel_pointer<SIMD_lane_width>::type &fn_AVX, 
                                          const FuncDouble &fn_double) {
     
     constexpr int n_rows = T::RowsAtCompileTime;
     constexpr int n_cols = T::ColsAtCompileTime;
     
     if constexpr (n_rows == 1 && n_cols == -1) {   // Row vector case
        
           fn_AVX_row_or_col_vector<SIMD_lane_width>(x_Ref, fn_AVX, fn_double);
       
     } else if constexpr (n_rows == -1 && n_cols == 1) {  // Column vector case
       
           fn_AVX_row_or_col_vector<SIMD_lane_width>(x_Ref, fn_AVX, fn_double);
       
     } else {   // General matrix case
       
           fn_AVX_matrix<SIMD_lane_width>(x_Ref, fn_AVX, fn_double);
       
     }
   
}
  
 
 

 
 
 
  

template <int SIMD_lane_width, typename T>
ALWAYS_INLINE  void    fn_process_double_AVX_sub_function(    Eigen::Ref<T> x_Ref,  
                                                              const typename BayesMVP_SIMD_kernel_pointer<SIMD_lane_width>::type &fn_fast_AVX_function,
                                                              const FuncDouble &fn_fast_double_function,
                                                              const typename BayesMVP_SIMD_kernel_pointer<SIMD_lane_width>::type &fn_fast_AVX_function_wo_checks,
                                                              const FuncDouble &fn_fast_double_function_wo_checks, 
                                                              const bool skip_checks) {
      
      if (skip_checks == false) {
        
           fn_AVX_dbl_Eigen<SIMD_lane_width>(x_Ref, fn_fast_AVX_function, fn_fast_double_function);
        
      } else {
        
           fn_AVX_dbl_Eigen<SIMD_lane_width>(x_Ref, fn_fast_AVX_function_wo_checks, fn_fast_double_function_wo_checks);
        
      }
  
}

 


 

#if BAYESMVP_COMPILED_AVX512_KERNELS
//// ---- 8-lane (AVX-512) processor: the body of the old AVX-512 fn_process_Ref_double_AVX, renamed (2026-09-22) so that it can
////      coexist with the 4-lane one below; the only other change is the <8> lane-width argument of the helpers.
 
template <typename T>
ALWAYS_INLINE  void       fn_process_Ref_double_AVX512(    Eigen::Ref<T> x_Ref,
                                                           const std::string &fn,
                                                           const bool &skip_checks) {
  
    if        (fn == "test_simple") {    
          std::cout << "Calling test_simple function" << std::endl;
          try { 
            // fn_process_double_AVX_sub_function(x_Ref, 
            //                                     test_simple_AVX512,  test_simple_double,  
            //                                     test_simple_AVX512,  test_simple_double, skip_checks) ;
            fn_AVX_dbl_Eigen<8>(x_Ref, test_simple_AVX512, test_simple_double);
            // fn_process_double_AVX_sub_function(x_Ref, test_simple_AVX512,  test_simple_double,   test_simple_AVX512, test_simple_double, skip_checks) ;
          } catch (const std::exception& e) { 
              std::cout << "Exception caught: " << e.what() << std::endl;
              throw;
          } catch (...) {
              std::cout << "Unknown exception caught" << std::endl;
              throw;
          }
    } else if (fn == "exp") {    
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              fast_exp_1_AVX512, mvp_std_exp, 
                                              fast_exp_1_wo_checks_AVX512, mvp_std_exp, skip_checks) ;
    } else if (fn == "log") {   
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              fast_log_1_AVX512, mvp_std_log, 
                                              fast_log_1_wo_checks_AVX512, mvp_std_log, skip_checks) ;
    } else if (fn == "log1p") {    
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              fast_log1p_1_AVX512, mvp_std_log1p, 
                                              fast_log1p_1_wo_checks_AVX512, mvp_std_log1p, skip_checks) ;
    } else if (fn == "log1m") {     
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              fast_log1m_1_AVX512, mvp_std_log1m, 
                                              fast_log1m_1_wo_checks_AVX512, mvp_std_log1m, skip_checks) ;
    } else if (fn == "logit") {        
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              fast_logit_AVX512, mvp_std_logit, 
                                              fast_logit_wo_checks_AVX512, mvp_std_logit, skip_checks) ;
    } else if (fn == "tanh") {  
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              fast_tanh_AVX512, mvp_std_tanh, 
                                              fast_tanh_wo_checks_AVX512, mvp_std_tanh, skip_checks) ;
    } else if (fn == "Phi_approx") {    
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              fast_Phi_approx_AVX512, mvp_std_Phi_approx, 
                                              fast_Phi_approx_wo_checks_AVX512, mvp_std_Phi_approx, skip_checks) ;
    } else if (fn == "log_Phi_approx") {      
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              fast_log_Phi_approx_AVX512, fast_log_Phi_approx, 
                                              fast_log_Phi_approx_wo_checks_AVX512, fast_log_Phi_approx_wo_checks, skip_checks) ;
    } else if (fn == "inv_Phi_approx") {      
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              fast_inv_Phi_approx_AVX512, fast_inv_Phi_approx, 
                                              fast_inv_Phi_approx_wo_checks_AVX512, fast_inv_Phi_approx_wo_checks, skip_checks) ;
    } else if (fn == "inv_Phi_approx_from_logit_prob") { 
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              fast_inv_Phi_approx_from_logit_prob_AVX512, fast_inv_Phi_approx_from_logit_prob, 
                                              fast_inv_Phi_approx_from_logit_prob_wo_checks_AVX512, fast_inv_Phi_approx_from_logit_prob_wo_checks, skip_checks) ;
    } else if (fn == "Phi") {             
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              fast_Phi_AVX512, mvp_std_Phi, 
                                              fast_Phi_wo_checks_AVX512, mvp_std_Phi, skip_checks) ;
    } else if (fn == "inv_Phi") {            
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              fast_inv_Phi_wo_checks_AVX512, mvp_std_inv_Phi, 
                                              fast_inv_Phi_wo_checks_AVX512, mvp_std_inv_Phi, skip_checks) ;
    } else if (fn == "inv_logit") {           
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              fast_inv_logit_AVX512, mvp_std_inv_logit, 
                                              fast_inv_logit_wo_checks_AVX512, mvp_std_inv_logit, skip_checks) ;
    } else if (fn == "log_inv_logit") {     
          fn_process_double_AVX_sub_function<8>( x_Ref, 
                                              fast_log_inv_logit_AVX512, fast_log_inv_logit, 
                                              fast_log_inv_logit_wo_checks_AVX512, fast_log_inv_logit_wo_checks, skip_checks) ;
    }

}
 

#endif
 
 
#if BAYESMVP_COMPILED_AVX2_KERNELS
//// ---- 4-lane (AVX2) processor: the body of the old AVX2 fn_process_Ref_double_AVX, renamed (2026-09-22). It used to be compiled
////      only when AVX-512 was NOT available (#elif USE_AVX2); it is now compiled on AVX-512 builds too, so vect_type = "AVX2" runs
////      the genuine 256-bit fast_*_AVX2 kernels there. The only other change is the <4> lane-width argument of the helpers.
 
template <typename T>
ALWAYS_INLINE  void       fn_process_Ref_double_AVX2(       Eigen::Ref<T> x_Ref,
                                                            const std::string &fn,
                                                            const bool &skip_checks) {
   
   if        (fn == "test_simple") {    
     std::cout << "Calling test_simple function" << std::endl;
     try { 
       // fn_process_double_AVX_sub_function(x_Ref, 
       //                                     test_simple_AVX2,  test_simple_double,  
       //                                     test_simple_AVX2,  test_simple_double, skip_checks) ;
       fn_AVX_dbl_Eigen<4>(x_Ref, test_simple_AVX2, test_simple_double);
       // fn_process_double_AVX_sub_function(x_Ref, test_simple_AVX2,  test_simple_double,   test_simple_AVX2, test_simple_double, skip_checks) ;
     } catch (const std::exception& e) { 
       std::cout << "Exception caught: " << e.what() << std::endl;
       throw;
     } catch (...) {
       std::cout << "Unknown exception caught" << std::endl;
       throw;
     }
   } else if (fn == "exp") {    
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         fast_exp_1_AVX2, mvp_std_exp, 
                                         fast_exp_1_wo_checks_AVX2, mvp_std_exp, skip_checks) ;
   } else if (fn == "log") {   
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         fast_log_1_AVX2, mvp_std_log, 
                                         fast_log_1_wo_checks_AVX2, mvp_std_log, skip_checks) ;
   } else if (fn == "log1p") {    
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         fast_log1p_1_AVX2, mvp_std_log1p, 
                                         fast_log1p_1_wo_checks_AVX2, mvp_std_log1p, skip_checks) ;
   } else if (fn == "log1m") {     
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         fast_log1m_1_AVX2, mvp_std_log1m, 
                                         fast_log1m_1_wo_checks_AVX2, mvp_std_log1m, skip_checks) ;
   } else if (fn == "logit") {        
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         fast_logit_AVX2, mvp_std_logit, 
                                         fast_logit_wo_checks_AVX2, mvp_std_logit, skip_checks) ;
   } else if (fn == "tanh") {  
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         fast_tanh_AVX2, mvp_std_tanh, 
                                         fast_tanh_wo_checks_AVX2, mvp_std_tanh, skip_checks) ;
   } else if (fn == "Phi_approx") {    
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         fast_Phi_approx_AVX2, mvp_std_Phi_approx, 
                                         fast_Phi_approx_wo_checks_AVX2, mvp_std_Phi_approx, skip_checks) ;
   } else if (fn == "log_Phi_approx") {      
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         fast_log_Phi_approx_AVX2, fast_log_Phi_approx, 
                                         fast_log_Phi_approx_wo_checks_AVX2, fast_log_Phi_approx_wo_checks, skip_checks) ;
   } else if (fn == "inv_Phi_approx") {      
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         fast_inv_Phi_approx_AVX2, fast_inv_Phi_approx, 
                                         fast_inv_Phi_approx_wo_checks_AVX2, fast_inv_Phi_approx_wo_checks, skip_checks) ;
   } else if (fn == "inv_Phi_approx_from_logit_prob") { 
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         fast_inv_Phi_approx_from_logit_prob_AVX2, fast_inv_Phi_approx_from_logit_prob, 
                                         fast_inv_Phi_approx_from_logit_prob_wo_checks_AVX2, fast_inv_Phi_approx_from_logit_prob_wo_checks, skip_checks) ;
   } else if (fn == "Phi") {             
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         fast_Phi_AVX2, mvp_std_Phi, 
                                         fast_Phi_wo_checks_AVX2, mvp_std_Phi, skip_checks) ;
   } else if (fn == "inv_Phi") {            
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         fast_inv_Phi_wo_checks_AVX2, mvp_std_inv_Phi, 
                                         fast_inv_Phi_wo_checks_AVX2, mvp_std_inv_Phi, skip_checks) ;
   } else if (fn == "inv_logit") {           
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         fast_inv_logit_AVX2, mvp_std_inv_logit, 
                                         fast_inv_logit_wo_checks_AVX2, mvp_std_inv_logit, skip_checks) ;
   } else if (fn == "log_inv_logit") {     
     fn_process_double_AVX_sub_function<4>( x_Ref, 
                                         fast_log_inv_logit_AVX2, fast_log_inv_logit, 
                                         fast_log_inv_logit_wo_checks_AVX2, fast_log_inv_logit_wo_checks, skip_checks) ;
   }
   
 }
 
#endif
 
 
//// ---- fn_process_Ref_double_AVX: the name NicoStan's SIMD_config.hpp forward-declares (whenever USE_AVX2 or USE_AVX_512 is defined).
////      Kept for compatibility (2026-09-22): it runs the HIGHEST compiled level, exactly what it did before. BayesMVP's own dispatch no
////      longer calls it; fn_EIGEN_Ref_double calls fn_process_Ref_double_AVX512 / fn_process_Ref_double_AVX2 directly.
template <typename T>
ALWAYS_INLINE  void       fn_process_Ref_double_AVX(         Eigen::Ref<T> x_Ref,
                                                             const std::string &fn,
                                                             const bool &skip_checks) {
   
    #if BAYESMVP_COMPILED_AVX512_KERNELS
          fn_process_Ref_double_AVX512(x_Ref, fn, skip_checks);
    #else
          fn_process_Ref_double_AVX2(x_Ref, fn, skip_checks);
    #endif

}
 
 

 
  
  


#endif



#endif

  
  