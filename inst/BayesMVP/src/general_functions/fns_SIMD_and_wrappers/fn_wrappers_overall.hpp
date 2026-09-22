
#pragma once 


#ifndef FN_WRAPPERS_OVERALL_HPP
#define FN_WRAPPERS_OVERALL_HPP

 
#include <stan/math/prim/prob/std_normal_log_qf.hpp>
  
#include <Eigen/Dense>
#include <Eigen/Core>
 
#include <immintrin.h>

 

 
 
///// ----------------------------------------------------------------------------- Colvec function callers / wrappers


template <typename T>
ALWAYS_INLINE  void               fn_EIGEN_Ref_double(      Eigen::Ref<T> x_Ref,
                                                            const std::string &fn,
                                                            const std::string &vect_type,
                                                            const bool &skip_checks
) {
        
       /////  stan::math::check_finite(fn.c_str(), "x", x);  // using c_str() to convert std::string to const char*
      
        if (fn == "inv_Phi_from_log_prob") {
      
               x_Ref = stan::math::std_normal_log_qf(x_Ref); 
      
        } else {
      
          if (vect_type == "Stan") {
      
                fn_void_Ref_double_Stan(x_Ref, fn, skip_checks);
       
          } else if ( (vect_type == "AVX2") || (vect_type == "AVX512") ) { 
      
              //// ---- SIMD request (2026-09-22 fix for audit item D3):
              ////      This build compiles exactly ONE SIMD level of fn_process_Ref_double_AVX (AVX-512 if the compiler
              ////      flags provide it, otherwise AVX2 - see SIMD_config.hpp / fn_wrappers_SIMD_AVX_general.hpp).
              ////      The R front end (fn_check_native_model_vect_types_and_Phi_types in R_fns_init_hard_coded_models.R)
              ////      stops before sampling if the requested level is not the compiled one. As a second line of defence
              ////      this branch NEVER leaves x unchanged: previously a request for the level that was not compiled
              ////      printed "Error: AVX2 is not available" (or "AVX-512") on every call and returned x UNCHANGED, so
              ////      every result computed that way was silently wrong. Now:
              ////        - requested level is compiled                -> use it;
              ////        - AVX2 requested on an AVX-512 build          -> use the compiled AVX-512 kernels (AVX-512 is a superset);
              ////        - AVX512 requested on an AVX2 build           -> use the compiled AVX2 kernels (same functions, narrower);
              ////        - neither AVX2 nor AVX-512 compiled           -> use the exact Stan-math path.
              ////      No per-call printing (this is called inside the likelihood hot loop).
              ////
              #if (defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__)) || defined(__AVX2__)
                         fn_process_Ref_double_AVX(x_Ref, fn, skip_checks); // the one compiled SIMD level (AVX-512 or AVX2)
              #else
                         fn_void_Ref_double_Stan(x_Ref, fn, skip_checks);
              #endif
      
          } else if (vect_type == "Loop") {
            
                fn_return_Loop(x_Ref, fn, skip_checks);
      
          } else { 
            
                //// ---- Unrecognised vect_type string (2026-09-22): previously fell through and left x UNCHANGED.
                ////      The R front end rejects unknown strings before sampling; here we compute the exact
                ////      Stan-math value rather than silently returning the input.
                fn_void_Ref_double_Stan(x_Ref, fn, skip_checks);
      
          }
      
        }

}
 
////// ------- "master" function  w/ return  -------------------------------------------------------------------------------------------------------
 
//// R-value
template <typename T>
inline  auto          fn_EIGEN_double(        T  &&x_R_val,
                                                     const std::string &fn,
                                                     const std::string &vect_type = "Stan",
                                                     const bool &skip_checks = false
) {
       
       using T_matrix_type = Eigen::Matrix<double, -1, -1>;
       T_matrix_type x_matrix = x_R_val;   
       fn_EIGEN_Ref_double(Eigen::Ref<T_matrix_type>(x_matrix), fn, vect_type, skip_checks);
       return x_matrix;
   
}

//// Eigen Ref (this will also accept L_value [&T] as well as other types)
template <typename T>
inline  auto          fn_EIGEN_double(   Eigen::Ref<T> x_L_val,
                                                const std::string &fn,
                                                const std::string &vect_type = "Stan",
                                                const bool &skip_checks = false
) {
  
       fn_EIGEN_Ref_double(x_L_val, fn, vect_type, skip_checks);
       return x_L_val;
  
}


//// const Eigen Ref (this will also accept L_value [&T] as well as other types)
template <typename T>
inline  auto          fn_EIGEN_double(   const Eigen::Ref<const T> x_L_val,
                                                const std::string &fn,
                                                const std::string &vect_type = "Stan",
                                                const bool &skip_checks = false
) {
  
       T x_copy = x_L_val;
       fn_EIGEN_Ref_double(Eigen::Ref<T>(x_copy), fn, vect_type, skip_checks);
       return x_copy;
  
}

 
//// blocks
template <typename T, int n_rows = Eigen::Dynamic, int n_cols = Eigen::Dynamic>
inline auto  fn_EIGEN_double(                 Eigen::Ref<Eigen::Block<T, n_rows, n_cols>> x_Ref,
                                                     const std::string &fn,
                                                     const std::string &vect_type = "Stan",
                                                     const bool &skip_checks = false
) {
       
       T x_matrix = x_Ref;
       fn_EIGEN_Ref_double(Eigen::Ref<T>(x_matrix), fn, vect_type, skip_checks);
       return x_matrix;
   
}
 
 
//// arrays
template <int n_rows = Eigen::Dynamic, int n_cols = Eigen::Dynamic>
inline auto  fn_EIGEN_double(         const Eigen::Array<double, n_rows, n_cols> &x,
                                             const std::string &fn, 
                                             const std::string &vect_type = "Stan",
                                             const bool &skip_checks = false
) {
   
       using T = Eigen::Matrix<double, n_rows, n_cols>;
       T x_matrix = x.matrix();
       fn_EIGEN_Ref_double(Eigen::Ref<T>(x_matrix), fn, vect_type, skip_checks);
       return x_matrix;
   
}
 
 
// New overload for general expressions
template <typename Derived>
inline auto fn_EIGEN_double(  const Eigen::EigenBase<Derived> &x,
                                     const std::string &fn,
                                     const std::string &vect_type = "Stan",
                                     const bool &skip_checks = false
) {
   
       using T = Eigen::Matrix<typename Derived::Scalar, 
                               Derived::RowsAtCompileTime, 
                               Derived::ColsAtCompileTime>;
       T x_copy = x;
       fn_EIGEN_Ref_double(Eigen::Ref<T>(x_copy), fn, vect_type, skip_checks);
       return x_copy;
   
}
 
 
 
// Additional Matrix expression overload
template <typename Derived>
inline auto fn_EIGEN_double(  const Eigen::MatrixBase<Derived> &x,
                                     const std::string &fn,
                                     const std::string &vect_type = "Stan",
                                     const bool &skip_checks = false
) {
   
       using T = Eigen::Matrix<typename Derived::Scalar,
                               Derived::RowsAtCompileTime,
                               Derived::ColsAtCompileTime>;
       T x_copy = x;
       fn_EIGEN_Ref_double(Eigen::Ref<T>(x_copy), fn, vect_type, skip_checks);
       return x_copy;
       
}
 
 
 
// Additional overload for array expressions
template <typename Derived>
inline auto fn_EIGEN_double(  const Eigen::ArrayBase<Derived> &x,
                                     const std::string &fn,
                                     const std::string &vect_type = "Stan", 
                                     const bool &skip_checks = false
) {
   
       using T = Eigen::Matrix<typename Derived::Scalar,
                               Derived::RowsAtCompileTime,
                               Derived::ColsAtCompileTime>;
       T x_copy = x.matrix();
       fn_EIGEN_Ref_double(Eigen::Ref<T>(x_copy), fn, vect_type, skip_checks);
       return x_copy;
   
}

 





#endif

  
  
  
  
  