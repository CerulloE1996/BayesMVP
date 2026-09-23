
#pragma once 


#ifndef FN_WRAPPERS_OVERALL_HPP
#define FN_WRAPPERS_OVERALL_HPP

 
#include <stan/math/prim/prob/std_normal_log_qf.hpp>
  
#include <Eigen/Dense>
#include <Eigen/Core>
 
#include <immintrin.h>

#include <string>
#include <stdexcept>

#include "fn_SIMD_level_resolver.hpp"
 

 
 
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
       
          } else if (vect_type == "AVX512") { 
      
              //// ---- SIMD requests (2026-09-22: BOTH levels are compiled on an AVX-512 build; replaces the D3 fallback, see History):
              ////        "AVX512" -> fn_process_Ref_double_AVX512 (8-lane fast_*_AVX512 kernels),
              ////        "AVX2"   -> fn_process_Ref_double_AVX2   (genuine 256-bit fast_*_AVX2 kernels; compiled on AVX2-only AND on
              ////                                                  AVX-512 builds).
              ////      A level this build did not compile (e.g. "AVX512" on an AVX2-only laptop build) THROWS std::invalid_argument. The
              ////      R front end (fn_check_native_model_vect_types_and_Phi_types in R_fns_init_hard_coded_models.R) stops before
              ////      sampling in that case, so the throw is a second line of defence. fn_BayesMVP_SIMD_lane_width_for_vect_type()
              ////      (fn_SIMD_level_resolver.hpp) encodes exactly these branches (same strings, same BAYESMVP_COMPILED_* macros); R's
              ////      R_fn_BayesMVP_SIMD_lane_width_for_vect_type() reports it AND probes this function through
              ////      Rcpp_wrapper_EIGEN_double_colvec to confirm the kernel width that really runs.
              ////      Plain string branches, AVX512 first: string comparisons cost several ns each and this runs once per vector
              ////      operation in the likelihood (a resolver call here added ~10 ns per call, 8-20% on 8-48-element vectors).
              ////      The AVX512 request now needs 2 comparisons (3 before). No per-element branching, no per-call printing.
              ////      History: until 2026-09-22 (audit item D3) a request for the level that was not compiled printed "Error: AVX2 is not
              ////      available" on every call and returned x UNCHANGED; the D3 fix then ran the ONE compiled level instead, so "AVX2" on
              ////      an AVX-512 build silently meant the 8-lane AVX-512 kernels (and "AVX512" on an AVX2 build the 4-lane ones).
              ////
              #if BAYESMVP_COMPILED_AVX512_KERNELS
                         fn_process_Ref_double_AVX512(x_Ref, fn, skip_checks);
              #else
                         throw std::invalid_argument(fn_BayesMVP_SIMD_level_not_compiled_message(vect_type));
              #endif
      
          } else if (vect_type == "AVX2") { 
      
              #if BAYESMVP_COMPILED_AVX2_KERNELS
                         fn_process_Ref_double_AVX2(x_Ref, fn, skip_checks);   //// 4-lane kernels, also on AVX-512 builds (see above)
              #else
                         throw std::invalid_argument(fn_BayesMVP_SIMD_level_not_compiled_message(vect_type));
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

  
  
  
  
  