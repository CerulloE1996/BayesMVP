
#pragma once 


#ifndef FN_WRAPPERS_OVERALL_HPP
#define FN_WRAPPERS_OVERALL_HPP

 
#include <stan/math/prim/prob/std_normal_log_qf.hpp>
  
#include <Eigen/Dense>
#include <Eigen/Core>
 
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)   //// (x86-64 only: ARM CPUs have no AVX)
#include <NicoSIMD/nico_simd_intrinsics.hpp>
#endif

#include <string>
#include <stdexcept>
#include <vector>

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
      
                fn_void_Ref_double_Stan( x_Ref,
                                         fn,
                                         skip_checks);
       
          } else if (vect_type == "AVX512") { 
      
              //// ---- SIMD requests (BOTH levels are compiled on an AVX-512 build):
              ////        "AVX512" -> fn_process_Ref_double_AVX512 (8-lane fast_*_AVX512 kernels),
              ////        "AVX2"   -> fn_process_Ref_double_AVX2   (genuine 256-bit fast_*_AVX2 kernels; compiled
              ////                                                  on AVX2-only AND on AVX-512 builds).
              ////      A level this build did not compile (e.g. "AVX512" on an AVX2-only laptop build) THROWS
              ////      std::invalid_argument. The R front end (fn_check_native_model_vect_types_and_Phi_types in
              ////      R_fns_init_hard_coded_models.R) stops before sampling in that case, so the throw is a
              ////      second line of defence. fn_BayesMVP_SIMD_lane_width_for_vect_type()
              ////      (fn_SIMD_level_resolver.hpp) encodes exactly these branches (same strings, same
              ////      BAYESMVP_COMPILED_* macros); R's R_fn_BayesMVP_SIMD_lane_width_for_vect_type() reports it
              ////      AND probes this function through Rcpp_wrapper_EIGEN_double_colvec to confirm the kernel
              ////      width that really runs.
              ////      Plain string branches, AVX512 first: string comparisons cost several ns each and this runs
              ////      once per vector operation in the likelihood (a resolver call here would add ~10 ns per
              ////      call, 8-20% on 8-48-element vectors), so the AVX512 request needs 2 comparisons. No
              ////      per-element branching, no per-call printing.
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
            
                fn_return_Loop( x_Ref,
                                fn,
                                skip_checks);
      
          } else { 
            
                //// ---- Unrecognised vect_type string: the exact Stan-math value (not x UNCHANGED).
                ////      The R front end rejects unknown strings before sampling; here the exact Stan-math
                ////      value is computed rather than silently returning the input.
                fn_void_Ref_double_Stan( x_Ref,
                                         fn,
                                         skip_checks);
      
          }
      
        }

}
 
////// ------- "master" function  w/ return  -------------------------------------------------------------------------------------------------------
 
//// R-value
template <typename T>
inline  auto          fn_EIGEN_double(        T  &&x_R_val,
                                                     const std::string &fn,
                                                     const std::string &vect_type,
                                                     const bool &skip_checks
) {
       
       Eigen::Matrix<double, -1, -1> x_matrix = x_R_val;   
       fn_EIGEN_Ref_double( Eigen::Ref<Eigen::Matrix<double,
                            -1,
                            -1>>(x_matrix),
                            fn,
                            vect_type,
                            skip_checks);
       return x_matrix;
   
}

//// Eigen Ref (this will also accept L_value [&T] as well as other types)
template <typename T>
inline  auto          fn_EIGEN_double(   Eigen::Ref<T> x_L_val,
                                                const std::string &fn,
                                                const std::string &vect_type,
                                                const bool &skip_checks
) {
  
       fn_EIGEN_Ref_double( x_L_val,
                            fn,
                            vect_type,
                            skip_checks);
       return x_L_val;
  
}


//// const Eigen Ref (this will also accept L_value [&T] as well as other types)
template <typename T>
inline  auto          fn_EIGEN_double(   const Eigen::Ref<const T> x_L_val,
                                                const std::string &fn,
                                                const std::string &vect_type,
                                                const bool &skip_checks
) {
  
       T x_copy = x_L_val;
       fn_EIGEN_Ref_double( Eigen::Ref<T>(x_copy),
                            fn,
                            vect_type,
                            skip_checks);
       return x_copy;
  
}

 
//// blocks
template <typename T, int n_rows = -1, int n_cols = -1>
inline auto  fn_EIGEN_double(                 Eigen::Ref<Eigen::Block<T, n_rows, n_cols>> x_Ref,
                                                     const std::string &fn,
                                                     const std::string &vect_type,
                                                     const bool &skip_checks
) {
       
       T x_matrix = x_Ref;
       fn_EIGEN_Ref_double( Eigen::Ref<T>(x_matrix),
                            fn,
                            vect_type,
                            skip_checks);
       return x_matrix;
   
}
 
 
//// arrays
template <int n_rows = -1, int n_cols = -1>
inline auto  fn_EIGEN_double(         const Eigen::Array<double, n_rows, n_cols> &x,
                                             const std::string &fn, 
                                             const std::string &vect_type,
                                             const bool &skip_checks
) {
   
       Eigen::Matrix<double, n_rows, n_cols> x_matrix = x.matrix();
       fn_EIGEN_Ref_double( Eigen::Ref<Eigen::Matrix<double,
                            n_rows,
                            n_cols>>(x_matrix),
                            fn,
                            vect_type,
                            skip_checks);
       return x_matrix;
   
}
 
 
//// New overload for general expressions
template <typename Derived>
inline auto fn_EIGEN_double(  const Eigen::EigenBase<Derived> &x,
                                     const std::string &fn,
                                     const std::string &vect_type,
                                     const bool &skip_checks
) {
   
       Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> x_copy = x;
       fn_EIGEN_Ref_double( Eigen::Ref<Eigen::Matrix<typename Derived::Scalar,
                            Derived::RowsAtCompileTime,
                            Derived::ColsAtCompileTime>>(x_copy),
                            fn,
                            vect_type,
                            skip_checks);
       return x_copy;
   
}
 
 
 
//// Additional Matrix expression overload
template <typename Derived>
inline auto fn_EIGEN_double(  const Eigen::MatrixBase<Derived> &x,
                                     const std::string &fn,
                                     const std::string &vect_type,
                                     const bool &skip_checks
) {
   
       Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> x_copy = x;
       fn_EIGEN_Ref_double( Eigen::Ref<Eigen::Matrix<typename Derived::Scalar,
                            Derived::RowsAtCompileTime,
                            Derived::ColsAtCompileTime>>(x_copy),
                            fn,
                            vect_type,
                            skip_checks);
       return x_copy;
       
}
 
 
 
//// Additional overload for array expressions
template <typename Derived>
inline auto fn_EIGEN_double(  const Eigen::ArrayBase<Derived> &x,
                                     const std::string &fn,
                                     const std::string &vect_type, 
                                     const bool &skip_checks
) {
   
       Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> x_copy = x.matrix();
       fn_EIGEN_Ref_double( Eigen::Ref<Eigen::Matrix<typename Derived::Scalar,
                            Derived::RowsAtCompileTime,
                            Derived::ColsAtCompileTime>>(x_copy),
                            fn,
                            vect_type,
                            skip_checks);
       return x_copy;

}




////// ------- in-place callers (no new vector on each call) ---------------------------------------------------

//// ---- Into an existing destination (a vector, a matrix, or a column or block of one): x is copied into the
////      destination and fn is applied there, in place, so that no new vector is allocated and no second copy is
////      made on each call (fn_EIGEN_double() returns a new vector, which is then copied into the destination).
////      The values are the same as with fn_EIGEN_double(), bit for bit. x may be the destination itself.
template <typename T_destination, typename T_x>
inline         void   fn_EIGEN_double_into(      T_destination &&destination,
                                                const T_x &x,
                                                const std::string &fn,
                                                const std::string &vect_type,
                                                const bool &skip_checks
) {

       destination.resize(x.rows(), x.cols());   //// (a plain vector or matrix takes x's size, as with =)
       destination.array() = x.array();
       fn_EIGEN_Ref_double<Eigen::Matrix<double, -1, -1>>( destination,
                                                           fn,
                                                           vect_type,
                                                           skip_checks);

}


//// ---- Into a per-thread buffer, for a call used inside a longer expression (a sum, an update with +=, a
////      product, rows picked by an index): the buffer grows to the largest size asked for and is never freed, so
////      that no new vector is allocated on each call. The values are the same as with fn_EIGEN_double(), bit for
////      bit. The next call on the same thread overwrites the buffer: one call per statement.
template <typename T_x>
inline  Eigen::Map<Eigen::Matrix<double, -1, -1>>   fn_EIGEN_double_in_a_per_thread_buffer(
                                                                    const T_x &x,
                                                                    const std::string &fn,
                                                                    const std::string &vect_type,
                                                                    const bool &skip_checks
) {

       static thread_local std::vector<double> buffer;
       const Eigen::Index n_rows = x.rows();
       const Eigen::Index n_cols = x.cols();
       if (static_cast<Eigen::Index>(buffer.size()) < n_rows * n_cols) buffer.resize(n_rows * n_cols);
       Eigen::Map<Eigen::Matrix<double, -1, -1>> x_in_the_buffer(buffer.data(), n_rows, n_cols);
       x_in_the_buffer.array() = x.array();
       fn_EIGEN_Ref_double<Eigen::Matrix<double, -1, -1>>( x_in_the_buffer,
                                                           fn,
                                                           vect_type,
                                                           skip_checks);
       return x_in_the_buffer;

}







#endif

  
  
  
  
  
















