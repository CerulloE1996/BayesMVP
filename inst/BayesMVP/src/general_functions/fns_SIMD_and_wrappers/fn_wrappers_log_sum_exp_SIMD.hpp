
#pragma once 


#ifndef FN_WRAPPERS_LOG_SUM_EXP_SIMD_HPP
#define FN_WRAPPERS_LOG_SUM_EXP_SIMD_HPP


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

#include <string>
#include <stdexcept>

#include "fn_SIMD_level_resolver.hpp"



using namespace Eigen;




//// 2026-09-22: compiled on AVX-512 builds too (the guard was "__AVX2__ && !AVX-512"), so that vect_type = "AVX2" runs the 4-lane
//// kernels there instead of falling through fn_log_sum_exp_2d_double below to Stan's scalar log_sum_exp.
#if BAYESMVP_COMPILED_AVX2_KERNELS
  
ALWAYS_INLINE Eigen::Matrix<double, -1, 1>   fast_log_sum_exp_2d_AVX2_double(  Eigen::Ref<Eigen::Matrix<double, -1, -1>>   x) {
    
      const int N = x.rows();
      const std::string vect_type_exp = "AVX2";
      const std::string vect_type_log = "AVX2";
      
      Eigen::Matrix<double, -1, 1> log_sum_abs_result(N);
      Eigen::Matrix<double, -1, 1> container_max_logs(N);
      
      log_sum_exp_general(x, 
                          vect_type_exp,
                          vect_type_log,
                          log_sum_abs_result,
                          container_max_logs);
      
      return log_sum_abs_result;
    
} 



#endif




#if BAYESMVP_COMPILED_AVX512_KERNELS
  
ALWAYS_INLINE Eigen::Matrix<double, -1, 1>   fast_log_sum_exp_2d_AVX512_double(  Eigen::Ref<Eigen::Matrix<double, -1, -1>> x) {
      
      const int N = x.rows();
      const std::string vect_type_exp = "AVX512";
      const std::string vect_type_log = "AVX512";
      
      Eigen::Matrix<double, -1, 1> log_sum_abs_result(N);
      Eigen::Matrix<double, -1, 1> container_max_logs(N);
      
      log_sum_exp_general(x, 
                          vect_type_exp,
                          vect_type_log,
                          log_sum_abs_result,
                          container_max_logs);
      
      return log_sum_abs_result;
    
}
  
  
#endif




ALWAYS_INLINE  Eigen::Matrix<double, -1, 1> fn_log_sum_exp_2d_double(      Eigen::Ref<Eigen::Matrix<double, -1, -1>>  x,    // Eigen::Matrix<double, -1, 2> &x,
                                                                           const std::string &vect_type = "Stan",
                                                                           const bool &skip_checks = false) {
    
    {
      if (vect_type == "Eigen") {
        return  log_sum_exp_2d_Eigen_double(x);
      } else if (vect_type == "Stan") {
        return  log_sum_exp_2d_Stan_double(x);
      } else if (vect_type == "AVX2") {
        //// 2026-09-22: a SIMD level this build did not compile now THROWS (as in fn_EIGEN_Ref_double; fn_SIMD_level_resolver.hpp).
        //// Before, "AVX2" was compiled out on AVX-512 builds (and "AVX512" on AVX2 builds) and the request silently fell through
        //// to the Stan return at the bottom of this function. Same string comparisons as before (no extra per-call cost).
#if BAYESMVP_COMPILED_AVX2_KERNELS
        if (skip_checks == false)   return  fast_log_sum_exp_2d_AVX2_double(x);
        else                        return  fast_log_sum_exp_2d_AVX2_double(x);
#else
        throw std::invalid_argument(fn_BayesMVP_SIMD_level_not_compiled_message(vect_type));
#endif
      } else if (vect_type == "AVX512") {
#if BAYESMVP_COMPILED_AVX512_KERNELS
        if (skip_checks == false)   return  fast_log_sum_exp_2d_AVX512_double(x);
        else                        return  fast_log_sum_exp_2d_AVX512_double(x);
#else
        throw std::invalid_argument(fn_BayesMVP_SIMD_level_not_compiled_message(vect_type));
#endif
      } else {
        return  log_sum_exp_2d_Stan_double(x);
      }
      
    }
    
    return  log_sum_exp_2d_Stan_double(x);
    
  }
  
  
  
  
  






#endif

  
  
  
  
  