#pragma once 
 
#ifndef FN_WRAPPERS_STAN_HPP
#define FN_WRAPPERS_STAN_HPP
 
  
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


 
 




inline double mvp_std_exp(double x) { 
     return stan::math::exp(x);
}
inline double mvp_std_log(double x) { 
  return stan::math::log(x);
}
inline double mvp_std_log1p(double x) { 
  return stan::math::log1p(x);
}
inline double mvp_std_log1m(double x) { 
  return stan::math::log1m(x);
} 
inline double mvp_std_logit(double x) { 
  return stan::math::logit(x);
} 
inline double mvp_std_tanh(double x) { 
  return stan::math::tanh(x);
} 
inline double mvp_std_Phi_approx(double x) { 
  return stan::math::Phi_approx(x);
}
inline double mvp_std_Phi(double x) { 
  return stan::math::Phi(x);
}
inline double mvp_std_inv_Phi(double x) { 
  return stan::math::inv_Phi(x);
}
inline double mvp_std_inv_logit(double x) { 
  return stan::math::inv_logit(x);
}
inline double mvp_std_log_inv_logit(double x) { 
  return stan::math::log_inv_logit(x); 
}



 


  
template <typename T>
ALWAYS_INLINE      void          fn_void_Ref_double_Stan(    Eigen::Ref<T> x,
                                                      const std::string &fn,
                                                      const bool &skip_checks) {

    
     if        (fn == "exp")   {   x =   stan::math::exp(x);
     } else if (fn == "log")   {   x =   stan::math::log(x);
     } else if (fn == "log1p") {   x =   stan::math::log1p(x);
     } else if (fn == "log1m") {   x =   stan::math::log1m(x);
     } else if (fn == "logit") {   x =   stan::math::logit(x);
     } else if (fn == "tanh")  {   x =   stan::math::tanh(x);
     } else if (fn == "Phi_approx") {      x =    stan::math::Phi_approx(x);
     } else if (fn == "log_Phi_approx") {    x =   stan::math::log_inv_logit((0.07056*x.array().square()*x.array()  +  1.5976*x.array()).matrix());  
     } else if (fn == "inv_Phi_approx") {  x =    inv_Phi_approx_Stan(x);
     } else if (fn == "Phi") {             x =    stan::math::Phi(x);
     } else if (fn == "inv_Phi") {         x =    stan::math::inv_Phi(x);
     } else if (fn == "inv_Phi_approx_from_logit_prob") {  x =  inv_Phi_approx_from_logit_prob_Stan(x);
     } else if (fn == "inv_Phi_from_log_prob") {  x =   stan::math::std_normal_log_qf(x.matrix());
     } else if (fn == "inv_logit") {  x =   stan::math::inv_logit(x.matrix());
     } else if (fn == "log_inv_logit") {  x =   stan::math::log_inv_logit(x.matrix());
     } else {
    
     }


}
  
  
  
   



// ALWAYS_INLINE   Eigen::Matrix<double, -1, 1>   log_sum_exp_2d_Stan_double( const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> x )  {
//   
//   using namespace stan::math;
//   
//   int N = x.rows();
//   Eigen::Matrix<double, -1, -1>   rowwise_maxes_2d_array(N, 2);
//   rowwise_maxes_2d_array.col(0) = x.array().rowwise().maxCoeff().matrix(); 
//   rowwise_maxes_2d_array.col(1) = rowwise_maxes_2d_array.col(0);
//   
//   //// Eigen::Matrix<double, -1, 1>  rowwise_maxes_1d_vec = rowwise_maxes_2d_array.col(0);
//   Eigen::Matrix<double, -1, 1>  sum_exp_vec =  stan::math::exp(  (x.array()  -  rowwise_maxes_2d_array.array()).matrix() ).rowwise().sum() ;
//   Eigen::Matrix<double, -1, 1>  out =  ( rowwise_maxes_2d_array.col(0).array()    +   stan::math::log(sum_exp_vec).array() ).matrix(); 
//   
//   return  out;
//   
// }
// 
// 

 




 






#endif

 
 
 
 
 
 
 
 