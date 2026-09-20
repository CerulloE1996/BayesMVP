
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


 
 




ALWAYS_INLINE double mvp_std_exp(double x) { 
     return stan::math::exp(x);
}
ALWAYS_INLINE double mvp_std_log(double x) { 
  return stan::math::log(x);
}
ALWAYS_INLINE double mvp_std_log1p(double x) { 
  return stan::math::log1p(x);
}
ALWAYS_INLINE double mvp_std_log1m(double x) { 
  return stan::math::log1m(x);
} 
ALWAYS_INLINE double mvp_std_logit(double x) { 
  return stan::math::logit(x);
} 
ALWAYS_INLINE double mvp_std_tanh(double x) { 
  return stan::math::tanh(x);
} 
ALWAYS_INLINE double mvp_std_Phi_approx(double x) { 
  return stan::math::Phi_approx(x);
}
ALWAYS_INLINE double mvp_std_Phi(double x) { 
  return stan::math::Phi(x);
}
ALWAYS_INLINE double mvp_std_inv_Phi(double x) { 
  return stan::math::inv_Phi(x);
}
ALWAYS_INLINE double mvp_std_inv_logit(double x) { 
  return stan::math::inv_logit(x);
}
ALWAYS_INLINE double mvp_std_log_inv_logit(double x) { 
  return stan::math::log_inv_logit(x); 
}



 


  
template <typename T>
ALWAYS_INLINE      void          fn_void_Ref_double_Stan(   Eigen::Ref<T> x,
                                                            const std::string &fn,
                                                            const bool &skip_checks) {

    
     if        (fn == "exp")   {   x =   stan::math::exp(x);
     } else if (fn == "log")   {   x =   stan::math::log(x);
     } else if (fn == "log1p") {   x =   stan::math::log1p(x);
     } else if (fn == "log1m") {   x =   stan::math::log1m(x);
     } else if (fn == "logit") {   x =   stan::math::logit(x);
     } else if (fn == "tanh")  {   x =   stan::math::tanh(x);
     } else if (fn == "Phi_approx") {      x =    stan::math::Phi_approx(x);
     } else if (fn == "log_Phi_approx") {  
           auto x_array = x.array(); 
           auto x_sq = x_array.square();  
           auto temp = 0.07056*x_sq*x_array  +  1.5976*x_array;
           x =   stan::math::log_inv_logit(temp.matrix());  
     } else if (fn == "inv_Phi_approx") {  x =    inv_Phi_approx_Stan(x);
     } else if (fn == "Phi") {             x =    stan::math::Phi(x);
     } else if (fn == "inv_Phi") {         x =    stan::math::inv_Phi(x);
     } else if (fn == "inv_Phi_approx_from_logit_prob") {  x =  inv_Phi_approx_from_logit_prob_Stan(x);
     } else if (fn == "inv_Phi_from_log_prob") {  x =   stan::math::std_normal_log_qf(x);
     } else if (fn == "inv_logit") {  x =   stan::math::inv_logit(x);
     } else if (fn == "log_inv_logit") {  x =   stan::math::log_inv_logit(x);
     } else {
    
     }


}
  
  
  
   



 




 






#endif

 
 
 
 
 
 
 
 