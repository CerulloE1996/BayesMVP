
#pragma once

 
 

#include <sstream>
#include <stdexcept>  
#include <complex>
///#include <dlfcn.h> // For dynamic loading 
#include <map>
#include <vector>  
#include <string> 
#include <stdexcept>
#include <stdio.h>
#include <iostream>
 
#include <stan/model/model_base.hpp>  
 
#include <stan/io/array_var_context.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/dump.hpp> 

 
#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/rapidjson_parser.hpp>   
 
 


#include <Eigen/Dense>
 
 


#if defined(__AVX2__) || defined(__AVX512F__) 
#include <immintrin.h>
#endif
 
 
 
 
 
 
 
//  
// inline  Eigen::Matrix<double, -1, 1>  fn_lp_grad_Return(          const std::string  Model_type,
//                                                           const bool force_autodiff,
//                                                           const bool force_PartialLog,
//                                                           const bool multi_attempts,
//                                                           const Eigen::Matrix<double, -1, 1> theta_main_vec_ref,
//                                                           const Eigen::Matrix<double, -1, 1> theta_us_vec_ref,
//                                                           const Eigen::Matrix<int, -1, -1> y_ref,
//                                                           const std::string grad_option,
//                                                           const Model_fn_args_struct  Model_args_as_cpp_struct) {
//    
//    const int N = Model_args_as_cpp_struct.N;
//    const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
//    int n_params_main = Model_args_as_cpp_struct.n_params_main;
//    int n_params = n_params_main + n_nuisance;
//    
//    Eigen::Matrix<double, -1, 1> lp_and_grad_outs(1 + N + n_params);
//    lp_and_grad_outs.setZero();
//    
//    if  ((Model_type == "latent_trait") || (Model_type == "MVP") || (Model_type == "LC_MVP")) { //// use a built-in, fast manual-gradient model
//      
//                  if (multi_attempts == true) {
//                    
//                    if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
//                      
//                      fn_lp_grad_MVP_multi_attempts_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct);
//                      
//                    } else if (Model_type == "latent_trait") {
//                      
//             //// #if COMPILE_LATENT_TRAIT
//                      fn_lp_grad_LT_LC_multi_attempts_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct);
//             //// #endif 
//                      
//                    }
//                    
//                  } else {
//                    
//                    if (force_autodiff == false) {  // Handle cases where autodiff is not forced
//                      
//                      if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
//                        
//                        if (force_PartialLog == true) {
//                          fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct);
//                        } else {
//                          fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct);
//                        }
//                        
//                      } else if (Model_type == "latent_trait") {
//            ////  #if COMPILE_LATENT_TRAIT
//                        if (force_PartialLog == true) {
//                          fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct);
//                        } else { 
//                          fn_lp_grad_LT_LC_NoLog_MD_and_AD_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct);
//                        } 
//            ////  #endif
//                      }
//                      
//                    } else { /// autodiff
//                      
//                      if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
//                        
//                        fn_lp_and_grad_MVP_Pinkney_AD_log_scale_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref,  grad_option, Model_args_as_cpp_struct); 
//                        
//                      } else if (Model_type == "latent_trait") {
//             //// #if COMPILE_LATENT_TRAIT
//                        fn_lp_and_grad_LC_LT_AD_log_scale_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref,  grad_option, Model_args_as_cpp_struct); 
//             //// #endif
//                         
//                      }
//                      
//                    }
//                    
//                  }
//      
//    }  else if (Model_type == "Stan" )  {
// 
//      
//      
//    }
//    
//    return lp_and_grad_outs;
//    
// }
//   
//   
//   
//   
//   
  
  
  
  
  
  
  

inline  void  fn_lp_grad_InPlace(                Eigen::Ref<Eigen::Matrix<double, -1, 1>> lp_and_grad_outs,
                                         const std::string  &Model_type,
                                         const bool force_autodiff,
                                         const bool force_PartialLog,
                                         const bool multi_attempts,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                         const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                         const std::string &grad_option,
                                         const Model_fn_args_struct  &Model_args_as_cpp_struct) {

  const int N = Model_args_as_cpp_struct.N;
  const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
  int n_params_main = Model_args_as_cpp_struct.n_params_main;
  int n_params = n_params_main + n_nuisance;
   
   if  ((Model_type == "latent_trait") || (Model_type == "MVP") || (Model_type == "LC_MVP")) { //// use a built-in, fast manual-gradient model
     
     
     if (multi_attempts == true) {
       
          if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
            
                            fn_lp_grad_MVP_multi_attempts_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct);
            
          } else if (Model_type == "latent_trait") {
            
                       //// #if COMPILE_LATENT_TRAIT
                           fn_lp_grad_LT_LC_multi_attempts_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct);
                       //// #endif
                           
          }
       
     } else {
     
                 
                    
                     if (force_autodiff == false) {  // Handle cases where autodiff is not forced
                       
                             if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
                               
                                   if (force_PartialLog == true) {
                                          fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct);
                                   } else {
                                          fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct);
                                   }
                                   
                             } else if (Model_type == "latent_trait") {
                  //// #if COMPILE_LATENT_TRAIT
                                   if (force_PartialLog == true) {
                                           fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct);
                                   } else { 
                                           fn_lp_grad_LT_LC_NoLog_MD_and_AD_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct);
                                   } 
                  //// #endif
                             }
                       
                     } else { /// autodiff
                       
                             if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
                               
                                       fn_lp_and_grad_MVP_Pinkney_AD_log_scale_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref,  grad_option, Model_args_as_cpp_struct); 
                               
                             } else if (Model_type == "latent_trait") {
            //// #if COMPILE_LATENT_TRAIT
                                       fn_lp_and_grad_LC_LT_AD_log_scale_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref,  grad_option, Model_args_as_cpp_struct); 
            //// #endif
                               
                             }
                           
                     }
         
     }
         
   }  else if (Model_type == "Stan" )  { 

              
   }
   
}










inline  Eigen::Matrix<double, -1, 1 >         fn_lp_grad(    const std::string  &Model_type,
                                                     const bool force_autodiff,
                                                     const bool force_PartialLog,
                                                     const bool multi_attempts,
                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                     const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                     const std::string &grad_option,
                                                     const Model_fn_args_struct  &Model_args_as_cpp_struct) {
 
 
  const int N = Model_args_as_cpp_struct.N;
  const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
  const int n_params_main = Model_args_as_cpp_struct.n_params_main;
  const int n_params = n_params_main + n_nuisance;
 
   Eigen::Matrix<double, -1, 1> lp_and_grad_outs =     Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
   
   
   fn_lp_grad_InPlace(        lp_and_grad_outs, 
                              Model_type, 
                              force_autodiff, force_PartialLog, multi_attempts,
                              theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option,
                              Model_args_as_cpp_struct);
   
   
   return  lp_and_grad_outs;
 
}


 
 
 
 
 
 
 
 
 
 
inline  void  fn_lp_only_InPlace(              double &lp,
                                       const std::string  &Model_type,
                                       const bool force_autodiff,
                                       const bool force_PartialLog,
                                       const bool multi_attempts,
                                       const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                       const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                       const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                       const Model_fn_args_struct  &Model_args_as_cpp_struct) {

  const int N = Model_args_as_cpp_struct.N;
  const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
  int n_params_main = Model_args_as_cpp_struct.n_params_main;
  int n_params = n_params_main + n_nuisance;

  const std::string  grad_option = "none";
  
  if  ((Model_type == "latent_trait") || (Model_type == "MVP") || (Model_type == "LC_MVP")) { //// use a built-in, fast manual-gradient model
        
          // Handle cases where autodiff is not forced
          if (force_autodiff == false) {
            
                  if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
                    
                        if (force_PartialLog == true) {
                          lp = (   fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD(theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct)  ).head(1).eval()(0);
                        } else {
                          lp =  (  fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD(theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
                        }
                        
                  } else if (Model_type == "latent_trait") {
/// #if COMPILE_LATENT_TRAIT
                    
                        if (force_PartialLog == true) {
                          lp = (   fn_lp_grad_LT_LC_PartialLog_MD_and_AD(theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct)  ).head(1).eval()(0);
                        } else { 
                          lp =  (  fn_lp_grad_LT_LC_NoLog_MD_and_AD(theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
                        }
/// #endif
                        
                  }
          } else {  /// use autodiff 
            
            if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
                    
                    lp =  (  fn_lp_and_grad_MVP_Pinkney_AD_log_scale(theta_main_vec_ref, theta_us_vec_ref, y_ref,  grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
                    
                  } else if (Model_type == "latent_trait") {
/// #if COMPILE_LATENT_TRAIT
                    lp =  (  fn_lp_and_grad_LC_LT_AD_log_scale(theta_main_vec_ref, theta_us_vec_ref, y_ref,  grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
//// #endif
                    
                  } 
            
          }

  }  else if (Model_type == "Stan" )  { 
    
 
  }
  
}


 
 
 
 
 
 
 
 
 
 










