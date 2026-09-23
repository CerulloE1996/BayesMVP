
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
 
 
 
 
 
 
 
inline  Eigen::Matrix<double, -1, 1>  fn_lp_grad_Return(    const std::string  Model_type,
                                                            const bool force_autodiff,
                                                            const bool force_PartialLog,
                                                            const bool multi_attempts,
                                                            const Eigen::Matrix<double, -1, 1> theta_main_vec_ref,
                                                            const Eigen::Matrix<double, -1, 1> theta_us_vec_ref,
                                                            const Eigen::Matrix<int, -1, -1> y_ref,
                                                            const std::string grad_option,
                                                            const Model_fn_args_struct  Model_args_as_cpp_struct,
                                                            const Stan_model_struct Stan_model_as_cpp_struct) {
   
   const int N = Model_args_as_cpp_struct.N;
   const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
   int n_params_main = Model_args_as_cpp_struct.n_params_main;
   int n_params = n_params_main + n_nuisance;
   
   Eigen::Matrix<double, -1, 1> lp_and_grad_outs(1 + N + n_params);
   lp_and_grad_outs.setZero();
   
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
                         //// fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct);
                         std::cout << "ERROR: For the latent_trait model, the MANUAL-LOG-SCALE lp_grad function isn't yet working fully, so please set force_PartialLog = FALSE" << std::endl;     //// bookmark
                         throw std::invalid_argument("force_PartialLog = TRUE is unsupported for Model_type = latent_trait; use force_PartialLog = FALSE.");
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
     
     //// #if HAS_BRIDGESTAN_H
                   
                   Eigen::Matrix<double, -1, 1> params(n_params);
                   if (n_nuisance > 10) {
                     params.head(n_nuisance) = theta_us_vec_ref;
                     params.tail(n_params_main) = theta_main_vec_ref;
                   } else {  
                     params.resize(n_params_main);
                     params = theta_main_vec_ref;
                     n_params = n_params_main;
                     n_params_main = theta_main_vec_ref.rows();
                   }
                   
                   if (n_nuisance > 10) {
                     if (grad_option == "main_only") { 
                       
                       lp_and_grad_outs.segment(1 + n_nuisance, n_params_main)  =    fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,
                                                params, n_params_main, n_nuisance, 
                                                lp_and_grad_outs.head(1 + n_params)).segment(1 + n_nuisance, n_params_main);
                       
                     } else if (grad_option == "us_only") {  // only nuisance 
                       
                       lp_and_grad_outs.head(1 + n_nuisance)  =  fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,
                                             params, n_params_main, n_nuisance,
                                             lp_and_grad_outs.head(1 + n_params)).head(1 + n_nuisance);
                       
                     } else {  /// otherwise compute entire gradient
                       
                       lp_and_grad_outs.head(1 + n_params)  =  fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct, 
                                             params, n_params_main, n_nuisance, 
                                             lp_and_grad_outs.head(1 + n_params)).head(1 + n_params);
                       
                     }
                   } else { 
                     lp_and_grad_outs.head(1 + n_params_main)  =    fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,  
                                           params, n_params_main, n_nuisance, 
                                           lp_and_grad_outs.head(1 + n_params_main)); 
                   }
                   
                   ////  #endif
     
   }
   
   return lp_and_grad_outs;
   
}
  
  
  
  
  
  
  
  
  
  
  
  


inline void  fn_lp_grad_InPlace(         Eigen::Ref<Eigen::Matrix<double, -1, 1>> lp_and_grad_outs,
                                         const std::string  &Model_type,
                                         const bool force_autodiff,
                                         const bool force_PartialLog,
                                         const bool multi_attempts,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                         const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                         const std::string &grad_option,
                                         const Model_fn_args_struct  &Model_args_as_cpp_struct,
                                         const Stan_model_struct &Stan_model_as_cpp_struct) {

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
            ////  #endif
                           
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
                                            
                                           //// fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace_process(lp_and_grad_outs, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct);
                                           //// bookmark
                                           std::cout << "ERROR: For the latent_trait model, the MANUAL-LOG-SCALE lp_grad function isn't yet working fully, so please set force_PartialLog = FALSE" << std::endl; 
                                           throw std::invalid_argument("force_PartialLog = TRUE is unsupported for Model_type = latent_trait; use force_PartialLog = FALSE.");
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
     
     //// #if HAS_BRIDGESTAN_H
                 
                 Eigen::Matrix<double, -1, 1> params(n_params);
                 if (n_nuisance > 10) {
                   params.head(n_nuisance) = theta_us_vec_ref;
                   params.tail(n_params_main) = theta_main_vec_ref;
                 } else { 
                   params.resize(n_params_main);
                   params = theta_main_vec_ref;
                   n_params = n_params_main;
                   n_params_main = theta_main_vec_ref.rows();
                 }
                 
            if (n_nuisance > 10) {
                 if (grad_option == "main_only") { 
                     
                  lp_and_grad_outs.segment(1 + n_nuisance, n_params_main)  =    fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,
                                                                                                              params, n_params_main, n_nuisance, 
                                                                                                              lp_and_grad_outs.head(1 + n_params)).segment(1 + n_nuisance, n_params_main);
                   
                 } else if (grad_option == "us_only") {  // only nuisance 
                    
                   lp_and_grad_outs.head(1 + n_nuisance)  =  fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,
                                                                                           params, n_params_main, n_nuisance,
                                                                                           lp_and_grad_outs.head(1 + n_params)).head(1 + n_nuisance);
                   
                 } else {  /// otherwise compute entire gradient
                   
                   lp_and_grad_outs.head(1 + n_params)  =  fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct, 
                                                                                         params, n_params_main, n_nuisance, 
                                                                                         lp_and_grad_outs.head(1 + n_params)).head(1 + n_params);
                   
                 }
            } else { 
                lp_and_grad_outs.head(1 + n_params_main)  =    fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,  
                                                                                             params, n_params_main, n_nuisance, 
                                                                                             lp_and_grad_outs.head(1 + n_params_main)); 
            }
            
            //// #endif
              
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
                                                     const Model_fn_args_struct  &Model_args_as_cpp_struct,
                                                    // MVP_ThreadLocalWorkspace &MVP_workspace,
                                                     const Stan_model_struct &Stan_model_as_cpp_struct) {
 
 
  const int N = Model_args_as_cpp_struct.N;
  const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
  const int n_params_main = Model_args_as_cpp_struct.n_params_main;
  const int n_params = n_params_main + n_nuisance;
 
   Eigen::Matrix<double, -1, 1> lp_and_grad_outs =     Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
   
   
   fn_lp_grad_InPlace(        lp_and_grad_outs, 
                              Model_type, 
                              force_autodiff, force_PartialLog, multi_attempts,
                              theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option,
                              Model_args_as_cpp_struct,// MVP_workspace, 
                              Stan_model_as_cpp_struct);
   
   
   return  lp_and_grad_outs;
 
}


 
 
 
 
 
 
 
 
 
 
 
inline  void  fn_lp_only_InPlace(      double &lp,
                                       const std::string  &Model_type,
                                       const bool force_autodiff,
                                       const bool force_PartialLog,
                                       const bool multi_attempts,
                                       const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                       const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                       const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                       const Model_fn_args_struct  &Model_args_as_cpp_struct,
                                       const Stan_model_struct &Stan_model_as_cpp_struct) {

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
                    //// #if COMPILE_LATENT_TRAIT
                    
                        if (force_PartialLog == true) {  
                          //// lp = (   fn_lp_grad_LT_LC_PartialLog_MD_and_AD(theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct)  ).head(1).eval()(0);
                          //// bookmark
                          std::cout << "ERROR: For the latent_trait model, the MANUAL-LOG-SCALE lp_grad function isn't yet working fully, so please set force_PartialLog = FALSE" << std::endl; 
                          throw std::invalid_argument("force_PartialLog = TRUE is unsupported for Model_type = latent_trait; use force_PartialLog = FALSE.");
                        } else { 
                          lp =  (  fn_lp_grad_LT_LC_NoLog_MD_and_AD(theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
                        }
                        //// #endif
                        
                  }
          } else {  /// use autodiff 
            
            if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
                    
                    lp =  (  fn_lp_and_grad_MVP_Pinkney_AD_log_scale(theta_main_vec_ref, theta_us_vec_ref, y_ref,  grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
                    
                  } else if (Model_type == "latent_trait") {
                    //// #if COMPILE_LATENT_TRAIT
                    lp =  (  fn_lp_and_grad_LC_LT_AD_log_scale(theta_main_vec_ref, theta_us_vec_ref, y_ref,  grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
                    //// #endif
                    
                  } 
            
          }

  }  else if (Model_type == "Stan" )  { 
    
    
              //// #if HAS_BRIDGESTAN_H
              
              Eigen::Matrix<double, -1, 1> params(n_params);
              if (n_nuisance > 10) {
                  params.head(n_nuisance) = theta_us_vec_ref;
                  params.tail(n_params_main) = theta_main_vec_ref;
              } else { 
                  params = theta_main_vec_ref;
                  n_params = n_params_main;
                  params.resize(n_params_main);
              }
              
              Eigen::Matrix<double, -1, 1> lp_and_grad_outs(1 + N + n_params);
              
              if (grad_option == "main_only") { 
                
                lp  =    fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,  params, n_params_main, n_nuisance, lp_and_grad_outs).head(1).eval()(0);
                
              } else if (grad_option == "us_only") {  // only nuisance 
                
                lp  =  fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,   params, n_params_main, n_nuisance, lp_and_grad_outs).head(1).eval()(0);
                
              } else {  /// otherwise compute entire gradient
                
                lp  =  fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,  params, n_params_main, n_nuisance, lp_and_grad_outs).head(1).eval()(0);
                
              }
              
              //// #endif
    
  }
  
}


 
 
 
 
 
 
 
 
 
 
