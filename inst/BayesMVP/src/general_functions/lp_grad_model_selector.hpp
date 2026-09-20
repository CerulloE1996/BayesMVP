
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
                                         const Stan_model_struct &Stan_model_as_cpp_struct,
                                         std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                         const int n_threads_WCP
) {

  const int N = Model_args_as_cpp_struct.N;
  const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
  int n_params_main = Model_args_as_cpp_struct.n_params_main;
  int n_params = n_params_main + n_nuisance;
   
   if  ((Model_type == "latent_trait") || (Model_type == "MVP") ||  (Model_type == "MVOP") || (Model_type == "LC_MVP") || (Model_type == "LC_MVOP")) { //// use a built-in, fast manual-gradient model
     
       if (multi_attempts == true) {
                     
                        if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
                          
                                          fn_lp_grad_MVP_multi_attempts_InPlace_process(lp_and_grad_outs,
                                                                                        theta_main_vec_ref,
                                                                                        theta_us_vec_ref,
                                                                                        y_ref,
                                                                                        grad_option,
                                                                                        Model_args_as_cpp_struct,
                                                                                        LC_MVP_ws_structs,
                                                                                        n_threads_WCP,
                                                                                        Model_args_as_cpp_struct.autodiff_fallback,
                                                                                        Model_type == "MVP");
                          
                        } else if (Model_type == "latent_trait") {
                          
                          //// #if COMPILE_LATENT_TRAIT
                                          fn_lp_grad_LT_LC_multi_attempts_InPlace_process(lp_and_grad_outs,
                                                                                          theta_main_vec_ref,
                                                                                          theta_us_vec_ref,
                                                                                          y_ref,
                                                                                          grad_option,
                                                                                          Model_args_as_cpp_struct,
                                                                                          !Model_args_as_cpp_struct.autodiff_fallback_option_is_set ||
                                                                                              Model_args_as_cpp_struct.autodiff_fallback);
                          ////  #endif
                                         
                        } else if ((Model_type == "LC_MVOP") || (Model_type == "MVOP")) {
                          
                                          fn_lp_grad_MVOP_multi_attempts_InPlace_process(lp_and_grad_outs,
                                                                                         theta_main_vec_ref,
                                                                                         theta_us_vec_ref,
                                                                                         y_ref,
                                                                                         grad_option,
                                                                                         Model_args_as_cpp_struct,
                                                                                         LC_MVP_ws_structs,
                                                                                         n_threads_WCP,
                                                                                         Model_args_as_cpp_struct.autodiff_fallback);
                          
                        }
           
       } else {
           
                        if (force_autodiff == false) {  // Handle cases where autodiff is not forced
                           
                                 if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
                                   
                                       if (force_PartialLog == true) {
                                         
                                              fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process(lp_and_grad_outs,
                                                                                                             theta_main_vec_ref, 
                                                                                                             theta_us_vec_ref,
                                                                                                             y_ref, 
                                                                                                             grad_option, 
                                                                                                             Model_args_as_cpp_struct,
                                                                                                             LC_MVP_ws_structs,
                                                                                                             n_threads_WCP);
                                         
                                       } else {
                                         
                                              fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(  lp_and_grad_outs, 
                                                                                                          theta_main_vec_ref,
                                                                                                          theta_us_vec_ref,
                                                                                                          y_ref, 
                                                                                                          grad_option, 
                                                                                                          Model_args_as_cpp_struct,
                                                                                                          LC_MVP_ws_structs,
                                                                                                          n_threads_WCP);
                                         
                                       }
                                       
                                 } else if (Model_type == "latent_trait") {
                                   
                                       //// #if COMPILE_LATENT_TRAIT
                                       if (force_PartialLog == true) {
                                                
                                                fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace_process(  lp_and_grad_outs, 
                                                                                                        theta_main_vec_ref,
                                                                                                        theta_us_vec_ref, 
                                                                                                        y_ref, 
                                                                                                        grad_option, 
                                                                                                        Model_args_as_cpp_struct);
                                               //// bookmark
                                               //  std::cout << "ERROR: For the latent_trait model, the MANUAL-LOG-SCALE lp_grad function isn't yet working fully, so please set force_PartialLog = FALSE" << std::endl;
                                              
                                       } else {
                                         
                                               fn_lp_grad_LT_LC_NoLog_MD_and_AD_InPlace_process(  lp_and_grad_outs,
                                                                                                  theta_main_vec_ref, 
                                                                                                  theta_us_vec_ref, 
                                                                                                  y_ref, 
                                                                                                  grad_option,
                                                                                                  Model_args_as_cpp_struct);
                                         
                                       }
                                       //// #endif
                                       
                                 } else if ((Model_type == "LC_MVOP") || (Model_type == "MVOP")) {
                                   
                                       fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process( lp_and_grad_outs, 
                                                                                                   theta_main_vec_ref,
                                                                                                   theta_us_vec_ref,
                                                                                                   y_ref, 
                                                                                                   grad_option, 
                                                                                                   Model_args_as_cpp_struct,
                                                                                                   LC_MVP_ws_structs,
                                                                                                   n_threads_WCP);
                                   
                                 }
                           
                        } else if (force_autodiff == true) { /// autodiff
                           
                                 if (Model_type == "LC_MVP") {
                                   
                                           fn_lp_and_grad_MVP_Pinkney_AD_log_scale_InPlace_process(lp_and_grad_outs,
                                                                                                   theta_main_vec_ref, 
                                                                                                   theta_us_vec_ref, 
                                                                                                   y_ref,  
                                                                                                   grad_option, 
                                                                                                   Model_args_as_cpp_struct,
                                                                                                   LC_MVP_ws_structs,
                                                                                                   n_threads_WCP);
                                   
                                 } else if (Model_type == "MVP") {
                                   
                                           fn_lp_and_grad_std_MVP_Pinkney_AD_log_scale_InPlace_process(lp_and_grad_outs,
                                                                                                       theta_main_vec_ref, 
                                                                                                       theta_us_vec_ref, 
                                                                                                       y_ref,  
                                                                                                       grad_option,
                                                                                                       Model_args_as_cpp_struct,
                                                                                                       LC_MVP_ws_structs,
                                                                                                       n_threads_WCP); 
                                   
                                 } else if (Model_type == "latent_trait") {
                                   
                                   //// #if COMPILE_LATENT_TRAIT
                                           fn_lp_and_grad_LC_LT_AD_log_scale_InPlace_process(lp_and_grad_outs, 
                                                                                             theta_main_vec_ref, 
                                                                                             theta_us_vec_ref, 
                                                                                             y_ref,
                                                                                             grad_option, 
                                                                                             Model_args_as_cpp_struct); 
                                   //// #endif
                                   
                                 } else if ((Model_type == "LC_MVOP") || (Model_type == "MVOP")) {
                                    
                                           fn_lp_and_grad_MVOP_Pinkney_AD_log_scale_InPlace_process(lp_and_grad_outs,
                                                                                                    theta_main_vec_ref,
                                                                                                    theta_us_vec_ref,
                                                                                                    y_ref,
                                                                                                    grad_option,
                                                                                                    Model_args_as_cpp_struct,
                                                                                                    LC_MVP_ws_structs,
                                                                                                    n_threads_WCP);
                                           
                                 }
                               
                        }
             
       }
         
  }  else if (Model_type == "Stan" )  { 
     
     //// #if HAS_BRIDGESTAN_H
                 
                 //// ---- Build the COMPLETE unconstrained vector: nuisance block FIRST,
                 //// main block SECOND. This must hold for EVERY n_nuisance (including 0
                 //// and small blocks <= 10): the Stan log-density is defined on all
                 //// declared coordinates. The previous "n_nuisance > 10" shortcut
                 //// silently dropped small nuisance blocks (and misplaced the gradient),
                 //// which is wrong whenever the model has nuisance parameters.
                 Eigen::Matrix<double, -1, 1> params(n_params);
                 params.head(n_nuisance) = theta_us_vec_ref;
                 params.tail(n_params_main) = theta_main_vec_ref;
                 
                 //// ---- full (log_prob, gradient) on the complete vector:
                 Eigen::Matrix<double, -1, 1> full_lp_grad(1 + n_params);
                 full_lp_grad = fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,
                                                              params, n_params_main, n_nuisance,
                                                              full_lp_grad);
                 
                 lp_and_grad_outs(0) = full_lp_grad(0);
                 
                 if (grad_option == "main_only") { 
                     
                     //// only the MAIN gradient block is copied back (nuisance slots stay 0)
                     lp_and_grad_outs.segment(1 + n_nuisance, n_params_main) = full_lp_grad.segment(1 + n_nuisance, n_params_main);
                   
                 } else if (grad_option == "us_only") {  // only nuisance 
                    
                     //// only the NUISANCE gradient block is copied back (main slots stay 0)
                     lp_and_grad_outs.segment(1, n_nuisance) = full_lp_grad.segment(1, n_nuisance);
                   
                 } else {  /// otherwise compute entire gradient
                   
                     lp_and_grad_outs.segment(1, n_params) = full_lp_grad.segment(1, n_params);
                   
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
                                                             const Stan_model_struct &Stan_model_as_cpp_struct,
                                                             std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                             const int n_threads_WCP
) {
 
     
      const int N = Model_args_as_cpp_struct.N;
      const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
      const int n_params_main = Model_args_as_cpp_struct.n_params_main;
      const int n_params = n_params_main + n_nuisance;
     
       Eigen::Matrix<double, -1, 1> lp_and_grad_outs =     Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
       
       
       fn_lp_grad_InPlace(        lp_and_grad_outs, 
                                  Model_type, 
                                  force_autodiff, force_PartialLog, multi_attempts,
                                  theta_main_vec_ref,
                                  theta_us_vec_ref, 
                                  y_ref,
                                  grad_option,
                                  Model_args_as_cpp_struct,
                                  Stan_model_as_cpp_struct,
                                  LC_MVP_ws_structs,
                                  n_threads_WCP);
       
       
       return  lp_and_grad_outs;
 
}


 
 
 
 
 
 
 
 
 
 
 
//   void  fn_lp_only_InPlace(      double &lp,
//                                        const std::string  &Model_type,
//                                        const bool force_autodiff,
//                                        const bool force_PartialLog,
//                                        const bool multi_attempts,
//                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
//                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
//                                        const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
//                                        const Model_fn_args_struct  &Model_args_as_cpp_struct,
//                                        const Stan_model_struct &Stan_model_as_cpp_struct) {
// 
//   const int N = Model_args_as_cpp_struct.N;
//   const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
//   int n_params_main = Model_args_as_cpp_struct.n_params_main;
//   int n_params = n_params_main + n_nuisance;
// 
//   const std::string  grad_option = "none";
//   
//   if  ((Model_type == "latent_trait") || (Model_type == "MVP") || (Model_type == "LC_MVP")) { //// use a built-in, fast manual-gradient model
//         
//           // Handle cases where autodiff is not forced
//           if (force_autodiff == false) {
//             
//                   if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
//                     
//                         if (force_PartialLog == true) {
//                           lp = (   fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD(theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct)  ).head(1).eval()(0);
//                         } else {
//                           lp =  (  fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD(theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
//                         }
//                         
//                   } else if (Model_type == "latent_trait") {
//                     //// #if COMPILE_LATENT_TRAIT
//                     
//                         if (force_PartialLog == true) {  
//                           lp = (   fn_lp_grad_LT_LC_PartialLog_MD_and_AD(theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct)  ).head(1).eval()(0);
//                           //// bookmark
//                           // std::cout << "ERROR: For the latent_trait model, the MANUAL-LOG-SCALE lp_grad function isn't yet working fully, so please set force_PartialLog = FALSE" << std::endl; 
//                         } else { 
//                           lp =  (  fn_lp_grad_LT_LC_NoLog_MD_and_AD(theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
//                         }
//                         //// #endif
//                         
//                   }
//           } else {  /// use autodiff 
//             
//             if ((Model_type == "LC_MVP") || (Model_type == "MVP")) {
//                     
//                     lp =  (  fn_lp_and_grad_MVP_Pinkney_AD_log_scale(theta_main_vec_ref, theta_us_vec_ref, y_ref,  grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
//                     
//                   } else if (Model_type == "latent_trait") {
//                     //// #if COMPILE_LATENT_TRAIT
//                     lp =  (  fn_lp_and_grad_LC_LT_AD_log_scale(theta_main_vec_ref, theta_us_vec_ref, y_ref,  grad_option, Model_args_as_cpp_struct) ).head(1).eval()(0);
//                     //// #endif
//                     
//                   } 
//             
//           }
// 
//   }  else if (Model_type == "Stan" )  { 
//     
//     
//               //// #if HAS_BRIDGESTAN_H
//               
//               Eigen::Matrix<double, -1, 1> params(n_params);
//               if (n_nuisance > 10) {
//                   params.head(n_nuisance) = theta_us_vec_ref;
//                   params.tail(n_params_main) = theta_main_vec_ref;
//               } else { 
//                   params = theta_main_vec_ref;
//                   n_params = n_params_main;
//                   params.resize(n_params_main);
//               }
//               
//               Eigen::Matrix<double, -1, 1> lp_and_grad_outs(1 + N + n_params);
//               
//               if (grad_option == "main_only") { 
//                 
//                 lp  =    fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,  params, n_params_main, n_nuisance, lp_and_grad_outs).head(1).eval()(0);
//                 
//               } else if (grad_option == "us_only") {  // only nuisance 
//                 
//                 lp  =  fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,   params, n_params_main, n_nuisance, lp_and_grad_outs).head(1).eval()(0);
//                 
//               } else {  /// otherwise compute entire gradient
//                 
//                 lp  =  fn_Stan_compute_log_prob_grad(Stan_model_as_cpp_struct,  params, n_params_main, n_nuisance, lp_and_grad_outs).head(1).eval()(0);
//                 
//               }
//               
//               //// #endif
//     
//   }
//   
// }


 
 
 
 
 
 
 
 
 
 
