
#pragma once

 

#include <Eigen/Dense>
 

 

 
using namespace Eigen;

 

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE

 
 
 
 
 
std::vector<double>                                   fn_find_initial_eps_main_and_us(      HMCResult &result_input,
                                                                                            const double seed,
                                                                                            const bool  burnin, 
                                                                                            const std::string &Model_type,
                                                                                            const bool  force_autodiff,
                                                                                            const bool  force_PartialLog,
                                                                                            const bool  multi_attempts,
                                                                                            const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                            const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                            EHMC_fn_args_struct  &EHMC_args_as_cpp_struct, /// pass by ref. to modify (???)
                                                                                            const EHMC_Metric_struct   &EHMC_Metric_struct_as_cpp_struct
) {
   
   stan::math::ChainableStack ad_tape;
   stan::math::nested_rev_autodiff nested;
   
   auto rng = dqrng::generator<dqrng::xoshiro256plusplus>(seed);
   
   const int N = Model_args_as_cpp_struct.N;
   const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
   const int n_params_main = Model_args_as_cpp_struct.n_params_main;
   const int n_params = n_params_main + n_nuisance;
   
   const std::string grad_option = "us_only";
   
   EHMC_args_as_cpp_struct.eps_us = 1.0;    /// starting value 
   EHMC_args_as_cpp_struct.tau_us = EHMC_args_as_cpp_struct.eps_us ; 
   
   EHMC_args_as_cpp_struct.eps_main = 1.0;    /// starting value 
   EHMC_args_as_cpp_struct.tau_main = EHMC_args_as_cpp_struct.eps_main ; 
   
   Stan_model_struct Stan_model_as_cpp_struct;
   
#if HAS_BRIDGESTAN_H
   if (Model_args_as_cpp_struct.model_so_file != "none") {
     
     Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(Model_args_as_cpp_struct.model_so_file, 
                                                            Model_args_as_cpp_struct.json_file_path, 
                                                            seed);
     
   }
#endif
   
   /// initial lp  
   double log_posterior = 0.0;
   //Eigen::Matrix<double, -1, 1>  lp_and_grad_outs = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
   fn_lp_grad_InPlace(   result_input.lp_and_grad_outs(), 
                         Model_type, 
                         force_autodiff, force_PartialLog, multi_attempts,
                         result_input.main_theta_vec(),  result_input.us_theta_vec(),
                         y_ref, grad_option,
                         Model_args_as_cpp_struct, //MVP_workspace, 
                         Stan_model_as_cpp_struct); 
   log_posterior =  result_input.lp_and_grad_outs()(0);
   
   
   for (int i = 0; i < 25; i++) { 
     
     log_posterior = 0.0; // reset 
     
     try {  
       
   
       ////  sample nuisance
       fn_Diffusion_HMC_nuisance_only_single_iter_InPlace_process(     result_input,  
                                                                       burnin,  rng, seed, 
                                                                       Model_type,  
                                                                       force_autodiff, force_PartialLog, multi_attempts, 
                                                                       y_ref,
                                                                       Model_args_as_cpp_struct,//  MVP_workspace, 
                                                                       EHMC_args_as_cpp_struct, 
                                                                       EHMC_Metric_struct_as_cpp_struct, 
                                                                       Stan_model_as_cpp_struct); 
       
       
       ////  sample main (cond. on nuisance)
       fn_Diffusion_HMC_nuisance_only_single_iter_InPlace_process(     result_input,  
                                                                       burnin,  rng, seed, 
                                                                       Model_type,  
                                                                       force_autodiff, force_PartialLog, multi_attempts, 
                                                                       y_ref,
                                                                       Model_args_as_cpp_struct,  //MVP_workspace, 
                                                                       EHMC_args_as_cpp_struct, 
                                                                       EHMC_Metric_struct_as_cpp_struct, 
                                                                       Stan_model_as_cpp_struct); 
       
       
       
 
       log_posterior =  result_input.lp_and_grad_outs()(0); // update lp
       
       
       if    ( (result_input.us_div() == 0) && (result_input.main_div() == 0) )  { // if no divs
         
         if (result_input.us_p_jump() < 0.80) {
           EHMC_args_as_cpp_struct.eps_us = 0.5 * EHMC_args_as_cpp_struct.eps_us;
           EHMC_args_as_cpp_struct.tau_us =       EHMC_args_as_cpp_struct.eps_us ;
         }
         
         if (result_input.main_p_jump() < 0.80) {
           EHMC_args_as_cpp_struct.eps_main = 0.5 * EHMC_args_as_cpp_struct.eps_main;
           EHMC_args_as_cpp_struct.tau_main =       EHMC_args_as_cpp_struct.eps_main ;
         } 
         
       } else {
         
         EHMC_args_as_cpp_struct.eps_us  =  0.5 * EHMC_args_as_cpp_struct.eps_us;
         EHMC_args_as_cpp_struct.tau_us =       EHMC_args_as_cpp_struct.eps_us ;
         
         EHMC_args_as_cpp_struct.eps_main  =  0.5 * EHMC_args_as_cpp_struct.eps_main;
         EHMC_args_as_cpp_struct.tau_main =       EHMC_args_as_cpp_struct.eps_main ;
         
       }
       
       
     } catch (...) { 
       
       EHMC_args_as_cpp_struct.eps_us  =  0.5 * EHMC_args_as_cpp_struct.eps_us;
       EHMC_args_as_cpp_struct.tau_us =       EHMC_args_as_cpp_struct.eps_us ;
       
       EHMC_args_as_cpp_struct.eps_main  =  0.5 * EHMC_args_as_cpp_struct.eps_main;
       EHMC_args_as_cpp_struct.tau_main =       EHMC_args_as_cpp_struct.eps_main ;
       
     }
     
     
   }
   
#if HAS_BRIDGESTAN_H
   // destroy Stan model object
   if (Model_args_as_cpp_struct.model_so_file != "none") {
     fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
   }
#endif
   
   
   
   std::vector<double>  outs(2);
   
   outs[0] = EHMC_args_as_cpp_struct.eps_us;
   outs[1] = EHMC_args_as_cpp_struct.eps_main;
   
   return outs;
   
 }
 
 
 
 
 
 
 
// 
// 
// double                                   fn_find_initial_eps_main(                HMCResult &result_input,
//                                                                                   const double seed,
//                                                                                   const bool  burnin, 
//                                                                                   const std::string &Model_type,
//                                                                                   const bool  force_autodiff,
//                                                                                   const bool  force_PartialLog,
//                                                                                   const bool  multi_attempts,
//                                                                                   const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
//                                                                                   const Model_fn_args_struct &Model_args_as_cpp_struct,
//                                                                                   EHMC_fn_args_struct  &EHMC_args_as_cpp_struct, /// pass by ref. to modify (???)
//                                                                                   const EHMC_Metric_struct   &EHMC_Metric_struct_as_cpp_struct
//  ) {
//    
//      
//       stan::math::ChainableStack ad_tape;
//       stan::math::nested_rev_autodiff nested;
//       
//      // auto rng = RNGManager::get_thread_local_rng(seed);
//   
//       std::mt19937 rng(static_cast<unsigned int>(seed));
//      
//       const int N = Model_args_as_cpp_struct.N;
//       const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
//       const int n_params_main = Model_args_as_cpp_struct.n_params_main;
//       const int n_params = n_params_main + n_nuisance;
//       
//       const std::string grad_option = "main_only";
//      
//      EHMC_args_as_cpp_struct.eps_main = 1.0;  /// starting value 
//      EHMC_args_as_cpp_struct.tau_main =  EHMC_args_as_cpp_struct.eps_main; 
//      
//      Stan_model_struct Stan_model_as_cpp_struct;
//      
// #if HAS_BRIDGESTAN_H
//      if (Model_args_as_cpp_struct.model_so_file != "none") {
//        
//        Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(Model_args_as_cpp_struct.model_so_file, 
//                                                               Model_args_as_cpp_struct.json_file_path, 
//                                                               seed);
//        
//      }
// #endif
//      
//      /// initial lp  
//      double log_posterior = 0.0;
//     // Eigen::Matrix<double, -1, 1>  lp_and_grad_outs = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
//      fn_lp_grad_InPlace(  result_input.lp_and_grad_outs,
//                           Model_type, 
//                           force_autodiff , force_PartialLog , multi_attempts,
//                           result_input.main_theta_vec,  result_input.us_theta_vec, y_ref, grad_option, 
//                           Model_args_as_cpp_struct, Stan_model_as_cpp_struct); 
//      log_posterior =  result_input.lp_and_grad_outs(0);
//  
//      for (int i = 0; i < 25; i++) { 
//        
//        log_posterior = 0.0; // reset 
//               
//                try {  
//                  
//                    stan::math::start_nested();
//                    fn_standard_HMC_main_only_single_iter_InPlace_process(            result_input, 
//                                                                                      burnin,  rng, seed, 
//                                                                                      Model_type,  
//                                                                                      force_autodiff, force_PartialLog, multi_attempts,
//                                                                                      y_ref,
//                                                                                      Model_args_as_cpp_struct,  EHMC_args_as_cpp_struct, EHMC_Metric_struct_as_cpp_struct, 
//                                                                                      Stan_model_as_cpp_struct); 
//                    stan::math::recover_memory_nested();
//                    log_posterior =  result_input.lp_and_grad_outs(0); // update lp
//                
//                    
//                      if     (result_input.main_div == 0)   { // if no div 
//                          
//                                if (result_input.main_p_jump < 0.80) {
//                                    EHMC_args_as_cpp_struct.eps_main = 0.5 * EHMC_args_as_cpp_struct.eps_main;
//                                    EHMC_args_as_cpp_struct.tau_main = EHMC_args_as_cpp_struct.eps_main; 
//                                } else { 
//                                    /// leave eps as it is 
//                                }
//                                
//                      } else { 
//                                 EHMC_args_as_cpp_struct.eps_main  =  0.5 * EHMC_args_as_cpp_struct.eps_main;
//                                 EHMC_args_as_cpp_struct.tau_main = EHMC_args_as_cpp_struct.eps_main; 
//                      }
//                          
//                } catch (...) { 
//                  
//                  EHMC_args_as_cpp_struct.eps_main  =  0.5 * EHMC_args_as_cpp_struct.eps_main;
//                  EHMC_args_as_cpp_struct.tau_main = EHMC_args_as_cpp_struct.eps_main; 
//                  
//                }
//            
//      }
//      
// #if HAS_BRIDGESTAN_H
//      // destroy Stan model object
//      if (Model_args_as_cpp_struct.model_so_file != "none") {
//        fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
//      }
// #endif
//  
//  return EHMC_args_as_cpp_struct.eps_main ;
//  
// }
//  
//  
//  
//  
//  
//  
//  
//  
// 
// double                                   fn_find_initial_eps_us(                 HMCResult &result_input,
//                                                                                  const double seed,
//                                                                                  const bool  burnin, 
//                                                                                  const std::string &Model_type,
//                                                                                  const bool  force_autodiff,
//                                                                                  const bool  force_PartialLog,
//                                                                                  const bool  multi_attempts,
//                                                                                  const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
//                                                                                  const Model_fn_args_struct &Model_args_as_cpp_struct,
//                                                                                  EHMC_fn_args_struct  &EHMC_args_as_cpp_struct, /// pass by ref. to modify (???)
//                                                                                  const EHMC_Metric_struct   &EHMC_Metric_struct_as_cpp_struct
// ) {
//   
//        stan::math::ChainableStack ad_tape;
//        stan::math::nested_rev_autodiff nested;
//       
//      // auto rng = RNGManager::get_thread_local_rng(seed);
//       std::mt19937 rng(static_cast<unsigned int>(seed));
//   
//       const int N = Model_args_as_cpp_struct.N;
//       const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
//       const int n_params_main = Model_args_as_cpp_struct.n_params_main;
//       const int n_params = n_params_main + n_nuisance;
//       
//       const std::string grad_option = "us_only";
//       
//       EHMC_args_as_cpp_struct.eps_us = 1.0;    /// starting value 
//       EHMC_args_as_cpp_struct.tau_us = EHMC_args_as_cpp_struct.eps_us ; 
//       
//       Stan_model_struct Stan_model_as_cpp_struct;
//       
// #if HAS_BRIDGESTAN_H
//       if (Model_args_as_cpp_struct.model_so_file != "none") {
//         
//         Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(Model_args_as_cpp_struct.model_so_file, 
//                                                                Model_args_as_cpp_struct.json_file_path, 
//                                                                seed);
//         
//       }
// #endif
//     
//       /// initial lp  
//       double log_posterior = 0.0;
//       //Eigen::Matrix<double, -1, 1>  lp_and_grad_outs = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);
//       fn_lp_grad_InPlace(   result_input.lp_and_grad_outs, 
//                             Model_type, 
//                             force_autodiff, force_PartialLog, multi_attempts,
//                             result_input.main_theta_vec,  result_input.us_theta_vec, y_ref, grad_option,
//                             Model_args_as_cpp_struct, Stan_model_as_cpp_struct); 
//       log_posterior =  result_input.lp_and_grad_outs(0);
//      
//      
//      for (int i = 0; i < 25; i++) { 
//        
//            log_posterior = 0.0; // reset 
//        
//              try {  
//                      
//                      stan::math::start_nested();
//                      fn_Diffusion_HMC_nuisance_only_single_iter_InPlace_process(     result_input,  
//                                                                                      burnin,  rng, seed, 
//                                                                                      Model_type,  
//                                                                                      force_autodiff, force_PartialLog, multi_attempts, 
//                                                                                      y_ref,
//                                                                                      Model_args_as_cpp_struct,  EHMC_args_as_cpp_struct, EHMC_Metric_struct_as_cpp_struct, 
//                                                                                      Stan_model_as_cpp_struct); 
//                      stan::math::recover_memory_nested();
//                      log_posterior =  result_input.lp_and_grad_outs(0); // update lp
//                     
//                      
//                      if     (result_input.us_div == 0)   { // if no div 
//                        
//                        if (result_input.us_p_jump < 0.80) {
//                          EHMC_args_as_cpp_struct.eps_us = 0.5 * EHMC_args_as_cpp_struct.eps_us;
//                          EHMC_args_as_cpp_struct.tau_us =       EHMC_args_as_cpp_struct.eps_us ;
//                        }
//                        
//                      } else {
//                        
//                        EHMC_args_as_cpp_struct.eps_us  =  0.5 * EHMC_args_as_cpp_struct.eps_us;
//                        EHMC_args_as_cpp_struct.tau_us =       EHMC_args_as_cpp_struct.eps_us ;
//                        
//                      }
//                      
//                     
//              } catch (...) { 
//                
//                EHMC_args_as_cpp_struct.eps_us  =  0.5 * EHMC_args_as_cpp_struct.eps_us;
//                EHMC_args_as_cpp_struct.tau_us =       EHMC_args_as_cpp_struct.eps_us ;
//                
//              }
//              
//          
//        }
//      
// #if HAS_BRIDGESTAN_H
//      // destroy Stan model object
//      if (Model_args_as_cpp_struct.model_so_file != "none") {
//        fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
//      }
// #endif
//            
//  
//  
//  return EHMC_args_as_cpp_struct.eps_us ;
//  
// }
// 
// 
//  
//  
//  
//  
 
 
 
 
 
 