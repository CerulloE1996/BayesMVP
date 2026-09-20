
#pragma once

 

#include <Eigen/Dense>
 

 

 
using namespace Eigen;

 

 
 
 
 
std::vector<double>                                   fn_find_initial_eps_main_and_us(      HMCResult &result_input,
                                                                                            const bool partitioned_HMC,
                                                                                            const int seed,
                                                                                            const bool  burnin, 
                                                                                            const std::string &Model_type,
                                                                                            const bool  force_autodiff,
                                                                                            const bool  force_PartialLog,
                                                                                            const bool  multi_attempts,
                                                                                            const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                            const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                            EHMC_fn_args_struct  &EHMC_args_as_cpp_struct,
                                                                                            const EHMC_Metric_struct   &EHMC_Metric_struct_as_cpp_struct
) {
   
   stan::math::ChainableStack ad_tape;
   // stan::math::nested_rev_autodiff nested;
     
   // #if RNG_TYPE_CPP_STD == 1
   //      std::mt19937 rng_i(seed);
   // #elif RNG_TYPE_pcg32 == 1
   //      pcg32 rng_i(seed, 1);
   // #elif RNG_TYPE_pcg64 == 1
   //      pcg64 rng_i(seed, 1);
   // #endif
   
   // // #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
   // //        dqrng::xoshiro256plus rng_i(seed);
   // // #elif RNG_TYPE_dqrng_pcg64 == 1
   //        RNG_TYPE_dqrng rng_main_i = dqrng::generator<pcg64>(seed, 1);
   //        RNG_TYPE_dqrng rng_nuisance_i = dqrng::generator<pcg64>(seed + 1e6, 1);
   // // #endif
   
   #if RNG_TYPE_CPP_STD == 1
       std::mt19937 rng_main_i(seed);
       std::mt19937 rng_nuisance_i(seed + 1e6);
   #elif RNG_TYPE_dqrng_xoshiro256plusplus == 1
       dqrng::xoshiro256plus rng_main_i(seed);
       dqrng::xoshiro256plus rng_nuisance_i(seed + 1e6);
   #endif
   
   const int N = Model_args_as_cpp_struct.N;
   const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
   const int n_params_main = Model_args_as_cpp_struct.n_params_main;
   const int n_params = n_params_main + n_nuisance;
   
   const std::string grad_option = "all";
   
   EHMC_args_as_cpp_struct.eps_us = 0.50;    /// starting value 
   EHMC_args_as_cpp_struct.tau_us = EHMC_args_as_cpp_struct.eps_us ; 
   
   EHMC_args_as_cpp_struct.eps_main = 0.50;    /// starting value 
   EHMC_args_as_cpp_struct.tau_main = EHMC_args_as_cpp_struct.eps_main ; 
   
   LC_MVP_workspace_struct LC_MVP_ws_struct; 
   Stan_model_struct Stan_model_as_cpp_struct; 
   
   if ((Model_type == "LC_MVP") || (Model_type == "LC_MVOP")) {
     
             const int n_tests = y_ref.cols();
             ////
             const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
             const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
             ////
             const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
             const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
             
             const int n_corrs = n_class * n_tests * (n_tests - 1) / 2;
             
             int n_covariates_total, n_covariates_max;
             if (n_class > 1) {
               const int n_covariates_total_nd = n_covariates_per_outcome_vec.row(0).sum();
               const int n_covariates_total_d  = n_covariates_per_outcome_vec.row(1).sum();
               n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
               const int n_covariates_max_nd = n_covariates_per_outcome_vec.row(0).maxCoeff();
               const int n_covariates_max_d  = n_covariates_per_outcome_vec.row(1).maxCoeff();
               n_covariates_max = std::max(n_covariates_max_nd, n_covariates_max_d);
             } else {
               n_covariates_total = n_covariates_per_outcome_vec.sum();
               n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();
             }
             
             int vec_size;
             if      (vect_type == "AVX512") vec_size = 8;
             else if (vect_type == "AVX2")   vec_size = 4;
             else if (vect_type == "AVX")    vec_size = 2;
             else                            vec_size = 1;
             
             ChunkSizeInfo chunk_info = calculate_chunk_sizes( N, 
                                                               vec_size, 
                                                               n_chunks);
             
             LC_MVP_ws_struct.allocate( chunk_info.normal_chunk_size,
                                        n_tests,
                                        n_class,
                                        n_covariates_max,
                                        n_corrs,
                                        n_covariates_total);
             // int padded_chunk = ((chunk_info.normal_chunk_size + vec_size - 1) / vec_size) * vec_size;
             // LC_MVP_ws_struct.allocate(  padded_chunk,
             //                             n_tests,
             //                             n_class,
             //                             n_covariates_max,
             //                             n_corrs,
             //                             n_covariates_total);
             
             std::cout << "find_initial_eps: Pre-allocated lp_grad workspace ("
                       << "chunk_size=" << chunk_info.normal_chunk_size 
                       << ", n_tests=" << n_tests << ")" << std::endl;
     
   } // else: LC_MVP_ws_struct[i].is_allocated stays false (dummy)
   
   if (g_debug_cutpoint_grads) std::cerr << "Point A" << std::endl << std::flush;
   
   if (Model_args_as_cpp_struct.model_so_file != "none") {
     
         Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(Model_args_as_cpp_struct.model_so_file, 
                                                                Model_args_as_cpp_struct.json_file_path, 
                                                                seed);
     
   }
   
   if (g_debug_cutpoint_grads) std::cerr << "Point B" << std::endl << std::flush;
   
   std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs;
   LC_MVP_ws_structs.resize(1);
   LC_MVP_ws_structs[0] = LC_MVP_ws_struct;
   
   if (g_debug_cutpoint_grads) std::cerr << "Point C" << std::endl << std::flush;
   
   fn_lp_grad_InPlace(   result_input.lp_and_grad_outs(), 
                         Model_type, 
                         force_autodiff, force_PartialLog, multi_attempts,
                         result_input.main_theta_vec(),  
                         result_input.us_theta_vec(),
                         y_ref, 
                         grad_option,
                         Model_args_as_cpp_struct,
                         Stan_model_as_cpp_struct, 
                         LC_MVP_ws_structs,
                         1);
   
   double log_posterior = result_input.lp_and_grad_outs()(0);

   if (!std::isfinite(log_posterior)) {
    throw std::runtime_error(
      "Initial values give a non-finite log-posterior (" + std::to_string(log_posterior) +
      "). For LC models this is almost always the identification constraint "
      "beta_d(test 1) > beta_nd(test 1) being violated by the inits.");
   }
   
   if (g_debug_cutpoint_grads) std::cerr << "Point D" << std::endl << std::flush;
   
   // //// initial lp
   // double log_posterior =  result_input.lp_and_grad_outs()(0);
   
    if (g_debug_cutpoint_grads) std::cerr << "Point E" << std::endl << std::flush;
    // The current state was evaluated above. The joint wrapper preserves its
    // evaluation after acceptance/rejection throughout this step-size search.
    bool current_lp_grad_valid = true;
   
   for (int i = 0; i < 25; i++) { 
     
     if (g_debug_cutpoint_grads) std::cerr << "LOOP iter " << i << " start" << std::endl << std::flush;
     
     log_posterior = 0.0; // reset 
     
     try {  
       
       if (partitioned_HMC == true) {
                   
                   if (g_debug_cutpoint_grads) std::cerr << "LOOP iter " << i << " before nuisance HMC" << std::endl << std::flush;
         
                   ////  sample nuisance
                   stan::math::start_nested();
                   fn_Diffusion_HMC_nuisance_only_single_iter_InPlace_process(     result_input,  
                                                                                   rng_nuisance_i,
                                                                                   Model_type,  
                                                                                   force_autodiff, force_PartialLog, multi_attempts, 
                                                                                   y_ref,
                                                                                   Model_args_as_cpp_struct,
                                                                                   EHMC_args_as_cpp_struct, 
                                                                                   EHMC_Metric_struct_as_cpp_struct, 
                                                                                   Stan_model_as_cpp_struct, 
                                                                                   LC_MVP_ws_structs,
                                                                                   1); 
                   stan::math::recover_memory_nested(); 
                   
                   if (g_debug_cutpoint_grads) std::cerr << "LOOP iter " << i << " before main HMC" << std::endl << std::flush;
                   
                   ////  sample main (cond. on nuisance)
                   stan::math::start_nested();
                   fn_standard_HMC_main_only_single_iter_InPlace_process(          result_input,  
                                                                                   rng_main_i,
                                                                                   Model_type,  
                                                                                   force_autodiff, force_PartialLog, multi_attempts, 
                                                                                   y_ref,
                                                                                   Model_args_as_cpp_struct,
                                                                                   EHMC_args_as_cpp_struct, 
                                                                                   EHMC_Metric_struct_as_cpp_struct, 
                                                                                   Stan_model_as_cpp_struct, 
                                                                                   LC_MVP_ws_structs,
                                                                                   1); 
                   stan::math::recover_memory_nested(); 
       
       } else { 
         
                   if (g_debug_cutpoint_grads) std::cerr << "LOOP iter " << i << " before dual HMC" << std::endl << std::flush;
             
                   //// Sample all params (standard HMC)
                   stan::math::start_nested();
                    fn_diffusion_HMC_dual_single_iter_InPlace_process( result_input,
                                                                     rng_main_i,
                                                                     rng_nuisance_i,
                                                                     Model_type,  
                                                                     force_autodiff, force_PartialLog, multi_attempts, 
                                                                     y_ref,
                                                                     Model_args_as_cpp_struct,
                                                                     EHMC_args_as_cpp_struct, 
                                                                     EHMC_Metric_struct_as_cpp_struct, 
                                                                     Stan_model_as_cpp_struct,
                                                                     LC_MVP_ws_structs,
                                                                      1,
                                                                      &current_lp_grad_valid);
                   stan::math::recover_memory_nested();
                   
                   if (g_debug_cutpoint_grads) std::cerr << "LOOP iter " << i << " after dual HMC" << std::endl << std::flush;
             
       }
 
       log_posterior =  result_input.lp_and_grad_outs()(0); // update lp
       
       if (partitioned_HMC == true) {
               
               ////
               if    ( (result_input.us_div() == 0) && (result_input.main_div() == 0) )  { // if no divs
                 
                     if (result_input.us_p_jump() < 0.80) {
                           EHMC_args_as_cpp_struct.eps_us = 0.5 * EHMC_args_as_cpp_struct.eps_us;
                           EHMC_args_as_cpp_struct.tau_us =       EHMC_args_as_cpp_struct.eps_us;
                     }
                     
                     if (result_input.main_p_jump() < 0.80) {
                           EHMC_args_as_cpp_struct.eps_main = 0.5 * EHMC_args_as_cpp_struct.eps_main;
                           EHMC_args_as_cpp_struct.tau_main =       EHMC_args_as_cpp_struct.eps_main;
                     } 
                 
               } else {
                     
                     EHMC_args_as_cpp_struct.eps_us  =  0.5 * EHMC_args_as_cpp_struct.eps_us;
                     EHMC_args_as_cpp_struct.tau_us =       EHMC_args_as_cpp_struct.eps_us;
                     
                     EHMC_args_as_cpp_struct.eps_main  =  0.5 * EHMC_args_as_cpp_struct.eps_main;
                     EHMC_args_as_cpp_struct.tau_main =       EHMC_args_as_cpp_struct.eps_main;
                 
               }
               ////
       
       } else if (partitioned_HMC == false) {
               
               ////
               if    (result_input.main_div() == 0) { // if no divs
                 
                     if (result_input.main_p_jump() < 0.80) {
                         EHMC_args_as_cpp_struct.eps_main = 0.5 * EHMC_args_as_cpp_struct.eps_main;
                         EHMC_args_as_cpp_struct.tau_main =       EHMC_args_as_cpp_struct.eps_main;
                     }
                 
               } else {
                     
                     EHMC_args_as_cpp_struct.eps_main  =  0.5 * EHMC_args_as_cpp_struct.eps_main;
                     EHMC_args_as_cpp_struct.tau_main =       EHMC_args_as_cpp_struct.eps_main;
                 
               }
               ////
         
       }
       
       
     } catch (...) { 
       
             EHMC_args_as_cpp_struct.eps_us  =  0.5 * EHMC_args_as_cpp_struct.eps_us;
             EHMC_args_as_cpp_struct.tau_us =       EHMC_args_as_cpp_struct.eps_us;
             
             EHMC_args_as_cpp_struct.eps_main  =  0.5 * EHMC_args_as_cpp_struct.eps_main;
             EHMC_args_as_cpp_struct.tau_main =       EHMC_args_as_cpp_struct.eps_main;
       
     }
     
   }
   
   if (g_debug_cutpoint_grads) std::cerr << "Point F" << std::endl << std::flush;
   
   // destroy Stan model object
   if (Model_args_as_cpp_struct.model_so_file != "none") {
     fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
   }
   
   if (g_debug_cutpoint_grads) std::cerr << "Point G" << std::endl << std::flush;
   
   std::vector<double>  outs(2);
   
   outs[0] = EHMC_args_as_cpp_struct.eps_main;
   outs[1] = EHMC_args_as_cpp_struct.eps_us;
   
   if (g_debug_cutpoint_grads) std::cerr << "Point H" << std::endl << std::flush;
   
   return outs;
   
 }
 
 
 
 
 

 
 
 
 
 

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
 
 
 
 
 
 
