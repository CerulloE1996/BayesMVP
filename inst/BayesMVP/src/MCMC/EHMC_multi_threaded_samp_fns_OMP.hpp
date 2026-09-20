#pragma once

 
 

 

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <tbb/concurrent_vector.h>

 
 
#include <chrono> 
#include <unordered_map>
#include <memory>
#include <thread>
#include <functional>

 
 
//// ANSI codes for different colors
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"

 
using namespace Rcpp;
using namespace Eigen;
 
 
 

     
     
     

     
     
     
     
     
     
     



//// --------------------------------- OpenMP  functions  -- SAMPLING fn ------------------------------------------------------------------------------------------------------------------------------------------- 



 

//// OpenMP  - SAMPLING
void EHMC_sampling_OpenMP(    const int  &n_threads,
                              const uint64_t  &global_seed,
                              const int  &n_iter,
                              const bool &partitioned_HMC,
                              const bool &diffusion_HMC,
                              const std::string &Model_type,
                              const bool &sample_nuisance,
                              const bool &force_autodiff,
                              const bool &force_PartialLog,
                              const bool &multi_attempts,
                              //// inputs
                              const Eigen::Matrix<double, -1, -1> &theta_main_vectors_all_chains_input_from_R_RcppPar,
                              const Eigen::Matrix<double, -1, -1> &theta_us_vectors_all_chains_input_from_R_RcppPar,
                              //// data
                              const std::vector<Eigen::Matrix<int, -1, -1>>   &y_copies,
                              //////////////  input structs
                              const std::vector<Model_fn_args_struct> &Model_args_as_cpp_struct_copies,
                              std::vector<EHMC_fn_args_struct> &EHMC_args_as_cpp_struct_copies,
                              const std::vector<EHMC_Metric_struct> &EHMC_Metric_as_cpp_struct_copies,
                              //// -------------- For POST-BURNIN only:
                              const int &n_nuisance_to_track,
                              std::vector<Eigen::Matrix<double, -1, -1>> &trace_output,
                              std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_divs,
                              std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_nuisance,
                              std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_log_lik,
                              Eigen::Matrix<double, -1, -1> &p_jump_main_output,
                              Eigen::Matrix<double, -1, -1> &p_jump_nuisance_output,
                              const int n_threads_WCP
) {
        
        const int N = Model_args_as_cpp_struct_copies[0].N;
        const int n_us =  Model_args_as_cpp_struct_copies[0].n_nuisance;
        const int n_params_main = Model_args_as_cpp_struct_copies[0].n_params_main;
        
        //// local storage:
        std::vector<HMC_output_single_chain> HMC_outputs;
        std::vector<HMCResult> HMC_inputs;
        ////
        std::vector<LC_MVP_workspace_struct> LC_MVP_ws_struct;
        // Reserve space:
        HMC_outputs.reserve(n_threads);
        HMC_inputs.reserve(n_threads);
        // Construct the structs:
        for (int i = 0; i < n_threads; ++i) {
          HMC_outputs.emplace_back(n_iter, n_nuisance_to_track, n_params_main, n_us, N);  // Construct directly in vector
          HMC_inputs.emplace_back(n_params_main, n_us, N);  // Construct directly in vector
        }
        
        // // Pre-allocate LC-MVP model lp_grad workspaces (one per thread):
        // // LC_MVP_ws_struct.resize(n_threads);
        // LC_MVP_ws_struct.reserve(n_threads);
        
        std::vector<std::vector<LC_MVP_workspace_struct>> LC_MVP_ws_structs_double_vec;
        LC_MVP_ws_structs_double_vec.resize(n_threads);
        
        if ((Model_type == "LC_MVP") || (Model_type == "LC_MVOP")) {
                
                const int n_tests = y_copies[0].cols();
                const int n_class = Model_args_as_cpp_struct_copies[0].Model_args_ints(1);
                const int n_chunks = Model_args_as_cpp_struct_copies[0].Model_args_ints(3);
                const std::string &vect_type = Model_args_as_cpp_struct_copies[0].Model_args_strings(0);
                const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct_copies[0].Model_args_mats_int[0];
                
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
                
                ChunkSizeInfo chunk_info = calculate_chunk_sizes(N, vec_size, n_chunks);
                
                for (int i = 0; i < n_threads; ++i) {
                  
                    LC_MVP_ws_structs_double_vec[i].reserve(n_threads_WCP);
                    
                    for (int t = 0; t < n_threads_WCP; ++t) {
                      
                      LC_MVP_ws_structs_double_vec[i].emplace_back(  chunk_info.normal_chunk_size,
                                                                     n_tests,
                                                                     n_class,
                                                                     n_covariates_max,
                                                                     n_corrs,
                                                                     n_covariates_total);
                      // int padded_chunk = ((chunk_info.normal_chunk_size + vec_size - 1) / vec_size) * vec_size;
                      // LC_MVP_ws_structs_double_vec[i].emplace_back(  padded_chunk, 
                      //                                                n_tests, 
                      //                                                n_class,
                      //                                                n_covariates_max,
                      //                                                n_corrs, 
                      //                                                n_covariates_total);
                    }
                  
                }
                
                std::cout << "Pre-allocated " << n_threads << " lp_grad workspaces ("
                          << "chunk_size=" << chunk_info.normal_chunk_size 
                          << ", n_tests=" << n_tests << ")" << std::endl;
          
        } // else: LC_MVP_ws_struct[i].is_allocated stays false (dummy)
        
        
        const int global_seed_main_int =      static_cast<int>(global_seed);
        const int global_seed_nuisance_int =  static_cast<int>(global_seed + 1e6);
          
        // #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
        //   dqrng::xoshiro256plus global_rng_main(global_seed_main_int);
        //   dqrng::xoshiro256plus global_rng_nuisance(global_seed_nuisance_int);
        // #endif
          
        omp_set_num_threads(n_threads);
        omp_set_dynamic(0);
        omp_set_max_active_levels(2);
        
        printf("Requested threads: %d\n", n_threads);
        printf("Max threads available: %d\n", omp_get_max_threads());
        
        //// parallel for-loop
        #pragma omp parallel for shared(HMC_outputs, HMC_inputs, y_copies, Model_args_as_cpp_struct_copies, EHMC_args_as_cpp_struct_copies, EHMC_Metric_as_cpp_struct_copies, theta_main_vectors_all_chains_input_from_R_RcppPar, theta_us_vectors_all_chains_input_from_R_RcppPar)
        for (int i = 0; i < n_threads; ++i) {  
                
                const int chain_id_int = static_cast<int>(i);
                const int seed_main_int_i =     global_seed_main_int +     n_iter*(1 + chain_id_int);
                const int seed_nuisance_int_i = global_seed_nuisance_int + n_iter*(1 + chain_id_int);
               
                #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
                        dqrng::xoshiro256plus rng_main_i; //(global_rng_main);      // make thread local copy of rng 
                        dqrng::xoshiro256plus rng_nuisance_i; //(global_rng_nuisance);      // make thread local copy of rng 
                        rng_main_i.seed(seed_main_int_i);  // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                        rng_nuisance_i.seed(seed_nuisance_int_i);  // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                #elif RNG_TYPE_CPP_STD == 1
                        std::mt19937 rng_main_i;  // Fresh RNG // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                        std::mt19937 rng_nuisance_i;  // Fresh RNG // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                        rng_main_i.seed(seed_main_int_i); // set / re-set the seed
                        rng_nuisance_i.seed(seed_nuisance_int_i); // set / re-set the seed
                #endif
                
                const int N = Model_args_as_cpp_struct_copies[i].N;
                const int n_us =  Model_args_as_cpp_struct_copies[i].n_nuisance;
                const int n_params_main = Model_args_as_cpp_struct_copies[i].n_params_main;
                
                stan::math::ChainableStack ad_tape;     // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                //// stan::math::nested_rev_autodiff nested; // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                
                const bool burnin_indicator = false;
                const int current_iter = 0; // gets assigned later for post-burnin
              
                ///////////////////////////////////////// perform iterations for adaptation interval
                HMC_inputs[i].main_theta_vec() =   theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
                HMC_inputs[i].main_theta_vec_0() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
                HMC_inputs[i].us_theta_vec() =     theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
                HMC_inputs[i].us_theta_vec_0() =   theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
              
                //////////////////////////////// perform iterations for chain i
                if (Model_type == "Stan") {  
                  
                        Stan_model_struct  Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(     Model_args_as_cpp_struct_copies[i].model_so_file,
                                                                                                       Model_args_as_cpp_struct_copies[i].json_file_path,
                                                                                                       seed_main_int_i);
                        
                        // //////////////////////////////// perform iterations for chain i:
                        // fn_sample_HMC_multi_iter_single_thread(   HMC_outputs[i] ,
                        //                                           HMC_inputs[i], 
                        //                                           burnin_indicator, 
                        //                                           chain_id_int, 
                        //                                           current_iter,
                        //                                           seed_main_int_i, 
                        //                                           seed_nuisance_int_i,
                        //                                           rng_main_i,
                        //                                           rng_nuisance_i,
                        //                                           n_iter,
                        //                                           partitioned_HMC,
                        //                                           diffusion_HMC,
                        //                                           Model_type,  sample_nuisance,
                        //                                           force_autodiff, force_PartialLog,  multi_attempts,  
                        //                                           n_nuisance_to_track, 
                        //                                           y_copies[i],
                        //                                           Model_args_as_cpp_struct_copies[i],
                        //                                           EHMC_args_as_cpp_struct_copies[i],
                        //                                           EHMC_Metric_as_cpp_struct_copies[i],
                        //                                           Stan_model_as_cpp_struct);
                        
                        
                        //////////////////////////////// perform iterations for chain i:
                        fn_sample_HMC_multi_iter_single_thread(  HMC_outputs[i],   
                                                                 HMC_inputs[i], 
                                                                 burnin_indicator, 
                                                                 chain_id_int, 
                                                                 current_iter,
                                                                 seed_main_int_i, 
                                                                 seed_nuisance_int_i,
                                                                 rng_main_i,
                                                                 rng_nuisance_i,
                                                                 n_iter,
                                                                 partitioned_HMC,
                                                                 diffusion_HMC,
                                                                 Model_type,  sample_nuisance, 
                                                                 force_autodiff, force_PartialLog,  multi_attempts,  
                                                                 n_nuisance_to_track, 
                                                                 y_copies[i],
                                                                 Model_args_as_cpp_struct_copies[i],
                                                                 EHMC_args_as_cpp_struct_copies[i], 
                                                                 EHMC_Metric_as_cpp_struct_copies[i],
                                                                 Stan_model_as_cpp_struct, 
                                                                 LC_MVP_ws_structs_double_vec[i],
                                                                 n_threads_WCP);
                        
                        
                        ////////////////////////////// end of iteration(s)
                        
                        //// destroy Stan model object:
                        fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);  
                  
                } else { 
                        
                        Stan_model_struct Stan_model_as_cpp_struct; ///  dummy struct
                        
                        //////////////////////////////// perform iterations for chain i:
                        fn_sample_HMC_multi_iter_single_thread(  HMC_outputs[i],
                                                                 HMC_inputs[i],
                                                                 burnin_indicator,
                                                                 chain_id_int,
                                                                 current_iter,
                                                                 seed_main_int_i,
                                                                 seed_nuisance_int_i,
                                                                 rng_main_i,
                                                                 rng_nuisance_i,
                                                                 n_iter,
                                                                 partitioned_HMC,
                                                                 diffusion_HMC,
                                                                 Model_type,  sample_nuisance,
                                                                 force_autodiff, force_PartialLog,  multi_attempts,
                                                                 n_nuisance_to_track, 
                                                                 y_copies[i],
                                                                 Model_args_as_cpp_struct_copies[i],
                                                                 EHMC_args_as_cpp_struct_copies[i],
                                                                 EHMC_Metric_as_cpp_struct_copies[i],
                                                                 Stan_model_as_cpp_struct,
                                                                 LC_MVP_ws_structs_double_vec[i],
                                                                 n_threads_WCP);
                        ////////////////////////////// end of iteration(s)
                  
                }
          
        }  //// end of parallel OpenMP loop
        
        #pragma omp barrier  // Make sure all threads are done before copying
        #pragma omp flush(HMC_outputs)  // Ensure all writes to HMC_outputs are visible
        // Rows are iterations; columns are chains. Copy diagnostics already recorded by the sampler.
        p_jump_main_output.resize(n_iter, n_threads);
        p_jump_nuisance_output.resize(n_iter, sample_nuisance ? n_threads : 0);
        
        for (int i = 0; i < n_threads; ++i) {
              p_jump_main_output.col(i) = HMC_outputs[i].diagnostics_p_jump_main();
              if (sample_nuisance) {
                  p_jump_nuisance_output.col(i) = HMC_outputs[i].diagnostics_p_jump_us();
              }
          
              for (int ii = 0; ii < n_iter; ++ii) {
                
                      trace_output[i].col(ii) = HMC_outputs[i].trace_main().col(ii);
                      trace_output_divs[i].col(ii) =  HMC_outputs[i].trace_div().col(ii);
                      if (sample_nuisance == true) {
                        const int n_us = HMC_outputs[i].trace_nuisance().rows();
                        if (trace_output_nuisance[i].rows() != n_us) {
                          trace_output_nuisance[i].resize(n_us, n_iter);
                        }
                        trace_output_nuisance[i].col(ii) =  HMC_outputs[i].trace_nuisance().col(ii);
                      }
                      if (Model_type != "Stan") {
                        trace_output_log_lik[i].col(ii) = HMC_outputs[i].trace_log_lik().col(ii);
                      }
                      
              }
          
        }
 
}



 





  
