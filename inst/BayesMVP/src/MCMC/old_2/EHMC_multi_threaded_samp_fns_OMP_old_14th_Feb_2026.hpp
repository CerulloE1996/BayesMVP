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
                              // const Eigen::Matrix<int, -1, -1> &y_copy,
                              //////////////  input structs
                              std::vector<Model_fn_args_struct> &Model_args_as_cpp_struct_copies,
                              // const Model_fn_args_struct &Model_args_as_cpp_struct,
                              std::vector<EHMC_fn_args_struct> &EHMC_args_as_cpp_struct_copies,
                              const EHMC_Metric_struct &EHMC_Metric_as_cpp_struct_copies,
                              // const EHMC_Metric_struct &EHMC_Metric_as_cpp_struct,
                              //// -------------- For POST-BURNIN only:
                              const int &n_nuisance_to_track,
                              std::vector<Eigen::Matrix<double, -1, -1>> &trace_output,
                              std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_divs,
                              // std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_nuisance,
                              std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_log_lik
) {
        
        const int N = Model_args_as_cpp_struct_copies[0].N;
        const int n_us =  Model_args_as_cpp_struct_copies[0].n_nuisance;
        const int n_params_main = Model_args_as_cpp_struct_copies[0].n_params_main;
        // const int N = Model_args_as_cpp_struct.N;
        // const int n_us =  Model_args_as_cpp_struct.n_nuisance;
        // const int n_params_main = Model_args_as_cpp_struct.n_params_main;
        
        //// local storage:
        std::vector<HMC_output_single_chain> HMC_outputs;
        std::vector<HMCResult> HMC_inputs;
        // Reserve space:
        HMC_outputs.reserve(n_threads);
        HMC_inputs.reserve(n_threads);
        // Construct the structs:
        for (int i = 0; i < n_threads; ++i) {
          HMC_outputs.emplace_back(n_iter, n_nuisance_to_track, n_params_main, n_us, N);  // Construct directly in vector
          HMC_inputs.emplace_back(n_params_main, n_us, N);  // Construct directly in vector
        }
        
        const int global_seed_main_int =      static_cast<int>(global_seed);
        const int global_seed_nuisance_int =  static_cast<int>(global_seed + 1e6);
          
        // #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
        //   dqrng::xoshiro256plus global_rng_main(global_seed_main_int);
        //   dqrng::xoshiro256plus global_rng_nuisance(global_seed_nuisance_int);
        // #endif
          
        omp_set_num_threads(n_threads);
        omp_set_dynamic(0);
        
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
                // const int N = Model_args_as_cpp_struct.N;
                // const int n_us =  Model_args_as_cpp_struct.n_nuisance;
                // const int n_params_main = Model_args_as_cpp_struct.n_params_main;
                // const int n_params = n_params_main + n_us;
                
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
                        // Stan_model_struct  Stan_model_as_cpp_struct = fn_load_Stan_model_and_data( Model_args_as_cpp_struct.model_so_file,
                        //                                                                            Model_args_as_cpp_struct.json_file_path, 
                        //                                                                            seed_main_int_i);
                        
                        //////////////////////////////// perform iterations for chain i:
                        fn_sample_HMC_multi_iter_single_thread(   HMC_outputs[i] ,
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
                                                                  Stan_model_as_cpp_struct);
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
                                                                 Stan_model_as_cpp_struct);
                        ////////////////////////////// end of iteration(s)
                  
                }
          
        }  //// end of parallel OpenMP loop
        
        #pragma omp barrier  // Make sure all threads are done before copying
        #pragma omp flush(HMC_outputs)  // Ensure all writes to HMC_outputs are visible
        
        for (int i = 0; i < n_threads; ++i) {
          
              for (int ii = 0; ii < n_iter; ++ii) {
                
                      trace_output[i].col(ii) = HMC_outputs[i].trace_main().col(ii);
                      trace_output_divs[i].col(ii) =  HMC_outputs[i].trace_div().col(ii);
                      // if (sample_nuisance == true) {
                      //   trace_output_nuisance[i].col(ii) =  HMC_outputs[i].trace_nuisance().col(ii);
                      // }
                      if (Model_type != "Stan") {
                        trace_output_log_lik[i].col(ii) = HMC_outputs[i].trace_log_lik().col(ii);
                      }
                      
              }
          
        }
 
}



 





  