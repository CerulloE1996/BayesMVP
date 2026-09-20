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
 
 
 
 
 


     
// --------------------------------- RcpParallel  functions  -- BURNIN fn --------------------------------------------------------------------------------------------------------------------------------
     
     
 
  
  
class RcppParallel_EHMC_burnin : public RcppParallel::Worker {
  
public:
  
               // //// Clear Eigen matrices:
               // void reset_Eigen() {
               //       theta_main_vectors_all_chains_input_from_R_RcppPar.resize(0, 0);
               //       theta_us_vectors_all_chains_input_from_R_RcppPar.resize(0, 0);
               // }
               // 
               // // //// Clear all tbb concurrent vectors:
               // // void reset_tbb() { 
               // //       HMC_outputs.clear();
               // //       HMC_inputs.clear();
               // //       y_copies.clear();
               // //       Model_args_as_cpp_struct_copies.clear();
               // //       EHMC_args_as_cpp_struct_copies.clear();
               // //       EHMC_Metric_as_cpp_struct_copies.clear();
               // //       EHMC_burnin_as_cpp_struct_copies.clear();
               // // }
               // 
               // //// Clear all:
               // void reset() {
               //       // reset_tbb();
               //       reset_Eigen();
               // } 
               
               //////////////////// ---- declare variables:
               // #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
               //               dqrng::xoshiro256plus global_rng_main;
               //               dqrng::xoshiro256plus global_rng_nuisance;
               // #endif
               
               const uint64_t global_seed;
               const int n_threads;
               const int n_iter;
               const bool partitioned_HMC;
               const bool diffusion_HMC;
               const std::string Model_type;
               const bool sample_nuisance; 
               const bool force_autodiff;
               const bool force_PartialLog;
               const bool multi_attempts;
               
               //// local storage:
               std::vector<HMC_output_single_chain> HMC_outputs;
               std::vector<HMCResult> HMC_inputs;
               
               //// Input data (to read):
               const Eigen::Matrix<double, -1, -1>  &theta_main_vectors_all_chains_input_from_R_RcppPar;
               const Eigen::Matrix<double, -1, -1>  &velocity_main_vectors_all_chains_input_from_R_RcppPar;
               
               //// Input data (to read):
               const Eigen::Matrix<double, -1, -1>  &theta_us_vectors_all_chains_input_from_R_RcppPar;
               const Eigen::Matrix<double, -1, -1>  &velocity_us_vectors_all_chains_input_from_R_RcppPar;
               
               //// data:
               const std::vector<Eigen::Matrix<int, -1, -1>> &y_copies;
               // const Eigen::Matrix<int, -1, -1> &y_copy;
               
               //// ---- input structs:
               const std::vector<Model_fn_args_struct>   &Model_args_as_cpp_struct_copies;
               // const Model_fn_args_struct &Model_args_as_cpp_struct;
               ////
               std::vector<EHMC_fn_args_struct> &EHMC_args_as_cpp_struct_copies;  //// This has to be modifiable
               ////
               const std::vector<EHMC_Metric_struct>     &EHMC_Metric_as_cpp_struct_copies;
               // const EHMC_Metric_struct &EHMC_Metric_as_cpp_struct;
               
               //////////////////// ---- declare BURNIN-SPECIFIC variables:
               //// The current burnin iteration:
               const int current_iter;
               //// burnin struct:
               // const std::vector<EHMC_burnin_struct> &EHMC_burnin_as_cpp_struct_copies;
               //// Outputs for main + nuisance:
               Rcpp::NumericMatrix  &theta_main_vectors_all_chains_output;
               Rcpp::NumericMatrix  &velocity_main_vectors_all_chains_output;
               Rcpp::NumericMatrix  &theta_us_vectors_all_chains_output;
               Rcpp::NumericMatrix  &velocity_us_vectors_all_chains_output;
               //// other main outputs:
               Rcpp::NumericMatrix  &theta_main_0_burnin_tau_adapt_all_chains_output;
               Rcpp::NumericMatrix  &theta_main_prop_burnin_tau_adapt_all_chains_output;
               Rcpp::NumericMatrix  &velocity_main_0_burnin_tau_adapt_all_chains_output;
               Rcpp::NumericMatrix  &velocity_main_prop_burnin_tau_adapt_all_chains_output;
               Rcpp::NumericMatrix  &other_main_out_vector_all_chains_output;
               //// other nuisance outputs:
               Rcpp::NumericMatrix  &theta_us_0_burnin_tau_adapt_all_chains_output;
               Rcpp::NumericMatrix  &theta_us_prop_burnin_tau_adapt_all_chains_output;
               Rcpp::NumericMatrix  &velocity_us_0_burnin_tau_adapt_all_chains_output;
               Rcpp::NumericMatrix  &velocity_us_prop_burnin_tau_adapt_all_chains_output;
               Rcpp::NumericMatrix  &other_us_out_vector_all_chains_output;
       
       ////////////// Constructor (initialise these with the SOURCE format)
       RcppParallel_EHMC_burnin(    const int &n_threads_R,
                                    const uint64_t &global_seed_R,
                                    const int &n_iter_R,
                                    const bool &partitioned_HMC_R,
                                    const bool &diffusion_HMC_R,
                                    const std::string &Model_type_R,
                                    const bool &sample_nuisance_R,
                                    const bool &force_autodiff_R,
                                    const bool &force_PartialLog_R,
                                    const bool &multi_attempts_R,
                                    //// inputs:
                                    const Eigen::Matrix<double, -1, -1> &theta_main_vectors_all_chains_input_from_R,
                                    const Eigen::Matrix<double, -1, -1> &velocity_main_vectors_all_chains_input_from_R,
                                    //// inputs:
                                    const Eigen::Matrix<double, -1, -1> &theta_us_vectors_all_chains_input_from_R,
                                    const Eigen::Matrix<double, -1, -1> &velocity_us_vectors_all_chains_input_from_R,
                                    ////  data:
                                    const std::vector<Eigen::Matrix<int, -1, -1>> &y_copies_,
                                    // const Eigen::Matrix<int, -1, -1> &y_copy_,
                                    ////  input structs:
                                    const std::vector<Model_fn_args_struct> &Model_args_as_cpp_struct_copies_,
                                    // const Model_fn_args_struct &Model_args_as_cpp_struct_,   // READ-ONLY
                                    std::vector<EHMC_fn_args_struct>  &EHMC_args_as_cpp_struct_copies_,
                                    const std::vector<EHMC_Metric_struct>   &EHMC_Metric_as_cpp_struct_copies_,
                                    // const EHMC_Metric_struct &EHMC_Metric_as_cpp_struct_, // READ-ONLY
                                    //// ---------  burnin-specific stuff:
                                    const int &current_iter_R,
                                    // const std::vector<EHMC_burnin_struct> &EHMC_burnin_as_cpp_struct_copies_,
                                    //// outputs:
                                    Rcpp::NumericMatrix  &theta_main_vectors_all_chains_output_,
                                    Rcpp::NumericMatrix  &velocity_main_vectors_all_chains_output_,
                                    Rcpp::NumericMatrix  &theta_us_vectors_all_chains_output_,
                                    Rcpp::NumericMatrix  &velocity_us_vectors_all_chains_output_,
                                    //// other main outputs:
                                    Rcpp::NumericMatrix  &theta_main_0_burnin_tau_adapt_all_chains_output_,
                                    Rcpp::NumericMatrix  &theta_main_prop_burnin_tau_adapt_all_chains_output_,
                                    Rcpp::NumericMatrix  &velocity_main_0_burnin_tau_adapt_all_chains_output_,
                                    Rcpp::NumericMatrix  &velocity_main_prop_burnin_tau_adapt_all_chains_output_,
                                    Rcpp::NumericMatrix  &other_main_out_vector_all_chains_output_,
                                    //// other nuisance outputs:
                                    Rcpp::NumericMatrix  &theta_us_0_burnin_tau_adapt_all_chains_output_,
                                    Rcpp::NumericMatrix  &theta_us_prop_burnin_tau_adapt_all_chains_output_,
                                    Rcpp::NumericMatrix  &velocity_us_0_burnin_tau_adapt_all_chains_output_,
                                    Rcpp::NumericMatrix  &velocity_us_prop_burnin_tau_adapt_all_chains_output_,
                                    Rcpp::NumericMatrix  &other_us_out_vector_all_chains_output_
                                    )
         :
         n_threads(n_threads_R),
         global_seed(global_seed_R),
         n_iter(n_iter_R),
         partitioned_HMC(partitioned_HMC_R),
         diffusion_HMC(diffusion_HMC_R),
         Model_type(Model_type_R),
         sample_nuisance(sample_nuisance_R),
         force_autodiff(force_autodiff_R),
         force_PartialLog(force_PartialLog_R),
         multi_attempts(multi_attempts_R) ,
         //// inputs:
         theta_main_vectors_all_chains_input_from_R_RcppPar(theta_main_vectors_all_chains_input_from_R),
         velocity_main_vectors_all_chains_input_from_R_RcppPar(velocity_main_vectors_all_chains_input_from_R),
         //// inputs:
         theta_us_vectors_all_chains_input_from_R_RcppPar(theta_us_vectors_all_chains_input_from_R),
         velocity_us_vectors_all_chains_input_from_R_RcppPar(velocity_us_vectors_all_chains_input_from_R),
         //// Data:
         y_copies(y_copies_),
         // y_copy(y_copy_),
         //// input structs:
         // Model_args_as_cpp_struct(Model_args_as_cpp_struct_),
         Model_args_as_cpp_struct_copies(Model_args_as_cpp_struct_copies_),
         EHMC_args_as_cpp_struct_copies(EHMC_args_as_cpp_struct_copies_),
         // EHMC_Metric_as_cpp_struct(EHMC_Metric_as_cpp_struct_),
         EHMC_Metric_as_cpp_struct_copies(EHMC_Metric_as_cpp_struct_copies_),
         //// -------------- For burnin only:
         current_iter(current_iter_R),
         // EHMC_burnin_as_cpp_struct_copies(EHMC_burnin_as_cpp_struct_copies_),
         /// outputs:
         theta_main_vectors_all_chains_output(theta_main_vectors_all_chains_output_),
         velocity_main_vectors_all_chains_output(velocity_main_vectors_all_chains_output_),
         ////
         theta_us_vectors_all_chains_output(theta_us_vectors_all_chains_output_),
         velocity_us_vectors_all_chains_output(velocity_us_vectors_all_chains_output_),
         //// other main outputs:
         theta_main_0_burnin_tau_adapt_all_chains_output(theta_main_0_burnin_tau_adapt_all_chains_output_),
         theta_main_prop_burnin_tau_adapt_all_chains_output(theta_main_prop_burnin_tau_adapt_all_chains_output_),
         velocity_main_0_burnin_tau_adapt_all_chains_output(velocity_main_0_burnin_tau_adapt_all_chains_output_),
         velocity_main_prop_burnin_tau_adapt_all_chains_output(velocity_main_prop_burnin_tau_adapt_all_chains_output_),
         other_main_out_vector_all_chains_output(other_main_out_vector_all_chains_output_),
         //// other nuisance outputs:
         theta_us_0_burnin_tau_adapt_all_chains_output(theta_us_0_burnin_tau_adapt_all_chains_output_),
         theta_us_prop_burnin_tau_adapt_all_chains_output(theta_us_prop_burnin_tau_adapt_all_chains_output_),
         velocity_us_0_burnin_tau_adapt_all_chains_output(velocity_us_0_burnin_tau_adapt_all_chains_output_),
         velocity_us_prop_burnin_tau_adapt_all_chains_output(velocity_us_prop_burnin_tau_adapt_all_chains_output_),
         other_us_out_vector_all_chains_output(other_us_out_vector_all_chains_output_)
       { 
         
                 // #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
                 //         global_rng_main.seed(global_seed_R);
                 //         global_rng_nuisance.seed(global_seed_R + 1e6);
                 // #endif
                 
                 const int N = Model_args_as_cpp_struct_copies[0].N;
                 const int n_nuisance =  Model_args_as_cpp_struct_copies[0].n_nuisance;
                 const int n_params_main = Model_args_as_cpp_struct_copies[0].n_params_main;
                 // const int N = Model_args_as_cpp_struct.N;
                 // const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
                 // const int n_params_main = Model_args_as_cpp_struct.n_params_main;
                 ////
                 const int n_nuisance_to_track = 1;
                 
                 HMC_outputs.reserve(n_threads_R);
                 HMC_inputs.reserve(n_threads_R);
                 
                 for (int i = 0; i < n_threads_R; ++i) {
                   HMC_outputs.emplace_back(n_iter_R, n_nuisance_to_track, n_params_main, n_nuisance, N);
                   HMC_inputs.emplace_back(n_params_main, n_nuisance, N);  // Construct HMCResult directly in vector
                 }
         
       }
       
   ////////////// RcppParallel Parallel operator
   void operator() (std::size_t begin, std::size_t end) {
     
               const uint64_t global_seed_main =     global_seed;
               const uint64_t global_seed_nuisance = global_seed + 1e6;
               const int global_seed_main_int =      static_cast<int>(global_seed_main);
               const int global_seed_nuisance_int =  static_cast<int>(global_seed_nuisance);
               
               //// Process all chains from begin to end:
               for (std::size_t i = begin; i < end; ++i) {
                     
                            const int chain_id_int = static_cast<int>(i);
                            const int seed_main_int_i =     global_seed_main_int +     n_iter*(1 + chain_id_int);
                            const int seed_nuisance_int_i = global_seed_nuisance_int + n_iter*(1 + chain_id_int);
                            
                             #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
                                         dqrng::xoshiro256plus rng_main_i; // (global_rng_main);      // make thread local copy of rng 
                                         dqrng::xoshiro256plus rng_nuisance_i; //(global_rng_nuisance);      // make thread local copy of rng 
                                         rng_main_i.seed(seed_main_int_i);  // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                                         rng_nuisance_i.seed(seed_nuisance_int_i);  // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                             #elif RNG_TYPE_CPP_STD == 1
                                         std::mt19937 rng_main_i;  // Fresh RNG // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                                         std::mt19937 rng_nuisance_i;  // Fresh RNG // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                                         rng_main_i.seed(seed_main_int_i); // set / re-set the seed
                                         rng_nuisance_i.seed(seed_nuisance_int_i); // set / re-set the seed
                             #endif
                                       
                            // #elif RNG_TYPE_pcg64 == 1
                            //            pcg_extras::seed_seq_from<std::random_device> global_seed;
                            //            pcg_extras::seed_seq_from<std::random_device> global_seed;
                            //            pcg64 rng_main_i(global_seed, n_iter*(1 + chain_id_int)); // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                            //            pcg64 rng_nuisance_i(global_seed, n_iter*(1 + chain_id_int)); // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                            // #elif RNG_TYPE_pcg32 == 1
                            //            pcg_extras::seed_seq_from<std::random_device> global_seed;
                            //            pcg_extras::seed_seq_from<std::random_device> global_seed;
                            //            pcg32 rng_main_i(global_seed, n_iter*(1 + chain_id_int)); // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                            //            pcg32 rng_nuisance_i(global_seed, n_iter*(1 + chain_id_int)); // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                            // #endif
                           
                            const int N = Model_args_as_cpp_struct_copies[i].N;
                            const int n_nuisance =  Model_args_as_cpp_struct_copies[i].n_nuisance;
                            const int n_params_main = Model_args_as_cpp_struct_copies[i].n_params_main;
                            const int n_params = n_params_main + n_nuisance;
                            // const int N = Model_args_as_cpp_struct.N;
                            // const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
                            // const int n_params_main = Model_args_as_cpp_struct.n_params_main;
                            // const int n_params = n_params_main + n_nuisance;
                           
                            stan::math::ChainableStack ad_tape;     // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                            //// stan::math::nested_rev_autodiff nested; // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                           
                            const bool burnin_indicator = true;
                            const int n_nuisance_to_track = 1;
                             
                            {
                              
                                  ///////////////////////////////////////// perform iterations for adaptation interval
                                  HMC_inputs[i].main_theta_vec() =   theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
                                  HMC_inputs[i].main_theta_vec_0() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
                                  ////
                                  HMC_inputs[i].us_theta_vec() =     theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
                                  HMC_inputs[i].us_theta_vec_0() =   theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
                                  
                                  {
                                    
                                    //////////////////////////////// perform iterations for chain i
                                    if (Model_type == "Stan") {  
                                          
                                          Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct_copies[i].model_so_file,
                                                                                                                     Model_args_as_cpp_struct_copies[i].json_file_path,
                                                                                                                     seed_main_int_i);
                                          // Stan_model_struct  Stan_model_as_cpp_struct = fn_load_Stan_model_and_data( Model_args_as_cpp_struct.model_so_file,
                                          //                                                                            Model_args_as_cpp_struct.json_file_path, 
                                          //                                                                            seed_main_int_i);
                                          
                                          fn_sample_HMC_multi_iter_single_thread(    HMC_outputs[i],
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
                                                                                     Model_type, sample_nuisance,
                                                                                     force_autodiff, force_PartialLog,  multi_attempts,  n_nuisance_to_track, 
                                                                                     y_copies[i],
                                                                                     // y_copy,
                                                                                     // Model_args_as_cpp_struct,
                                                                                     Model_args_as_cpp_struct_copies[i],
                                                                                     EHMC_args_as_cpp_struct_copies[i], 
                                                                                     // EHMC_Metric_as_cpp_struct,
                                                                                     EHMC_Metric_as_cpp_struct_copies[i],
                                                                                     Stan_model_as_cpp_struct);
                                          //// destroy Stan model object:
                                          fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
                                      
                                    } else { 
                                      
                                          Stan_model_struct Stan_model_as_cpp_struct; ///  dummy struct
                                          
                                          fn_sample_HMC_multi_iter_single_thread(    HMC_outputs[i],
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
                                                                                     Model_type, sample_nuisance,
                                                                                     force_autodiff, force_PartialLog,  multi_attempts,  n_nuisance_to_track, 
                                                                                     y_copies[i],
                                                                                     // y_copy,
                                                                                     // Model_args_as_cpp_struct,
                                                                                     Model_args_as_cpp_struct_copies[i],
                                                                                     EHMC_args_as_cpp_struct_copies[i], 
                                                                                     // EHMC_Metric_as_cpp_struct,
                                                                                     EHMC_Metric_as_cpp_struct_copies[i],
                                                                                     Stan_model_as_cpp_struct);
                                          
                                          
                                    }
                                    
                                  } /// end of big local block
                                  
                            }
                             
                           
                    }  //// end of all parallel work// Definition of static thread_local variable
                 
                   
  } /// end of void RcppParallel operator
       
       // Copy results directly to R matrices
       void copy_results_to_output() {
         
               for (int i = 0; i < n_threads; ++i) {
                 
                       //////// Write results back to the shared array - MAIN PARAMS:
                       theta_main_vectors_all_chains_output.column(i) =                  fn_convert_EigenVec_to_RcppVec_dbl(HMC_inputs[i].main_theta_vec());
                       theta_main_0_burnin_tau_adapt_all_chains_output.column(i) =       fn_convert_EigenVec_to_RcppVec_dbl(HMC_inputs[i].main_theta_vec_0());
                       theta_main_prop_burnin_tau_adapt_all_chains_output.column(i) =    fn_convert_EigenVec_to_RcppVec_dbl(HMC_inputs[i].main_theta_vec_proposed());
                       velocity_main_vectors_all_chains_output.column(i) =               fn_convert_EigenVec_to_RcppVec_dbl(HMC_inputs[i].main_velocity_vec());
                       velocity_main_0_burnin_tau_adapt_all_chains_output.column(i) =    fn_convert_EigenVec_to_RcppVec_dbl(HMC_inputs[i].main_velocity_0_vec());
                       velocity_main_prop_burnin_tau_adapt_all_chains_output.column(i) = fn_convert_EigenVec_to_RcppVec_dbl(HMC_inputs[i].main_velocity_vec_proposed());
                       
                       //////// Write results back to the shared array - NUISANCE PARAMS:
                       theta_us_vectors_all_chains_output.column(i) =                  fn_convert_EigenVec_to_RcppVec_dbl(HMC_inputs[i].us_theta_vec());
                       theta_us_0_burnin_tau_adapt_all_chains_output.column(i) =       fn_convert_EigenVec_to_RcppVec_dbl(HMC_inputs[i].us_theta_vec_0());
                       theta_us_prop_burnin_tau_adapt_all_chains_output.column(i) =    fn_convert_EigenVec_to_RcppVec_dbl(HMC_inputs[i].us_theta_vec_proposed());
                       velocity_us_vectors_all_chains_output.column(i) =               fn_convert_EigenVec_to_RcppVec_dbl(HMC_inputs[i].us_velocity_vec());
                       velocity_us_0_burnin_tau_adapt_all_chains_output.column(i) =    fn_convert_EigenVec_to_RcppVec_dbl(HMC_inputs[i].us_velocity_0_vec());
                       velocity_us_prop_burnin_tau_adapt_all_chains_output.column(i) = fn_convert_EigenVec_to_RcppVec_dbl(HMC_inputs[i].us_velocity_vec_proposed());
                       
                       other_main_out_vector_all_chains_output(0, i) =  HMC_outputs[i].diagnostics_p_jump_main().sum() / static_cast<double>(n_iter);
                       other_main_out_vector_all_chains_output(1, i) =  static_cast<double>(HMC_outputs[i].diagnostics_div_main().sum());
                       if (sample_nuisance == true)  {
                         other_us_out_vector_all_chains_output(0, i) =  HMC_outputs[i].diagnostics_p_jump_us().sum() /  static_cast<double>(n_iter);
                         other_us_out_vector_all_chains_output(1, i) =  static_cast<double>(HMC_outputs[i].diagnostics_div_us().sum());
                       }
                       
                       //// other outputs (once all iterations finished) - main:
                       // other_main_out_vector_all_chains_output(2, i) = EHMC_burnin_as_cpp_struct_copies[i].tau_m_adam_main;
                       // other_main_out_vector_all_chains_output(3, i) = EHMC_burnin_as_cpp_struct_copies[i].tau_v_adam_main;
                       other_main_out_vector_all_chains_output(4, i) = EHMC_args_as_cpp_struct_copies[i].tau_main;
                       other_main_out_vector_all_chains_output(5, i) = EHMC_args_as_cpp_struct_copies[i].tau_main_ii;
                       // other_main_out_vector_all_chains_output(6, i) = EHMC_burnin_as_cpp_struct_copies[i].eps_m_adam_main;
                       // other_main_out_vector_all_chains_output(7, i) = EHMC_burnin_as_cpp_struct_copies[i].eps_v_adam_main;
                       other_main_out_vector_all_chains_output(8, i) = EHMC_args_as_cpp_struct_copies[i].eps_main;
                       //// other outputs (once all iterations finished) - nuisance:
                       if (sample_nuisance == true)  {
                         
                           // other_us_out_vector_all_chains_output(2, i) = EHMC_burnin_as_cpp_struct_copies[i].tau_m_adam_us;
                           // other_us_out_vector_all_chains_output(3, i) = EHMC_burnin_as_cpp_struct_copies[i].tau_v_adam_us;
                           other_us_out_vector_all_chains_output(4, i) = EHMC_args_as_cpp_struct_copies[i].tau_us;
                           other_us_out_vector_all_chains_output(5, i) = EHMC_args_as_cpp_struct_copies[i].tau_us_ii;
                           // other_us_out_vector_all_chains_output(6, i) = EHMC_burnin_as_cpp_struct_copies[i].eps_m_adam_us;
                           // other_us_out_vector_all_chains_output(7, i) = EHMC_burnin_as_cpp_struct_copies[i].eps_v_adam_us;
                           other_us_out_vector_all_chains_output(8, i) = EHMC_args_as_cpp_struct_copies[i].eps_us;
                           
                       }
                 
               }
               
       }
       
    
};
























 





  