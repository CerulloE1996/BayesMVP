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
               ////
               // std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs;
               std::vector<std::vector<LC_MVP_workspace_struct>> LC_MVP_ws_structs_double_vec;

               const int n_threads_WCP;

               //// Input data (to read):
               const Eigen::Matrix<double, -1, -1>  &theta_main_vectors_all_chains_input_from_R_RcppPar;
               const Eigen::Matrix<double, -1, -1>  &velocity_main_vectors_all_chains_input_from_R_RcppPar;

               //// Input data (to read):
               const Eigen::Matrix<double, -1, -1>  &theta_us_vectors_all_chains_input_from_R_RcppPar;
               const Eigen::Matrix<double, -1, -1>  &velocity_us_vectors_all_chains_input_from_R_RcppPar;

               //// data:
               const std::vector<Eigen::Matrix<int, -1, -1>> &y_copies;

               //// ---- input structs:
               const std::vector<Model_fn_args_struct>   &Model_args_as_cpp_struct_copies;
               ////
               std::vector<EHMC_fn_args_struct> &EHMC_args_as_cpp_struct_copies;  //// This has to be modifiable
               ////
               const std::vector<EHMC_Metric_struct>     &EHMC_Metric_as_cpp_struct_copies;

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

               // ////
               // const bool use_disk;
               // const std::string trace_dir;

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
                                    ////  input structs:
                                    const std::vector<Model_fn_args_struct> &Model_args_as_cpp_struct_copies_,
                                    std::vector<EHMC_fn_args_struct>  &EHMC_args_as_cpp_struct_copies_,
                                    const std::vector<EHMC_Metric_struct>   &EHMC_Metric_as_cpp_struct_copies_,
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
                                    Rcpp::NumericMatrix  &other_us_out_vector_all_chains_output_,
                                    ////
                                    const int &n_threads_WCP_
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
         //// input structs:
         Model_args_as_cpp_struct_copies(Model_args_as_cpp_struct_copies_),
         EHMC_args_as_cpp_struct_copies(EHMC_args_as_cpp_struct_copies_),
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
         other_us_out_vector_all_chains_output(other_us_out_vector_all_chains_output_),
         ////
         n_threads_WCP(n_threads_WCP_)
       {

                 // #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
                 //         global_rng_main.seed(global_seed_R);
                 //         global_rng_nuisance.seed(global_seed_R + 1e6);
                 // #endif

                 const int N = Model_args_as_cpp_struct_copies[0].N;
                 const int n_nuisance =  Model_args_as_cpp_struct_copies[0].n_nuisance;
                 const int n_params_main = Model_args_as_cpp_struct_copies[0].n_params_main;
                 ////
                 //// must equal n_nuisance: store_iteration() writes the full nuisance
                 //// column when sample_nuisance == TRUE.
                 const int n_nuisance_to_track = n_nuisance;

                 HMC_outputs.reserve(n_threads_R);
                 HMC_inputs.reserve(n_threads_R);

                 for (int i = 0; i < n_threads_R; ++i) {
                   HMC_outputs.emplace_back( n_iter_R, n_nuisance_to_track, n_params_main, n_nuisance, N);
                   HMC_inputs.emplace_back( n_params_main, n_nuisance, N);  // Construct HMCResult directly in vector
                 }

                 // // LC_MVP_ws_struct.resize(n_threads_R);
                 // LC_MVP_ws_struct.reserve(n_threads_R);

                 LC_MVP_ws_structs_double_vec.resize(n_threads_R);

                 if ((Model_type == "LC_MVP") || (Model_type == "LC_MVOP")) {

                         const int n_tests = y_copies_[0].cols();
                         const int n_class = Model_args_as_cpp_struct_copies_[0].Model_args_ints(1);
                         const int n_chunks = Model_args_as_cpp_struct_copies_[0].Model_args_ints(3);
                         const std::string &vect_type = Model_args_as_cpp_struct_copies_[0].Model_args_strings(0);
                         const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct_copies_[0].Model_args_mats_int[0];

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

                         // std::cout << "DEBUG: N=" << N << " vec_size=" << vec_size
                         //           << " n_chunks=" << n_chunks
                         //           << "Model_args_as_cpp_struct_copies[0].N = " << Model_args_as_cpp_struct_copies[0].N  << std::endl;

                         for (int i = 0; i < n_threads_R; ++i) {

                             LC_MVP_ws_structs_double_vec[i].reserve(n_threads_WCP_);

                             for (int t = 0; t < n_threads_WCP_; ++t) {

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

                         // std::cout << "Pre-allocated " << n_threads_R << " lp_grad workspaces ("
                         //           << "chunk_size=" << chunk_info.normal_chunk_size
                         //           << ", n_tests=" << n_tests << ")" << std::endl;

                 } // else: LC_MVP_ws_struct[i].is_allocated stays false (dummy)

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

                            stan::math::ChainableStack ad_tape;     // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                            //// stan::math::nested_rev_autodiff nested; // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)

                            const bool burnin_indicator = true;
                            //// must equal n_nuisance: store_iteration() writes the full
                            //// nuisance column when sample_nuisance == TRUE.
                            const int n_nuisance_to_track = n_nuisance;
                            // const int n_nuisance_to_track = n_nuisance;

                            {

                                  ///////////////////////////////////////// perform iterations for adaptation interval
                                  HMC_inputs[i].main_theta_vec() =   theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
                                  HMC_inputs[i].main_theta_vec_0() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
                                  HMC_inputs[i].us_theta_vec() =     theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
                                  HMC_inputs[i].us_theta_vec_0() =   theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);

                                  {

                                    //////////////////////////////// perform iterations for chain i
                                    if (Model_type == "Stan") {

                                          Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct_copies[i].model_so_file,
                                                                                                                     Model_args_as_cpp_struct_copies[i].json_file_path,
                                                                                                                     seed_main_int_i);

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
                                                                                     Model_args_as_cpp_struct_copies[i],
                                                                                     EHMC_args_as_cpp_struct_copies[i],
                                                                                     EHMC_Metric_as_cpp_struct_copies[i],
                                                                                     Stan_model_as_cpp_struct,
                                                                                     LC_MVP_ws_structs_double_vec[i],
                                                                                     n_threads_WCP);
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
                                                                                     Model_args_as_cpp_struct_copies[i],
                                                                                     EHMC_args_as_cpp_struct_copies[i],
                                                                                     EHMC_Metric_as_cpp_struct_copies[i],
                                                                                     Stan_model_as_cpp_struct,
                                                                                     LC_MVP_ws_structs_double_vec[i],
                                                                                     n_threads_WCP);


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






//// ---------------------------------------------------------------------------------------------------
//// PERSISTENT BURNIN WORKER
////
//// Construct ONCE per burnin phase; run one HMC iteration per call with all heavy
//// state (workspaces, HMC buffers, struct copies, y copies) resident in C++.
////
//// Replaces the per-iteration construction inside fn_R_RcppParallel_EHMC_single_iter_burnin:
////   - HMC_outputs / HMC_inputs allocation           -> now allocated once
////   - LC_MVP_ws_structs_double_vec allocation       -> now allocated once
////   - y_copies replication (8x N x n_tests)         -> now replicated once
////   - Model_args / EHMC args / Metric replication   -> Model_args once; args/metric
////                                                      overwritten in-place each iter
////   - theta input marshalling                       -> resident in HMC_inputs
////
//// Put this in the SAME translation unit as fn_R_RcppParallel_EHMC_single_iter_burnin
//// (it needs the same headers / helper functions / structs already visible there:
////  HMC_output_single_chain, HMCResult, LC_MVP_workspace_struct, Model_fn_args_struct,
////  EHMC_fn_args_struct, EHMC_Metric_struct, convert_R_List_*, replicate_*,
////  fn_sample_HMC_multi_iter_single_thread, fn_load_Stan_model_and_data,
////  fn_bs_destroy_Stan_model, calculate_chunk_sizes, vec_of_mats, warmUpThreads,
////  fn_convert_EigenVec_to_RcppVec_dbl).
//// ---------------------------------------------------------------------------------------------------


// void warmUpThreads(int n_threads); // fwd. declaration

struct WarmUp : public RcppParallel::Worker {
  void operator()(std::size_t begin, std::size_t end) override {
    // Perform a dummy operation
    for (std::size_t i = begin; i < end; ++i) {
      volatile double x = i * 0.1; // Prevent compiler optimization
    }
  }
};

// Call this before starting your main function
void warmUpThreads(std::size_t nThreads) {
  WarmUp warmUpTask;
  RcppParallel::parallelFor(0, nThreads, warmUpTask);
}

class PersistentBurninState {
  
public:
  //// Each chain owns a separate bool (not packed std::vector<bool> storage).
  struct LpGradCacheState { bool valid = false; };
  std::vector<LpGradCacheState> lp_grad_cache;
  
  //// ---- fixed config (set once at construction):
  const int n_threads;
  const bool partitioned_HMC;
  const bool diffusion_HMC;
  const std::string Model_type;
  const bool sample_nuisance;
  const bool force_autodiff;
  const bool force_PartialLog;
  const bool multi_attempts;
  const int n_threads_WCP;
  ////
  int n_params_main;
  int n_nuisance;
  int N;
  
  //// ---- resident heavy state (allocated ONCE):
  std::vector<HMC_output_single_chain> HMC_outputs;
  std::vector<HMCResult>               HMC_inputs;
  std::vector<std::vector<LC_MVP_workspace_struct>> LC_MVP_ws_structs_double_vec;
  std::vector<Eigen::Matrix<int, -1, -1>> y_copies;
  ////
  std::vector<Model_fn_args_struct> Model_args_copies;   //// read-only after construction
  std::vector<EHMC_fn_args_struct>  EHMC_args_copies;    //// overwritten by update_adaptation()
  std::vector<EHMC_Metric_struct>   EHMC_Metric_copies;  //// overwritten by update_adaptation() -- MUTABLE now (was const& in old worker)
  
  //// ---------------------------------------------------------------- Constructor: pays ALL the setup cost, once.
  PersistentBurninState(  const int n_threads_R,
                          const bool partitioned_HMC_R,
                          const bool diffusion_HMC_R,
                          const std::string &Model_type_R,
                          const bool sample_nuisance_R,
                          const bool force_autodiff_R,
                          const bool force_PartialLog_R,
                          const bool multi_attempts_R,
                          const Eigen::Matrix<int, -1, -1> &y_Eigen_R,
                          const Rcpp::List &Model_args_as_Rcpp_List,
                          const Rcpp::List &EHMC_args_as_Rcpp_List,
                          const Rcpp::List &EHMC_Metric_as_Rcpp_List,
                          const int n_threads_WCP_R)
    :
    n_threads(n_threads_R),
    partitioned_HMC(partitioned_HMC_R),
    diffusion_HMC(diffusion_HMC_R),
    Model_type(Model_type_R),
    sample_nuisance(sample_nuisance_R),
    force_autodiff(force_autodiff_R),
    force_PartialLog(force_PartialLog_R),
    multi_attempts(multi_attempts_R),
    n_threads_WCP(n_threads_WCP_R)
  {
    
        //// ---- convert R lists to C++ structs ONCE:
        Model_fn_args_struct  Model_args_as_cpp_struct  = convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List);
        EHMC_fn_args_struct   EHMC_args_as_cpp_struct   = convert_R_List_EHMC_fn_args_struct(EHMC_args_as_Rcpp_List);
        EHMC_Metric_struct    EHMC_Metric_as_cpp_struct = convert_R_List_EHMC_Metric_struct(EHMC_Metric_as_Rcpp_List);
        
        //// ---- replicate per chain ONCE:
        Model_args_copies  = replicate_Model_fn_args_struct(Model_args_as_cpp_struct,  n_threads);
        EHMC_args_copies   = replicate_EHMC_fn_args_struct(EHMC_args_as_cpp_struct,   n_threads);
        EHMC_Metric_copies = replicate_EHMC_Metric_struct(EHMC_Metric_as_cpp_struct, n_threads);
        
        //// ---- y copies ONCE:
        y_copies = vec_of_mats<int>(y_Eigen_R.rows(), y_Eigen_R.cols(), n_threads);
        for (int kk = 0; kk < n_threads; ++kk)  y_copies[kk] = y_Eigen_R;
        
        N             = Model_args_copies[0].N;
        n_nuisance    = Model_args_copies[0].n_nuisance;
        n_params_main = Model_args_copies[0].n_params_main;
        
        //// ---- HMC buffers ONCE:
        lp_grad_cache.resize(n_threads);
        //// n_nuisance_to_track_alloc must equal the struct's n_nuisance: store_iteration()
        //// writes the full nuisance column when sample_nuisance == TRUE, so allocating
        //// only 1 row here would write out of bounds for any n_nuisance > 1.
        const int n_nuisance_to_track_alloc = n_nuisance;
        const int n_iter_local = 1;
        
        HMC_outputs.reserve(n_threads);
        HMC_inputs.reserve(n_threads);
    
        for (int i = 0; i < n_threads; ++i) {
          HMC_outputs.emplace_back(n_iter_local, n_nuisance_to_track_alloc, n_params_main, n_nuisance, N);
          HMC_inputs.emplace_back(n_params_main, n_nuisance, N);
        }
        
        //// ---- LC-MVP / LC-MVOP cache-blocked workspaces ONCE:
        LC_MVP_ws_structs_double_vec.resize(n_threads);
        if ((Model_type == "LC_MVP") || (Model_type == "LC_MVOP")) {
          
              const int n_tests = y_copies[0].cols();
              const int n_class  = Model_args_copies[0].Model_args_ints(1);
              const int n_chunks = Model_args_copies[0].Model_args_ints(3);
              const std::string &vect_type = Model_args_copies[0].Model_args_strings(0);
              const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_copies[0].Model_args_mats_int[0];
              
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
                }
              }
              
        } //// else: dummy (is_allocated stays false)
        
        //// ---- warm up TBB pool once:
        warmUpThreads(n_threads);
    
  }
  
  //// ---------------------------------------------------------------- Load initial theta from R (call ONCE before the loop):
  void set_theta(const Eigen::Matrix<double, -1, -1> &theta_main,
                 const Eigen::Matrix<double, -1, -1> &theta_us) {
    
        if (theta_main.cols() != n_threads || theta_us.cols() != n_threads) {
          Rcpp::stop("set_theta: number of columns must equal n_threads (one column per chain).");
        }
        for (int i = 0; i < n_threads; ++i) {
          HMC_inputs[i].main_theta_vec()   = theta_main.col(i);
          lp_grad_cache[i].valid = false;
          HMC_inputs[i].main_theta_vec_0() = theta_main.col(i);
          HMC_inputs[i].us_theta_vec()     = theta_us.col(i);
          HMC_inputs[i].us_theta_vec_0()   = theta_us.col(i);
        }
    
  }
  
  //// ---------------------------------------------------------------- Push this iteration's eps / tau / metric (call each iter -- a few KB + one nuisance-length vec):
  //// Wholesale overwrite from the R lists == exactly what the old wrapper did every call,
  //// so adaptation semantics are IDENTICAL; we just skip re-replicating Model_args, y, workspaces.
  void update_adaptation(const Rcpp::List &EHMC_args_as_Rcpp_List,
                         const Rcpp::List &EHMC_Metric_as_Rcpp_List) {
    
        EHMC_fn_args_struct  args   = convert_R_List_EHMC_fn_args_struct(EHMC_args_as_Rcpp_List);
        EHMC_Metric_struct   metric = convert_R_List_EHMC_Metric_struct(EHMC_Metric_as_Rcpp_List);
        for (int i = 0; i < n_threads; ++i) {
          EHMC_args_copies[i]   = args;
          EHMC_Metric_copies[i] = metric;
        }
    
  }
  
  //// ---------------------------------------------------------------- One chain, one iteration (called from the parallel worker):
  void run_chain(const int i,
                 const int seed, 
                 const int current_iter) {
    
        const int n_iter_local = 1;
        //// seed construction mirrors the old worker (n_iter member was 1 there too):
        const int seed_main_int_i     = seed                          + n_iter_local * (1 + i);
        const int seed_nuisance_int_i = seed + static_cast<int>(1e6)  + n_iter_local * (1 + i);
        
        #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
            dqrng::xoshiro256plus rng_main_i;
            dqrng::xoshiro256plus rng_nuisance_i;
            rng_main_i.seed(seed_main_int_i);
            rng_nuisance_i.seed(seed_nuisance_int_i);
        #elif RNG_TYPE_CPP_STD == 1
            std::mt19937 rng_main_i;
            std::mt19937 rng_nuisance_i;
            rng_main_i.seed(seed_main_int_i);
            rng_nuisance_i.seed(seed_nuisance_int_i);
        #endif
        
        stan::math::ChainableStack ad_tape;
        
        const bool burnin_indicator = true;
        //// must equal n_nuisance: store_iteration() writes the full nuisance column
        //// when sample_nuisance == TRUE.
        const int n_nuisance_to_track_local = n_nuisance; // n_nuisance;
        
        //// ---- carry state forward: current theta becomes this iteration's initial point
        //// (old flow: R passed last iter's output back in, and theta_vec_0 was set = input):
        HMC_inputs[i].main_theta_vec_0() = HMC_inputs[i].main_theta_vec();
        HMC_inputs[i].us_theta_vec_0()   = HMC_inputs[i].us_theta_vec();
        
        //// ---- share_tau_ii_across_chains (burn-in only): ONE trajectory length tau_ii per iteration for ALL chains.
        //// Every chain computes the same value from the same (seed, iteration) with a SEPARATE generator (parameter
        //// block 2), so the chains' own momentum streams (blocks 0 and 1) are untouched and momenta stay independent.
        //// All chains then take the same number of leapfrog steps, so an iteration no longer waits for whichever
        //// chain drew the longest trajectory (E[max of k draws of U(0, 2 tau)] = 2 tau k / (k + 1) grows with k).
        EHMC_args_copies[i].use_given_tau_main_ii = false;
        if (EHMC_args_copies[i].share_tau_ii_across_chains) {
              #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
                  dqrng::xoshiro256plus rng_shared_tau_ii;
              #elif RNG_TYPE_CPP_STD == 1
                  std::mt19937 rng_shared_tau_ii;
              #endif
              fn_seed_burnin_rng(rng_shared_tau_ii, seed, 0, current_iter, 2);
              //// same tau floor the samplers apply before drawing:
              const double tau_main_for_shared_draw = std::max(EHMC_args_copies[i].tau_main, EHMC_args_copies[i].eps_main);
              EHMC_args_copies[i].tau_main_ii = generate_random_tau_ii(tau_main_for_shared_draw, rng_shared_tau_ii);
              EHMC_args_copies[i].use_given_tau_main_ii = true;
        }
        
        if (Model_type == "Stan") {
          
              Stan_model_struct Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_copies[i].model_so_file,
                                                                                         Model_args_copies[i].json_file_path,
                                                                                         seed_main_int_i);
              fn_sample_HMC_multi_iter_single_thread(   HMC_outputs[i],
                                                        HMC_inputs[i],
                                                        burnin_indicator,
                                                        i,
                                                        current_iter,
                                                        seed_main_int_i,
                                                        seed_nuisance_int_i,
                                                        rng_main_i,
                                                        rng_nuisance_i,
                                                        n_iter_local,
                                                        partitioned_HMC,
                                                        diffusion_HMC,
                                                        Model_type, sample_nuisance,
                                                        force_autodiff, force_PartialLog, multi_attempts,
                                                        n_nuisance_to_track_local,
                                                        y_copies[i],
                                                        Model_args_copies[i],
                                                        EHMC_args_copies[i],
                                                        EHMC_Metric_copies[i],
                                                        Stan_model_as_cpp_struct,
                                                        LC_MVP_ws_structs_double_vec[i],
                                                        n_threads_WCP);
              fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
          
        } else {
          
              Stan_model_struct Stan_model_as_cpp_struct; //// dummy
          
              fn_sample_HMC_multi_iter_single_thread(   HMC_outputs[i],
                                                        HMC_inputs[i],
                                                        burnin_indicator,
                                                        i,
                                                        current_iter,
                                                        seed_main_int_i,
                                                        seed_nuisance_int_i,
                                                        rng_main_i,
                                                        rng_nuisance_i,
                                                        n_iter_local,
                                                        partitioned_HMC,
                                                        diffusion_HMC,
                                                        Model_type, sample_nuisance,
                                                        force_autodiff, force_PartialLog, multi_attempts,
                                                        n_nuisance_to_track_local,
                                                        y_copies[i],
                                                        Model_args_copies[i],
                                                        EHMC_args_copies[i],
                                                        EHMC_Metric_copies[i],
                                                        Stan_model_as_cpp_struct,
                                                        LC_MVP_ws_structs_double_vec[i],
                                                         n_threads_WCP,
                                                         &lp_grad_cache[i].valid);
          
        }
    
  }
  
};



//// ---------------------------------------------------------------- Thin RcppParallel worker: just dispatches chains to the resident state.
struct BurninChainProfile {
    double elapsed_seconds = 0.0;
    double start_offset_seconds = 0.0;
    int n_leapfrog_steps = 0;
};

class BurninIterWorker : public RcppParallel::Worker {
  
public:
  
      PersistentBurninState* st;
      const int seed;
      const int current_iter;
      std::vector<BurninChainProfile> *chain_profiles;
      std::chrono::steady_clock::time_point parallel_start;
      
      BurninIterWorker(PersistentBurninState* st_, const int seed_, const int current_iter_,
                       std::vector<BurninChainProfile> *chain_profiles_ = nullptr,
                       std::chrono::steady_clock::time_point parallel_start_ = std::chrono::steady_clock::time_point())
        : st(st_), seed(seed_), current_iter(current_iter_), chain_profiles(chain_profiles_), parallel_start(parallel_start_) {}
      
      void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
          if (chain_profiles == nullptr) {
            st->run_chain(static_cast<int>(i), seed, current_iter);
          } else {
            BurninChainProfile &profile = (*chain_profiles)[i];
            int *previous_counter = st->Model_args_copies[i].burnin_leapfrog_steps;
            st->Model_args_copies[i].burnin_leapfrog_steps = &profile.n_leapfrog_steps;
            const auto chain_start = std::chrono::steady_clock::now();
            try {
              st->run_chain(static_cast<int>(i), seed, current_iter);
            } catch (...) {
              st->Model_args_copies[i].burnin_leapfrog_steps = previous_counter;
              throw;
            }
            const auto chain_end = std::chrono::steady_clock::now();
            st->Model_args_copies[i].burnin_leapfrog_steps = previous_counter;
            profile.elapsed_seconds = std::chrono::duration<double>(chain_end - chain_start).count();
            profile.start_offset_seconds = std::chrono::duration<double>(chain_start - parallel_start).count();
          }
        }
      }
  
};


















 





  
