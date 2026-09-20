
#pragma once

 
#include <stan/model/model_base.hpp>  
 
 
 
#include <stan/io/array_var_context.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/dump.hpp> 
 
 
#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/rapidjson_parser.hpp>  
 
 
#include <sstream>
#include <stdexcept>  
#include <complex>
  
////#include <dlfcn.h> // For dynamic loading 
 
 
#include <map>
#include <vector>   
#include <string>
 
 
 
 
#include <stdexcept> 
#include <stdio.h>
#include <iostream>
 
 
  
#include <Eigen/Dense>
  
  
  
  
// #if __has_include("bridgestan.h")
//     #define HAS_BRIDGESTAN_H 1
//     #include "bridgestan.h" 
//     #include "version.hpp"
//     #include "model_rng.hpp" 
// #else 
//     #define HAS_BRIDGESTAN_H 0
// #endif
//   
//   
//   
  
  
 
 
 
/// using namespace Eigen;
  

using std_vec_of_EigenVecs_dbl = std::vector<Eigen::Matrix<double, -1, 1>>;
using std_vec_of_EigenVecs_int = std::vector<Eigen::Matrix<int, -1, 1>>;
using std_vec_of_EigenMats_dbl = std::vector<Eigen::Matrix<double, -1, -1>>;
using std_vec_of_EigenMats_int = std::vector<Eigen::Matrix<int, -1, -1>>;

using two_layer_std_vec_of_EigenVecs_dbl = std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>;
using two_layer_std_vec_of_EigenVecs_int = std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>;
using two_layer_std_vec_of_EigenMats_dbl = std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>;
using two_layer_std_vec_of_EigenMats_int = std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>;


using three_layer_std_vec_of_EigenVecs_dbl =  std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>>;
using three_layer_std_vec_of_EigenVecs_int =  std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>>;
using three_layer_std_vec_of_EigenMats_dbl = std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>>;
using three_layer_std_vec_of_EigenMats_int = std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>>;


 
 
 
 
// struct for other function arguments to make function signitures more general  easier to manage
// the struct name becomes a return type. So can use as a return argument to functions.
struct   Model_fn_args_struct {
    //// Optional, chain-local diagnostic counter; never used by likelihood calculations.
    //// Counts entered integrator steps (both blocks if partitioned), including a partially completed divergent step.
    int *burnin_leapfrog_steps = nullptr;
    //// Snapshot on the R thread; likelihood workers never read R options.
    bool autodiff_fallback = false;
    bool autodiff_fallback_option_is_set = false;
   
               int N;
               int n_nuisance;
               int n_params_main;
               
               //// ---- Ordinal params:
               int n_binary_tests;
               int n_ordinal_tests;
               
               std::string model_so_file;
               std::string json_file_path;
               
               Eigen::Matrix<bool, -1, 1>        Model_args_bools;       
               Eigen::Matrix<int, -1, 1>         Model_args_ints;       
               Eigen::Matrix<double, -1, 1>      Model_args_doubles;    
               Eigen::Matrix<std::string, -1, 1> Model_args_strings;     
               
               std_vec_of_EigenVecs_dbl  Model_args_col_vecs_double;
               std_vec_of_EigenVecs_int  Model_args_col_vecs_int;
               std_vec_of_EigenMats_dbl  Model_args_mats_double;
               std_vec_of_EigenMats_int  Model_args_mats_int;
               
               two_layer_std_vec_of_EigenVecs_dbl      Model_args_vecs_of_col_vecs_double;
               two_layer_std_vec_of_EigenVecs_int      Model_args_vecs_of_col_vecs_int;
               two_layer_std_vec_of_EigenMats_dbl      Model_args_vecs_of_mats_double; // X goes here for non-LC MVP
               two_layer_std_vec_of_EigenMats_int      Model_args_vecs_of_mats_int;
               
               three_layer_std_vec_of_EigenVecs_dbl   Model_args_2_layer_vecs_of_col_vecs_double ;
               three_layer_std_vec_of_EigenVecs_int   Model_args_2_layer_vecs_of_col_vecs_int ;
               three_layer_std_vec_of_EigenMats_dbl   Model_args_2_layer_vecs_of_mats_double ;/// X goees here  for LC-MVP (4D array)
               three_layer_std_vec_of_EigenMats_int   Model_args_2_layer_vecs_of_mats_int ;
          
         // default constructor
         Model_fn_args_struct(int N, int n_nuisance, int n_params_main, 
                              int n_binary_tests, int n_ordinal_tests, //// ordinal-only
                              int n_bools, int n_ints, int n_doubles, int n_strings,
                              int n_col_vecs_dbl,  int n_col_vecs_int,  int n_mats_dbl,  int n_mats_int,
                              int n_vecs_of_col_vecs_dbl,  int n_vecs_of_col_vecs_int,  int n_vecs_of_mats_dbl,  int n_vecs_of_mats_int,
                              int n_2_layer_vecs_of_col_vecs_dbl,  int n_2_layer_vecs_of_col_vecs_int,  int n_2_layer_vecs_of_mats_dbl,  int n_2_layer_vecs_of_mats_int)
           :
           N(N),
           n_nuisance(n_nuisance),
           n_params_main(n_params_main),
           n_binary_tests(n_binary_tests), //// ordinal-only
           n_ordinal_tests(n_ordinal_tests), //// ordinal-only
           model_so_file("none"),
           json_file_path("none") 
         {
             
               // initialize vectors with default sizes
               Model_args_bools.resize(n_bools);     
               Model_args_ints.resize(n_ints);      
               Model_args_doubles.resize(n_doubles);   
               Model_args_strings.resize(n_strings);  
               
               Model_args_col_vecs_double.resize(n_col_vecs_dbl);
               Model_args_col_vecs_int.resize(n_col_vecs_int);
               Model_args_mats_double.resize(n_mats_dbl);
               Model_args_mats_int.resize(n_mats_int);
               
               Model_args_vecs_of_col_vecs_double.resize(n_vecs_of_col_vecs_dbl);
               Model_args_vecs_of_col_vecs_int.resize(n_vecs_of_col_vecs_int);
               Model_args_vecs_of_mats_double.resize(n_vecs_of_mats_dbl);
               Model_args_vecs_of_mats_int.resize(n_vecs_of_mats_int);
               
               Model_args_2_layer_vecs_of_col_vecs_double.resize(n_2_layer_vecs_of_col_vecs_dbl);
               Model_args_2_layer_vecs_of_col_vecs_int.resize(n_2_layer_vecs_of_col_vecs_int);
               Model_args_2_layer_vecs_of_mats_double.resize(n_2_layer_vecs_of_mats_dbl);
               Model_args_2_layer_vecs_of_mats_int.resize(n_2_layer_vecs_of_mats_int);
          
         }
         
         // constructor w/ default values to handle optional/empty args
         Model_fn_args_struct(
               const int &N_,
               const int &n_nuisance_,
               const int &n_params_main_,
               const int &n_binary_tests_, //// ordinal-only
               const int &n_ordinal_tests_, //// ordinal-only
               const std::string  &model_so_file_ = "none",
               const std::string  &json_file_path_ = "none",
               const Eigen::Matrix<bool, -1, 1>         &Model_args_bools_ = Eigen::Matrix<bool, -1, 1>(),
               const Eigen::Matrix<int, -1, 1>          &Model_args_ints_ = Eigen::Matrix<int, -1, 1>(),
               const Eigen::Matrix<double, -1, 1>       &Model_args_doubles_ = Eigen::Matrix<double, -1, 1>(),
               const Eigen::Matrix<std::string, -1, 1>  &Model_args_strings_ = Eigen::Matrix<std::string, -1, 1>(),
               const std_vec_of_EigenVecs_dbl &Model_args_col_vecs_double_ = std_vec_of_EigenVecs_dbl(),
               const std_vec_of_EigenVecs_int &Model_args_col_vecs_int_ = std_vec_of_EigenVecs_int(),
               const std_vec_of_EigenMats_dbl &Model_args_mats_double_ = std_vec_of_EigenMats_dbl(),
               const std_vec_of_EigenMats_int &Model_args_mats_int_ = std_vec_of_EigenMats_int(),
               const two_layer_std_vec_of_EigenVecs_dbl &Model_args_vecs_of_col_vecs_double_ = two_layer_std_vec_of_EigenVecs_dbl(),
               const two_layer_std_vec_of_EigenVecs_int &Model_args_vecs_of_col_vecs_int_ = two_layer_std_vec_of_EigenVecs_int(),
               const two_layer_std_vec_of_EigenMats_dbl &Model_args_vecs_of_mats_double_ = two_layer_std_vec_of_EigenMats_dbl(),
               const two_layer_std_vec_of_EigenMats_int &Model_args_vecs_of_mats_int_ = two_layer_std_vec_of_EigenMats_int(),
               const three_layer_std_vec_of_EigenVecs_dbl &Model_args_2_layer_vecs_of_col_vecs_double_ = three_layer_std_vec_of_EigenVecs_dbl(),
               const three_layer_std_vec_of_EigenVecs_int &Model_args_2_layer_vecs_of_col_vecs_int_ = three_layer_std_vec_of_EigenVecs_int(),
               const three_layer_std_vec_of_EigenMats_dbl &Model_args_2_layer_vecs_of_mats_double_ = three_layer_std_vec_of_EigenMats_dbl(),
               const three_layer_std_vec_of_EigenMats_int &Model_args_2_layer_vecs_of_mats_int_ = three_layer_std_vec_of_EigenMats_int()
         ) : 
           N(N_),
           n_nuisance(n_nuisance_),
           n_params_main(n_params_main_),
           n_binary_tests(n_binary_tests_), //// ordinal-only
           n_ordinal_tests(n_ordinal_tests_), //// ordinal-only
           model_so_file(model_so_file_),
           json_file_path(json_file_path_),
           Model_args_bools(Model_args_bools_),
           Model_args_ints(Model_args_ints_),
           Model_args_doubles(Model_args_doubles_),
           Model_args_strings(Model_args_strings_),
           Model_args_col_vecs_double(Model_args_col_vecs_double_),
           Model_args_col_vecs_int(Model_args_col_vecs_int_),
           Model_args_mats_double(Model_args_mats_double_),
           Model_args_mats_int(Model_args_mats_int_),
           Model_args_vecs_of_col_vecs_double(Model_args_vecs_of_col_vecs_double_),
           Model_args_vecs_of_col_vecs_int(Model_args_vecs_of_col_vecs_int_),
           Model_args_vecs_of_mats_double(Model_args_vecs_of_mats_double_),
           Model_args_vecs_of_mats_int(Model_args_vecs_of_mats_int_),
           Model_args_2_layer_vecs_of_col_vecs_double(Model_args_2_layer_vecs_of_col_vecs_double_),
           Model_args_2_layer_vecs_of_col_vecs_int(Model_args_2_layer_vecs_of_col_vecs_int_),
           Model_args_2_layer_vecs_of_mats_double(Model_args_2_layer_vecs_of_mats_double_),
           Model_args_2_layer_vecs_of_mats_int(Model_args_2_layer_vecs_of_mats_int_)
         {}
         
};
 
 
 
 

   
 
 
 
 
 
 
 
 // struct for other function arguments to make function signitures more general  easier to manage
 // the struct name becomes a return type. So can use as a return argument to functions.
 struct EHMC_fn_args_struct {
    
    //// for main params
    double tau_main;
    double tau_main_ii;
    double eps_main;
    //// for nuisance params
    double tau_us;
    double tau_us_ii;
    double eps_us;
    
    //// general 
    bool diffusion_HMC;
    //// which diffusion dual integrator ordering to use:
    ////   "kick_flow_kick" (default) or "flow_kick_flow"
    std::string diffusion_HMC_integrator;
    //// BURN-IN ONLY: draw ONE tau_ii per iteration, shared by all chains (set from the R list element
    //// "share_tau_ii_across_chains"; FALSE if absent). Only the trajectory LENGTH is shared - each chain's
    //// momentum still comes from its own RNG stream. See PersistentBurninState::run_chain.
    bool share_tau_ii_across_chains = false;
    //// Optional burn-in KE objective diagnostics; no extra target evaluations or sampling-phase work.
    bool record_kinetic_energy_tau_derivatives = false;
    //// set per iteration by the burn-in worker when share_tau_ii_across_chains is TRUE: the joint (dual)
    //// samplers then use tau_main_ii as given instead of drawing their own. Never set during sampling.
    bool use_given_tau_main_ii = false;
    //// SAMPLING: keep the N x n_iter log-lik trace per chain (set from the R list element "store_log_lik_trace";
    //// TRUE if absent). FALSE when nothing will read it (create_summary_and_traces(save_log_lik_trace = FALSE)):
    //// at N = 10,000, 180 chains, 50 iterations it is 720 MB zero-filled up front and copied into R for nothing.
    bool store_log_lik_trace = true;
    
    /////// constructor
    EHMC_fn_args_struct(
      const double &tau_main_,
      const double &tau_main_ii_,
      const double &eps_main_,
      const double &tau_us_,
      const double &tau_us_ii_,
      const double &eps_us_,
      const bool &diffusion_HMC_,
      const std::string &diffusion_HMC_integrator_ = "kick_flow_kick"
    ) : 
      tau_main(tau_main_),
      tau_main_ii(tau_main_ii_),
      eps_main(eps_main_),
      tau_us(tau_us_),
      tau_us_ii(tau_us_ii_),
      eps_us(eps_us_),
      diffusion_HMC(diffusion_HMC_),
      diffusion_HMC_integrator(diffusion_HMC_integrator_)
    {} 

 }; 
 
  


 
 
 
 
 // struct for other function arguments to make function signitures more general  easier to manage
 // the struct name becomes a return type. So can use as a return argument to functions.
 struct   EHMC_burnin_struct {  // all params in this struct are shared between all chains
   
   // for main params
   double adapt_delta_main; // shared between all chains
   double LR_main; // shared between all chains
   double eps_m_adam_main; // shared between all chains
   double eps_v_adam_main; // shared between all chains
   double tau_m_adam_main; // shared between all chains
   double tau_v_adam_main; // shared between all chains
   double eigen_max_main;   // shared between all chains
   Eigen::VectorXi  index_main; // shared between all chains
   Eigen::Matrix<double, -1, -1> M_dense_sqrt; // shared between all chains
   Eigen::Matrix<double, -1, 1>  snaper_m_vec_main;   // shared between all chains
   Eigen::Matrix<double, -1, 1>  snaper_s_vec_main_empirical;  // shared between all chains
   Eigen::Matrix<double, -1, 1>  snaper_w_vec_main;  // shared between all chains
   Eigen::Matrix<double, -1, 1>  eigen_vector_main;   // shared between all chains
   
   // for nuisance params
   double adapt_delta_us;
   double LR_us;
   double eps_m_adam_us;
   double eps_v_adam_us;
   double tau_m_adam_us;
   double tau_v_adam_us;
   double eigen_max_us;  
   Eigen::VectorXi index_us;
   Eigen::Matrix<double, -1, 1>  sqrt_M_us_vec;
   Eigen::Matrix<double, -1, 1>  snaper_m_vec_us; 
   Eigen::Matrix<double, -1, 1>  snaper_s_vec_us_empirical; 
   Eigen::Matrix<double, -1, 1>  snaper_w_vec_us;  
   Eigen::Matrix<double, -1, 1>  eigen_vector_us;   
   
   /////// constructor
   EHMC_burnin_struct(
      ///// main params
      double adapt_delta_main_,
      double LR_main_,
      double eps_m_adam_main_,
      double eps_v_adam_main_,
      double tau_m_adam_main_,
      double tau_v_adam_main_,
      double eigen_max_main_,  
      Eigen::VectorXi  index_main_,
      Eigen::Matrix<double, -1, -1> M_dense_sqrt_,
      Eigen::Matrix<double, -1, 1>  snaper_m_vec_main_,
      Eigen::Matrix<double, -1, 1>  snaper_s_vec_main_empirical_,
      Eigen::Matrix<double, -1, 1>  snaper_w_vec_main_,
      Eigen::Matrix<double, -1, 1>  eigen_vector_main_,   
      ///// nuisance
      double adapt_delta_us_,
      double LR_us_,
      double eps_m_adam_us_,
      double eps_v_adam_us_,
      double tau_m_adam_us_,
      double tau_v_adam_us_,
      double eigen_max_us_,  
      Eigen::VectorXi  index_us_,
      Eigen::Matrix<double, -1, 1>  sqrt_M_us_vec_,
      Eigen::Matrix<double, -1, 1>  snaper_m_vec_us_, 
      Eigen::Matrix<double, -1, 1>  snaper_s_vec_us_empirical_, 
      Eigen::Matrix<double, -1, 1>  snaper_w_vec_us_,
      Eigen::Matrix<double, -1, 1>  eigen_vector_us_   
   ) :  
     ///// main params
     adapt_delta_main(adapt_delta_main_),
     LR_main(LR_main_),
     eps_m_adam_main(eps_m_adam_main_),
     eps_v_adam_main(eps_v_adam_main_),
     tau_m_adam_main(tau_m_adam_main_),
     tau_v_adam_main(tau_v_adam_main_),
     eigen_max_main(eigen_max_main_),   
     index_main(index_main_),
     M_dense_sqrt(M_dense_sqrt_),
     snaper_m_vec_main(snaper_m_vec_main_),
     snaper_s_vec_main_empirical(snaper_s_vec_main_empirical_),
     snaper_w_vec_main(snaper_w_vec_main_),
     eigen_vector_main(eigen_vector_main_),  
     /////// nuisance
     adapt_delta_us(adapt_delta_us_),
     LR_us(LR_us_),
     eps_m_adam_us(eps_m_adam_us_),
     eps_v_adam_us(eps_v_adam_us_),
     tau_m_adam_us(tau_m_adam_us_),
     tau_v_adam_us(tau_v_adam_us_),
     eigen_max_us(eigen_max_us_),   
     index_us(index_us_),
     sqrt_M_us_vec(sqrt_M_us_vec_),
     snaper_m_vec_us(snaper_m_vec_us_),
     snaper_s_vec_us_empirical(snaper_s_vec_us_empirical_),
     snaper_w_vec_us(snaper_w_vec_us_),
     eigen_vector_us(eigen_vector_us_)   
     {} 
   
 };  
 
 
 
 
 
 
 
 
 // struct for other function arguments to make function signitures more general  easier to manage
 // the struct name becomes a return type. So can use as a return argument to functions.
 
//// we dont want to modify any of these so can use const & throughout the struct. 
struct  EHMC_Metric_struct { // all params in this struct are shared between all chains
   
   //// for main params
   Eigen::Matrix<double, -1, -1> M_dense_main; // using & here AND in the constructor (bit below) is very efficient but have to be careful when calling constructor 
   Eigen::Matrix<double, -1, -1> M_inv_dense_main;
   Eigen::Matrix<double, -1, -1> M_inv_dense_main_chol;
   Eigen::Matrix<double, -1, 1>  M_inv_main_vec;
   //// for nuisance params
   Eigen::Matrix<double, -1, 1>  M_inv_us_vec;
   Eigen::Matrix<double, -1, 1>  M_us_vec;
   Eigen::Matrix<double, -1, 1>  theta_hat_us_vec;
   
   std::string metric_shape_main;
   
   /////// constructor w/ eevrything (some can be empty)
   EHMC_Metric_struct(
     Eigen::Matrix<double, -1, -1>  M_dense_main_,
     Eigen::Matrix<double, -1, -1>  M_inv_dense_main_,
     Eigen::Matrix<double, -1, -1>  M_inv_dense_main_chol_,
     Eigen::Matrix<double, -1, 1>   M_inv_main_vec_,
     Eigen::Matrix<double, -1, 1>   M_inv_us_vec_,
     Eigen::Matrix<double, -1, 1>   M_us_vec_,
     Eigen::Matrix<double, -1, 1>   theta_hat_us_vec_,
     std::string  metric_shape_main_
   ) : 
     M_dense_main(M_dense_main_),
     M_inv_dense_main(M_inv_dense_main_),
     M_inv_dense_main_chol(M_inv_dense_main_chol_),
     M_inv_main_vec(M_inv_main_vec_),
     M_inv_us_vec(M_inv_us_vec_),
     M_us_vec(M_us_vec_),
     theta_hat_us_vec(theta_hat_us_vec_),
     metric_shape_main(metric_shape_main_)
   {}
   
}; 
 
 
 
 
 
 
 
 



struct ChunkSizeInfo {
  
        int chunk_size;
        int chunk_size_orig;
        int normal_chunk_size;
        int last_chunk_size;
        int n_total_chunks;
        int n_full_chunks;
        
};


// ChunkSizeInfo calculate_chunk_sizes(const int N,
//                                     const int vec_size,
//                                     const int desired_n_chunks) {
// 
//         ChunkSizeInfo info;
// 
//         if (desired_n_chunks == 1) {
// 
//               info.chunk_size = N;
//               info.chunk_size_orig = N;
//               info.normal_chunk_size = N;
//               info.last_chunk_size = N;
//               info.n_total_chunks = 1;
//               info.n_full_chunks = 1;
// 
//               return info;
// 
//         }
// 
//         const double N_double = static_cast<double>(N);
//         const double vec_size_double = static_cast<double>(vec_size);
//         const double desired_n_chunks_double = static_cast<double>(desired_n_chunks);
// 
//         info.normal_chunk_size = vec_size_double * std::floor(N_double / (vec_size_double * desired_n_chunks_double));
//         info.n_full_chunks = std::floor(N_double / static_cast<double>(info.normal_chunk_size));
//         info.last_chunk_size = N_double - (static_cast<double>(info.n_full_chunks) * static_cast<double>(info.normal_chunk_size));
// 
//         info.n_total_chunks = (info.last_chunk_size == 0) ? info.n_full_chunks : info.n_full_chunks + 1;
// 
//         info.chunk_size = info.normal_chunk_size;
//         info.chunk_size_orig = info.normal_chunk_size;
// 
//         return info;
// 
// }


// ChunkSizeInfo calculate_chunk_sizes(const int N,
//                                     const int vec_size,
//                                     const int desired_n_chunks) {
// 
//   ChunkSizeInfo info;
// 
//   if (desired_n_chunks == 1) {
// 
//     info.normal_chunk_size = (N / vec_size) * vec_size;  // round DOWN to multiple of vec_size
//     info.last_chunk_size = N - info.normal_chunk_size;
// 
//     if (info.last_chunk_size == 0) {
//       info.n_full_chunks = 1;
//       info.n_total_chunks = 1;
//     } else {
//       info.n_full_chunks = 1;
//       info.n_total_chunks = 2;
//     }
// 
//     info.chunk_size = info.normal_chunk_size;
//     info.chunk_size_orig = info.normal_chunk_size;
// 
//     return info;
// 
//   }
// 
//   const double N_double = static_cast<double>(N);
//   const double vec_size_double = static_cast<double>(vec_size);
//   const double desired_n_chunks_double = static_cast<double>(desired_n_chunks);
// 
//   info.normal_chunk_size = vec_size_double * std::floor(N_double / (vec_size_double * desired_n_chunks_double));
//   info.n_full_chunks = std::floor(N_double / static_cast<double>(info.normal_chunk_size));
//   info.last_chunk_size = N_double - (static_cast<double>(info.n_full_chunks) * static_cast<double>(info.normal_chunk_size));
// 
//   info.n_total_chunks = (info.last_chunk_size == 0) ? info.n_full_chunks : info.n_full_chunks + 1;
// 
//   info.chunk_size = info.normal_chunk_size;
//   info.chunk_size_orig = info.normal_chunk_size;
// 
//   return info;
// 
// }

ChunkSizeInfo calculate_chunk_sizes(const int N, 
                                    const int vec_size, 
                                    const int desired_n_chunks) {
  
        ChunkSizeInfo info;
        
        const int effective_chunks = std::max(desired_n_chunks, 1);
        
        info.normal_chunk_size = vec_size * (N / (vec_size * effective_chunks));
        
        if (info.normal_chunk_size == 0) {
          // N smaller than vec_size * effective_chunks
          info.normal_chunk_size = (N / vec_size) * vec_size;
          if (info.normal_chunk_size == 0) {
            // N < vec_size, everything goes through scalar
            info.n_full_chunks = 0;
            info.last_chunk_size = N;
            info.n_total_chunks = 1;
            info.chunk_size = N;
            info.chunk_size_orig = N;
            return info;
          }
        }
        
        info.n_full_chunks = N / info.normal_chunk_size;
        info.last_chunk_size = N - (info.n_full_chunks * info.normal_chunk_size);
        info.n_total_chunks = (info.last_chunk_size == 0) ? info.n_full_chunks : info.n_full_chunks + 1;
        
        info.chunk_size = info.normal_chunk_size;
        info.chunk_size_orig = info.normal_chunk_size;
        
        return info;
  
}



// ChunkSizeInfo calculate_chunk_sizes(const int N, 
//                                     const int vec_size, 
//                                     const int desired_n_chunks) {
//   
//       ChunkSizeInfo info;
//       
//       const int effective_chunks = std::max(desired_n_chunks, 1);
//       
//       info.normal_chunk_size = vec_size * (N / (vec_size * effective_chunks));
//       
//       if (info.normal_chunk_size == 0) {
//         // N too small for even one SIMD-aligned chunk
//         info.normal_chunk_size = N;
//         info.n_full_chunks = 0;
//         info.last_chunk_size = N;
//         info.n_total_chunks = 1;
//       } else {
//         info.n_full_chunks = N / info.normal_chunk_size;
//         info.last_chunk_size = N - (info.n_full_chunks * info.normal_chunk_size);
//         info.n_total_chunks = (info.last_chunk_size == 0) ? info.n_full_chunks : info.n_full_chunks + 1;
//       }
//       
//       info.chunk_size = info.normal_chunk_size;
//       info.chunk_size_orig = info.normal_chunk_size;
//       
//       return info;
//       
// }


// 
// 
// 
// 
// struct LC_MVP_workspace_struct {
//   
//   // Constructors
//   LC_MVP_workspace_struct() = default;
//   
//   LC_MVP_workspace_struct(int chunk_size, 
//                           int n_tests, 
//                           int n_class,
//                           int n_covariates_max,
//                           int n_corrs, 
//                           int n_covariates_total) {
//     
//           allocate(chunk_size, 
//                    n_tests, 
//                    n_class, 
//                    n_covariates_max, 
//                    n_corrs, 
//                    n_covariates_total);
//     
//   }
//   
//   void reset() {
//         
//         // // First restore sizes if they were shrunk by last-chunk resize
//         if (is_allocated && y1_log_prob.rows() != allocated_chunk_size) {
//           
//               allocate(allocated_chunk_size, 
//                        stored_n_tests, 
//                        stored_n_class,
//                        stored_n_covariates_max, 
//                        stored_n_corrs, 
//                        stored_n_covariates_total);
//                     
//               return;  // allocate already zeros everything
//           
//         }
//         
//   }
//   
//   void reset_sizes() {
//     if (is_allocated && y1_log_prob.rows() != allocated_chunk_size) {
//       restore_sizes();
//     }
//   }
//   
//   //// Always size 2 — 1-class model just uses [0]
//   ///////////////////////////////////////////////
//   std::array<Eigen::Matrix<double, -1, -1>, 2> Z_std_norm;
//   std::array<Eigen::Matrix<double, -1, -1>, 2> Bound_Z;
//   std::array<Eigen::Matrix<double, -1, -1>, 2> Bound_U_Phi_Bound_Z;
//   std::array<Eigen::Matrix<double, -1, -1>, 2> prob;
//   std::array<Eigen::Matrix<double, -1, -1>, 2> Phi_Z;
//   ///////////////////////////////////////////////
//   Eigen::Matrix<double, -1, -1> y1_log_prob;
//   Eigen::Matrix<double, -1, -1> phi_Z_recip;
//   Eigen::Matrix<double, -1, -1> phi_Bound_Z;
//   ///////////////////////////////////////////////
//   Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk;
//   ///////////////////////////////////////////////
//   Eigen::Matrix<double, -1, -1> common_grad_term_1;
//   Eigen::Matrix<double, -1, -1> y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip;
//   Eigen::Matrix<double, -1, -1> y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip;
//   Eigen::Matrix<double, -1, -1> prob_rowwise_prod_temp;
//   Eigen::Matrix<double, -1, -1> prob_recip_rowwise_prod_temp;
//   ///////////////////////////////////////////////
//   Eigen::Matrix<double, -1, 1> prod_container_or_inc_array;
//   Eigen::Matrix<double, -1, 1> derivs_chain_container_vec;
//   Eigen::Matrix<double, -1, 1> prob_rowwise_prod_temp_all;
//   ///////////////////////////////////////////////
//   Eigen::Matrix<double, -1, -1> grad_prob;
//   Eigen::Matrix<double, -1, -1> z_grad_term;
//   ///////////////////////////////////////////////
//   Eigen::Matrix<double, -1, -1> y_chunk;
//   Eigen::Matrix<double, -1, -1> u_array;
//   Eigen::Matrix<double, -1, -1> y_sign;
//   Eigen::Matrix<double, -1, -1> y_m_y_sign_x_u;
//   ///////////////////////////////////////////////
//   Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk_block;
//   ///////////////////////////////////////////////
//   Eigen::Matrix<double, -1, 1> u_unc_vec_chunk;
//   Eigen::Matrix<double, -1, 1> u_vec_chunk;
//   Eigen::Matrix<double, -1, 1> du_wrt_duu_chunk;
//   Eigen::Matrix<double, -1, 1> d_J_wrt_duu_chunk;
//   ///////////////////////////////////////////////
//   Eigen::Matrix<double, -1, -1> lp_array;
//   ///////////////////////////////////////////////
//   Eigen::Matrix<double, -1, 1> prob_n;
//   Eigen::Matrix<double, -1, 1> prob_n_recip;
//   Eigen::Matrix<double, -1, 1> log_sum_result;
//   Eigen::Matrix<double, -1, 1> container_max_logs;
//   ///////////////////////////////////////////////
//   Eigen::Matrix<double, -1, 1> rowwise_log_sum;
//   Eigen::Matrix<double, -1, 1> rowwise_prod;
//   Eigen::Matrix<double, -1, 1> rowwise_sum;
//   Eigen::Matrix<double, -1, 1> log_lik_chunk;
//   ///////////////////////////////////////////////
//   Eigen::Matrix<double, -1, -1> prob_recip;
//   ///////////////////////////////////////////////
//   std::vector<Eigen::Matrix<double, -1, -1>> Upper_Bound_Z;        // for ordinal (MVOP) - (chunk_size × n_tests)
//   ///////////////////////////////////////////////
//   Eigen::Matrix<double, -1, -1> phi_Upper_Bound_Z;    // for ordinal (MVOP) - (chunk_size × n_tests)
//   Eigen::Matrix<double, -1, -1> dphi_over_L;          // for ordinal (MVOP) - (chunk_size × n_tests)
//   Eigen::Matrix<double, -1, -1> dZ_dmu_neg;           // for ordinal (MVOP) - (chunk_size × n_tests)
//   
//   // ---- tracking ----
//   bool is_allocated = false;
//   int allocated_chunk_size = 0;
//   
//   int stored_n_tests = 0;
//   int stored_n_class = 0;
//   int stored_n_covariates_max = 0;
//   int stored_n_corrs = 0;
//   int stored_n_covariates_total = 0;
//   
//   void allocate(int chunk_size,
//                 int n_tests, 
//                 int n_class, 
//                 int n_covariates_max,
//                 int n_corrs, 
//                 int n_covariates_total) {
//     
//             stored_n_tests = n_tests;
//             stored_n_class = n_class;
//             stored_n_covariates_max = n_covariates_max;
//             stored_n_corrs = n_corrs;
//             stored_n_covariates_total = n_covariates_total;
// 
//             const int dim_choose_2 = n_tests * (n_tests - 1) / 2;
//             
//             ////////////////////////////////////////////////
//             Z_std_norm =          array_of_mats<double, 2>(chunk_size, n_tests);
//             Bound_Z =             array_of_mats<double, 2>(chunk_size, n_tests);
//             Bound_U_Phi_Bound_Z = array_of_mats<double, 2>(chunk_size, n_tests);
//             prob =                array_of_mats<double, 2>(chunk_size, n_tests);
//             Phi_Z =               array_of_mats<double, 2>(chunk_size, n_tests);
//             ////////////////////////////////////////////////
//             y1_log_prob =            Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             phi_Z_recip =            Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             phi_Bound_Z =            Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             ////////////////////////////////////////////////
//             u_grad_array_CM_chunk =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             ////////////////////////////////////////////////
//             common_grad_term_1 =     Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             prob_rowwise_prod_temp =         Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             prob_recip_rowwise_prod_temp =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             ////////////////////////////////////////////////
//             prod_container_or_inc_array =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
//             derivs_chain_container_vec =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
//             prob_rowwise_prod_temp_all =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
//             ////////////////////////////////////////////////
//             grad_prob =              Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             z_grad_term =            Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             ////////////////////////////////////////////////
//             y_chunk =                Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             u_array =                Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             y_sign =                 Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             y_m_y_sign_x_u =         Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             ////////////////////////////////////////////////
//             u_grad_array_CM_chunk_block = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             ////////////////////////////////////////////////
//             u_unc_vec_chunk =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
//             u_vec_chunk =        Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
//             du_wrt_duu_chunk =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
//             d_J_wrt_duu_chunk =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
//             ////////////////////////////////////////////////
//             lp_array =               Eigen::Matrix<double, -1, -1>::Zero(chunk_size, 2);
//             ////////////////////////////////////////////////
//             prob_n =                       Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
//             prob_n_recip =                 Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
//             log_sum_result =               Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
//             container_max_logs =           Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
//             ////////////////////////////////////////////////
//             rowwise_log_sum =              Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
//             rowwise_prod =                 Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
//             rowwise_sum =                  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
//             log_lik_chunk =                Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
//             ////////////////////////////////////////////////
//             prob_recip =             Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
//             ////////////////////////////////////////////////
//             Upper_Bound_Z     = vec_of_mats<double>(chunk_size, n_tests, 2); // ordinal-only
//             ////////////////////////////////////////////////
//             phi_Upper_Bound_Z = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests); // ordinal-only
//             dphi_over_L       = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests); // ordinal-only
//             dZ_dmu_neg        = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests); // ordinal-only
//             ////////////////////////////////////////////////
//         
//             is_allocated = true;
//             allocated_chunk_size = chunk_size;
//     
//   }
//   
//   void restore_sizes() {
//             
//             ////////////////////////////////////////////////
//             for (int c = 0; c < 2; c++) {
//               Z_std_norm[c].resize(allocated_chunk_size, stored_n_tests);
//               Bound_Z[c].resize(allocated_chunk_size, stored_n_tests);
//               Bound_U_Phi_Bound_Z[c].resize(allocated_chunk_size, stored_n_tests);
//               prob[c].resize(allocated_chunk_size, stored_n_tests);
//               Phi_Z[c].resize(allocated_chunk_size, stored_n_tests);
//             }
//             ////////////////////////////////////////////////
//             y1_log_prob.resize(allocated_chunk_size, stored_n_tests);
//             phi_Z_recip.resize(allocated_chunk_size, stored_n_tests);
//             phi_Bound_Z.resize(allocated_chunk_size, stored_n_tests);
//             ////////////////////////////////////////////////
//             u_grad_array_CM_chunk.resize(allocated_chunk_size, stored_n_tests);
//             ////////////////////////////////////////////////
//             common_grad_term_1.resize(allocated_chunk_size, stored_n_tests);
//             y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(allocated_chunk_size, stored_n_tests);
//             y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(allocated_chunk_size, stored_n_tests);
//             prob_rowwise_prod_temp.resize(allocated_chunk_size, stored_n_tests);
//             prob_recip_rowwise_prod_temp.resize(allocated_chunk_size, stored_n_tests);
//             ////////////////////////////////////////////////
//             prod_container_or_inc_array.resize(allocated_chunk_size);
//             derivs_chain_container_vec.resize(allocated_chunk_size);
//             prob_rowwise_prod_temp_all.resize(allocated_chunk_size);
//             ////////////////////////////////////////////////
//             grad_prob.resize(allocated_chunk_size, stored_n_tests);
//             z_grad_term.resize(allocated_chunk_size, stored_n_tests);
//             ////////////////////////////////////////////////
//             y_chunk.resize(allocated_chunk_size, stored_n_tests);
//             u_array.resize(allocated_chunk_size, stored_n_tests);
//             y_sign.resize(allocated_chunk_size, stored_n_tests);
//             y_m_y_sign_x_u.resize(allocated_chunk_size, stored_n_tests);
//             ////////////////////////////////////////////////
//             u_grad_array_CM_chunk_block.resize(allocated_chunk_size, stored_n_tests);
//             ////////////////////////////////////////////////
//             u_unc_vec_chunk.resize(allocated_chunk_size * stored_n_tests);
//             u_vec_chunk.resize(allocated_chunk_size * stored_n_tests);
//             du_wrt_duu_chunk.resize(allocated_chunk_size * stored_n_tests);
//             d_J_wrt_duu_chunk.resize(allocated_chunk_size * stored_n_tests);
//             ////////////////////////////////////////////////
//             lp_array.resize(allocated_chunk_size, 2);
//             ////////////////////////////////////////////////
//             prob_n.resize(allocated_chunk_size);
//             prob_n_recip.resize(allocated_chunk_size);
//             log_sum_result.resize(allocated_chunk_size);
//             container_max_logs.resize(allocated_chunk_size);
//             ////////////////////////////////////////////////
//             rowwise_log_sum.resize(allocated_chunk_size);
//             rowwise_prod.resize(allocated_chunk_size);
//             rowwise_sum.resize(allocated_chunk_size);
//             log_lik_chunk.resize(allocated_chunk_size);
//             ////////////////////////////////////////////////
//             prob_recip.resize(allocated_chunk_size, stored_n_tests);
//             ////////////////////////////////////////////////
//             for (int c = 0; c < 2; c++) {
//                Upper_Bound_Z[c].resize(allocated_chunk_size, stored_n_tests);
//             }
//             ////////////////////////////////////////////////
//             phi_Upper_Bound_Z.resize(allocated_chunk_size, stored_n_tests);
//             dphi_over_L.resize(allocated_chunk_size, stored_n_tests);
//             dZ_dmu_neg.resize(allocated_chunk_size, stored_n_tests);
//     
//   } 
//   
// };
// 
// 
// 












struct LC_MVP_workspace_struct {
  
  // Constructors
  LC_MVP_workspace_struct() = default;
  
  LC_MVP_workspace_struct(int chunk_size, 
                          int n_tests, 
                          int n_class,
                          int n_covariates_max,
                          int n_corrs, 
                          int n_covariates_total) {
    
    allocate(chunk_size, 
             n_tests, 
             n_class, 
             n_covariates_max, 
             n_corrs, 
             n_covariates_total);
    
  } 
  
  void reset() {
    
    // // First restore sizes if they were shrunk by last-chunk resize
    if (is_allocated && y1_log_prob.rows() != allocated_chunk_size) {
      
      allocate(allocated_chunk_size, 
               stored_n_tests, 
               stored_n_class,
               stored_n_covariates_max, 
               stored_n_corrs, 
               stored_n_covariates_total);
       
      return;  // allocate already zeros everything
      
    }
    
  }
  
  void reset_sizes() {
    if (is_allocated && y1_log_prob.rows() != allocated_chunk_size) {
      restore_sizes();
    } 
  }
  
  //// Always size 2 — 1-class model just uses [0]
  ///////////////////////////////////////////////
  std::vector<Eigen::Matrix<double, -1, -1>> Z_std_norm;
  std::vector<Eigen::Matrix<double, -1, -1>> Bound_Z;
  std::vector<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z;
  std::vector<Eigen::Matrix<double, -1, -1>> prob;
  std::vector<Eigen::Matrix<double, -1, -1>> Phi_Z;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> y1_log_prob;
  Eigen::Matrix<double, -1, -1> phi_Z_recip;
  Eigen::Matrix<double, -1, -1> phi_Bound_Z;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> common_grad_term_1;
  Eigen::Matrix<double, -1, -1> y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip;
  Eigen::Matrix<double, -1, -1> y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip;
  Eigen::Matrix<double, -1, -1> prob_rowwise_prod_temp;
  Eigen::Matrix<double, -1, -1> prob_recip_rowwise_prod_temp;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> prod_container_or_inc_array;
  Eigen::Matrix<double, -1, 1> derivs_chain_container_vec;
  Eigen::Matrix<double, -1, 1> prob_rowwise_prod_temp_all;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> dphi_direct_Cj; // ordinal-only
  Eigen::Matrix<double, -1, 1> dZ_direct_Cj; // ordinal-only
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> grad_prob;
  Eigen::Matrix<double, -1, -1> z_grad_term;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> y_chunk;
  Eigen::Matrix<double, -1, -1> u_array;
  Eigen::Matrix<double, -1, -1> y_sign;
  Eigen::Matrix<double, -1, -1> y_m_y_sign_x_u;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk_block;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> u_unc_vec_chunk;
  Eigen::Matrix<double, -1, 1> u_vec_chunk;
  Eigen::Matrix<double, -1, 1> du_wrt_duu_chunk;
  Eigen::Matrix<double, -1, 1> d_J_wrt_duu_chunk;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> lp_array;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> prob_n;
  Eigen::Matrix<double, -1, 1> prob_n_recip;
  Eigen::Matrix<double, -1, 1> log_sum_result;
  Eigen::Matrix<double, -1, 1> container_max_logs;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> rowwise_log_sum;
  Eigen::Matrix<double, -1, 1> rowwise_prod;
  Eigen::Matrix<double, -1, 1> rowwise_sum;
  Eigen::Matrix<double, -1, 1> log_lik_chunk;
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> prob_recip;
  ///////////////////////////////////////////////
  std::vector<Eigen::Matrix<double, -1, -1>> Upper_Bound_Z;        // for ordinal (MVOP) - (chunk_size × n_tests)
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> phi_Upper_Bound_Z;    // for ordinal (MVOP) - (chunk_size × n_tests)
  Eigen::Matrix<double, -1, -1> dphi_over_L;          // for ordinal (MVOP) - (chunk_size × n_tests)
  Eigen::Matrix<double, -1, -1> dZ_dmu_neg;           // for ordinal (MVOP) - (chunk_size × n_tests)
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, -1> dphi_times_bz;        // for ordinal (MVOP) - (chunk_size × n_tests)
  Eigen::Matrix<double, -1, -1> dZ_times_bz;          // for ordinal (MVOP) - (chunk_size × n_tests)
  ///////////////////////////////////////////////
  Eigen::Matrix<double, -1, 1> prev_per_obs_given_c;
  Eigen::Matrix<double, -1, 1> log_prev_per_obs_given_c;
  ///////////////////////////////////////////////
  
  // ---- tracking ----
  bool is_allocated = false;
  int allocated_chunk_size = 0;
  
  int stored_n_tests = 0;
  int stored_n_class = 0;
  int stored_n_covariates_max = 0;
  int stored_n_corrs = 0;
  int stored_n_covariates_total = 0;
  
  void allocate(int chunk_size,
                int n_tests, 
                int n_class, 
                int n_covariates_max,
                int n_corrs, 
                int n_covariates_total) {
    
    stored_n_tests = n_tests;
    stored_n_class = n_class;
    stored_n_covariates_max = n_covariates_max;
    stored_n_corrs = n_corrs;
    stored_n_covariates_total = n_covariates_total;
    
    const int dim_choose_2 = n_tests * (n_tests - 1) / 2;
    
    ////////////////////////////////////////////////
    Z_std_norm =          vec_of_mats<double>(chunk_size, n_tests, 2);
    Bound_Z =             vec_of_mats<double>(chunk_size, n_tests, 2);
    Bound_U_Phi_Bound_Z = vec_of_mats<double>(chunk_size, n_tests, 2);
    prob =                vec_of_mats<double>(chunk_size, n_tests, 2);
    Phi_Z =               vec_of_mats<double>(chunk_size, n_tests, 2);
    ////////////////////////////////////////////////
    y1_log_prob =            Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    phi_Z_recip =            Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    phi_Bound_Z =            Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ////////////////////////////////////////////////
    u_grad_array_CM_chunk =  Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ////////////////////////////////////////////////
    common_grad_term_1 =     Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    prob_rowwise_prod_temp =         Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    prob_recip_rowwise_prod_temp =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ////////////////////////////////////////////////
    prod_container_or_inc_array =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    derivs_chain_container_vec =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    prob_rowwise_prod_temp_all =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    ////////////////////////////////////////////////
    dphi_direct_Cj = Eigen::Matrix<double, -1, 1>::Zero(chunk_size); // ordinal-only
    dZ_direct_Cj =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size); // ordinal-only
    ////////////////////////////////////////////////
    grad_prob =              Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    z_grad_term =            Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ////////////////////////////////////////////////
    y_chunk =                Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    u_array =                Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    y_sign =                 Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    y_m_y_sign_x_u =         Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ////////////////////////////////////////////////
    u_grad_array_CM_chunk_block = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ////////////////////////////////////////////////
    u_unc_vec_chunk =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
    u_vec_chunk =        Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
    du_wrt_duu_chunk =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
    d_J_wrt_duu_chunk =  Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
    ////////////////////////////////////////////////
    lp_array =               Eigen::Matrix<double, -1, -1>::Zero(chunk_size, 2);
    ////////////////////////////////////////////////
    prob_n =                       Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    prob_n_recip =                 Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    log_sum_result =               Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    container_max_logs =           Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    ////////////////////////////////////////////////
    rowwise_log_sum =              Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    rowwise_prod =                 Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    rowwise_sum =                  Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    log_lik_chunk =                Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    ////////////////////////////////////////////////
    prob_recip =             Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    ////////////////////////////////////////////////
    Upper_Bound_Z     = vec_of_mats<double>(chunk_size, n_tests, 2); // ordinal-only
    ////////////////////////////////////////////////
    phi_Upper_Bound_Z = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests); // ordinal-only
    dphi_over_L       = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests); // ordinal-only
    dZ_dmu_neg        = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests); // ordinal-only
    //////////////////////////////////////////////// 
    dphi_times_bz     = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests); // ordinal-only
    dZ_times_bz       = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests); // ordinal-only
    //////////////////////////////////////////////// 
    log_prev_per_obs_given_c  = Eigen::Matrix<double, -1, 1>::Zero(chunk_size); 
    prev_per_obs_given_c      = Eigen::Matrix<double, -1, 1>::Zero(chunk_size); 
    //////////////////////////////////////////////// 
    
    is_allocated = true;
    allocated_chunk_size = chunk_size;
    
  } 
  
  void restore_sizes() {
    
    ////////////////////////////////////////////////
    for (int c = 0; c < 2; c++) { 
      Z_std_norm[c].resize(allocated_chunk_size, stored_n_tests);
      Bound_Z[c].resize(allocated_chunk_size, stored_n_tests);
      Bound_U_Phi_Bound_Z[c].resize(allocated_chunk_size, stored_n_tests);
      prob[c].resize(allocated_chunk_size, stored_n_tests);
      Phi_Z[c].resize(allocated_chunk_size, stored_n_tests);
    }
    ////////////////////////////////////////////////
    y1_log_prob.resize(allocated_chunk_size, stored_n_tests);
    phi_Z_recip.resize(allocated_chunk_size, stored_n_tests);
    phi_Bound_Z.resize(allocated_chunk_size, stored_n_tests);
    ////////////////////////////////////////////////
    u_grad_array_CM_chunk.resize(allocated_chunk_size, stored_n_tests);
    ////////////////////////////////////////////////
    common_grad_term_1.resize(allocated_chunk_size, stored_n_tests);
    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(allocated_chunk_size, stored_n_tests);
    y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(allocated_chunk_size, stored_n_tests);
    prob_rowwise_prod_temp.resize(allocated_chunk_size, stored_n_tests);
    prob_recip_rowwise_prod_temp.resize(allocated_chunk_size, stored_n_tests);
    ////////////////////////////////////////////////
    prod_container_or_inc_array.resize(allocated_chunk_size);
    derivs_chain_container_vec.resize(allocated_chunk_size);
    prob_rowwise_prod_temp_all.resize(allocated_chunk_size);
    ////////////////////////////////////////////////
    dphi_direct_Cj.resize(allocated_chunk_size); // ordinal-only
    dZ_direct_Cj.resize(allocated_chunk_size); // ordinal-only
    ////////////////////////////////////////////////
    grad_prob.resize(allocated_chunk_size, stored_n_tests);
    z_grad_term.resize(allocated_chunk_size, stored_n_tests);
    ////////////////////////////////////////////////
    y_chunk.resize(allocated_chunk_size, stored_n_tests);
    u_array.resize(allocated_chunk_size, stored_n_tests);
    y_sign.resize(allocated_chunk_size, stored_n_tests);
    y_m_y_sign_x_u.resize(allocated_chunk_size, stored_n_tests);
    ////////////////////////////////////////////////
    u_grad_array_CM_chunk_block.resize(allocated_chunk_size, stored_n_tests);
    ////////////////////////////////////////////////
    u_unc_vec_chunk.resize(allocated_chunk_size * stored_n_tests);
    u_vec_chunk.resize(allocated_chunk_size * stored_n_tests);
    du_wrt_duu_chunk.resize(allocated_chunk_size * stored_n_tests);
    d_J_wrt_duu_chunk.resize(allocated_chunk_size * stored_n_tests);
    ////////////////////////////////////////////////
    lp_array.resize(allocated_chunk_size, 2);
    ////////////////////////////////////////////////
    prob_n.resize(allocated_chunk_size);
    prob_n_recip.resize(allocated_chunk_size);
    log_sum_result.resize(allocated_chunk_size);
    container_max_logs.resize(allocated_chunk_size);
    ////////////////////////////////////////////////
    rowwise_log_sum.resize(allocated_chunk_size);
    rowwise_prod.resize(allocated_chunk_size);
    rowwise_sum.resize(allocated_chunk_size);
    log_lik_chunk.resize(allocated_chunk_size);
    ////////////////////////////////////////////////
    prob_recip.resize(allocated_chunk_size, stored_n_tests);
    ////////////////////////////////////////////////
    for (int c = 0; c < 2; c++) {
      Upper_Bound_Z[c].resize(allocated_chunk_size, stored_n_tests); // ordinal-only
    }
    ////////////////////////////////////////////////
    phi_Upper_Bound_Z.resize(allocated_chunk_size, stored_n_tests); // ordinal-only
    dphi_over_L.resize(allocated_chunk_size, stored_n_tests); // ordinal-only
    dZ_dmu_neg.resize(allocated_chunk_size, stored_n_tests); // ordinal-only
    ////////////////////////////////////////////////
    dphi_times_bz.resize(allocated_chunk_size, stored_n_tests); // ordinal-only
    dZ_times_bz.resize(allocated_chunk_size, stored_n_tests); // ordinal-only
    //////////////////////////////////////////////// 
    log_prev_per_obs_given_c.resize(allocated_chunk_size);
    prev_per_obs_given_c.resize(allocated_chunk_size);
    //////////////////////////////////////////////
    
  }
  
};






























  
