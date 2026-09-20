

//// #include <Rcpp.h>

#define using_dqrng 1
////
#define RNG_TYPE_CPP_STD                  0
////
#define RNG_TYPE_pcg32                    0
#define RNG_TYPE_pcg54                    0
////
#define RNG_TYPE_dqrng_pcg64              0
#define RNG_TYPE_dqrng_xoshiro256plusplus 1

// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp17)]]

#include <iostream>
#include <sstream>
#include <ostream> 
#include <stdexcept>    
#include <complex>
#include <vector>   
#include <stdio.h>  
#include <algorithm>
#include <iomanip>
#include <string>
#include <map>
//// #include <random> 
#include <cmath>

// (dqrng, BH, sitmo)

#if using_dqrng == 1
//[[Rcpp::depends(dqrng)]]
#endif


#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE


#define ENABLE_OPEN_MP 1
#if ENABLE_OPEN_MP == 1
// [[Rcpp::plugins(openmp)]]
  #if __has_include("omp.h")
      #include "omp.h"
  #endif
#endif


//// ---------  Stan includes
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <stan/math.hpp>
////
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/prim/err/check_pos_definite.hpp>
#include <stan/math/prim/err/check_square.hpp>
#include <stan/math/prim/err/check_symmetric.hpp>
#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/sqrt.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/transpose.hpp>
#include <stan/math/prim/fun/dot_product.hpp>
#include <stan/math/prim/fun/norm2.hpp>
#include <stan/math/prim/fun/diagonal.hpp>
#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/eigenvalues_sym.hpp>
#include <stan/math/prim/fun/diag_post_multiply.hpp>
#include <stan/math/prim/prob/multi_normal_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/lkj_corr_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/weibull_lpdf.hpp>
#include <stan/math/prim/prob/gamma_lpdf.hpp>
#include <stan/math/prim/prob/beta_lpdf.hpp>


//// ---------  Eigen C++ lib. includes
#undef OUT
#include <RcppEigen.h>
#include <unsupported/Eigen/SpecialFunctions>
//// #include <unsupported/Eigen/CXX11/Tensor>

 
//// --------- BayesMVP config. include
#include "initial_config.hpp" //// Other config. 
#include "SIMD_config.hpp"  ////  config. (must be included BEFORE eigen_config.hpp)
#include "eigen_config.hpp" //// Eigen C++ lib. config.  


//// ---------  Other Stan includes  
#include <stan/model/model_base.hpp>  
#include <stan/io/array_var_context.hpp> 
#include <stan/io/var_context.hpp> 
#include <stan/io/dump.hpp>  
#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/rapidjson_parser.hpp>   



    
#include <RcppParallel.h>
//// #include <RcppEigen.h>

//// --------- BayesMVP includes - General functions (e.g. fast exp() and log() approximations). Most of these are not model-specific.
#include "general_functions/var_fns.hpp"
#include "general_functions/double_fns.hpp"


//// --------- BayesMVP includes - General functions
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_Stan.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_Loop.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fast_and_approx_AVX2_fns.hpp" // will only compile if  AVX2 is available
#include "general_functions/fns_SIMD_and_wrappers/fast_and_approx_AVX512_fns.hpp" // will only compile if  AVX-512 is available
// #include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_SIMD_AVX2.hpp" // will only compile if AVX2 is available
//// #include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_SIMD_AVX2_test.hpp" ///////////  FOR DEBUG ONLY
// #include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_SIMD_AVX512.hpp" // will only compile if AVX-512 is available
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_SIMD_AVX_general.hpp" // will only compile if AVX-512 (1st choice) or AVX2 available
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_overall.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_log_sum_exp_dbl.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_log_sum_exp_SIMD.hpp"


//// --------- BayesMVP includes - General functions
#include "general_functions/array_creators_Eigen_fns.hpp"
#include "general_functions/array_creators_other_fns.hpp"
#include "general_functions/structures.hpp"
#include "general_functions/classes.hpp"


#include <Rcpp.h>

//// --------- BayesMVP includes - General functions
#include "general_functions/misc_helper_fns_1.hpp"
#include "general_functions/misc_helper_fns_2.hpp" /// needs Rcpp.h
#include "general_functions/compute_diagnostics.hpp"

//////// #include "BayesMVP_Stan_fast_approx_fns.hpp"

//// --------- BayesMVP includes - MVP-specific (and MVP-LC) model fns
#include "MVP_functions/MVP_manual_grad_calc_fns.hpp"
#include "MVP_functions/MVP_log_scale_grad_calc_fns.hpp"
#include "MVP_functions/MVP_manual_trans_and_J_fns.hpp"
#include "MVP_functions/std_MVP_lp_grad_AD_fns.hpp"
#include "MVP_functions/MVP_lp_grad_AD_fns.hpp"
#include "MVP_functions/MVP_lp_grad_MD_AD_fns.hpp"
#include "MVP_functions/MVP_lp_grad_log_scale_MD_AD_fns.hpp"
//// #include "MVP_functions/MVP_manual_Hessian_calc_fns.hpp"
#include "MVP_functions/MVOP_lp_grad_AD_fns.hpp"
#include "MVP_functions/MVP_lp_grad_multi_attempts.hpp"


//// --------- BayesMVP includes - Latent trait model fns
#include "LC_LT_functions/LC_LT_manual_grad_calc_fns.hpp"
#include "LC_LT_functions/LC_LT_log_scale_grad_calc_fns.hpp"
#include "LC_LT_functions/LC_LT_lp_grad_AD_fns.hpp"
#include "LC_LT_functions/LC_LT_lp_grad_MD_AD_fns.hpp" 
#include "LC_LT_functions/LC_LT_lp_grad_log_scale_MD_AD_fns.hpp" //// bookmark - (LT_b's not currently working)
#include "LC_LT_functions/LT_LC_lp_grad_multi_attempts.hpp"

    
//// ---------  BridgeStan includes 
#include "bridgestan.h" 
#include "version.hpp"
#include "model_rng.hpp" 

//// --------- BayesMVP includes - Stan model helper functions
#include "general_functions/Stan_model_helper_fns.hpp"
#include "general_functions/Stan_model_helper_fns_parallel.hpp"

//// --------- BayesMVP includes - general lp_grad fns
#include "general_functions/lp_grad_model_selector.hpp"

    
//// --------- BayesMVP includes - MCMC includes - ADAM / SNAPER-HMC fns (general)
#include "MCMC/EHMC_adapt_eps_fn.hpp"
#include "MCMC/EHMC_adapt_tau_fns.hpp"
#include "MCMC/EHMC_adapt_M_Hessian_fns.hpp"



#include <Rcpp.h>

//// --------- RNG (using dqrng) includes:
// #define PCG_EMULATED_OFFSETOF
#if RNG_TYPE_pcg64 == 1
    #include "RNG/pcg_random.hpp"
    //// #include "pcg_extras.hpp"
#endif
#if RNG_TYPE_pcg32 == 1
    #include "RNG/pcg_random.hpp"
    //// #include "pcg_extras.hpp" 
#endif
#if using_dqrng == 1
    //// #include "RNG/pcg_random.hpp"
    #include <dqrng_distribution.h>
    #include <dqrng_generator.h>
    //// #include <dqrng_sample.h>
    #include <xoshiro.h>
#endif
    
    
#if RNG_TYPE_dqrng_pcg64 == 1
    using RNG_TYPE_dqrng = Rcpp::XPtr<dqrng::random_64bit_generator>; 
#elif RNG_TYPE_dqrng_xoshiro256plusplus == 1
    using RNG_TYPE_dqrng = dqrng::xoshiro256plus;
#elif RNG_TYPE_CPP_STD == 1
    using RNG_TYPE_dqrng = std::mt19937;
#endif

//// --------- BayesMVP includes - MCMC includes - Standard-HMC (EHMC) + Diffusion-space-HMC sampler fns
#define COMPILE_MCMC_MAIN 1
#if COMPILE_MCMC_MAIN
  #include "MCMC/EHMC_random_draw_fns.hpp"
  #include "MCMC/EHMC_main_sampler_fns.hpp"
  #include "MCMC/EHMC_nuisance_sampler_fns.hpp"
  #include "MCMC/EHMC_dual_sampler_fns.hpp"
  #include "MCMC/EHMC_find_initial_eps_fns.hpp"
  #include "MCMC/EHMC_single_threaded_samp_fns.hpp"
  #include "MCMC/EHMC_burn_multi_thread_samp_fns_RCPP.hpp"
  #include "MCMC/EHMC_pb_multi_thread_samp_fns_RCPP.hpp"
#if ENABLE_OPEN_MP == 1
    #include "MCMC/EHMC_multi_threaded_samp_fns_OMP.hpp" //// needs OpenMP
  #endif
#endif



#undef OUT



#include <tbb/task_arena.h>






using namespace Rcpp;
using namespace Eigen;




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





// ANSI codes for different colors
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"









Model_fn_args_struct   convert_R_List_to_Model_fn_args_struct(Rcpp::List R_List) {

     // handles empty fields by using defaults
     const int N = R_List["N"];
     const int n_nuisance = R_List["n_nuisance"];
     const int n_params_main = R_List["n_params_main"];

     const std::string model_so_file = R_List["model_so_file"];
     const std::string json_file_path = R_List["json_file_path"];

     const Eigen::Matrix<bool, -1, 1> Model_args_bools = Rcpp::as<Eigen::Matrix<bool, -1, 1>>(R_List.containsElementNamed("Model_args_bools") ? R_List["Model_args_bools"] : Rcpp::LogicalMatrix(0));
     const Eigen::Matrix<int, -1, 1> Model_args_ints = Rcpp::as<Eigen::Matrix<int, -1, 1>>(R_List.containsElementNamed("Model_args_ints") ? R_List["Model_args_ints"] : Rcpp::IntegerMatrix(0));
     const Eigen::Matrix<double, -1, 1> Model_args_doubles = Rcpp::as<Eigen::Matrix<double, -1, 1>>(R_List.containsElementNamed("Model_args_doubles") ? R_List["Model_args_doubles"] : Rcpp::NumericMatrix(0));
     const Eigen::Matrix<std::string, -1, 1> Model_args_strings = Rcpp::as<Eigen::Matrix<std::string, -1, 1>>(R_List.containsElementNamed("Model_args_strings") ? R_List["Model_args_strings"] : Rcpp::StringMatrix(0));

     const std_vec_of_EigenVecs_dbl Model_args_col_vecs_double = Rcpp::as<std_vec_of_EigenVecs_dbl>(R_List.containsElementNamed("Model_args_col_vecs_double") ? R_List["Model_args_col_vecs_double"] : Rcpp::List(0));
     const std_vec_of_EigenVecs_int Model_args_col_vecs_int = Rcpp::as<std_vec_of_EigenVecs_int>(R_List.containsElementNamed("Model_args_col_vecs_int") ? R_List["Model_args_col_vecs_int"] : Rcpp::List(0));
     const std_vec_of_EigenMats_dbl Model_args_mats_double = Rcpp::as<std_vec_of_EigenMats_dbl>(R_List.containsElementNamed("Model_args_mats_double") ? R_List["Model_args_mats_double"] : Rcpp::List(0));
     const std_vec_of_EigenMats_int Model_args_mats_int = Rcpp::as<std_vec_of_EigenMats_int>(R_List.containsElementNamed("Model_args_mats_int") ? R_List["Model_args_mats_int"] : Rcpp::List(0));

     const two_layer_std_vec_of_EigenVecs_dbl Model_args_vecs_of_col_vecs_double = Rcpp::as<two_layer_std_vec_of_EigenVecs_dbl>(R_List.containsElementNamed("Model_args_vecs_of_col_vecs_double") ? R_List["Model_args_vecs_of_col_vecs_double"] : Rcpp::List(0));
     const two_layer_std_vec_of_EigenVecs_int Model_args_vecs_of_col_vecs_int = Rcpp::as<two_layer_std_vec_of_EigenVecs_int>(R_List.containsElementNamed("Model_args_vecs_of_col_vecs_int") ? R_List["Model_args_vecs_of_col_vecs_int"] : Rcpp::List(0));
     const two_layer_std_vec_of_EigenMats_dbl Model_args_vecs_of_mats_double = Rcpp::as<two_layer_std_vec_of_EigenMats_dbl>(R_List.containsElementNamed("Model_args_vecs_of_mats_double") ? R_List["Model_args_vecs_of_mats_double"] : Rcpp::List(0));
     const two_layer_std_vec_of_EigenMats_int Model_args_vecs_of_mats_int = Rcpp::as<two_layer_std_vec_of_EigenMats_int>(R_List.containsElementNamed("Model_args_vecs_of_mats_int") ? R_List["Model_args_vecs_of_mats_int"] : Rcpp::List(0));

     const three_layer_std_vec_of_EigenVecs_dbl Model_args_2_layer_vecs_of_col_vecs_double = Rcpp::as<three_layer_std_vec_of_EigenVecs_dbl>(R_List.containsElementNamed("Model_args_2_layer_vecs_of_col_vecs_double") ? R_List["Model_args_2_layer_vecs_of_col_vecs_double"] : Rcpp::List(0));
     const three_layer_std_vec_of_EigenVecs_int Model_args_2_layer_vecs_of_col_vecs_int = Rcpp::as<three_layer_std_vec_of_EigenVecs_int>(R_List.containsElementNamed("Model_args_2_layer_vecs_of_col_vecs_int") ? R_List["Model_args_2_layer_vecs_of_col_vecs_int"] : Rcpp::List(0));
     const three_layer_std_vec_of_EigenMats_dbl Model_args_2_layer_vecs_of_mats_double = Rcpp::as<three_layer_std_vec_of_EigenMats_dbl>(R_List.containsElementNamed("Model_args_2_layer_vecs_of_mats_double") ? R_List["Model_args_2_layer_vecs_of_mats_double"] : Rcpp::List(0));
     const three_layer_std_vec_of_EigenMats_int Model_args_2_layer_vecs_of_mats_int = Rcpp::as<three_layer_std_vec_of_EigenMats_int>(R_List.containsElementNamed("Model_args_2_layer_vecs_of_mats_int") ? R_List["Model_args_2_layer_vecs_of_mats_int"] : Rcpp::List(0));

     return Model_fn_args_struct(
       N,
       n_nuisance,
       n_params_main,
       model_so_file,
       json_file_path,
       Model_args_bools,
       Model_args_ints,
       Model_args_doubles,
       Model_args_strings,
       Model_args_col_vecs_double,
       Model_args_col_vecs_int,
       Model_args_mats_double,
       Model_args_mats_int,
       Model_args_vecs_of_col_vecs_double,
       Model_args_vecs_of_col_vecs_int,
       Model_args_vecs_of_mats_double,
       Model_args_vecs_of_mats_int,
       Model_args_2_layer_vecs_of_col_vecs_double,
       Model_args_2_layer_vecs_of_col_vecs_int,
       Model_args_2_layer_vecs_of_mats_double,
       Model_args_2_layer_vecs_of_mats_int
     );

}








//////  Function to convert from R List -> C++ struct (so can call fn's from R without having to use Rcpp::List as not thread-safe, and slower etc)
EHMC_Metric_struct convert_R_List_EHMC_Metric_struct(const Rcpp::List &R_List) {

  ////
  const Eigen::Matrix<double, -1, -1> M_dense_main = Rcpp::as<Eigen::Matrix<double, -1, -1>>(R_List["M_dense_main"]);
  const Eigen::Matrix<double, -1, -1> M_inv_dense_main = Rcpp::as<Eigen::Matrix<double, -1, -1>>(R_List["M_inv_dense_main"]);
  const Eigen::Matrix<double, -1, -1> M_inv_dense_main_chol = Rcpp::as<Eigen::Matrix<double, -1, -1>>(R_List["M_inv_dense_main_chol"]);
  ////
  const Eigen::Matrix<double, -1, 1>  M_inv_main_vec = Rcpp::as<Eigen::Matrix<double, -1, 1>>(R_List["M_inv_main_vec"]);
  ////
  const Eigen::Matrix<double, -1, 1>  M_inv_us_vec = Rcpp::as<Eigen::Matrix<double, -1, 1>>(R_List["M_inv_us_vec"]);
  const Eigen::Matrix<double, -1, 1>  M_us_vec = Rcpp::as<Eigen::Matrix<double, -1, 1>>(R_List["M_us_vec"]);
  ////
  const Eigen::Matrix<double, -1, 1>  theta_hat_us_vec = Rcpp::as<Eigen::Matrix<double, -1, 1>>(R_List["theta_hat_us_vec"]);
  ////
  const std::string metric_shape_main = R_List["metric_shape_main"]  ;

  return EHMC_Metric_struct(M_dense_main,
                            M_inv_dense_main,
                            M_inv_dense_main_chol,
                            M_inv_main_vec,
                            M_inv_us_vec,
                            M_us_vec,
                            theta_hat_us_vec,
                            metric_shape_main);
}








// /////  Function to convert from R List -> C++ struct (so can call fn's from R without having to use Rcpp::List as not thread-safe, and slower etc)
EHMC_fn_args_struct convert_R_List_EHMC_fn_args_struct(Rcpp::List R_List) {

        // Convert the R list elements to the appropriate C++ types
        // for main params
        double tau_main = (R_List["tau_main"]);
        double tau_main_ii = (R_List["tau_main_ii"]);
        double eps_main = (R_List["eps_main"]);

        // for nuisance params
        double tau_us = (R_List["tau_us"]);
        double tau_us_ii = (R_List["tau_us_ii"]);
        double eps_us = (R_List["eps_us"]);

        // general
        bool diffusion_HMC = (R_List["diffusion_HMC"]);

        return EHMC_fn_args_struct( tau_main,
                                    tau_main_ii,
                                    eps_main,
                                    // for nuisance params
                                    tau_us,
                                    tau_us_ii,
                                    eps_us,
                                    diffusion_HMC);

}








// /////  Function to convert from R List -> C++ struct (so can call fn's from R without having to use Rcpp::List as not thread-safe, and slower etc)
EHMC_burnin_struct convert_R_List_EHMC_burnin_struct(Rcpp::List R_List) {

  // Convert the R list elements to the appropriate C++ types
  // for main params
  double adapt_delta_main = (R_List["adapt_delta_main"]);
  double LR_main = (R_List["LR_main"]);
  double eps_m_adam_main = (R_List["eps_m_adam_main"]);
  double eps_v_adam_main = (R_List["eps_v_adam_main"]);
  double tau_m_adam_main = (R_List["tau_m_adam_main"]);
  double tau_v_adam_main = (R_List["tau_v_adam_main"]);
  double eigen_max_main =  (R_List["eigen_max_main"]);
  Eigen::VectorXi index_main = (R_List["index_main"]);
  Eigen::Matrix<double, -1, -1> M_dense_sqrt = (R_List["M_dense_sqrt"]);
  Eigen::Matrix<double, -1, 1>  snaper_m_vec_main = (R_List["snaper_m_vec_main"]);
  Eigen::Matrix<double, -1, 1>  snaper_s_vec_main_empirical = (R_List["snaper_s_vec_main_empirical"]);
  Eigen::Matrix<double, -1, 1>  snaper_w_vec_main = (R_List["snaper_w_vec_main"]);
  Eigen::Matrix<double, -1, 1>  eigen_vector_main = (R_List["eigen_vector_main"]);

  // for nuisance params
  double adapt_delta_us = (R_List["adapt_delta_us"]);
  double LR_us = (R_List["LR_us"]);
  double eps_m_adam_us = (R_List["eps_m_adam_us"]);
  double eps_v_adam_us = (R_List["eps_v_adam_us"]);
  double tau_m_adam_us = (R_List["tau_m_adam_us"]);
  double tau_v_adam_us = (R_List["tau_v_adam_us"]);
  double eigen_max_us =  (R_List["eigen_max_us"]);
  Eigen::VectorXi index_us = (R_List["index_us"]);
  Eigen::Matrix<double, -1, 1>  sqrt_M_us_vec = (R_List["sqrt_M_us_vec"]);
  Eigen::Matrix<double, -1, 1>  snaper_m_vec_us = (R_List["snaper_m_vec_us"]);
  Eigen::Matrix<double, -1, 1>  snaper_s_vec_us_empirical = (R_List["snaper_s_vec_us_empirical"]);
  Eigen::Matrix<double, -1, 1>  snaper_w_vec_us = (R_List["snaper_w_vec_us"]);
  Eigen::Matrix<double, -1, 1>  eigen_vector_us = (R_List["eigen_vector_us"]);

  return EHMC_burnin_struct(     adapt_delta_main,
                                 LR_main,
                                 eps_m_adam_main,
                                 eps_v_adam_main,
                                 tau_m_adam_main,
                                 tau_v_adam_main,
                                 eigen_max_main,
                                 index_main,
                                 M_dense_sqrt,
                                 snaper_m_vec_main,
                                 snaper_s_vec_main_empirical,
                                 snaper_w_vec_main,
                                 eigen_vector_main,
                                 /////// nuisance
                                 adapt_delta_us,
                                 LR_us,
                                 eps_m_adam_us,
                                 eps_v_adam_us,
                                 tau_m_adam_us,
                                 tau_v_adam_us,
                                 eigen_max_us,
                                 index_us,
                                 sqrt_M_us_vec,
                                 snaper_m_vec_us,
                                 snaper_s_vec_us_empirical,
                                 snaper_w_vec_us,
                                 eigen_vector_us);

}




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















// [[Rcpp::export]]
Eigen::Matrix<double, -1, -1>     Rcpp_wrapper_EIGEN_double_mat(             const Eigen::Matrix<double, -1, -1> x,
                                                                             const std::string fn,
                                                                             const std::string vect_type,
                                                                             const bool skip_checks
) {

    Eigen::Matrix<double, -1, -1> x_copy = x;
    Eigen::Ref<Eigen::Matrix<double, -1, -1>> x_copy_ref(x_copy);
    
    #if (defined(USE_AVX2) || defined(USE_AVX_512))
        if (fn == "test_simple_debug") {
          // const std::string test_simple_string = "test_simple";
          // TEST_fn_process_Ref_double_AVX2(x_copy_ref, fn, skip_checks);  // modify in-place
        } else {
          fn_process_Ref_double_AVX(x_copy_ref, fn, skip_checks);  // modify in-place
        }
    #endif
    
    return x_copy_ref;

}




 




 
 
 
 
 
 
 
 
 
 
 
 
 
// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>     Rcpp_wrapper_EIGEN_double_colvec(           const Eigen::Matrix<double, -1, 1> x,
                                                                             const std::string fn,
                                                                             const std::string vect_type,
                                                                             const bool skip_checks
) {

  Eigen::Matrix<double, -1, 1> x_copy = x;
  Eigen::Ref<Eigen::Matrix<double, -1, 1>> x_copy_ref(x_copy);
  
  #if (defined(HAS_AVX2) || defined(HAS_AVX_512))
    if (fn == "test_simple_debug") {
      // const std::string test_simple_string = "test_simple";
      // TEST_fn_process_Ref_double_AVX2(x_copy_ref, test_simple_string, skip_checks);  // modify in-place
    } else {
      fn_process_Ref_double_AVX(x_copy_ref, fn, skip_checks);  // modify in-place
    }
  #endif

  return x_copy_ref;

}











// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>        Rcpp_wrapper_fn_lp_grad(             const std::string Model_type,
                                                                         const bool force_autodiff,
                                                                         const bool force_PartialLog,
                                                                         const bool multi_attempts,
                                                                         const Eigen::Matrix<double, -1, 1> theta_main_vec,
                                                                         const Eigen::Matrix<double, -1, 1> theta_us_vec,
                                                                         const Eigen::Matrix<int, -1, -1>  y,
                                                                         const std::string grad_option,
                                                                         const Rcpp::List Model_args_as_Rcpp_List
) {

  const int N = y.rows();
  const int n_us = theta_us_vec.rows()  ;
  const int n_params_main =  theta_main_vec.rows()  ;
  const int n_params = n_params_main + n_us;

  /// convert to Eigen
  const Eigen::Matrix<double, -1, 1> theta_main_vec_Ref =  theta_main_vec;
  const Eigen::Matrix<double, -1, 1> theta_us_vec_Ref =  theta_us_vec;
  const Eigen::Matrix<int, -1, -1>   y_Ref =  y;

  const Model_fn_args_struct Model_args_as_cpp_struct = convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List);

  Eigen::Matrix<double, -1, 1> lp_grad_outs = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);

  Stan_model_struct Stan_model_as_cpp_struct;

  if (Model_type == "Stan") {

    unsigned int seed = 123;
    const std::string model_so_file = Model_args_as_cpp_struct.model_so_file;
    const std::string json_file_path = Model_args_as_cpp_struct.json_file_path;
    Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(model_so_file, json_file_path, seed);

  }

  fn_lp_grad_InPlace(   lp_grad_outs,
                        Model_type,
                        force_autodiff, force_PartialLog, multi_attempts,
                        theta_main_vec_Ref, theta_us_vec_Ref,
                        y_Ref,
                        grad_option,
                        Model_args_as_cpp_struct,//MVP_workspace,
                        Stan_model_as_cpp_struct);

  fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);

  return lp_grad_outs;

}







// [[Rcpp::export]]
Rcpp::List Rcpp_compute_chain_stats(const std::vector<Eigen::Matrix<double, -1, -1>> mcmc_3D_array,
                                    const std::string stat_type,
                                    const int n_threads) {

  const int n_params = mcmc_3D_array.size();
  Rcpp::NumericMatrix output(n_params, 3);

  ComputeStatsParallel parallel_worker(n_params,
                                       stat_type,
                                       mcmc_3D_array,
                                       output);

  RcppParallel::parallelFor(0, n_params, parallel_worker);

  return Rcpp::List::create(Rcpp::Named("statistics") = output);

}








// [[Rcpp::export]]
Rcpp::List  Rcpp_compute_MCMC_diagnostics(     const std::vector<Eigen::Matrix<double, -1, -1>> mcmc_3D_array,
                                               const std::string diagnostic,
                                               const int n_threads
) {

      const int n_params = mcmc_3D_array.size();
      Rcpp::NumericMatrix  output(n_params, 2);

      //// Create the parallel worker
      ComputeDiagnosticParallel parallel_worker(n_params,
                                                diagnostic,
                                                mcmc_3D_array,
                                                output);

      //// Run parallelFor
      RcppParallel::parallelFor(0, n_params, parallel_worker); // RCppParallel will distribute the load across the n_threads

      //// output
      return Rcpp::List::create(Rcpp::Named("diagnostics") = output);

}







// [[Rcpp::export]]
Rcpp::String  detect_vectorization_support() {

    #if defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__) // use AVX-512 if available
      return "AVX512";
    #elif defined(__AVX2__) && ( !(defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)) ) // use AVX2
      return "AVX2";
    #endif
    // #elif defined(__AVX__) && !(defined(__AVX2__)) &&  ( !(defined(__AVX512VL__) && defined(__AVX512F__)  && defined(__AVX512DQ__)) ) // use AVX2 // use AVX
    //   return "AVX";
    // #endif
    
    return "Stan";

}


















// [[Rcpp::export]]
Rcpp::List    fn_Rcpp_compute_PD_Hessian_main(        const double shrinkage_factor,
                                                      const double num_diff_e,
                                                      const std::string  Model_type,
                                                      const bool force_autodiff,
                                                      const bool force_PartialLog,
                                                      const bool multi_attempts,
                                                      const Eigen::Matrix<double, -1, 1> theta_main_vec,
                                                      const Eigen::Matrix<double, -1, 1> theta_us_vec,
                                                      const Eigen::Matrix<int, -1, -1> y,
                                                      const Rcpp::List Model_args_as_Rcpp_List
) {
  
          const int N = y.rows();
          const int n_us = theta_us_vec.rows()  ;
          const int n_params_main =  theta_main_vec.rows()  ;
          const int n_params = n_params_main + n_us;
          
          const Model_fn_args_struct Model_args_as_cpp_struct = convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List);
          
          
          Eigen::Matrix<double, -1, -1> Hessian =   compute_PD_Hessian_main(   shrinkage_factor,
                                                                               num_diff_e,
                                                                               Model_type,
                                                                               force_autodiff,
                                                                               force_PartialLog,
                                                                               multi_attempts,
                                                                               theta_main_vec,
                                                                               theta_us_vec,
                                                                               y,
                                                                               Model_args_as_cpp_struct);
          
          return Rcpp::List::create(
            Rcpp::Named("Hessian") = Hessian
          );
  
}





 
 

// [[Rcpp::export]]
Rcpp::List    fn_Rcpp_wrapper_update_M_dense_main_Hessian(            Eigen::Matrix<double, -1, -1> M_dense_main,  /// to be updated
                                                                      Eigen::Matrix<double, -1, -1> M_inv_dense_main, /// to be updated
                                                                      Eigen::Matrix<double, -1, -1> M_inv_dense_main_chol, /// to be updated
                                                                      const double shrinkage_factor,
                                                                      const double ratio_Hess_main,
                                                                      const int interval_width,
                                                                      const double num_diff_e,
                                                                      const std::string  Model_type,
                                                                      const bool force_autodiff,
                                                                      const bool force_PartialLog,
                                                                      const bool multi_attempts,
                                                                      const Eigen::Matrix<double, -1, 1> theta_main_vec,
                                                                      const Eigen::Matrix<double, -1, 1> theta_us_vec,
                                                                      const Eigen::Matrix<int, -1, -1> y,
                                                                      const Rcpp::List Model_args_as_Rcpp_List,
                                                                      const double   ii,
                                                                      const double   n_burnin,
                                                                      const std::string metric_type
) {

         const int N = y.rows();
         const int n_us = theta_us_vec.rows()  ;
         const int n_params_main =  theta_main_vec.rows()  ;
         const int n_params = n_params_main + n_us;

         const Model_fn_args_struct Model_args_as_cpp_struct = convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List);

         Eigen::Matrix<double, -1, -1> M_dense_main_copy = M_dense_main;
         Eigen::Matrix<double, -1, -1> M_inv_dense_main_copy = M_inv_dense_main;
         Eigen::Matrix<double, -1, -1> M_inv_dense_main_chol_copy = M_inv_dense_main_chol;


         update_M_dense_main_Hessian_InPlace(     M_dense_main_copy,
                                                  M_inv_dense_main_copy,
                                                  M_inv_dense_main_chol_copy,
                                                  shrinkage_factor,
                                                  ratio_Hess_main,
                                                  interval_width,
                                                  num_diff_e,
                                                  Model_type,
                                                  force_autodiff,
                                                  force_PartialLog,
                                                  multi_attempts,
                                                  theta_main_vec,
                                                  theta_us_vec,
                                                  y,
                                                  Model_args_as_cpp_struct,
                                                  ii,
                                                  n_burnin,
                                                  metric_type);


         return Rcpp::List::create(
           Rcpp::Named("M_dense_main_copy") = M_dense_main_copy,
           Rcpp::Named("M_inv_dense_main_copy") = M_inv_dense_main_copy,
           Rcpp::Named("M_inv_dense_main_chol_copy") = M_inv_dense_main_chol_copy
         );



}

 





 
// [[Rcpp::export]]
Rcpp::List    fn_Rcpp_compute_PD_Hessian_diag(     const double shrinkage_factor,
                                                   const double num_diff_e,
                                                   const std::string  Model_type,
                                                   const bool force_autodiff,
                                                   const bool force_PartialLog,
                                                   const bool multi_attempts,
                                                   const Eigen::Matrix<double, -1, 1> theta_main_vec,
                                                   const Eigen::Matrix<double, -1, 1> theta_us_vec,
                                                   const Eigen::Matrix<int, -1, -1> y,
                                                   const Rcpp::List Model_args_as_Rcpp_List
 ) {
   
         const int N = y.rows();
         const int n_us = theta_us_vec.rows()  ;
         const int n_params_main =  theta_main_vec.rows()  ;
         const int n_params = n_params_main + n_us;
         
         const Model_fn_args_struct Model_args_as_cpp_struct = convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List);
         
         
         Eigen::Matrix<double, -1, 1> Hessian =   compute_PD_Hessian_diag_main(   shrinkage_factor,
                                                                                  num_diff_e,
                                                                                  Model_type,
                                                                                  force_autodiff,
                                                                                  force_PartialLog,
                                                                                  multi_attempts,
                                                                                  theta_main_vec,
                                                                                  theta_us_vec,
                                                                                  y,
                                                                                  Model_args_as_cpp_struct);
         
         return Rcpp::List::create(
           Rcpp::Named("Hessian") = Hessian
         );
   
}





 


// [[Rcpp::export]]
Rcpp::List    fn_Rcpp_wrapper_update_M_diag_Hessian(      Eigen::Matrix<double, -1, 1> M_main_vec,  /// to be updated
                                                          Eigen::Matrix<double, -1, 1> M_inv_main_vec, /// to be updated
                                                          const double shrinkage_factor,
                                                          const double ratio_Hess_main,
                                                          const int interval_width,
                                                          const double num_diff_e,
                                                          const std::string  Model_type,
                                                          const bool force_autodiff,
                                                          const bool force_PartialLog,
                                                          const bool multi_attempts,
                                                          const Eigen::Matrix<double, -1, 1> theta_main_vec,
                                                          const Eigen::Matrix<double, -1, 1> theta_us_vec,
                                                          const Eigen::Matrix<int, -1, -1> y,
                                                          const Rcpp::List Model_args_as_Rcpp_List,
                                                          const double   ii,
                                                          const double   n_burnin,
                                                          const std::string metric_type
) {
  
        const int N = y.rows();
        const int n_us = theta_us_vec.rows()  ;
        const int n_params_main =  theta_main_vec.rows()  ;
        const int n_params = n_params_main + n_us;
        
        const Model_fn_args_struct Model_args_as_cpp_struct = convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List);
        
        Eigen::Matrix<double, -1, 1> M_main_vec_copy = M_main_vec;
        Eigen::Matrix<double, -1, 1> M_inv_main_vec_copy = M_inv_main_vec;
        
        
        update_M_diag_main_Hessian_InPlace(      M_main_vec_copy,
                                                 M_inv_main_vec_copy,
                                                 shrinkage_factor,
                                                 ratio_Hess_main,
                                                 interval_width,
                                                 num_diff_e,
                                                 Model_type,
                                                 force_autodiff,
                                                 force_PartialLog,
                                                 multi_attempts, 
                                                 theta_main_vec,
                                                 theta_us_vec,
                                                 y,
                                                 Model_args_as_cpp_struct,
                                                 ii,
                                                 n_burnin,
                                                 metric_type);
        
        
        return Rcpp::List::create(
          Rcpp::Named("M_main_vec_copy") = M_main_vec_copy,
          Rcpp::Named("M_inv_main_vec_copy") = M_inv_main_vec_copy
        );
  
}

 

 


//////////////////////   ADAM / SNAPER-HMC wrapper fns  -------------------------------------------------------------------------------------------------------------------------------



// [[Rcpp::export]]
Rcpp::List                         fn_find_initial_eps_main_and_us(           Eigen::Matrix<double, -1, 1> theta_main_vec_initial_ref,
                                                                              Eigen::Matrix<double, -1, 1> theta_us_vec_initial_ref,
                                                                              const bool partitioned_HMC,
                                                                              const double seed,
                                                                              const std::string Model_type,
                                                                              const bool  force_autodiff,
                                                                              const bool  force_PartialLog,
                                                                              const bool  multi_attempts,
                                                                              Eigen::Matrix<int, -1, -1> y_ref,
                                                                              const Rcpp::List Model_args_as_Rcpp_List,
                                                                              Rcpp::List  EHMC_args_as_Rcpp_List, /// pass by ref. to modify (???)
                                                                              const Rcpp::List   EHMC_Metric_as_Rcpp_List
) {
  
      const bool burnin = false;
      const int n_params_main = theta_main_vec_initial_ref.rows();
      const int n_nuisance =    theta_us_vec_initial_ref.rows();
      const int n_params = n_params_main + n_nuisance;
      const int N = y_ref.rows();

      HMCResult result_input(n_params_main, n_nuisance, N);
      result_input.main_theta_vec() = theta_main_vec_initial_ref;
      result_input.main_theta_vec_0()  = theta_main_vec_initial_ref;
      result_input.main_theta_vec_proposed()  = theta_main_vec_initial_ref;
      result_input.main_velocity_0_vec()  = theta_main_vec_initial_ref;
      result_input.main_velocity_vec_proposed()  = theta_main_vec_initial_ref;
      result_input.main_velocity_vec()  = theta_main_vec_initial_ref;

      // convert Rcpp::List to cpp structs and pass by reference
      const Model_fn_args_struct Model_args_as_cpp_struct = convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List);
      EHMC_fn_args_struct  EHMC_args_as_cpp_struct =  convert_R_List_EHMC_fn_args_struct(EHMC_args_as_Rcpp_List);
      const EHMC_Metric_struct   EHMC_Metric_as_cpp_struct =  convert_R_List_EHMC_Metric_struct(EHMC_Metric_as_Rcpp_List);

      std::vector<double> eps_pair =  fn_find_initial_eps_main_and_us(   result_input,
                                                                         partitioned_HMC,
                                                                         seed, burnin,  Model_type,
                                                                         force_autodiff, force_PartialLog, multi_attempts,
                                                                         y_ref,
                                                                         Model_args_as_cpp_struct, // MVP_workspace,
                                                                         EHMC_args_as_cpp_struct, EHMC_Metric_as_cpp_struct);

      Rcpp::List outs(2);
      outs(0) = eps_pair[0];
      outs(1) = eps_pair[1];
      return outs;

}











// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>     fn_Rcpp_wrapper_adapt_eps_ADAM(   double eps,   //// updating this
                                                                   double eps_m_adam,   //// updating this
                                                                   double eps_v_adam,  //// updating this
                                                                   const int iter,
                                                                   const int n_burnin,
                                                                   const double LR,  /// ADAM learning rate
                                                                   const double p_jump,
                                                                   const double adapt_delta,
                                                                   const double beta1_adam,
                                                                   const double beta2_adam,
                                                                   const double eps_adam
) {


  return    adapt_eps_ADAM(  eps,
                             eps_m_adam,
                             eps_v_adam,
                             iter,
                             n_burnin,
                             LR,
                             p_jump,
                             adapt_delta,
                             beta1_adam,
                             beta2_adam,
                             eps_adam);




}



 






// [[Rcpp::export]]
Eigen::Matrix<double, -1, -1>  fn_update_snaper_m_and_s(     Eigen::Matrix<double, -1, 1> snaper_m,  // to be updated
                                                             Eigen::Matrix<double, -1, 1> snaper_s_empirical,   // to be updated
                                                             const Eigen::Matrix<double, -1, 1>  theta_vec_mean,  // mean theta_vec across all K chains
                                                             const double ii
) {


        const double kappa  =  8.0;
        const double eta_m  =  1.0 / (std::ceil(static_cast<double>(ii)/kappa) + 1.0);
  
        // update snaper_m
        if (static_cast<int>(ii) < 2) {
          snaper_m = theta_vec_mean;
        } else {
          snaper_m = (1.0 - eta_m)*snaper_m + eta_m*theta_vec_mean;
        }
  
        // update posterior variances (snaper_s_empirical)
        Eigen::Matrix<double, -1, 1> theta_vec_mean_m_snaper_m = (theta_vec_mean  - snaper_m);
        Eigen::Matrix<double, -1, 1> current_variances = ( theta_vec_mean_m_snaper_m.array() * theta_vec_mean_m_snaper_m.array() ).matrix() ;
        snaper_s_empirical = (1.0 - eta_m)*snaper_s_empirical   +   eta_m*current_variances;
  
        Eigen::Matrix<double, -1, -1> out_mat(snaper_m.rows(), 2);
        out_mat.col(0) = snaper_m;
        out_mat.col(1) = snaper_s_empirical;
  
        return out_mat;

}


 








// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1> fn_update_snaper_w_dense_M(    Eigen::Matrix<double, -1, 1>  snaper_w_vec,    //// NOT const as updating!
                                                            const Eigen::Matrix<double, -1, 1>  eigen_vector,
                                                            const double eigen_max,
                                                            const Eigen::Matrix<double, -1, 1>  theta_vec,
                                                            const Eigen::Matrix<double, -1, 1>  snaper_m_vec,
                                                            const double ii,
                                                            const Eigen::Matrix<double, -1, -1> M_dense_sqrt
) {


    const double eta_w = 3.0;

    //// update W (for DENSE M) - this part varies from the diag_M tau-adaptation function!
    Eigen::Matrix<double, -1, 1> x_c = M_dense_sqrt * (theta_vec - snaper_m_vec); // this is the only part which is different from diag (and of course the inputs).
    if (eigen_max > 0.0) {
      double x_c_eigen_vector_dot_prod =  (x_c.array() * eigen_vector.array()).sum() ;
      Eigen::Matrix<double, -1, 1> current_w =   x_c * x_c_eigen_vector_dot_prod  ;
      snaper_w_vec = ( snaper_w_vec.array() * ((ii - eta_w) / (ii + 1.0)) + ((eta_w + 1.0) / (ii + 1.0)) * current_w.array() ).matrix() ; /// update snaper_w_vec
    } else {
      snaper_w_vec = x_c;
    }

    return snaper_w_vec;

}









// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1>  fn_update_snaper_w_diag_M(       Eigen::Matrix<double, -1, 1>  snaper_w_vec,    //// NOT const as updating!
                                                               const Eigen::Matrix<double, -1, 1>  eigen_vector,
                                                               const double eigen_max,
                                                               const Eigen::Matrix<double, -1, 1>  theta_vec,
                                                               const Eigen::Matrix<double, -1, 1>  snaper_m_vec,
                                                               const double ii,
                                                               const Eigen::Matrix<double, -1, 1>  sqrt_M_vec
) {
  
      const double eta_w = 3.0;
    
      //// update W (for DIAG M)
      const Eigen::Matrix<double, -1, 1> x_c = ( sqrt_M_vec.array() * (theta_vec - snaper_m_vec).array() ).matrix() ; // this is the only part which is different from diag (and of course the inputs).
      if (eigen_max > 0.0) {
        double x_c_eigen_vector_dot_prod =  (x_c.array() * eigen_vector.array()).sum();
        Eigen::Matrix<double, -1, 1> current_w = ( x_c * x_c_eigen_vector_dot_prod ).matrix();
        snaper_w_vec = ( snaper_w_vec.array() * ((ii - eta_w) / (ii + 1.0)) + ((eta_w + 1.0) / (ii + 1.0)) * current_w.array() ).matrix() ; /// update snaper_w_vec
      } else {
        snaper_w_vec = x_c;
      } 
    
      return snaper_w_vec; 

}










// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1> fn_update_eigen_max_and_eigen_vec( const Eigen::Matrix<double, -1, 1>  snaper_w_vec) {
  
          // compute L2-norm of W (sum of elements squared)
          const double w_norm_sq =  snaper_w_vec.array().square().sum();
          const double w_norm  =  stan::math::sqrt(w_norm_sq) ; // snaper_w_vec.array().square().sum();
          
          //// new eigen max 
          const double eigen_max = w_norm;
          
          //// update eigen vector
          Eigen::Matrix<double, -1, 1> eigen_vector = Eigen::Matrix<double, -1, 1>::Zero(snaper_w_vec.size());
          if (eigen_max > 0) {
            eigen_vector = snaper_w_vec / w_norm;
          }
          
          Eigen::Matrix<double, -1, 1> out_vec(eigen_vector.size() + 1);
          out_vec(0) = eigen_max;
          out_vec.tail(eigen_vector.size()) = eigen_vector;
          
          return out_vec;
  
}








// [[Rcpp::export]]
Eigen::Matrix<double, -1, 1> fn_Rcpp_wrapper_update_tau_w_diag_M_ADAM(   const Eigen::Matrix<double, -1, 1> eigen_vector,
                                                                         const double eigen_max,
                                                                         const Eigen::Matrix<double, -1, 1> theta_vec_initial,
                                                                         const Eigen::Matrix<double, -1, 1> theta_vec_prop,
                                                                         const Eigen::Matrix<double, -1, 1> snaper_m_vec,
                                                                         const Eigen::Matrix<double, -1, 1> velocity_prop,
                                                                         const Eigen::Matrix<double, -1, 1> velocity_0,
                                                                         double tau,  /// updating this
                                                                         const double LR,
                                                                         const double ii,
                                                                         const double n_burnin, 
                                                                         const Eigen::Matrix<double, -1, 1> sqrt_M_vec,
                                                                         double tau_m_adam,   /// updating this
                                                                         double tau_v_adam,  /// updating this
                                                                         const double tau_ii,
                                                                         const double beta1_adam,
                                                                         const double beta2_adam,
                                                                         const double eps_adam
) {

       //  Eigen::Matrix<double, -1, 1> out_vec(3); 
    
         ////// for main (this also updates snaper_w)
      return   fn_update_tau_w_diag_M_ADAM(
                                           eigen_vector,
                                           eigen_max,
                                           theta_vec_initial,
                                           theta_vec_prop,
                                           snaper_m_vec,   // READ-ONLY in this function
                                           velocity_prop,
                                           velocity_0,
                                           tau,  // being modified !!! 
                                           LR,
                                           ii,
                                           n_burnin,
                                           sqrt_M_vec,  // READ-ONLY in this function
                                           tau_m_adam,
                                           tau_v_adam,  // being modified !!!
                                           tau_ii,
                                           beta1_adam,
                                           beta2_adam,
                                           eps_adam
                                         );


} 






 // [[Rcpp::export]]
 Eigen::Matrix<double, -1, 1> fn_Rcpp_wrapper_update_tau_w_dense_M_ADAM(   const Eigen::Matrix<double, -1, 1> eigen_vector,
                                                                           const double eigen_max,
                                                                           const Eigen::Matrix<double, -1, 1> theta_vec_initial,
                                                                           const Eigen::Matrix<double, -1, 1> theta_vec_prop,
                                                                           const Eigen::Matrix<double, -1, 1> snaper_m_vec,
                                                                           const Eigen::Matrix<double, -1, 1> velocity_prop,
                                                                           const Eigen::Matrix<double, -1, 1> velocity_0,
                                                                           double tau,   /// updating this
                                                                           const double LR,
                                                                           const double ii,
                                                                           const double n_burnin,
                                                                           const Eigen::Matrix<double, -1, -1> M_dense_sqrt,
                                                                           double tau_m_adam,   /// updating this
                                                                           double tau_v_adam,  /// updating this
                                                                           const double tau_ii,
                                                                           const double beta1_adam,
                                                                           const double beta2_adam,
                                                                           const double eps_adam
 ) {

    // Eigen::Matrix<double, -1, 1> out_vec(3);

     ////// for main (this also updates snaper_w)
     return fn_update_tau_w_dense_M_ADAM(
                                         eigen_vector,
                                         eigen_max,
                                         theta_vec_initial,
                                         theta_vec_prop,
                                         snaper_m_vec,   // READ-ONLY in this function
                                         velocity_prop,
                                         velocity_0,
                                         tau,  // being modified !!!
                                         LR,
                                         ii,
                                         n_burnin,
                                         M_dense_sqrt,  // READ-ONLY in this function
                                         tau_m_adam,
                                         tau_v_adam,  // being modified !!!
                                         tau_ii,
                                         beta1_adam,
                                         beta2_adam,
                                         eps_adam
                                       );

}










// Some R / C++ helper functions -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


// [[Rcpp::export]]
double   Rcpp_det(const Eigen::Matrix<double, -1, -1>  &mat) {

     return(    (mat).determinant()   ) ;

}




// [[Rcpp::export]]
double   Rcpp_log_det(const Eigen::Matrix<double, -1, -1>  &mat) {

      return (  stan::math::log( stan::math::abs(  (mat).determinant())  )  ) ;

}





// [[Rcpp::export]]
Eigen::Matrix<double, -1, -1>    Rcpp_solve(const Eigen::Matrix<double, -1, -1>  &mat) {

      return mat.inverse(); //  fn_convert_EigenMat_to_RcppMat_dbl(fn_convert_RcppMat_to_EigenMat(mat).inverse());

}




// [[Rcpp::export]]
Eigen::Matrix<double, -1, -1>  Rcpp_Chol(const Eigen::Matrix<double, -1, -1>  &mat) {

      Eigen::Matrix<double, -1, -1>    res_Eigen = (  (mat).llt().matrixL() ).toDenseMatrix().matrix() ;
      return  (res_Eigen);

}








// [[Rcpp::export]]
Eigen::Matrix<double, -1, -1>  Rcpp_near_PD(const Eigen::Matrix<double, -1, -1>  &mat) {
  
        if (!(is_positive_definite(mat))) { 
          
              Eigen::Matrix<double, -1, -1>    res_Eigen = near_PD(mat);
              return res_Eigen;
          
        } else { 
        
              Eigen::Matrix<double, -1, -1> res_Eigen = mat;
              return res_Eigen;
          
        }
  
} 

 

// [[Rcpp::export]]
Eigen::Matrix<double, -1, -1>  Rcpp_shrink_matrix( const Eigen::Matrix<double, -1, -1>  &mat,
                                                   const double shrinkage_factor) {
   
   return shrink_hessian(mat, shrinkage_factor);
   
} 
 




template<typename T>
inline bool is_multiple(T ii, T interval) {
  
  if constexpr (std::is_integral<T>::value) {
    return ii % interval == 0;
  } else {
    return std::fmod(ii, interval) < 1e-6;
  }
  
}






// Function to clean up NaN/Inf and outlier elements in the Eigen vector
inline void clean_vector(Eigen::Matrix<double, -1, 1> &vec) {

  Eigen::Array<bool, -1, 1> valid_mask = vec.array().isFinite();
  Eigen::Matrix<double, -1, 1> valid_elements = vec(valid_mask);

  if (valid_elements.size() == 0) {
    throw std::runtime_error("All elements are NaN and/or Inf!");
  }


  double mean = valid_elements.mean();
  double stddev = std::sqrt((valid_elements.array() - mean).square().mean());


  for (int i = 0; i < vec.size(); ++i) {
    if (!std::isfinite(vec(i))) {
      vec(i) = mean;
    }
  }

  for (int i = 0; i < vec.size(); ++i) {
    if (std::abs(vec(i) - mean) > 10 * stddev) {
      if (vec(i) > mean) {
        vec(i) = valid_elements(valid_elements.array() <= mean + 10 * stddev).maxCoeff();
      } else {
        vec(i) = valid_elements(valid_elements.array() >= mean - 10 * stddev).minCoeff();
      }
    }
  }

}






// 
// 
// 
// 
// // [[Rcpp::export]]
// Rcpp::List     Rcpp_wrapper_fn_sample_HMC_multi_iter_single_thread(    const int chain_id,
//                                                                        const int seed,
//                                                                        const int n_iter,
//                                                                        const bool partitioned_HMC,
//                                                                        const std::string Model_type,
//                                                                        const bool sample_nuisance,
//                                                                        const bool force_autodiff,
//                                                                        const bool force_PartialLog,
//                                                                        const bool multi_attempts,
//                                                                        const int n_nuisance_to_track,
//                                                                        const Eigen::Matrix<double, -1, 1>  theta_main_vector_from_single_chain_input_from_R,
//                                                                        const Eigen::Matrix<double, -1, 1>  theta_us_vector_from_single_chain_input_from_R,
//                                                                        const Eigen::Matrix<int, -1, -1> y_Eigen_i,
//                                                                        const Rcpp::List  &Model_args_as_Rcpp_List,  ///// ALWAYS read-only
//                                                                        const Rcpp::List  &EHMC_args_as_Rcpp_List,
//                                                                        const Rcpp::List  &EHMC_Metric_as_Rcpp_List)  {
// 
//   int N = y_Eigen_i.rows();
//   int n_params_main = theta_main_vector_from_single_chain_input_from_R.size();
//   int n_us = theta_us_vector_from_single_chain_input_from_R.size();
//   int n_params = n_params_main + n_us;
// 
//   const bool burnin_indicator = false;
// 
//   //// convert lists to C++ structs
//   const Model_fn_args_struct     Model_args_as_cpp_struct =   convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List); ///// ALWAYS read-only
//   EHMC_fn_args_struct      EHMC_args_as_cpp_struct =    convert_R_List_EHMC_fn_args_struct(EHMC_args_as_Rcpp_List);
//   const EHMC_Metric_struct       EHMC_Metric_as_cpp_struct =  convert_R_List_EHMC_Metric_struct(EHMC_Metric_as_Rcpp_List);
// 
//   Stan_model_struct Stan_model_as_cpp_struct;
// 
//   if (Model_args_as_cpp_struct.model_so_file != "none" && Stan_model_as_cpp_struct.bs_model_ptr == nullptr) {
// 
//       Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(Model_args_as_cpp_struct.model_so_file,
//                                                              Model_args_as_cpp_struct.json_file_path,
//                                                              123);
// 
//   }
// 
// 
//   HMCResult result_input(n_params_main, n_us, N);
// 
//   result_input.main_theta_vec_0() = theta_main_vector_from_single_chain_input_from_R;
//   result_input.main_theta_vec() = theta_main_vector_from_single_chain_input_from_R;
// 
//   result_input.us_theta_vec_0() = theta_us_vector_from_single_chain_input_from_R;
//   result_input.us_theta_vec() = theta_us_vector_from_single_chain_input_from_R;
// 
//   HMC_output_single_chain  HMC_output_single_chain_i(n_iter, n_nuisance_to_track, n_params_main, n_us, N);
//   
//   // #if RNG_TYPE_CPP_STD == 1
//   //     std::mt19937 rng_i(seed); 
//   // #elif RNG_TYPE_pcg32 == 1
//   //     pcg32 rng_i(seed, 1);
//   // #elif RNG_TYPE_pcg64 == 1
//   //     pcg64 rng_i(seed, 1);
//   // #endif
//   
//   uint64_t seed_uint64_t = static_cast<uint64_t>(seed);
//   
//   #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
//     RNG_TYPE_dqrng global_rng = dqrng::generator<dqrng::xoshiro256plusplus>(); // seeded from R's RNG
//     std::unique_ptr<dqrng::random_64bit_generator> rng_i = global_rng->clone(1);
//   #elif RNG_TYPE_aqrng_pcg64 == 1
//     RNG_TYPE_dqrng rng_i = dqrng::generator<pcg64>(seed_uint64_t, 1);
//   #endif
// 
//   fn_sample_HMC_multi_iter_single_thread(       HMC_output_single_chain_i,
//                                                 result_input,
//                                                 burnin_indicator,
//                                                 1, // chain_id
//                                                 1, // current_iter
//                                                 seed,
//                                                 rng_i,
//                                                 n_iter,
//                                                 partitioned_HMC,
//                                                 Model_type,
//                                                 sample_nuisance,
//                                                 force_autodiff,
//                                                 force_PartialLog,
//                                                 multi_attempts,
//                                                 n_nuisance_to_track,
//                                                 y_Eigen_i,
//                                                 Model_args_as_cpp_struct,
//                                                 EHMC_args_as_cpp_struct,
//                                                 EHMC_Metric_as_cpp_struct,
//                                                 Stan_model_as_cpp_struct);
// 
//   // destroy Stan model object
//   if (Model_args_as_cpp_struct.model_so_file != "none" && Stan_model_as_cpp_struct.bs_model_ptr != nullptr) {
//     fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
//   }
// 
//    Rcpp::List out_list(3);
// 
//    out_list(0) = HMC_output_single_chain_i.trace_main();
//    out_list(1) = HMC_output_single_chain_i.trace_div();
//    out_list(2) = HMC_output_single_chain_i.trace_nuisance();
// 
//    return  out_list;
// 
// }
// 
// 
// 
// 
// 






// [[Rcpp::export]]
Rcpp::List    fn_compute_param_constrain_from_trace_parallel(   const std::vector<Eigen::Matrix<double, -1, -1>> unc_params_trace_input_main,
                                                                const std::vector<int> pars_indicies_to_track,
                                                                const int n_params_full,
                                                                const int n_params_main,
                                                                const std::string model_so_file,
                                                                const std::string json_file_path,
                                                                const bool use_disk = false,
                                                                const std::string trace_dir = "/tmp/constrain_traces"
) {
      
      const int n_chains = unc_params_trace_input_main.size();
      const int n_iter = unc_params_trace_input_main[0].cols();
      
      std::cout << "1: n_chains=" << n_chains 
                << " n_iter=" << n_iter 
                << " n_params_full=" << n_params_full 
                << " n_params_main=" << n_params_main << std::flush;
      
      if (use_disk == true) {
        std::filesystem::create_directories(trace_dir);
      }  
      
      // Eigen storage - use n_params_to_track, not n_params_full!
      // Only allocate if NOT using disk
      std::vector<Eigen::Matrix<double, -1, -1>> all_param_outs_eigen(n_chains);
      if (use_disk == false) {
        for (int i = 0; i < n_chains; ++i) {
          std::cout << "2: allocating chain " << i << std::flush;
          // all_param_outs_eigen[i].resize(n_params_main, n_iter);
          all_param_outs_eigen[i].resize(n_params_full, n_iter);
          all_param_outs_eigen[i].setZero();
        }
      }
      
      std::cout << "3: creating worker" << std::flush;
    
      //// Create worker
      ParamConstrainWorker worker(  n_chains,
                                    unc_params_trace_input_main,
                                    n_params_full,
                                    n_params_main,
                                    model_so_file,
                                    json_file_path,
                                    all_param_outs_eigen,
                                    use_disk, 
                                    trace_dir);
      
      std::cout << "4: worker created" << std::flush;
    
      //// Run parallel chains
      // RcppParallel::parallelFor(0, n_chains, worker, 1);
      RcppParallel::parallelFor(0, n_chains, worker);
      // worker(0, n_chains);  // run everything on main thread (for debug)
      
      std::cout << "5: done" << std::flush;
      
      //// IMPORTANT: Clean up model AFTER parallelFor completes
      worker.cleanup_model();
      
      if (use_disk == true) { // Return metadata only:
        
            return Rcpp::List::create(
              Rcpp::Named("use_disk") = true,
              Rcpp::Named("trace_dir") = trace_dir,
              Rcpp::Named("n_chains") = n_chains,
              Rcpp::Named("n_iter") = n_iter,
              Rcpp::Named("n_params") = n_params_full);
        
      } else {
        
            Rcpp::List out(n_chains);
            for (int i = 0; i < n_chains; ++i) {
              
                Rcpp::NumericMatrix mat(n_params_full, n_iter);
                const auto &eigen_mat = all_param_outs_eigen[i];
                
                for (int ii = 0; ii < n_iter; ++ii) {
                  for (int p = 0; p < n_params_full; ++p) {
                    mat(p, ii) = eigen_mat(p, ii);
                  }
                }
                
                out[i] = mat;  // immediately protected by Rcpp::List
                
            }
            
            return out;
        
      }
      
}


// [[Rcpp::export]]
Rcpp::List fn_compute_param_constrain_from_trace_v2(  const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_main,
                                                      const int n_params_full,
                                                      const int n_params_main,
                                                      const bool include_nuisance,
                                                      const std::string model_so_file,
                                                      const std::string json_file_path
) {

      const int n_chains = unc_params_trace_input_main.size();
      const int n_iter = unc_params_trace_input_main[0].cols();
      // const int n_params = n_nuisance + n_params_main;
      
      // Pre-allocate output - but only allocate if NOT using disk
      Rcpp::List out(n_chains);
        for (int kk = 0; kk < n_chains; ++kk) {
          out[kk] = Rcpp::NumericMatrix(n_params_full, n_iter);
        }
      
      // Process chains sequentially
      for (int kk = 0; kk < n_chains; ++kk) {
        
            // Rcpp::Rcout << "Chain " << kk + 1 << " / " << n_chains << std::endl;
               std::cout << "Chain " <<  kk + 1  << " / " << n_chains << std::endl;
            
            char* error_msg = nullptr;
            
            // Load model once per chain
            Stan_model_struct Stan_model_as_cpp_struct;
            bs_rng* bs_rng_object = nullptr;
            
            if (model_so_file != "none") {
                  Stan_model_as_cpp_struct = fn_load_Stan_model_and_data( model_so_file,
                                                                          json_file_path, 
                                                                          123 + kk);
                  bs_rng_object = Stan_model_as_cpp_struct.bs_rng_construct( 123 + kk, 
                                                                             &error_msg);
            }
            
            // Output matrix (only if NOT using disk)
            Rcpp::NumericMatrix out_mat;
            out_mat = Rcpp::as<Rcpp::NumericMatrix>(out[kk]);
            
            // Reusable vectors
            Eigen::Matrix<double, -1, 1> theta_unc(n_params_main);
            Eigen::Matrix<double, -1, 1> theta_con(n_params_full);
            
            for (int ii = 0; ii < n_iter; ++ii) {
              
                      theta_unc.setZero();
                      theta_unc = unc_params_trace_input_main[kk].col(ii);
                      
                      theta_con.setZero();
                      int result = Stan_model_as_cpp_struct.bs_param_constrain( Stan_model_as_cpp_struct.bs_model_ptr,
                                                                                true,
                                                                                true,
                                                                                theta_unc.data(),
                                                                                theta_con.data(),
                                                                                bs_rng_object,
                                                                                &error_msg);
                      
                      if (result != 0) {
                        Rcpp::stop("Constraint failed: %s", error_msg ? error_msg : "Unknown");
                      }
                     
                      for (int p = 0; p < n_params_full; ++p) {
                        out_mat(p, ii) = theta_con(p);
                      }
              
            }
            
            // Cleanup
            if (bs_rng_object) {
              Stan_model_as_cpp_struct.bs_rng_destruct(bs_rng_object);
            }
            fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
        
      }
      
      // if (use_disk) { // Return metadata only:
      //   
      //       return Rcpp::List::create(
      //         Rcpp::Named("use_disk") = true,
      //         Rcpp::Named("trace_dir") = trace_dir,
      //         Rcpp::Named("n_chains") = n_chains,
      //         Rcpp::Named("n_iter") = n_iter,
      //         Rcpp::Named("n_params") = n_params_full);
      //   
      // } else {
        
            return out;
            
      // }

}











// 
// // [[Rcpp::export]]
// Rcpp::List fn_compute_param_constrain_from_trace_v2_disk(
//     const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_main,
//     const std::string &nuisance_trace_dir,  // Path to nuisance files instead of matrices!
//     const int n_params_full,
//     const int n_nuisance,
//     const int n_params_main,
//     const bool include_nuisance,
//     const std::string model_so_file,
//     const std::string json_file_path,
//     const std::string trace_dir = "/tmp/constrain_traces"
// ) {
//   
//   const int n_chains = unc_params_trace_input_main.size();
//   const int n_iter = unc_params_trace_input_main[0].cols();
//   const int n_params = n_nuisance + n_params_main;
//   
//   std::filesystem::create_directories(trace_dir);
//   
//   for (int kk = 0; kk < n_chains; ++kk) {
//     
//     Rcpp::Rcout << "Chain " << kk + 1 << " / " << n_chains << std::endl;
//     
//     char* error_msg = nullptr;
//     
//     Stan_model_struct Stan_model_as_cpp_struct;
//     bs_rng* bs_rng_object = nullptr;
//     
//     if (model_so_file != "none") {
//       Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(model_so_file, json_file_path, 123 + kk);
//       bs_rng_object = Stan_model_as_cpp_struct.bs_rng_construct(123 + kk, &error_msg);
//     }
//     
//     // READ NUISANCE FOR THIS CHAIN FROM DISK
//     Eigen::Matrix<double, -1, -1> nuisance_chain;
//     if (include_nuisance) {
//       std::string nuisance_path = nuisance_trace_dir + "/chain_" + std::to_string(kk) + "_nuisance.bin";
//       std::ifstream in_file(nuisance_path, std::ios::binary);
//       if (in_file.is_open()) {
//         nuisance_chain.resize(n_nuisance, n_iter);
//         in_file.read(reinterpret_cast<char*>(nuisance_chain.data()), n_nuisance * n_iter * sizeof(double));
//         in_file.close();
//       }
//     }
//     
//     // Output file
//     std::string out_path = trace_dir + "/chain_" + std::to_string(kk) + "_constrained.bin";
//     std::ofstream out_file(out_path, std::ios::binary);
//     
//     Eigen::VectorXd theta_unc(n_params);
//     Eigen::VectorXd theta_con(n_params_full);
//     
//     for (int ii = 0; ii < n_iter; ++ii) {
//       
//       theta_unc.setZero();
//       theta_unc.tail(n_params_main) = unc_params_trace_input_main[kk].col(ii);
//       
//       if (include_nuisance) {
//         theta_unc.head(n_nuisance) = nuisance_chain.col(ii);
//       }
//       
//       theta_con.setZero();
//       int result = Stan_model_as_cpp_struct.bs_param_constrain(
//         Stan_model_as_cpp_struct.bs_model_ptr,
//         true, true,
//         theta_unc.data(),
//         theta_con.data(),
//         bs_rng_object,
//         &error_msg
//       );
//       
//       if (result != 0) {
//         Rcpp::stop("Constraint failed: %s", error_msg ? error_msg : "Unknown");
//       }
//       
//       out_file.write(
//         reinterpret_cast<const char*>(theta_con.data()),
//         n_params_full * sizeof(double)
//       );
//     }
//     
//     out_file.close();
//     
//     // nuisance_chain goes out of scope here - memory freed!
//     
//     if (bs_rng_object) {
//       Stan_model_as_cpp_struct.bs_rng_destruct(bs_rng_object);
//     }
//     fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
//   }
//   
//   return Rcpp::List::create(
//     Rcpp::Named("use_disk") = true,
//     Rcpp::Named("trace_dir") = trace_dir,
//     Rcpp::Named("n_chains") = n_chains,
//     Rcpp::Named("n_iter") = n_iter,
//     Rcpp::Named("n_params") = n_params_full
//   );
// }


 
//  
// // [[Rcpp::export]]
// Rcpp::List fn_compute_param_constrain_from_trace_v2_disk(    const std::vector<Eigen::Matrix<double, -1, -1>> &unc_params_trace_input_main,
//                                                              const int n_params_full,
//                                                              const int n_nuisance,
//                                                              const int n_params_main,
//                                                              const bool include_nuisance,
//                                                              const std::string model_so_file,
//                                                              const std::string json_file_path,
//                                                              const bool use_disk = false,
//                                                              const std::string trace_dir = "/tmp/constrain_traces" // OUTPUT directory
// ) {
//    
//        const int n_chains = unc_params_trace_input_main.size();
//        const int n_iter = unc_params_trace_input_main[0].cols();
//        const int n_params = n_nuisance + n_params_main;
//        
//        std::filesystem::create_directories(trace_dir);
//        ////
//        //// Process chains sequentially:
//        ////
//        for (int kk = 0; kk < n_chains; ++kk) {
//                
//                std::cout << "Chain " << kk + 1 << " / " << n_chains << std::endl;
//                
//                char* error_msg = nullptr;
//                
//                // Load model once per chain
//                Stan_model_struct Stan_model_as_cpp_struct;
//                bs_rng* bs_rng_object = nullptr;
//                
//                if (model_so_file != "none") {
//                    
//                        Stan_model_as_cpp_struct = fn_load_Stan_model_and_data( model_so_file,
//                                                                                json_file_path,
//                                                                                123 + kk);
//                    
//                        bs_rng_object = Stan_model_as_cpp_struct.bs_rng_construct( 123 + kk,
//                                                                                   &error_msg);
//                      
//                }
//                
//                // READ NUISANCE FOR THIS CHAIN FROM DISK - READ NUISANCE FROM nuisance_trace_dir (NOT trace_dir!)
//                Eigen::Matrix<double, -1, -1> nuisance_chain;
//                if (include_nuisance) {
//                  std::string nuisance_path = nuisance_trace_dir + "/chain_" + std::to_string(kk) + "_nuisance.bin";
//                  std::ifstream in_file(nuisance_path, std::ios::binary);
//                  
//                  if (!in_file.is_open()) {
//                    Rcpp::stop("Cannot open nuisance file: %s", nuisance_path.c_str());
//                  }
//                  
//                  nuisance_chain.resize(n_nuisance, n_iter);
//                  in_file.read(reinterpret_cast<char*>(nuisance_chain.data()), n_nuisance * n_iter * sizeof(double));
//                  in_file.close();
//                }
// 
//                // Output file (to trace_dir)
//                std::string out_path = trace_dir + "/chain_" + std::to_string(kk) + "_constrained.bin";
//                std::ofstream out_file(out_path, std::ios::binary);
//                
//                // // File for this chain
//                // std::ofstream out_file;
//                // std::string filepath = trace_dir + "/chain_" + std::to_string(kk) + "_constrained.bin";
//                // out_file.open(filepath, std::ios::binary);
//                
//                // Reusable vectors
//                Eigen::Matrix<double, -1, 1> theta_unc(n_params_main);
//                Eigen::Matrix<double, -1, 1> theta_con(n_params_full);
//                
//                for (int ii = 0; ii < n_iter; ++ii) {
//                  
//                        theta_unc.setZero();
//                        theta_unc = unc_params_trace_input_main[kk].col(ii);
//                        
//                        // if (include_nuisance) {
//                        //   theta_unc.head(n_nuisance) = unc_params_trace_input_nuisance[kk].col(ii);
//                        // theta_unc.head(n_nuisance) = nuisance_chain.col(ii);
//                        // }
//                        
//                        theta_con.setZero();
//                        int result = Stan_model_as_cpp_struct.bs_param_constrain( Stan_model_as_cpp_struct.bs_model_ptr,
//                                                                                  true,
//                                                                                  true,
//                                                                                  theta_unc.data(),
//                                                                                  theta_con.data(),
//                                                                                  bs_rng_object,
//                                                                                  &error_msg);
//                        
//                        if (result != 0) {
//                          Rcpp::stop("Constraint failed: %s", error_msg ? error_msg : "Unknown");
//                        }
//                          
//                        out_file.write(
//                          reinterpret_cast<const char*>(theta_con.data()),
//                          n_params_full * sizeof(double)
//                        );
//                  
//                }
//                
//                out_file.close();
//                
//                // Cleanup
//                if (bs_rng_object) {
//                  Stan_model_as_cpp_struct.bs_rng_destruct(bs_rng_object);
//                }
//                fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
//          
//        }
//          
//              return Rcpp::List::create(
//                Rcpp::Named("use_disk") = true,
//                Rcpp::Named("trace_dir") = trace_dir,
//                Rcpp::Named("n_chains") = n_chains,
//                Rcpp::Named("n_iter") = n_iter,
//                Rcpp::Named("n_params") = n_params_full);
//    
// }
// 
// 
//  







//// --------------------------------- RcpParallel  functions  ----------------------------------------------------------------------------------------------------------------------------------------------------------


// [[Rcpp::export]]
Rcpp::List Rcpp_fn_RcppParallel_EHMC_sampling( const int n_threads_R,
                                               const int seed_R,
                                               const int n_iter_R,
                                               const bool iter_one_by_one,
                                               const bool partitioned_HMC_R,
                                               const bool diffusion_HMC_R,
                                               const std::string Model_type_R,
                                               const bool sample_nuisance_R,
                                               const bool force_autodiff_R,
                                               const bool force_PartialLog_R,
                                               const bool multi_attempts_R,
                                               const int n_nuisance_to_track,
                                               const Eigen::Matrix<double, -1, -1> theta_main_vectors_all_chains_input_from_R,
                                               const Eigen::Matrix<double, -1, -1> theta_us_vectors_all_chains_input_from_R,
                                               const Eigen::Matrix<int, -1, -1> y_Eigen_R,
                                               const Rcpp::List Model_args_as_Rcpp_List,  ///// ALWAYS read-only
                                               const Rcpp::List EHMC_args_as_Rcpp_List,
                                               const Rcpp::List EHMC_Metric_as_Rcpp_List,
                                               const bool use_disk,
                                               const std::string trace_dir
) {
  
    //// key dimensions:
    const int n_params_main = theta_main_vectors_all_chains_input_from_R.rows();
    // const int n_nuisance = theta_us_vectors_all_chains_input_from_R.rows();
  
    //// main:
    Eigen::Matrix<double, -1, -1> theta_main_vectors_all_chains_output_to_R = theta_main_vectors_all_chains_input_from_R;  
    //// nuisance:
    Eigen::Matrix<double, -1, -1> theta_us_vectors_all_chains_output_to_R = theta_us_vectors_all_chains_input_from_R;
  
    //// convert lists to C++ structs:
    const Model_fn_args_struct Model_args_as_cpp_struct  = convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List); //// read-only
    const EHMC_fn_args_struct  EHMC_args_as_cpp_struct   = convert_R_List_EHMC_fn_args_struct(EHMC_args_as_Rcpp_List);
    const EHMC_Metric_struct   EHMC_Metric_as_cpp_struct = convert_R_List_EHMC_Metric_struct(EHMC_Metric_as_Rcpp_List); //// read-only
    //// replicate these structs for thread-safety as we will be modifying them for burnin:
    std::vector<Model_fn_args_struct> Model_args_as_cpp_struct_copies_R  = replicate_Model_fn_args_struct( Model_args_as_cpp_struct, n_threads_R); // read-only
    std::vector<EHMC_fn_args_struct>  EHMC_args_as_cpp_struct_copies_R   = replicate_EHMC_fn_args_struct( EHMC_args_as_cpp_struct, n_threads_R); // need to edit these !!
    std::vector<EHMC_Metric_struct>   EHMC_Metric_as_cpp_struct_copies_R = replicate_EHMC_Metric_struct( EHMC_Metric_as_cpp_struct, n_threads_R); // read-only
  
    ///// Traces:
    const int N = Model_args_as_cpp_struct.N;
    std::vector<Eigen::Matrix<double, -1, -1>> trace_output; // = vec_of_mats<double>(n_params_main, n_iter_R, n_threads_R);
    std::vector<Eigen::Matrix<double, -1, -1>> trace_output_divs; // = vec_of_mats<double>(1, n_iter_R, n_threads_R);
    // std::vector<Eigen::Matrix<double, -1, -1>> trace_output_nuisance; // = vec_of_mats<double>(n_nuisance_to_track, n_iter_R, n_threads_R);
    std::vector<Eigen::Matrix<double, -1, -1>> trace_output_log_lik; // = vec_of_mats<double>(N, n_iter_R, n_threads_R);  //// possibly dummy
  
    //// data copies:
    std::vector<Eigen::Matrix<int, -1, -1>> y_copies_R = vec_of_mats<int>(y_Eigen_R.rows(), y_Eigen_R.cols(), n_threads_R);
    for (int kk = 0; kk < n_threads_R; ++kk) {
       y_copies_R[kk] = y_Eigen_R;
    }
    // const Eigen::Matrix<int, -1, -1> y_copy = y_Eigen_R;

    // warmUpThreads(n_threads_R);
    
    const int global_seed_R_int = seed_R;
    const uint64_t global_seed_R_uint64_t = static_cast<uint64_t>(global_seed_R_int);

    //// create worker:
    RcppParallel_EHMC_sampling parallel_hmc_sampling(  n_threads_R,
                                                       global_seed_R_uint64_t,
                                                       n_iter_R,
                                                       partitioned_HMC_R,
                                                       diffusion_HMC_R,
                                                       Model_type_R,
                                                       sample_nuisance_R,
                                                       force_autodiff_R,
                                                       force_PartialLog_R,
                                                       multi_attempts_R,
                                                       ///// inputs:
                                                       theta_main_vectors_all_chains_output_to_R,
                                                       theta_us_vectors_all_chains_output_to_R,
                                                       ///// data:
                                                       y_copies_R,
                                                       // y_copy,
                                                       ///// structs:
                                                       Model_args_as_cpp_struct_copies_R,
                                                       // Model_args_as_cpp_struct,
                                                       EHMC_args_as_cpp_struct_copies_R,
                                                       EHMC_Metric_as_cpp_struct_copies_R,
                                                       // EHMC_Metric_as_cpp_struct,
                                                       ///// traces:
                                                       n_nuisance_to_track,
                                                       trace_output,
                                                       trace_output_divs,
                                                       // trace_output_nuisance,
                                                       trace_output_log_lik,
                                                       ////
                                                       use_disk,
                                                       trace_dir);
    //// Call parallelFor:
    tbb::task_arena arena(n_threads_R);
    arena.execute([&]() {
      RcppParallel::parallelFor(0, n_threads_R, parallel_hmc_sampling);
    });
    
    // // GET THESE BEFORE move_results_to_output() clears HMC_outputs!
    // const bool use_disk = parallel_hmc_sampling.is_using_disk();
    // const std::string trace_dir = parallel_hmc_sampling.get_trace_dir();

    ////  copy / store trace:
    parallel_hmc_sampling.move_results_to_output();
    
    // //// FREE INTERNAL STORAGE BEFORE RCPP CONVERSION
    // parallel_hmc_sampling.clear_internal_storage();

    // //// Reset everything:
    // parallel_hmc_sampling.reset();

    const double zero_dbl = 0.0;
   
    //// Return results:
    return Rcpp::List::create( trace_output,
                               trace_output_divs,
                               zero_dbl, // trace_output_nuisance,
                               zero_dbl,
                               zero_dbl,
                               trace_output_log_lik);

}









 








// [[Rcpp::export]]
Rcpp::List                                        fn_R_RcppParallel_EHMC_single_iter_burnin(  int n_threads_R,
                                                                                              int seed_R,
                                                                                              int n_iter_R,
                                                                                              int current_iter_R,
                                                                                              int n_adapt,
                                                                                              const bool burnin_indicator,
                                                                                              std::string Model_type_R,
                                                                                              bool sample_nuisance_R,
                                                                                              bool force_autodiff_R,
                                                                                              bool force_PartialLog_R,
                                                                                              bool multi_attempts_R,
                                                                                              const int n_nuisance_to_track,
                                                                                              const double max_eps_main,
                                                                                              const double max_eps_us,
                                                                                              bool partitioned_HMC_R,
                                                                                              bool diffusion_HMC_R,
                                                                                              const std::string metric_type_main,
                                                                                              double shrinkage_factor,
                                                                                              const std::string metric_type_nuisance,
                                                                                              const double tau_main_target,
                                                                                              const double tau_us_target,
                                                                                              const int clip_iter,
                                                                                              const int gap,
                                                                                              const bool main_L_manual,
                                                                                              const bool us_L_manual,
                                                                                              const int L_main_if_manual,
                                                                                              const int L_us_if_manual,
                                                                                              const int max_L,
                                                                                              const double tau_mult,
                                                                                              const double ratio_M_us,
                                                                                              const double ratio_Hess_main,
                                                                                              const int M_interval_width,
                                                                                              Eigen::Matrix<double, -1, -1>  theta_main_vectors_all_chains_input_from_R,
                                                                                              Eigen::Matrix<double, -1, -1>  velocity_main_vectors_all_chains_input_from_R,
                                                                                              Eigen::Matrix<double, -1, -1>  theta_us_vectors_all_chains_input_from_R,
                                                                                              Eigen::Matrix<double, -1, -1>  velocity_us_vectors_all_chains_input_from_R,
                                                                                              const Eigen::Matrix<int, -1, -1> y_Eigen_R,
                                                                                              const Rcpp::List Model_args_as_Rcpp_List,  ///// ALWAYS read-only
                                                                                              Rcpp::List EHMC_args_as_Rcpp_List,
                                                                                              Rcpp::List EHMC_Metric_as_Rcpp_List,
                                                                                              Rcpp::List EHMC_burnin_as_Rcpp_List
) {
  
  // key dimensions
  const int n_params_main = theta_main_vectors_all_chains_input_from_R.rows();
  const int n_nuisance = theta_us_vectors_all_chains_input_from_R.rows();

  //// create EMPTY OUTPUT / containers* to be filled (each col filled from different thread w/ each col corresponding to a different chain)
  Rcpp::NumericMatrix  theta_main_vectors_all_chains_output_to_R =  fn_convert_EigenMat_to_RcppMat_dbl(theta_main_vectors_all_chains_input_from_R);   // write to this
  Rcpp::NumericMatrix  velocity_main_vectors_all_chains_output_to_R =  fn_convert_EigenMat_to_RcppMat_dbl(velocity_main_vectors_all_chains_input_from_R);   // write to this
  Rcpp::NumericMatrix  other_main_out_vector_all_chains_output_to_R(10, n_threads_R);  // write to this

  //// nuisance:
  Rcpp::NumericMatrix  theta_us_vectors_all_chains_output_to_R  = fn_convert_EigenMat_to_RcppMat_dbl(theta_us_vectors_all_chains_input_from_R);
  Rcpp::NumericMatrix  velocity_us_vectors_all_chains_output_to_R  = fn_convert_EigenMat_to_RcppMat_dbl(velocity_us_vectors_all_chains_input_from_R);
  Rcpp::NumericMatrix  other_us_out_vector_all_chains_output_to_R(10, n_threads_R);  // write to this

  //// convert lists to C++ structs
  Model_fn_args_struct     Model_args_as_cpp_struct =   convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List); ///// ALWAYS read-only
  EHMC_fn_args_struct      EHMC_args_as_cpp_struct =    convert_R_List_EHMC_fn_args_struct(EHMC_args_as_Rcpp_List);
  EHMC_Metric_struct       EHMC_Metric_as_cpp_struct =  convert_R_List_EHMC_Metric_struct(EHMC_Metric_as_Rcpp_List);
  // EHMC_burnin_struct       EHMC_burnin_as_cpp_struct  = convert_R_List_EHMC_burnin_struct(EHMC_burnin_as_Rcpp_List);

  //// replicate these structs for thread-safety as we will be modifying them for burnin
  const std::vector<Model_fn_args_struct> Model_args_as_cpp_struct_copies_R =     replicate_Model_fn_args_struct( Model_args_as_cpp_struct,  n_threads_R); // read-only
  std::vector<EHMC_fn_args_struct>  EHMC_args_as_cpp_struct_copies_R =      replicate_EHMC_fn_args_struct(  EHMC_args_as_cpp_struct,   n_threads_R); // need to edit these !!
  const std::vector<EHMC_Metric_struct>   EHMC_Metric_as_cpp_struct_copies_R =    replicate_EHMC_Metric_struct(   EHMC_Metric_as_cpp_struct, n_threads_R); // read-only
  // const std::vector<EHMC_burnin_struct>   EHMC_burnin_as_cpp_struct_copies_R =    replicate_EHMC_burnin_struct(   EHMC_burnin_as_cpp_struct, n_threads_R); // read-only

  ///// data copies
  std::vector<Eigen::Matrix<int, -1, -1>> y_copies_R = vec_of_mats<int>(y_Eigen_R.rows(), y_Eigen_R.cols(), n_threads_R);
  for (int kk = 0; kk < n_threads_R; ++kk) {
    y_copies_R[kk] = y_Eigen_R;
  }
  // const Eigen::Matrix<int, -1, -1> y_copy_R = y_Eigen_R;

  /////// containers for burnin outputs ONLY (not needed for sampling) - stores: theta_0, theta_prop, velocity_0, velocity_prop
  Rcpp::NumericMatrix  theta_main_0_burnin_tau_adapt_all_chains_input_from_R(n_params_main, n_threads_R);  // write to this
  Rcpp::NumericMatrix  theta_main_prop_burnin_tau_adapt_all_chains_input_from_R(n_params_main, n_threads_R);  // write to this
  Rcpp::NumericMatrix  velocity_main_0_burnin_tau_adapt_all_chains_input_from_R(n_params_main, n_threads_R); // write to this
  Rcpp::NumericMatrix  velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R(n_params_main, n_threads_R); // write to this
  ////
  Rcpp::NumericMatrix  theta_us_0_burnin_tau_adapt_all_chains_input_from_R(n_nuisance, n_threads_R); // write to this
  Rcpp::NumericMatrix  theta_us_prop_burnin_tau_adapt_all_chains_input_from_R(n_nuisance, n_threads_R); // write to this
  Rcpp::NumericMatrix  velocity_us_0_burnin_tau_adapt_all_chains_input_from_R(n_nuisance, n_threads_R); // write to this
  Rcpp::NumericMatrix  velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R(n_nuisance, n_threads_R); // write to this

  int n_iter_for_fn_call = n_iter_R;
  if (burnin_indicator == true) n_iter_for_fn_call = 1;

  const int one = 1;
  
  // tbb::task_scheduler_init init(n_threads_R);
  warmUpThreads(n_threads_R);
  
  const int global_seed_R_int = seed_R;
  const uint64_t global_seed_R_uint64_t = static_cast<uint64_t>(global_seed_R_int);

  //// create worker
  RcppParallel_EHMC_burnin parallel_hmc_burnin(       n_threads_R,
                                                      global_seed_R_uint64_t,
                                                      one,
                                                      partitioned_HMC_R,
                                                      diffusion_HMC_R,
                                                      Model_type_R,
                                                      sample_nuisance_R,
                                                      force_autodiff_R,
                                                      force_PartialLog_R,
                                                      multi_attempts_R,
                                                      ///// inputs:
                                                      theta_main_vectors_all_chains_input_from_R,
                                                      velocity_main_vectors_all_chains_input_from_R,
                                                      //// inputs:
                                                      theta_us_vectors_all_chains_input_from_R,
                                                      velocity_us_vectors_all_chains_input_from_R,
                                                      ///// data:
                                                      y_copies_R,
                                                      // y_copy_R,
                                                      ///// input structs:
                                                      Model_args_as_cpp_struct_copies_R,
                                                      // Model_args_as_cpp_struct,
                                                      EHMC_args_as_cpp_struct_copies_R,
                                                      EHMC_Metric_as_cpp_struct_copies_R,
                                                      // EHMC_Metric_as_cpp_struct,
                                                      //// ---------  burnin-specific stuff:
                                                      current_iter_R,
                                                      // EHMC_burnin_as_cpp_struct_copies_R,
                                                      //// outputs:
                                                      theta_main_vectors_all_chains_output_to_R,
                                                      velocity_main_vectors_all_chains_output_to_R,
                                                      theta_us_vectors_all_chains_output_to_R,
                                                      velocity_us_vectors_all_chains_output_to_R,
                                                      //// other main outputs:
                                                      theta_main_0_burnin_tau_adapt_all_chains_input_from_R,
                                                      theta_main_prop_burnin_tau_adapt_all_chains_input_from_R,
                                                      velocity_main_0_burnin_tau_adapt_all_chains_input_from_R,
                                                      velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R,
                                                      other_main_out_vector_all_chains_output_to_R,
                                                      //// other nuisance outputs:
                                                      theta_us_0_burnin_tau_adapt_all_chains_input_from_R,
                                                      theta_us_prop_burnin_tau_adapt_all_chains_input_from_R,
                                                      velocity_us_0_burnin_tau_adapt_all_chains_input_from_R,
                                                      velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R,
                                                      other_us_out_vector_all_chains_output_to_R);

  //// Call parallelFor:
  parallelFor(0, n_threads_R, parallel_hmc_burnin);      

  ////  copy / store stuff needed for next burnin iteration:
  parallel_hmc_burnin.copy_results_to_output();
  
  // //// Reset everything
  // parallel_hmc_burnin.reset();
  
  const double zero_dbl = 0.0;


  if (   (burnin_indicator == false)   ) {

    // Return results
    return Rcpp::List::create(
      Rcpp::Named("theta_main_vectors_all_chains_output_to_R") = theta_main_vectors_all_chains_output_to_R,
      Rcpp::Named("velocity_main_vectors_all_chains_output_to_R") = velocity_main_vectors_all_chains_output_to_R,
      Rcpp::Named("other_main_out_vector_all_chains_output_to_R") = other_main_out_vector_all_chains_output_to_R,
      Rcpp::Named("theta_us_vectors_all_chains_output_to_R") = theta_us_vectors_all_chains_output_to_R,
      Rcpp::Named("velocity_us_vectors_all_chains_output_to_R") = velocity_us_vectors_all_chains_output_to_R,
      Rcpp::Named("other_us_out_vector_all_chains_output_to_R") = other_us_out_vector_all_chains_output_to_R,
      Rcpp::Named("theta_main_0_burnin_tau_adapt_all_chains_input_from_R") = theta_main_0_burnin_tau_adapt_all_chains_input_from_R,
      Rcpp::Named("theta_main_prop_burnin_tau_adapt_all_chains_input_from_R") = theta_main_prop_burnin_tau_adapt_all_chains_input_from_R,
      Rcpp::Named("velocity_main_0_burnin_tau_adapt_all_chains_input_from_R") = velocity_main_0_burnin_tau_adapt_all_chains_input_from_R,
      Rcpp::Named("velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R") = velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R,
      Rcpp::Named("theta_us_0_burnin_tau_adapt_all_chains_input_from_R") = theta_us_0_burnin_tau_adapt_all_chains_input_from_R,
      Rcpp::Named("theta_us_prop_burnin_tau_adapt_all_chains_input_from_R") = theta_us_prop_burnin_tau_adapt_all_chains_input_from_R,
      Rcpp::Named("velocity_us_0_burnin_tau_adapt_all_chains_input_from_R") = velocity_us_0_burnin_tau_adapt_all_chains_input_from_R,
      Rcpp::Named("velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R") = velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R
    );


  }

  
  
  // Return results
  return Rcpp::List::create(
    Rcpp::Named("theta_main_vectors_all_chains_output_to_R") = theta_main_vectors_all_chains_output_to_R,
    Rcpp::Named("velocity_main_vectors_all_chains_output_to_R") = velocity_main_vectors_all_chains_output_to_R,
    Rcpp::Named("other_main_out_vector_all_chains_output_to_R") = other_main_out_vector_all_chains_output_to_R,
    Rcpp::Named("theta_us_vectors_all_chains_output_to_R") = theta_us_vectors_all_chains_output_to_R,
    Rcpp::Named("velocity_us_vectors_all_chains_output_to_R") = velocity_us_vectors_all_chains_output_to_R,
    Rcpp::Named("other_us_out_vector_all_chains_output_to_R") = other_us_out_vector_all_chains_output_to_R,
    Rcpp::Named("theta_main_0_burnin_tau_adapt_all_chains_input_from_R") = theta_main_0_burnin_tau_adapt_all_chains_input_from_R,
    Rcpp::Named("theta_main_prop_burnin_tau_adapt_all_chains_input_from_R") = theta_main_prop_burnin_tau_adapt_all_chains_input_from_R,
    Rcpp::Named("velocity_main_0_burnin_tau_adapt_all_chains_input_from_R") = velocity_main_0_burnin_tau_adapt_all_chains_input_from_R,
    Rcpp::Named("velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R") = velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R,
    Rcpp::Named("theta_us_0_burnin_tau_adapt_all_chains_input_from_R") = theta_us_0_burnin_tau_adapt_all_chains_input_from_R,
    Rcpp::Named("theta_us_prop_burnin_tau_adapt_all_chains_input_from_R") = theta_us_prop_burnin_tau_adapt_all_chains_input_from_R,
    Rcpp::Named("velocity_us_0_burnin_tau_adapt_all_chains_input_from_R") = velocity_us_0_burnin_tau_adapt_all_chains_input_from_R,
    Rcpp::Named("velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R") = velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R
  );




}

















 


// [[Rcpp::export]]
Rcpp::List                                   Rcpp_fn_OpenMP_EHMC_sampling(        const int n_threads_R,
                                                                                  const int seed_R,
                                                                                  const int n_iter_R, 
                                                                                  const bool iter_one_by_one,
                                                                                  const bool partitioned_HMC_R,
                                                                                  const bool diffusion_HMC_R,
                                                                                  const std::string Model_type_R,
                                                                                  const bool sample_nuisance_R,
                                                                                  const bool force_autodiff_R,
                                                                                  const bool force_PartialLog_R,
                                                                                  const bool multi_attempts_R,
                                                                                  const int n_nuisance_to_track,
                                                                                  const Eigen::Matrix<double, -1, -1>  theta_main_vectors_all_chains_input_from_R,
                                                                                  const Eigen::Matrix<double, -1, -1>  theta_us_vectors_all_chains_input_from_R,
                                                                                  const Eigen::Matrix<int, -1, -1> y_Eigen_R,
                                                                                  const Rcpp::List Model_args_as_Rcpp_List,  //// ALWAYS read-only
                                                                                  const Rcpp::List EHMC_args_as_Rcpp_List,
                                                                                  const Rcpp::List EHMC_Metric_as_Rcpp_List
) {
            
            //// key dimensions:
            const int n_params_main = theta_main_vectors_all_chains_input_from_R.rows();
            const int n_us = theta_us_vectors_all_chains_input_from_R.rows();
            
            //// main:
            Eigen::Matrix<double, -1, -1> theta_main_vectors_all_chains_output_to_R =  theta_main_vectors_all_chains_input_from_R;  
            //// nuisance:
            Eigen::Matrix<double, -1, -1> theta_us_vectors_all_chains_output_to_R  = theta_us_vectors_all_chains_input_from_R;
            
            //// convert lists to C++ structs:
            const Model_fn_args_struct     Model_args_as_cpp_struct =   convert_R_List_to_Model_fn_args_struct(Model_args_as_Rcpp_List); ///// ALWAYS read-only
            const EHMC_fn_args_struct      EHMC_args_as_cpp_struct =    convert_R_List_EHMC_fn_args_struct(EHMC_args_as_Rcpp_List);
            const EHMC_Metric_struct       EHMC_Metric_as_cpp_struct =  convert_R_List_EHMC_Metric_struct(EHMC_Metric_as_Rcpp_List);
            //// replicate these structs for thread-safety as we will be modifying them for burnin:
            const std::vector<Model_fn_args_struct> Model_args_as_cpp_struct_copies_R =     replicate_Model_fn_args_struct( Model_args_as_cpp_struct,  n_threads_R); // read-only
            std::vector<EHMC_fn_args_struct>  EHMC_args_as_cpp_struct_copies_R = replicate_EHMC_fn_args_struct( EHMC_args_as_cpp_struct,   n_threads_R); // need to edit these !!
            const std::vector<EHMC_Metric_struct>   EHMC_Metric_as_cpp_struct_copies_R =    replicate_EHMC_Metric_struct(   EHMC_Metric_as_cpp_struct, n_threads_R); // read-only
            
            ///// Traces:
            const int N = Model_args_as_cpp_struct.N;
            std::vector<Eigen::Matrix<double, -1, -1>> trace_output =  vec_of_mats<double>(n_params_main, n_iter_R, n_threads_R);
            std::vector<Eigen::Matrix<double, -1, -1>> trace_output_divs =  vec_of_mats<double>(1, n_iter_R, n_threads_R);
            // std::vector<Eigen::Matrix<double, -1, -1>> trace_output_nuisance =  vec_of_mats<double>(n_nuisance_to_track, n_iter_R, n_threads_R);
            std::vector<Eigen::Matrix<double, -1, -1>> trace_output_log_lik = vec_of_mats<double>(N, n_iter_R, n_threads_R);  //// possibly dummy
            
            ///// data copies:
            std::vector<Eigen::Matrix<int, -1, -1>> y_copies_R = vec_of_mats<int>(y_Eigen_R.rows(), y_Eigen_R.cols(), n_threads_R);
            for (int kk = 0; kk < n_threads_R; ++kk) {
              y_copies_R[kk] = y_Eigen_R;
            }
            // const Eigen::Matrix<int, -1, -1> y_copy = y_Eigen_R;
            
            // warmUpThreads(n_threads_R);
            
            const int global_seed_R_int = seed_R;
            const uint64_t global_seed_R_uint64_t = static_cast<uint64_t>(global_seed_R_int);
            
            // std::vector<int> core_ids(n_threads_R);
            // for (int i = 0; i < n_threads_R; ++i) {
            //   core_ids[i] = i;
            // }
            // PinningObserver pin_observer(core_ids);
       
            EHMC_sampling_OpenMP(    n_threads_R,
                                     global_seed_R_uint64_t,
                                     n_iter_R,
                                     partitioned_HMC_R, 
                                     diffusion_HMC_R, 
                                     Model_type_R,
                                     sample_nuisance_R,
                                     force_autodiff_R,
                                     force_PartialLog_R,
                                     multi_attempts_R,
                                     ///// inputs:
                                     theta_main_vectors_all_chains_output_to_R,
                                     theta_us_vectors_all_chains_output_to_R,
                                     ///// data:
                                     y_copies_R,
                                     // y_copy,
                                     ///// structs:
                                     // Model_args_as_cpp_struct,
                                     Model_args_as_cpp_struct_copies_R,
                                     EHMC_args_as_cpp_struct_copies_R,
                                     // EHMC_Metric_as_cpp_struct,
                                     EHMC_Metric_as_cpp_struct_copies_R,
                                     ///// traces:
                                     n_nuisance_to_track,
                                     trace_output,
                                     trace_output_divs,
                                     // trace_output_nuisance,
                                     trace_output_log_lik);
          
          // ////  copy / store trace:
          // parallel_hmc_sampling.copy_results_to_output();
          // 
          // //// Reset everything:
          // parallel_hmc_sampling.reset();
          
          const double zero_dbl = 0.0;
          
          //// Return results:
          return Rcpp::List::create( trace_output,
                                     trace_output_divs,
                                     zero_dbl, //// trace_output_nuisance,
                                     zero_dbl,
                                     zero_dbl,
                                     trace_output_log_lik);
  
}
















 
 
