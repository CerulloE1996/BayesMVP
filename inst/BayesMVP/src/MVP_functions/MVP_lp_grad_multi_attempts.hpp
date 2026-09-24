
#pragma once

 

#include <stan/math/rev.hpp>


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



 
#include <Eigen/Dense>
 



#include <unsupported/Eigen/SpecialFunctions>


 
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
 
 
 

 
using namespace Eigen;

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE




using std_vec_of_EigenVecs = std::vector<Eigen::Matrix<double, -1, 1>>;
using std_vec_of_EigenVecs_int = std::vector<Eigen::Matrix<int, -1, 1>>;

using std_vec_of_EigenMats = std::vector<Eigen::Matrix<double, -1, -1>>;
using std_vec_of_EigenMats_int = std::vector<Eigen::Matrix<int, -1, -1>>;

using two_layer_std_vec_of_EigenVecs =  std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>;
using two_layer_std_vec_of_EigenVecs_int = std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>;

using two_layer_std_vec_of_EigenMats = std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>;
using two_layer_std_vec_of_EigenMats_int = std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>;


using three_layer_std_vec_of_EigenVecs =  std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>>;
using three_layer_std_vec_of_EigenVecs_int =  std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>>;

using three_layer_std_vec_of_EigenMats = std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>>; 
using three_layer_std_vec_of_EigenMats_int = std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>>;




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// This model ccan be either the "standard" MVP model or the latent class MVP model (w/ 2 classes) for analysis of test accuracy data. 
inline void fn_lp_grad_MVP_multi_attempts_InPlace_process( Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                           const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                           const std::string grad_option,
                                                           const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                           std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                           const int n_threads_WCP,
                                                           const bool use_autodiff_fallback = false,
                                                           const bool use_standard_MVP_autodiff = false
) {
 
        int NaN_or_Inf_indicator = 0;  
        
        //bool use_autodiff = false;
        //bool use_PartialLog = false;
        
        Eigen::Matrix<double, -1, 1> out_mat_orig = out_mat; // store initial input 
        
        //// Attempts 1 and 2 are now inside try/catch, mirroring attempt 3: previously a
        //// Stan-math exception in attempt 1 or 2 (e.g. "Phi: x is nan" at |bound| ~ 39 on vect_type "Stan") escaped this function
        //// instead of moving on to the next attempt. Because the catch blocks catch std::exception, the settings strings are checked
        //// first, outside any try, so that a mistyped setting still stops with a clear message instead of being turned into a NaN.
        fn_check_settings_strings_before_multi_attempts(  Model_args_as_cpp_struct,
                                                          false,
                                                          "fn_lp_grad_MVP_multi_attempts_InPlace_process");
        
        ///// 1st attempt
        { //// if  ( (force_autodiff == false) && (force_PartialLog == false)  )  {   // NOT log-scale and NOT autodiff (least numerically stable but fastest)
              
              NaN_or_Inf_indicator = 0;  // Reset NaN_or_Inf indicator
              
              try {
                    fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process( out_mat,
                                                                               theta_main_vec_ref,
                                                                               theta_us_vec_ref,
                                                                               y_ref,
                                                                               grad_option,
                                                                               Model_args_as_cpp_struct,
                                                                               LC_MVP_ws_structs,
                                                                               n_threads_WCP);
                    
                    if (is_NaN_or_Inf_Eigen(out_mat)) { 
                      NaN_or_Inf_indicator = 1;
                    }
              } catch (const std::exception &) {
                    NaN_or_Inf_indicator = 1;
              }
          
        }
        
        ///// 2nd attempt (if first fails) - uses log-scale but NOT autodiff
        if ( (NaN_or_Inf_indicator == 1) ) { ///  if ( (NaN_or_Inf_indicator == 1) || ( (force_autodiff == false) && (force_PartialLog == true)   ) ) {

              NaN_or_Inf_indicator = 0;  // Reset main_div indicator
              out_mat = out_mat_orig;

              try {   //// see above
                    fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process(  out_mat,
                                                                                     theta_main_vec_ref,
                                                                                     theta_us_vec_ref,
                                                                                     y_ref,
                                                                                     grad_option,
                                                                                     Model_args_as_cpp_struct,
                                                                                     LC_MVP_ws_structs,
                                                                                     n_threads_WCP);

                    if (is_NaN_or_Inf_Eigen(out_mat)) {
                      NaN_or_Inf_indicator = 1;
                    }
              } catch (const std::exception &) {
                    NaN_or_Inf_indicator = 1;
              }

        }
        
        ///// Optional 3rd evaluation at EXACTLY the same position, before the caller consumes the gradient.
        ///// No position/velocity update, trajectory restart, random draw or Metropolis decision happens here.
        ///// The energy-divergence check belongs to the sampler, not this position-only evaluator.
        if (NaN_or_Inf_indicator == 1 && use_autodiff_fallback) {

              out_mat = out_mat_orig;
              try {
                    if (use_standard_MVP_autodiff) {
                          fn_lp_and_grad_std_MVP_Pinkney_AD_log_scale_InPlace_process( out_mat,
                                                                                         theta_main_vec_ref,
                                                                                         theta_us_vec_ref,
                                                                                         y_ref,
                                                                                         grad_option,
                                                                                         Model_args_as_cpp_struct,
                                                                                         LC_MVP_ws_structs,
                                                                                         n_threads_WCP);
                    } else {
                    fn_lp_and_grad_MVP_Pinkney_AD_log_scale_InPlace_process( out_mat,
                                                                              theta_main_vec_ref,
                                                                              theta_us_vec_ref,
                                                                              y_ref,
                                                                              grad_option,
                                                                              Model_args_as_cpp_struct,
                                                                              LC_MVP_ws_structs,
                                                                              n_threads_WCP);
                    }
                    NaN_or_Inf_indicator = is_NaN_or_Inf_Eigen(out_mat) ? 1 : 0;
              } catch (const std::exception &) {
                    NaN_or_Inf_indicator = 1;
              }

        }

        if ( (NaN_or_Inf_indicator == 1) ) {   //// all enabled attempts failed: leave a clean, unambiguous failure state
                    
              out_mat = out_mat_orig;                                        // restore caller's contents
              out_mat(0) = std::numeric_limits<double>::quiet_NaN();         // failure flag in the lp slot only
            
        }
        
        // ///// 3rd attempt (if second fails)
        // if ( (NaN_or_Inf_indicator == 1) ) {    /// if ( (NaN_or_Inf_indicator == 1) ||  ( (force_autodiff == true) && (force_PartialLog == true)  )  )  {
        // 
        //       NaN_or_Inf_indicator = 0;  // Reset main_div indicator
        //       out_mat = out_mat_orig;
        // 
        //       fn_lp_and_grad_MVP_Pinkney_AD_log_scale_InPlace_process(   out_mat,
        //                                                                  theta_main_vec_ref,
        //                                                                  theta_us_vec_ref,
        //                                                                  y_ref,
        //                                                                  grad_option,
        //                                                                  Model_args_as_cpp_struct,
        //                                                                  LC_MVP_ws_structs,
        //                                                                  n_threads_WCP);
        // 
        //       if (is_NaN_or_Inf_Eigen(out_mat)) {
        //         NaN_or_Inf_indicator = 1;
        //       }
        // 
        // }

}




inline void fn_lp_grad_MVOP_multi_attempts_InPlace_process( Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                            const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                            const std::string grad_option,
                                                            const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                            std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                            const int n_threads_WCP,
                                                            const bool use_autodiff_fallback = false
) {
  
        int NaN_or_Inf_indicator = 0;  
        
        Eigen::Matrix<double, -1, 1> out_mat_orig = out_mat; // store initial input 
        
        //// as in fn_lp_grad_MVP_multi_attempts_InPlace_process above: settings checked first,
        //// then attempts 1 and 2 inside try/catch, mirroring the autodiff attempt.
        fn_check_settings_strings_before_multi_attempts(  Model_args_as_cpp_struct,
                                                          false,
                                                          "fn_lp_grad_MVOP_multi_attempts_InPlace_process");
        
        ///// 1st attempt
        { //// if  ( (force_autodiff == false) && (force_PartialLog == false)  )  {   // NOT log-scale and NOT autodiff (least numerically stable but fastest)

              NaN_or_Inf_indicator = 0;  // Reset NaN_or_Inf indicator
              
              try {
                    fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(  out_mat,
                                                                                 theta_main_vec_ref,
                                                                                 theta_us_vec_ref,
                                                                                 y_ref,
                                                                                 grad_option,
                                                                                 Model_args_as_cpp_struct,
                                                                                 LC_MVP_ws_structs,
                                                                                 n_threads_WCP);

                    if (is_NaN_or_Inf_Eigen(out_mat)) {
                      NaN_or_Inf_indicator = 1;
                    }
              } catch (const std::exception &) {
                    NaN_or_Inf_indicator = 1;
              }

        }
        
        // ///// 2nd attempt (if first fails) - uses log-scale but NOT autodiff
        if ( (NaN_or_Inf_indicator == 1) ) { ///  if ( (NaN_or_Inf_indicator == 1) || ( (force_autodiff == false) && (force_PartialLog == true)   ) ) {

              NaN_or_Inf_indicator = 0;  // Reset main_div indicator
              out_mat = out_mat_orig;

              try {   //// see above
                    fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_Inplace_process(   out_mat,
                                                                                       theta_main_vec_ref,
                                                                                       theta_us_vec_ref,
                                                                                       y_ref,
                                                                                       grad_option,
                                                                                       Model_args_as_cpp_struct,
                                                                                       LC_MVP_ws_structs,
                                                                                       n_threads_WCP);

                    if (is_NaN_or_Inf_Eigen(out_mat)) {
                      NaN_or_Inf_indicator = 1;
                    }
              } catch (const std::exception &) {
                    NaN_or_Inf_indicator = 1;
              }

        }
        
        ///// Optional autodiff evaluation at the same position; no integrator or random-number calls.
        if (NaN_or_Inf_indicator == 1 && use_autodiff_fallback) {

              out_mat = out_mat_orig;
              try {
                    fn_lp_and_grad_MVOP_Pinkney_AD_log_scale_InPlace_process( out_mat,
                                                                               theta_main_vec_ref,
                                                                               theta_us_vec_ref,
                                                                               y_ref,
                                                                               grad_option,
                                                                               Model_args_as_cpp_struct,
                                                                               LC_MVP_ws_structs,
                                                                               n_threads_WCP);
                    NaN_or_Inf_indicator = is_NaN_or_Inf_Eigen(out_mat) ? 1 : 0;
              } catch (const std::exception &) {
                    NaN_or_Inf_indicator = 1;
              }
        }

        if (NaN_or_Inf_indicator == 1) {
              out_mat = out_mat_orig;
              out_mat(0) = std::numeric_limits<double>::quiet_NaN();
        }

        // /// 3rd attempt (if second fails)
        // if ( (NaN_or_Inf_indicator == 1) ) {
        //       
        //       NaN_or_Inf_indicator = 0;  // Reset main_div indicator
        //       out_mat = out_mat_orig;
        //       
        //       fn_lp_and_grad_MVOP_Pinkney_AD_log_scale_InPlace_process(  out_mat, 
        //                                                                  theta_main_vec_ref,
        //                                                                  theta_us_vec_ref,
        //                                                                  y_ref,
        //                                                                  grad_option,
        //                                                                  Model_args_as_cpp_struct,
        //                                                                  LC_MVP_ws_structs,
        //                                                                  n_threads_WCP);
        //       
        //       if (is_NaN_or_Inf_Eigen(out_mat)) { 
        //         NaN_or_Inf_indicator = 1; 
        //       } 
        //   
        // }
  
}
