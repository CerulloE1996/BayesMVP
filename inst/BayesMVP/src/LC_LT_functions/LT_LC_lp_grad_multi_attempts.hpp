
#pragma once


 

 

// #include <stan/math/rev.hpp>
// ////
// #include <stan/math/prim/fun/Eigen.hpp>
// #include <stan/math/prim/fun/typedefs.hpp>
// #include <stan/math/prim/fun/value_of_rec.hpp>
// #include <stan/math/prim/err/check_pos_definite.hpp>
// #include <stan/math/prim/err/check_square.hpp>
// #include <stan/math/prim/err/check_symmetric.hpp>
// ////
// #include <stan/math/prim/fun/cholesky_decompose.hpp>
// #include <stan/math/prim/fun/sqrt.hpp>
// #include <stan/math/prim/fun/log.hpp>
// #include <stan/math/prim/fun/transpose.hpp>
// #include <stan/math/prim/fun/dot_product.hpp>
// #include <stan/math/prim/fun/norm2.hpp>
// #include <stan/math/prim/fun/diagonal.hpp>
// #include <stan/math/prim/fun/cholesky_decompose.hpp>
// #include <stan/math/prim/fun/eigenvalues_sym.hpp>
// #include <stan/math/prim/fun/diag_post_multiply.hpp>
// ////
// #include <stan/math/prim/prob/multi_normal_cholesky_lpdf.hpp>
// #include <stan/math/prim/prob/lkj_corr_cholesky_lpdf.hpp>
// #include <stan/math/prim/prob/weibull_lpdf.hpp>
// #include <stan/math/prim/prob/gamma_lpdf.hpp>
// #include <stan/math/prim/prob/beta_lpdf.hpp>



 
#include <Eigen/Dense>
 



#include <unsupported/Eigen/SpecialFunctions>


 

 

// 
// using std_vec_of_EigenVecs = std::vector<Eigen::Matrix<double, -1, 1>>;
// using std_vec_of_EigenVecs_int = std::vector<Eigen::Matrix<int, -1, 1>>;
// 
// using std_vec_of_EigenMats = std::vector<Eigen::Matrix<double, -1, -1>>;
// using std_vec_of_EigenMats_int = std::vector<Eigen::Matrix<int, -1, -1>>;
// 
// using two_layer_std_vec_of_EigenVecs =  std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>;
// using two_layer_std_vec_of_EigenVecs_int = std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>;
// 
// using two_layer_std_vec_of_EigenMats = std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>;
// using two_layer_std_vec_of_EigenMats_int = std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>;
// 
// 
// using three_layer_std_vec_of_EigenVecs =  std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>>;
// using three_layer_std_vec_of_EigenVecs_int =  std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>>;
// 
// using three_layer_std_vec_of_EigenMats = std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>>; 
// using three_layer_std_vec_of_EigenMats_int = std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>>;
// 
// 
// 







 

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////







 
  
  

///// This model ccan be either the "standard" MVP model or the latent class MVP model (w/ 2 classes) for analysis of test accuracy data. 
void                             fn_lp_grad_LT_LC_multi_attempts_InPlace_process(     Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
                                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                      const std::string grad_option,
                                                                                      const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                      const bool use_autodiff_fallback = true
) {

 
 
 
  int NaN_or_Inf_indicator = 0;  
  
  //// bool use_autodiff = false;
  //// bool use_PartialLog = false;
  
  Eigen::Matrix<double, -1, 1> out_mat_orig = out_mat; // store initial input 
  
  ///// 1st attempt
  { //// if  ( (force_autodiff == false) && (force_PartialLog == false)  )  {   // NOT log-scale and NOT autodiff (least numerically stable but fastest)
        
        NaN_or_Inf_indicator = 0;  // Reset NaN_or_Inf indicator
    
        fn_lp_grad_LT_LC_NoLog_MD_and_AD_InPlace_process(  out_mat, 
                                                           theta_main_vec_ref,
                                                           theta_us_vec_ref,
                                                           y_ref,
                                                           grad_option,
                                                           Model_args_as_cpp_struct);
        
        if (is_NaN_or_Inf_Eigen(out_mat)) { 
          NaN_or_Inf_indicator = 1;
        } 
    
  } 
  
  // // NOTE: LOG-SCALE FN NOT YET WORKING FOR THE LATENT_TRAIT MODEL (LT_b's part) - SO JUST GO STRAIGHT TO AD FUNCTION
  // ///// 2nd attempt (if first fails) - uses log-scale but NOT autodiff
  // if   (NaN_or_Inf_indicator == 1) {  ///  if ( (NaN_or_Inf_indicator == 1) || ( (force_autodiff == false) && (force_PartialLog == true)   ) ) {
  // 
  //       NaN_or_Inf_indicator = 0;  // Reset main_div indicator
  //       out_mat = out_mat_orig;
  // 
  //       fn_lp_grad_LT_LC_PartialLog_MD_and_AD_InPlace_process(   out_mat,
  //                                                                theta_main_vec_ref,
  //                                                                theta_us_vec_ref,
  //                                                                y_ref,
  //                                                                grad_option,
  //                                                                Model_args_as_cpp_struct);
  // 
  //       if (is_NaN_or_Inf_Eigen(out_mat)) {
  //         NaN_or_Inf_indicator = 1;
  //       }
  // 
  // }
  
  ///// 3rd attempt (if second fails)
  if (NaN_or_Inf_indicator == 1 && use_autodiff_fallback) {  //// preserve the existing standard -> autodiff fallback unless explicitly disabled
    
        NaN_or_Inf_indicator = 0;  // Reset main_div indicator
        out_mat = out_mat_orig;
        
        try {
        fn_lp_and_grad_LC_LT_AD_log_scale_InPlace_process(     out_mat, 
                                                               theta_main_vec_ref,
                                                               theta_us_vec_ref,
                                                               y_ref,
                                                               grad_option,
                                                               Model_args_as_cpp_struct);
        
        if (is_NaN_or_Inf_Eigen(out_mat)) { 
          NaN_or_Inf_indicator = 1;
        } 
        } catch (const std::exception &) {
          NaN_or_Inf_indicator = 1;
        }
    
  }

 
 
 
  if (NaN_or_Inf_indicator == 1) {
        out_mat = out_mat_orig;
        out_mat(0) = std::numeric_limits<double>::quiet_NaN();
  }

}
