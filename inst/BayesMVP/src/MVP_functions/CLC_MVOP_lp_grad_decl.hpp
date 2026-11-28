#pragma once

//// -------------------------------------------------------------------------------------
//// CLC_MVOP_lp_grad_decl.hpp
////
//// Declarations of the CLC_MVOP log-density and gradient entry points compiled in
//// src/CLC_MVOP_lp_grad.cpp.
//// -------------------------------------------------------------------------------------

#include <Eigen/Dense>
#include <string>
#include <vector>


void fn_lp_and_grad_CLC_MVOP_Pinkney_AD_log_scale_InPlace_process(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const std::string &grad_option,
      const Model_fn_args_struct &Model_args_as_cpp_struct,
      std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
      const int n_threads_WCP
);


void fn_lp_grad_CLC_MVOP_NoLog_MD_and_AD_Inplace_process(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const std::string &grad_option,
      const Model_fn_args_struct &Model_args_as_cpp_struct,
      std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
      const int n_threads_WCP
);


void fn_lp_grad_CLC_MVOP_PartialLog_MD_and_AD_Inplace_process(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const std::string &grad_option,
      const Model_fn_args_struct &Model_args_as_cpp_struct,
      std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
      const int n_threads_WCP,
      const int n_chunks_multiplier
);


void fn_lp_grad_CLC_MVOP_multi_attempts_InPlace_process(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const std::string &grad_option,
      const Model_fn_args_struct &Model_args_as_cpp_struct,
      std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
      const int n_threads_WCP,
      const bool force_autodiff,
      const bool force_PartialLog
);
