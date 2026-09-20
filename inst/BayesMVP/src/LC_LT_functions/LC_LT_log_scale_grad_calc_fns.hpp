#pragma once


 

#include <Eigen/Dense>
 

 
 



#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE





 
 
 inline void fn_latent_trait_compute_bs_grad_log_scale(   const std::vector<int> &n_problem_array,
                                                          const std::vector<std::vector<int>> &problem_index_array,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>>   grad_pi_wrt_b_raw,  
                                                          Eigen::Ref<Eigen::Matrix<double, -1, 1>>    log_abs_bs_grad_array_col_for_each_n,   
                                                          Eigen::Ref<Eigen::Matrix<double, -1, 1>>    sign_bs_grad_array_col_for_each_n,   
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>>   log_abs_deriv_Bound_Z_x_L,   
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>>   sign_deriv_Bound_Z_x_L,  
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>>   log_abs_deriv_Bound_Z_x_L_comp,   
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>>   sign_deriv_Bound_Z_x_L_comp,  
                                                          const int c,   
                                                          const std::vector<Eigen::Matrix<double, -1, -1>> &log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double,
                                                          const std::vector<Eigen::Matrix<double, -1, -1>> &sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_log_Bound_Z,   
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_sign_Bound_Z,   
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_log_Z_std_norm,   
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_sign_Z_std_norm, 
                                                          const Eigen::Matrix<double, -1, -1> &L_Omega_double,
                                                          const Eigen::Matrix<double, -1, -1> &log_abs_L_Omega_double,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_log_phi_Bound_Z,   
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_log_phi_Z_recip,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_log_abs_y_sign_chunk,  
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_sign_y_sign_chunk,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_log_abs_y_m_y_sign_x_u,  
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_sign_y_m_y_sign_x_u,   
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_y1_log_prob,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_log_prob_rowwise_prod_temp,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_log_common_grad_term_1,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> B_log_abs_grad_bound_z,   
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> B_sign_grad_bound_z,   
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> B_log_abs_grad_Phi_bound_z,   
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> B_sign_grad_Phi_bound_z,   
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> B_log_abs_z_grad_term,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> B_sign_z_grad_term,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> B_log_abs_grad_prob,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> B_sign_grad_prob,  
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> B_log_abs_derivs_chain_container_vec_comp,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> B_sign_derivs_chain_container_vec_comp,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, 1>>  log_sum_result,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, 1>>  sign_sum_result,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> log_terms,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> sign_terms,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, 1>>  container_max_logs,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, 1>>  container_sum_exp_signed,
                                                          const Model_fn_args_struct &Model_args_as_cpp_struct
 ) {
   
   const int  n_class = Model_args_as_cpp_struct.Model_args_ints(1);
   const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
   
   const int n_tests = B_log_Bound_Z.cols();
   const int chunk_size = B_log_Bound_Z.rows();
   
   ///////////////////////// Process each diagonal element
   //////// Last diagonal (t1 = n_tests - 1)
   {
     int t1 = n_tests - 1; 
     
     if (n_problem_array[t1] > 0) {
       
       // Resize containers for this problem size
       log_abs_deriv_Bound_Z_x_L_comp.resize(n_problem_array[t1], n_tests);
       sign_deriv_Bound_Z_x_L_comp.resize(n_problem_array[t1], n_tests);
       log_sum_result.resize(n_problem_array[t1]);
       sign_sum_result.resize(n_problem_array[t1]);
       log_terms.resize(n_problem_array[t1], n_tests);
       sign_terms.resize(n_problem_array[t1], n_tests);
       container_max_logs.resize(n_problem_array[t1]);
       container_sum_exp_signed.resize(n_problem_array[t1]);
       
       double log_abs_deriv_L_T_T_inv = -2.0 * log_abs_L_Omega_double(t1, t1) + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1);
       double sign_deriv_L_T_T_inv = -1.0 * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1);
       
       // Compute deriv_Bound_Z_x_L for problem indices
       // for (int t = 0; t < t1; t++) {
       //   log_abs_deriv_Bound_Z_x_L_comp.col(t) = B_log_Z_std_norm.col(t)(problem_index_array[t1]) + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
       //   sign_deriv_Bound_Z_x_L_comp.col(t).array() = B_sign_Z_std_norm.col(t)(problem_index_array[t1]).array() * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
       // }
       for (int t = 0; t < t1; t++) {
         Eigen::Matrix<double, -1, 1> temp_log = B_log_Z_std_norm.col(t)(problem_index_array[t1]);
         log_abs_deriv_Bound_Z_x_L_comp.col(t).array() = temp_log.array() + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
         sign_deriv_Bound_Z_x_L_comp.col(t).array() = B_sign_Z_std_norm.col(t)(problem_index_array[t1]).array() * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
       }
       
       // Create temporaries EXACTLY like MVP function
       Eigen::Matrix<double, -1, 1> temp_log_abs_deriv = Eigen::Matrix<double, -1, 1>::Constant(n_problem_array[t1], -700.0);
       Eigen::Matrix<double, -1, 1> temp_sign_deriv = Eigen::Matrix<double, -1, 1>::Ones(n_problem_array[t1]);
       
       // Sum using log_sum_exp
       log_abs_sum_exp_general_v2(log_abs_deriv_Bound_Z_x_L_comp.leftCols(t1),
                                  sign_deriv_Bound_Z_x_L_comp.leftCols(t1),
                                  vect_type, vect_type,
                                  temp_log_abs_deriv,
                                  temp_sign_deriv,
                                  container_max_logs,
                                  container_sum_exp_signed);
       
       log_abs_deriv_Bound_Z_x_L.col(0)(problem_index_array[t1]) = temp_log_abs_deriv;
       sign_deriv_Bound_Z_x_L.col(0)(problem_index_array[t1]) = temp_sign_deriv;
       sign_deriv_Bound_Z_x_L.col(0)(problem_index_array[t1]).array() *= -1.0;
       
       // Compute grad_bound_z
       log_terms.col(0).array() = log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1)(problem_index_array[t1]).array() + log_abs_L_Omega_double(t1, t1);
       log_terms.col(1).array() = -log_abs_L_Omega_double(t1, t1) + log_abs_deriv_Bound_Z_x_L.col(0)(problem_index_array[t1]).array();
       sign_terms.col(0).array() = sign_deriv_L_T_T_inv * B_sign_Bound_Z.col(t1)(problem_index_array[t1]).array() * stan::math::sign(L_Omega_double(t1, t1));
       sign_terms.col(1) = stan::math::sign(1.0 / L_Omega_double(t1, t1)) * sign_deriv_Bound_Z_x_L.col(0)(problem_index_array[t1]);
       
       // Create temporaries for grad_bound_z
       Eigen::Matrix<double, -1, 1> temp_log_abs_grad_bound = Eigen::Matrix<double, -1, 1>::Constant(n_problem_array[t1], -700.0);
       Eigen::Matrix<double, -1, 1> temp_sign_grad_bound = Eigen::Matrix<double, -1, 1>::Ones(n_problem_array[t1]);
       
       log_abs_sum_exp_general_v2(log_terms.leftCols(2),
                                  sign_terms.leftCols(2),
                                  vect_type, vect_type,
                                  temp_log_abs_grad_bound,
                                  temp_sign_grad_bound,
                                  container_max_logs,
                                  container_sum_exp_signed);
       
       B_log_abs_grad_bound_z.col(0)(problem_index_array[t1]) = temp_log_abs_grad_bound;
       B_sign_grad_bound_z.col(0)(problem_index_array[t1]) = temp_sign_grad_bound;
       
       // Compute grad_Phi_bound_z and grad_prob
       B_log_abs_grad_Phi_bound_z.col(0)(problem_index_array[t1]).array() = B_log_phi_Bound_Z.col(t1)(problem_index_array[t1]).array() + B_log_abs_grad_bound_z.col(0)(problem_index_array[t1]).array();
       B_sign_grad_Phi_bound_z.col(0)(problem_index_array[t1]) = B_sign_grad_bound_z.col(0)(problem_index_array[t1]);
       
       B_log_abs_grad_prob.col(0)(problem_index_array[t1]).array() = B_log_abs_y_sign_chunk.col(t1)(problem_index_array[t1]).array() + B_log_abs_grad_Phi_bound_z.col(0)(problem_index_array[t1]).array();
       B_sign_grad_prob.col(0)(problem_index_array[t1]).array() = -1.0 * B_sign_y_sign_chunk.col(t1)(problem_index_array[t1]).array() * B_sign_grad_Phi_bound_z.col(0)(problem_index_array[t1]).array();
       
       // Final gradient computation
       log_abs_bs_grad_array_col_for_each_n(problem_index_array[t1]).array() = B_log_common_grad_term_1.col(t1)(problem_index_array[t1]).array() + B_log_abs_grad_prob.col(0)(problem_index_array[t1]).array();
       sign_bs_grad_array_col_for_each_n(problem_index_array[t1]) = B_sign_grad_prob.col(0)(problem_index_array[t1]);
       
       LogSumVecSingedResult log_sum_vec_signed_struct = log_sum_vec_signed_v1( log_abs_bs_grad_array_col_for_each_n(problem_index_array[t1]),
                                                                                sign_bs_grad_array_col_for_each_n(problem_index_array[t1]), 
                                                                                vect_type);
       
       grad_pi_wrt_b_raw(c, t1) += stan::math::exp(log_sum_vec_signed_struct.log_sum) * log_sum_vec_signed_struct.sign;
       
     }
   }
   
   //////// Second-to-last diagonal (t1 = n_tests - 2)
   {
     int t1 = n_tests - 2; 
     
     if (n_problem_array[t1] > 0) {
       
       log_abs_deriv_Bound_Z_x_L_comp.resize(n_problem_array[t1], n_tests);
       sign_deriv_Bound_Z_x_L_comp.resize(n_problem_array[t1], n_tests);
       log_sum_result.resize(n_problem_array[t1]);
       sign_sum_result.resize(n_problem_array[t1]);
       log_terms.resize(n_problem_array[t1], n_tests);
       sign_terms.resize(n_problem_array[t1], n_tests);
       container_max_logs.resize(n_problem_array[t1]);
       container_sum_exp_signed.resize(n_problem_array[t1]);
       
       double log_abs_deriv_L_T_T_inv = -2.0 * log_abs_L_Omega_double(t1, t1) + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1);
       double sign_deriv_L_T_T_inv = -1.0 * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1);
       
       // for (int t = 0; t < t1; t++) {
       //   log_abs_deriv_Bound_Z_x_L_comp.col(t) = B_log_Z_std_norm.col(t)(problem_index_array[t1]) + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
       //   sign_deriv_Bound_Z_x_L_comp.col(t).array() = B_sign_Z_std_norm.col(t)(problem_index_array[t1]).array() * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
       // }
       for (int t = 0; t < t1; t++) {
         Eigen::Matrix<double, -1, 1> temp_log = B_log_Z_std_norm.col(t)(problem_index_array[t1]);
         log_abs_deriv_Bound_Z_x_L_comp.col(t).array() = temp_log.array() + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
         sign_deriv_Bound_Z_x_L_comp.col(t).array() = B_sign_Z_std_norm.col(t)(problem_index_array[t1]).array() * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
       }
       
       // Create temporaries
       Eigen::Matrix<double, -1, 1> temp_log_abs_deriv_0 = Eigen::Matrix<double, -1, 1>::Constant(n_problem_array[t1], -700.0);
       Eigen::Matrix<double, -1, 1> temp_sign_deriv_0 = Eigen::Matrix<double, -1, 1>::Ones(n_problem_array[t1]);
       
       log_abs_sum_exp_general_v2(log_abs_deriv_Bound_Z_x_L_comp.leftCols(t1), sign_deriv_Bound_Z_x_L_comp.leftCols(t1),
                                  vect_type, vect_type, temp_log_abs_deriv_0,
                                  temp_sign_deriv_0,
                                  container_max_logs, container_sum_exp_signed);
       
       log_abs_deriv_Bound_Z_x_L.col(0)(problem_index_array[t1]) = temp_log_abs_deriv_0;
       sign_deriv_Bound_Z_x_L.col(0)(problem_index_array[t1]) = temp_sign_deriv_0;
       sign_deriv_Bound_Z_x_L.col(0)(problem_index_array[t1]).array() *= -1.0;
       
       log_terms.col(0).array() = log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1)(problem_index_array[t1]).array() + log_abs_L_Omega_double(t1, t1);
       log_terms.col(1).array() = -log_abs_L_Omega_double(t1, t1) + log_abs_deriv_Bound_Z_x_L.col(0)(problem_index_array[t1]).array();
       sign_terms.col(0).array() = sign_deriv_L_T_T_inv * B_sign_Bound_Z.col(t1)(problem_index_array[t1]).array() * stan::math::sign(L_Omega_double(t1, t1));
       sign_terms.col(1) = stan::math::sign(1.0 / L_Omega_double(t1, t1)) * sign_deriv_Bound_Z_x_L.col(0)(problem_index_array[t1]);
       
       Eigen::Matrix<double, -1, 1> temp_log_abs_grad_bound_0 = Eigen::Matrix<double, -1, 1>::Constant(n_problem_array[t1], -700.0);
       Eigen::Matrix<double, -1, 1> temp_sign_grad_bound_0 = Eigen::Matrix<double, -1, 1>::Ones(n_problem_array[t1]);
       
       log_abs_sum_exp_general_v2(log_terms.leftCols(2), sign_terms.leftCols(2), vect_type, vect_type,
                                  temp_log_abs_grad_bound_0, 
                                  temp_sign_grad_bound_0,
                                  container_max_logs, container_sum_exp_signed);
       
       B_log_abs_grad_bound_z.col(0)(problem_index_array[t1]) = temp_log_abs_grad_bound_0;
       B_sign_grad_bound_z.col(0)(problem_index_array[t1]) = temp_sign_grad_bound_0;
       
       B_log_abs_grad_Phi_bound_z.col(0)(problem_index_array[t1]).array() = B_log_phi_Bound_Z.col(t1)(problem_index_array[t1]).array() + B_log_abs_grad_bound_z.col(0)(problem_index_array[t1]).array();
       B_sign_grad_Phi_bound_z.col(0)(problem_index_array[t1]) = B_sign_grad_bound_z.col(0)(problem_index_array[t1]);
       
       B_log_abs_grad_prob.col(0)(problem_index_array[t1]).array() = B_log_abs_y_sign_chunk.col(t1)(problem_index_array[t1]).array() + B_log_abs_grad_Phi_bound_z.col(0)(problem_index_array[t1]).array();
       B_sign_grad_prob.col(0)(problem_index_array[t1]).array() = -1.0 * B_sign_y_sign_chunk.col(t1)(problem_index_array[t1]).array() * B_sign_grad_Phi_bound_z.col(0)(problem_index_array[t1]).array();
       
       // z_grad_term
       B_log_abs_z_grad_term.col(0)(problem_index_array[t1]).array() = B_log_abs_y_m_y_sign_x_u.col(t1)(problem_index_array[t1]).array() + B_log_phi_Z_recip.col(t1)(problem_index_array[t1]).array() + 
         B_log_phi_Bound_Z.col(t1)(problem_index_array[t1]).array() + B_log_abs_grad_bound_z.col(0)(problem_index_array[t1]).array();
       B_sign_z_grad_term.col(0)(problem_index_array[t1]).array() = B_sign_y_m_y_sign_x_u.col(t1)(problem_index_array[t1]).array() * B_sign_grad_bound_z.col(0)(problem_index_array[t1]).array();
       
       // 2nd grad_prob term
       log_abs_deriv_L_T_T_inv = -2.0 * log_abs_L_Omega_double(t1 + 1, t1 + 1) + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + 1, t1 + 1);
       sign_deriv_L_T_T_inv = stan::math::sign(-1.0 / (L_Omega_double(t1 + 1, t1 + 1) * L_Omega_double(t1 + 1, t1 + 1))) * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + 1, t1 + 1);
       
       log_terms.col(0).array() = log_abs_L_Omega_double(t1 + 1, t1) + B_log_abs_z_grad_term.col(0)(problem_index_array[t1]).array();
       log_terms.col(1).array() = B_log_Z_std_norm.col(t1)(problem_index_array[t1]).array() + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
       sign_terms.col(0).array() = stan::math::sign(L_Omega_double(t1+1, t1)) * B_sign_z_grad_term.col(0)(problem_index_array[t1]).array();
       sign_terms.col(1).array() = B_sign_Z_std_norm.col(t1)(problem_index_array[t1]).array() * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
       
       Eigen::Matrix<double, -1, 1> temp_log_abs_deriv_1 = Eigen::Matrix<double, -1, 1>::Constant(n_problem_array[t1], -700.0);
       Eigen::Matrix<double, -1, 1> temp_sign_deriv_1 = Eigen::Matrix<double, -1, 1>::Ones(n_problem_array[t1]);
       
       log_abs_sum_exp_general_v2(log_terms.leftCols(2), sign_terms.leftCols(2), vect_type, vect_type,
                                  temp_log_abs_deriv_1, 
                                  temp_sign_deriv_1,
                                  container_max_logs, container_sum_exp_signed);
       
       log_abs_deriv_Bound_Z_x_L.col(1)(problem_index_array[t1]) = temp_log_abs_deriv_1;
       sign_deriv_Bound_Z_x_L.col(1)(problem_index_array[t1]) = temp_sign_deriv_1;
       
       log_terms.col(0).array() = log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1 + 1)(problem_index_array[t1]).array() + log_abs_L_Omega_double(t1 + 1, t1 + 1);
       log_terms.col(1).array() = -log_abs_L_Omega_double(t1 + 1, t1 + 1) + log_abs_deriv_Bound_Z_x_L.col(1)(problem_index_array[t1]).array();
       sign_terms.col(0).array() = sign_deriv_L_T_T_inv * B_sign_Bound_Z.col(t1 + 1)(problem_index_array[t1]).array() * stan::math::sign(L_Omega_double(t1 + 1, t1 + 1));
       sign_terms.col(1).array() = stan::math::sign(1.0 / L_Omega_double(t1 + 1, t1 + 1)) * -1.0 * sign_deriv_Bound_Z_x_L.col(1)(problem_index_array[t1]).array();
       
       Eigen::Matrix<double, -1, 1> temp_log_abs_grad_bound_1 = Eigen::Matrix<double, -1, 1>::Constant(n_problem_array[t1], -700.0);
       Eigen::Matrix<double, -1, 1> temp_sign_grad_bound_1 = Eigen::Matrix<double, -1, 1>::Ones(n_problem_array[t1]);
       
       log_abs_sum_exp_general_v2( log_terms.leftCols(2), sign_terms.leftCols(2), vect_type, vect_type,
                                   temp_log_abs_grad_bound_1,  
                                   temp_sign_grad_bound_1,
                                   container_max_logs, container_sum_exp_signed);
       
       B_log_abs_grad_bound_z.col(1)(problem_index_array[t1]) = temp_log_abs_grad_bound_1;
       B_sign_grad_bound_z.col(1)(problem_index_array[t1]) = temp_sign_grad_bound_1;
       
       B_log_abs_grad_Phi_bound_z.col(1)(problem_index_array[t1]).array() = B_log_phi_Bound_Z.col(t1 + 1)(problem_index_array[t1]).array() + B_log_abs_grad_bound_z.col(1)(problem_index_array[t1]).array();
       B_sign_grad_Phi_bound_z.col(1)(problem_index_array[t1]) = B_sign_grad_bound_z.col(1)(problem_index_array[t1]);
       
       B_log_abs_grad_prob.col(1)(problem_index_array[t1]).array() = B_log_abs_y_sign_chunk.col(t1 + 1)(problem_index_array[t1]).array() + B_log_abs_grad_Phi_bound_z.col(1)(problem_index_array[t1]).array();
       B_sign_grad_prob.col(1)(problem_index_array[t1]).array() = -1.0 * B_sign_y_sign_chunk.col(t1 + 1)(problem_index_array[t1]).array() * B_sign_grad_Phi_bound_z.col(1)(problem_index_array[t1]).array();
       
       // final computation
       log_terms.col(0) = B_y1_log_prob.col(t1 + 1)(problem_index_array[t1]) + B_log_abs_grad_prob.col(0)(problem_index_array[t1]);
       log_terms.col(1) = B_y1_log_prob.col(t1 + 0)(problem_index_array[t1]) + B_log_abs_grad_prob.col(1)(problem_index_array[t1]);
       sign_terms.col(0) = B_sign_grad_prob.col(0)(problem_index_array[t1]);
       sign_terms.col(1) = B_sign_grad_prob.col(1)(problem_index_array[t1]);
       
       log_abs_sum_exp_general_v2(log_terms.leftCols(2), 
                                  sign_terms.leftCols(2),
                                  vect_type, vect_type,
                                  log_sum_result,
                                  sign_sum_result, 
                                  container_max_logs,
                                  container_sum_exp_signed);
       
       log_abs_bs_grad_array_col_for_each_n(problem_index_array[t1]).array() = B_log_common_grad_term_1.col(t1)(problem_index_array[t1]).array() + log_sum_result.array();
       sign_bs_grad_array_col_for_each_n(problem_index_array[t1]) = sign_sum_result;
       
       LogSumVecSingedResult log_sum_vec_signed_struct = log_sum_vec_signed_v1(log_abs_bs_grad_array_col_for_each_n(problem_index_array[t1]), 
                                                                               sign_bs_grad_array_col_for_each_n(problem_index_array[t1]), 
                                                                               vect_type);
       grad_pi_wrt_b_raw(c, t1) += stan::math::exp(log_sum_vec_signed_struct.log_sum) * log_sum_vec_signed_struct.sign;
       
     }
   }
   
   
   //////// Remaining diagonals (t1 = n_tests - 3, ..., 0)
   for (int i = 3; i < n_tests + 1; i++) {
     
     int t1 = n_tests - i;
     
     if (n_problem_array[t1] > 0) {
       
       log_abs_deriv_Bound_Z_x_L_comp.resize(n_problem_array[t1], n_tests);
       sign_deriv_Bound_Z_x_L_comp.resize(n_problem_array[t1], n_tests);
       log_abs_deriv_Bound_Z_x_L.resize(n_problem_array[t1], n_tests);
       sign_deriv_Bound_Z_x_L.resize(n_problem_array[t1], n_tests);
       B_log_abs_grad_bound_z.resize(n_problem_array[t1], n_tests);
       B_sign_grad_bound_z.resize(n_problem_array[t1], n_tests);
       B_log_abs_grad_Phi_bound_z.resize(n_problem_array[t1], n_tests);
       B_sign_grad_Phi_bound_z.resize(n_problem_array[t1], n_tests);
       B_log_abs_z_grad_term.resize(n_problem_array[t1], n_tests);
       B_sign_z_grad_term.resize(n_problem_array[t1], n_tests);
       B_log_abs_grad_prob.resize(n_problem_array[t1], n_tests);
       B_sign_grad_prob.resize(n_problem_array[t1], n_tests);
       B_log_abs_derivs_chain_container_vec_comp.resize(n_problem_array[t1], n_tests);
       B_sign_derivs_chain_container_vec_comp.resize(n_problem_array[t1], n_tests);
       log_sum_result.resize(n_problem_array[t1]);
       sign_sum_result.resize(n_problem_array[t1]);
       log_terms.resize(n_problem_array[t1], n_tests);
       sign_terms.resize(n_problem_array[t1], n_tests);
       container_max_logs.resize(n_problem_array[t1]);
       container_sum_exp_signed.resize(n_problem_array[t1]);
       
       B_log_abs_grad_prob.setConstant(-700.0);
       B_sign_grad_prob.setOnes();
       
       double log_abs_deriv_L_T_T_inv = -2.0 * log_abs_L_Omega_double(t1, t1) + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1);
       double sign_deriv_L_T_T_inv = -1.0 * stan::math::sign(1.0 / (L_Omega_double(t1,t1) * L_Omega_double(t1,t1))) * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1);
       
       // for (int t = 0; t < t1; t++) {
       //   log_abs_deriv_Bound_Z_x_L_comp.col(t) = B_log_Z_std_norm.col(t)(problem_index_array[t1]) + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
       //   sign_deriv_Bound_Z_x_L_comp.col(t).array() = B_sign_Z_std_norm.col(t)(problem_index_array[t1]).array() * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
       // }
       for (int t = 0; t < t1; t++) {
         Eigen::Matrix<double, -1, 1> temp_log = B_log_Z_std_norm.col(t)(problem_index_array[t1]);
         log_abs_deriv_Bound_Z_x_L_comp.col(t).array() = temp_log.array() + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
         sign_deriv_Bound_Z_x_L_comp.col(t).array() = B_sign_Z_std_norm.col(t)(problem_index_array[t1]).array() * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
       }
       
       log_abs_sum_exp_general_v2(log_abs_deriv_Bound_Z_x_L_comp.leftCols(t1), 
                                  sign_deriv_Bound_Z_x_L_comp.leftCols(t1),
                                  vect_type, vect_type, 
                                  log_abs_deriv_Bound_Z_x_L.col(0), 
                                  sign_deriv_Bound_Z_x_L.col(0),
                                  container_max_logs,
                                  container_sum_exp_signed);
       sign_deriv_Bound_Z_x_L.col(0).array() *= -1.0;
       
       log_terms.col(0).array() = log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1)(problem_index_array[t1]).array() + log_abs_L_Omega_double(t1, t1);
       log_terms.col(1).array() = -log_abs_L_Omega_double(t1, t1) + log_abs_deriv_Bound_Z_x_L.col(0).array();
       sign_terms.col(0).array() = sign_deriv_L_T_T_inv * B_sign_Bound_Z.col(t1)(problem_index_array[t1]).array() * stan::math::sign(L_Omega_double(t1, t1));
       sign_terms.col(1) = stan::math::sign(1.0 / L_Omega_double(t1, t1)) * sign_deriv_Bound_Z_x_L.col(0);
       
       log_abs_sum_exp_general_v2(log_terms.leftCols(2), 
                                  sign_terms.leftCols(2), 
                                  vect_type, vect_type,
                                  B_log_abs_grad_bound_z.col(0), 
                                  B_sign_grad_bound_z.col(0),
                                  container_max_logs, 
                                  container_sum_exp_signed);
       
       B_log_abs_grad_Phi_bound_z.col(0).array() = B_log_phi_Bound_Z.col(t1)(problem_index_array[t1]).array() + B_log_abs_grad_bound_z.col(0).array();
       B_sign_grad_Phi_bound_z.col(0) = B_sign_grad_bound_z.col(0);
       
       B_log_abs_grad_prob.col(0).array() = B_log_abs_y_sign_chunk.col(t1)(problem_index_array[t1]).array() + B_log_abs_grad_Phi_bound_z.col(0).array();
       B_sign_grad_prob.col(0).array() = -1.0 * B_sign_y_sign_chunk.col(t1)(problem_index_array[t1]).array() * B_sign_grad_Phi_bound_z.col(0).array();
       
       B_log_abs_z_grad_term.col(0).array() = B_log_abs_y_m_y_sign_x_u.col(t1)(problem_index_array[t1]).array() + B_log_phi_Z_recip.col(t1)(problem_index_array[t1]).array() + 
         B_log_phi_Bound_Z.col(t1)(problem_index_array[t1]).array() + B_log_abs_grad_bound_z.col(0).array();
       B_sign_z_grad_term.col(0).array() = B_sign_y_m_y_sign_x_u.col(t1)(problem_index_array[t1]).array() * B_sign_grad_bound_z.col(0).array();
       
       log_abs_deriv_L_T_T_inv = -2.0 * log_abs_L_Omega_double(t1 + 1, t1 + 1) + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + 1, t1 + 1);
       sign_deriv_L_T_T_inv = stan::math::sign(-1.0 / (L_Omega_double(t1 + 1, t1 + 1) * L_Omega_double(t1 + 1, t1 + 1))) * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + 1, t1 + 1);
       
       log_terms.col(0).array() = log_abs_L_Omega_double(t1+1, t1) + B_log_abs_z_grad_term.col(0).array();
       
       // Create temporary for indexed view
       Eigen::Matrix<double, -1, 1> temp_col = B_log_Z_std_norm.col(t1)(problem_index_array[t1]);
       log_terms.col(1).array() = temp_col.array() + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
       
       sign_terms.col(0).array() = stan::math::sign(L_Omega_double(t1+1, t1)) * B_sign_z_grad_term.col(0).array();
       sign_terms.col(1).array() = B_sign_Z_std_norm.col(t1)(problem_index_array[t1]).array() * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
       
       log_abs_sum_exp_general_v2(log_terms.leftCols(2), 
                                  sign_terms.leftCols(2), 
                                  vect_type, vect_type,
                                  log_abs_deriv_Bound_Z_x_L.col(1), 
                                  sign_deriv_Bound_Z_x_L.col(1),
                                  container_max_logs, 
                                  container_sum_exp_signed);
       
       log_terms.col(0).array() = log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1 + 1)(problem_index_array[t1]).array() + log_abs_L_Omega_double(t1 + 1, t1 + 1);
       log_terms.col(1).array() = -log_abs_L_Omega_double(t1 + 1, t1 + 1) + log_abs_deriv_Bound_Z_x_L.col(1).array();
       sign_terms.col(0).array() = sign_deriv_L_T_T_inv * B_sign_Bound_Z.col(t1 + 1)(problem_index_array[t1]).array() * stan::math::sign(L_Omega_double(t1 + 1, t1 + 1));
       sign_terms.col(1).array() = stan::math::sign(1.0 / L_Omega_double(t1 + 1, t1 + 1)) * -1.0 * sign_deriv_Bound_Z_x_L.col(1).array();
       
       log_abs_sum_exp_general_v2(log_terms.leftCols(2), 
                                  sign_terms.leftCols(2), 
                                  vect_type, vect_type,
                                  B_log_abs_grad_bound_z.col(1), 
                                  B_sign_grad_bound_z.col(1),
                                  container_max_logs, 
                                  container_sum_exp_signed);
       
       B_log_abs_grad_Phi_bound_z.col(1).array() = B_log_phi_Bound_Z.col(t1 + 1)(problem_index_array[t1]).array() + B_log_abs_grad_bound_z.col(1).array();
       B_sign_grad_Phi_bound_z.col(1) = B_sign_grad_bound_z.col(1);
       
       B_log_abs_grad_prob.col(1).array() = B_log_abs_y_sign_chunk.col(t1 + 1)(problem_index_array[t1]).array() + B_log_abs_grad_Phi_bound_z.col(1).array();
       B_sign_grad_prob.col(1).array() = -1.0 * B_sign_y_sign_chunk.col(t1 + 1)(problem_index_array[t1]).array() * B_sign_grad_Phi_bound_z.col(1).array();
       
       for (int ii = 1; ii < i - 1; ii++) {
         
         B_log_abs_z_grad_term.col(ii).array() = B_log_abs_y_m_y_sign_x_u.col(t1 + ii)(problem_index_array[t1]).array() + 
           B_log_phi_Z_recip.col(t1 + ii)(problem_index_array[t1]).array() + 
           B_log_phi_Bound_Z.col(t1 + ii)(problem_index_array[t1]).array() + 
           B_log_abs_grad_bound_z.col(ii).array();
         B_sign_z_grad_term.col(ii).array() = B_sign_y_m_y_sign_x_u.col(t1 + ii)(problem_index_array[t1]).array() * B_sign_grad_bound_z.col(ii).array();
         
         log_abs_deriv_L_T_T_inv = -2.0 * log_abs_L_Omega_double(t1 + ii + 1, t1 + ii + 1) + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + ii + 1, t1 + ii + 1);
         sign_deriv_L_T_T_inv = stan::math::sign(-1.0 / (L_Omega_double(t1 + ii + 1, t1 + ii + 1) * L_Omega_double(t1 + ii + 1, t1 + ii + 1))) * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + ii + 1, t1 + ii + 1);
         
         for (int jj = 0; jj < ii + 1; jj++) {
           // EXACTLY like MVP - create temporaries for indexed views
           Eigen::Matrix<double, -1, 1> temp_log_z_std = B_log_Z_std_norm.col(t1 + jj)(problem_index_array[t1]);
           
           log_abs_deriv_Bound_Z_x_L_comp.col(jj).array() = log_abs_L_Omega_double(t1 + ii + 1, t1 + jj) + 
             B_log_abs_z_grad_term.col(jj).array() + 
             temp_log_z_std.array() + 
             log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + ii + 1, t1 + jj);
             
             sign_deriv_Bound_Z_x_L_comp.col(jj).array() = stan::math::sign(L_Omega_double(t1 + ii + 1, t1 + jj)) * 
               B_sign_z_grad_term.col(jj).array() * 
               B_sign_Z_std_norm.col(t1 + jj)(problem_index_array[t1]).array() * 
               sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + ii + 1, t1 + jj);
         }
         
         log_abs_sum_exp_general_v2(log_abs_deriv_Bound_Z_x_L_comp.leftCols(ii + 1), 
                                    sign_deriv_Bound_Z_x_L_comp.leftCols(ii + 1),
                                    vect_type, vect_type, 
                                    log_abs_deriv_Bound_Z_x_L.col(ii + 1), 
                                    sign_deriv_Bound_Z_x_L.col(ii + 1),
                                    container_max_logs, 
                                    container_sum_exp_signed);
         
         log_terms.col(0).array() = log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1 + ii + 1)(problem_index_array[t1]).array() + log_abs_L_Omega_double(t1 + ii + 1, t1 + ii + 1);
         log_terms.col(1).array() = -log_abs_L_Omega_double(t1 + ii + 1, t1 + ii + 1) + log_abs_deriv_Bound_Z_x_L.col(ii + 1).array();
         sign_terms.col(0).array() = sign_deriv_L_T_T_inv * B_sign_Bound_Z.col(t1 + ii + 1)(problem_index_array[t1]).array() * stan::math::sign(L_Omega_double(t1 + ii + 1, t1 + ii + 1));
         sign_terms.col(1).array() = stan::math::sign(1.0 / L_Omega_double(t1 + ii + 1, t1 + ii + 1)) * -1.0 * sign_deriv_Bound_Z_x_L.col(ii + 1).array();
         
         log_abs_sum_exp_general_v2(log_terms.leftCols(2),
                                    sign_terms.leftCols(2), 
                                    vect_type, vect_type,
                                    B_log_abs_grad_bound_z.col(ii + 1),
                                    B_sign_grad_bound_z.col(ii + 1),
                                    container_max_logs, 
                                    container_sum_exp_signed);
         
         B_log_abs_grad_Phi_bound_z.col(ii + 1).array() = B_log_phi_Bound_Z.col(t1 + ii + 1)(problem_index_array[t1]).array() + B_log_abs_grad_bound_z.col(ii + 1).array();
         B_sign_grad_Phi_bound_z.col(ii + 1) = B_sign_grad_bound_z.col(ii + 1);
         
         B_log_abs_grad_prob.col(ii + 1).array() = B_log_abs_y_sign_chunk.col(t1 + ii + 1)(problem_index_array[t1]).array() + B_log_abs_grad_Phi_bound_z.col(ii + 1).array();
         B_sign_grad_prob.col(ii + 1).array() = -1.0 * B_sign_y_sign_chunk.col(t1 + ii + 1)(problem_index_array[t1]).array() * B_sign_grad_Phi_bound_z.col(ii + 1).array();
         
       }
       
       for (int iii = 0; iii < i; iii++) {
         B_log_abs_derivs_chain_container_vec_comp.col(iii).array() = B_log_abs_grad_prob.col(iii).array() + 
           B_log_prob_rowwise_prod_temp.col(t1)(problem_index_array[t1]).array() +
           (-B_y1_log_prob.col(t1 + iii)(problem_index_array[t1])).array();
         B_sign_derivs_chain_container_vec_comp.col(iii) = B_sign_grad_prob.col(iii);
       }
       
       log_abs_sum_exp_general_v2(B_log_abs_derivs_chain_container_vec_comp.leftCols(i), 
                                  B_sign_derivs_chain_container_vec_comp.leftCols(i),
                                  vect_type, vect_type, 
                                  log_sum_result, 
                                  sign_sum_result,
                                  container_max_logs, 
                                  container_sum_exp_signed);
       
       log_abs_bs_grad_array_col_for_each_n(problem_index_array[t1]).array() = B_log_common_grad_term_1.col(t1)(problem_index_array[t1]).array() + log_sum_result.array();
       sign_bs_grad_array_col_for_each_n(problem_index_array[t1]) = sign_sum_result;
       
       LogSumVecSingedResult log_sum_vec_signed_struct = log_sum_vec_signed_v1( log_abs_bs_grad_array_col_for_each_n(problem_index_array[t1]), 
                                                                                sign_bs_grad_array_col_for_each_n(problem_index_array[t1]), 
                                                                                vect_type);
       grad_pi_wrt_b_raw(c, t1) += stan::math::exp(log_sum_vec_signed_struct.log_sum) * log_sum_vec_signed_struct.sign;
       
     }
     
   }
   
   // Resize containers back to chunk_size
   log_abs_deriv_Bound_Z_x_L_comp.resize(chunk_size, n_tests);
   sign_deriv_Bound_Z_x_L_comp.resize(chunk_size, n_tests);
   log_abs_deriv_Bound_Z_x_L.resize(chunk_size, n_tests);
   sign_deriv_Bound_Z_x_L.resize(chunk_size, n_tests);
   B_log_abs_grad_bound_z.resize(chunk_size, n_tests);
   B_sign_grad_bound_z.resize(chunk_size, n_tests);
   B_log_abs_grad_Phi_bound_z.resize(chunk_size, n_tests);
   B_sign_grad_Phi_bound_z.resize(chunk_size, n_tests);
   B_log_abs_z_grad_term.resize(chunk_size, n_tests);
   B_sign_z_grad_term.resize(chunk_size, n_tests);
   B_log_abs_grad_prob.resize(chunk_size, n_tests);
   B_sign_grad_prob.resize(chunk_size, n_tests);
   B_log_abs_derivs_chain_container_vec_comp.resize(chunk_size, n_tests);
   B_sign_derivs_chain_container_vec_comp.resize(chunk_size, n_tests);
   log_sum_result.resize(chunk_size);
   sign_sum_result.resize(chunk_size);
   log_terms.resize(chunk_size, n_tests);
   sign_terms.resize(chunk_size, n_tests);
   container_max_logs.resize(chunk_size);
   container_sum_exp_signed.resize(chunk_size);
   
 }
 
 
 
 
 

 