#pragma once


 

#include <Eigen/Dense>
 

 
 
using namespace Eigen;

 

 



#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE





inline void fn_latent_trait_compute_bs_grad_log_scale(      Eigen::Ref<Eigen::Matrix<double, -1, -1>>   grad_pi_wrt_b_raw,  
                                                            Eigen::Ref<Eigen::Matrix<double, -1, 1>>    log_abs_bs_grad_array_col_for_each_n,   
                                                            Eigen::Ref<Eigen::Matrix<double, -1, 1>>    sign_bs_grad_array_col_for_each_n,   
                                                            Eigen::Ref<Eigen::Matrix<double, -1, -1>>   log_abs_deriv_Bound_Z_x_L,   
                                                            Eigen::Ref<Eigen::Matrix<double, -1, -1>>   sign_deriv_Bound_Z_x_L,  
                                                            Eigen::Ref<Eigen::Matrix<double, -1, -1>>   log_abs_deriv_Bound_Z_x_L_comp,   
                                                            Eigen::Ref<Eigen::Matrix<double, -1, -1>>   sign_deriv_Bound_Z_x_L_comp,  
                                                            const int c,   
                                                            const std::vector<Eigen::Matrix<double, -1, -1>> &log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double,  ////
                                                            const std::vector<Eigen::Matrix<double, -1, -1>> &sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double,  ////
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_log_Bound_Z,   
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_sign_Bound_Z,   
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_log_Z_std_norm,   
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_sign_Z_std_norm, 
                                                            const Eigen::Matrix<double, -1, -1> &L_Omega_double,
                                                            const Eigen::Matrix<double, -1, -1> &log_abs_L_Omega_double,
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_log_phi_Bound_Z,   
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_log_phi_Z_recip,
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_log_abs_y_sign_chunk,  
                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> B_sign_y_sign_chunk,  //// note: sign of y_sign same as y_sign !!
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
     
     ///////////////////////// deriv of diagonal elements (not needed if using the "standard" or "Stan" Cholesky parameterisation of Omega)
     //////// w.r.t last diagonal first
     {
       int  t1 = n_tests - 1;
       
       // non-log-scale: double deriv_L_T_T_inv =  ( - 1.0 /  ( L_Omega_double(t1,t1)  * L_Omega_double(t1,t1) ) )   *   Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1)  ;
       double log_abs_deriv_L_T_T_inv = -2.0 * log_abs_L_Omega_double(t1, t1)  +  log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1);
       double sign_deriv_L_T_T_inv = -1.0 * stan::math::sign(1.0 / (L_Omega_double(t1,t1) * L_Omega_double(t1,t1))) * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1);
       
       // non-log-scale: deriv_Bound_Z_x_L.col(0).array() = 0.0;
       log_abs_deriv_Bound_Z_x_L.col(0).setConstant(-700.0); // initialise
       sign_deriv_Bound_Z_x_L.col(0).setOnes() ; // initialise
       for (int t = 0; t < t1; t++) {
         // non-log-scale: deriv_Bound_Z_x_L.col(0).array() +=   Z_std_norm.col(t).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
         log_abs_deriv_Bound_Z_x_L_comp.col(t) =   B_log_Z_std_norm.col(t).array() + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
         sign_deriv_Bound_Z_x_L_comp.col(t).array() =   B_sign_Z_std_norm.col(t).array() * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
       }
       
       ///// then compute log_abs and sign of deriv_Bound_Z_x_L.col(0) using log_abs_sum_exp:
       log_abs_sum_exp_general_v2(log_abs_deriv_Bound_Z_x_L_comp.leftCols(t1),
                                  log_abs_deriv_Bound_Z_x_L_comp.leftCols(t1),
                                  vect_type, vect_type,
                                  log_abs_deriv_Bound_Z_x_L.col(0),
                                  sign_deriv_Bound_Z_x_L.col(0),
                                  container_max_logs,
                                  container_sum_exp_signed);
       
       // non-log-scale: double deriv_a = 0 ; //  stan::math::pow( (1 + (bs_mat_double(c, t1)*bs_mat_double(c, t1)) ), -0.5) * bs_mat_double(c, t1)  * LT_theta(c, t1)  ;
       // non-log-scale: deriv_Bound_Z_x_L.col(0).array()   = deriv_a -     deriv_Bound_Z_x_L.col(0).array();
       sign_deriv_Bound_Z_x_L.col(0) = -1.0 * sign_deriv_Bound_Z_x_L.col(0);
       
       ///// compute log_abs and sign of grad_bound_z using log_abs_sum_exp
       // non-log-scale: grad_bound_z.col(0).array() =   deriv_L_T_T_inv * (Bound_Z.col(t1).array() * L_Omega_double(t1, t1)  ) +  (1.0 / L_Omega_double(t1, t1)) *   deriv_Bound_Z_x_L.col(0).array()  ;
       // log_abs_a.array() = log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1).array() + log_abs_L_Omega_double(t1, t1);
       // sign_a.array() = sign_deriv_L_T_T_inv * (B_sign_Bound_Z.col(t1).array() * stan::math::sign(L_Omega_double(t1, t1)) );
       // log_abs_b.array() =  - log_abs_L_Omega_double(t1, t1) +  log_abs_deriv_Bound_Z_x_L.col(0).array();
       // sign_b.array() =  stan::math::sign(1.0 / L_Omega_double(t1, t1))  *   sign_deriv_Bound_Z_x_L.col(0).array();
       log_terms.col(0) =  log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1).array() + log_abs_L_Omega_double(t1, t1);
       log_terms.col(1)  = - log_abs_L_Omega_double(t1, t1) +  log_abs_deriv_Bound_Z_x_L.col(0).array();
       sign_terms.col(0) =  sign_deriv_L_T_T_inv * (B_sign_Bound_Z.col(t1).array() * stan::math::sign(L_Omega_double(t1, t1)) );
       sign_terms.col(1) =  stan::math::sign(1.0 / L_Omega_double(t1, t1))  *   sign_deriv_Bound_Z_x_L.col(0).array();
       
       ///// then compute log_abs and sign using log_abs_sum_exp:
       log_abs_sum_exp_general_v2(log_terms.leftCols(2),
                                  sign_terms.leftCols(2),
                                  vect_type, vect_type,
                                  B_log_abs_grad_bound_z.col(0),
                                  B_sign_grad_bound_z.col(0),
                                  container_max_logs,
                                  container_sum_exp_signed);
         
       // non-log-scale:  grad_Phi_bound_z.col(0)  =  ( phi_Bound_Z.col(t1).array() *  (  grad_bound_z.col(0).array() )  ) .matrix();   // correct  (standard form)
       B_log_abs_grad_Phi_bound_z.col(0).array() = B_log_phi_Bound_Z.col(t1).array() + B_log_abs_grad_bound_z.col(0).array(); 
       B_sign_grad_Phi_bound_z.col(0).array()  =     B_sign_grad_bound_z.col(0).array();    // note: sign of phi_Bound_Z = +1 as density!!
         
       // non-log-scale:  A_grad_prob.col(0)   =  (   - y_sign_chunk.col(t1).array()  *   grad_Phi_bound_z.col(0).array() ).matrix() ;     // correct  (standard form)
       B_log_abs_grad_prob.col(0).array() = B_log_abs_y_sign_chunk.col(t1).array() +  B_log_abs_grad_Phi_bound_z.col(0).array(); 
       B_sign_grad_prob.col(0).array() =  -1.0 * B_sign_y_sign_chunk.col(t1).array() *  B_sign_grad_Phi_bound_z.col(0).array(); 
         
       /////// final grad computations 
       // non-log-scale:   grad_pi_wrt_b_raw(c, t1)  +=   (   A_common_grad_term_1.col(t1).array()    *            A_grad_prob.col(0).array()    ).matrix().sum()   ; // correct  (standard form)
       log_abs_bs_grad_array_col_for_each_n.array() = B_log_common_grad_term_1.col(t1).array() + B_log_abs_grad_prob.col(0).array(); 
       sign_bs_grad_array_col_for_each_n.array() =    B_sign_grad_prob.col(0).array(); // note: sign of  "common_grad_term_1" = +'ve !!
       
       /////// Final scalar grad using log-sum-exp
       LogSumVecSingedResult log_sum_vec_signed_struct = log_sum_vec_signed_v1(log_abs_bs_grad_array_col_for_each_n,
                                                                               sign_bs_grad_array_col_for_each_n, 
                                                                               vect_type);
       /////// compute final (scalar) grad term
       grad_pi_wrt_b_raw(c, t1) +=   stan::math::exp(log_sum_vec_signed_struct.log_sum) * log_sum_vec_signed_struct.sign;
       
       
     }
     
     //////// then w.r.t the second-to-last diagonal
     {
       
       int  t1 = n_tests - 2;
       
       
       // non-log-scale: double deriv_L_T_T_inv =  ( - 1.0 /  ( L_Omega_double(t1,t1)  * L_Omega_double(t1,t1) ) )   *   Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1)  ;
       double log_abs_deriv_L_T_T_inv = -2.0 * log_abs_L_Omega_double(t1, t1)  +  log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1);
       double sign_deriv_L_T_T_inv = -1.0 * stan::math::sign(1.0 / (L_Omega_double(t1,t1) * L_Omega_double(t1,t1))) * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1);
       
       // non-log-scale: deriv_Bound_Z_x_L.col(0).array() = 0.0;
       log_abs_deriv_Bound_Z_x_L.col(0).setConstant(-700.0); // initialise
       sign_deriv_Bound_Z_x_L.col(0).setOnes() ; // initialise
       for (int t = 0; t < t1; t++) {
         // non-log-scale: deriv_Bound_Z_x_L.col(0).array() +=   Z_std_norm.col(t).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
         log_abs_deriv_Bound_Z_x_L_comp.col(t) =   B_log_Z_std_norm.col(t).array() + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
         sign_deriv_Bound_Z_x_L_comp.col(t).array() =   B_sign_Z_std_norm.col(t).array() * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
       } 
       
       ///// then compute log_abs and sign of deriv_Bound_Z_x_L.col(0) using log_abs_sum_exp:
       log_abs_sum_exp_general_v2(log_abs_deriv_Bound_Z_x_L_comp.leftCols(t1),
                                  sign_deriv_Bound_Z_x_L_comp.leftCols(t1),
                                  vect_type, vect_type,
                                  log_abs_deriv_Bound_Z_x_L.col(0),
                                  sign_deriv_Bound_Z_x_L.col(0),
                                  container_max_logs,
                                  container_sum_exp_signed);
       
       // non-log-scale: double deriv_a = 0 ; //  stan::math::pow( (1 + (bs_mat_double(c, t1)*bs_mat_double(c, t1)) ), -0.5) * bs_mat_double(c, t1)  * LT_theta(c, t1)  ;
       // non-log-scale: deriv_Bound_Z_x_L.col(0).array()   = deriv_a -     deriv_Bound_Z_x_L.col(0).array();
       sign_deriv_Bound_Z_x_L.col(0) = -1.0 * sign_deriv_Bound_Z_x_L.col(0);
       
       ///// compute log_abs and sign of grad_bound_z using log_abs_sum_exp
       // non-log-scale: grad_bound_z.col(0).array() =   deriv_L_T_T_inv * (Bound_Z.col(t1).array() * L_Omega_double(t1, t1)  ) +  (1.0 / L_Omega_double(t1, t1)) *   deriv_Bound_Z_x_L.col(0).array()  ;
       // log_abs_a.array() = log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1).array() + log_abs_L_Omega_double(t1, t1); 
       // sign_a.array() = sign_deriv_L_T_T_inv * (B_sign_Bound_Z.col(t1).array() * stan::math::sign(L_Omega_double(t1, t1)) );
       // log_abs_b.array() =  - log_abs_L_Omega_double(t1, t1) +  log_abs_deriv_Bound_Z_x_L.col(0).array();
       // sign_b.array() =  stan::math::sign(1.0 / L_Omega_double(t1, t1))  *   sign_deriv_Bound_Z_x_L.col(0).array();
       log_terms.col(0) = log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1).array() + log_abs_L_Omega_double(t1, t1); 
       log_terms.col(1) =  - log_abs_L_Omega_double(t1, t1) +  log_abs_deriv_Bound_Z_x_L.col(0).array();
       sign_terms.col(0) = sign_deriv_L_T_T_inv * (B_sign_Bound_Z.col(t1).array() * stan::math::sign(L_Omega_double(t1, t1)) );
       sign_terms.col(1) =  stan::math::sign(1.0 / L_Omega_double(t1, t1))  *   sign_deriv_Bound_Z_x_L.col(0).array();
       
       log_abs_sum_exp_general_v2(log_terms.leftCols(2),
                                  sign_terms.leftCols(2),
                                  vect_type, vect_type,
                                  B_log_abs_grad_bound_z.col(0),
                                  B_sign_grad_bound_z.col(0),
                                  container_max_logs,
                                  container_sum_exp_signed);
        
       // non-log-scale:  grad_Phi_bound_z.col(0)  =  ( phi_Bound_Z.col(t1).array() *  (  grad_bound_z.col(0).array() )  ) .matrix();   // correct  (standard form)
       B_log_abs_grad_Phi_bound_z.col(0).array() = B_log_phi_Bound_Z.col(t1).array() + B_log_abs_grad_bound_z.col(0).array(); 
       B_sign_grad_Phi_bound_z.col(0).array()  =     B_sign_grad_bound_z.col(0).array();    // note: sign of phi_Bound_Z = +1 as density!!
         
       // non-log-scale:  A_grad_prob.col(0)   =  (   - y_sign_chunk.col(t1).array()  *   grad_Phi_bound_z.col(0).array() ).matrix() ;     // correct  (standard form)
       B_log_abs_grad_prob.col(0).array() = B_log_abs_y_sign_chunk.col(t1).array() +  B_log_abs_grad_Phi_bound_z.col(0).array(); 
       B_sign_grad_prob.col(0).array() =  -1.0 * B_sign_y_sign_chunk.col(t1).array() *  B_sign_grad_Phi_bound_z.col(0).array(); 
       
       ////// NOTE: from here it differs from computation for t1 = n_tests - 1
       // non-log-scale: A_z_grad_term.col(0).array()  =  (  ( (  y_m_y_sign_x_u.col(t1).array()  * phi_Z_recip.col(t1).array()  ).array()    * phi_Bound_Z.col(t1).array() *   grad_bound_z.col(0).array()  ).array() )  ;  // correct  (standard form)
       B_log_abs_z_grad_term.col(0).array() = B_log_abs_y_m_y_sign_x_u.col(t1).array() + B_log_phi_Z_recip.col(t1).array() + B_log_phi_Bound_Z.col(t1).array() + B_log_abs_grad_bound_z.col(0).array();
       B_sign_z_grad_term.col(0).array()  =        B_sign_y_m_y_sign_x_u.col(t1).array()  *  B_sign_grad_bound_z.col(0).array()  ; // NOTE: B_sign_phi_Z_recip and B_sign_phi_Bound_Z are +'ve as densities !!
       
       ///// now compute 2nd grad_prob term 
       // non-log-scale: deriv_L_T_T_inv =  ( - 1 /  ( L_Omega_double(t1 + 1, t1 + 1)  * L_Omega_double(t1 + 1, t1 + 1)  )  )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + 1, t1 + 1)  ;
       log_abs_deriv_L_T_T_inv = -2.0 * log_abs_L_Omega_double(t1 + 1, t1 + 1)  + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + 1, t1 + 1);
       sign_deriv_L_T_T_inv =  stan::math::sign( - 1.0 /  ( L_Omega_double(t1 + 1, t1 + 1)  * L_Omega_double(t1 + 1, t1 + 1)  )  )  * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + 1, t1 + 1);
       
       // non-log-scale: deriv_Bound_Z_x_L.col(1).array()  =    L_Omega_double(t1+1,t1) *   A_z_grad_term.col(0).array()     +   Z_std_norm.col(t1).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
       // log_abs_a.array() =  log_abs_L_Omega_double(t1+1, t1)  +   B_log_abs_z_grad_term.col(0).array(); 
       // sign_a.array() = stan::math::sign(L_Omega_double(t1+1, t1)) *   B_sign_z_grad_term.col(0).array()  ;
       // log_abs_b.array() =  B_log_Z_std_norm.col(t1).array()  +   log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
       // sign_b.array() = B_sign_Z_std_norm.col(t1).array()  *   sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
       log_terms.col(0) = log_abs_L_Omega_double(t1+1, t1)  +   B_log_abs_z_grad_term.col(0).array(); 
       log_terms.col(1) =  B_log_Z_std_norm.col(t1).array()  +   log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
       sign_terms.col(0) = stan::math::sign(L_Omega_double(t1+1, t1)) *   B_sign_z_grad_term.col(0).array()  ;
       sign_terms.col(1) =  B_sign_Z_std_norm.col(t1).array()  *   sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
       
       ///// then compute log_abs and sign of deriv_Bound_Z_x_L.col(1) using log_abs_sum_exp:
       log_abs_sum_exp_general_v2(log_terms.leftCols(2),
                                  sign_terms.leftCols(2),
                                  vect_type, vect_type,
                                  log_abs_deriv_Bound_Z_x_L.col(1),
                                  sign_deriv_Bound_Z_x_L.col(1),
                                  container_max_logs,
                                  container_sum_exp_signed);
       
      
       ///// compute log_abs and sign of grad_bound_z using log_abs_sum_exp
       // non-log-scale: grad_bound_z.col(1).array() =  deriv_L_T_T_inv * (Bound_Z.col(t1+1).array() * L_Omega_double(t1+1,t1+1)  ) +  (1.0 / L_Omega_double(t1+1, t1+1)) * -   deriv_Bound_Z_x_L.col(1).array()  ;
       // log_abs_a.array() = log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1 + 1).array() + log_abs_L_Omega_double(t1 + 1, t1 + 1); 
       // sign_a.array() = sign_deriv_L_T_T_inv * (B_sign_Bound_Z.col(t1 + 1).array() * stan::math::sign(L_Omega_double(t1 + 1, t1 + 1)) );
       // log_abs_b.array() =  - log_abs_L_Omega_double(t1 + 1, t1 + 1) +  log_abs_deriv_Bound_Z_x_L.col(1).array();
       // sign_b.array() =  stan::math::sign(1.0 / L_Omega_double(t1 + 1, t1 + 1))  *   -1.0 * sign_deriv_Bound_Z_x_L.col(1).array();
       log_terms.col(0) = log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1 + 1).array() + log_abs_L_Omega_double(t1 + 1, t1 + 1); 
       log_terms.col(1) =   - log_abs_L_Omega_double(t1 + 1, t1 + 1) +  log_abs_deriv_Bound_Z_x_L.col(1).array();
       sign_terms.col(0) =  sign_deriv_L_T_T_inv * (B_sign_Bound_Z.col(t1 + 1).array() * stan::math::sign(L_Omega_double(t1 + 1, t1 + 1)) );
       sign_terms.col(1) =  stan::math::sign(1.0 / L_Omega_double(t1 + 1, t1 + 1))  *   -1.0 * sign_deriv_Bound_Z_x_L.col(1).array();
       
       log_abs_sum_exp_general_v2(log_terms.leftCols(2),
                                  sign_terms.leftCols(2),
                                  vect_type, vect_type,
                                  B_log_abs_grad_bound_z.col(1),
                                  B_sign_grad_bound_z.col(1),
                                  container_max_logs,
                                  container_sum_exp_signed);
       
       // non-log-scale: grad_Phi_bound_z.col(1) =   ( phi_Bound_Z.col(t1 + 1).array() *  (    grad_bound_z.col(1).array()   ) ).matrix();   // correct  (standard form)
       B_log_abs_grad_Phi_bound_z.col(1).array() = B_log_phi_Bound_Z.col(t1 + 1).array() + B_log_abs_grad_bound_z.col(1).array(); 
       B_sign_grad_Phi_bound_z.col(1).array()  =     B_sign_grad_bound_z.col(1).array();    // note: sign of phi_Bound_Z = +1 as density!!
         
       // non-log-scale: A_grad_prob.col(1)   =   (  - y_sign_chunk.col(t1 + 1).array()  *     grad_Phi_bound_z.col(1).array()  ).array().matrix() ;    // correct   (standard form)
       B_log_abs_grad_prob.col(1).array() = B_log_abs_y_sign_chunk.col(t1 + 1).array() +  B_log_abs_grad_Phi_bound_z.col(1).array(); 
       B_sign_grad_prob.col(1).array() =  -1.0 * B_sign_y_sign_chunk.col(t1 + 1).array() *   B_sign_grad_Phi_bound_z.col(1).array(); 

       /////// final grad computations 
       // non-log-scale: grad_pi_wrt_b_raw(c, t1) +=   ( ( A_common_grad_term_1.col(t1).array() )    *   ( prob.col(t1 + 1).array() * A_grad_prob.col(0).array()  +   prob.col(t1).array() * A_grad_prob.col(1).array()   ) ).sum() ;
       // use log_sum_exp_pair to do on log-scale 
       // log_abs_a.array() =  B_y1_log_prob.col(t1 + 1).array() + B_log_abs_grad_prob.col(0).array();
       // sign_a.array() =     B_sign_grad_prob.col(0).array(); // note: sign of prob always +'ve !!!
       // log_abs_b.array()  = B_y1_log_prob.col(t1 + 0).array() + B_log_abs_grad_prob.col(1).array();
       // sign_b.array() =     B_sign_grad_prob.col(1).array();  // note: sign of prob always +'ve !!!
       log_terms.col(0) =  B_y1_log_prob.col(t1 + 1).array() + B_log_abs_grad_prob.col(0).array();
       log_terms.col(1) =  B_y1_log_prob.col(t1 + 0).array() + B_log_abs_grad_prob.col(1).array();
       sign_terms.col(0) =  B_sign_grad_prob.col(0).array(); // note: sign of prob always +'ve !!!
       sign_terms.col(1) = B_sign_grad_prob.col(1).array();  // note: sign of prob always +'ve !!!
       
       ///// then compute log_abs and sign of deriv_Bound_Z_x_L.col(1) using log_abs_sum_exp:
       log_abs_sum_exp_general_v2(log_terms.leftCols(2),
                                  sign_terms.leftCols(2),
                                  vect_type, vect_type,
                                  log_sum_result,
                                  sign_sum_result,
                                  container_max_logs,
                                  container_sum_exp_signed);
       
       log_abs_bs_grad_array_col_for_each_n.array() = B_log_common_grad_term_1.col(t1).array() + log_sum_result.array(); 
       sign_bs_grad_array_col_for_each_n.array() =    sign_sum_result.array() ; //   note: sign of  "common_grad_term_1" = +'ve !!
       
       /////// Final scalar grad using log-sum-exp
       LogSumVecSingedResult log_sum_vec_signed_struct = log_sum_vec_signed_v1(log_abs_bs_grad_array_col_for_each_n,
                                                                               sign_bs_grad_array_col_for_each_n, 
                                                                               vect_type);
       /////// compute final (scalar) grad term
       grad_pi_wrt_b_raw(c, t1) +=   stan::math::exp(log_sum_vec_signed_struct.log_sum) * log_sum_vec_signed_struct.sign;
       
     }
     
     
     //////// then w.r.t the third-to-last diagonal .... etc
     {
       
       for (int i = 3; i < n_tests + 1; i++) {
         
         B_log_abs_grad_prob.setConstant(-700.0);
         B_sign_grad_prob.setOnes();
         
         int  t1 = n_tests - i;
         
         
         // non-log-scale: double deriv_L_T_T_inv =  ( - 1.0 /  ( L_Omega_double(t1,t1)  * L_Omega_double(t1,t1) ) )   *   Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1)  ;
         double log_abs_deriv_L_T_T_inv = -2.0 * log_abs_L_Omega_double(t1, t1)  +  log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1);
         double sign_deriv_L_T_T_inv = -1.0 * stan::math::sign(1.0 / (L_Omega_double(t1,t1) * L_Omega_double(t1,t1))) * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1);
         
         // non-log-scale: deriv_Bound_Z_x_L.col(0).array() = 0.0;
         log_abs_deriv_Bound_Z_x_L.col(0).setConstant(-700.0); // initialise
         sign_deriv_Bound_Z_x_L.col(0).setOnes() ; // initialise
         for (int t = 0; t < t1; t++) {
           // non-log-scale: deriv_Bound_Z_x_L.col(0).array() +=   Z_std_norm.col(t).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
           log_abs_deriv_Bound_Z_x_L_comp.col(t) =   B_log_Z_std_norm.col(t).array() + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
           sign_deriv_Bound_Z_x_L_comp.col(t).array() =   B_sign_Z_std_norm.col(t).array() * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
         } 
         
         ///// then compute log_abs and sign of deriv_Bound_Z_x_L.col(0) using log_abs_sum_exp:
         log_abs_sum_exp_general_v2(log_abs_deriv_Bound_Z_x_L_comp.leftCols(t1),
                                    sign_deriv_Bound_Z_x_L_comp.leftCols(t1),
                                    vect_type, vect_type,
                                    log_abs_deriv_Bound_Z_x_L.col(0),
                                    sign_deriv_Bound_Z_x_L.col(0),
                                    container_max_logs,
                                    container_sum_exp_signed);
         
         // non-log-scale: double deriv_a = 0 ; //  stan::math::pow( (1 + (bs_mat_double(c, t1)*bs_mat_double(c, t1)) ), -0.5) * bs_mat_double(c, t1)  * LT_theta(c, t1)  ;
         // non-log-scale: deriv_Bound_Z_x_L.col(0).array()   = deriv_a -     deriv_Bound_Z_x_L.col(0).array();
         sign_deriv_Bound_Z_x_L.col(0) = -1.0 * sign_deriv_Bound_Z_x_L.col(0);
         
         ///// compute log_abs and sign of grad_bound_z using log_abs_sum_exp
         // non-log-scale: grad_bound_z.col(0).array() =   deriv_L_T_T_inv * (Bound_Z.col(t1).array() * L_Omega_double(t1, t1)  ) +  (1.0 / L_Omega_double(t1, t1)) *   deriv_Bound_Z_x_L.col(0).array()  ;
         // log_abs_a.array() = log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1).array() + log_abs_L_Omega_double(t1, t1); 
         // sign_a.array() = sign_deriv_L_T_T_inv * (B_sign_Bound_Z.col(t1).array() * stan::math::sign(L_Omega_double(t1, t1)) );
         // log_abs_b.array() =  - log_abs_L_Omega_double(t1, t1) +  log_abs_deriv_Bound_Z_x_L.col(0).array();
         // sign_b.array() =  stan::math::sign(1.0 / L_Omega_double(t1, t1))  *   sign_deriv_Bound_Z_x_L.col(0).array();
         log_terms.col(0) =  log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1).array() + log_abs_L_Omega_double(t1, t1); 
         log_terms.col(1) =  - log_abs_L_Omega_double(t1, t1) +  log_abs_deriv_Bound_Z_x_L.col(0).array();
         sign_terms.col(0) =   sign_deriv_L_T_T_inv * (B_sign_Bound_Z.col(t1).array() * stan::math::sign(L_Omega_double(t1, t1)) );
         sign_terms.col(1) =   stan::math::sign(1.0 / L_Omega_double(t1, t1))  *   sign_deriv_Bound_Z_x_L.col(0).array();
         
         ///// then compute log_abs and sign using log_abs_sum_exp:
         log_abs_sum_exp_general_v2(log_terms.leftCols(2),
                                    sign_terms.leftCols(2),
                                    vect_type, vect_type,
                                    B_log_abs_grad_bound_z.col(0),
                                    B_sign_grad_bound_z.col(0),
                                    container_max_logs,
                                    container_sum_exp_signed);
         
         //// non-log-scale:  grad_Phi_bound_z.col(0)  =  ( phi_Bound_Z.col(t1).array() *  (  grad_bound_z.col(0).array() )  ) .matrix();   // correct  (standard form)
         B_log_abs_grad_Phi_bound_z.col(0).array() = B_log_phi_Bound_Z.col(t1).array() + B_log_abs_grad_bound_z.col(0).array(); 
         B_sign_grad_Phi_bound_z.col(0).array()  =     B_sign_grad_bound_z.col(0).array();    // note: sign of phi_Bound_Z = +1 as density!!
           
         //// non-log-scale:  A_grad_prob.col(0)   =  (   - y_sign_chunk.col(t1).array()  *   grad_Phi_bound_z.col(0).array() ).matrix() ;     // correct  (standard form)
         B_log_abs_grad_prob.col(0).array() = B_log_abs_y_sign_chunk.col(t1).array() +  B_log_abs_grad_Phi_bound_z.col(0).array(); 
         B_sign_grad_prob.col(0).array() =  -1.0 * B_sign_y_sign_chunk.col(t1).array() *  B_sign_grad_Phi_bound_z.col(0).array(); 
         
         ////// NOTE: from here it differs from computation for t1 = n_tests - 1
         //// non-log-scale: A_z_grad_term.col(0).array()  =  (  ( (  y_m_y_sign_x_u.col(t1).array()  * phi_Z_recip.col(t1).array()  ).array()    * phi_Bound_Z.col(t1).array() *   grad_bound_z.col(0).array()  ).array() )  ;  // correct  (standard form)
         B_log_abs_z_grad_term.col(0).array() = B_log_abs_y_m_y_sign_x_u.col(t1).array() + B_log_phi_Z_recip.col(t1).array() + B_log_phi_Bound_Z.col(t1).array() + B_log_abs_grad_bound_z.col(0).array();
         B_sign_z_grad_term.col(0).array()  =        B_sign_y_m_y_sign_x_u.col(t1).array()  *  B_sign_grad_bound_z.col(0).array()  ; // NOTE: B_sign_phi_Z_recip and B_sign_phi_Bound_Z are +'ve as densities !!
         
         ///// now compute 2nd grad_prob term 
         //// non-log-scale: deriv_L_T_T_inv =  ( - 1 /  ( L_Omega_double(t1 + 1, t1 + 1)  * L_Omega_double(t1 + 1, t1 + 1)  )  )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + 1, t1 + 1)  ;
         log_abs_deriv_L_T_T_inv = -2.0 * log_abs_L_Omega_double(t1 + 1, t1 + 1)  + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + 1, t1 + 1);
         sign_deriv_L_T_T_inv =  stan::math::sign( - 1.0 /  ( L_Omega_double(t1 + 1, t1 + 1)  * L_Omega_double(t1 + 1, t1 + 1)  )  )  * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + 1, t1 + 1);
         
         //// non-log-scale: deriv_Bound_Z_x_L.col(1).array()  =    L_Omega_double(t1+1,t1) *   A_z_grad_term.col(0).array()     +   Z_std_norm.col(t1).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
         // log_abs_a.array() =  log_abs_L_Omega_double(t1+1, t1)  +   B_log_abs_z_grad_term.col(0).array(); 
         // sign_a.array() = stan::math::sign(L_Omega_double(t1+1, t1)) *   B_sign_z_grad_term.col(0).array()  ;
         // log_abs_b.array() =  B_log_Z_std_norm.col(t1).array()  +   log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
         // sign_b.array() = B_sign_Z_std_norm.col(t1).array()  *   sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
         log_terms.col(0) =  log_abs_L_Omega_double(t1+1, t1)  +   B_log_abs_z_grad_term.col(0).array(); 
         log_terms.col(1)  = B_log_Z_std_norm.col(t1).array()  +   log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
         sign_terms.col(0) =  stan::math::sign(L_Omega_double(t1+1, t1)) *   B_sign_z_grad_term.col(0).array()  ;
         sign_terms.col(1) =  B_sign_Z_std_norm.col(t1).array()  *   sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
         
         ///// then compute log_abs and sign using log_abs_sum_exp:
         log_abs_sum_exp_general_v2(log_terms.leftCols(2),
                                    sign_terms.leftCols(2),
                                    vect_type, vect_type,
                                    log_abs_deriv_Bound_Z_x_L.col(1),
                                    sign_deriv_Bound_Z_x_L.col(1),
                                    container_max_logs,
                                    container_sum_exp_signed);
         
         ///// compute log_abs and sign of grad_bound_z using log_abs_sum_exp
         // non-log-scale: grad_bound_z.col(1).array() =  deriv_L_T_T_inv * (Bound_Z.col(t1+1).array() * L_Omega_double(t1+1,t1+1)  ) +  (1.0 / L_Omega_double(t1+1, t1+1)) * -   deriv_Bound_Z_x_L.col(1).array()  ;
         // log_abs_a.array() = log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1 + 1).array() + log_abs_L_Omega_double(t1 + 1, t1 + 1); 
         // sign_a.array() = sign_deriv_L_T_T_inv * (B_sign_Bound_Z.col(t1 + 1).array() * stan::math::sign(L_Omega_double(t1 + 1, t1 + 1)) );
         // log_abs_b.array() =  - log_abs_L_Omega_double(t1 + 1, t1 + 1) +  log_abs_deriv_Bound_Z_x_L.col(1).array();
         // sign_b.array() =  stan::math::sign(1.0 / L_Omega_double(t1 + 1, t1 + 1))  *   -1.0 * sign_deriv_Bound_Z_x_L.col(1).array();
         log_terms.col(0)  =   log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1 + 1).array() + log_abs_L_Omega_double(t1 + 1, t1 + 1); 
         log_terms.col(1)  = - log_abs_L_Omega_double(t1 + 1, t1 + 1) +  log_abs_deriv_Bound_Z_x_L.col(1).array();
         sign_terms.col(0) =  sign_deriv_L_T_T_inv * (B_sign_Bound_Z.col(t1 + 1).array() * stan::math::sign(L_Omega_double(t1 + 1, t1 + 1)) );
         sign_terms.col(1) =  stan::math::sign(1.0 / L_Omega_double(t1 + 1, t1 + 1))  *   -1.0 * sign_deriv_Bound_Z_x_L.col(1).array();
         
         ///// then compute log_abs and sign using log_abs_sum_exp:
         log_abs_sum_exp_general_v2(log_terms.leftCols(2),
                                    sign_terms.leftCols(2),
                                    vect_type, vect_type,
                                    B_log_abs_grad_bound_z.col(1),
                                    B_sign_grad_bound_z.col(1),
                                    container_max_logs,
                                    container_sum_exp_signed);
         
         // non-log-scale: grad_Phi_bound_z.col(1) =   ( phi_Bound_Z.col(t1 + 1).array() *  (    grad_bound_z.col(1).array()   ) ).matrix();   // correct  (standard form)
         B_log_abs_grad_Phi_bound_z.col(1).array() = B_log_phi_Bound_Z.col(t1 + 1).array() + B_log_abs_grad_bound_z.col(1).array(); 
         B_sign_grad_Phi_bound_z.col(1).array()  =     B_sign_grad_bound_z.col(1).array();    // note: sign of phi_Bound_Z = +1 as density!!
           
         // non-log-scale: A_grad_prob.col(1)   =   (  - y_sign_chunk.col(t1 + 1).array()  *     grad_Phi_bound_z.col(1).array()  ).array().matrix() ;    // correct   (standard form)
         B_log_abs_grad_prob.col(1).array() = B_log_abs_y_sign_chunk.col(t1 + 1).array() +  B_log_abs_grad_Phi_bound_z.col(1).array(); 
         B_sign_grad_prob.col(1).array() =  -1.0 * B_sign_y_sign_chunk.col(t1 + 1).array() *   B_sign_grad_Phi_bound_z.col(1).array(); 
         
         // double deriv_L_T_T_inv =  ( - 1 /   ( L_Omega_double(t1,t1)  * L_Omega_double(t1,t1) ) )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1)  ;
         // 
         // deriv_Bound_Z_x_L.col(0).array() = 0;
         // for (int t = 0; t < t1; t++) {
         //   deriv_Bound_Z_x_L.col(0).array() += Z_std_norm.col(t).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
         // }
         // 
         // double deriv_a = 0 ; //  stan::math::pow( (1 + (bs_mat_double(c, t1)*bs_mat_double(c, t1)) ), -0.5) * bs_mat_double(c, t1) *   LT_theta(c, t1)   ;
         // deriv_Bound_Z_x_L.col(0).array()   = deriv_a -     deriv_Bound_Z_x_L.col(0).array();
         // 
         // grad_bound_z.col(0).array() =  deriv_L_T_T_inv * (Bound_Z.col(t1).array() * L_Omega_double(t1,t1)  ) +  (1 / L_Omega_double(t1, t1)) *    deriv_Bound_Z_x_L.col(0).array()  ;
         // grad_Phi_bound_z.col(0)  =  ( phi_Bound_Z.col(t1).array() *  (  grad_bound_z.col(0).array() )  ) .matrix();   // correct  (standard form)
         // A_grad_prob.col(0)   =  (   - y_sign_chunk.col(t1).array()  *   grad_Phi_bound_z.col(0).array() ).matrix() ;     // correct  (standard form)
         // 
         // A_z_grad_term.col(0).array()  =      (  ( (  y_m_y_sign_x_u.col(t1).array()  * phi_Z_recip.col(t1).array()  ).array()    * phi_Bound_Z.col(t1).array() *   grad_bound_z.col(0).array()  ).array() ).matrix()  ;  // correct  (standard form)
         // 
         // deriv_L_T_T_inv =  ( - 1 /  ( L_Omega_double(t1+1,t1+1)  * L_Omega_double(t1+1,t1+1)  )  )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1+1)  ;
         // deriv_Bound_Z_x_L.col(1).array()  =    L_Omega_double(t1+1,t1) *   A_z_grad_term.col(0).array()     +   Z_std_norm.col(t1).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
         // grad_bound_z.col(1).array() =  deriv_L_T_T_inv * (Bound_Z.col(t1+1).array() * L_Omega_double(t1+1,t1+1)  ) +  (1 / L_Omega_double(t1+1, t1+1)) * -  deriv_Bound_Z_x_L.col(1).array()  ;
         // 
         // grad_Phi_bound_z.col(1) =   ( phi_Bound_Z.col(t1 + 1).array() *  (    grad_bound_z.col(1).array()   ) ).matrix();   // correct  (standard form)
         // A_grad_prob.col(1)  =   (  - y_sign_chunk.col(t1 + 1).array()  *     grad_Phi_bound_z.col(1).array()  ).array().matrix() ;    // correct   (standard form)
         
         //////  UP TO HERE SAME AS PREVIOUS TERM
         
         for (int ii = 1; ii < i - 1; ii++) {
               
               ////// compute next z term 
               // non-log-scale: A_z_grad_term.col(ii).array()  =    (  ( (  y_m_y_sign_x_u.col(t1 + ii).array()  * phi_Z_recip.col(t1 + ii).array()  ).array()    * phi_Bound_Z.col(t1 + ii).array() *   grad_bound_z.col(ii).array()  ).array() ).matrix() ;     // correct  (standard form)
               B_log_abs_z_grad_term.col(ii).array() = B_log_abs_y_m_y_sign_x_u.col(t1 + ii).array() + B_log_phi_Z_recip.col(t1 + ii).array() + B_log_phi_Bound_Z.col(t1 + ii).array() + B_log_abs_grad_bound_z.col(ii).array();
               B_sign_z_grad_term.col(ii).array()  =        B_sign_y_m_y_sign_x_u.col(t1 + ii).array()  *  B_sign_grad_bound_z.col(ii).array()  ; // NOTE: B_sign_phi_Z_recip and B_sign_phi_Bound_Z are +'ve as densities !!
               
               ////// update deriv_L_T_T_inv log_abs and sign terms 
               // non-log-scale: deriv_L_T_T_inv =  ( - 1 /  (  L_Omega_double(t1 + ii + 1, t1 + ii + 1)  * L_Omega_double(t1 + ii + 1, t1 + ii + 1) )  )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + ii + 1, t1 + ii + 1)  ;
               log_abs_deriv_L_T_T_inv = -2.0 * log_abs_L_Omega_double(t1 + ii + 1, t1 + ii + 1)  + log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + ii + 1, t1 + ii + 1);
               sign_deriv_L_T_T_inv =  stan::math::sign( - 1.0 /  ( L_Omega_double(t1 + ii + 1, t1 + ii + 1)  * L_Omega_double(t1 + ii + 1, t1 + ii + 1)  )  )  * sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + ii + 1, t1 + ii + 1);
               
               
               // non-log-scale: deriv_Bound_Z_x_L.col(0).array() = 0.0;
               for (int jj = 0; jj < ii + 1; jj++) { 
                 // non-log-scale:  deriv_Bound_Z_x_L.col(ii + 1).array()  +=    L_Omega_double(t1 + ii + 1,t1 + jj)     *   A_z_grad_term.col(jj).array()     +   Z_std_norm.col(t1 + jj).array()     *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + ii + 1, t1 + jj) ;// +
                 log_abs_deriv_Bound_Z_x_L_comp.col(jj).array() =  log_abs_L_Omega_double(t1 + ii + 1, t1 + jj) +   B_log_abs_z_grad_term.col(jj).array()   +  
                                                                   B_log_Z_std_norm.col(t1 + jj).array() +  log_abs_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + ii + 1, t1 + jj) ; 
                 sign_deriv_Bound_Z_x_L_comp.col(jj).array() =   stan::math::sign(L_Omega_double(t1 + ii + 1, t1 + jj)) *   B_sign_z_grad_term.col(jj).array()   *  
                                                                 B_sign_Z_std_norm.col(t1 + jj).array() *  sign_Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + ii + 1, t1 + jj) ; 
               } 
               
               ///// then compute log_abs and sign of deriv_Bound_Z_x_L.col(0) using log_abs_sum_exp:
               log_abs_sum_exp_general_v2(log_abs_deriv_Bound_Z_x_L_comp.leftCols(ii + 1),
                                          sign_deriv_Bound_Z_x_L_comp.leftCols(ii + 1),
                                          vect_type, vect_type,
                                          log_abs_deriv_Bound_Z_x_L.col(ii + 1),
                                          sign_deriv_Bound_Z_x_L.col(ii + 1),
                                          container_max_logs,
                                          container_sum_exp_signed);
               
               ///// update grad_bound_z log_abs and sign terms
               // non-log-scale: grad_bound_z.col(ii + 1).array() =  deriv_L_T_T_inv * (Bound_Z.col(t1 + ii + 1).array() * L_Omega_double(t1 + ii + 1,t1 + ii + 1)  ) +  (1 / L_Omega_double(t1 + ii + 1, t1 + ii + 1)) * -   deriv_Bound_Z_x_L.col(ii + 1).array()  ;
               // log_abs_a.array() = log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1 + ii + 1).array() + log_abs_L_Omega_double(t1 + ii + 1, t1 + ii + 1); 
               // sign_a.array() = sign_deriv_L_T_T_inv * (B_sign_Bound_Z.col(t1 + ii + 1).array() * stan::math::sign(L_Omega_double(t1 + ii + 1, t1 + ii + 1)) );
               // log_abs_b.array() =  - log_abs_L_Omega_double(t1 + ii + 1, t1 + ii + 1) +  log_abs_deriv_Bound_Z_x_L.col(ii + 1).array();
               // sign_b.array() =  stan::math::sign(1.0 / L_Omega_double(t1 + ii + 1, t1 + ii + 1))  *   -1.0 * sign_deriv_Bound_Z_x_L.col(ii + 1).array();
               log_terms.col(0)  =   log_abs_deriv_L_T_T_inv + B_log_Bound_Z.col(t1 + ii + 1).array() + log_abs_L_Omega_double(t1 + ii + 1, t1 + ii + 1); 
               log_terms.col(1)  = - log_abs_L_Omega_double(t1 + ii + 1, t1 + ii + 1) +  log_abs_deriv_Bound_Z_x_L.col(ii + 1).array();
               sign_terms.col(0) =  sign_deriv_L_T_T_inv * (B_sign_Bound_Z.col(t1 + ii + 1).array() * stan::math::sign(L_Omega_double(t1 + ii + 1, t1 + ii + 1)) );
               sign_terms.col(1) =  stan::math::sign(1.0 / L_Omega_double(t1 + ii + 1, t1 + ii + 1))  *   -1.0 * sign_deriv_Bound_Z_x_L.col(ii + 1).array();
               
               ///// then compute log_abs and sign using log_abs_sum_exp:
               log_abs_sum_exp_general_v2(log_terms.leftCols(2),
                                          sign_terms.leftCols(2),
                                          vect_type, vect_type,
                                          B_log_abs_grad_bound_z.col(ii + 1),
                                          B_sign_grad_bound_z.col(ii + 1),
                                          container_max_logs,
                                          container_sum_exp_signed);
               
               ///// update grad_Phi_bound_z log_abs and sign terms
               // non-log-scale:  grad_Phi_bound_z.col(ii + 1).array()  =     phi_Bound_Z.col(t1 + ii + 1).array()  *   grad_bound_z.col(ii + 1).array() ;   // correct  (standard form)
               B_log_abs_grad_Phi_bound_z.col(ii + 1).array() = B_log_phi_Bound_Z.col(t1 + ii + 1).array() + B_log_abs_grad_bound_z.col(ii + 1).array(); 
               B_sign_grad_Phi_bound_z.col(ii + 1).array()  =     B_sign_grad_bound_z.col(ii + 1).array();    // note: sign of phi_Bound_Z = +1 as density!!
                 
               ///// update A_grad_prob log_abs and sign terms
               // non-log-scale:  A_grad_prob.col(ii + 1).array()  =   ( - y_sign_chunk.col(t1 + ii + 1).array()  ) *    grad_Phi_bound_z.col(ii + 1).array() ;  // correct  (standard form)
               B_log_abs_grad_prob.col(ii + 1).array() = B_log_abs_y_sign_chunk.col(t1 + ii + 1).array() +  B_log_abs_grad_Phi_bound_z.col(ii + 1).array(); 
               B_sign_grad_prob.col(ii + 1).array() =  -1.0 * B_sign_y_sign_chunk.col(t1 + ii + 1).array() *   B_sign_grad_Phi_bound_z.col(ii + 1).array(); 
           
         }
         
         /////// final computations
         //// NON-log-scale:
         // A_derivs_chain_container_vec.array() = 0.0;
         // ///// attempt at vectorising  // bookmark
         // for (int iii = 0; iii <  i; iii++) {
         //   A_derivs_chain_container_vec.array() +=  (    A_grad_prob.col(iii).array()  * (   prob.block(0, t1 + 0, chunk_size, i).rowwise().prod().array()  /  prob.col(t1 + iii).array()  ).array() ).array() ; // correct  (standard form)
         // }
         // grad_pi_wrt_b_raw(c, t1) +=        ( A_common_grad_term_1.col(t1).array()   *  A_derivs_chain_container_vec.array() ).sum();
         
         /////// final computations (log-scale)
         {
               for (int iii = 0; iii <  i; iii++) {
                 B_log_abs_derivs_chain_container_vec_comp.col(iii).array() =  B_log_abs_grad_prob.col(iii).array() +  B_log_prob_rowwise_prod_temp.col(t1).array() +  (-B_y1_log_prob).col(t1 + iii).array();
                 B_sign_derivs_chain_container_vec_comp.col(iii).array() = B_sign_grad_prob.col(iii).array();
               }
               
               log_abs_sum_exp_general_v2( B_log_abs_derivs_chain_container_vec_comp.leftCols(i),
                                           B_sign_derivs_chain_container_vec_comp.leftCols(i),
                                           vect_type, vect_type,
                                           log_sum_result, sign_sum_result,
                                           container_max_logs,
                                           container_sum_exp_signed);
               
               log_abs_bs_grad_array_col_for_each_n = B_log_common_grad_term_1.col(t1).array() + log_sum_result.array();
               sign_bs_grad_array_col_for_each_n = sign_sum_result;
               
               // Final scalar grad using log-sum-exp
               LogSumVecSingedResult log_sum_vec_signed_struct = log_sum_vec_signed_v1(log_abs_bs_grad_array_col_for_each_n,
                                                                                       sign_bs_grad_array_col_for_each_n, 
                                                                                       vect_type);
               grad_pi_wrt_b_raw(c, t1) +=   stan::math::exp(log_sum_vec_signed_struct.log_sum) * log_sum_vec_signed_struct.sign;
               
         }
         
         
       }
       
     }
     
    


}
  
  
  
  
  
  
  
  
  

 