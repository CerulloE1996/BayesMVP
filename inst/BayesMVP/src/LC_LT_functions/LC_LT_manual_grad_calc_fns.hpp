#pragma once


 
 
#include <Eigen/Dense>
 

 
 


 
 










// Gradient computation function template (no need to have any template parameters as not very modular e.g. only double's)
inline void fn_LC_LT_compute_bs_grad_v1(      Eigen::Ref<Eigen::Matrix<double, -1, -1>>   grad_pi_wrt_b_raw,
                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>>   deriv_Bound_Z_x_L,
                                              const int c,
                                              const std::vector<Eigen::Matrix<double, -1, -1>> &Jacobian_d_L_Sigma_wrt_b_3d_arrays_double,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  A_common_grad_term_1,
                                              const Eigen::Matrix<double, -1, -1> &L_Omega_double,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  prob,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  prob_recip,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  Bound_Z,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  Z_std_norm,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  phi_Bound_Z,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  phi_Z_recip,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  y_sign_chunk,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  y_m_y_sign_x_u,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  A_prop_rowwise_prod_temp, 
                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>>   grad_bound_z,
                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>>   grad_Phi_bound_z,
                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>>   A_z_grad_term,
                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>>   A_grad_prob,
                                              Eigen::Ref<Eigen::Matrix<double, -1, 1>>    A_prod_container,
                                              Eigen::Ref<Eigen::Matrix<double, -1, 1>>    A_derivs_chain_container_vec,
                                              const bool compute_final_scalar_grad,
                                              const Model_fn_args_struct &Model_args_as_cpp_struct

) {

  const int  n_class = Model_args_as_cpp_struct.Model_args_ints(1);

  const int chunk_size = Z_std_norm.rows();
  const int n_tests = Z_std_norm.cols();
  
  ///////////////////////// deriv of diagonal elements (not needed if using the "standard" or "Stan" Cholesky parameterisation of Omega)
  //////// w.r.t last diagonal first
  {
    int  t1 = n_tests - 1;
    
    double deriv_L_T_T_inv =  ( - 1 /  ( L_Omega_double(t1,t1)  * L_Omega_double(t1,t1) ) )   *   Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1)  ;
    
    deriv_Bound_Z_x_L.col(0).array() = 0.0;
    for (int t = 0; t < t1; t++) {
      deriv_Bound_Z_x_L.col(0).array() +=   Z_std_norm.col(t).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
    }
    
    double deriv_a = 0 ; //  stan::math::pow( (1 + (bs_mat_double(c, t1)*bs_mat_double(c, t1)) ), -0.5) * bs_mat_double(c, t1)  * LT_theta(c, t1)  ;
    deriv_Bound_Z_x_L.col(0).array()   = deriv_a -     deriv_Bound_Z_x_L.col(0).array();
    
    grad_bound_z.col(0).array() =   deriv_L_T_T_inv * (Bound_Z.col(t1).array() * L_Omega_double(t1,t1)  ) +  (1 / L_Omega_double(t1, t1)) *   deriv_Bound_Z_x_L.col(0).array()  ;
    grad_Phi_bound_z.col(0)  =  ( phi_Bound_Z.col(t1).array() *  (  grad_bound_z.col(0).array() )  ) .matrix();   
    A_grad_prob.col(0)   =  (   - y_sign_chunk.col(t1).array()  *   grad_Phi_bound_z.col(0).array() ).matrix() ;     
    
    grad_pi_wrt_b_raw(c, t1)  +=   (   A_common_grad_term_1.col(t1).array()    *            A_grad_prob.col(0).array()    ).matrix().sum()   ; 
    
  }
  
  //////// then w.r.t the second-to-last diagonal
  {
    
    int  t1 = n_tests - 2;
    
    double deriv_L_T_T_inv =  ( - 1.0 /   ( L_Omega_double(t1,t1)  * L_Omega_double(t1,t1) ) )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1)  ;
    
    deriv_Bound_Z_x_L.col(0).array() = 0.0;
    for (int t = 0; t < t1; t++) {
      deriv_Bound_Z_x_L.col(0).array() += Z_std_norm.col(t).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
    }
    
    double deriv_a = 0 ; //   stan::math::pow( (1.0 + (bs_mat_double(c, t1)*bs_mat_double(c, t1)) ), -0.5) * bs_mat_double(c, t1)   * LT_theta(c, t1)   ;
    deriv_Bound_Z_x_L.col(0).array()   = deriv_a -     deriv_Bound_Z_x_L.col(0).array();
    
    grad_bound_z.col(0).array() =  deriv_L_T_T_inv * (Bound_Z.col(t1).array() * L_Omega_double(t1,t1)  ) +  (1.0 / L_Omega_double(t1, t1)) *     deriv_Bound_Z_x_L.col(0).array()  ;
    grad_Phi_bound_z.col(0)  =  ( phi_Bound_Z.col(t1).array() *  (  grad_bound_z.col(0).array() )  ) .matrix();   
    A_grad_prob.col(0)   =  (   - y_sign_chunk.col(t1).array()  *   grad_Phi_bound_z.col(0).array() ).matrix() ;     
    
    A_z_grad_term.col(0).array()  =      (  ( (  y_m_y_sign_x_u.col(t1).array()  * phi_Z_recip.col(t1).array()  ).array()    * phi_Bound_Z.col(t1).array() *   grad_bound_z.col(0).array()  ).array() ).matrix()  ; 
    ////////
    
    deriv_L_T_T_inv =  ( - 1 /  ( L_Omega_double(t1 + 1, t1 + 1)  * L_Omega_double(t1 + 1, t1 + 1)  )  )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + 1, t1 + 1)  ;
    deriv_Bound_Z_x_L.col(1).array()  =    L_Omega_double(t1+1,t1) *   A_z_grad_term.col(0).array()     +   Z_std_norm.col(t1).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
    grad_bound_z.col(1).array() =  deriv_L_T_T_inv * (Bound_Z.col(t1+1).array() * L_Omega_double(t1+1,t1+1)  ) +  (1.0 / L_Omega_double(t1+1, t1+1)) * -   deriv_Bound_Z_x_L.col(1).array()  ;
    
    grad_Phi_bound_z.col(1) =   ( phi_Bound_Z.col(t1 + 1).array() *  (    grad_bound_z.col(1).array()   ) ).matrix();   
    A_grad_prob.col(1)   =   (  - y_sign_chunk.col(t1 + 1).array()  *     grad_Phi_bound_z.col(1).array()  ).array().matrix() ;    
    
    grad_pi_wrt_b_raw(c, t1) +=   ( ( A_common_grad_term_1.col(t1).array() )    *   ( prob.col(t1 + 1).array()  *      A_grad_prob.col(0).array()  +   prob.col(t1).array()  *         A_grad_prob.col(1).array()   ) ).sum() ;
    
  }
  
  
  //////// then w.r.t the third-to-last diagonal .... etc
  {
    
    for (int i = 3; i < n_tests + 1; i++) {
      
      A_grad_prob.array()   = 0.0;
      A_z_grad_term.array() = 0.0;
      
      int  t1 = n_tests - i;
      
      double deriv_L_T_T_inv =  ( - 1 /   ( L_Omega_double(t1,t1)  * L_Omega_double(t1,t1) ) )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t1)  ;
      
      deriv_Bound_Z_x_L.col(0).array() = 0;
      for (int t = 0; t < t1; t++) {
        deriv_Bound_Z_x_L.col(0).array() += Z_std_norm.col(t).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1, t);
      }
      
      double deriv_a = 0 ; //  stan::math::pow( (1 + (bs_mat_double(c, t1)*bs_mat_double(c, t1)) ), -0.5) * bs_mat_double(c, t1) *   LT_theta(c, t1)   ;
      deriv_Bound_Z_x_L.col(0).array()   = deriv_a -     deriv_Bound_Z_x_L.col(0).array();
      
      grad_bound_z.col(0).array() =  deriv_L_T_T_inv * (Bound_Z.col(t1).array() * L_Omega_double(t1,t1)  ) +  (1 / L_Omega_double(t1, t1)) *    deriv_Bound_Z_x_L.col(0).array()  ;
      grad_Phi_bound_z.col(0)  =  ( phi_Bound_Z.col(t1).array() *  (  grad_bound_z.col(0).array() )  ) .matrix();   
      A_grad_prob.col(0)   =  (   - y_sign_chunk.col(t1).array()  *   grad_Phi_bound_z.col(0).array() ).matrix() ;     
      
      A_z_grad_term.col(0).array()  =      (  ( (  y_m_y_sign_x_u.col(t1).array()  * phi_Z_recip.col(t1).array()  ).array()    * phi_Bound_Z.col(t1).array() *   grad_bound_z.col(0).array()  ).array() ).matrix()  ;
      //////
      
      
      
      deriv_L_T_T_inv =  ( - 1 /  ( L_Omega_double(t1+1,t1+1)  * L_Omega_double(t1+1,t1+1)  )  )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1+1)  ;
      deriv_Bound_Z_x_L.col(1).array()  =    L_Omega_double(t1+1,t1) *   A_z_grad_term.col(0).array()     +   Z_std_norm.col(t1).array() *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1+1, t1);
      grad_bound_z.col(1).array() =  deriv_L_T_T_inv * (Bound_Z.col(t1+1).array() * L_Omega_double(t1+1,t1+1)  ) +  (1 / L_Omega_double(t1+1, t1+1)) * -  deriv_Bound_Z_x_L.col(1).array()  ;
      
      grad_Phi_bound_z.col(1) =   ( phi_Bound_Z.col(t1 + 1).array() *  (    grad_bound_z.col(1).array()   ) ).matrix();   
      A_grad_prob.col(1)  =   (  - y_sign_chunk.col(t1 + 1).array()  *     grad_Phi_bound_z.col(1).array()  ).array().matrix() ;  
      
      
      for (int ii = 1; ii < i - 1; ii++) {
        A_z_grad_term.col(ii).array()  =    (  ( (  y_m_y_sign_x_u.col(t1 + ii).array()  * phi_Z_recip.col(t1 + ii).array()  ).array()    * phi_Bound_Z.col(t1 + ii).array() *   grad_bound_z.col(ii).array()  ).array() ).matrix() ;     
        
        deriv_L_T_T_inv =  ( - 1 /  (  L_Omega_double(t1 + ii + 1, t1 + ii + 1)  * L_Omega_double(t1 + ii + 1, t1 + ii + 1) )  )  * Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + ii + 1, t1 + ii + 1)  ;
        
        deriv_Bound_Z_x_L.col(ii + 1).array()   =  0.0;
        for (int jj = 0; jj < ii + 1; jj++) {
          deriv_Bound_Z_x_L.col(ii + 1).array()  +=    L_Omega_double(t1 + ii + 1,t1 + jj)     *   A_z_grad_term.col(jj).array()     +   Z_std_norm.col(t1 + jj).array()     *  Jacobian_d_L_Sigma_wrt_b_3d_arrays_double[t1](t1 + ii + 1, t1 + jj) ;// +
        }
        grad_bound_z.col(ii + 1).array() =  deriv_L_T_T_inv * (Bound_Z.col(t1 + ii + 1).array() * L_Omega_double(t1 + ii + 1,t1 + ii + 1)  ) +  (1 / L_Omega_double(t1 + ii + 1, t1 + ii + 1)) * -   deriv_Bound_Z_x_L.col(ii + 1).array()  ;
        grad_Phi_bound_z.col(ii + 1).array()  =     phi_Bound_Z.col(t1 + ii + 1).array()  *   grad_bound_z.col(ii + 1).array() ;   
        A_grad_prob.col(ii + 1).array()  =   ( - y_sign_chunk.col(t1 + ii + 1).array()  ) *    grad_Phi_bound_z.col(ii + 1).array() ;  
        
      }
      
      A_derivs_chain_container_vec.array() = 0.0;
      
      ///// attempt at vectorising  // bookmark
      for (int iii = 0; iii <  i; iii++) {
        A_derivs_chain_container_vec.array() +=  (    A_grad_prob.col(iii).array()  * (   prob.block(0, t1 + 0, chunk_size, i).rowwise().prod().array()  /  prob.col(t1 + iii).array()  ).array() ).array() ; 
      }
      
      grad_pi_wrt_b_raw(c, t1) +=        ( A_common_grad_term_1.col(t1).array()   *  A_derivs_chain_container_vec.array() ).sum();
      
    }
    
  }
  
} 

 




 















