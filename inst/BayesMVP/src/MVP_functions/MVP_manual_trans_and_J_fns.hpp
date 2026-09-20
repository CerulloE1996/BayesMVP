
#pragma once


 

#include <Eigen/Dense>
 

 
 
using namespace Eigen;

 

 

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE





 
 
 
 
 
ALWAYS_INLINE  void  fn_MVP_compute_nuisance(         Eigen::Matrix<double, -1, 1> &u_vec,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec,
                                                      const Model_fn_args_struct &Model_args_as_cpp_struct
) {
  
          const int  n_class = Model_args_as_cpp_struct.Model_args_ints(1);
          const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
          
          const std::string &nuisance_transformation = Model_args_as_cpp_struct.Model_args_strings(12);
          const std::string &vect_type_tanh = Model_args_as_cpp_struct.Model_args_strings(6);
          const std::string &vect_type_Phi = Model_args_as_cpp_struct.Model_args_strings(7);
          
          u_vec.setZero();
        
          const int n_us = u_unc_vec.size();
          
          if (nuisance_transformation == "Phi") {
        
              u_vec.array() +=      fn_EIGEN_double( u_unc_vec, "Phi", vect_type_Phi, false).array();
        
          } else if (nuisance_transformation == "Phi_approx") {
        
              u_vec.array() +=      fn_EIGEN_double( u_unc_vec, "Phi_approx", vect_type_Phi, false).array();
        
          } else if (nuisance_transformation == "Phi_approx_rough") {   ;
        
              u_vec.array() +=      fn_EIGEN_double( 1.702 * u_unc_vec, "inv_logit", vect_type_Phi, false).array();
        
          } else if (nuisance_transformation == "tanh") {
        
              u_vec.array() +=   ( 0.5 * ( fn_EIGEN_double( u_unc_vec, "tanh", vect_type_tanh, false).matrix().array() + 1.0).array() ).array();
        
          }
          
          // const double u_lo = 1e-12, u_hi = 1.0 - 1e-12;
          // u_vec = u_vec.array().max(u_lo).min(u_hi).matrix();

}







ALWAYS_INLINE double fn_MVP_compute_nuisance_log_jac_u(       const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_vec, // Eigen::Matrix<double, -1, 1>   &u_vec,
                                                              const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec,
                                                              const Model_fn_args_struct &Model_args_as_cpp_struct
) {

          const double a = 0.07056;
          const double b = 1.5976;
          const double a_times_3 = 3.0 * 0.07056;
          
          const std::string &nuisance_transformation = Model_args_as_cpp_struct.Model_args_strings(12);
          const std::string &vect_type_log = Model_args_as_cpp_struct.Model_args_strings(4);
          
          double log_jac_u = 0.0;
          
          if (nuisance_transformation == "Phi") {
        
              log_jac_u  =  - 0.5 * stan::math::log(2 * M_PI)  - 0.5 * u_unc_vec.array().square().sum() ;
        
          } else if (nuisance_transformation == "Phi_approx") {
        
              log_jac_u   =  fn_EIGEN_double((a_times_3 * u_unc_vec.array().square() +  b).matrix(), "log", vect_type_log).sum();
              log_jac_u   += fn_EIGEN_double(u_vec, "log", vect_type_log).sum(); 
              log_jac_u   += fn_EIGEN_double(u_vec, "log1m", vect_type_log).sum();
        
          } else if (nuisance_transformation == "Phi_approx_rough") {
        
              log_jac_u  = stan::math::log(1.702) + fn_EIGEN_double( u_vec, "log", vect_type_log).sum() + fn_EIGEN_double( u_vec, "log1m", vect_type_log).sum()  ;
        
        
          } else if (nuisance_transformation == "tanh") {
        
              log_jac_u  = - stan::math::log(2.0) + fn_EIGEN_double( u_vec, "log", vect_type_log ).sum() + fn_EIGEN_double( u_vec , "log1m", vect_type_log ).sum();
        
          }
          
          
          /////  log_jac_u  =  - 0.5 * stan::math::log(2 * M_PI)  - 0.5 * u_unc_vec.array().square().sum() ;   
          
          return log_jac_u;

}








// Gradient computation function template (no need to have any template parameters as not very modular e.g. only double's)
ALWAYS_INLINE void fn_MVP_nuisance_first_deriv(     Eigen::Matrix<double, -1, 1> &du_wrt_duu,
                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_vec,
                                                    const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec,
                                                    const Model_fn_args_struct &Model_args_as_cpp_struct
) {

          const double a = 0.07056;
          const double b = 1.5976;
          const double a_times_3 = 3.0 * 0.07056;
          const double sqrt_2_pi_recip =   1.0 / sqrt(2.0 * M_PI) ; //  0.3989422804;
          
          du_wrt_duu.setZero();
          
          const std::string &nuisance_transformation = Model_args_as_cpp_struct.Model_args_strings(12);
          const std::string &vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);
        
          const int n_us = u_unc_vec.size();
        
          if (nuisance_transformation == "Phi") { 
      
              du_wrt_duu.array() +=    ( sqrt_2_pi_recip * fn_EIGEN_double(  ( -0.5 * (u_unc_vec.array().square()) ).matrix() , "exp", vect_type_exp) ).array() ;  
        
          } else if (nuisance_transformation == "Phi_approx") {  
             
              du_wrt_duu.array()  +=   (   (a_times_3 * u_unc_vec.array().square() +  b).array() * u_vec.array() * (1.0 - u_vec.array()) ).array() ;    
        
          } else if (nuisance_transformation == "Phi_approx_rough") {   ;   
        
              du_wrt_duu.array()  +=   1.702 * u_vec.array() * ( 1.0 - u_vec.array() )  ;    
        
          } else if (nuisance_transformation == "tanh") {
        
              du_wrt_duu.array() +=    2.0 * u_vec.array() * (1.0 - u_vec.array() ) ;     
        
          }
        
          //  return du_wrt_duu;

}










// Gradient computation function template (no need to have any template parameters as not very modular e.g. only double's)
ALWAYS_INLINE void  fn_MVP_nuisance_deriv_of_log_det_J(   Eigen::Matrix<double, -1, 1> &d_J_wrt_duu,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_vec,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> du_wrt_duu,
                                                          const Model_fn_args_struct &Model_args_as_cpp_struct
) {
  
          const double a = 0.07056;
          const double b = 1.5976;
          const double a_times_3 = 3.0 * 0.07056;
          
          d_J_wrt_duu.setZero();
          
          const std::string &nuisance_transformation = Model_args_as_cpp_struct.Model_args_strings(12);
          
          const int n_us = u_unc_vec.size();
          
          if (nuisance_transformation == "Phi") { 
        
                d_J_wrt_duu.array() +=  ( - u_unc_vec.array() ).array() ; 
            
          } else if (nuisance_transformation == "Phi_approx") {   
             
                d_J_wrt_duu.array() += (  ( du_wrt_duu.array() * (  (1.0 - 2.0 * u_vec.array()  ) / ( u_vec.array() * (1.0 - u_vec.array() ) )  )   ).array()  + (  (  1.0 - 2.0 * u_vec.array() )  / (a_times_3*u_unc_vec.array().square() + b).array() ) ).array() ;    
            
          } else if (nuisance_transformation == "Phi_approx_rough") {   ;    
            
                d_J_wrt_duu.array() +=  1.702 * (1.0 - 2.0 * u_vec.array() ) ;    
            
          } else if (nuisance_transformation == "tanh") { 
            
                d_J_wrt_duu.array() +=  2.0 * (1.0 - 2.0 * u_vec.array() ) ;   
            
          }
          
          //  return d_J_wrt_duu;
  
}







 