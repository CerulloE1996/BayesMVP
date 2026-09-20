
#pragma once

 

#include <Eigen/Dense>
 
#include <iostream>
#include <vector>


 

 
using namespace Eigen;

 

 

 

   
 
/// updates tau and also snaper_w_vec
Eigen::Matrix<double, -1, 1>  fn_update_tau_w_diag_M_ADAM(   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  eigen_vector, 
                                             const double eigen_max,
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  theta_vec_initial, 
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  theta_vec_prop, 
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  snaper_m_vec,  
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  velocity_prop,
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  velocity_0,
                                             const double tau,  
                                             const double LR,
                                             const double ii, 
                                             const double n_burnin,
                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  sqrt_M_vec, 
                                             const double tau_m_adam,   
                                             const double tau_v_adam,  
                                             const double tau_ii,
                                             const double beta1_adam,
                                             const double beta2_adam,
                                             const double eps_adam
 ) {

   
   const double eta_w = 3.0;
   // const double beta1_adam = 0.0; // ADAM hyperparameter 1   
   // const double beta2_adam = 0.95; // ADAM hyperparameter 2
   // const double eps_adam = 1e-8; // ADAM "eps" for numerical stability
   const double rho = 1.0;
   
   const double tau_initial = tau;
   const double tau_m_adam_initial = tau_m_adam;
   const double tau_v_adam_initial = tau_v_adam;
   
   Eigen::Matrix<double, -1, 1> out_vec(3);
   
   if (ii > eta_w && eigen_max > 0) {
     
       //////////  ------------- this part is done for all K chains (parallel) --------------------------------
       const Eigen::Matrix<double, -1, 1> theta_diff_initial = theta_vec_initial - snaper_m_vec;
       const Eigen::Matrix<double, -1, 1> theta_diff_prop = theta_vec_prop - snaper_m_vec;
       
        
       Eigen::Matrix<double, -1, 1> pos_c_array = eigen_vector.array() * sqrt_M_vec.array() * theta_diff_initial.array();
       pos_c_array = pos_c_array.unaryExpr([](double x) { return std::isinf(x) ? 0.0 : x; });
       const double pos_c_per_chain = pos_c_array.sum();
      
       Eigen::Matrix<double, -1, 1> prop_c_array = eigen_vector.array() * sqrt_M_vec.array() * theta_diff_prop.array();
       prop_c_array = prop_c_array.unaryExpr([](double x) { return std::isinf(x) ? 0.0 : x; });
       const double prop_c_per_chain = prop_c_array.sum();
   
       const double diff_sq = prop_c_per_chain * prop_c_per_chain - pos_c_per_chain * pos_c_per_chain;
       
       const Eigen::Matrix<double, -1, 1> pos_c_grad_per_chain =  2.0 * pos_c_per_chain  * sqrt_M_vec.array() * eigen_vector.array();
       const Eigen::Matrix<double, -1, 1> prop_c_grad_per_chain = 2.0 * prop_c_per_chain * sqrt_M_vec.array() * eigen_vector.array();
   
       double tau_noisy_grad = diff_sq * (prop_c_grad_per_chain.dot(velocity_prop) +  pos_c_grad_per_chain.dot(velocity_0)) - (1.0 / tau_ii) * diff_sq * diff_sq;
       
       if (std::isnan(tau_noisy_grad)) {
         tau_noisy_grad = 0.0;
       }
       //////////  ------------- END OF PARALLEL PART (take averages of tau_noisy_grad across the K chains) --------------------------------
       
         // Update tau 
         // double diff_sq = std::pow(tau_ii, 2);
         // double tau_noisy_grad = diff_sq * (eigen_vector.dot(velocity_prop) +  eigen_vector.dot(velocity_0));
         const double tau_m_adam_new = beta1_adam * tau_m_adam + (1.0 - beta1_adam) * tau_noisy_grad;
         const double tau_v_adam_new = beta2_adam * tau_v_adam + (1.0 - beta2_adam) * std::pow(tau_noisy_grad, 2);
         const double tau_m_hat = tau_m_adam / (1.0 - std::pow(beta1_adam, ii));
         const double tau_v_hat = tau_v_adam / (1.0 - std::pow(beta2_adam, ii));
         
         const double current_alpha = LR * (1.0 - ((1.0 - LR) * ii) / n_burnin);
         const double log_tau = std::log(stan::math::fabs(tau)) + current_alpha * tau_m_hat / (std::sqrt(tau_v_hat) + eps_adam);
         const double tau_new = std::exp(log_tau);
         
         out_vec(0) = tau_new;
         out_vec(1) = tau_m_adam_new;
         out_vec(2) = tau_v_adam_new;
     
   }
   
   if (is_NaN_or_Inf_Eigen(out_vec) == true) {
     out_vec(0) = tau_initial;
     out_vec(1) = tau_m_adam_initial;
     out_vec(2) = tau_v_adam_initial;
   }
   
   return out_vec;
   
   
 }
 
 

 
 
 
 
 
 
 
 
 
Eigen::Matrix<double, -1, 1> fn_update_tau_w_dense_M_ADAM(  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  eigen_vector, 
                                            const double eigen_max,
                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  theta_vec_initial, 
                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  theta_vec_prop, 
                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  snaper_m_vec,  
                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  velocity_prop,
                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  velocity_0,
                                            const double tau,   
                                            const double LR, 
                                            const double ii, 
                                            const double n_burnin,
                                            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> M_dense_sqrt, 
                                            const double tau_m_adam,   
                                            const double tau_v_adam,  
                                            const double  tau_ii,
                                            const double beta1_adam,
                                            const double beta2_adam,
                                            const double eps_adam
) {
  
  
  const double eta_w = 3.0;
  // const double beta1_adam = 0.0; // ADAM hyperparameter 1   
  // const double beta2_adam = 0.95; // ADAM hyperparameter 2
  // const double eps_adam = 1e-8; // ADAM "eps" for numerical stability
  const double rho = 1.0;
  
  const double tau_initial = tau;
  const double tau_m_adam_initial = tau_m_adam;
  const double tau_v_adam_initial = tau_v_adam;
  
  Eigen::Matrix<double, -1, 1> out_vec(3);
    
    if (ii > eta_w && eigen_max > 0) {
      
              //////////  ------------- this part is done for all K chains (parallel) --------------------------------
         
              Eigen::Matrix<double, -1, 1>  pos_c_array =  ( eigen_vector.array() *  ( M_dense_sqrt * (theta_vec_initial - snaper_m_vec) ).array() ).matrix() ;
              pos_c_array = pos_c_array.unaryExpr([](double x) { return std::isinf(x) ? 0.0 : x; });
              const double pos_c_per_chain = pos_c_array.sum();
          
              Eigen::Matrix<double, -1, 1> prop_c_array =  ( eigen_vector.array() *  ( M_dense_sqrt * (theta_vec_prop -   snaper_m_vec) ).array() ).matrix() ;
              prop_c_array = prop_c_array.unaryExpr([](double x) { return std::isinf(x) ? 0.0 : x; });
              const double prop_c_per_chain = prop_c_array.sum();
              
              const double diff_sq = prop_c_per_chain * prop_c_per_chain - pos_c_per_chain * pos_c_per_chain;
             
              const Eigen::Matrix<double, -1, 1>   M_dense_sqrt_x_eigen_vec = (M_dense_sqrt * eigen_vector); // col vector
              const Eigen::Matrix<double, -1, 1>   pos_c_grad_per_chain =  2.0 * pos_c_per_chain   * M_dense_sqrt_x_eigen_vec;
              const Eigen::Matrix<double, -1, 1>   prop_c_grad_per_chain = 2.0 * prop_c_per_chain  * M_dense_sqrt_x_eigen_vec;
              
              // Update tau noisy gradient
              double tau_noisy_grad = diff_sq * (prop_c_grad_per_chain.dot(velocity_prop) +  pos_c_grad_per_chain.dot(velocity_0)) - 
                                      (0.5 * (1.0 + std::min(1.0, rho))/(tau_ii)) * (diff_sq * diff_sq);
              
              if (std::isnan(tau_noisy_grad)) {
                tau_noisy_grad = 0.0;
              }
              
              /////// Update tau (NOT done in parallel)
              const double tau_m_adam_new = beta1_adam * tau_m_adam + (1.0 - beta1_adam) * tau_noisy_grad;
              const double tau_v_adam_new = beta2_adam * tau_v_adam + (1.0 - beta2_adam) * std::pow(tau_noisy_grad, 2);
              const double tau_m_hat = tau_m_adam / (1.0 - std::pow(beta1_adam, ii));
              const double tau_v_hat = tau_v_adam / (1.0 - std::pow(beta2_adam, ii));
              
              const double current_alpha = LR * (1.0 - ((1.0 - LR) * ii) / n_burnin);
              const double log_tau = std::log(stan::math::fabs(tau)) + current_alpha * tau_m_hat / (std::sqrt(tau_v_hat) + eps_adam);
              const double tau_new  = std::exp(log_tau);
              
              out_vec(0) = tau_new;
              out_vec(1) = tau_m_adam_new;
              out_vec(2) = tau_v_adam_new;
    
  }
    
    if (is_NaN_or_Inf_Eigen(out_vec) == true) {
      out_vec(0) = tau_initial;
      out_vec(1) = tau_m_adam_initial;
      out_vec(2) = tau_v_adam_initial;
    }
    
    return out_vec;
    
  

}

 
 
 
 
 
 
 
 
 
 