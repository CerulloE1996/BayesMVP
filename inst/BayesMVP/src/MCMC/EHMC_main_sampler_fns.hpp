
#pragma once

 


#include <random>

  

#include <Eigen/Dense>
 
 
 

#include <unsupported/Eigen/SpecialFunctions>
 
 
 
 
using namespace Eigen;
 
 
 
 
 

 
// HMC sampler functions   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


 
 
 
 
 
ALWAYS_INLINE  bool check_divergence(   double log_ratio,
                                        HMCResult &result_input) {
  
  
      if (result_input.check_state_valid() == false) { 
        return true;
      }
  
      if (!std::isfinite(log_ratio)) {
        return true;  // Reject non-finite proposals
      }
      
      // Eigen::Matrix<double, -1, -1> lp_and_grad_outs_copy = result_input.lp_and_grad_outs();
      
      /// first check is any NaN or Inf values 
      // if (is_NaN_or_Inf_Eigen(lp_and_grad_outs_copy) == true) {
      if (is_NaN_or_Inf_Eigen(result_input.lp_and_grad_outs()) == true) {
     
            // for (int i = 0; i < lp_and_grad_outs.size(); ++i) {
            //   if (std::isnan(lp_and_grad_outs(i)) || std::isinf(lp_and_grad_outs(i))) {
            //     std::cout  << "NaN or Inf detected at indices: "  << "\n"   ;
            //     std::cout  << i << " "  << "\n"    << std::endl; 
            //   }
            // }
            
            return true;
        
      }
      
      // //  now check for large energy changes
      // {
      //     const double energy_diff = std::fabs(log_ratio);
      //     const double energy_threshold = 100000000.0; // arbitrary
      //   
      //     if (energy_diff > energy_threshold) {
      //       return true;
      //     }
      // }
      
      // if (std::abs(log_ratio) > 10000) {
      //   return true;  // Reject suspiciously large energy differences
      // }
      if (std::abs(log_ratio) > 1000.0) return true;
    
      return false; // If no issues detected
  
} 



 
 
 
 
 
// ALWAYS_INLINE  bool proposal_div(     const double log_ratio,
//                                       const double energy_new,
//                                       const double energy_old) {
// 
//    if (!std::isfinite(log_ratio)) {
//      return true;  // Reject non-finite proposals
//    }
// 
//    // if (std::abs(log_ratio) > 10000) {
//    //   return true;  // Reject suspiciously large energy differences
//    // }
// 
//    return false;
// 
// }

//  
//  
// 
// ALWAYS_INLINE  void check_numeric_stability(const Eigen::Matrix<double, -1, 1> &vec, 
//                              const std::string &name) {
//   
//   if (vec.hasNaN()) {
//     throw std::runtime_error("NaN detected in " + name);
//   }
//   if (!vec.allFinite()) {
//     throw std::runtime_error("Inf detected in " + name);
//   }
//   
// }
// 







ALWAYS_INLINE  void leapfrog_integrator_dense_M_standard_HMC_main_InPlace(    Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
                                                                              Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
                                                                              Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                                                              const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
                                                                              const Eigen::Matrix<double, -1, -1> &M_inv_dense_main,
                                                                              const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                              const int L_ii,
                                                                              const double eps,
                                                                              const std::string &Model_type,
                                                                              const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
                                                                              const std::string &grad_option,
                                                                              const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                              const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                              std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                              const int n_threads_WCP,
                                                                              std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                                                                                                 const std::string &,
                                                                                                 const bool, const bool, const bool,
                                                                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                 const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                                                                                                 const std::string &,
                                                                                                 const Model_fn_args_struct &,
                                                                                                 const Stan_model_struct &,
                                                                                                 std::vector<LC_MVP_workspace_struct> &,
                                                                                                 const int &)> fn_lp_grad_InPlace

) {

      const int N = Model_args_as_cpp_struct.N;
      const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
      const int n_params_main = Model_args_as_cpp_struct.n_params_main;
      const int n_params = n_params_main + n_nuisance;

      Eigen::Matrix<double, -1, 1> grad_main =  lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);

      for (int l = 0; l < L_ii; l++) {
            if (Model_args_as_cpp_struct.burnin_leapfrog_steps != nullptr) {
                ++(*Model_args_as_cpp_struct.burnin_leapfrog_steps);
            }

            // Update velocity (first half step)
            //Eigen::Matrix<double, -1, 1> temp_1 = M_inv_dense_main * grad_main;
            velocity_main_vec_proposed_ref.array() +=    0.5 * eps * (M_inv_dense_main * grad_main).array() ;

            //// updae params by full step
            theta_main_vec_proposed_ref.array()  +=  eps *     velocity_main_vec_proposed_ref.array() ;

            // Update lp and gradients
            fn_lp_grad_InPlace(lp_and_grad_outs,
                               Model_type, force_autodiff, force_PartialLog, multi_attempts,
                               theta_main_vec_proposed_ref, theta_us_vec_initial_ref, y_ref, grad_option,
                               Model_args_as_cpp_struct,
                               Stan_model_as_cpp_struct, 
                               LC_MVP_ws_structs,
                               n_threads_WCP);
            if (!std::isfinite(lp_and_grad_outs(0))) return; 
            grad_main =   lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);

            // Update velocity (second half step)
            //Eigen::Matrix<double, -1, 1> temp_2 = M_inv_dense_main * grad_main;
            velocity_main_vec_proposed_ref.array() +=    0.5 * eps * (M_inv_dense_main * grad_main).array() ;

      } // End of leapfrog steps

}






 
 
 
 
 
 
 
 
 
 
 
 
 
 
 



ALWAYS_INLINE  void leapfrog_integrator_diag_M_standard_HMC_main_InPlace(       Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
                                                                                                Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
                                                                                                Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                                                                                const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
                                                                                                const Eigen::Matrix<double, -1, -1> &M_inv_main_vec,
                                                                                                const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                                                const int L_ii,
                                                                                                const double eps,
                                                                                                const std::string &Model_type,
                                                                                                const bool force_autodiff, const bool force_PartialLog,  const bool multi_attempts,
                                                                                                const std::string &grad_option,
                                                                                                const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                                const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                                                std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                                const int n_threads_WCP,
                                                                                                std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                                                                                                                   const std::string &,
                                                                                                                   const bool, const bool, const bool,
                                                                                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                                   const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                                                                                                                   const std::string &,
                                                                                                                   const Model_fn_args_struct &,
                                                                                                                   const Stan_model_struct &, 
                                                                                                                   std::vector<LC_MVP_workspace_struct> &,
                                                                                                                   const int &)> fn_lp_grad_InPlace

) {

      const int N = Model_args_as_cpp_struct.N;
      const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
      const int n_params_main = Model_args_as_cpp_struct.n_params_main;
      const int n_params = n_params_main + n_nuisance;

      Eigen::Matrix<double, -1, 1> grad_main =   lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);

      for (int l = 0; l < L_ii; l++) {
            if (Model_args_as_cpp_struct.burnin_leapfrog_steps != nullptr) {
                ++(*Model_args_as_cpp_struct.burnin_leapfrog_steps);
            }

            // Update velocity (first half step)
            velocity_main_vec_proposed_ref.array() +=   0.5 * eps * M_inv_main_vec.array() *  grad_main.array()   ;

            //// updae params by full step
            theta_main_vec_proposed_ref.array()  +=  eps *     velocity_main_vec_proposed_ref.array() ;

            // Update lp and gradients
            fn_lp_grad_InPlace(lp_and_grad_outs,
                               Model_type, force_autodiff, force_PartialLog, multi_attempts,
                               theta_main_vec_proposed_ref, theta_us_vec_initial_ref, y_ref, grad_option,
                               Model_args_as_cpp_struct,
                               Stan_model_as_cpp_struct, 
                               LC_MVP_ws_structs,
                               n_threads_WCP);
            if (!std::isfinite(lp_and_grad_outs(0))) return; 
            grad_main =   lp_and_grad_outs.segment(1 + n_nuisance, n_params_main);

            // Update velocity (second half step)
            velocity_main_vec_proposed_ref.array() +=  0.5 * eps * M_inv_main_vec.array() *  grad_main.array()   ;

      } // End of leapfrog steps

}












 
 
 
 
 
 
 
 
 
 


 
 
template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE  void                                        fn_standard_HMC_main_only_single_iter_InPlace_process(  HMCResult &result_input,
                                                                                                                   T &rng_main,
                                                                                                                   const std::string &Model_type,
                                                                                                                   const bool  force_autodiff,
                                                                                                                   const bool  force_PartialLog,
                                                                                                                   const bool  multi_attempts,
                                                                                                                   const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                                                                   const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                                                   EHMC_fn_args_struct  &EHMC_args_as_cpp_struct, /// pass by ref. to modify (???)
                                                                                                                   const EHMC_Metric_struct   &EHMC_Metric_struct_as_cpp_struct,
                                                                                                                   const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                                                                   std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                                                   const int n_threads_WCP
) {


    //// important params
    const int N = Model_args_as_cpp_struct.N;
    const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
    const int n_params_main = Model_args_as_cpp_struct.n_params_main;
    const int n_params = n_params_main + n_nuisance;
  
    const std::string grad_option = "main_only";  
     
    const std::string metric_shape_main = EHMC_Metric_struct_as_cpp_struct.metric_shape_main;
    
    double U_x_initial =  0.0 ; 
    double U_x_prop =  0.0 ; 
    double log_posterior_prop =  0.0 ; 
    double log_posterior_0  =   0.0 ;  
    double log_ratio = 0.0;
    double energy_old = 0.0;
    double energy_new = 0.0;
    
    /// result_input.store_current_state(); // sets initial theta and velocity to current theta and velocity
    result_input.main_theta_vec_0() =  result_input.main_theta_vec();
    result_input.us_theta_vec_0() =    result_input.us_theta_vec();
   
  {

      {
          //// Eigen::Matrix<double, -1, 1> std_norm_vec_main = Eigen::Matrix<double, -1, 1>::Zero(n_params_main);
          generate_random_std_norm_vec_InPlace(result_input.main_velocity_0_vec(), rng_main);
          if (metric_shape_main == "dense") result_input.main_velocity_0_vec()  = EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main_chol * result_input.main_velocity_0_vec();
          if (metric_shape_main == "diag")  result_input.main_velocity_0_vec().array() = result_input.main_velocity_0_vec().array() *  (EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec).array().sqrt() ; 
      }
     
    {
      
 
      { 
        
      try {
        
                result_input.main_velocity_vec_proposed()  =   result_input.main_velocity_0_vec(); // set initial velocity
                result_input.main_theta_vec_proposed() =       result_input.main_theta_vec();   // set to CURRENT theta   
      
                //// ---------------------------------------------------------------------------------------------------------------///    Perform L leapfrogs   ///------------------------------------------
                if (EHMC_args_as_cpp_struct.tau_main < EHMC_args_as_cpp_struct.eps_main) {
                        EHMC_args_as_cpp_struct.tau_main = EHMC_args_as_cpp_struct.eps_main; 
                }
                // Draw + assign tau_ii:
                EHMC_args_as_cpp_struct.tau_main_ii = generate_random_tau_ii(EHMC_args_as_cpp_struct.tau_main, rng_main);
                if (EHMC_args_as_cpp_struct.tau_main_ii < EHMC_args_as_cpp_struct.eps_main) { 
                        EHMC_args_as_cpp_struct.tau_main_ii = EHMC_args_as_cpp_struct.eps_main; 
                }
                // Compute L_ii:
                int  L_ii = std::ceil(  EHMC_args_as_cpp_struct.tau_main_ii / EHMC_args_as_cpp_struct.eps_main );
                if (L_ii < 1) { L_ii = 1 ; }
                
                //// initial lp  (and grad)
                fn_lp_grad_InPlace(     result_input.lp_and_grad_outs(), 
                                        Model_type, 
                                        force_autodiff, force_PartialLog, multi_attempts,
                                        result_input.main_theta_vec(),  result_input.us_theta_vec(), 
                                        y_ref, 
                                        grad_option,
                                        Model_args_as_cpp_struct,  
                                        Stan_model_as_cpp_struct, 
                                        LC_MVP_ws_structs,
                                        n_threads_WCP);
                
                result_input.lp_and_grad_outs_0() = result_input.lp_and_grad_outs(); // ----
                
                log_posterior_0 =  result_input.lp_and_grad_outs()(0);
                U_x_initial = - log_posterior_0; //// initial energy
                
                if (metric_shape_main == "dense") { 
                  
                          leapfrog_integrator_dense_M_standard_HMC_main_InPlace(      result_input.main_velocity_vec_proposed(),
                                                                                      result_input.main_theta_vec_proposed(),
                                                                                      result_input.lp_and_grad_outs(),
                                                                                      result_input.us_theta_vec(),
                                                                                      EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main,
                                                                                      y_ref,
                                                                                      L_ii,
                                                                                      EHMC_args_as_cpp_struct.eps_main,
                                                                                      Model_type,
                                                                                      force_autodiff, force_PartialLog, multi_attempts,
                                                                                      grad_option,
                                                                                      Model_args_as_cpp_struct,
                                                                                      Stan_model_as_cpp_struct, 
                                                                                      LC_MVP_ws_structs,
                                                                                      n_threads_WCP,
                                                                                      fn_lp_grad_InPlace);
                  
                } else { 
                  
                          leapfrog_integrator_diag_M_standard_HMC_main_InPlace(       result_input.main_velocity_vec_proposed(),
                                                                                      result_input.main_theta_vec_proposed(),
                                                                                      result_input.lp_and_grad_outs(),
                                                                                      result_input.us_theta_vec(),
                                                                                      EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec,
                                                                                      y_ref,
                                                                                      L_ii,
                                                                                      EHMC_args_as_cpp_struct.eps_main,
                                                                                      Model_type,
                                                                                      force_autodiff, force_PartialLog, multi_attempts,
                                                                                      grad_option,
                                                                                      Model_args_as_cpp_struct,
                                                                                      Stan_model_as_cpp_struct, 
                                                                                      LC_MVP_ws_structs,
                                                                                      n_threads_WCP,
                                                                                      fn_lp_grad_InPlace);
                  
                }
                
                //// proposed lp  
                log_posterior_prop =  result_input.lp_and_grad_outs()(0);
                if (EHMC_args_as_cpp_struct.record_kinetic_energy_tau_derivatives) result_input.record_kinetic_energy_rate_main();
                U_x_prop = - log_posterior_prop; // initial energy
              
                //////////////////////////////////////////////////////////////////    M-H acceptance step  (i.e, Accept/Reject step)
                energy_old = U_x_initial;
                energy_new = U_x_prop;
                
                if (metric_shape_main == "dense") {
                    
                    energy_old += 0.5 * result_input.main_velocity_0_vec().transpose() * EHMC_Metric_struct_as_cpp_struct.M_dense_main * result_input.main_velocity_0_vec();
                    energy_new += 0.5 * result_input.main_velocity_vec_proposed().transpose() * EHMC_Metric_struct_as_cpp_struct.M_dense_main * result_input.main_velocity_vec_proposed();
                    log_ratio = - energy_new + energy_old;
                    
                } else if (metric_shape_main == "diag") {
                  
                    energy_old +=  0.5 * (result_input.main_velocity_0_vec().array().square() * (1.0 / EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec.array())).sum();
                    energy_new +=  0.5 * (result_input.main_velocity_vec_proposed().array().square() * (1.0 / EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec.array())).sum();
                    log_ratio = - energy_new + energy_old;
                  
                }
       
              if  (check_divergence(log_ratio, result_input) == true)    { /// if main_div, reject proposal 
                  
                          //// if  div, reject proposal 
                          result_input.main_div() = 1;  
                          result_input.main_p_jump() = 0.0;
                          result_input.reject_proposal_main();  // # reject proposal
                
              } else {  // if no main_div, carry on with MH step 
                
                          result_input.main_div() = 0;
                          result_input.main_p_jump() = std::min(1.0, stan::math::exp(log_ratio));
                          
                          const double rand_unif = generate_random_std_uniform(rng_main);
                          const double log_rand_unif = stan::math::log(rand_unif);
    
                          if  (log_rand_unif > log_ratio)   {  // # reject proposal
                                 result_input.reject_proposal_main();  // # reject proposal
                          } else {   
                                 result_input.accept_proposal_main(); // # accept proposal
                          }
                      
              }

      } catch (...) {
        
                // std::cout << "  Could not evaluate lp_grad function when sampling main parameters " << ")\n";
          
                result_input.main_div() = 1; // record as div 
                result_input.main_p_jump() = 0.0; // Set p_jump to zero
                result_input.reject_proposal_main(); // # reject proposal
             
              
      }
      


    }   
      
    }
    

  }
   


}








 
 
 
 
 
 
 
 
 
 
 
