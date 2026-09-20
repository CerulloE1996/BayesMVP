
#pragma once
 

#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/SpecialFunctions>
 
 
 
 
using namespace Eigen;

 
 
 
 
ALWAYS_INLINE  void leapfrog_integrator_dense_M_standard_HMC_dual_InPlace(    Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
                                                                                               Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
                                                                                     Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
                                                                                     Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
                                                                                     Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                                                                     const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
                                                                                     const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
                                                                                     const Eigen::Matrix<double, -1, -1> &M_inv_dense_main,
                                                                                     const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
                                                                                     const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                                     const int L_ii, // Number of leapfrog steps
                                                                                     const double eps,
                                                                                     const std::string &Model_type,
                                                                                     const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
                                                                                     const std::string &grad_option,
                                                                                     const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                     const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                                     std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                                                                                                        const std::string,
                                                                                                        const bool, const bool, const bool,
                                                                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                        const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                                                                                                        const std::string,
                                                                                                        const Model_fn_args_struct &,
                                                                                                        const Stan_model_struct &)> fn_lp_grad_InPlace
                                                                                                 
 ) {
   
               const int N = Model_args_as_cpp_struct.N;
               const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
               const int n_params_main = Model_args_as_cpp_struct.n_params_main;
               const int n_params = n_params_main + n_nuisance;
               
               Eigen::Matrix<double, -1, 1> grad_main =  ( lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array()).matrix();
               
               for (int l = 0; l < L_ii; l++) {
                 
                         // Update velocity (first half step) - main params
                         velocity_main_vec_proposed_ref.array() +=  ( 0.5 * eps * M_inv_dense_main *  grad_main ).array() ; 
                         // Update velocity (first half step) - nuisance params
                         velocity_us_vec_proposed_ref.array() += (0.5 * eps * lp_and_grad_outs.segment(1, n_nuisance).array()  * M_inv_us_vec.array());
                         
                         //// update params by full step - main params
                         theta_main_vec_proposed_ref.array()  +=  eps *     velocity_main_vec_proposed_ref.array() ;
                         //// update params by full step - nuisance params
                         theta_us_vec_proposed_ref.array()  +=  eps *     velocity_us_vec_proposed_ref.array() ;
                         
                         // Update lp and gradients
                         fn_lp_grad_InPlace(lp_and_grad_outs, 
                                            Model_type, force_autodiff, force_PartialLog, multi_attempts,
                                            theta_main_vec_proposed_ref, theta_us_vec_proposed_ref, y_ref, grad_option, 
                                            Model_args_as_cpp_struct, //MVP_workspace,
                                            Stan_model_as_cpp_struct);
                         grad_main =  ( lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array()).matrix();
                         
                         // Update velocity (second half step)
                         velocity_main_vec_proposed_ref.array() +=  ( 0.5 * eps * M_inv_dense_main *  grad_main ).array() ;   
                         // Update velocity (first half step) - nuisance params
                         velocity_us_vec_proposed_ref.array() += (0.5 * eps * lp_and_grad_outs.segment(1, n_nuisance).array()  * M_inv_us_vec.array());
                 
               } // End of leapfrog steps 
                                       
 }
 
 
 
 
 
 
 
 
 
 ALWAYS_INLINE  void leapfrog_integrator_diag_M_standard_HMC_dual_InPlace(     Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
                                                                               Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
                                                                               Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
                                                                               Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
                                                                               Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                                                               const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
                                                                               const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
                                                                               const Eigen::Matrix<double, -1, 1> &M_inv_main_vec,
                                                                               const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
                                                                               const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                               const int L_ii, // Number of leapfrog steps
                                                                               const double eps,
                                                                               const std::string &Model_type,
                                                                               const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
                                                                               const std::string &grad_option,
                                                                               const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                               const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                               std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                                                                                                  const std::string,
                                                                                                  const bool, const bool, const bool,
                                                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                                  const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                                                                                                  const std::string,
                                                                                                  const Model_fn_args_struct &,
                                                                                                  const Stan_model_struct &)> fn_lp_grad_InPlace
                                                                                 
 ) {
   
   const int N = Model_args_as_cpp_struct.N;
   const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
   const int n_params_main = Model_args_as_cpp_struct.n_params_main;
   const int n_params = n_params_main + n_nuisance;
   
   Eigen::Matrix<double, -1, 1> grad_main =  ( lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array()).matrix();
   
   for (int l = 0; l < L_ii; l++) {
     
     // Update velocity (first half step) - main params
     velocity_main_vec_proposed_ref.array() +=  ( 0.5 * eps * M_inv_main_vec.array() *  grad_main.array() ).array() ;
     // Update velocity (first half step) - nuisance params
     velocity_us_vec_proposed_ref.array() += (0.5 * eps * lp_and_grad_outs.segment(1, n_nuisance).array()  * M_inv_us_vec.array());
     
     //// update params by full step - main params 
     theta_main_vec_proposed_ref.array()  +=  eps *     velocity_main_vec_proposed_ref.array() ;
     //// update params by full step - nuisance params
     theta_us_vec_proposed_ref.array()  +=  eps *     velocity_us_vec_proposed_ref.array() ;
     
     // Update lp and gradients
     fn_lp_grad_InPlace(lp_and_grad_outs, 
                        Model_type, force_autodiff, force_PartialLog, multi_attempts,
                        theta_main_vec_proposed_ref, theta_us_vec_proposed_ref, y_ref, grad_option, 
                        Model_args_as_cpp_struct, //MVP_workspace,
                        Stan_model_as_cpp_struct);
     grad_main =  ( lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array()).matrix();
     
     // Update velocity (second half step)
     velocity_main_vec_proposed_ref.array() +=  ( 0.5 * eps * M_inv_main_vec.array() *  grad_main.array() ).array() ; 
     // Update velocity (first half step) - nuisance params
     velocity_us_vec_proposed_ref.array() += (0.5 * eps * lp_and_grad_outs.segment(1, n_nuisance).array()  * M_inv_us_vec.array());
     
   } // End of leapfrog steps 
   
 }


 
 
 
 
 
 
 
 
 
 
 
 
template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE  void                         fn_standard_HMC_dual_single_iter_InPlace_process(    HMCResult &result_input,
                                                                                                 T &rng_main,
                                                                                                 T &rng_nuisance,
                                                                                                 const std::string &Model_type,
                                                                                                 const bool  force_autodiff,
                                                                                                 const bool  force_PartialLog,
                                                                                                 const bool  multi_attempts,
                                                                                                 const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                                                 const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                                 EHMC_fn_args_struct  &EHMC_args_as_cpp_struct, /// pass by ref. to modify (???)
                                                                                                 const EHMC_Metric_struct   &EHMC_Metric_struct_as_cpp_struct,
                                                                                                 const Stan_model_struct &Stan_model_as_cpp_struct
) {


    //// important params
    const int N = Model_args_as_cpp_struct.N;
    const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
    const int n_params_main = Model_args_as_cpp_struct.n_params_main;
    const int n_params = n_params_main + n_nuisance;
  
    const std::string grad_option = "all";
     
    const std::string metric_shape_main = EHMC_Metric_struct_as_cpp_struct.metric_shape_main;
    
    double U_x_initial =  0.0 ; 
    double U_x_prop =  0.0 ; 
    double log_posterior_prop =  0.0 ; 
    double log_posterior_0  =   0.0 ;   
    double log_ratio = 0.0;
    double energy_old = 0.0;
    double energy_new = 0.0;
    
    //// result_input.store_current_state(); // sets initial theta and velocity to current theta and velocity
    result_input.main_theta_vec_0() =  result_input.main_theta_vec();
    result_input.us_theta_vec_0() =    result_input.us_theta_vec();

  {

      { /// draw velocity for main:
          Eigen::Matrix<double, -1, 1> std_norm_vec_main = Eigen::Matrix<double, -1, 1>::Zero(n_params_main);
          generate_random_std_norm_vec_InPlace(std_norm_vec_main, rng_main);
          if (metric_shape_main == "dense") result_input.main_velocity_0_vec()  = EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main_chol * std_norm_vec_main;
          if (metric_shape_main == "diag")  result_input.main_velocity_0_vec().array() = std_norm_vec_main.array() *  (EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec).array().sqrt() ; 
      }
      { /// draw velocity for nuisance:
          Eigen::Matrix<double, -1, 1> std_norm_vec_us = Eigen::Matrix<double, -1, 1>::Zero(n_nuisance);
          generate_random_std_norm_vec_InPlace(std_norm_vec_us, rng_nuisance);
          result_input.us_velocity_0_vec().array() = std_norm_vec_us.array() *  (EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec).array().sqrt();
      }

    {
 
      { 
        
      try {
        
        result_input.main_velocity_vec_proposed()  =   result_input.main_velocity_0_vec(); // set to initial velocity
        result_input.main_theta_vec_proposed() =       result_input.main_theta_vec_0();   // set to initial theta   
        result_input.us_velocity_vec_proposed()  =   result_input.us_velocity_0_vec(); // set to initial velocity
        result_input.us_theta_vec_proposed() =       result_input.us_theta_vec_0();   // set to initial theta 

          //// ---------------------------------------------------------------------------------------------------------------///    Perform L leapfrogs   ///-----------------------------------------
          if (EHMC_args_as_cpp_struct.tau_main < EHMC_args_as_cpp_struct.eps_main) { 
            EHMC_args_as_cpp_struct.tau_main = EHMC_args_as_cpp_struct.eps_main; 
          }
          //
          EHMC_args_as_cpp_struct.tau_main_ii = generate_random_tau_ii(EHMC_args_as_cpp_struct.tau_main, rng_main);
          if (EHMC_args_as_cpp_struct.tau_main_ii < EHMC_args_as_cpp_struct.eps_main) { 
            EHMC_args_as_cpp_struct.tau_main_ii = EHMC_args_as_cpp_struct.eps_main; 
          }
          //
          int    L_ii = std::ceil( EHMC_args_as_cpp_struct.tau_main_ii / EHMC_args_as_cpp_struct.eps_main );
          if (L_ii < 1) { L_ii = 1 ; }
          
          
          //// initial lp  
          fn_lp_grad_InPlace(     result_input.lp_and_grad_outs(), 
                                  Model_type, 
                                  force_autodiff, force_PartialLog, multi_attempts,
                                  result_input.main_theta_vec(),  result_input.us_theta_vec(),
                                  y_ref, 
                                  grad_option,
                                  Model_args_as_cpp_struct, //MVP_workspace, 
                                  Stan_model_as_cpp_struct);
          
          log_posterior_0 =  result_input.lp_and_grad_outs()(0);
          U_x_initial = - log_posterior_0; //// initial energy
          
          /////////  do leapfrogs for ALL parameters
          if (metric_shape_main == "dense") {
                                                                                                 
                 leapfrog_integrator_dense_M_standard_HMC_dual_InPlace(     result_input.main_velocity_vec_proposed(), 
                                                                            result_input.us_velocity_vec_proposed(), 
                                                                            result_input.main_theta_vec_proposed(), 
                                                                            result_input.us_theta_vec_proposed(), 
                                                                            result_input.lp_and_grad_outs(), 
                                                                            result_input.main_theta_vec(),
                                                                            result_input.us_theta_vec(),
                                                                            EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main,
                                                                            EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec,
                                                                            y_ref,
                                                                            L_ii, EHMC_args_as_cpp_struct.eps_main,
                                                                            Model_type, 
                                                                            force_autodiff, force_PartialLog, multi_attempts, 
                                                                            grad_option,
                                                                            Model_args_as_cpp_struct, 
                                                                            Stan_model_as_cpp_struct, 
                                                                            fn_lp_grad_InPlace);
            
          } else if (metric_shape_main == "diag") {
            
                leapfrog_integrator_diag_M_standard_HMC_dual_InPlace(      result_input.main_velocity_vec_proposed(), 
                                                                           result_input.us_velocity_vec_proposed(), 
                                                                           result_input.main_theta_vec_proposed(), 
                                                                           result_input.us_theta_vec_proposed(), 
                                                                           result_input.lp_and_grad_outs(), 
                                                                           result_input.main_theta_vec(),
                                                                           result_input.us_theta_vec(),
                                                                           EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec,
                                                                           EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec,
                                                                           y_ref,
                                                                           L_ii, EHMC_args_as_cpp_struct.eps_main,
                                                                           Model_type, 
                                                                           force_autodiff, force_PartialLog, multi_attempts, 
                                                                           grad_option,
                                                                           Model_args_as_cpp_struct, 
                                                                           Stan_model_as_cpp_struct, 
                                                                           fn_lp_grad_InPlace);
            
          }
 
          
          // /// proposed lp  
          log_posterior_prop =  result_input.lp_and_grad_outs()(0);
          U_x_prop = - log_posterior_prop; // initial energy
        
          //////////////////////////////////////////////////////////////////    M-H acceptance step  (i.e, Accept/Reject step)
          if (metric_shape_main == "dense") {
       
                  const Eigen::Matrix<double, 1, -1>  velocity_0_x_M_dense_main = result_input.main_velocity_0_vec().transpose() * EHMC_Metric_struct_as_cpp_struct.M_dense_main; // row-vec
                  energy_old = U_x_initial ;
                  energy_old  +=  0.5 * (velocity_0_x_M_dense_main * result_input.main_velocity_0_vec() ).eval()(0, 0) ; // for main
                  //// for nuisance:
                  energy_old  +=  0.5 * (  result_input.us_velocity_0_vec().array()  *  result_input.us_velocity_0_vec().array() * ( EHMC_Metric_struct_as_cpp_struct.M_us_vec ).array() ).sum() ;  
                
                  const Eigen::Matrix<double, 1, -1>  velocity_prop_x_M_dense_main =  result_input.main_velocity_vec_proposed().transpose() * EHMC_Metric_struct_as_cpp_struct.M_dense_main; // row-vec
                  energy_new  =  U_x_prop;  
                  energy_new  +=  0.5 * (velocity_prop_x_M_dense_main *  result_input.main_velocity_vec_proposed()).eval()(0, 0) ; // for main
                  //// for nuisance:
                  energy_new  +=  0.5 * (result_input.us_velocity_vec_proposed().array() * result_input.us_velocity_vec_proposed().array() * (EHMC_Metric_struct_as_cpp_struct.M_us_vec ).array() ).sum();  
        
                  log_ratio = - energy_new + energy_old;
              
          } else if (metric_shape_main == "diag") {
            
                  energy_old = U_x_initial ;
                  //// for main:
                  energy_old  +=  0.5 * (result_input.main_velocity_0_vec().array() * result_input.main_velocity_0_vec().array() * (stan::math::inv(EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec).array()).array()).sum(); 
                  //// for nuisance:
                  energy_old  +=  0.5 * (result_input.us_velocity_0_vec().array() * result_input.us_velocity_0_vec().array() * (EHMC_Metric_struct_as_cpp_struct.M_us_vec).array()).sum() ; 
                  
                  energy_new  =  U_x_prop;  
                  //// for main:
                  energy_new  +=  0.5 * (result_input.main_velocity_vec_proposed().array() * result_input.main_velocity_vec_proposed().array() * (stan::math::inv(EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec).array()).array()).sum(); 
                  //// for nuisance:
                  energy_new  +=  0.5 * (result_input.us_velocity_vec_proposed().array() * result_input.us_velocity_vec_proposed().array() * (EHMC_Metric_struct_as_cpp_struct.M_us_vec).array() ).sum(); 
                  
                  log_ratio = - energy_new + energy_old; 
            
          }
 
        if  (check_divergence(log_ratio, result_input) == true)    { /// if main_div, reject proposal 
            
                    //// if  div, reject proposal 
                    result_input.main_div() = 1;  
                    result_input.main_p_jump() = 0.0;
                    result_input.us_div() = 1;
                    result_input.us_p_jump() = 0.0;
                    result_input.reject_proposal_main();  // # reject proposal
                    result_input.reject_proposal_us();  // # reject proposal
        
          
        }  else {  // if no main_div, carry on with MH step 
          
                    result_input.main_div() = 0;
                    result_input.main_p_jump() = std::min(1.0, stan::math::exp(log_ratio));
                    result_input.us_div()   = 0;
                    result_input.us_p_jump() =   std::min(1.0, stan::math::exp(log_ratio));
                    
                    const double rand_unif = generate_random_std_uniform(rng_main);
                    const double log_rand_unif = stan::math::log(rand_unif);

                    if  (log_rand_unif > log_ratio)   {  // # reject proposal
                          result_input.reject_proposal_main();  // # reject proposal
                          result_input.reject_proposal_us();  // # reject proposal
                    } else {   // # accept proposal
                          result_input.accept_proposal_main();  // # accept proposal
                          result_input.accept_proposal_us();  // # accept proposal
                    }
 
                
        }

      } catch (...) {
        
              // std::cout << "  Could not evaluate lp_grad function when sampling main + nuisance parameters " << ")\n";
        
                result_input.us_div() = 1; // record as div 
                result_input.us_p_jump() = 0.0; // set p_jump to zero
                result_input.reject_proposal_us(); // # reject proposal
                
                result_input.main_div() = 1; // record as div 
                result_input.main_p_jump() = 0.0; // set p_jump to zero
                result_input.reject_proposal_main(); // # reject proposal
  
              
      }
      


    }   
      
    }
    

  }
 


}










 
 
 
 