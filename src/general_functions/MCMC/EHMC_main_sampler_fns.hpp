
#pragma once

 

// [[Rcpp::depends(StanHeaders)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppEigen)]]

 


#include <random>

  

#include <Eigen/Dense>
 
 
 

#include <unsupported/Eigen/SpecialFunctions>
 
 
 
 // [[Rcpp::plugins(cpp17)]]
 
 
 
using namespace Eigen;

 
 
#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
 
 

 
 
//  
// #include <Ziggurat.h>
//  
// static Ziggurat::Ziggurat::Ziggurat zigg;
//  
//  
 
 
 
 
// HMC sampler functions   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



inline void generate_random_std_norm_vec(Eigen::Matrix<double, -1, 1> &std_norm_vec,
                                       int n_params, 
                                       std::mt19937  &rng) {

  std::normal_distribution<double> dist(0.0, 1.0); 
  for (int d = 0; d < n_params; d++) {
    std_norm_vec(d) = dist(rng);
  }

}



// inline void generate_random_std_norm_vec_zigg(Eigen::Matrix<double, -1, 1> &std_norm_vec,
//                                                const int n_params, 
//                                                const int seed) {
//   
//   for (int d = 0; d < n_params; d++) {
//     std_norm_vec(d) = zigg.norm();
//   }
//   
// }
//  




// Another function using the same RNG
inline void generate_random_tau_ii(double &tau, 
                                   double &tau_ii, 
                                   std::mt19937  &rng) {

  std::uniform_real_distribution<double> dist(0.0, 2.0 * tau);
  tau_ii = dist(rng); // assign random value

}







inline bool check_divergence_Eigen(     const HMCResult &result_input,
                                        const Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                        const double &hamiltonian_energy,
                                        const double &previous_hamiltonian_energy) {
  
  /// first check is any NaN or Inf values 
  if ( (is_NaN_or_Inf_Eigen(lp_and_grad_outs) == true)  )  { 
 
        for (int i = 0; i < lp_and_grad_outs.size(); ++i) {
          if (std::isnan(lp_and_grad_outs(i)) || std::isinf(lp_and_grad_outs(i))) {
            std::cout  << "NaN or Inf detected at indices: "  << "\n"   ;
            std::cout  << i << " "  << "\n"    << std::endl; 
          }
        }
        
        return true;
    
  }
  
 

  
  // //  now check for large energy changes
  // {
  //     const double energy_diff = std::fabs(hamiltonian_energy - previous_hamiltonian_energy);
  //     const double energy_threshold = 100000000.0; // arbitrary
  //   
  //     if (energy_diff > energy_threshold) {
  //       return true;
  //     }
  // }

  return false; // If no issues detected
  
} 



 




// Define the function that performs the leapfrog integration
inline void leapfrog_integrator_dense_M_standard_HMC_main_InPlace(    Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
                                                                      Eigen::Matrix<double, -1, 1> &theta_main_vec_proposed_ref,
                                                                      Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                                                      const Eigen::Matrix<double, -1, 1> &theta_us_vec_initial_ref,
                                                                      const Eigen::Matrix<double, -1, -1> &M_inv_dense_main,
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
        
        // Update velocity (first half step)
        velocity_main_vec_proposed_ref.array() +=  ( 0.5 * eps * M_inv_dense_main *  grad_main ).array() ; 
        
        //// updae params by full step
        theta_main_vec_proposed_ref.array()  +=  eps *     velocity_main_vec_proposed_ref.array() ;
        
        // Update lp and gradients
        fn_lp_grad_InPlace(lp_and_grad_outs, 
                           Model_type, force_autodiff, force_PartialLog, multi_attempts,
                           theta_main_vec_proposed_ref, theta_us_vec_initial_ref, y_ref, grad_option, 
                           Model_args_as_cpp_struct, Stan_model_as_cpp_struct);
        grad_main =  ( lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array()).matrix();
        
        // Update velocity (second half step)
        velocity_main_vec_proposed_ref.array() +=  ( 0.5 * eps * M_inv_dense_main *  grad_main ).array() ;   
    
  } // End of leapfrog steps 
  
}








// Define the function that performs the leapfrog integration
inline void leapfrog_integrator_diag_M_standard_HMC_main_InPlace(         Eigen::Matrix<double, -1, 1> &velocity_main_vec_proposed_ref,
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
                                                                          std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
                                                                                             const std::string,
                                                                                             const bool, const bool, const bool,
                                                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
                                                                                             const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
                                                                                             const std::string,
                                                                                             const Model_fn_args_struct&,
                                                                                             const Stan_model_struct& )> fn_lp_grad_InPlace
                                                                      
) {
  
  const int N = Model_args_as_cpp_struct.N;
  const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
  const int n_params_main = Model_args_as_cpp_struct.n_params_main;
  const int n_params = n_params_main + n_nuisance;
  
  Eigen::Matrix<double, -1, 1> grad_main =  ( lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array()).matrix();
  
  for (int l = 0; l < L_ii; l++) {
    
        // Update velocity (first half step)
        velocity_main_vec_proposed_ref.array() +=  ( 0.5 * eps * M_inv_main_vec.array() *  grad_main.array() ).array() ; 
        
        //// updae params by full step
        theta_main_vec_proposed_ref.array()  +=  eps *     velocity_main_vec_proposed_ref.array() ;
        
        // Update lp and gradients
        fn_lp_grad_InPlace(lp_and_grad_outs, 
                           Model_type, force_autodiff, force_PartialLog, multi_attempts,
                           theta_main_vec_proposed_ref, theta_us_vec_initial_ref, y_ref, grad_option, 
                           Model_args_as_cpp_struct, Stan_model_as_cpp_struct);
        grad_main =  ( lp_and_grad_outs.segment(1 + n_nuisance, n_params_main).array()).matrix();
        
        // Update velocity (second half step)
        velocity_main_vec_proposed_ref.array() +=  ( 0.5 * eps * M_inv_main_vec.array() *  grad_main.array() ).array() ;  
    
  } // End of leapfrog steps  
  
}











void                                        fn_standard_HMC_main_only_single_iter_InPlace_process(   HMCResult &result_input,
                                                                                                     const bool  burnin, 
                                                                                                     std::mt19937  &rng,
                                                                                                     const int seed,
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
  
    const std::string grad_option = "main_only";
     
    const std::string metric_shape_main = EHMC_Metric_struct_as_cpp_struct.metric_shape_main;
    
    double U_x_initial =  0.0 ; 
    double U_x_prop =  0.0 ; 
    double log_posterior_prop =  0.0 ; 
    double log_posterior_0  =   0.0 ;  
    double log_ratio = 0.0;
    double energy_old = 0.0;
    double energy_new = 0.0;
    
    result_input.main_theta_vec_0 =  result_input.main_theta_vec;
    result_input.us_theta_vec_0 =  result_input.us_theta_vec;

  {

      {
          Eigen::Matrix<double, -1, 1>  std_norm_vec_main(n_params_main);
          generate_random_std_norm_vec(std_norm_vec_main, n_params_main, rng);
          if (metric_shape_main == "dense") result_input.main_velocity_0_vec  = EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main_chol * std_norm_vec_main;
          if (metric_shape_main == "diag")  result_input.main_velocity_0_vec.array() = std_norm_vec_main.array() *  (EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec).array().sqrt() ; 
      }
      
   

    {
      
 
      { 
        
      try {
        
        result_input.main_velocity_vec_proposed  =   result_input.main_velocity_0_vec; // set initial velocity
        result_input.main_theta_vec_proposed =       result_input.main_theta_vec_0;   // set initial theta   

        // ---------------------------------------------------------------------------------------------------------------///    Perform L leapfrogs   ///-----------------------------------------------------------------------------------------------------------------------------------------
          generate_random_tau_ii(   EHMC_args_as_cpp_struct.tau_main,    EHMC_args_as_cpp_struct.tau_main_ii, rng);
          int    L_ii = std::ceil( EHMC_args_as_cpp_struct.tau_main_ii / EHMC_args_as_cpp_struct.eps_main );
          if (L_ii < 1) { L_ii = 1 ; }
          
          //// initial lp  
          fn_lp_grad_InPlace(     result_input.lp_and_grad_outs, 
                                  Model_type, 
                                  force_autodiff, force_PartialLog, multi_attempts,
                                  result_input.main_theta_vec,  result_input.us_theta_vec, y_ref, grad_option,
                                  Model_args_as_cpp_struct, Stan_model_as_cpp_struct);
          
          log_posterior_0 =  result_input.lp_and_grad_outs(0);
          U_x_initial = - log_posterior_0; //// initial energy
          
          if (metric_shape_main == "dense") {
 
              leapfrog_integrator_dense_M_standard_HMC_main_InPlace(    result_input.main_velocity_vec_proposed, 
                                                                        result_input.main_theta_vec_proposed, 
                                                                        result_input.lp_and_grad_outs, 
                                                                        result_input.us_theta_vec,
                                                                        EHMC_Metric_struct_as_cpp_struct.M_inv_dense_main,
                                                                        y_ref,
                                                                        L_ii, EHMC_args_as_cpp_struct.eps_main,
                                                                        Model_type, 
                                                                        force_autodiff, force_PartialLog, multi_attempts, 
                                                                        grad_option,
                                                                        Model_args_as_cpp_struct, Stan_model_as_cpp_struct, 
                                                                        fn_lp_grad_InPlace);
            
          } else if (metric_shape_main == "diag") {
            
                leapfrog_integrator_diag_M_standard_HMC_main_InPlace(     result_input.main_velocity_vec_proposed, 
                                                                          result_input.main_theta_vec_proposed, 
                                                                          result_input.lp_and_grad_outs, 
                                                                          result_input.us_theta_vec,
                                                                          EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec,
                                                                          y_ref,
                                                                          L_ii, EHMC_args_as_cpp_struct.eps_main,
                                                                          Model_type, 
                                                                          force_autodiff, force_PartialLog, multi_attempts, 
                                                                          grad_option,
                                                                          Model_args_as_cpp_struct, Stan_model_as_cpp_struct, 
                                                                          fn_lp_grad_InPlace);
          }
 
          
          // /// proposed lp  
          log_posterior_prop =  result_input.lp_and_grad_outs(0);
          U_x_prop = - log_posterior_prop; // initial energy
        
          //////////////////////////////////////////////////////////////////    M-H acceptance step  (i.e, Accept/Reject step)
          if (metric_shape_main == "dense") {
       
              Eigen::Matrix<double, 1, -1>  velocity_0_x_M_dense_main = result_input.main_velocity_0_vec.transpose() * EHMC_Metric_struct_as_cpp_struct.M_dense_main; // row-vec
              energy_old = U_x_initial  +   0.5 * ( velocity_0_x_M_dense_main * result_input.main_velocity_0_vec ).eval()(0, 0) ;
              
              Eigen::Matrix<double, 1, -1>  velocity_prop_x_M_dense_main =  result_input.main_velocity_vec_proposed.transpose() * EHMC_Metric_struct_as_cpp_struct.M_dense_main; // row-vec
              energy_new  =  U_x_prop +  0.5 * ( velocity_prop_x_M_dense_main *  result_input.main_velocity_vec_proposed  ).eval()(0, 0) ;
    
              log_ratio = - energy_new + energy_old;
              
          } else if (metric_shape_main == "diag") {
            
              energy_old  = U_x_initial +  0.5 * (   result_input.main_velocity_0_vec .array().square()          * ( 1.0 / EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec.array()    ).array() ).matrix().sum() ; 
              energy_new  = U_x_prop +   0.5 * (     result_input.main_velocity_vec_proposed.array().square()    * ( 1.0 / EHMC_Metric_struct_as_cpp_struct.M_inv_main_vec.array()    ).array() ).matrix().sum() ;
              log_ratio = - energy_new + energy_old;
            
          }
 
        if  (check_divergence_Eigen(result_input,   result_input.lp_and_grad_outs, energy_old, energy_new) == true)      {     /// if main_div, reject proposal 
            
                result_input.main_p_jump = 0.0;
                result_input.main_theta_vec  =   result_input.main_theta_vec_0;    
                result_input.main_velocity_vec = result_input.main_velocity_0_vec;
                result_input.main_div = 1;  // and set main_div indiator to 1
              //  return result_input;
          
        }  else {  // if no main_div, carry on with MH step 
          
                    result_input.main_div = 0;
                    result_input.main_p_jump = std::min(1.0, stan::math::exp(log_ratio));
                
                if  ( (R::runif(0, 1) > result_input.main_p_jump) ) {  // # reject proposal
                      result_input.main_theta_vec  =   result_input.main_theta_vec_0;  
                      result_input.main_velocity_vec = result_input.main_velocity_0_vec;
                } else {   // # accept proposal
                      result_input.main_theta_vec  =    result_input.main_theta_vec_proposed ; 
                      result_input.main_velocity_vec =  result_input.main_velocity_vec_proposed ;
                }
            
               // return result_input;
                
        }

      } catch (...) {
        
              std::cout << "  Could not evaluate lp_grad function when sampling main parameters " << ")\n";
        
              result_input.main_div = 1;
              result_input.main_p_jump = 0.0;
              result_input.main_theta_vec  =   result_input.main_theta_vec_0; 
              result_input.main_velocity_vec = result_input.main_velocity_0_vec;
             // return result_input;
              
      }
      


    }   
      
    }
    

  }
  
 // return result_input;


}












// 
// 
// 
// 
// void fn_standard_HMC_main_only_single_iter_multi_attempts_InPlace_process(      HMCResult &result_input,
//                                                                                 const bool &burnin, 
//                                                                                 std::mt19937 &rng,
//                                                                                 const int seed,
//                                                                                 const std::string &Model_type,
//                                                                                 const bool &force_autodiff,
//                                                                                 const bool &force_PartialLog,
//                                                                                 const Eigen::Matrix<int, -1, -1> &y_ref,
//                                                                                 const Model_fn_args_struct &Model_args_as_cpp_struct,
//                                                                                 EHMC_fn_args_struct &EHMC_args_as_cpp_struct, 
//                                                                                 const EHMC_Metric_struct &EHMC_Metric_struct_as_cpp_struct,
//                                                                                 const Stan_model_struct &Stan_model_as_cpp_struct
// ) {
//   
//  
//  
//            result_input.main_div = 0;  // init main_div
//            HMCResult result_input_orig = result_input;
//   
//             bool use_autodiff = false;
//             bool use_PartialLog = false;
//             
//             ///// 1st attempt
//             if  ( (force_autodiff == false) && (force_PartialLog == false)  )  {   // NOT log-scale and NOT autodiff (least numerically stable but fastest)
//               result_input.main_div = 0;  // Reset main_div indicator
//               fn_standard_HMC_main_only_single_iter_InPlace_process(result_input,  
//                                                                                    burnin, rng, seed,
//                                                                                    Model_type, use_autodiff, use_PartialLog, y_ref,
//                                                                                    Model_args_as_cpp_struct, EHMC_args_as_cpp_struct, EHMC_Metric_struct_as_cpp_struct,
//                                                                                    Stan_model_as_cpp_struct);
//             }
//             
//             ///// 2nd attempt (if first fails) - uses log-scale but NOT autodiff 
//             if ( (result_input.main_div == 1) || ( (force_autodiff == false) && (force_PartialLog == true)   ) ) {
//               result_input = result_input_orig;
//               result_input.main_div = 0;  // Reset main_div indicator
//               use_PartialLog = true;
//               fn_standard_HMC_main_only_single_iter_InPlace_process(   result_input,  
//                                                                                       burnin, rng, seed, 
//                                                                                       Model_type, use_autodiff, use_PartialLog, y_ref,
//                                                                                       Model_args_as_cpp_struct, EHMC_args_as_cpp_struct, EHMC_Metric_struct_as_cpp_struct,
//                                                                                       Stan_model_as_cpp_struct);
//               // if  (result_input.main_div == 0) { 
//               //   return result_input;
//               // }
//             }
//             
//             ///// 3rd attempt (if second fails)
//             if ( (result_input.main_div == 1) ||  ( (force_autodiff == true) && (force_PartialLog == true)  )  )  {
//               result_input.main_div = 0;  // Reset main_div indicator
//               result_input = result_input_orig;
//               use_autodiff = true;
//               use_PartialLog = true;
//               fn_standard_HMC_main_only_single_iter_InPlace_process( result_input,  
//                                                                                     burnin, rng, seed, 
//                                                                                     Model_type, use_autodiff, use_PartialLog, y_ref,
//                                                                                     Model_args_as_cpp_struct, EHMC_args_as_cpp_struct, EHMC_Metric_struct_as_cpp_struct,
//                                                                                     Stan_model_as_cpp_struct);
//               // if  (result_input.main_div == 0) { 
//               //   return result_input;
//               // } 
//             }
//             
//            // return result_input;
//  
// }
// 
// 
//  
//  
//  
//  
//  
//  
 
 
 
 
 
 
 
 