
#pragma once

 
 
#include <random>

 

#include <Eigen/Dense>
 
 

#include <unsupported/Eigen/SpecialFunctions>
 
 
 
using namespace Eigen;
 
  
  
 
 
 
// Diffusion-HMC sampler functions   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------






ALWAYS_INLINE  void leapfrog_integrator_diag_M_standard_HMC_nuisance_InPlace(   Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
                                                                                Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
                                                                                Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                                                                const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
                                                                                const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
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

      const int n_nuisance = velocity_us_vec_proposed_ref.size();
      // #ifdef _WIN32
        Eigen::Matrix<double, -1, 1> grad_us =  lp_and_grad_outs.segment(1, n_nuisance);
      // #endif

      for (int l = 0; l < L_ii; l++) {
            if (Model_args_as_cpp_struct.burnin_leapfrog_steps != nullptr) {
                ++(*Model_args_as_cpp_struct.burnin_leapfrog_steps);
            }
      
                // Update velocity (first half step)
                // #ifdef _WIN32
                           velocity_us_vec_proposed_ref.array() += 0.5 * eps * grad_us.array()  * M_inv_us_vec.array();
                // #else
                //            velocity_us_vec_proposed_ref.array() += 0.5 * eps * lp_and_grad_outs.segment(1, n_nuisance).array()  * M_inv_us_vec.array();
                // #endif
  
                //// updae params by full step
                theta_us_vec_proposed_ref.array()  +=  eps *     velocity_us_vec_proposed_ref.array() ;
  
                // Update lp and gradients
                fn_lp_grad_InPlace(lp_and_grad_outs, 
                                   Model_type, force_autodiff, force_PartialLog, multi_attempts,
                                   theta_main_vec_initial_ref,
                                   theta_us_vec_proposed_ref,
                                   y_ref,
                                   grad_option,
                                   Model_args_as_cpp_struct,
                                   Stan_model_as_cpp_struct, 
                                   LC_MVP_ws_structs,
                                   n_threads_WCP);
                // #ifdef _WIN32
                           grad_us =  lp_and_grad_outs.segment(1, n_nuisance);
                // #endif
  
                // #ifdef _WIN32
                           velocity_us_vec_proposed_ref.array() += 0.5 * eps * grad_us.array()  * M_inv_us_vec.array();
                // #else
                //            velocity_us_vec_proposed_ref.array() += 0.5 * eps * lp_and_grad_outs.segment(1, n_nuisance).array()  * M_inv_us_vec.array();
                // #endif

      } // End of leapfrog steps

}






















  
// ALWAYS_INLINE  void leapfrog_integrator_diag_M_diffusion_HMC_nuisance_InPlace(  Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
//                                                                         Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
//                                                                         Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
//                                                                         Eigen::Matrix<double, -1, 1> &us_theta_vec_current_segment,
//                                                                         Eigen::Matrix<double, -1, 1> &us_velocity_vec_current_segment,
//                                                                         const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
//                                                                         const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
//                                                                         const Eigen::Matrix<int, -1, -1> &y_ref,
//                                                                         const int L_ii,
//                                                                         const double eps_1, const double eps_2, const double cos_eps_2, const double sin_eps_2,
//                                                                         const std::string &Model_type,
//                                                                         const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
//                                                                         const std::string &grad_option,
//                                                                         const Model_fn_args_struct &Model_args_as_cpp_struct,
//                                                                         const Stan_model_struct &Stan_model_as_cpp_struct,
//                                                                         std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
//                                                                                            const std::string &,
//                                                                                            const bool, const bool, const bool,
//                                                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
//                                                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
//                                                                                            const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
//                                                                                            const std::string &,
//                                                                                            const Model_fn_args_struct &,
//                                                                                            const Stan_model_struct &)> fn_lp_grad_InPlace
// 
// ) {
// 
//   const int n_nuisance = velocity_us_vec_proposed_ref.size();
//   // #ifdef _WIN32
//     Eigen::Matrix<double, -1, 1> grad_us =  lp_and_grad_outs.segment(1, n_nuisance);
//   // #endif
//   
//   // Precompute per-component rotation angles
//   Eigen::ArrayXd angle = (1.0 / M_inv_us_vec.array()).array() * eps_2;
//   Eigen::ArrayXd cos_t = angle.cos();
//   Eigen::ArrayXd sin_t = angle.sin();
// 
//   for (int l = 0; l < L_ii; l++) {
// 
//             // Update velocity (first half step)
//             // velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * M_inv_us_vec.array() * grad_us.array()
//             //                                       + 0.5 * eps_1 * theta_us_vec_proposed_ref.array();
//             velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * grad_us.array() * M_inv_us_vec.array();
//     
//             // #ifdef _WIN32
//                          // velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * grad_us.array()  * M_inv_us_vec.array();
//                          // Eigen::ArrayXd grad_corrected = grad_us.array() + (1.0 - ((1.0/M_inv_us_vec.array()).array())) * theta_us_vec_proposed_ref.array();
//                          // velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * M_inv_us_vec.array() * grad_corrected;
//             // #else
//             //           velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * lp_and_grad_outs.segment(1, n_nuisance).array()  * M_inv_us_vec.array();
//             // #endif
//   
//             // Full step for position + update velocity again (only for Gaussian)
//             us_theta_vec_current_segment.array() = theta_us_vec_proposed_ref.array();
//             us_velocity_vec_current_segment.array() = velocity_us_vec_proposed_ref.array();
//             // theta_us_vec_proposed_ref.array() =    (cos_eps_2 * us_theta_vec_current_segment.array()     + sin_eps_2 * us_velocity_vec_current_segment.array());
//             // velocity_us_vec_proposed_ref.array() = (cos_eps_2 * us_velocity_vec_current_segment.array()  - sin_eps_2 * us_theta_vec_current_segment.array());
//             //
//             // Eigen::Matrix<double, -1, 1> angle = ((1.0/M_inv_us_vec.array()).array() * eps_2).matrix();
//             // Eigen::Matrix<double, -1, 1> cos_t = angle.array().cos().matrix();
//             // Eigen::Matrix<double, -1, 1> sin_t = angle.array().sin().matrix();
//             // theta_us_vec_proposed_ref.array()    = cos_t.array() * us_theta_vec_current_segment.array() + sin_t.array() * us_velocity_vec_current_segment.array();
//             // velocity_us_vec_proposed_ref.array() = cos_t.array() * us_velocity_vec_current_segment.array() - sin_t.array() * us_theta_vec_current_segment.array();
// 
//             theta_us_vec_proposed_ref.array() = cos_t * us_theta_vec_current_segment.array() 
//                                               + sin_t * us_velocity_vec_current_segment.array();
//             velocity_us_vec_proposed_ref.array() = cos_t * us_velocity_vec_current_segment.array() 
//                                                  - sin_t * us_theta_vec_current_segment.array();
//             
//             // Update lp and gradients
//             fn_lp_grad_InPlace(lp_and_grad_outs, Model_type, force_autodiff, force_PartialLog, multi_attempts,
//                                theta_main_vec_initial_ref, theta_us_vec_proposed_ref, y_ref, grad_option,
//                                Model_args_as_cpp_struct,
//                                Stan_model_as_cpp_struct);
//             // #ifdef _WIN32
//                       grad_us =  lp_and_grad_outs.segment(1, n_nuisance);
//             // #endif
//             
//             // Update velocity (second half step)
//             // velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * M_inv_us_vec.array() * grad_us.array()
//             //                                       + 0.5 * eps_1 * theta_us_vec_proposed_ref.array();
//             velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * grad_us.array() * M_inv_us_vec.array();
//             // #ifdef _WIN32
//                          // velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * grad_us.array()  * M_inv_us_vec.array();
//                          // grad_corrected = grad_us.array() + (1.0 - ((1.0/M_inv_us_vec.array()).array())) * theta_us_vec_proposed_ref.array();
//                          // velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * M_inv_us_vec.array() * grad_corrected;
//             // #else
//             //           velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * lp_and_grad_outs.segment(1, n_nuisance).array() * M_inv_us_vec.array();
//             // #endif
// 
//   } // End of leapfrog steps
// 
// }




// ALWAYS_INLINE void leapfrog_integrator_diag_M_diffusion_HMC_nuisance_InPlace(
//     Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
//     Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
//     Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
//     Eigen::Matrix<double, -1, 1> &us_theta_vec_current_segment,
//     Eigen::Matrix<double, -1, 1> &us_velocity_vec_current_segment,
//     const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
//     const Eigen::Matrix<double, -1, 1> &M_us_vec,
//     const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
//     const Eigen::Matrix<int, -1, -1> &y_ref,
//     const int L_ii,
//     const double eps_1,
//     const double eps_2,
//     const std::string &Model_type,
//     const bool force_autodiff, const bool force_PartialLog, const bool multi_attempts,
//     const std::string &grad_option,
//     const Model_fn_args_struct &Model_args_as_cpp_struct,
//     const Stan_model_struct &Stan_model_as_cpp_struct,
//     std::function<void(Eigen::Ref<Eigen::Matrix<double, -1, 1>>,
//                        const std::string &,
//                        const bool, const bool, const bool,
//                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
//                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>,
//                        const Eigen::Ref<const Eigen::Matrix<int, -1, -1>>,
//                        const std::string &,
//                        const Model_fn_args_struct &,
//                        const Stan_model_struct &)> fn_lp_grad_InPlace
// ) {
//               
//             const int n_nuisance = velocity_us_vec_proposed_ref.size();
//             Eigen::Matrix<double, -1, 1> grad_us = lp_and_grad_outs.segment(1, n_nuisance);
//             
//             // ---- FIX 1: Rotation frequencies and angles ----
//             // ω_i = √(A_i / M_i) = 1/√M_i  (since A = I)
//             // angle_i = ω_i * eps_2 = eps_2 / √M_i
//             //
//             // OLD (WRONG): angle = M_us_vec.array() * eps_2
//             Eigen::ArrayXd sqrt_M     = M_us_vec.array().sqrt();
//             Eigen::ArrayXd inv_sqrt_M = M_inv_us_vec.array().sqrt();  // = 1/√M
//             Eigen::ArrayXd omega      = inv_sqrt_M;                    // ω_i = 1/√M_i
//             Eigen::ArrayXd angle      = omega * eps_2;                 // ω_i * eps_2
//             Eigen::ArrayXd cos_t      = angle.cos();
//             Eigen::ArrayXd sin_t      = angle.sin();
//             
//             for (int l = 0; l < L_ii; l++) {
//                       
//                       // ---- FIX 3: Velocity half-kick ----
//                       // Kick: p += (ε/2) · (-∇Φ) = (ε/2) · (grad_post + θ)
//                       // In velocity: v += (ε/2) · M⁻¹ · (grad_post + θ)
//                       //
//                       // OLD (WRONG): v += 0.5 * eps_1 * M_inv * grad_us
//                       velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * M_inv_us_vec.array() * (grad_us.array() + theta_us_vec_proposed_ref.array());
//                       
//                       // ---- FIX 2: Exact rotation for H₀ ----
//                       // In (θ,v) space with diagonal M and A=I:
//                       //   θ_new = cos(ωt)·θ + (1/ω)·sin(ωt)·v = cos(ωt)·θ + √M·sin(ωt)·v
//                       //   v_new = cos(ωt)·v −    ω ·sin(ωt)·θ = cos(ωt)·v − (1/√M)·sin(ωt)·θ
//                       //
//                       // OLD (WRONG): simple rotation θ_new = cos·θ + sin·v, v_new = cos·v - sin·θ
//                       us_theta_vec_current_segment    = theta_us_vec_proposed_ref;
//                       us_velocity_vec_current_segment = velocity_us_vec_proposed_ref;
//                       
//                       theta_us_vec_proposed_ref.array()    = cos_t * us_theta_vec_current_segment.array()    + sqrt_M * sin_t * us_velocity_vec_current_segment.array();
//                       velocity_us_vec_proposed_ref.array() = cos_t * us_velocity_vec_current_segment.array() - inv_sqrt_M * sin_t * us_theta_vec_current_segment.array();
//                       
//                       // Recompute gradient at new θ
//                       fn_lp_grad_InPlace(lp_and_grad_outs, Model_type,
//                                          force_autodiff, force_PartialLog, multi_attempts,
//                                          theta_main_vec_initial_ref, theta_us_vec_proposed_ref,
//                                          y_ref, grad_option,
//                                          Model_args_as_cpp_struct, Stan_model_as_cpp_struct);
//                       grad_us = lp_and_grad_outs.segment(1, n_nuisance);
//                       
//                       // ---- FIX 3 (again): Velocity half-kick with NEW θ and gradient ----
//                       velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * M_inv_us_vec.array() * (grad_us.array() + theta_us_vec_proposed_ref.array());
//   }
//             
// }



ALWAYS_INLINE void leapfrog_integrator_diag_M_diffusion_HMC_nuisance_InPlace( Eigen::Matrix<double, -1, 1> &velocity_us_vec_proposed_ref,
                                                                              Eigen::Matrix<double, -1, 1> &theta_us_vec_proposed_ref,
                                                                              Eigen::Matrix<double, -1, 1> &lp_and_grad_outs,
                                                                              Eigen::Matrix<double, -1, 1> &us_theta_vec_current_segment,
                                                                              Eigen::Matrix<double, -1, 1> &us_velocity_vec_current_segment,
                                                                              const Eigen::Matrix<double, -1, 1> &theta_main_vec_initial_ref,
                                                                              const Eigen::Matrix<double, -1, 1> &M_inv_us_vec,
                                                                              const Eigen::Matrix<double, -1, 1> &theta_hat_us_vec,  // posterior mean from warmup
                                                                              const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                              const int L_ii,
                                                                              const double eps_1,
                                                                              const double eps_2,
                                                                              const double cos_eps_2,
                                                                              const double sin_eps_2,
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
  
        const int n_nuisance = velocity_us_vec_proposed_ref.size();
        Eigen::Matrix<double, -1, 1> grad_us = lp_and_grad_outs.segment(1, n_nuisance);
        
        // Per-component frequencies: omega_i = 1/sqrt(M_i)  (A = I reference measure)
        const Eigen::Matrix<double, -1, 1> inv_sqrt_M = M_inv_us_vec.array().sqrt();   // = 1/sqrt(M_i)
        const Eigen::Matrix<double, -1, 1> sqrt_M     = inv_sqrt_M.array().inverse();          // = sqrt(M_i)
        const Eigen::Matrix<double, -1, 1> angle      = eps_2 * inv_sqrt_M;            // omega_i * eps_2
        const Eigen::Matrix<double, -1, 1> cos_t      = angle.array().cos();
        const Eigen::Matrix<double, -1, 1> sin_t      = angle.array().sin();
        
        for (int l = 0; l < L_ii; l++) {
            if (Model_args_as_cpp_struct.burnin_leapfrog_steps != nullptr) {
                ++(*Model_args_as_cpp_struct.burnin_leapfrog_steps);
            }
          
                 // ---- Velocity half-kick ----
                 // v += (eps_1/2) * (M⁻¹ * grad_log_post + (θ - θ̂))
                 // velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * (M_inv_us_vec.array() * grad_us.array() + theta_us_vec_proposed_ref.array() - theta_hat_us_vec.array());
                 velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * M_inv_us_vec.array() * (grad_us.array() + theta_us_vec_proposed_ref.array() - theta_hat_us_vec.array());
          
                 // ---- Exact uniform rotation (all ω = 1) ---- // Shift to center: θ̃ = θ - θ̂
                 us_theta_vec_current_segment.array() = theta_us_vec_proposed_ref.array() - theta_hat_us_vec.array(); 
                 us_velocity_vec_current_segment = velocity_us_vec_proposed_ref;
                 
                 // Rotate (simple, uniform — cos/sin are scalars)  // shift back:
                 // theta_us_vec_proposed_ref.array() = cos_eps_2 * us_theta_vec_current_segment.array() + sin_eps_2 * us_velocity_vec_current_segment.array() + theta_hat_us_vec.array(); 
                 // velocity_us_vec_proposed_ref.array() = cos_eps_2 * us_velocity_vec_current_segment.array() - sin_eps_2 * us_theta_vec_current_segment.array();
                 theta_us_vec_proposed_ref.array()    = cos_t.array() * us_theta_vec_current_segment.array()
                                                      + sqrt_M.array() * sin_t.array() * us_velocity_vec_current_segment.array() + theta_hat_us_vec.array();
                 velocity_us_vec_proposed_ref.array() = cos_t.array() * us_velocity_vec_current_segment.array() - inv_sqrt_M.array() * sin_t.array() * us_theta_vec_current_segment.array();
                   
                 // ---- Recompute gradient at new θ ----
                 fn_lp_grad_InPlace(lp_and_grad_outs, Model_type,
                                    force_autodiff, force_PartialLog, multi_attempts,
                                    theta_main_vec_initial_ref, theta_us_vec_proposed_ref,
                                    y_ref, grad_option,
                                    Model_args_as_cpp_struct, 
                                    Stan_model_as_cpp_struct,
                                    LC_MVP_ws_structs,
                                    n_threads_WCP);
                 grad_us = lp_and_grad_outs.segment(1, n_nuisance);
                 
                 // ---- Velocity half-kick (with new θ and gradient) ----
                 // velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * (M_inv_us_vec.array() * grad_us.array() + theta_us_vec_proposed_ref.array() - theta_hat_us_vec.array());
                 velocity_us_vec_proposed_ref.array() += 0.5 * eps_1 * M_inv_us_vec.array() * (grad_us.array() + theta_us_vec_proposed_ref.array() - theta_hat_us_vec.array());
        }
  
}







template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE  void         fn_Diffusion_HMC_nuisance_only_single_iter_InPlace_process(        HMCResult &result_input,
                                                                                               T &rng_nuisance,
                                                                                               const std::string &Model_type,
                                                                                               const bool force_autodiff,
                                                                                               const bool force_PartialLog,
                                                                                               const bool multi_attempts, 
                                                                                               const Eigen::Matrix<int, -1, -1> &y_ref,
                                                                                               const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                                                               EHMC_fn_args_struct  &EHMC_args_as_cpp_struct,
                                                                                               const EHMC_Metric_struct  &EHMC_Metric_struct_as_cpp_struct,
                                                                                               const Stan_model_struct &Stan_model_as_cpp_struct,
                                                                                               std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs,
                                                                                               const int n_threads_WCP
) {

      //// important params
      const int N = Model_args_as_cpp_struct.N;
      const int n_nuisance =  Model_args_as_cpp_struct.n_nuisance;
      const int n_params_main = Model_args_as_cpp_struct.n_params_main;
      const int n_params = n_params_main + n_nuisance;
  
      const std::string grad_option = "us_only";
      
      const double eps_1 = EHMC_args_as_cpp_struct.eps_us;
      const double eps_1_sq = eps_1 * eps_1;
      const double cos_eps_2 = (1.0 - 0.25 * eps_1_sq) / (1.0 + 0.25 * eps_1_sq);
      const double eps_2 = std::acos(cos_eps_2);
      const double sin_eps_2 = std::sqrt(1.0 - (cos_eps_2 * cos_eps_2));
  
      double U_x_initial = 0.0;
      double U_x_prop = 0.0;
      double log_posterior_prop = 0.0;
      double log_posterior_0  = 0.0;
      double log_ratio = 0.0;
      double energy_old = 0.0;
      double energy_new = 0.0;
      
      //// result_input.store_current_state(); // sets initial theta and velocity to current theta and velocity
      result_input.main_theta_vec_0() =  result_input.main_theta_vec();
      result_input.us_theta_vec_0() =    result_input.us_theta_vec();
     
  {
  
      {
         //// Eigen::Matrix<double, -1, 1> std_norm_vec_us = Eigen::Matrix<double, -1, 1>::Zero(n_nuisance);
         generate_random_std_norm_vec_InPlace(result_input.us_velocity_0_vec(), rng_nuisance);
         result_input.us_velocity_0_vec().array() = ( result_input.us_velocity_0_vec().array() * (EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec).array().sqrt() );  //// .cast<float>() ;  
      }
     
    {
      
      {
 
      try {
          
              result_input.us_velocity_vec_proposed() =  result_input.us_velocity_0_vec() ;    // set TO initial velocity
              result_input.us_theta_vec_proposed() =  result_input.us_theta_vec();             // set TO CURRENT theta   
    
              //// --------------------------------------------------------------------------------------------------------------- ////    Perform L leapfrogs  //// ------------------------------------
              if (EHMC_args_as_cpp_struct.tau_us < EHMC_args_as_cpp_struct.eps_us) { 
                EHMC_args_as_cpp_struct.tau_us = EHMC_args_as_cpp_struct.eps_us;
              }
              // Draw + assign tau_ii:
              EHMC_args_as_cpp_struct.tau_us_ii = generate_random_tau_ii(EHMC_args_as_cpp_struct.tau_us, rng_nuisance);
              if (EHMC_args_as_cpp_struct.tau_us_ii < EHMC_args_as_cpp_struct.eps_us) { 
                EHMC_args_as_cpp_struct.tau_us_ii = EHMC_args_as_cpp_struct.eps_us; 
              }
              // Compute L_ii:
              int L_ii;
              if (EHMC_args_as_cpp_struct.diffusion_HMC == true)   L_ii = std::ceil( EHMC_args_as_cpp_struct.tau_us_ii /  eps_1 );
              if (EHMC_args_as_cpp_struct.diffusion_HMC == false)  L_ii = std::ceil( EHMC_args_as_cpp_struct.tau_us_ii /  EHMC_args_as_cpp_struct.eps_us );
              if (L_ii < 1) { L_ii = 1 ; }

              // if (grad_option_temp != "no_testing")  { //// --------------------------------------------------------------------------------
              //        const bool force_PartialLog_temp = true;
              //        //  const std::string grad_option_temp = "none";  // seems fine (windows) w/ 500 ---
              //        // const std::string grad_option_temp = "test";     // seems fine (windows) w/ 500 ---
              //        //   const std::string grad_option_temp = "us_only";     // seems fine (windows) w/ 500  --- seems fine (windows) w/ 1
              //        // const std::string grad_option_temp = "coeff_only";   // seems fine (windows)  w/ 500 ---
              //        // const std::string grad_option_temp = "corr_only";     // seems fine (windows)  w/ 500 ---
              //        //  const std::string grad_option_temp = "prev_only";      // seems fine (windows)  w/ 500 --- seems fine (windows) w/ 1
              // 
              //         //// initial lp / grad (for TESTING/DEBUG)
              //         fn_lp_grad_InPlace(     result_input.lp_and_grad_outs(),
              //                                 Model_type,
              //                                 force_autodiff, force_PartialLog_temp, multi_attempts,
              //                                 result_input.main_theta_vec(), result_input.us_theta_vec(),
              //                                 y_ref,  grad_option_temp,
              //                                 Model_args_as_cpp_struct,
              //                                 Stan_model_as_cpp_struct);
              // } //// --------------------------------------------------------------------------------
              
              //// initial lp  (and grad)
              fn_lp_grad_InPlace(     result_input.lp_and_grad_outs(), 
                                      Model_type, 
                                      force_autodiff, force_PartialLog, multi_attempts,
                                      result_input.main_theta_vec(),  result_input.us_theta_vec(), 
                                      y_ref,  grad_option,
                                      Model_args_as_cpp_struct, 
                                      Stan_model_as_cpp_struct,
                                      LC_MVP_ws_structs,
                                      n_threads_WCP);
              
              result_input.lp_and_grad_outs_0() = result_input.lp_and_grad_outs(); // ----
              
              log_posterior_0 =  result_input.lp_and_grad_outs()(0);
              U_x_initial = - log_posterior_0; //// initial energy
      
              if (EHMC_args_as_cpp_struct.diffusion_HMC == true) {
  
                         //// make params/containers for sampling u's using advanced HMC
                         // Eigen::Matrix<double, -1, 1>  us_theta_vec_current_segment =     result_input.us_theta_vec();
                         // Eigen::Matrix<double, -1, 1>  us_velocity_vec_current_segment =  result_input.us_velocity_vec();
      
                         leapfrog_integrator_diag_M_diffusion_HMC_nuisance_InPlace(   result_input.us_velocity_vec_proposed(),
                                                                                      result_input.us_theta_vec_proposed(),
                                                                                      result_input.lp_and_grad_outs(),
                                                                                      result_input.us_theta_vec_current_segment(),
                                                                                      result_input.us_velocity_vec_current_segment(),
                                                                                      result_input.main_theta_vec(),
                                                                                      EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec,
                                                                                      EHMC_Metric_struct_as_cpp_struct.theta_hat_us_vec,
                                                                                      y_ref,
                                                                                      L_ii,
                                                                                      eps_1, eps_2, cos_eps_2, sin_eps_2,
                                                                                      Model_type,
                                                                                      force_autodiff, force_PartialLog, multi_attempts,
                                                                                      grad_option,
                                                                                      Model_args_as_cpp_struct,
                                                                                      Stan_model_as_cpp_struct, 
                                                                                      LC_MVP_ws_structs,
                                                                                      n_threads_WCP,
                                                                                      fn_lp_grad_InPlace);
  
              } else  {
  
                        leapfrog_integrator_diag_M_standard_HMC_nuisance_InPlace(   result_input.us_velocity_vec_proposed(),
                                                                                    result_input.us_theta_vec_proposed(),
                                                                                    result_input.lp_and_grad_outs(),
                                                                                    result_input.main_theta_vec(),
                                                                                    EHMC_Metric_struct_as_cpp_struct.M_inv_us_vec,
                                                                                    y_ref,
                                                                                    L_ii,
                                                                                    EHMC_args_as_cpp_struct.eps_us,
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
              log_posterior_prop =   result_input.lp_and_grad_outs()(0);
              if (EHMC_args_as_cpp_struct.record_kinetic_energy_tau_derivatives) result_input.record_kinetic_energy_rate_us();
              U_x_prop = - log_posterior_prop;
                  
            //////////////////////////////////////////////////////////////////    M-H acceptance step  (i.e, Accept/Reject step)
            {
                
                  // // ============================================================
                  // // FIX 4: ENERGY CALCULATION (in MH accept/reject)
                  // // ============================================================
                  // // H = -log_post(θ) + ½ v' M v
                  // //
                  // // The -log_post ALREADY contains ½θ'θ from the Jacobian.
                  // // Do NOT add a separate ½θ'Mθ term — that would change the target!
                  // //
                  // // OLD (WRONG):
                  // //   energy  = -log_post
                  // //   energy += 0.5 * (v² * M).sum()
                  // //   energy += 0.5 * (θ² * M).sum()   ← REMOVE THIS LINE
                  // //
                  // // CORRECT:
                  // //   energy  = -log_post
                  // //   energy += 0.5 * (v² * M).sum()
                  // //
                  // // For old state:
                  // energy_old  = U_x_initial;   // = -log_post(θ_0)
                  // energy_old += 0.5 * (result_input.us_velocity_0_vec().array().square()
                  //                        * M_us_vec.array()).sum();
                  // // NO θ term!
                  // 
                  // // For proposed state:
                  // energy_new  = U_x_proposed;  // = -log_post(θ_prop)
                  // energy_new += 0.5 * (result_input.us_velocity_vec_proposed().array().square()
                  //                        * M_us_vec.array()).sum();
                  // // NO θ term!
                      
                    energy_old = U_x_initial;
                    energy_new = U_x_prop;
                    
                    energy_old +=  0.5 * (result_input.us_velocity_0_vec().array().square() * EHMC_Metric_struct_as_cpp_struct.M_us_vec.array()).sum();
                    energy_new +=  0.5 * (result_input.us_velocity_vec_proposed().array().square() * EHMC_Metric_struct_as_cpp_struct.M_us_vec.array()).sum();
                    
                    // if (EHMC_args_as_cpp_struct.diffusion_HMC == true)  {
                    //   // energy_old +=  0.5 * (result_input.us_theta_vec_0().array().square() * EHMC_Metric_struct_as_cpp_struct.M_us_vec.array()).sum();
                    //   // energy_new +=  0.5 * (result_input.us_theta_vec_proposed().array().square() * EHMC_Metric_struct_as_cpp_struct.M_us_vec.array()).sum();
                    //   // energy_old += 0.5 * result_input.us_theta_vec_0().squaredNorm();
                    //   // energy_new += 0.5 * result_input.us_theta_vec_proposed().squaredNorm();
                    //   energy_old += 0.5 * (result_input.us_theta_vec_0().array().square() * EHMC_Metric_struct_as_cpp_struct.M_us_vec.array()).sum();
                    //   energy_new += 0.5 * (result_input.us_theta_vec_proposed().array().square() * EHMC_Metric_struct_as_cpp_struct.M_us_vec.array()).sum();
                    //   
                    // }
          
                    log_ratio = - energy_new + energy_old;
    
            }
            
          if  (check_divergence(log_ratio, result_input) == true)    { /// if main_div, reject proposal 
                  
                  //// if  div, reject proposal 
                  result_input.us_div() = 1;
                  result_input.us_p_jump() = 0.0;
                  result_input.reject_proposal_us();  // # reject proposal
            
          }  else {  // if no us_div, carry on with MH step 
            
                    result_input.us_div() = 0;
                    result_input.us_p_jump() = std::min(1.0, stan::math::exp(log_ratio));
                    
                    const double rand_unif = generate_random_std_uniform(rng_nuisance);
                    const double log_rand_unif = stan::math::log(rand_unif);
                    
                    if  (log_rand_unif > log_ratio)   {  // # reject proposal
                       result_input.reject_proposal_us();  // # reject proposal
                    } else {   // # accept proposal
                       result_input.accept_proposal_us(); // # accept proposal
                    }
                  
          }

      } catch (...) { // if iteration fails (recorded as us_div)
        
                // std::cout << "  Could not evaluate lp_grad function when sampling nuisance parameters " << ")\n";

                result_input.us_div() = 1; // record as div 
                result_input.us_p_jump() = 0.0; // set p_jump to zero
                result_input.reject_proposal_us(); // # reject proposal

      }
      
    }  
 

  }

 
}
  
  
  //return result_input;
  
  
}


 
 
 
  


 
