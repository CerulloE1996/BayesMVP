
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
        
          } else if (nuisance_transformation == "inv_logit") {
            
              //// "inv_logit" branch added. Previously the chain had no "inv_logit" branch and no
              //// final else, so latent_trait with nuisance_transformation = "inv_logit" left u_vec at zero and returned NaN lp
              //// silently. Same transform as fn_MVP_compute_nuisance_T (MVP_helpers_migrated.hpp): u = inv_logit(u_unc).
              u_vec.array() +=      fn_EIGEN_double( u_unc_vec, "inv_logit", vect_type_Phi, false).array();
            
          } else {
            
              //// unknown strings now stop with a clear message instead of silently leaving u_vec at zero.
              throw std::invalid_argument("fn_MVP_compute_nuisance: unknown nuisance_transformation '" + nuisance_transformation +
                                          "' (allowed: Phi, Phi_approx, Phi_approx_rough, tanh, inv_logit).");
            
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
        
              log_jac_u  =  - 0.5 * u_unc_vec.size() * stan::math::log(2 * M_PI)
                            - 0.5 * u_unc_vec.array().square().sum() ;
        
          } else if (nuisance_transformation == "Phi_approx") {
        
              log_jac_u   =  fn_EIGEN_double((a_times_3 * u_unc_vec.array().square() +  b).matrix(), "log", vect_type_log).sum();
              log_jac_u   += fn_EIGEN_double(u_vec, "log", vect_type_log).sum(); 
              log_jac_u   += fn_EIGEN_double(u_vec, "log1m", vect_type_log).sum();
        
          } else if (nuisance_transformation == "Phi_approx_rough") {
        
              log_jac_u  = u_unc_vec.size() * stan::math::log(1.702)
                            + fn_EIGEN_double( u_vec, "log", vect_type_log).sum()
                            + fn_EIGEN_double( u_vec, "log1m", vect_type_log).sum()  ;
        
        
          } else if (nuisance_transformation == "tanh") {
        
              log_jac_u  = u_unc_vec.size() * stan::math::log(2.0)
                            + fn_EIGEN_double( u_vec, "log", vect_type_log ).sum()
                            + fn_EIGEN_double( u_vec , "log1m", vect_type_log ).sum();
        
          } else if (nuisance_transformation == "inv_logit") {
            
              //// du/dx = u (1 - u), so log J = sum log(u) + log(1 - u) (no constant), as in
              //// fn_MVP_compute_nuisance_log_jac_u_T (MVP_helpers_migrated.hpp).
              log_jac_u  =    fn_EIGEN_double( u_vec, "log", vect_type_log ).sum()
                            + fn_EIGEN_double( u_vec , "log1m", vect_type_log ).sum();
            
          } else {
            
              throw std::invalid_argument("fn_MVP_compute_nuisance_log_jac_u: unknown nuisance_transformation '" + nuisance_transformation +
                                          "' (allowed: Phi, Phi_approx, Phi_approx_rough, tanh, inv_logit).");
            
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
        
          } else if (nuisance_transformation == "inv_logit") {
            
              du_wrt_duu.array() +=    u_vec.array() * (1.0 - u_vec.array() ) ;     //// du/dx = u (1 - u)
            
          } else {
            
              throw std::invalid_argument("fn_MVP_nuisance_first_deriv: unknown nuisance_transformation '" + nuisance_transformation +
                                          "' (allowed: Phi, Phi_approx, Phi_approx_rough, tanh, inv_logit).");
            
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
             
                //// d/dx [ log(g'(x)) + log(u) + log(1 - u) ], with g(x) = a x^3 + b x, u = inv_logit(g(x)), g' = 3 a x^2 + b:
                ////   = g' (1 - 2u) + g''/g',   g'' = 6 a x   (matches fn_MVP_nuisance_deriv_of_log_det_J_T in MVP_helpers_migrated.hpp).
                //// The second term was (1 - 2u)/g' previously, which is wrong everywhere except x = 0.
                d_J_wrt_duu.array() += (  ( du_wrt_duu.array() * (  (1.0 - 2.0 * u_vec.array()  ) / ( u_vec.array() * (1.0 - u_vec.array() ) )  )   ).array()  + (  ( 6.0 * a * u_unc_vec.array() )  / (a_times_3*u_unc_vec.array().square() + b).array() ) ).array() ;
            
          } else if (nuisance_transformation == "Phi_approx_rough") {   ;    
            
                d_J_wrt_duu.array() +=  1.702 * (1.0 - 2.0 * u_vec.array() ) ;    
            
          } else if (nuisance_transformation == "tanh") { 
            
                d_J_wrt_duu.array() +=  2.0 * (1.0 - 2.0 * u_vec.array() ) ;   
            
          } else if (nuisance_transformation == "inv_logit") {
            
                d_J_wrt_duu.array() +=  (1.0 - 2.0 * u_vec.array() ) ;   //// d/dx [log(u) + log(1 - u)] = 1 - 2u
            
          } else {
            
                throw std::invalid_argument("fn_MVP_nuisance_deriv_of_log_det_J: unknown nuisance_transformation '" + nuisance_transformation +
                                            "' (allowed: Phi, Phi_approx, Phi_approx_rough, tanh, inv_logit).");
            
          }
          
          //  return d_J_wrt_duu;
  
}







 




////
//// ---- Phi_type / inv_Phi_type handling shared by the four autodiff (stan::math::var) reference copies:
////
//// Used by MVP_lp_grad_AD_fns.hpp (LC_MVP), std_MVP_lp_grad_AD_fns.hpp (MVP),
//// MVOP_lp_grad_AD_fns.hpp (MVOP and LC_MVOP) and LC_LT_lp_grad_AD_fns.hpp (latent_trait). Previously:
////   - the autodiff copies of MVP, MVOP and LC_MVOP computed the standard-scale (interior) GHK step with the EXACT Phi / inv_Phi
////     whatever Phi_type was, so Phi_type = "Phi_approx" only reached their tails;
////   - every autodiff copy tied the inverse CDF to Phi_type and ignored inv_Phi_type, whereas the manual-gradient paths choose the
////     CDF from Phi_type and the inverse from inv_Phi_type independently (KernelChoice, MVP_helpers_migrated.hpp);
////   - an unknown Phi_type string was not rejected: in the LC_MVP and latent_trait copies it matched no branch, so the likelihood
////     was silently left out of lp; in the MVP and MVOP copies it was treated as "Phi_approx" in the tails and as "Phi" elsewhere.
//// (The missing "inv_logit" nuisance branch of the same copies, is fixed inline in each copy.)
//// The string rules here are the ones kernel_choice_from_args applies to the manual paths, so manual and autodiff agree:
////   Phi_type:     "Phi" -> exact CDF;  "Phi_approx" or "Phi_approx_2" -> Phi_approx(x) = inv_logit(0.07056 x^3 + 1.5976 x)
////                 (Bowling et al., 2009);
////   inv_Phi_type: "inv_Phi" -> exact inverse (Wichura, 1988, AS241);  "inv_Phi_approx" -> exact inverse of Phi_approx;
//// anything else throws std::invalid_argument, so no setting is silently ignored.
////
struct AD_Phi_setting_struct {
      bool use_Phi_approx;       //// CDF: Phi_approx (true) or exact Phi (false)
      bool use_inv_Phi_approx;   //// inverse CDF: inv_Phi_approx (true) or exact inv_Phi (false)
};



inline AD_Phi_setting_struct fn_AD_Phi_setting_from_strings(  const std::string &Phi_type,
                                                              const std::string &inv_Phi_type,
                                                              const std::string &caller_name
) {
  
      AD_Phi_setting_struct AD_Phi_setting;
      
      if (Phi_type == "Phi") {
            AD_Phi_setting.use_Phi_approx = false;
      } else if ((Phi_type == "Phi_approx") || (Phi_type == "Phi_approx_2")) {
            AD_Phi_setting.use_Phi_approx = true;
      } else {
            throw std::invalid_argument(caller_name + ": unknown Phi_type '" + Phi_type + "' (allowed: Phi, Phi_approx, Phi_approx_2).");
      }
      
      if (inv_Phi_type == "inv_Phi") {
            AD_Phi_setting.use_inv_Phi_approx = false;
      } else if (inv_Phi_type == "inv_Phi_approx") {
            AD_Phi_setting.use_inv_Phi_approx = true;
      } else {
            throw std::invalid_argument(caller_name + ": unknown inv_Phi_type '" + inv_Phi_type + "' (allowed: inv_Phi, inv_Phi_approx).");
      }
      
      return AD_Phi_setting;
  
}



//// CDF on the standard (interior) scale: exact Phi or Phi_approx (stan::math::Phi_approx uses the same 0.07056 / 1.5976 constants).
inline stan::math::var fn_AD_Phi_var(  const stan::math::var &x_var,
                                       const bool use_Phi_approx
) {
      if (use_Phi_approx) return stan::math::Phi_approx(x_var);
      return stan::math::Phi(x_var);
}



//// Inverse CDF on the standard (interior) scale: exact AS241 (stan::math::inv_Phi) or inv_Phi_approx_var (var_fns.hpp).
inline stan::math::var fn_AD_inv_Phi_var(  const stan::math::var &probability_var,
                                           const bool use_inv_Phi_approx
) {
      if (use_inv_Phi_approx) return inv_Phi_approx_var(probability_var);
      return stan::math::inv_Phi(probability_var);
}



//// Inverse CDF in the tails, from the two consistent log-scale representations (log p, log(1 - p)) of the same probability:
//// exact (inv_Phi_from_log_p_exact_var, double_fns.hpp) or inv_Phi_approx from the logit, logit(p) = log p - log(1 - p).
inline stan::math::var fn_AD_inv_Phi_from_log_probs_var(  const stan::math::var &log_probability_var,
                                                          const stan::math::var &log_1m_probability_var,
                                                          const bool use_inv_Phi_approx
) {
      if (use_inv_Phi_approx) return inv_Phi_approx_from_logit_prob_var(log_probability_var - log_1m_probability_var);
      return inv_Phi_from_log_p_exact_var(log_probability_var, log_1m_probability_var);
}



////
//// ---- Up-front check of the settings strings for the multi_attempts evaluators:
////
//// This check accompanies the try/catch around attempts 1 and 2 of fn_lp_grad_MVP_multi_attempts_InPlace_process,
//// fn_lp_grad_MVOP_multi_attempts_InPlace_process (MVP_lp_grad_multi_attempts.hpp) and attempt 1 of fn_lp_grad_LT_LC_multi_attempts_InPlace_process
//// (LT_LC_lp_grad_multi_attempts.hpp). Those try/catch blocks catch std::exception, like the existing attempt 3, so that a
//// numerical exception (Stan math's "Phi: x is nan", inv_Phi out of range, fast_inv_Phi_approx's out-of-range argument, ...) moves on to
//// the next attempt. A mistyped SETTING must not be swallowed that way, so it is checked here, before any attempt runs:
////   nuisance_transformation: Phi, Phi_approx, tanh, inv_logit, and Phi_approx_rough only where the manual path accepts it
////                            (latent_trait; the MVP / LC_MVP / MVOP / LC_MVOP manual paths reject it in kernel_choice_from_args);
////   Phi_type / inv_Phi_type: the rules of fn_AD_Phi_setting_from_strings (= kernel_choice_from_args).
////
inline void fn_check_settings_strings_before_multi_attempts(  const Model_fn_args_struct &Model_args_as_cpp_struct,
                                                              const bool allow_Phi_approx_rough,
                                                              const std::string &caller_name
) {
  
      const std::string &nuisance_transformation = Model_args_as_cpp_struct.Model_args_strings(12);
      
      const bool nuisance_transformation_is_allowed =    (nuisance_transformation == "Phi")
                                                      || (nuisance_transformation == "Phi_approx")
                                                      || (nuisance_transformation == "tanh")
                                                      || (nuisance_transformation == "inv_logit")
                                                      || (allow_Phi_approx_rough && (nuisance_transformation == "Phi_approx_rough"));
      if (nuisance_transformation_is_allowed == false) {
            throw std::invalid_argument(caller_name + ": unknown or unsupported nuisance_transformation '" + nuisance_transformation + "' (allowed: Phi, Phi_approx, tanh, inv_logit" +
                                        (allow_Phi_approx_rough ? std::string(", Phi_approx_rough") : std::string("")) + ").");
      }
      
      fn_AD_Phi_setting_from_strings(  Model_args_as_cpp_struct.Model_args_strings(1),
                                       Model_args_as_cpp_struct.Model_args_strings(2),
                                       caller_name);
  
}
