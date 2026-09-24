
//// MVP_manual_grad_calc_fns.hpp

#pragma once

 
 
 
#include <Eigen/Dense>
 

 
 

 
using namespace Eigen;

 

 



#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE



 
 
 

////
//// ---- forward declaration of the templated version defined further down this file,
////      so that the string-dispatch version (still used by the latent_trait NoLog / PartialLog paths) can delegate to it for
////      exact Phi and pick up the right-side-seam reflection derived there.
////
template <Vec vec>
inline void fn_MVP_compute_lp_GHK_cols_T(   const int t,
                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_chunk,
                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array,
                                          const KernelChoice &k);

//// fn that computes important quantities needed for the GHK parameterisation of the MVP / LC_MVP (and latent_trait) models
ALWAYS_INLINE  void fn_MVP_compute_lp_GHK_cols(   const int t,
                                                  Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
                                                  Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
                                                  Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                                  Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
                                                  Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  Bound_Z,
                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  y_chunk,
                                                  const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  u_array,
                                                  const Model_fn_args_struct &Model_args_as_cpp_struct
) {

       const std::string vect_type_log = Model_args_as_cpp_struct.Model_args_strings(4);
       const std::string Phi_type = Model_args_as_cpp_struct.Model_args_strings(1);
       const std::string vect_type_Phi = Model_args_as_cpp_struct.Model_args_strings(7);
       const std::string inv_Phi_type = Model_args_as_cpp_struct.Model_args_strings(2);
       const std::string vect_type_inv_Phi = Model_args_as_cpp_struct.Model_args_strings(9);
       
       ////
       //// ---- exact Phi (Phi_type = "Phi") -> templated version with the right-side-seam reflection
       ////      (prob = Phi((1 - 2y) Bound_Z), Z = (1 - 2y) Phi^{-1}((y - (2y - 1) u) prob); derivation in fn_MVP_compute_lp_GHK_cols_T).
       ////      Same Phi / inv_Phi / log kernels as the string dispatch (stan::math for "Stan", fast_*_AVX2 / _AVX512 otherwise).
       ////
       if (Phi_type == "Phi") {
             KernelChoice kernel_choice_for_GHK_cols;
             kernel_choice_for_GHK_cols.Phi_approx     = false;
             kernel_choice_for_GHK_cols.inv_Phi_approx = (inv_Phi_type == "inv_Phi_approx");
             kernel_choice_for_GHK_cols.nuisance       = NuisTf::Phi;   //// unused by fn_MVP_compute_lp_GHK_cols_T
             const Vec vec_for_GHK_cols = vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0));
             DISPATCH_VEC(vec_for_GHK_cols, fn_MVP_compute_lp_GHK_cols_T,
                          t, Bound_U_Phi_Bound_Z, Phi_Z, Z_std_norm, prob, y1_log_prob, Bound_Z, y_chunk, u_array, kernel_choice_for_GHK_cols);
             return;
       }

       ////// Compute on NON-log-scale for all observations using ".col(t)".
       Bound_U_Phi_Bound_Z.col(t) =   fn_EIGEN_double( Bound_Z.col(t), Phi_type, vect_type_Phi );

       Phi_Z.col(t).array() = y_chunk.col(t).array() * Bound_U_Phi_Bound_Z.col(t).array() +   (y_chunk.col(t).array() -  Bound_U_Phi_Bound_Z.col(t).array()) *
                            ((y_chunk.col(t).array()  + (y_chunk.col(t).array()  - 1.0)) * u_array.col(t).array());

       Z_std_norm.col(t) =   fn_EIGEN_double( Phi_Z.col(t),   inv_Phi_type, vect_type_inv_Phi);


       prob.col(t).array() =    y_chunk.col(t).array()  * (1.0 - Bound_U_Phi_Bound_Z.col(t).array() ) + ( y_chunk.col(t).array()  -  1.0)  *
                                Bound_U_Phi_Bound_Z.col(t).array() * ( y_chunk.col(t).array()  +  (  y_chunk.col(t).array()  - 1.0)  )  ;
       // prob.col(t).array() = prob.col(t).array().max(1e-30);

       y1_log_prob.col(t)  =    fn_EIGEN_double( prob.col(t),  "log", vect_type_log);

}
// template <Vec vec>
// inline void fn_MVP_compute_lp_GHK_cols(   const int t,
//                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
//                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
//                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
//                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
//                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
//                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
//                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_chunk,
//                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array,
//                                           const KernelChoice &k
// ) {
//   
//         Bound_U_Phi_Bound_Z.col(t) = Bound_Z.col(t);
//         Phi_col_inplace<vec>(Bound_U_Phi_Bound_Z, t, k);
//       
//         Phi_Z.col(t).array() = y_chunk.col(t).array() * Bound_U_Phi_Bound_Z.col(t).array()
//                              + (y_chunk.col(t).array() - Bound_U_Phi_Bound_Z.col(t).array())
//                              * ((2.0 * y_chunk.col(t).array() - 1.0) * u_array.col(t).array());
//       
//         Z_std_norm.col(t) = Phi_Z.col(t);
//         inv_Phi_col_inplace<vec>(Z_std_norm, t, k);
//       
//         prob.col(t).array() = y_chunk.col(t).array() * (1.0 - Bound_U_Phi_Bound_Z.col(t).array())
//                             + (y_chunk.col(t).array() - 1.0) * Bound_U_Phi_Bound_Z.col(t).array()
//                             * (2.0 * y_chunk.col(t).array() - 1.0);
//       
//         y1_log_prob.col(t) = prob.col(t);
//         apply_col_inplace<vec, Fn::log>(y1_log_prob, t);
//   
// }




//// =====================================================================================
//// PART C: fn_MVP_compute_lp_GHK_cols (the binary-only MVP version the driver calls).
//// =====================================================================================
template <Vec vec>
inline void fn_MVP_compute_lp_GHK_cols_T(   const int t,
                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_chunk,
                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array,
                                          const KernelChoice &k
) {
  
        if (k.Phi_approx) {
          
                //// ---- Phi_approx setting: original code, unchanged ----
                Bound_U_Phi_Bound_Z.col(t) = Bound_Z.col(t);
                Phi_col_inplace<vec>(Bound_U_Phi_Bound_Z, t, k);
              
                Phi_Z.col(t).array() = y_chunk.col(t).array() * Bound_U_Phi_Bound_Z.col(t).array()
                                     + (y_chunk.col(t).array() - Bound_U_Phi_Bound_Z.col(t).array())
                                     * ((2.0 * y_chunk.col(t).array() - 1.0) * u_array.col(t).array());
              
                Z_std_norm.col(t) = Phi_Z.col(t);
                inv_Phi_col_inplace<vec>(Z_std_norm, t, k);
              
                prob.col(t).array() = y_chunk.col(t).array() * (1.0 - Bound_U_Phi_Bound_Z.col(t).array())
                                    + (y_chunk.col(t).array() - 1.0) * Bound_U_Phi_Bound_Z.col(t).array()
                                    * (2.0 * y_chunk.col(t).array() - 1.0);
              
                y1_log_prob.col(t) = prob.col(t);
                apply_col_inplace<vec, Fn::log>(y1_log_prob, t);
                
                return;
          
        }
        
        ////
        //// ---- Exact Phi (Phi_type = "Phi"): right-side seam fixed by reflection.
        ////
        //// The old code formed prob = 1 - Phi(Bound_Z) for y = 1 and Phi_Z = Phi(B) + u (1 - Phi(B)), Z = Phi^{-1}(Phi_Z).
        //// Near Bound_Z = +7.5 (just inside overflow_threshold), 1 - Phi(B) ~ 3e-14 is the difference of two numbers
        //// near 1, so it kept only ~2-3 significant digits (up to ~1e-3 nats in log prob), and Phi_Z rounded to 1
        //// (Z = +Inf) once u > ~0.996. Using the symmetry Phi(-x) = 1 - Phi(x) (exact for Phi and for the AS241
        //// inverse, inv_Phi(1 - q) = -inv_Phi(q)), with sign_y = 1 - 2y (+1 for y = 0, -1 for y = 1):
        ////
        ////   prob          = Phi(sign_y * Bound_Z)                 [ = Phi(B) for y = 0,  Phi(-B) = 1 - Phi(B) for y = 1 ]
        ////   q             = (y - (2y - 1) u) * prob               [ = u Phi(B) = Phi_Z for y = 0,  (1 - u)(1 - Phi(B)) = 1 - Phi_Z for y = 1 ]
        ////   Z             = sign_y * Phi^{-1}(q)                  [ = Phi^{-1}(Phi_Z) in both cases ]
        ////   Phi_Z         = y + sign_y * q
        ////   Phi(Bound_Z)  = y + sign_y * prob                     (stored for completeness; only the Phi_approx derivative reads it)
        ////
        //// For y = 0 every quantity is mathematically identical to before and uses the same products (up to compiler FMA contraction). The derivatives the
        //// gradient code uses are unchanged, because they are the analytic derivatives of the same functions:
        ////   d prob / d Bound_Z = sign_y * phi(Bound_Z),   d Z / d q = sign_y / phi(Z)  (so dZ/dPhi_Z = 1/phi(Z) as before).
        ////
        prob.col(t).array() = (1.0 - 2.0 * y_chunk.col(t).array()) * Bound_Z.col(t).array();
        Phi_col_inplace<vec>(prob, t, k);
        
        Bound_U_Phi_Bound_Z.col(t).array() = y_chunk.col(t).array()
                                           + (1.0 - 2.0 * y_chunk.col(t).array()) * prob.col(t).array();
        
        Z_std_norm.col(t).array() = (y_chunk.col(t).array() - (2.0 * y_chunk.col(t).array() - 1.0) * u_array.col(t).array())
                                  * prob.col(t).array();                                                             //// q
        Phi_Z.col(t).array() = y_chunk.col(t).array()
                             + (1.0 - 2.0 * y_chunk.col(t).array()) * Z_std_norm.col(t).array();                    //// y + sign_y * q
        inv_Phi_col_inplace<vec>(Z_std_norm, t, k);                                                                    //// Phi^{-1}(q)
        Z_std_norm.col(t).array() *= (1.0 - 2.0 * y_chunk.col(t).array());                                             //// sign_y * Phi^{-1}(q)
        
        y1_log_prob.col(t) = prob.col(t);
        apply_col_inplace<vec, Fn::log>(y1_log_prob, t);
  
}
 
 
 
 
 
//// fn that computes phi_Bound_Z needed for the gradient of the GHK parameterisation of the MVP / LC_MVP (and latent_trait) models
ALWAYS_INLINE  void fn_MVP_compute_phi_Bound_Z_cols(      const int t,
                                                          Eigen::Matrix<double, -1, -1> &phi_Bound_Z, //// updating this
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  Bound_U_Phi_Bound_Z,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  Bound_Z,
                                                          const Model_fn_args_struct &Model_args_as_cpp_struct
) {
   
   
         const double sqrt_2_pi_recip = 1.0 / std::sqrt(2.0 * M_PI);
         const double a = 0.07056;
         const double b = 1.5976;
         const double a_times_3 = 3.0 * a;
         
         const std::string Phi_type = Model_args_as_cpp_struct.Model_args_strings(1);
         const std::string vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);
         
         phi_Bound_Z.col(t).setZero();
     
         ///////// grad stuff
         if ( (Phi_type == "Phi_approx") || (Phi_type == "Phi_approx_2") ) { // vect_type
              const Eigen::Matrix<double, -1, 1> Bound_Z_col_t_sq = stan::math::square(Bound_Z.col(t));
              Eigen::Matrix<double, -1, 1> temp = (a_times_3 * Bound_Z_col_t_sq.array() + b).matrix();
              temp.array() *= Bound_U_Phi_Bound_Z.col(t).array();
              temp.array() *= (1.0 - Bound_U_Phi_Bound_Z.col(t).array()).array();
              phi_Bound_Z.col(t).array() += temp.array();
         }  else if (Phi_type == "Phi")   {  
              // const Eigen::Matrix<double, -1, 1> Bound_Z_col_t_sq = stan::math::square(Bound_Z.col(t));
              // Eigen::Matrix<double, -1, 1> temp = (-0.5)*Bound_Z_col_t_sq; 
              // temp = fn_EIGEN_double(temp, "exp", vect_type_exp);
              // temp = sqrt_2_pi_recip*temp;
              // phi_Bound_Z.col(t).array() +=   temp.array();
              ////
              phi_Bound_Z.col(t).array() += (sqrt_2_pi_recip*fn_EIGEN_double((-0.5)*stan::math::square(Bound_Z.col(t)), "exp", vect_type_exp)).array();
         }
         
         // ///////// grad stuff
         // if ( (Phi_type == "Phi_approx") || (Phi_type == "Phi_approx_2") ) { // vect_type
         //   phi_Bound_Z.col(t).array()  +=         ( a_times_3 * Bound_Z.col(t).array().square() + b  ).array()  *  Bound_U_Phi_Bound_Z.col(t).array() * (1.0 -  Bound_U_Phi_Bound_Z.col(t).array() )   ;
         // }  else if (Phi_type == "Phi")   {  
         //   phi_Bound_Z.col(t).array()  +=         sqrt_2_pi_recip * fn_EIGEN_double( ( - 0.5 * Bound_Z.col(t).array().square() ).matrix(),  "exp", vect_type_exp).array();
         // }
         
   
}





 
 
 
 
// fn that computes phi_Z_recip auto for the gradient of the GHK parameterisation of the MVP / LC_MVP (and latent_trait) models
ALWAYS_INLINE  void fn_MVP_compute_phi_Z_recip_cols(     const int t,
                                                         Eigen::Matrix<double, -1, -1> &phi_Z_recip,   //// updating this
                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  Phi_Z,
                                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  Z_std_norm,
                                                         const Model_fn_args_struct &Model_args_as_cpp_struct
) { 
   
       const double sqrt_2_pi_recip = 1.0 / std::sqrt(2.0 * M_PI);
       const double a = 0.07056;
       const double b = 1.5976;
       const double a_times_3 = 3.0 * a;
       
       const std::string Phi_type = Model_args_as_cpp_struct.Model_args_strings(1);
       const std::string vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);
       
       phi_Z_recip.col(t).setZero();
       
       ///////// grad stuff
       if ( (Phi_type == "Phi_approx") || (Phi_type == "Phi_approx_2") ) { // vect_type
             
               const Eigen::Matrix<double, -1, 1> Z_col_t_sq = stan::math::square(Z_std_norm.col(t));
               Eigen::Matrix<double, -1, 1> temp = (a_times_3 * Z_col_t_sq.array() + b).matrix();
               temp.array() *= Phi_Z.col(t).array();
               temp.array() *= (1.0 - Phi_Z.col(t).array()).array();
               temp = stan::math::inv(temp);
               phi_Z_recip.col(t).array() += temp.array();
           
       }  else if (Phi_type == "Phi")   {  
         
               // const Eigen::Matrix<double, -1, 1> Z_col_t_sq = stan::math::square(Z_std_norm.col(t));
               // Eigen::Matrix<double, -1, 1> temp = (-0.5)*stan::math::square(Z_std_norm.col(t)); 
               // temp = fn_EIGEN_double((-0.5)*stan::math::square(Z_std_norm.col(t)), "exp", vect_type_exp);
               // temp = sqrt_2_pi_recip*fn_EIGEN_double((-0.5)*stan::math::square(Z_std_norm.col(t)), "exp", vect_type_exp);
               // temp = stan::math::inv(sqrt_2_pi_recip*fn_EIGEN_double((-0.5)*stan::math::square(Z_std_norm.col(t)), "exp", vect_type_exp));
               ////
               phi_Z_recip.col(t).array() += (stan::math::inv(sqrt_2_pi_recip*fn_EIGEN_double((-0.5)*stan::math::square(Z_std_norm.col(t)), "exp", vect_type_exp))).array();
         
       }
       
       // ///////// grad stuff
       // if ( (Phi_type == "Phi_approx") || (Phi_type == "Phi_approx_2") ) { // vect_type
       //     phi_Z_recip.col(t).array()  +=    1.0 / ((  ( a_times_3 * Z_std_norm.col(t).array().square() + b  ).array()  ).array()*  Phi_Z.col(t).array() * (1.0 -  Phi_Z.col(t).array())  ).array() ;
       // }  else if (Phi_type == "Phi")   {  
       //     phi_Z_recip.col(t).array()  +=    1.0 /  (   sqrt_2_pi_recip * fn_EIGEN_double( ( - 0.5 * Z_std_norm.col(t).array().square() ).matrix(),  "exp", vect_type_exp) ).array();
       // } 
        
   
} 




 
 
 
 
// Helper functions for matrix products
template<typename M1, typename M2>
ALWAYS_INLINE void compute_rowwise_products( Eigen::Ref<M1> prod_temp,
                                             Eigen::Ref<M2> recip_temp, 
                                             const Eigen::Ref<const Eigen::Matrix<double,-1,-1>> prob,
                                             const Eigen::Ref<const Eigen::Matrix<double,-1,-1>> prob_recip,
                                             const int chunk_size,
                                             const int n_tests
) {
         
         // for (int i = 0; i < n_tests; i++) {  
         //   int t = n_tests - (i + 1);
         //   prod_temp.col(t).array() = prob.block(0, t + 0, chunk_size, i + 1).rowwise().prod().array();
         //   recip_temp.col(t).array() = prob_recip.block(0, t + 0, chunk_size, i + 1).rowwise().prod().array();
         // }
         
          prod_temp.col(n_tests-1) = prob.col(n_tests-1);
          recip_temp.col(n_tests-1) = prob_recip.col(n_tests-1);
          for (int t = n_tests - 2; t >= 0; t--) {
            prod_temp.col(t) = prod_temp.col(t+1).array() * prob.col(t).array();
            recip_temp.col(t) = recip_temp.col(t+1).array() * prob_recip.col(t).array();
          } 
         
}
 
 
// Helper for latent class calculations
template<typename M>
ALWAYS_INLINE void compute_latent_class_terms(   Eigen::Ref<M> common_grad,
                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> prod_temp_all,
                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> recip_temp,
                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> prob_n_recip,
                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> prev_per_obs,
                                                 const int n_tests
) {
   
   for (int i = 0; i < n_tests; i++) {  
     int t = n_tests - (i + 1);
     common_grad.col(t) = ((prev_per_obs.array() * prob_n_recip.array()) *   (prod_temp_all.array() * recip_temp.col(t).array()).array());
   }
   
}
 
 
 
// Helper for final transformations
template<typename M1, typename M2>
ALWAYS_INLINE void compute_final_terms(  Eigen::Ref<M1> y_sign_out,
                                         Eigen::Ref<M2> y_m_ysign_out,
                                         const Eigen::Ref<const Eigen::Matrix<double,-1,-1>> y_sign_chunk,
                                         const Eigen::Ref<const Eigen::Matrix<double,-1,-1>> y_m_y_sign_x_u,
                                         const Eigen::Ref<const Eigen::Matrix<double,-1,-1>> phi_Z_recip,
                                         const Eigen::Ref<const Eigen::Matrix<double,-1,-1>> phi_Bound_Z,
                                         const Eigen::Ref<const Eigen::Matrix<double,-1,-1>> L_Omega_recip_double,
                                         const int n_tests
) {
   
   for (int t = 0; t < n_tests; t++) {
       y_sign_out.col(t).array() = y_sign_chunk.col(t).array() * phi_Bound_Z.col(t).array() * L_Omega_recip_double(t, t);
       y_m_ysign_out.col(t).array() = y_m_y_sign_x_u.col(t).array() * phi_Z_recip.col(t).array() * phi_Bound_Z.col(t).array() *  L_Omega_recip_double(t, t);
   }
   
}
 
 
 
 
// y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip(n, t) // wrong!!!!!!!!!!

//// fn that computes important quantities needed for the gradient of the GHK parameterisation of the MVP / LC_MVP (and latent_trait) models
ALWAYS_INLINE  void fn_MVP_grad_prep(            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob,
                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_sign_chunk,
                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_m_y_sign_x_u, // 10
                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_recip_double,
                                                 ////
                                                 // const double prev_double,
                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> prev_per_obs,
                                                 ////
                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  prob_n_recip,
                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> phi_Z_recip,
                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> phi_Bound_Z, // 15
                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob_recip,
                                                 Eigen::Ref<Eigen::Matrix<double, -1, -1>> prop_rowwise_prod_temp,
                                                 Eigen::Ref<Eigen::Matrix<double, -1, -1>> prop_recip_rowwise_prod_temp,
                                                 Eigen::Ref<Eigen::Matrix<double, -1, 1>>  prop_rowwise_prod_temp_all, // 20
                                                 Eigen::Ref<Eigen::Matrix<double, -1, -1>> common_grad_term_1,
                                                 Eigen::Ref<Eigen::Matrix<double, -1, -1>> y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip, // 22
                                                 Eigen::Ref<Eigen::Matrix<double, -1, -1>> y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                 const Model_fn_args_struct &Model_args_as_cpp_struct
) {    
  
       const int  n_class = Model_args_as_cpp_struct.Model_args_ints(1);
       const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
       const std::string vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
       
       // const std::string &Phi_type = Model_args_as_cpp_struct.Model_args_strings(8);
       // const std::string &vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);
       // const bool skip_checks_exp =   Model_args_as_cpp_struct.Model_args_bools(6);
   
       const double a_times_3 = 3.0 * 0.07056; 
       const double b = 1.5976;
       const double sqrt_2_pi_recip = 1.0 / sqrt(2.0 * M_PI);
       
       const int chunk_size = y_sign_chunk.rows();
       const int n_tests = y_sign_chunk.cols();
       
       // Calculate row-wise products
       compute_rowwise_products( prop_rowwise_prod_temp,
                                 prop_recip_rowwise_prod_temp,
                                 prob,
                                 prob_recip,
                                 chunk_size,
                                 n_tests);
       
       // Calculate full row products
       prop_rowwise_prod_temp_all.array() = prob.rowwise().prod().array();
       
       // Handle latent class case
       if (n_class > 1) {
         compute_latent_class_terms(   common_grad_term_1,
                                       prop_rowwise_prod_temp_all,
                                       prop_recip_rowwise_prod_temp,
                                       prob_n_recip,
                                       prev_per_obs,
                                       n_tests);
       } else {
         common_grad_term_1.setOnes();
       }
       
       // Final transformations
       compute_final_terms(  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                             y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                             y_sign_chunk,
                             y_m_y_sign_x_u,
                             phi_Z_recip,
                             phi_Bound_Z,
                             L_Omega_recip_double,
                             n_tests);
       
     //   for (int i = 0; i < n_tests; i++) {  
     //     int t = n_tests - (i+1) ;
     //     prop_rowwise_prod_temp.col(t).array()   =   prob.block(0, t + 0, chunk_size, i + 1).rowwise().prod().array() ;
     //     prop_recip_rowwise_prod_temp.col(t).array()   =   prob_recip.block(0, t + 0, chunk_size, i + 1).rowwise().prod().array() ;
     //   }
     // 
     //   prop_rowwise_prod_temp_all.array() =  prob.rowwise().prod().array()  ;
     // 
     //   if (n_class > 1) {  /// only compute if using latent class MVP
     //       for (int i = 0; i < n_tests; i++) {  
     //         int t = n_tests - (i + 1) ;
     //         common_grad_term_1.col(t) =   (  ( prev_double * prob_n_recip.array() ) * (    prop_rowwise_prod_temp_all.array() * prop_recip_rowwise_prod_temp.col(t).array()  ).array() )  ;
     //       }
     //   } else { 
     //         common_grad_term_1.setOnes();
     //   }
     // 
     // for (int t = 0; t < n_tests; t++) {  
     //     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()  =    y_sign_chunk.col(t).array() *    phi_Bound_Z.col(t).array()  *   L_Omega_recip_double(t, t) ; 
     //     y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array()  = y_m_y_sign_x_u.col(t).array() * phi_Z_recip.col(t).array()  *  phi_Bound_Z.col(t).array() * L_Omega_recip_double(t, t);
     // }
     //   
       
       
}
 
 
 
 
 
 
 
 
 
 




// Gradient computation function template (no need to have any template parameters as not very modular e.g. only double's)
ALWAYS_INLINE void fn_MVP_compute_nuisance_grad_v2(     Eigen::Ref<Eigen::Matrix<double, -1, -1>>   u_grad_array_CM_chunk,
                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>   phi_Z_recip,
                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>   common_grad_term_1,
                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>   L_Omega_double,
                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>   prob,
                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>   prob_recip,
                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>   prop_rowwise_prod_temp,
                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>   y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                        Eigen::Ref<Eigen::Matrix<double, -1, -1>>   z_grad_term,
                                                        Eigen::Ref<Eigen::Matrix<double, -1, -1>>   grad_prob,
                                                        Eigen::Ref<Eigen::Matrix<double, -1, 1>>    prod_container,
                                                        Eigen::Ref<Eigen::Matrix<double, -1, 1>>    derivs_chain_container_vec,
                                                        const Model_fn_args_struct &Model_args_as_cpp_struct
) {

        const int  n_class = Model_args_as_cpp_struct.Model_args_ints(1);
        const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
        const std::string vect_type = Model_args_as_cpp_struct.Model_args_strings(0);

        const int chunk_size = u_grad_array_CM_chunk.rows();
        const int n_tests = u_grad_array_CM_chunk.cols();
        
        // if (n_class == 1) {  /// i.e. if using standard MVP
        //   common_grad_term_1.setOnes();
        //   prop_rowwise_prod_temp.setOnes();
        // }
      
       {
          
            int t = n_tests - 1;  ///// then second-to-last term (test T - 1)
          
            if (n_class != 1) {  //// if latent class
              u_grad_array_CM_chunk.col(n_tests - 2).array()  =  (  common_grad_term_1.col(t).array()  * (y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()  * 
                                                                    L_Omega_double(t,t - 1) * ( phi_Z_recip.col(t-1).array() )  *  prob.col(t-1).array()) ).array()  ;
            } else {  ////  if standard MVP
              u_grad_array_CM_chunk.col(n_tests - 2).array() +=     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array() * L_Omega_double(t,t - 1) * 
                                                                    phi_Z_recip.col(t-1).array() * prob.col(t-1).array()  *  (prob_recip.col(t).array()).array() ;
            }
          
       }
      
       { ///// then third-to-last term (test T - 2)
         
             int t = n_tests - 2;
             
             z_grad_term.col(0) = ( phi_Z_recip.col(t-1).array())  *  prob.col(t-1).array() ;
             grad_prob.col(0) =        y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   *  L_Omega_double(t, t - 1) *   z_grad_term.col(0).array() ; // lp(T-1) - part 2;
             z_grad_term.col(1).array()  =      L_Omega_double(t,t-1) *   z_grad_term.col(0).array() *  y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array() ;
             grad_prob.col(1)  =         (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t+1).array()   ) * 
                                           (  z_grad_term.col(0).array() *  L_Omega_double(t + 1,t - 1)  -   z_grad_term.col(1).array()  * L_Omega_double(t+1,t)) ;
             
             if (n_class != 1) {  //// if latent class
                 u_grad_array_CM_chunk.col(n_tests - 3).array()  =   ( common_grad_term_1.col(t).array()   * 
                                                                     ( grad_prob.col(1).array() *  prob.col(t).array()  +      grad_prob.col(0).array() *   prob.col(t+1).array()  )  )   ;
             } else {  //// if standard MVP
                 u_grad_array_CM_chunk.col(n_tests - 3).array()  = (  (  grad_prob.col(1).array() *  prob_recip.col(t + 1).array()  +   grad_prob.col(0).array() * prob_recip.col(t).array()  )  )   ;
             }
             
       }
      
      for (int i = 1; i < n_tests - 2; i++)  {   // then rest of terms (i = 2)
      
            int t = n_tests - (i+2);
          
            grad_prob.setZero(); // .array()   = 0.0;
            z_grad_term.setZero(); // .array() = 0.0;
          
            z_grad_term.col(0) = ( phi_Z_recip.col(t-1).array())  *  prob.col(t-1).array() ;
            grad_prob.col(0) =        y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   *  L_Omega_double(t, t - 1) *   z_grad_term.col(0).array() ; 
          
              prod_container.array() =  L_Omega_double(t, t - 1) *   z_grad_term.col(0).array() ;
          
              for (int ii = 1; ii < i + 2; ii++) {
                z_grad_term.col(ii).array() =    y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii - 1).array() *   prod_container.array() ;   
                prod_container.array() =    z_grad_term.col(0).array() *  L_Omega_double(t + ii, t - 1)  -    ( z_grad_term.block(0, 1, chunk_size, ii)  * L_Omega_double.row(t + ii).segment(t + 0, ii).transpose() ).array()  ;
                grad_prob.col(ii)  =           y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + ii).array()    * prod_container.array() ;   
              }
          
          
          if (n_class != 1) {  //// if latent class
                
                  derivs_chain_container_vec.setZero(); //  = 0.0;
                  for (int ii = 0; ii < i + 2; ii++) {
                    derivs_chain_container_vec.array()  +=  ( grad_prob.col(ii).array()    * (       prop_rowwise_prod_temp.col(t).array() * prob_recip.col(t + ii).array()  ).array() ).array()  ;
                  }
                  u_grad_array_CM_chunk.col(n_tests - (i + 3)).array()    =   (  ( (   common_grad_term_1.col(t).array()   *  derivs_chain_container_vec.array() ) ).array()  ).array() ;
                
          }  else {  //// if standard MVP
            
                  derivs_chain_container_vec.array() = 0.0;
                  for (int ii = 0; ii < i + 2; ii++) {
                    derivs_chain_container_vec.array()   +=  grad_prob.col(ii).array()    *   prob_recip.col(t + ii).array();
                  }
                  u_grad_array_CM_chunk.col(n_tests - (i + 3)).array()    +=     derivs_chain_container_vec.array()  ;
            
          }
      
      }

// no "return" unlike std::function since it modifies u_grad_array_CM_chunk by reference!

}




 

 

 
 
 
 
 
  

// Gradient computation function template (no need to have any template parameters as not very modular e.g. only double's)
ALWAYS_INLINE void fn_MVP_compute_coefficients_grad_v2(   const int c, // latent class (0 for standard MVP)
                                                          Eigen::Matrix<double, -1, -1> &coefficients_array,
                                                          const std::vector<Eigen::Matrix<double, -1, -1>> &X,
                                                          const Eigen::Ref<const Eigen::Matrix<int, -1, 1>> n_covariates_per_outcome_vec,
                                                          std::vector<Eigen::Matrix<double, -1, -1>> &beta_grad_array_for_each_n,
                                                          const int &chunk_counter,
                                                          const int &n_covariates_max,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> common_grad_term_1,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob_recip,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prop_rowwise_prod_temp,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                          const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>>  z_grad_term,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, -1>>  grad_prob,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, 1>>   prod_container,
                                                          Eigen::Ref<Eigen::Matrix<double, -1, 1>>   derivs_chain_container_vec,
                                                          const bool compute_final_scalar_grad,
                                                          const Model_fn_args_struct &Model_args_as_cpp_struct
) {

      const int chunk_size = common_grad_term_1.rows();
      const int n_tests = common_grad_term_1.cols();

      const int  n_class = Model_args_as_cpp_struct.Model_args_ints(1);
      const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
      const std::string vect_type = Model_args_as_cpp_struct.Model_args_strings(0);

      // const std::vector<Eigen::Matrix<double, -1, -1>>  &X =  Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0][c];
      // const Eigen::Matrix<int, -1, 1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0].row(c).transpose();

      const double a = 0.07056;
      const double b = 1.5976;
      const double a_times_3 = 3.0 * 0.07056;
      const double sqrt_2_pi_recip =   1.0 / stan::math::sqrt(2.0 * M_PI) ;

      // if (n_class == 1) {  /// i.e. if using standard MVP
      //     common_grad_term_1.setOnes();
      //     prop_rowwise_prod_temp.setOnes();
      // }

      if (n_covariates_max == 1) { /// only possible for latent class LC-MVP as standard-MVP always has covariates!!

              {
                int t = n_tests - 1;
                beta_grad_array_for_each_n[0].col(t) =    (  common_grad_term_1.col(t).array()  *   (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()   ) );
                if (compute_final_scalar_grad)  coefficients_array(0, t)   +=       beta_grad_array_for_each_n[0].col(t).sum();
              }

              {   ///// then second-to-last term (test T - 1)
                int t = n_tests - 2;

                grad_prob.col(0) =           y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()    ;
                z_grad_term.col(0)   =     -      y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array()         ;
                grad_prob.col(1)  =       (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t  + 1).array()    )   * (   L_Omega_double(t + 1, t) * z_grad_term.col(0).array() ) ;
                beta_grad_array_for_each_n[0].col(t) =   (  common_grad_term_1.col(t).array()   * ( grad_prob.col(1).array() *  prob.col(t).array() +         grad_prob.col(0).array() *   prob.col(t  + 1).array() ) );
                if (compute_final_scalar_grad)  coefficients_array(0, t)   +=       beta_grad_array_for_each_n[0].col(t).sum();
              }

              for (int i = 1; i < n_tests - 1; i++) {     // then rest of terms

                    int t = n_tests - (i + 2) ;

                    grad_prob.col(0)  =     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()    ;
                    z_grad_term.col(0)  =        -     y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array()       ;
                    grad_prob.col(1) =        y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 1).array()    *    (   L_Omega_double(t + 1,t) *   z_grad_term.col(0).array() ) ;

                    for (int ii = 1; ii < i+1; ii++) {    // rest of components
                      if (ii == 1)  prod_container  = (    z_grad_term.leftCols(ii)  *   L_Omega_double.row( t + (ii - 1) + 1).segment(t + 0, ii + 0).transpose()  );
                      z_grad_term.col(ii)  =        -     y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii).array()    *   prod_container.array();
                      prod_container  = (    z_grad_term.leftCols(ii + 1)  *   L_Omega_double.row( t + (ii) + 1).segment(t + 0, ii + 1).transpose()  );
                      grad_prob.col(ii + 1) =      (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + ii + 1).array()  ).array()  *  prod_container.array();
                    }

                    {
                      derivs_chain_container_vec.setZero();// .array() = 0.0;
                      for (int ii = 0; ii < i + 2; ii++) {
                        derivs_chain_container_vec.array() +=  ( grad_prob.col(ii).array()  * (        prop_rowwise_prod_temp.col(t).array() *   prob_recip.col(t + ii).array()  ).array() ).array() ;
                      }
                      beta_grad_array_for_each_n[0].col(t) =    common_grad_term_1.col(t).array()   *  derivs_chain_container_vec.array() ;
                      if (compute_final_scalar_grad)  coefficients_array(0, t)   +=       beta_grad_array_for_each_n[0].col(t).sum();
                    }

              }


      } else {

                {

                    int t = n_tests - 1;

                    for (int k = 0; k < n_covariates_per_outcome_vec(t); ++k)  {

                          if (n_class != 1) {  ///// if latent class
                            if (compute_final_scalar_grad)  coefficients_array(k, t) +=      (  common_grad_term_1.col(t).array()  *   (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()  *
                                                                          X[t].block(chunk_size * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(t)).col(k).array().cast<double>()    )).sum();
                          } else {   ////// if standard MVP
                                Eigen::Matrix<double, -1, 1> y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip_col_t =    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array() *
                                                                                                                                        X[t].block(chunk_size * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(t)).col(k).array().cast<double>() ;
                            beta_grad_array_for_each_n[k].col(t) = y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip_col_t.array() * prob_recip.col(t).array();
                            if (compute_final_scalar_grad)  coefficients_array(k, t) +=   beta_grad_array_for_each_n[k].col(t).sum();
                          }

                    }

                }

                {   ///// then second-to-last term (test T - 1)

                    int t = n_tests - 2;

                    for (int k = 0; k < n_covariates_per_outcome_vec(t); ++k)  {

                          Eigen::Matrix<double, -1, 1>  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip_col_t  =     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array() * X[t].matrix().block(chunk_size * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(t)).col(k).array().cast<double>() ;
                          Eigen::Matrix<double, -1, 1>  y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip_for_coeffs_col_t  =   X[t].matrix().block(chunk_size * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(t)).col(k).array().cast<double>()  * y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array()  ;
                          grad_prob.col(0).array() =           y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip_col_t.array();
                          z_grad_term.col(0).array()   =     - y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip_for_coeffs_col_t.array();
                          grad_prob.col(1).array()  =       (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 1).array()    )   * (   L_Omega_double(t + 1, t) *      z_grad_term.col(0).array() ).array() ;

                          if (n_class != 1) {  ///// if latent class
                              beta_grad_array_for_each_n[k].col(t) =   (  common_grad_term_1.col(t).array()   * ( grad_prob.col(1).array() *  prob.col(t).array() + grad_prob.col(0).array() *   prob.col(t  + 1).array() ) );
                              if (compute_final_scalar_grad)  coefficients_array(k, t) +=   beta_grad_array_for_each_n[k].col(t).sum();
                          } else {  //// if standard MVP
                              beta_grad_array_for_each_n[k].col(t) =    (  grad_prob.col(1).array() *  prob_recip.col(t + 1).array()   +   grad_prob.col(0).array() *   prob_recip.col(t + 0).array()      );
                              if (compute_final_scalar_grad)  coefficients_array(k, t) +=   beta_grad_array_for_each_n[k].col(t).sum();
                          }

                    }

                }

                for (int i = 1; i < n_tests - 1; i++) {     // then rest of terms

                      int t = n_tests - (i + 2) ;

                      for (int k = 0; k < n_covariates_per_outcome_vec(t); ++k)  {

                            Eigen::Matrix<double, -1, 1>  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip_col_t  =     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array() *  X[t].matrix().block(chunk_size * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(t)).col(k).array().cast<double>() ;
                            Eigen::Matrix<double, -1, 1>  y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip_for_coeffs_col_t  =   X[t].matrix().block(chunk_size * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(t)).col(k).array().cast<double>()   * y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array()  ;
                            grad_prob.col(0).array()  =     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip_col_t.array();
                            z_grad_term.col(0).array()  =        - y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip_for_coeffs_col_t.array()   ;
                            grad_prob.col(1).array() =        y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 1).array()    *    (   L_Omega_double(t + 1,t) *   z_grad_term.col(0).array() ) ;

                            for (int ii = 1; ii < i + 1; ii++) {    // rest of components

                                  if (ii == 1)  prod_container  = (    z_grad_term.leftCols(ii)  *   L_Omega_double.row( t + (ii - 1) + 1).segment(t + 0, ii + 0).transpose()  );
                                  z_grad_term.col(ii)  =        -     y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii).array()    *   prod_container.array();
                                  prod_container  = (    z_grad_term.leftCols(ii + 1)  *   L_Omega_double.row( t + (ii) + 1).segment(t + 0, ii + 1).transpose()  );
                                  grad_prob.col(ii + 1) =      (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + ii + 1).array()  ).array()     *  prod_container.array();

                            }

                           if (n_class == 1) {  //// if standard MVP

                                  derivs_chain_container_vec.array() = 0.0;
                                  for (int ii = 0; ii < i + 2; ii++) {
                                    derivs_chain_container_vec.array() +=   grad_prob.col(ii).array()  *   prob_recip.col(t + ii).array()  ;
                                  }
                                  beta_grad_array_for_each_n[k].col(t) =         derivs_chain_container_vec;
                                  if (compute_final_scalar_grad)  coefficients_array(k, t) +=   beta_grad_array_for_each_n[k].col(t).sum();

                           } else {  //// if latent class

                                 derivs_chain_container_vec.array() = 0.0;
                                 for (int ii = 0; ii < i + 2; ii++) {
                                   derivs_chain_container_vec.array() +=  ( grad_prob.col(ii).array()  * (        prop_rowwise_prod_temp.col(t).array() *   prob_recip.col(t + ii).array()  ).array() ).array() ;
                                 }
                                 beta_grad_array_for_each_n[k].col(t) =         (   common_grad_term_1.col(t).array()   *  derivs_chain_container_vec.array() );
                                 if (compute_final_scalar_grad)  coefficients_array(k, t) +=   beta_grad_array_for_each_n[k].col(t).sum();

                           }

                      }

                }

      }



}












 
 
// Gradient computation function template (no need to have any template parameters as not very modular e.g. only double's)
ALWAYS_INLINE void fn_MVP_compute_coefficients_grad_v3(      const int c, // latent class (0 for standard MVP)
                                                             Eigen::Matrix<double, -1, -1> &coefficients_array,
                                                             const std::vector<Eigen::Matrix<double, -1, -1>> &X,
                                                             const Eigen::Ref<const Eigen::Matrix<int, -1, 1>> n_covariates_per_outcome_vec,
                                                             // std::vector<Eigen::Matrix<double, -1, -1>> &beta_grad_array_for_each_n,
                                                             // const int &chunk_counter,
                                                             const int row_start,
                                                             const int &n_covariates_max,
                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> common_grad_term_1,
                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob,
                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob_recip,
                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prop_rowwise_prod_temp,
                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                             Eigen::Ref<Eigen::Matrix<double, -1, -1>>  z_grad_term,
                                                             Eigen::Ref<Eigen::Matrix<double, -1, -1>>  grad_prob,
                                                             Eigen::Ref<Eigen::Matrix<double, -1, 1>>   prod_container,
                                                             Eigen::Ref<Eigen::Matrix<double, -1, 1>>   derivs_chain_container_vec,
                                                             const bool compute_final_scalar_grad,
                                                             const Model_fn_args_struct &Model_args_as_cpp_struct
) {
   
   const int chunk_size = common_grad_term_1.rows();
   const int n_tests = common_grad_term_1.cols();
   
   const int  n_class = Model_args_as_cpp_struct.Model_args_ints(1);
   const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
   const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
   
   // const std::vector<Eigen::Matrix<double, -1, -1>>  &X =  Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0][c];
   // const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0]; // .row(c).transpose();
   
   const double a = 0.07056;
   const double b = 1.5976;
   const double a_times_3 = 3.0 * 0.07056;
   const double sqrt_2_pi_recip =   1.0 / stan::math::sqrt(2.0 * M_PI) ;
   
   // if (n_class == 1) {  /// i.e. if using standard MVP
   //     common_grad_term_1.setOnes();
   //     prop_rowwise_prod_temp.setOnes();
   // } 
   
   if (n_covariates_max == 1 && n_class > 1) { // Intercept-only mixture shortcut; single-class models use the general derivative below.
     
             {
               int t = n_tests - 1;
               if (compute_final_scalar_grad) coefficients_array(0, t) += ( common_grad_term_1.col(t).array() * ( y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array() ) ).sum();
             }
             
             // {   ///// then second-to-last term (test T - 1)
             //   int t = n_tests - 2;
             //   
             //   grad_prob.col(0) = - y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array();
             //   grad_prob.col(1) = (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t  + 1).array() ) * ( L_Omega_double(t + 1, t) * z_grad_term.col(0).array() );
             //   if (compute_final_scalar_grad) coefficients_array(0, t) += ( common_grad_term_1.col(t).array() * ( grad_prob.col(1).array() * prob.col(t).array() + grad_prob.col(0).array() * prob.col(t  + 1).array() ) ).sum();
             // }
     
             {   ///// then second-to-last term (test T - 1)
               int t = n_tests - 2;
               
               grad_prob.col(0) = y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array();
               z_grad_term.col(0) = - y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array();
               grad_prob.col(1) = ( y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t+1).array() ) * ( L_Omega_double(t+1, t) * z_grad_term.col(0).array() );
               if (compute_final_scalar_grad) coefficients_array(0, t) += ( common_grad_term_1.col(t).array() * ( grad_prob.col(1).array() * prob.col(t).array() + grad_prob.col(0).array() * prob.col(t+1).array() ) ).sum();
             }
             
             for (int i = 1; i < n_tests - 1; i++) {     // then rest of terms
               
               int t = n_tests - (i + 2) ;
               
               grad_prob.col(0) = y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array();
               z_grad_term.col(0) = - y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array();
               grad_prob.col(1) = y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 1).array() * ( L_Omega_double(t + 1,t) * z_grad_term.col(0).array() ) ;
               
               for (int ii = 1; ii < i+1; ii++) {    // rest of components
                 if (ii == 1)  prod_container  = ( z_grad_term.leftCols(ii) * L_Omega_double.row( t + (ii - 1) + 1).segment(t + 0, ii + 0).transpose() );
                 z_grad_term.col(ii)  = - y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii).array() * prod_container.array();
                 prod_container  = ( z_grad_term.leftCols(ii + 1)  *   L_Omega_double.row( t + (ii) + 1).segment(t + 0, ii + 1).transpose() );
                 grad_prob.col(ii + 1) = ( y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + ii + 1).array() ).array() * prod_container.array();
               }
               
               {
                 derivs_chain_container_vec.setZero();// .array() = 0.0;
                 for (int ii = 0; ii < i + 2; ii++) {
                   derivs_chain_container_vec.array() += ( grad_prob.col(ii).array() * ( prop_rowwise_prod_temp.col(t).array() * prob_recip.col(t + ii).array() ).array() ).array() ;
                 }
                 if (compute_final_scalar_grad) coefficients_array(0, t) += ( common_grad_term_1.col(t).array() * derivs_chain_container_vec.array() ).sum();
               }
               
             }
     
   } else {
     
     {
       
       int t = n_tests - 1;
       
       for (int k = 0; k < n_covariates_per_outcome_vec(t); ++k)  {
         
         if (n_class != 1) {  ///// if latent class
           if (compute_final_scalar_grad)  coefficients_array(k, t) += ( common_grad_term_1.col(t).array()  * ( y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array()  * 
                                                                         X[t].block(row_start, 0, chunk_size, n_covariates_per_outcome_vec(t)).col(k).array().cast<double>() )).sum();
         } else {   ////// if standard MVP
           Eigen::Matrix<double, -1, 1> y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip_col_t = y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array() * 
                                                                                                    // X[t].block(chunk_size * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(t)).col(k).array().cast<double>();
                                                                                                    X[t].block(row_start, 0, chunk_size, n_covariates_per_outcome_vec(t)).col(k).array().cast<double>();
           if (compute_final_scalar_grad)  coefficients_array(k, t) += ( y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip_col_t.array() * prob_recip.col(t).array() ).sum();
         }
         
       }
       
     }
     
     {   ///// then second-to-last term (test T - 1)
       
       int t = n_tests - 2;
       
       for (int k = 0; k < n_covariates_per_outcome_vec(t); ++k)  {
         
         Eigen::Matrix<double, -1, 1>  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip_col_t  =     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array() *
                                                                                                        // X[t].matrix().block(chunk_size * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(t)).col(k).array().cast<double>() ;
                                                                                                        X[t].matrix().block(row_start, 0, chunk_size, n_covariates_per_outcome_vec(t)).col(k).array().cast<double>() ;
         Eigen::Matrix<double, -1, 1>  y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip_for_coeffs_col_t = X[t].matrix().block(row_start, 0, chunk_size, n_covariates_per_outcome_vec(t)).col(k).array().cast<double>() * 
                                                                                                                                     y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array();
         grad_prob.col(0).array() =           y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip_col_t.array();
         z_grad_term.col(0).array()   =     - y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip_for_coeffs_col_t.array();
         grad_prob.col(1).array()  =       (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 1).array()    )   * (   L_Omega_double(t + 1, t) *      z_grad_term.col(0).array() ).array() ;
         
         
         
         if (n_class != 1) {  ///// if latent class
           if (compute_final_scalar_grad)  coefficients_array(k, t) +=   (  common_grad_term_1.col(t).array()   * ( grad_prob.col(1).array() *  prob.col(t).array() + grad_prob.col(0).array() *   prob.col(t  + 1).array() ) ).sum();
         } else {  //// if standard MVP
           if (compute_final_scalar_grad)  coefficients_array(k, t) +=   (  grad_prob.col(1).array() *  prob_recip.col(t + 1).array()   +   grad_prob.col(0).array() *   prob_recip.col(t + 0).array()      ).sum();
         }
         
       }
       
     }
     
     for (int i = 1; i < n_tests - 1; i++) {     // then rest of terms
       
       int t = n_tests - (i + 2) ;
       
       for (int k = 0; k < n_covariates_per_outcome_vec(t); ++k)  {
         
         Eigen::Matrix<double, -1, 1>  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip_col_t = y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t).array() * 
                                                                                                   X[t].matrix().block(row_start, 0, chunk_size, n_covariates_per_outcome_vec(t)).col(k).array().cast<double>() ;
         Eigen::Matrix<double, -1, 1>  y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip_for_coeffs_col_t = X[t].matrix().block(row_start, 0, chunk_size, n_covariates_per_outcome_vec(t)).col(k).array().cast<double>() 
                                                                                                                                     * y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t).array()  ;
         grad_prob.col(0).array()  = y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip_col_t.array();
         z_grad_term.col(0).array() = - y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip_for_coeffs_col_t.array()   ;
         grad_prob.col(1).array() = y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + 1).array() * (   L_Omega_double(t + 1, t) *   z_grad_term.col(0).array() ) ;
         
         for (int ii = 1; ii < i + 1; ii++) {    // rest of components
           
               if (ii == 1)  prod_container = ( z_grad_term.leftCols(ii) * L_Omega_double.row( t + (ii - 1) + 1).segment(t + 0, ii + 0).transpose()  );
               z_grad_term.col(ii) = - y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t + ii).array() * prod_container.array();
               prod_container  = ( z_grad_term.leftCols(ii + 1)  *   L_Omega_double.row( t + (ii) + 1).segment(t + 0, ii + 1).transpose() );
               grad_prob.col(ii + 1) = ( y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t + ii + 1).array() ).array() * prod_container.array();
           
         }
         
         if (n_class == 1) {  //// if standard MVP
           
               derivs_chain_container_vec.array() = 0.0;
               for (int ii = 0; ii < i + 2; ii++) {
                 derivs_chain_container_vec.array() +=   grad_prob.col(ii).array()  *   prob_recip.col(t + ii).array()  ;
               }
               // beta_grad_array_for_each_n[k].col(t) =         derivs_chain_container_vec;
               if (compute_final_scalar_grad)  coefficients_array(k, t) +=  derivs_chain_container_vec.sum();
           
         } else {  //// if latent class
           
               derivs_chain_container_vec.array() = 0.0;
               for (int ii = 0; ii < i + 2; ii++) {
                 derivs_chain_container_vec.array() +=  ( grad_prob.col(ii).array()  * (        prop_rowwise_prod_temp.col(t).array() *   prob_recip.col(t + ii).array()  ).array() ).array() ;
               }
               if (compute_final_scalar_grad)  coefficients_array(k, t) +=   (   common_grad_term_1.col(t).array()   *  derivs_chain_container_vec.array() ).sum();
               
         }
         
       }
       
     }
     
   }
   
   
   
}
 
 
 
 
 
 
 
 
 
 
 
 
 
 





// Gradient computation function template (no need to have any template parameters as not very modular e.g. only double's)
ALWAYS_INLINE void fn_MVP_compute_L_Omega_grad_v2(   Eigen::Ref<Eigen::Matrix<double, -1, -1>>   U_Omega_grad_array,
                                              std::vector<Eigen::Matrix<double, -1, -1>> &Omega_grad_array_for_each_n,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  common_grad_term_1,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob_recip,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  prop_rowwise_prod_temp,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                              const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>>   z_grad_term,
                                              Eigen::Ref<Eigen::Matrix<double, -1, -1>>   grad_prob,
                                              Eigen::Ref<Eigen::Matrix<double, -1, 1>>    prod_container,
                                              Eigen::Ref<Eigen::Matrix<double, -1, 1>>    derivs_chain_container_vec,
                                              const bool compute_final_scalar_grad,
                                              const Model_fn_args_struct &Model_args_as_cpp_struct

) {

  const int  n_class = Model_args_as_cpp_struct.Model_args_ints(1);
  const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
  const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);

  // const int chunk_size = common_grad_term_1.rows();
  const int n_tests = common_grad_term_1.cols();

  // if (n_class == 1) {  /// i.e. if using standard MVP
  //   common_grad_term_1.setOnes();
  //   prop_rowwise_prod_temp.setOnes();
  // }

  {

    z_grad_term.setZero(); // .array() = 0.0 ;
    grad_prob.setZero();// .array() = 0.0;

    {
      ///////////////////////// deriv of diagonal elements (not needed if using the "standard" or "Stan" Cholesky parameterisation of Omega)
      //////// w.r.t last diagonal first
      {

        int  t1 = n_tests - 1;

        if (n_class == 1)  {
          Omega_grad_array_for_each_n[t1].col(t1) =    prob_recip.col(t1).array()   *   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()  *   Bound_Z.col(t1).array();
          if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) +=   Omega_grad_array_for_each_n[t1].col(t1).sum();
        }  else  {
          Omega_grad_array_for_each_n[t1].col(t1) =  (   common_grad_term_1.col(t1).array()   *   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()  *   Bound_Z.col(t1).array()       );
          if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) +=   Omega_grad_array_for_each_n[t1].col(t1).sum();
        }

      }

      //////// then w.r.t the second-to-last diagonal
      int  t1 = n_tests - 2;
      grad_prob.col(0).array()  =       (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array() *     Bound_Z.col(t1).array() ).array()    ;     // correct  (standard form)
      z_grad_term.col(0).array()  =    (  (   -      y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1).array()  )    *  Bound_Z.col(t1).array()  ).array()  ;  // correct
      prod_container.array()  =   (  L_Omega_double(t1 + 1, t1)    *   z_grad_term.col(0).array()   ).array() ; // sequence
      grad_prob.col(1).array()  =  (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + 1).array()   *          prod_container.array()     ).array()    ;    // correct   (standard form)

      if (n_class == 1) {
          Omega_grad_array_for_each_n[t1].col(t1) =    (  prob_recip.col(t1).array()  *   grad_prob.col(0).array()  +    prob_recip.col(t1 + 1).array()  *      grad_prob.col(1).array()   );
          if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) +=   Omega_grad_array_for_each_n[t1].col(t1).sum();
      }  else   {
          Omega_grad_array_for_each_n[t1].col(t1) =    ((common_grad_term_1.col(t1).array() )  *  (  prob.col(t1 + 1).array() * grad_prob.col(0).array()  +  prob.col(t1).array()  * grad_prob.col(1).array()   )  );
          if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) +=   Omega_grad_array_for_each_n[t1].col(t1).sum();
      }

    }
    ////////// then w.r.t the third-to-last diagonal .... etc
    {

      for (int i = 3; i < n_tests + 1; i++) {

        int  t1 = n_tests - i;

        //////// 1st component
        grad_prob.col(0).array()  =   (  (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array() )  *  ( Bound_Z.col(t1).array()   ).array() ).array() ; // correct  (standard form)
        z_grad_term.col(0).array()  =   (   ( -     y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1).array()  )  *  Bound_Z.col(t1).array()   ).array()    ;   // correct  (standard form)
        prod_container.array() =    ( L_Omega_double(t1 + 1, t1)   * z_grad_term.col(0).array() ).array()   ; // correct  (standard form)
        grad_prob.col(1).array()  =   (  (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + 1).array()  )   *  (    prod_container.array() ).array() ).array()  ; // correct  (standard form)

        for (int ii = 1; ii < i - 1; ii++) {
          z_grad_term.col(ii).array()  =    (   (-      y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1 + ii).array()  )    *   prod_container.array()  ).array()  ;   // correct  (standard form)       // grad_z term
          prod_container.matrix() =   (  (  L_Omega_double.row(t1 + ii + 1).segment(t1, ii + 1) *   z_grad_term.leftCols(ii + 1).transpose() ).transpose().matrix()  ).array() ; // correct  (standard form)
          grad_prob.col(ii + 1).array()  =   (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + ii + 1).array()    *   prod_container.array() ).array()  ;  // correct  (standard form)     //    grad_prob term
        }

       if (n_class != 1) {  //// if latent class

              derivs_chain_container_vec.setZero(); // .array() = 0.0;
              for (int iii = 0; iii <  i; iii++) {
                derivs_chain_container_vec.array()  +=    ( grad_prob.col(iii).array()  * ( prop_rowwise_prod_temp.col(t1).array()    *   prob_recip.col(t1 + iii).array()  ).array() ).array()   ;  // correct  (standard form)
              }
              Omega_grad_array_for_each_n[t1].col(t1) =     (   common_grad_term_1.col(t1).array()   * derivs_chain_container_vec.array() );
              if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) +=   Omega_grad_array_for_each_n[t1].col(t1).sum();

       } else { //// if standard MVP

             derivs_chain_container_vec.setZero(); // .array() = 0.0;
             for (int iii = 0; iii <  i; iii++) {
               derivs_chain_container_vec.array()  +=    grad_prob.col(iii).array()  *  prob_recip.col(t1 + iii).array()    ;
             }
             Omega_grad_array_for_each_n[t1].col(t1) =       derivs_chain_container_vec;
             if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) +=   Omega_grad_array_for_each_n[t1].col(t1).sum();

       }

      }

    }

  }

  {

        {
          int t1_dash = 0;  // t1 = n_tests - 1
          int t1 = n_tests - (t1_dash + 1); //  starts at n_tests - 1;  // if t1_dash = 0 -> t1 = T - 1
          int t2 = n_tests - (t1_dash + 2); //  starts at n_tests - 2;

          if (n_class == 1) {
            Omega_grad_array_for_each_n[t1].col(t2) =  (prob_recip.col(t1).array()    *    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()  *  Z_std_norm.col(t2).array()   );
            if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=   Omega_grad_array_for_each_n[t1].col(t2).sum();
          }  else {
            Omega_grad_array_for_each_n[t1].col(t2) =   (common_grad_term_1.col(t1).array()      *    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()   *    Z_std_norm.col(t2).array()   );
            if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=   Omega_grad_array_for_each_n[t1].col(t2).sum();
          }

          if (t1 > 1) { // starts at  L_{T, T-2}

              t2 =   n_tests - (t1_dash + 3); // starts at n_tests - 3;

              if (n_class == 1) {
                Omega_grad_array_for_each_n[t1].col(t2) =  (prob_recip.col(t1).array()   *    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()   *    Z_std_norm.col(t2).array()   );
                if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=   Omega_grad_array_for_each_n[t1].col(t2).sum();
              } else  {
                Omega_grad_array_for_each_n[t1].col(t2) =  (common_grad_term_1.col(t1).array()  * y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()  *    Z_std_norm.col(t2).array()   );
                if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=   Omega_grad_array_for_each_n[t1].col(t2).sum();
              }

          }

          if (t1 > 2) { // starts at  L_{T, T-3}

              for (int t2_dash = 3; t2_dash < n_tests; t2_dash++ ) { // t2 < t1

                t2 = n_tests - (t1_dash + t2_dash + 1); // starts at T - 4

                if (t2 < n_tests - 1) {
                  if (n_class == 1)  {
                    Omega_grad_array_for_each_n[t1].col(t2) =   (prob_recip.col(t1).array() * y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array() * Z_std_norm.col(t2).array());
                    if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=   Omega_grad_array_for_each_n[t1].col(t1).sum();
                  } else {
                    Omega_grad_array_for_each_n[t1].col(t2) =  (common_grad_term_1.col(t1).array() * y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array() * Z_std_norm.col(t2).array());
                    if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=   Omega_grad_array_for_each_n[t1].col(t2).sum();
                  }
                }

              }

          }

        }

  }

  {

    z_grad_term.setZero(); // .array() = 0.0 ;
    grad_prob.setZero(); // .array() = 0.0;


    /////////////////// then rest of rows (second-to-last row, then third-to-last row, .... , then first row)
    for (int t1_dash = 1; t1_dash <  n_tests - 1;  t1_dash++) {
      int  t1 = n_tests - (t1_dash + 1);

      for (int t2_dash = t1_dash + 1; t2_dash <  n_tests;  t2_dash++) {
        int t2 = n_tests - (t2_dash + 1); // starts at t1 - 1, then t1 - 2, up to 0

        {
          prod_container.array()  =  Z_std_norm.col(t2).array() ; // block(0, t2, index_size, t1 - t2) * deriv_L_t1.head(t1 - t2) ;
          grad_prob.col(0).array()  =      (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()   *           prod_container.array()  ).array()  ;
          z_grad_term.col(0).array()  =       (                 y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1).array()     *   (   -    prod_container.array()     )    ).array()   ;

          if (t1_dash > 0) {
            for (int t1_dash_dash = 1; t1_dash_dash <  t1_dash + 1;  t1_dash_dash++) {
              if (t1_dash_dash > 1)  z_grad_term.col(t1_dash_dash - 1).array()   =  (   y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1 + t1_dash_dash - 1).array()   *  (- prod_container.array())  ).array() ;
              prod_container.array()  =            (   z_grad_term.leftCols(t1_dash_dash) *   L_Omega_double.row(t1 + t1_dash_dash).segment(t1, t1_dash_dash).transpose()   ).array() ;
              grad_prob.col(t1_dash_dash).array()  =          (     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + t1_dash_dash).array()    *      prod_container.array() ).array()   ;
            }
          }

          if (n_class == 1)   {  ////// if standard MVP

                derivs_chain_container_vec.setZero(); // .array() = 0.0;
                for (int ii = 0; ii <  t1_dash + 1; ii++) {
                  derivs_chain_container_vec.array() +=   grad_prob.col(ii).array()  *     prob_recip.col(t1 + ii).array()   ; // correct i think
                }
                Omega_grad_array_for_each_n[t1].col(t2) =         derivs_chain_container_vec;
                if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=   Omega_grad_array_for_each_n[t1].col(t2).sum();

          } else {  //// if latent class

                derivs_chain_container_vec.setZero(); // .array() = 0.0;
                for (int ii = 0; ii <  t1_dash + 1; ii++) {
                  derivs_chain_container_vec.array() += ( grad_prob.col(ii).array()  * (   prop_rowwise_prod_temp.col(t1).array()    *  prob_recip.col(t1 + ii).array()  ).array() ).array() ; // correct i think
                }
                Omega_grad_array_for_each_n[t1].col(t2) =      (   common_grad_term_1.col(t1).array()   * derivs_chain_container_vec.array() );
                if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=   Omega_grad_array_for_each_n[t1].col(t2).sum();

          }

        }
      }
    }

  }


  // no "return" unlike std::function since it modifies u_grad_array_CM_chunk by reference!

}









 
 
 
 
 
 
 
// Gradient computation function template (no need to have any template parameters as not very modular e.g. only double's)
ALWAYS_INLINE void fn_MVP_compute_L_Omega_grad_v3(   Eigen::Ref<Eigen::Matrix<double, -1, -1>>   U_Omega_grad_array,
                                             //  std::vector<Eigen::Matrix<double, -1, -1>> &Omega_grad_array_for_each_n,
                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  common_grad_term_1,
                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob,
                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob_recip,
                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  prop_rowwise_prod_temp,
                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>>   z_grad_term,
                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>>   grad_prob,
                                               Eigen::Ref<Eigen::Matrix<double, -1, 1>>    prod_container,
                                               Eigen::Ref<Eigen::Matrix<double, -1, 1>>    derivs_chain_container_vec,
                                               const bool compute_final_scalar_grad,
                                               const Model_fn_args_struct &Model_args_as_cpp_struct
                                                 
 ) {
   
   const int  n_class = Model_args_as_cpp_struct.Model_args_ints(1);
   const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
   const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
   
   // const int chunk_size = common_grad_term_1.rows();
   const int n_tests = common_grad_term_1.cols();
   
   // if (n_class == 1) {  /// i.e. if using standard MVP
   //   common_grad_term_1.setOnes();
   //   prop_rowwise_prod_temp.setOnes();
   // } 
   
   {
     
     z_grad_term.setZero(); // .array() = 0.0 ;
     grad_prob.setZero();// .array() = 0.0;
     
     {
       ///////////////////////// deriv of diagonal elements (not needed if using the "standard" or "Stan" Cholesky parameterisation of Omega)
       //////// w.r.t last diagonal first
       {
         
         int  t1 = n_tests - 1;
         
         if (n_class == 1)  {
           if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) +=   ( prob_recip.col(t1).array()   *   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()  *   Bound_Z.col(t1).array() ).sum();
         }  else  {
           if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) +=  (   common_grad_term_1.col(t1).array()   *   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()  *   Bound_Z.col(t1).array()       ).sum();
         }
         
       }
       
       //////// then w.r.t the second-to-last diagonal
       int  t1 = n_tests - 2;
       grad_prob.col(0).array()  =       (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array() *     Bound_Z.col(t1).array() ).array()    ;     // correct  (standard form)
       z_grad_term.col(0).array()  =    (  (   -      y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1).array()  )    *  Bound_Z.col(t1).array()  ).array()  ;  // correct
       prod_container.array()  =   (  L_Omega_double(t1 + 1, t1)    *   z_grad_term.col(0).array()   ).array() ; // sequence
       grad_prob.col(1).array()  =  (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + 1).array()   *          prod_container.array()     ).array()    ;    // correct   (standard form)
       
       if (n_class == 1) {
         if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) +=   (  prob_recip.col(t1).array()  *   grad_prob.col(0).array()  +    prob_recip.col(t1 + 1).array()  *      grad_prob.col(1).array()   ).sum();
       }  else   {
         if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) +=   ((common_grad_term_1.col(t1).array() )  *  (  prob.col(t1 + 1).array() * grad_prob.col(0).array()  +  prob.col(t1).array()  * grad_prob.col(1).array()   )  ).sum();
       }
       
     }
     ////////// then w.r.t the third-to-last diagonal .... etc
     {
       
       for (int i = 3; i < n_tests + 1; i++) {
         
         int  t1 = n_tests - i;
         
         //////// 1st component
         grad_prob.col(0).array()  =   (  (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array() )  *  ( Bound_Z.col(t1).array()   ).array() ).array() ; // correct  (standard form)
         z_grad_term.col(0).array()  =   (   ( -     y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1).array()  )  *  Bound_Z.col(t1).array()   ).array()    ;   // correct  (standard form)
         prod_container.array() =    ( L_Omega_double(t1 + 1, t1)   * z_grad_term.col(0).array() ).array()   ; // correct  (standard form)
         grad_prob.col(1).array()  =   (  (  y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + 1).array()  )   *  (    prod_container.array() ).array() ).array()  ; // correct  (standard form)
         
         for (int ii = 1; ii < i - 1; ii++) {
           z_grad_term.col(ii).array()  =    (   (-      y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1 + ii).array()  )    *   prod_container.array()  ).array()  ;   // correct  (standard form)       // grad_z term
           prod_container.matrix() =   (  (  L_Omega_double.row(t1 + ii + 1).segment(t1, ii + 1) *   z_grad_term.leftCols(ii + 1).transpose() ).transpose().matrix()  ).array() ; // correct  (standard form)
           grad_prob.col(ii + 1).array()  =   (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + ii + 1).array()    *   prod_container.array() ).array()  ;  // correct  (standard form)     //    grad_prob term
         }
         
         if (n_class != 1) {  //// if latent class
           
           derivs_chain_container_vec.setZero(); // .array() = 0.0;
           for (int iii = 0; iii <  i; iii++) {
             derivs_chain_container_vec.array()  +=    ( grad_prob.col(iii).array()  * ( prop_rowwise_prod_temp.col(t1).array()    *   prob_recip.col(t1 + iii).array()  ).array() ).array()   ;  // correct  (standard form)
           }
           if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) +=  (   common_grad_term_1.col(t1).array()   * derivs_chain_container_vec.array() ).sum();
           
         } else { //// if standard MVP
           
           derivs_chain_container_vec.setZero(); // .array() = 0.0;
           for (int iii = 0; iii <  i; iii++) {
             derivs_chain_container_vec.array()  +=    grad_prob.col(iii).array()  *  prob_recip.col(t1 + iii).array()    ;  
           }
           if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) +=   derivs_chain_container_vec.sum();
           
         }
         
       }
       
     }
     
   }
   
   {
     
     {
       int t1_dash = 0;  // t1 = n_tests - 1
       int t1 = n_tests - (t1_dash + 1); //  starts at n_tests - 1;  // if t1_dash = 0 -> t1 = T - 1
       int t2 = n_tests - (t1_dash + 2); //  starts at n_tests - 2;
       
       if (n_class == 1) {
         if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=   (prob_recip.col(t1).array()    *    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()  *  Z_std_norm.col(t2).array()   ).sum();
       }  else {
         if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=  (common_grad_term_1.col(t1).array()      *    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()   *    Z_std_norm.col(t2).array()   ).sum();
       }
       
       if (t1 > 1) { // starts at  L_{T, T-2}
         
         t2 =   n_tests - (t1_dash + 3); // starts at n_tests - 3;
         
         if (n_class == 1) {
           if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=    (prob_recip.col(t1).array()   *    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()   *    Z_std_norm.col(t2).array()   ).sum();
         } else  {
           if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=    (common_grad_term_1.col(t1).array()  * y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()  *    Z_std_norm.col(t2).array()   ).sum();
         }
         
       }
       
       if (t1 > 2) { // starts at  L_{T, T-3}
         
         for (int t2_dash = 3; t2_dash < n_tests; t2_dash++ ) { // t2 < t1
           
           t2 = n_tests - (t1_dash + t2_dash + 1); // starts at T - 4
           
           if (t2 < n_tests - 1) {
             if (n_class == 1)  {
               if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=    (prob_recip.col(t1).array() * y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array() * Z_std_norm.col(t2).array()).sum();
             } else {
               if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=    (common_grad_term_1.col(t1).array() * y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array() * Z_std_norm.col(t2).array()).sum();
             }
           }
           
         }
         
       }
       
     }
     
   }
   
   {
     
     z_grad_term.setZero(); // .array() = 0.0 ;
     grad_prob.setZero(); // .array() = 0.0;
     
     
     /////////////////// then rest of rows (second-to-last row, then third-to-last row, .... , then first row)
     for (int t1_dash = 1; t1_dash <  n_tests - 1;  t1_dash++) {
       int  t1 = n_tests - (t1_dash + 1);
       
       for (int t2_dash = t1_dash + 1; t2_dash <  n_tests;  t2_dash++) {
         int t2 = n_tests - (t2_dash + 1); // starts at t1 - 1, then t1 - 2, up to 0
         
         {
           prod_container.array()  =  Z_std_norm.col(t2).array() ; // block(0, t2, index_size, t1 - t2) * deriv_L_t1.head(t1 - t2) ;
           grad_prob.col(0).array()  =      (   y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1).array()   *           prod_container.array()  ).array()  ;
           z_grad_term.col(0).array()  =       (                 y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1).array()     *   (   -    prod_container.array()     )    ).array()   ;
           
           if (t1_dash > 0) {
             for (int t1_dash_dash = 1; t1_dash_dash <  t1_dash + 1;  t1_dash_dash++) {
               if (t1_dash_dash > 1)  z_grad_term.col(t1_dash_dash - 1).array()   =  (   y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.col(t1 + t1_dash_dash - 1).array()   *  (- prod_container.array())  ).array() ;
               prod_container.array()  =            (   z_grad_term.leftCols(t1_dash_dash) *   L_Omega_double.row(t1 + t1_dash_dash).segment(t1, t1_dash_dash).transpose()   ).array() ;
               grad_prob.col(t1_dash_dash).array()  =          (     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.col(t1 + t1_dash_dash).array()    *      prod_container.array() ).array()   ;
             }
           }
           
           if (n_class == 1)   {  ////// if standard MVP
             
             derivs_chain_container_vec.setZero(); // .array() = 0.0;
             for (int ii = 0; ii <  t1_dash + 1; ii++) {
               derivs_chain_container_vec.array() +=   grad_prob.col(ii).array()  *     prob_recip.col(t1 + ii).array()   ; // correct i think
             }
             if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=  derivs_chain_container_vec.sum();
             
           } else {  //// if latent class
             
             derivs_chain_container_vec.setZero(); // .array() = 0.0;
             for (int ii = 0; ii <  t1_dash + 1; ii++) {
               derivs_chain_container_vec.array() += ( grad_prob.col(ii).array()  * (   prop_rowwise_prod_temp.col(t1).array()    *  prob_recip.col(t1 + ii).array()  ).array() ).array() ; // correct i think
             }
             if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) +=      (   common_grad_term_1.col(t1).array()   * derivs_chain_container_vec.array() ).sum();
             
           }
           
         }
       }
     }
     
   }
   
   
   // no "return" unlike std::function since it modifies u_grad_array_CM_chunk by reference!
   
 }
 
 
 
 
 
 




// 
// // Gradient computation function template (no need to have any template parameters as not very modular e.g. only double's)
//   ALWAYS_INLINE  void fn_MVP_compute_nuisance_grad_v1(  Eigen::Ref<Eigen::Matrix<double, -1, -1>>   u_grad_array_CM_chunk,
//                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>>  prob_n_recip,
//                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>>  phi_Z_recip,
//                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>>  phi_Bound_Z,
//                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>>  y_sign_chunk,
//                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>>  y_m_y_sign_x_u,
//                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>>  common_grad_term_1,
//                                               Eigen::Matrix<double, -1, -1> &L_Omega_double,
//                                               Eigen::Matrix<double, -1, -1> &L_Omega_recip_double,
//                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
//                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>>  grad_bound_z,
//                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>>  grad_Phi_bound_z,
//                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>>   z_grad_term,
//                                               Eigen::Ref<Eigen::Matrix<double, -1, -1>>   grad_prob,
//                                               Eigen::Ref<Eigen::Matrix<double, -1, 1>>    prod_container,
//                                               Eigen::Ref<Eigen::Matrix<double, -1, 1>>    derivs_chain_container_vec ) {
// 
// 
// 
// 
// 
// 
//   const int chunk_size = u_grad_array_CM_chunk.rows();
//   const int n_tests = u_grad_array_CM_chunk.cols();
// 
//   ///// last term is zero
// 
//   ///// then second-to-last term (test T - 1)
//   {
//     int t = n_tests - 2;
// 
//     z_grad_term.col(0).array()  =  phi_Z_recip.col(t).array() *  prob.col(t).array() ;
//     prod_container.array()  =    L_Omega_double(t + 1, t)   * z_grad_term.col(0).array() ;
//     grad_bound_z.col(0).array()  =   (  L_Omega_recip_double(t + 1, t + 1)  ) *  ( -  prod_container.array()  ) ;
//     grad_Phi_bound_z.col(0).array()  =  phi_Bound_Z.col(t + 1).array() * grad_bound_z.col(0).array();
//     grad_prob.col(0).array()    =   (  - y_sign_chunk.col(t + 1).array()  )  * grad_Phi_bound_z.col(0).array();
// 
//     u_grad_array_CM_chunk.col( n_tests - 2).array() +=   (  common_grad_term_1.col(t + 1).array()   *     grad_prob.col(0).array()   ) ;
// 
//   }
// 
//   ///// then third-to-last term
//   {
// 
//     int t = n_tests - 3;
// 
//     // 1st set of z_grad_term and grad_prob terms
//     z_grad_term.col(0).array()  =  phi_Z_recip.col(t).array() *  prob.col(t).array() ;
//     prod_container.array()  =    L_Omega_double(t + 1, t)   * z_grad_term.col(0).array() ;
//     grad_bound_z.col(0).array()  =   (   L_Omega_recip_double(t + 1, t + 1)  ) *  ( -  prod_container.array()  ) ;
//     grad_Phi_bound_z.col(0).array()  =  phi_Bound_Z.col(t + 1).array() * grad_bound_z.col(0).array();
//     grad_prob.col(0).array()    =   (  - y_sign_chunk.col(t + 1).array()  )  * grad_Phi_bound_z.col(0).array();
// 
// 
//     // 2nd set of z_grad_term and grad_prob terms
//     z_grad_term.col(1).array()  =  ( phi_Z_recip.col(t + 1).array() ) *     y_m_y_sign_x_u.col(t + 1).array() *    grad_Phi_bound_z.col(0).array() ;
//     prod_container.array()  =   L_Omega_double(t + 2, t)   * z_grad_term.col(0).array()   +  L_Omega_double(t + 2, t + 1)   * z_grad_term.col(1).array()    ;
//     grad_bound_z.col(1).array()  =   (  L_Omega_recip_double(t + 2, t + 2)  ) *  ( -  prod_container.array()  ) ;
//     grad_Phi_bound_z.col(1).array()  =  phi_Bound_Z.col(t + 2).array() * grad_bound_z.col(1).array();
//     grad_prob.col(1).array()    =   (  - y_sign_chunk.col(t + 2).array()  )  * grad_Phi_bound_z.col(1).array();
// 
// 
//     u_grad_array_CM_chunk.col(n_tests - 3).array()  +=      (common_grad_term_1.col(t + 1).array()   *  (  grad_prob.col(1).array()  * prob.col(t + 1).array()  +      grad_prob.col(0).array()  *  prob.col(t + 2).array() ).array() ).array() ;
// 
//   }
// 
// 
//   ///// then rest of terms
//   {
// 
//     for (int i = 1; i < n_tests - 2; i++ ) {
// 
//       int t = n_tests - (i + 3);
// 
//       // 1st set of z_grad_term and grad_prob terms
//       z_grad_term.col(0).array()  =   phi_Z_recip.col(t).array()  *  prob.col(t).array() ;
//       prod_container.array()  =    L_Omega_double(t + 1, t)   * z_grad_term.col(0).array() ;
//       grad_bound_z.col(0).array()  =   (  L_Omega_recip_double(t + 1, t + 1)  ) *  ( -  prod_container.array()  ) ;
// 
//       grad_Phi_bound_z.col(0).array()  =  phi_Bound_Z.col(t + 1).array() * grad_bound_z.col(0).array();
//       grad_prob.col(0).array()    =   (  - y_sign_chunk.col(t + 1).array()  )  * grad_Phi_bound_z.col(0).array();
// 
// 
//       for (int ii = 1; ii < i + 2; ii++ ) {
//         z_grad_term.col(ii).array()  =  (  phi_Z_recip.col(t + ii).array() ) *     y_m_y_sign_x_u.col(t + ii).array() *    grad_Phi_bound_z.col(ii - 1).array() ;
//         prod_container.matrix()  =  ( z_grad_term.block(0, 0, chunk_size, ii + 1)  *  L_Omega_double.row(t + ii + 1).segment(t, ii + 1).transpose() ).matrix()  ;
//         grad_bound_z.col(ii).array()  =   (    L_Omega_recip_double(t + ii + 1, t + ii + 1)  ) *  ( -  prod_container.array()  ) ;
//         grad_Phi_bound_z.col(ii).array()  =  phi_Bound_Z.col(t + ii + 1).array() * grad_bound_z.col(ii).array();
//         grad_prob.col(ii).array()    =   (  - y_sign_chunk.col(t + ii + 1).array()  )  * grad_Phi_bound_z.col(ii).array();
//       }
// 
// 
//       {
//         derivs_chain_container_vec.array() = 0.0;
//         for (int ii = 0; ii < i + 2; ii++) {
//           //  derivs_chain_container_vec.array()  +=  ( grad_prob.col(ii).array()    * (       prop_rowwise_prod_temp.col(t).array() * prob_recip.col(t + ii + 1).array()  ).array() ).array()  ;
//           derivs_chain_container_vec.array()  +=    ( grad_prob.col(ii).array()  * (    prob.block(0, t + 1, chunk_size, i + 2).rowwise().prod().array() /  prob.col(t + ii + 1).array()  ).array() ).array() ;
//         }
//         u_grad_array_CM_chunk.col(n_tests - (i+3)).array()    +=   (  ( (   common_grad_term_1.col(t + 1).array()   *  derivs_chain_container_vec.array() ) ).array()  ).array() ;
//       }
// 
//     }
// 
//   }
// 
//   // no "return" unlike std::function since it modifies u_grad_array_CM_chunk by reference!
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
//  
//  
//  
//  
//  
//  
//  
//  
//  
// 
// // Gradient computation function template (no need to have any template parameters as not very modular e.g. only double's)
//   ALWAYS_INLINE void fn_MVP_compute_coefficients_grad_v1(  Eigen::Ref<Eigen::Matrix<double, -1, -1>>   u_coefficients_array_chunk,
//                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  prob_n_recip,
//                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  phi_Z_recip,
//                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  phi_Bound_Z,
//                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  y_sign_chunk,
//                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  y_m_y_sign_x_u,
//                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  common_grad_term_1,
//                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  L_Omega_double,
//                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  L_Omega_recip_double,
//                                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  prob,
//                                                     Eigen::Ref<Eigen::Matrix<double, -1, -1>>  grad_bound_z,
//                                                     Eigen::Ref<Eigen::Matrix<double, -1, -1>>  grad_Phi_bound_z,
//                                                     Eigen::Ref<Eigen::Matrix<double, -1, -1>>  z_grad_term,
//                                                     Eigen::Ref<Eigen::Matrix<double, -1, -1>>  grad_prob,
//                                                     Eigen::Ref<Eigen::Matrix<double, -1, 1>>   prod_container,
//                                                     Eigen::Ref<Eigen::Matrix<double, -1, 1>>   derivs_chain_container_vec
// ) {
// 
// 
// 
//   const int chunk_size = u_coefficients_array_chunk.rows();
//   const int n_tests = u_coefficients_array_chunk.cols();
// 
//   {
// 
//     { // last term first
//       int t = n_tests - 1;
//       u_coefficients_array_chunk(0, t)  +=    (  common_grad_term_1.col(t).array()  *   ( y_sign_chunk.col(t).array() *  phi_Bound_Z.col(t).array() *  L_Omega_recip_double(t, t) ) ).sum();
//     }
// 
//     {   ///// then second-to-last term (test T - 1)
// 
//       { ///////// new
//         int t = n_tests - 2;
// 
//         // 1st (and last) grad_Z term and 1st grad_prob term (simplest terms)
//         grad_bound_z.col(0).array()  =  ( -   L_Omega_recip_double(t, t)  ) ; //   CHECKED
//         grad_Phi_bound_z.col(0).array()  =     phi_Bound_Z.col(t).array() *  grad_bound_z.col(0).array() ; //  CHECKED
//         grad_prob.col(0).array()  =   ( - y_sign_chunk.col(t).array() ) *    grad_Phi_bound_z.col(0).array() ;  //  CHECKED
// 
//         z_grad_term.col(0).array()  =   y_m_y_sign_x_u.col(t).array() * (   phi_Z_recip.col(t).array() ) *   grad_Phi_bound_z.col(0).array()  ;  //  CHECKED
// 
//         // 2nd (and last)    grad_prob  term
//         prod_container =     ( L_Omega_double(t + 1, t)   * z_grad_term.col(0).array() ).matrix() ; //  CHECKED
//         grad_bound_z.col(1).array()  = ( -   L_Omega_recip_double(t + 1, t + 1) ) * prod_container.array() ; //  CHECKED
//         grad_Phi_bound_z.col(1).array()  =     phi_Bound_Z.col(t + 1).array() *  grad_bound_z.col(1).array() ;  //
//         grad_prob.col(1).array()  =   ( - y_sign_chunk.col(t + 1).array() ) *    grad_Phi_bound_z.col(1).array() ;  //
// 
//         u_coefficients_array_chunk(0, t)  +=   (  common_grad_term_1.col(t).array()   * ( grad_prob.col(1).array() *  prob.col(t).array() +         grad_prob.col(0).array() *   prob.col(t  + 1).array() ) ).sum() ;
// 
// 
// 
//       }
// 
//     }
// 
// 
// 
// 
//     for (int i = 1; i < n_tests - 1; i++) {     // then rest of terms
// 
//       grad_bound_z.array() = 0.0;
//       grad_Phi_bound_z.array() = 0.0;
//       z_grad_term.array() = 0.0;
//       grad_prob.array() = 0.0;
//       prod_container.array() = 0.0;
//       derivs_chain_container_vec.array() = 0.0;
// 
//       int t = n_tests - (i + 2) ;
// 
//       {  ////// new
// 
//         //////// 1st component
//         // 1st grad_Z term and 1st grad_prob term (simplest terms)
//         grad_bound_z.col(0).array()  =  ( -   L_Omega_recip_double(t, t)  ) ;  // ---------
//         grad_Phi_bound_z.col(0).array()  =     phi_Bound_Z.col(t).array() *  grad_bound_z.col(0).array() ; // ---------
//         grad_prob.col(0).array()  =   ( - y_sign_chunk.col(t).array() ) *    grad_Phi_bound_z.col(0).array() ;  // ---------
// 
//         z_grad_term.col(0).array()  =    y_m_y_sign_x_u.col(t).array() *  (  phi_Z_recip.col(t).array() )  *   grad_Phi_bound_z.col(0).array()  ;   // ---------
// 
//         // 2nd   grad_Z term and 2nd grad_prob  (more complicated than 1st term)
//         prod_container =    L_Omega_double(t + 1, t)   * z_grad_term.col(0).array()  ;  // ---------
//         grad_bound_z.col(1).array()  = ( -  L_Omega_recip_double(t + 1, t + 1) ) * prod_container.array() ;   // ---------
//         grad_Phi_bound_z.col(1).array()  =     phi_Bound_Z.col(t + 1).array()  *  grad_bound_z.col(1).array() ;     // ---------
//         grad_prob.col(1).array()  =   ( - y_sign_chunk.col(t + 1).array()  ) *    grad_Phi_bound_z.col(1).array() ;  // ---------
// 
//         z_grad_term.col(1).array()  =    y_m_y_sign_x_u.col(t + 1).array()  * (   phi_Z_recip.col(t + 1).array()  ) *   grad_Phi_bound_z.col(1).array()  ;   // ---------
// 
//         // 3rd   grad_prob term
//         prod_container =   L_Omega_double(t + 2, t)   * z_grad_term.col(0).array()    +  L_Omega_double(t + 2, t + 1)   * z_grad_term.col(1).array()  ;  // ---------
//         grad_bound_z.col(2).array()  = ( -  L_Omega_recip_double(t + 2, t + 2)  ) * prod_container.array();  // ---------
//         grad_Phi_bound_z.col(2).array()  =     phi_Bound_Z.col(t + 2).array()  *  grad_bound_z.col(2).array() ;     // ---------
//         grad_prob.col(2).array()  =   ( - y_sign_chunk.col(t + 2).array()  ) *   grad_Phi_bound_z.col(2).array() ;  // ---------
// 
// 
//         for (int ii = 2; ii < n_tests - 1; ii++) { // if i = 1, ii goes from 0 to 1
// 
//           if (i > ii - 1) {
// 
//             ///////////////
//             // grad_z term
//             z_grad_term.col(ii).array()  =    y_m_y_sign_x_u.col(t + ii).array()  *   phi_Z_recip.col(t + ii).array()   *   grad_Phi_bound_z.col(ii).array()  ;     // ---------
// 
//             //    grad_prob term
//             prod_container.array() =   (  L_Omega_double.row(t + ii + 1).segment(t, ii + 1) *   z_grad_term.block(0, 0, chunk_size, ii + 1).transpose() ).transpose().array();
//             grad_bound_z.col(ii + 1).array()  = ( -  L_Omega_recip_double(t + ii + 1, t + ii + 1)  ) * prod_container.array();  // ---------
//             grad_Phi_bound_z.col(ii + 1).array()  =     phi_Bound_Z.col(t + ii + 1).array()  *  grad_bound_z.col(ii + 1).array() ;    // ---------
//             grad_prob.col(ii + 1).array()  =   ( - y_sign_chunk.col(t + ii + 1).array()  ) *   grad_Phi_bound_z.col(ii + 1).array() ;   // ----------------
// 
// 
//           }
// 
// 
//         }
// 
//         {
//           derivs_chain_container_vec.array() = 0.0;
//           for (int ii = 0; ii < i + 2; ii++) {
//             //  derivs_chain_container_vec.array() +=  ( grad_prob.col(ii).array()  * (        prop_rowwise_prod_temp.col(t).array() *   prob_recip.col(t + ii).array()  ).array() ).array() ;
//             derivs_chain_container_vec.array() +=  ( grad_prob.col(ii).array()  * (    prob.block(0, t + 0, chunk_size, i + 2).rowwise().prod().array() /  prob.col(t + ii).array()  ).array() ).array() ;
//           }
//           u_coefficients_array_chunk(0, n_tests - (i + 2))   +=         (   common_grad_term_1.col(t).array()   *  derivs_chain_container_vec.array() ).sum();
//         }
// 
// 
// 
//       }
// 
// 
//     }
// 
// 
// 
//   }
// 
//   // no "return" unlike std::function since it modifies u_grad_array_CM_chunk by reference!
// 
// }
// //
// //
// 
// // 
// // 
// // 
// 
// 











 
 