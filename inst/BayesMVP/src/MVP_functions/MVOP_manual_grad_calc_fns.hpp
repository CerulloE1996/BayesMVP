
//// MVOP_manual_grad_calc_fns.hpp

#pragma once

 
 
 
#include <Eigen/Dense>



using namespace Eigen;




#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE



// extern bool g_debug_cutpoint_grads;
 
 
// ============================================================================
// fn_MVOP_compute_lp_GHK_cols.hpp
//
// Mixed binary + ordinal GHK column computation.
// Replaces fn_MVP_compute_lp_GHK_cols for the MVOP model.
//
// Key difference from MVP version:
//   - Binary tests (t < n_bin): identical to MVP (uses y_sign trick)
//   - Ordinal tests (t >= n_bin): uses lower/upper bounds from cutpoints
//
// The function computes the same outputs as the binary version:
//   - Bound_U_Phi_Bound_Z (now stores Φ(lower_bz) for ordinal tests)  
//   - Phi_Z, Z_std_norm, prob, y1_log_prob
// Plus new outputs for ordinal tests:
//   - Upper_Bound_Z: upper bound in std normal space (ordinal only)
//   - Phi_Upper_Bound_Z: Φ(upper_bz) (ordinal only)
//
// NOTE: For ordinal tests, Bound_Z stores the LOWER bound and 
//       Upper_Bound_Z stores the UPPER bound.
// ============================================================================
 
 
// #include <Eigen/Dense>
// #include <stan/math.hpp>
// #include "MVP_lp_grad_MD_AD_fns.hpp.hpp"
 
// ============================================================================
// Unified GHK column computation for mixed binary + ordinal
// ============================================================================
ALWAYS_INLINE void fn_MVOP_compute_lp_GHK_cols(    const int t,
                                                   Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,    // Φ(lower_bz) for ordinal, Φ(BZ) for binary
                                                   Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
                                                   Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                                   Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
                                                   Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                                   ////
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,     // lower_bz for ordinal
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_chunk,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array,
                                                   ////
                                                   // --- New ordinal arguments ---
                                                   Eigen::Ref<Eigen::Matrix<double, -1, -1>> Upper_Bound_Z,           // upper_bz (ordinal only, ignored for binary)
                                                   // Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Upper_Bound_Z,       // Φ(upper_bz) (ordinal only)
                                                   ////
                                                   const bool is_binary,                                                // slot type (was: const int n_bin)
                                                   const Model_fn_args_struct &Model_args_as_cpp_struct
                                                   // const int n_bin,                                                     // number of binary tests
                                                   // const Model_fn_args_struct &Model_args_as_cpp_struct
) {

       const std::string vect_type_log = Model_args_as_cpp_struct.Model_args_strings(4);
       const std::string Phi_type      = Model_args_as_cpp_struct.Model_args_strings(1);
       const std::string vect_type_Phi = Model_args_as_cpp_struct.Model_args_strings(7);
       const std::string inv_Phi_type  = Model_args_as_cpp_struct.Model_args_strings(2);
       const std::string vect_type_inv_Phi = Model_args_as_cpp_struct.Model_args_strings(9);

       if (is_binary) {

               // ================================================================
               // BINARY TEST: identical to existing MVP code (i.e., same as the "fn_MVP_compute_lp_GHK_cols" function)
               // ================================================================
               Eigen::Matrix<double, -1, -1> y_sign_col = y_chunk.col(t).array() + (y_chunk.col(t).array() - 1.0);  // 2y - 1

               Bound_U_Phi_Bound_Z.col(t) = fn_EIGEN_double( Bound_Z.col(t), Phi_type, vect_type_Phi);

               Phi_Z.col(t).array() = y_chunk.col(t).array() * Bound_U_Phi_Bound_Z.col(t).array()
                                    + (y_chunk.col(t).array() - Bound_U_Phi_Bound_Z.col(t).array())
                                    * (y_sign_col.array() * u_array.col(t).array());

               Z_std_norm.col(t) = fn_EIGEN_double(Phi_Z.col(t), inv_Phi_type, vect_type_inv_Phi);

               prob.col(t).array() = y_chunk.col(t).array() * (1.0 - Bound_U_Phi_Bound_Z.col(t).array())
                                   + (y_chunk.col(t).array() - 1.0) * Bound_U_Phi_Bound_Z.col(t).array()
                                   * (y_sign_col.array());

               y1_log_prob.col(t) = fn_EIGEN_double(prob.col(t), "log", vect_type_log);

       } else {

               // ================================================================
               // ORDINAL TEST: uses lower and upper bounds from cutpoints
               // ================================================================
               // At this point:
               //   Bound_Z.col(t)       = lower_bz = (C_{k-1} - mu_t) / L(t,t)  [or -big if k=1]
               //   Upper_Bound_Z.col(t) = upper_bz = (C_k - mu_t) / L(t,t)      [or +big if k=K]
               //
               // These must be pre-computed by the caller before calling this fn.

               // Φ(lower_bz) — store in Bound_U_Phi_Bound_Z for consistency
               Bound_U_Phi_Bound_Z.col(t) = fn_EIGEN_double(Bound_Z.col(t), Phi_type, vect_type_Phi);

               // // Φ(upper_bz)
               // Phi_Upper_Bound_Z.col(t) = fn_EIGEN_double(Upper_Bound_Z.col(t), Phi_type, vect_type_Phi);

               // prob = Φ(upper_bz) - Φ(lower_bz)
               prob.col(t).array() = fn_EIGEN_double(Upper_Bound_Z.col(t), Phi_type, vect_type_Phi).array() - Bound_U_Phi_Bound_Z.col(t).array();
               // prob.col(t).array() = prob.col(t).array().max(1e-30);

               // Truncated uniform -> truncated normal:
               // Φ(Z) = Φ(lower_bz) + u * prob
               Phi_Z.col(t).array() = Bound_U_Phi_Bound_Z.col(t).array() + u_array.col(t).array() * prob.col(t).array();

               // Z = Φ⁻¹(Φ(Z))no
               Z_std_norm.col(t) = fn_EIGEN_double(Phi_Z.col(t), inv_Phi_type, vect_type_inv_Phi);

               // log(prob)
               y1_log_prob.col(t) = fn_EIGEN_double(prob.col(t), "log", vect_type_log);

       }

}
// //// =====================================================================================
// //// PART A: fn_MVOP_compute_lp_GHK_cols — rewritten. Same outputs, same maths, no strings,
// ////         no temporaries. Note the pattern: copy the input column INTO the output column,
// ////         then transform the output column in place.
// //// =====================================================================================
// template <Vec vec>
// inline void fn_MVOP_compute_lp_GHK_cols_T(   const int t,
//                                            Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
//                                            Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
//                                            Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
//                                            Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
//                                            Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
//                                            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
//                                            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_chunk,
//                                            const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array,
//                                            Eigen::Ref<Eigen::Matrix<double, -1, -1>> Upper_Bound_Z,
//                                            const bool is_binary,
//                                            const KernelChoice &k
// ) {
// 
//         if (is_binary) {
//       
//               //// Phi(BZ):
//               Bound_U_Phi_Bound_Z.col(t) = Bound_Z.col(t);
//               Phi_col_inplace<vec>(Bound_U_Phi_Bound_Z, t, k);
//       
//               //// Phi_Z = y*Phi(BZ) + (y - Phi(BZ)) * (2y-1) * u      [(2y-1) written inline, no y_sign temp]
//               Phi_Z.col(t).array() = y_chunk.col(t).array() * Bound_U_Phi_Bound_Z.col(t).array()
//                                    + (y_chunk.col(t).array() - Bound_U_Phi_Bound_Z.col(t).array())
//                                    * ((2.0 * y_chunk.col(t).array() - 1.0) * u_array.col(t).array());
//       
//               //// Z = inv_Phi(Phi_Z):
//               Z_std_norm.col(t) = Phi_Z.col(t);
//               inv_Phi_col_inplace<vec>(Z_std_norm, t, k);
//       
//               //// prob = y*(1 - Phi(BZ)) + (y-1)*Phi(BZ)*(2y-1):
//               prob.col(t).array() = y_chunk.col(t).array() * (1.0 - Bound_U_Phi_Bound_Z.col(t).array())
//                                   + (y_chunk.col(t).array() - 1.0) * Bound_U_Phi_Bound_Z.col(t).array()
//                                   * (2.0 * y_chunk.col(t).array() - 1.0);
//       
//               //// log(prob):
//               y1_log_prob.col(t) = prob.col(t);
//               apply_col_inplace<vec, Fn::log>(y1_log_prob, t);
//       
//         } else {
//       
//               //// Phi(lower_bz) -> Bound_U_Phi_Bound_Z:
//               Bound_U_Phi_Bound_Z.col(t) = Bound_Z.col(t);
//               Phi_col_inplace<vec>(Bound_U_Phi_Bound_Z, t, k);
//       
//               //// prob = Phi(upper_bz) - Phi(lower_bz). Use prob.col(t) itself as the scratch for Phi(ub):
//               prob.col(t) = Upper_Bound_Z.col(t);
//               Phi_col_inplace<vec>(prob, t, k);
//               prob.col(t).array() -= Bound_U_Phi_Bound_Z.col(t).array();
//       
//               //// Phi(Z) = Phi(lb) + u * prob:
//               Phi_Z.col(t).array() = Bound_U_Phi_Bound_Z.col(t).array() + u_array.col(t).array() * prob.col(t).array();
//       
//               //// Z = inv_Phi(Phi(Z)):
//               Z_std_norm.col(t) = Phi_Z.col(t);
//               inv_Phi_col_inplace<vec>(Z_std_norm, t, k);
//       
//               //// log(prob):
//               y1_log_prob.col(t) = prob.col(t);
//               apply_col_inplace<vec, Fn::log>(y1_log_prob, t);
//       
//         }
// 
// }




// ============================================================================
// fn_MVOP_compute_phi_Bound_Z_cols: Compute φ(lower_bz) and φ(upper_bz) 
// for ordinal tests, or φ(BZ) for binary tests.
//
// For binary tests: identical to fn_MVP_compute_phi_Bound_Z_cols
// For ordinal tests: computes both phi_Lower_Bound_Z and phi_Upper_Bound_Z
// ============================================================================
ALWAYS_INLINE void fn_MVOP_compute_phi_Bound_Z_cols(   const int t,
                                                       Eigen::Matrix<double, -1, -1> &phi_Bound_Z,           // φ(lower_bz) for ordinal, φ(BZ) for binary
                                                       Eigen::Matrix<double, -1, -1> &phi_Upper_Bound_Z_out, // φ(upper_bz) for ordinal (unused for binary)
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Phi_Upper_Bound_Z,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
                                                       const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Upper_Bound_Z,
                                                       const bool is_binary,                                            // slot type (was: const int n_bin)
                                                       const Model_fn_args_struct &Model_args_as_cpp_struct
                                                       // const int n_bin,
                                                       // const Model_fn_args_struct &Model_args_as_cpp_struct
) {

       const double sqrt_2_pi_recip = 1.0 / std::sqrt(2.0 * M_PI);
       const double a = 0.07056;
       const double b = 1.5976;
       const double a_times_3 = 3.0 * a;

       const std::string Phi_type = Model_args_as_cpp_struct.Model_args_strings(1);
       const std::string vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);

       phi_Bound_Z.col(t).setZero();

       // φ(lower_bz) or φ(BZ) — same computation regardless of binary/ordinal
       if ((Phi_type == "Phi_approx") || (Phi_type == "Phi_approx_2")) {

             const Eigen::Matrix<double, -1, 1> BZ_sq = stan::math::square(Bound_Z.col(t));
             Eigen::Matrix<double, -1, 1> temp = (a_times_3 * BZ_sq.array() + b).matrix();
             temp.array() *= Bound_U_Phi_Bound_Z.col(t).array();
             temp.array() *= (1.0 - Bound_U_Phi_Bound_Z.col(t).array()).array();
             phi_Bound_Z.col(t).array() += temp.array();

       } else if (Phi_type == "Phi") {

             phi_Bound_Z.col(t).array() += (sqrt_2_pi_recip * fn_EIGEN_double( (-0.5) * stan::math::square(Bound_Z.col(t)), "exp", vect_type_exp)).array();

       }

       // For ordinal tests: also compute φ(upper_bz)
       if (!is_binary) {

             phi_Upper_Bound_Z_out.col(t).setZero();

             if ((Phi_type == "Phi_approx") || (Phi_type == "Phi_approx_2")) {

                   const Eigen::Matrix<double, -1, 1> UBZ_sq = stan::math::square(Upper_Bound_Z.col(t));
                   Eigen::Matrix<double, -1, 1> temp = (a_times_3 * UBZ_sq.array() + b).matrix();
                   temp.array() *= Phi_Upper_Bound_Z.col(t).array();
                   temp.array() *= (1.0 - Phi_Upper_Bound_Z.col(t).array()).array();
                   phi_Upper_Bound_Z_out.col(t).array() += temp.array();

             } else if (Phi_type == "Phi") {

                   phi_Upper_Bound_Z_out.col(t).array() += (sqrt_2_pi_recip * fn_EIGEN_double((-0.5) * stan::math::square(Upper_Bound_Z.col(t)), "exp", vect_type_exp)).array();

             }

       }

}
// template <Vec vec>
// inline void fn_MVOP_compute_phi_Bound_Z_cols_T(  const int t,
//                                                Eigen::Matrix<double, -1, -1> &phi_Bound_Z,
//                                                Eigen::Matrix<double, -1, -1> &phi_Upper_Bound_Z_out,
//                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
//                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Phi_Upper_Bound_Z,
//                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
//                                                const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Upper_Bound_Z,
//                                                const bool is_binary,
//                                                const KernelChoice &k
// ) {
// 
//         const double sqrt_2_pi_recip = 1.0 / std::sqrt(2.0 * M_PI);
//         const double a_times_3 = 3.0 * 0.07056;
//         const double b = 1.5976;
//       
//         //// phi(lower_bz) or phi(BZ):
//         if (k.Phi_approx) {
//               //// d/dx Phi_approx(x) = Phi_approx(x) (1 - Phi_approx(x)) (3 a x^2 + b)
//               phi_Bound_Z.col(t).array() = (a_times_3 * Bound_Z.col(t).array().square() + b)
//                                          * Bound_U_Phi_Bound_Z.col(t).array()
//                                          * (1.0 - Bound_U_Phi_Bound_Z.col(t).array());
//         } else {
//               //// phi(x) = exp(-x^2/2) / sqrt(2 pi):  write -x^2/2 into the output column, exp it in place, scale.
//               phi_Bound_Z.col(t).array() = -0.5 * Bound_Z.col(t).array().square();
//               apply_col_inplace<vec, Fn::exp>(phi_Bound_Z, t);
//               phi_Bound_Z.col(t).array() *= sqrt_2_pi_recip;
//         }
//       
//         //// Ordinal: also phi(upper_bz):
//         if (!is_binary) {
//               if (k.Phi_approx) {
//                     phi_Upper_Bound_Z_out.col(t).array() = (a_times_3 * Upper_Bound_Z.col(t).array().square() + b)
//                                                          * Phi_Upper_Bound_Z.col(t).array()
//                                                          * (1.0 - Phi_Upper_Bound_Z.col(t).array());
//               } else {
//                     phi_Upper_Bound_Z_out.col(t).array() = -0.5 * Upper_Bound_Z.col(t).array().square();
//                     apply_col_inplace<vec, Fn::exp>(phi_Upper_Bound_Z_out, t);
//                     phi_Upper_Bound_Z_out.col(t).array() *= sqrt_2_pi_recip;
//               }
//         }
// 
// }

















 
ALWAYS_INLINE void fn_MVOP_grad_prep(    const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_sign_chunk,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_m_y_sign_x_u,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_recip_double,
                                         ////
                                         // const double prev_double,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1,1>> prev_per_obs,
                                         ////
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, 1>>  prob_n_recip,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> phi_Z_recip,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> phi_Bound_Z,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> phi_Upper_Bound_Z,
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,        // NEW
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Upper_Bound_Z,  // NEW
                                         const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob_recip,
                                         Eigen::Ref<Eigen::Matrix<double, -1, -1>> prop_rowwise_prod_temp,
                                         Eigen::Ref<Eigen::Matrix<double, -1, -1>> prop_recip_rowwise_prod_temp,
                                         Eigen::Ref<Eigen::Matrix<double, -1, 1>>  prop_rowwise_prod_temp_all,
                                         Eigen::Ref<Eigen::Matrix<double, -1, -1>> common_grad_term_1,
                                         Eigen::Ref<Eigen::Matrix<double, -1, -1>> dphi_over_L,
                                         Eigen::Ref<Eigen::Matrix<double, -1, -1>> dZ_dmu_neg,
                                         Eigen::Ref<Eigen::Matrix<double, -1, -1>> dphi_times_bz,   // NEW OUTPUT
                                         Eigen::Ref<Eigen::Matrix<double, -1, -1>> dZ_times_bz,     // NEW OUTPUT
                                         ////
                                         // const int n_binary_tests,
                                         const Eigen::Matrix<int, -1, 1> &ord_idx_of_test,   // -1 = binary slot; else slot-order ordinal index
                                         const Model_fn_args_struct &Model_args_as_cpp_struct
) {
        
         const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
         const int chunk_size = prob.rows();
         const int n_tests = prob.cols();
         
         compute_rowwise_products(prop_rowwise_prod_temp,
                                  prop_recip_rowwise_prod_temp,
                                  prob, prob_recip,
                                  chunk_size, n_tests);
         
         prop_rowwise_prod_temp_all.array() = prob.rowwise().prod().array();
         
         if (n_class > 1) {
           compute_latent_class_terms(common_grad_term_1,
                                      prop_rowwise_prod_temp_all,
                                      prop_recip_rowwise_prod_temp,
                                      prob_n_recip,
                                      prev_per_obs,
                                      n_tests);
         } else {
           common_grad_term_1.setOnes();
         }
         
         for (int t = 0; t < n_tests; ++t) {
           
               // if (t < n_binary_tests) {
               if (ord_idx_of_test(t) < 0) {   // binary
                 
                      dphi_over_L.col(t).array() = y_sign_chunk.col(t).array() * phi_Bound_Z.col(t).array() * L_Omega_recip_double(t, t);
                     
                      dZ_dmu_neg.col(t).array() = y_m_y_sign_x_u.col(t).array() * phi_Z_recip.col(t).array() * phi_Bound_Z.col(t).array() * L_Omega_recip_double(t, t);
                     
                      // Binary: Bound_Z always finite
                      dphi_times_bz.col(t).array() = dphi_over_L.col(t).array() * Bound_Z.col(t).array();
                      dZ_times_bz.col(t).array()   = dZ_dmu_neg.col(t).array()  * Bound_Z.col(t).array();
                   
               } else {
                 
                     dphi_over_L.col(t).array() = (phi_Bound_Z.col(t).array() - phi_Upper_Bound_Z.col(t).array()) * L_Omega_recip_double(t, t);
                     
                     for (int n = 0; n < chunk_size; ++n) {
                       
                           double lbz = Bound_Z(n, t);
                           double ubz = Upper_Bound_Z(n, t);
                           
                           // dZ_dmu_neg
                           double numer = (1.0 - u_array(n, t)) * phi_Bound_Z(n, t) + u_array(n, t) * phi_Upper_Bound_Z(n, t);
                           dZ_dmu_neg(n, t) = (numer == 0.0) ? 0.0 : numer * phi_Z_recip(n, t) * L_Omega_recip_double(t, t);
                           
                           // dphi_times_bz = (phi(lbz)*lbz - phi(ubz)*ubz) / L
                           // with guards: x*phi(x) -> 0 as x -> +-Inf
                           double phi_lbz_lbz = std::isinf(lbz) ? 0.0 : phi_Bound_Z(n, t) * lbz;
                           double phi_ubz_ubz = std::isinf(ubz) ? 0.0 : phi_Upper_Bound_Z(n, t) * ubz;
                           dphi_times_bz(n, t) = (phi_lbz_lbz - phi_ubz_ubz) * L_Omega_recip_double(t, t);
                           
                           // // dZ_times_bz = dZ_dmu_neg * lbz, guard on lbz
                           // dZ_times_bz(n, t) = std::isinf(lbz) ? 0.0 : dZ_dmu_neg(n, t) * lbz;
                           
                           // dZ_times_bz for ordinal: correct formula
                           // double phi_lbz_lbz = std::isinf(lbz) ? 0.0 : phi_Bound_Z(n, t) * lbz;
                           // double phi_ubz_ubz = std::isinf(ubz) ? 0.0 : phi_Upper_Bound_Z(n, t) * ubz;
                           double numer_bz = (1.0 - u_array(n, t)) * phi_lbz_lbz + u_array(n, t) * phi_ubz_ubz;
                           dZ_times_bz(n, t) = (numer_bz == 0.0) ? 0.0 : numer_bz * phi_Z_recip(n, t) * L_Omega_recip_double(t, t);
                       
                     }
                 
               }
           
         }
   
}


 
 

// ============================================================================
// fn_MVOP_compute_cutpoint_grad
//
// Gradient w.r.t. cutpoints of a single ordinal test.
//
// For ordinal test at global index t (= n_bin + t_ord), with K_t categories
// and K_t - 1 cutpoints C_0 < C_1 < ... < C_{K_t-2}:
//
// Cutpoint C_j (0-indexed) is:
//   - UPPER bound for observations with y = j + 1
//   - LOWER bound for observations with y = j + 2
//
// For each cutpoint j:
//   ∂prob_t/∂C_j:
//     y = j+1 (upper):  +φ(ubz) / L(t,t)
//     y = j+2 (lower):  -φ(lbz) / L(t,t)
//   ∂Z_t/∂C_j:
//     y = j+1 (upper):  u * φ(ubz) / (φ(Z) * L(t,t))
//     y = j+2 (lower):  (1-u) * φ(lbz) / (φ(Z) * L(t,t))
//
// Then the chain-rule propagation to future tests t' > t uses the SAME
// recursive structure as fn_MVP_compute_coefficients_grad_v3:
//   z_grad_term tracks ∂Z_{t'}/∂C_j through L_Omega
//   grad_prob tracks ∂prob_{t'}/∂C_j
//
// NOTE: This function computes gradients w.r.t. ORDERED cutpoints C,
//       not raw C_raw. The chain rule C → C_raw is applied afterwards
//       using chain_rule_C_to_C_raw().
// ============================================================================


// ============================================================================
// fn_MVOP_compute_cutpoint_grad  —  FIXED VERSION
//
// BUG FIX: The n_future==0 and n_future==1 special-case accumulation formulas
// for n_class > 1 used "prob.col(t)" and "prob.col(t+1)" directly instead of
// "prop_rowwise_prod_temp.col(t) * prob_recip.col(t+ii)", which is what the
// general case (and the hand-written coefficient gradient) correctly uses.
//
// The n_class==1 formulas were already correct.
//
// Additional fixes:
//   - dphi_direct_Cj / dZ_direct_Cj are real vectors (not aliases to workspace)
//   - Inf guard on phi_Z_recip
// ============================================================================
ALWAYS_INLINE void fn_MVOP_compute_cutpoint_grad(  const int c,
                                                   Eigen::Ref<Eigen::Matrix<double, -1, -1>> cutpoint_grad_array,
                                                   const int t,
                                                   const int t_ord,
                                                   const int K_t,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> common_grad_term_1,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob_recip,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prop_rowwise_prod_temp,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> phi_Bound_Z,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> phi_Upper_Bound_Z,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> phi_Z_recip,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_chunk,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> dphi_over_L,
                                                   const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> dZ_dmu_neg,
                                                   Eigen::Ref<Eigen::Matrix<double, -1, -1>> z_grad_term,
                                                   Eigen::Ref<Eigen::Matrix<double, -1, -1>> grad_prob,
                                                   Eigen::Ref<Eigen::Matrix<double, -1, 1>> prod_container,
                                                   Eigen::Ref<Eigen::Matrix<double, -1, 1>> derivs_chain_container_vec,
                                                   Eigen::Ref<Eigen::Matrix<double, -1, 1>> dphi_direct_Cj, //// ---- ordinal-only + NEW
                                                   Eigen::Ref<Eigen::Matrix<double, -1, 1>> dZ_direct_Cj, //// ---- ordinal-only + NEW
                                                   const bool compute_final_scalar_grad,
                                                   const Model_fn_args_struct &Model_args_as_cpp_struct
) {
  
        const int n_class    = Model_args_as_cpp_struct.Model_args_ints(1);
        const int chunk_size = y_chunk.rows();
        const int n_tests    = y_chunk.cols();
        const int n_thr_t    = K_t - 1;
        const double L_tt     = L_Omega_double(t, t);
        const double L_tt_inv = 1.0 / L_tt;
        const int n_future   = n_tests - t - 1;
        
        grad_prob.setZero(); // .array()   = 0.0;
        z_grad_term.setZero(); // .array() = 0.0;
        
        // // ---- Actual vectors, NOT aliases into workspace (avoids clobbering) ----
        // Eigen::Matrix<double, -1, 1> dphi_direct_Cj(chunk_size);
        // Eigen::Matrix<double, -1, 1> dZ_direct_Cj(chunk_size);
        
        // ====================================================================
        // Loop over each cutpoint j
        // ====================================================================
        for (int j = 0; j < n_thr_t; ++j) {
          
              dphi_direct_Cj.setZero();
              dZ_direct_Cj.setZero();
              
              // ================================================================
              // Step 1: Compute per-observation seeds for cutpoint C_j
              // ================================================================
              for (int i = 0; i < chunk_size; ++i) {
            
                    const int y_i = static_cast<int>(y_chunk(i, t));
                    const double pzr = phi_Z_recip(i, t);
                    const bool pzr_ok = std::isfinite(pzr);
                    
                    if (y_i == j + 1 && y_i < K_t) {
                      
                          // C_j is UPPER bound for this observation
                          dphi_direct_Cj(i) = phi_Upper_Bound_Z(i, t) * L_tt_inv;
                          double uph = u_array(i, t) * phi_Upper_Bound_Z(i, t);
                          dZ_direct_Cj(i) = (uph == 0.0 || !pzr_ok) ? 0.0 : uph * pzr * L_tt_inv;
                      
                    } else if (y_i == j + 2) {
                      
                          // C_j is LOWER bound for this observation
                          dphi_direct_Cj(i) = -phi_Bound_Z(i, t) * L_tt_inv;
                          double omup = (1.0 - u_array(i, t)) * phi_Bound_Z(i, t);
                          dZ_direct_Cj(i) = (omup == 0.0 || !pzr_ok) ? 0.0 : omup * pzr * L_tt_inv;
                      
                    }
                    // else: C_j doesn't affect this observation -> stays zero
                
              }
              // ================================================================
              // Step 2: Propagation + accumulation
              // ================================================================
              if (n_future == 0) {
                
                    // ---- Last test: no chain-rule propagation needed ----
                    if (compute_final_scalar_grad) {
                          
                          if (n_class != 1) {
                            
                                // FIXED: use prop_rowwise_prod_temp * prob_recip (matching general case)
                                cutpoint_grad_array(j, t_ord) += (common_grad_term_1.col(t).array() * dphi_direct_Cj.array() * prop_rowwise_prod_temp.col(t).array() 
                                                                 * prob_recip.col(t).array()).sum();
                            
                          } else {
                            
                                cutpoint_grad_array(j, t_ord) += (dphi_direct_Cj.array() * prob_recip.col(t).array()).sum();
                          }
                          
                    }
                  
              } else {
                  
                    // ---- n_future >= 1: propagation through future tests ----
                    grad_prob.col(0).array()   = dphi_direct_Cj.array();
                    z_grad_term.col(0).array() = dZ_direct_Cj.array();
                    
                    // First future test effect
                    grad_prob.col(1).array() = dphi_over_L.col(t + 1).array() * (L_Omega_double(t + 1, t) * z_grad_term.col(0).array());
                    
                    // Remaining future tests (only runs if n_future > 1)
                    for (int ii = 1; ii < n_future; ++ii) {
                      
                          if (ii == 1) {
                            prod_container = z_grad_term.leftCols(ii) * L_Omega_double.row(t + (ii - 1) + 1).segment(t, ii).transpose();
                          }
                          
                          z_grad_term.col(ii).array() = -dZ_dmu_neg.col(t + ii).array() * prod_container.array();
                          
                          prod_container = z_grad_term.leftCols(ii + 1) * L_Omega_double.row(t + ii + 1).segment(t, ii + 1).transpose();
                          
                          grad_prob.col(ii + 1).array() = dphi_over_L.col(t + ii + 1).array() * prod_container.array();
                      
                    }
                    
                    // ---- Accumulation (same formula for ALL n_future >= 1) ----
                    if (compute_final_scalar_grad) {
                          
                          if (n_class != 1) {
                            
                                // FIXED: use prop_rowwise_prod_temp * prob_recip (matching general case + coeff grad)
                                derivs_chain_container_vec.setZero();
                                for (int ii = 0; ii < n_future + 1; ++ii) {
                                  derivs_chain_container_vec.array() += grad_prob.col(ii).array() * prop_rowwise_prod_temp.col(t).array() * prob_recip.col(t + ii).array();
                                }
                                cutpoint_grad_array(j, t_ord) += (common_grad_term_1.col(t).array() * derivs_chain_container_vec.array()).sum();
                            
                          } else {
                                derivs_chain_container_vec.setZero();
                                for (int ii = 0; ii < n_future + 1; ++ii) {
                                  derivs_chain_container_vec.array() += grad_prob.col(ii).array() * prob_recip.col(t + ii).array();
                                }
                                cutpoint_grad_array(j, t_ord) += derivs_chain_container_vec.sum();
                          }
                      
                    }
                
              }
              
        }  // end loop over cutpoints j
  
}

 





ALWAYS_INLINE void fn_MOVP_compute_L_Omega_grad_v3(   Eigen::Ref<Eigen::Matrix<double, -1, -1>>   U_Omega_grad_array,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>>  common_grad_term_1,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_Omega_double,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prob_recip,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> prop_rowwise_prod_temp,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> dphi_over_L,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> dZ_dmu_neg,
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> dphi_times_bz,  // NEW (replaces dphi_over_L*Bound_Z)
                                                      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> dZ_times_bz,    // NEW (replaces dZ_dmu_neg*Bound_Z)
                                                      Eigen::Ref<Eigen::Matrix<double, -1, -1>>   z_grad_term,
                                                      Eigen::Ref<Eigen::Matrix<double, -1, -1>>   grad_prob,
                                                      Eigen::Ref<Eigen::Matrix<double, -1, 1>>    prod_container,
                                                      Eigen::Ref<Eigen::Matrix<double, -1, 1>>    derivs_chain_container_vec,
                                                      const bool compute_final_scalar_grad,
                                                      const Model_fn_args_struct &Model_args_as_cpp_struct
) {
  
        const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
        const int n_tests = common_grad_term_1.cols();
        
        {
          z_grad_term.setZero();
          grad_prob.setZero();
          
          // ---- last diagonal ----
          {
                int t1 = n_tests - 1;
                
                if (n_class == 1) {
                  
                      if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) += (prob_recip.col(t1).array() * dphi_times_bz.col(t1).array()).sum();
                  
                } else {
                  
                     if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) += (common_grad_term_1.col(t1).array() * dphi_times_bz.col(t1).array()).sum();
                  
                }
          }
          
          // ---- second-to-last diagonal ----
          {
                int t1 = n_tests - 2;
            
                grad_prob.col(0).array()   = dphi_times_bz.col(t1).array();
                z_grad_term.col(0).array() = -dZ_times_bz.col(t1).array();  // was: -dZ_dmu_neg*Bound_Z
                prod_container.array()     = L_Omega_double(t1 + 1, t1) * z_grad_term.col(0).array();
                grad_prob.col(1).array()   = dphi_over_L.col(t1 + 1).array() * prod_container.array();
                
                if (n_class == 1) {
                  
                      if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) += (prob_recip.col(t1).array() * grad_prob.col(0).array() + prob_recip.col(t1+1).array() * grad_prob.col(1).array()).sum();
                  
                } else {
                  
                      if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) += (common_grad_term_1.col(t1).array() * (prob.col(t1+1).array() * grad_prob.col(0).array() + prob.col(t1).array()   * grad_prob.col(1).array())).sum();
                  
                }
          }
          
          // ---- remaining diagonals ----
          for (int i = 3; i < n_tests + 1; i++) {
            
                int t1 = n_tests - i;
                
                grad_prob.col(0).array()   = dphi_times_bz.col(t1).array();
                z_grad_term.col(0).array() = -dZ_times_bz.col(t1).array();  // was: -dZ_dmu_neg*Bound_Z
                prod_container.array()     = L_Omega_double(t1+1, t1) * z_grad_term.col(0).array();
                grad_prob.col(1).array()   = dphi_over_L.col(t1+1).array() * prod_container.array();
                
                for (int ii = 1; ii < i - 1; ii++) {
                  z_grad_term.col(ii).array()  = -dZ_dmu_neg.col(t1+ii).array() * prod_container.array();
                  prod_container.array()       = (L_Omega_double.row(t1+ii+1).segment(t1, ii+1) * z_grad_term.leftCols(ii+1).transpose()).transpose().array();
                  grad_prob.col(ii+1).array()  = dphi_over_L.col(t1+ii+1).array() * prod_container.array();
                }
                
                if (n_class != 1) {
                  
                      derivs_chain_container_vec.setZero();
                      for (int iii = 0; iii < i; iii++) {
                        derivs_chain_container_vec.array() += grad_prob.col(iii).array() * prop_rowwise_prod_temp.col(t1).array() * prob_recip.col(t1+iii).array();
                      }
                      if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) += (common_grad_term_1.col(t1).array() * derivs_chain_container_vec.array()).sum();
                  
                } else {
                  
                      derivs_chain_container_vec.setZero();
                      for (int iii = 0; iii < i; iii++) {
                        derivs_chain_container_vec.array() += grad_prob.col(iii).array() * prob_recip.col(t1+iii).array();
                      }
                      if (compute_final_scalar_grad) U_Omega_grad_array(t1, t1) += derivs_chain_container_vec.sum();
                  
                }
            
          }
          
        }
        
        // ---- off-diagonals: last row ----
        {
          int t1_dash = 0;
          int t1 = n_tests - 1;
          int t2 = n_tests - 2;
          
          if (n_class == 1) {
            if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) += (prob_recip.col(t1).array() * dphi_over_L.col(t1).array() * Z_std_norm.col(t2).array()).sum();
          } else {
            if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) += (common_grad_term_1.col(t1).array() * dphi_over_L.col(t1).array() * Z_std_norm.col(t2).array()).sum();
          }
          
          if (t1 > 1) {
            t2 = n_tests - 3;
            if (n_class == 1) {
              if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) += (prob_recip.col(t1).array() * dphi_over_L.col(t1).array() * Z_std_norm.col(t2).array()).sum();
            } else {
              if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) += (common_grad_term_1.col(t1).array() * dphi_over_L.col(t1).array() * Z_std_norm.col(t2).array()).sum();
            }
          }
          
          if (t1 > 2) {
            for (int t2_dash = 3; t2_dash < n_tests; t2_dash++) {
              t2 = n_tests - (t1_dash + t2_dash + 1);
              if (t2 < n_tests - 1) {
                if (n_class == 1) {
                  if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) += (prob_recip.col(t1).array() * dphi_over_L.col(t1).array() * Z_std_norm.col(t2).array()).sum();
                } else {
                  if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) += (common_grad_term_1.col(t1).array() * dphi_over_L.col(t1).array() * Z_std_norm.col(t2).array()).sum();
                }
              }
            }
          }
        }
        
        // ---- off-diagonals: remaining rows ----
        {
          z_grad_term.setZero();
          grad_prob.setZero();
          
          for (int t1_dash = 1; t1_dash < n_tests - 1; t1_dash++) {
            int t1 = n_tests - (t1_dash + 1);
            
            for (int t2_dash = t1_dash + 1; t2_dash < n_tests; t2_dash++) {
                  
                  int t2 = n_tests - (t2_dash + 1);
                  
                  prod_container.array()     = Z_std_norm.col(t2).array();
                  grad_prob.col(0).array()   = dphi_over_L.col(t1).array() * prod_container.array();
                  z_grad_term.col(0).array() = dZ_dmu_neg.col(t1).array() * (-prod_container.array());
                  
                  if (t1_dash > 0) {
                    
                      for (int t1_dash_dash = 1; t1_dash_dash < t1_dash + 1; t1_dash_dash++) {
                        
                          if (t1_dash_dash > 1)  z_grad_term.col(t1_dash_dash-1).array() = dZ_dmu_neg.col(t1+t1_dash_dash-1).array() * (-prod_container.array());
                          prod_container.array() = (z_grad_term.leftCols(t1_dash_dash) * L_Omega_double.row(t1+t1_dash_dash).segment(t1, t1_dash_dash).transpose()).array();
                          grad_prob.col(t1_dash_dash).array() = dphi_over_L.col(t1+t1_dash_dash).array() * prod_container.array();
                        
                      }
                    
                  }
                  
                  if (n_class == 1) {
                    
                        derivs_chain_container_vec.setZero();
                        for (int ii = 0; ii < t1_dash + 1; ii++) {
                          derivs_chain_container_vec.array() += grad_prob.col(ii).array() * prob_recip.col(t1+ii).array();
                        }
                        if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) += derivs_chain_container_vec.sum();
                    
                  } else {
                    
                        derivs_chain_container_vec.setZero();
                        for (int ii = 0; ii < t1_dash + 1; ii++) {
                          derivs_chain_container_vec.array() += grad_prob.col(ii).array() * prop_rowwise_prod_temp.col(t1).array() * prob_recip.col(t1+ii).array();
                        }
                        if (compute_final_scalar_grad) U_Omega_grad_array(t1, t2) += (common_grad_term_1.col(t1).array() * derivs_chain_container_vec.array()).sum();
                    
                  }
                  
            }
            
          }
          
        }
        
}

// ============================================================================
// chain_rule_C_to_C_raw: Chain rule from ∂f/∂C to ∂f/∂C_raw
//
// Given: grad_wrt_C (length n_thr, gradient w.r.t. ordered cutpoints C)
//        C_raw_vec (length n_thr, unconstrained parameters)
//
// The transformation is:
//   C[0] = C_raw[0]
//   C[k] = C[k-1] + exp(C_raw[k])  for k >= 1
//
// The Jacobian ∂C/∂C_raw is lower-triangular:
//   ∂C[j]/∂C_raw[0] = 1               for all j
//   ∂C[j]/∂C_raw[k] = exp(C_raw[k])   for j >= k, k >= 1
//   ∂C[j]/∂C_raw[k] = 0               for j < k
//
// So: ∂f/∂C_raw[0] = Σ_j ∂f/∂C[j]
//     ∂f/∂C_raw[k] = exp(C_raw[k]) * Σ_{j>=k} ∂f/∂C[j]   for k >= 1
//
// Computed efficiently using a suffix sum (right to left).
// ============================================================================
// inline Eigen::Matrix<double, -1, 1> chain_rule_C_to_C_raw( const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> grad_wrt_C,
//                                                            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> C_raw_vec
// ) {
//         
//         const int n_thr = C_raw_vec.size();
//         Eigen::Matrix<double, -1, 1> grad_wrt_C_raw(n_thr);
//         
//         // Suffix sum from right to left
//         double suffix_sum = 0.0;
//         for (int k = n_thr - 1; k >= 1; --k) {
//           suffix_sum += grad_wrt_C(k);
//           grad_wrt_C_raw(k) = std::exp(C_raw_vec(k)) * suffix_sum;
//         }
//         
//         suffix_sum += grad_wrt_C(0);
//         grad_wrt_C_raw(0) = suffix_sum;
//         
//         return grad_wrt_C_raw;
//   
// }


inline Eigen::Matrix<double, -1, 1> chain_rule_C_to_C_raw( const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> grad_wrt_C,
                                                           const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> C_raw_vec
) {
  
          const int n_thr = C_raw_vec.size();
          Eigen::Matrix<double, -1, 1> grad_wrt_C_raw(n_thr);
          
          double suffix_sum = 0.0;
          for (int k = n_thr - 1; k >= 1; --k) {
            
              suffix_sum += grad_wrt_C(k);
              double exp_val = std::exp(C_raw_vec(k));
              ////
              double result = exp_val * suffix_sum;
              if ((std::isinf(result)) || (std::isnan(result))) { 
                grad_wrt_C_raw(k) = 0.0;
              } else { 
                grad_wrt_C_raw(k) = result;
              }
              // grad_wrt_C_raw(k) = std::isfinite(result) ? result : 0.0;
              // grad_wrt_C_raw(k) = exp_val * suffix_sum;
              
              if (g_debug_cutpoint_grads && (std::isnan(grad_wrt_C_raw(k)) || std::isinf(grad_wrt_C_raw(k)))) {
                std::cout << "CHAIN_RULE NaN: k=" << k
                          << " C_raw=" << C_raw_vec(k)
                          << " exp=" << exp_val
                          << " suffix_sum=" << suffix_sum
                          << " grad_C(k)=" << grad_wrt_C(k)
                          << " result=" << grad_wrt_C_raw(k) << std::endl;
                std::cout << "  full C_raw: " << C_raw_vec.transpose() << std::endl;
                std::cout << "  full grad_C: " << grad_wrt_C.transpose() << std::endl;
                std::cout.flush();
              }
            
          }
          
          suffix_sum += grad_wrt_C(0);
          grad_wrt_C_raw(0) = suffix_sum;
          
          return grad_wrt_C_raw;
  
}


// ============================================================================
// For reference: how to call these in the lp_grad function
// ============================================================================
//
// In the chunk loop, AFTER computing phi_Bound_Z and phi_Upper_Bound_Z:
//
// 1. Call fn_MVOP_grad_prep instead of fn_MVP_grad_prep:
//    ────────────────────────────────────────────────────
//    fn_MVOP_grad_prep(prob[c], y_sign, y_m_y_sign_x_u, u_array,
//                      L_Omega_recip_double[c],
//                      prev(0, c), prob_n_recip,
//                      phi_Z_recip, phi_Bound_Z, phi_Upper_Bound_Z,
//                      prob_recip,
//                      prob_rowwise_prod_temp, prob_recip_rowwise_prod_temp,
//                      prob_rowwise_prod_temp_all, common_grad_term_1,
//                      dphi_over_L, dZ_dmu_neg,      // <── OUTPUT (replaces old names)
//                      n_binary_tests, Model_args_as_cpp_struct);
//
// 2. Call EXISTING gradient functions with dphi_over_L and dZ_dmu_neg
//    as the "y_sign_chunk_times..." and "y_m_ysign_x_u_array_times..." arguments:
//    ─────────────────────────────────────────────────────────────────────────────
//    fn_MVP_compute_nuisance_grad_v2(u_grad_array_CM_chunk_block,
//                                    phi_Z_recip,
//                                    common_grad_term_1,
//                                    L_Omega_double[c],
//                                    prob[c], prob_recip,
//                                    prob_rowwise_prod_temp,
//                                    dphi_over_L,      // <── was y_sign_chunk_times...
//                                    dZ_dmu_neg,        // <── was y_m_ysign_x_u_array_times...
//                                    z_grad_term, grad_prob,
//                                    prod_container_or_inc_array,
//                                    derivs_chain_container_vec,
//                                    Model_args_as_cpp_struct);
//
//    fn_MVP_compute_coefficients_grad_v3(c, beta_grad_local[c],
//                                        chunk_counter, n_covariates_max,
//                                        common_grad_term_1,
//                                        L_Omega_double[c],
//                                        prob[c], prob_recip,
//                                        prob_rowwise_prod_temp,
//                                        dphi_over_L,      // <── was y_sign_chunk_times...
//                                        dZ_dmu_neg,        // <── was y_m_ysign_x_u_array_times...
//                                        z_grad_term, grad_prob,
//                                        prod_container_or_inc_array,
//                                        derivs_chain_container_vec,
//                                        true, Model_args_as_cpp_struct);
//
//    fn_MVP_compute_L_Omega_grad_v3(U_Omega_grad_local[c],
//                                   common_grad_term_1,
//                                   L_Omega_double[c],
//                                   prob[c], prob_recip,
//                                   Bound_Z[c], Z_std_norm[c],
//                                   prob_rowwise_prod_temp,
//                                   dphi_over_L,      // <── was y_sign_chunk_times...
//                                   dZ_dmu_neg,        // <── was y_m_ysign_x_u_array_times...
//                                   z_grad_term, grad_prob,
//                                   prod_container_or_inc_array,
//                                   derivs_chain_container_vec,
//                                   true, Model_args_as_cpp_struct);
//
// 3. Call fn_MVOP_compute_cutpoint_grad for each ordinal test:
//    ──────────────────────────────────────────────────────────
//    for (int t_ord = 0; t_ord < n_ordinal_tests; ++t_ord) {
//        int t = n_binary_tests + t_ord;
//        int K_t = n_cutpoints_per_outcome_vec(t) + 1;  // n_cat
//        
//        fn_MVOP_compute_cutpoint_grad(
//            c, cutpoint_grad_local[c],  // (n_cutpoints_max × n_ordinal_tests)
//            t, t_ord, K_t,
//            common_grad_term_1,
//            L_Omega_double[c],
//            prob[c], prob_recip,
//            prob_rowwise_prod_temp,
//            phi_Bound_Z, phi_Upper_Bound_Z,
//            phi_Z_recip, u_array, y_chunk,
//            dphi_over_L, dZ_dmu_neg,
//            z_grad_term, grad_prob,
//            prod_container_or_inc_array,
//            derivs_chain_container_vec,
//            true, Model_args_as_cpp_struct);
//    }
//
// 4. After the chunk loop, chain-rule cutpoint grads through C_raw → C:
//    ─────────────────────────────────────────────────────────────────────
//    int out_idx = 1 + n_us + n_corrs + n_covariates_total;
//    for (int c = 0; c < n_class; ++c) {
//        for (int t_ord = 0; t_ord < n_ordinal_tests; ++t_ord) {
//            int n_thr_t = n_cutpoints_per_outcome_vec(n_binary_tests + t_ord);
//            Eigen::VectorXd grad_wrt_C = cutpoint_grad_array[c].col(t_ord).head(n_thr_t);
//            Eigen::VectorXd grad_wrt_C_raw = chain_rule_C_to_C_raw(
//                grad_wrt_C, C_raw[c].col(t_ord).head(n_thr_t));
//            out_mat.segment(out_idx, n_thr_t) += grad_wrt_C_raw;
//            out_idx += n_thr_t;
//        }
//    }

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 