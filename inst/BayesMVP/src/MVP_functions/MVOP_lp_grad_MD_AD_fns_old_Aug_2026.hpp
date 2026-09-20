
//// MVOP_lp_grad_MD_AD_fns.hpp

#pragma once




#include <Eigen/Dense>

#include <unsupported/Eigen/SpecialFunctions>























//////////////////////////////////////////////////------------------------------------------------------------------------------------------------------------------------
//// Resize the workspace matrices for a given chunk size (only called when the current ws
//// dimensions differ - i.e. once, when switching from the full-chunk size to the remainder size):
inline void fn_MVOP_resize_ws_for_chunk(   LC_MVP_workspace_struct &ws,
                                                  const int chunk_size,
                                                  const int n_tests,
                                                  const int n_class
) {
      
      for (int c = 0; c < 2; c++) {
          ws.Z_std_norm[c].resize(chunk_size, n_tests);
          ws.Bound_Z[c].resize(chunk_size, n_tests);
          ws.Bound_U_Phi_Bound_Z[c].resize(chunk_size, n_tests);
          ws.prob[c].resize(chunk_size, n_tests);
          ws.Phi_Z[c].resize(chunk_size, n_tests);
      }
      ///////////////////////////////////////////////
      ws.y1_log_prob.resize(chunk_size, n_tests);
      ws.phi_Z_recip.resize(chunk_size, n_tests);
      ws.phi_Bound_Z.resize(chunk_size, n_tests);
      ws.u_grad_array_CM_chunk.resize(chunk_size, n_tests);
      ws.common_grad_term_1.resize(chunk_size, n_tests);
      ws.y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(chunk_size, n_tests);
      ws.y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(chunk_size, n_tests);
      ws.prob_rowwise_prod_temp.resize(chunk_size, n_tests);
      ws.prob_recip_rowwise_prod_temp.resize(chunk_size, n_tests);
      ///////////////////////////////////////////////
      ws.prod_container_or_inc_array.resize(chunk_size);
      ws.derivs_chain_container_vec.resize(chunk_size);
      ws.prob_rowwise_prod_temp_all.resize(chunk_size);
      ///////////////////////////////////////////////
      ws.dphi_direct_Cj.resize(chunk_size); //// ordinal-only
      ws.dZ_direct_Cj.resize(chunk_size); //// ordinal-only
      ///////////////////////////////////////////////
      ws.grad_prob.resize(chunk_size, n_tests);
      ws.z_grad_term.resize(chunk_size, n_tests);
      ws.y_chunk.resize(chunk_size, n_tests);
      ws.u_array.resize(chunk_size, n_tests);
      ws.y_sign.resize(chunk_size, n_tests);
      ws.y_m_y_sign_x_u.resize(chunk_size, n_tests);
      ws.u_grad_array_CM_chunk_block.resize(chunk_size, n_tests);
      ws.u_unc_vec_chunk.resize(chunk_size * n_tests);
      ws.u_vec_chunk.resize(chunk_size * n_tests);
      ws.du_wrt_duu_chunk.resize(chunk_size * n_tests);
      ws.d_J_wrt_duu_chunk.resize(chunk_size * n_tests);
      ///////////////////////////////////////////////
      ws.lp_array.resize(chunk_size, n_class);
      ///////////////////////////////////////////////
      ws.prob_n.resize(chunk_size);
      ws.prob_n_recip.resize(chunk_size);
      ws.log_sum_result.resize(chunk_size);
      ws.container_max_logs.resize(chunk_size);
      ws.rowwise_log_sum.resize(chunk_size);
      ws.rowwise_prod.resize(chunk_size);
      ws.rowwise_sum.resize(chunk_size);
      ws.log_lik_chunk.resize(chunk_size);
      ws.prob_recip.resize(chunk_size, n_tests);
      ///////////////////////////////////////////////
      for (int c = 0; c < n_class; c++) {
        ws.Upper_Bound_Z[c].resize(chunk_size, n_tests); //// ordinal-only
      }
      ///////////////////////////////////////////////
      ws.phi_Upper_Bound_Z.resize(chunk_size, n_tests); //// ordinal-only
      ws.dphi_over_L.resize(chunk_size, n_tests); //// ordinal-only
      ws.dZ_dmu_neg.resize(chunk_size, n_tests); //// ordinal-only
      ///////////////////////////////////////////////
      ws.dphi_times_bz.resize(chunk_size, n_tests); //// ordinal-only
      ws.dZ_times_bz.resize(chunk_size, n_tests); //// ordinal-only
      ///////////////////////////////////////////////
      ws.log_prev_per_obs_given_c.resize(chunk_size); 
      ws.prev_per_obs_given_c.resize(chunk_size);
  
}


// //////////////////////////////////////////////////------------------------------------------------------------------------------------------------------------------------
// //// Internal per-chunk routine (lp + grad for ONE chunk). Called once per full chunk with the
// //// SIMD-configured struct, and once for the remainder chunk with the "Stan" (scalar) copy
// //// ("Model_args_last_chunk") - so ALL helper fns run scalar on the remainder chunk:
// inline void fn_lp_grad_MVOP_LC_Pinkney_NoLog_process_chunk(     Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
//                                                                        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
//                                                                        const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
//                                                                        const std::string &grad_option,
//                                                                        const int chunk_counter,
//                                                                        const int chunk_size,
//                                                                        const int chunk_size_orig,
//                                                                        const int N,
//                                                                        const int n_params,
//                                                                        const int n_tests,
//                                                                        const int n_class,
//                                                                        const int n_binary_tests,
//                                                                        const int n_ordinal_tests,
//                                                                        const Eigen::Matrix<int, -1, 1> &ord_idx_of_test, //// ----
//                                                                        const int n_covariates_max,
//                                                                        const int n_pops,
//                                                                        const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec,
//                                                                        const Eigen::Matrix<int, -1, 1> &n_cat_per_ord_test,
//                                                                        const Eigen::Matrix<int, -1, 1> &n_thr_per_ord_test,
//                                                                        const Eigen::Matrix<int, -1, 1> &pop_ind,
//                                                                        const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> &X,
//                                                                        const std::vector<Eigen::Matrix<double, -1, -1>> &beta_double_array,
//                                                                        const std::vector<Eigen::Matrix<double, -1, -1>> &C,
//                                                                        const std::vector<Eigen::Matrix<double, -1, -1>> &L_Omega_double,
//                                                                        const std::vector<Eigen::Matrix<double, -1, -1>> &L_Omega_recip_double,
//                                                                        const Eigen::Matrix<double, -1, -1> &prev_mat,
//                                                                        const Eigen::Matrix<double, -1, -1> &log_prev_mat_small,
//                                                                        double &log_jac_u,
//                                                                        std::vector<Eigen::Matrix<double, -1, -1>> &beta_grad_array,
//                                                                        std::vector<Eigen::Matrix<double, -1, -1>> &U_Omega_grad_array,
//                                                                        std::vector<Eigen::Matrix<double, -1, -1>> &cutpoint_grad_array,
//                                                                        Eigen::Matrix<double, -1, -1> &prev_grad_mat,
//                                                                        LC_MVP_workspace_struct &ws,
//                                                                        const Model_fn_args_struct &Model_args_for_chunk
// ) {
//   
//       const double sqrt_2_pi_recip = 1.0 / sqrt(2.0 * M_PI);
//       const double Inf = std::numeric_limits<double>::infinity();
//       
//       // const double BOUND_CLAMP = 8.0; //// finite stand-in for +-Inf: Phi(+-15) = 0/1 to double precision
//       
//       //// direct fn_EIGEN_double / log_sum_exp_general calls read the vect-type strings from the
//       //// PASSED struct, so the remainder chunk automatically runs scalar:
//       const std::string vect_type_exp = Model_args_for_chunk.Model_args_strings(3);
//       const std::string vect_type_log = Model_args_for_chunk.Model_args_strings(4);
//       
//       //// resize the workspace if this chunk size differs from the current ws size
//       //// (happens once, on the remainder chunk):
//       if ( (ws.y_chunk.rows() != chunk_size) || (ws.y_chunk.cols() != n_tests) ) {
//         fn_MVOP_resize_ws_for_chunk(ws, chunk_size, n_tests, n_class);
//       }
//       
//       ////////////////////////////////////////////////
//       // ---- workspace references ----
//       ////////////////////////////////////////////////
//       std::vector<Eigen::Matrix<double, -1, -1>> &Z_std_norm = ws.Z_std_norm;
//       std::vector<Eigen::Matrix<double, -1, -1>> &Bound_Z = ws.Bound_Z;
//       std::vector<Eigen::Matrix<double, -1, -1>> &Bound_U_Phi_Bound_Z = ws.Bound_U_Phi_Bound_Z;
//       std::vector<Eigen::Matrix<double, -1, -1>> &prob = ws.prob;  
//       std::vector<Eigen::Matrix<double, -1, -1>> &Phi_Z = ws.Phi_Z; 
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, -1> &y1_log_prob = ws.y1_log_prob;
//       Eigen::Matrix<double, -1, -1> &phi_Z_recip = ws.phi_Z_recip;
//       Eigen::Matrix<double, -1, -1> &phi_Bound_Z = ws.phi_Bound_Z;
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, -1> &u_grad_array_CM_chunk = ws.u_grad_array_CM_chunk;
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, -1> &common_grad_term_1 = ws.common_grad_term_1;
//       Eigen::Matrix<double, -1, -1> &y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = ws.y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip;
//       Eigen::Matrix<double, -1, -1> &y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = ws.y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip;
//       Eigen::Matrix<double, -1, -1> &prob_rowwise_prod_temp = ws.prob_rowwise_prod_temp;
//       Eigen::Matrix<double, -1, -1> &prob_recip_rowwise_prod_temp = ws.prob_recip_rowwise_prod_temp;
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, 1> &prod_container_or_inc_array = ws.prod_container_or_inc_array;
//       Eigen::Matrix<double, -1, 1> &derivs_chain_container_vec = ws.derivs_chain_container_vec;
//       Eigen::Matrix<double, -1, 1> &prob_rowwise_prod_temp_all = ws.prob_rowwise_prod_temp_all;
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, 1> &dphi_direct_Cj = ws.dphi_direct_Cj; // ordinal-only
//       Eigen::Matrix<double, -1, 1> &dZ_direct_Cj = ws.dZ_direct_Cj; // ordinal-only
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, -1> &grad_prob = ws.grad_prob;
//       Eigen::Matrix<double, -1, -1> &z_grad_term = ws.z_grad_term;
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, -1> &y_chunk = ws.y_chunk;
//       Eigen::Matrix<double, -1, -1> &u_array = ws.u_array;
//       Eigen::Matrix<double, -1, -1> &y_sign = ws.y_sign;
//       Eigen::Matrix<double, -1, -1> &y_m_y_sign_x_u = ws.y_m_y_sign_x_u;
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, -1> &u_grad_array_CM_chunk_block = ws.u_grad_array_CM_chunk_block;
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, 1> &u_unc_vec_chunk = ws.u_unc_vec_chunk;
//       Eigen::Matrix<double, -1, 1> &u_vec_chunk = ws.u_vec_chunk;
//       Eigen::Matrix<double, -1, 1> &du_wrt_duu_chunk = ws.du_wrt_duu_chunk;
//       Eigen::Matrix<double, -1, 1> &d_J_wrt_duu_chunk = ws.d_J_wrt_duu_chunk;
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, -1> &lp_array = ws.lp_array;
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, 1> &prob_n = ws.prob_n;
//       Eigen::Matrix<double, -1, 1> &prob_n_recip = ws.prob_n_recip;
//       Eigen::Matrix<double, -1, 1> &log_sum_result = ws.log_sum_result;
//       Eigen::Matrix<double, -1, 1> &container_max_logs = ws.container_max_logs;
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, 1> &rowwise_log_sum = ws.rowwise_log_sum;
//       Eigen::Matrix<double, -1, 1> &rowwise_prod = ws.rowwise_prod;
//       Eigen::Matrix<double, -1, 1> &rowwise_sum = ws.rowwise_sum;
//       Eigen::Matrix<double, -1, 1> &log_lik_chunk = ws.log_lik_chunk;
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, -1> &prob_recip = ws.prob_recip;
//       ////////////////////////////////////////////////
//       std::vector<Eigen::Matrix<double, -1, -1>> &Upper_Bound_Z = ws.Upper_Bound_Z; // ordinal-only
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, -1> &phi_Upper_Bound_Z = ws.phi_Upper_Bound_Z; // ordinal-only
//       Eigen::Matrix<double, -1, -1> &dphi_over_L       = ws.dphi_over_L; // ordinal-only
//       Eigen::Matrix<double, -1, -1> &dZ_dmu_neg        = ws.dZ_dmu_neg; // ordinal-only
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, -1> &dphi_times_bz = ws.dphi_times_bz; //// ordinal-only
//       Eigen::Matrix<double, -1, -1> &dZ_times_bz   = ws.dZ_times_bz; //// ordinal-only
//       ////////////////////////////////////////////////
//       Eigen::Matrix<double, -1, 1> &log_prev_per_obs_given_c = ws.log_prev_per_obs_given_c;
//       Eigen::Matrix<double, -1, 1> &prev_per_obs_given_c     = ws.prev_per_obs_given_c;
//       
//       ////-----------------------------------------------
//       u_grad_array_CM_chunk.setZero(); //// reset between chunks as re-using same container
//       
//       y_chunk = y_ref.middleRows( chunk_size_orig * chunk_counter, chunk_size).cast<double>();
//       
//       //// Nuisance parameter transformation step
//       u_unc_vec_chunk = theta_us_vec_ref.segment( chunk_size_orig * n_tests * chunk_counter, chunk_size * n_tests);
//       fn_MVP_compute_nuisance( u_vec_chunk, u_unc_vec_chunk, Model_args_for_chunk);
//       log_jac_u += fn_MVP_compute_nuisance_log_jac_u( u_vec_chunk, u_unc_vec_chunk, Model_args_for_chunk);
//       
//       u_array = u_vec_chunk.reshaped(chunk_size, n_tests);
//       y_sign.array() =  y_chunk.array() + (y_chunk.array() - 1.0) ;
//       y_m_y_sign_x_u.array() =  y_chunk.array() - (y_sign.array() * u_array.array());
//       ////-----------------------------------------------
//       
//       // ---- Class loop (identical to original) ----
//       //// START of c loop
//       for (int c = 0; c < n_class; c++) {
//         
//         ////-----------------------------------------------
//         if (n_class > 1) {
//           
//           for (int n = 0; n < chunk_size; ++n) {
//             int n_global = chunk_size_orig * chunk_counter + n;
//             int g = pop_ind(n_global);                          // was missing `int`
//             log_prev_per_obs_given_c(n) = log_prev_mat_small(g, c);
//             prev_per_obs_given_c(n) = prev_mat(g, c);           // was `prev`, now `prev_mat`
//           }
//           
//         }
//         ////---------------------------------------------
//         
//         prod_container_or_inc_array.setZero(); //// reset to 0
//         
//         //// start of t loop
//         for (int t = 0; t < n_tests; t++) {
//           
//           ////-----------------------------------------------
//           if (n_covariates_max > 1) {
//             
//             Eigen::Matrix<double, -1, 1> Xbeta_given_class_c_col_t = X[c][t].block(chunk_size_orig * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(c, t)).cast<double>()
//                                                                    * beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t));
//             
//             const int t_ord_slot = ord_idx_of_test(t);
//             if (t_ord_slot < 0) { // binary
//             // if (t < n_binary_tests) { // For binary: Bound_Z = L_recip * (-Xbeta - prod) 
//               
//                   Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * (-1.0*( Xbeta_given_class_c_col_t.array() + prod_container_or_inc_array.array())) ;
//               
//             } else { // Store μ_t temporarily, compute bounds per-observation
//               
//                   // const int t_ord = t - n_binary_tests;
//                   const int t_ord = t_ord_slot;
//                   const int n_thr_t = n_thr_per_ord_test(t_ord);
//                   const int K_t = n_cat_per_ord_test(t_ord);
//                   
//                   for (int n = 0; n < chunk_size; ++n) {
//                     
//                         double mu_t = Xbeta_given_class_c_col_t(n) + prod_container_or_inc_array(n);
//                         int y_n = static_cast<int>(y_chunk(n, t));
//                         double L_tt_inv = L_Omega_recip_double[c](t, t);
//                         
//                         // double lb = (y_n == 1)   ? -BOUND_CLAMP : (C[c](y_n - 2, t_ord) - mu_t) * L_tt_inv; // Lower bound
//                         // double ub = (y_n == K_t) ?  BOUND_CLAMP : (C[c](y_n - 1, t_ord) - mu_t) * L_tt_inv; // Upper bound
//                         double lb = (y_n == 1)   ? -Inf : (C[c](y_n - 2, t_ord) - mu_t) * L_tt_inv; // Lower bound
//                         double ub = (y_n == K_t) ?  Inf : (C[c](y_n - 1, t_ord) - mu_t) * L_tt_inv; // Upper bound
//                         
//                         //// Store for GHK:
//                         Bound_Z[c](n, t) = lb;           // lower_bz (re-using Bound_Z container for GHK lower-bound)
//                         Upper_Bound_Z[c](n, t) = ub;     // upper_bz (using the "Upper_Bound_Z", ordinal-only workspace matrix)
//                     
//                   } // end of n loop
//               
//             }
//             
//           } else {  // intercept-only
//             
//                 const int t_ord_slot = ord_idx_of_test(t);
//                 if (t_ord_slot < 0) { // binary
//                 // if (t < n_binary_tests) { // For binary: Bound_Z = L_recip * (-Xbeta - prod) 
//                   
//                       Bound_Z[c].col(t).array() = L_Omega_recip_double[c](t, t) * ( -1.0*( beta_double_array[c](0, t) + prod_container_or_inc_array.array() ) ) ;
//                   
//                 } else { 
//                       
//                       // const int t_ord = t - n_binary_tests;
//                       const int t_ord = t_ord_slot;
//                       const int n_thr_t = n_thr_per_ord_test(t_ord);
//                       const int K_t = n_cat_per_ord_test(t_ord);
//                       
//                       for (int n = 0; n < chunk_size; ++n) {
//                             
//                             double mu_t = beta_double_array[c](0, t) + prod_container_or_inc_array(n);
//                             int y_n = static_cast<int>(y_chunk(n, t));
//                             double L_tt_inv = L_Omega_recip_double[c](t, t);
//                             
//                             // double lb = (y_n == 1)   ? -BOUND_CLAMP : (C[c](y_n - 2, t_ord) - mu_t) * L_tt_inv; // Lower bound
//                             // double ub = (y_n == K_t) ?  BOUND_CLAMP : (C[c](y_n - 1, t_ord) - mu_t) * L_tt_inv; // Upper bound
//                             double lb = (y_n == 1)   ? -Inf : (C[c](y_n - 2, t_ord) - mu_t) * L_tt_inv; // Lower bound
//                             double ub = (y_n == K_t) ?  Inf : (C[c](y_n - 1, t_ord) - mu_t) * L_tt_inv; // Upper bound
//                             
//                             //// Store for GHK:
//                             Bound_Z[c](n, t) = lb;           // lower_bz (re-using Bound_Z container for GHK lower-bound)
//                             Upper_Bound_Z[c](n, t) = ub;     // upper_bz (using the "Upper_Bound_Z", ordinal-only workspace matrix)
//                         
//                       } // end of n loop
//                   
//                 }
//             
//           }
//           
//           //-----------------------------------------------
//           //-----------------------------------------------
//           // compute/update important log-lik quantities for GHK-MVP
//           fn_MVOP_compute_lp_GHK_cols(  t,
//                                         Bound_U_Phi_Bound_Z[c],
//                                         Phi_Z[c],
//                                         Z_std_norm[c],
//                                         prob[c],
//                                         y1_log_prob,
//                                         ////
//                                         Bound_Z[c],
//                                         y_chunk,
//                                         u_array,
//                                         ////
//                                         Upper_Bound_Z[c],
//                                         ////
//                                         (ord_idx_of_test(t) < 0),   // is_binary
//                                         Model_args_for_chunk);
//           ////-----------------------------------------------
//           
//           ////-----------------------------------------------
//           if (t < n_tests - 1) {
//             auto L_Omega_row = L_Omega_double[c].row(t + 1);
//             prod_container_or_inc_array = Z_std_norm[c].leftCols(t + 1)  *   L_Omega_row.head(t+1).transpose();
//           }
//           ////-----------------------------------------------
//           
//         }   //// end of t loop
//         
//         ////-----------------------------------------------
//         if (n_class > 1) {
//           rowwise_sum = y1_log_prob.rowwise().sum();
//           rowwise_sum.array() += log_prev_per_obs_given_c.array();
//           lp_array.col(c) = rowwise_sum;
//         } else {
//           rowwise_sum = y1_log_prob.rowwise().sum();
//           lp_array.col(0) = rowwise_sum;
//         }
//         ////-----------------------------------------------
//         
//       }  //// end of c loop
//       
//       ////-----------------------------------------------  
//       if (n_class > 1) {
//         
//         log_sum_exp_general( lp_array,
//                              vect_type_exp,
//                              vect_type_log,
//                              log_sum_result,
//                              container_max_logs);
//         const int index_start = 1 + n_params + chunk_size_orig * chunk_counter;
//         out_mat.segment(index_start, chunk_size) = log_sum_result;
//         
//       } else {
//         
//         out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size)  = lp_array.col(0);
//         
//       }
//       ////-----------------------------------------------  
//       ////-----------------------------------------------
//       log_lik_chunk = out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size);
//       prob_n  =  fn_EIGEN_double(log_lik_chunk, "exp",  vect_type_exp);
//       prob_n_recip  =  stan::math::inv(prob_n); // this CANNOT be a temporary otherwise it created a "dangling reference", since prob_n is a temporary!!
//       ////-----------------------------------------------
//       
//       /////////////////  ------------------------- compute grad  ---------------------------------------------------------------------------------
//       for (int c = 0; c < n_class; c++) {
//         
//         //// ---- re-fill prev for this class (needed for fn_MVOP_grad_prep) ----
//         if (n_class > 1) {
//           for (int n = 0; n < chunk_size; ++n) {
//             int n_global = chunk_size_orig * chunk_counter + n;
//             int g = pop_ind(n_global);
//             prev_per_obs_given_c(n) = prev_mat(g, c);
//           }
//         }
//         
//         ////-----------------------------------------------
//         prob_recip = stan::math::inv(prob[c]);  
//         ////-----------------------------------------------
//         
//         ////-----------------------------------------------
//         //// compute/update important log-lik quantities for GHK-MVP
//         for (int t = 0; t < n_tests; t++) {
//           
//               fn_MVP_compute_phi_Z_recip_cols(   t,
//                                                  phi_Z_recip, // computing this
//                                                  Phi_Z[c], Z_std_norm[c], Model_args_for_chunk);
//               
//               // if (t < n_binary_tests) {
//               if (ord_idx_of_test(t) < 0) {
//                 
//                     fn_MVP_compute_phi_Bound_Z_cols( t, 
//                                                      phi_Bound_Z, // computing this
//                                                      Bound_U_Phi_Bound_Z[c], Bound_Z[c], Model_args_for_chunk);
//                 
//               } else { // for ordinal tests
//                 
//                     // φ(lower_bz) — stored in phi_Bound_Z
//                     // phi_Bound_Z.col(t).array() = (-(Bound_Z[c].col(t).array().square()) * 0.5).exp() * sqrt_2_pi_recip;
//                     phi_Bound_Z.col(t).array() = fn_EIGEN_double((-(Bound_Z[c].col(t).array().square()) * 0.5), "exp", vect_type_exp) * sqrt_2_pi_recip;
//                     
//                     // φ(upper_bz) — stored in phi_Upper_Bound_Z
//                     // phi_Upper_Bound_Z.col(t).array() = (-(Upper_Bound_Z[c].col(t).array().square()) * 0.5).exp() * sqrt_2_pi_recip;
//                     phi_Upper_Bound_Z.col(t).array() = fn_EIGEN_double((-(Upper_Bound_Z[c].col(t).array().square()) * 0.5), "exp", vect_type_exp) * sqrt_2_pi_recip;
//                     
//                     // Handle ±∞ bounds → φ = 0
//                     for (int i = 0; i < chunk_size; ++i) {
//                       if (std::isinf(Bound_Z[c](i, t)))       phi_Bound_Z(i, t) = 0.0;
//                       if (std::isinf(Upper_Bound_Z[c](i, t))) phi_Upper_Bound_Z(i, t) = 0.0;
//                     }
//                 
//               }
//           
//         }
//         ////-----------------------------------------------
//         if (grad_option != "none") {
//           
//               fn_MVOP_grad_prep( prob[c],
//                                  y_sign,
//                                  y_m_y_sign_x_u,
//                                  u_array,
//                                  L_Omega_recip_double[c],
//                                  ////
//                                  prev_per_obs_given_c, //// prev(0, c),
//                                  ////
//                                  prob_n_recip, 
//                                  phi_Z_recip,
//                                  phi_Bound_Z, 
//                                  phi_Upper_Bound_Z, //// for ordinal-only fn
//                                  ////
//                                  Bound_Z[c], //// for ordinal-only fn
//                                  Upper_Bound_Z[c], //// for ordinal-only fn
//                                  ////
//                                  prob_recip,
//                                  prob_rowwise_prod_temp, 
//                                  prob_recip_rowwise_prod_temp,
//                                  prob_rowwise_prod_temp_all,
//                                  common_grad_term_1,
//                                  ////
//                                  dphi_over_L, //// for ordinal-only fn     // y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
//                                  dZ_dmu_neg, //// for ordinal-only fn   // y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
//                                  ////
//                                  dphi_times_bz, //// for ordinal-only fn  ---- NEW
//                                  dZ_times_bz, //// for ordinal-only fn ---- NEW
//                                  ////
//                                  // n_binary_tests, //// for ordinal-only fn
//                                  ord_idx_of_test,
//                                  ////
//                                  Model_args_for_chunk);
//               
//         }
//         ////////////////////////////////////////////////////////////////////////////////////  Grad of nuisance parameters / u's (manual)
//         if ( (grad_option == "us_only") || (grad_option == "all") ) {
//           
//               u_grad_array_CM_chunk_block =  u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests);
//               
//               //// NOTE: this uses the Call BINARY gradient fn's with "dphi_over_L" and "dZ_dmu_neg"
//               ////    as the "y_sign_chunk_times..." and "y_m_ysign_x_u_array_times..." arguments:
//               fn_MVP_compute_nuisance_grad_v2(  u_grad_array_CM_chunk_block,
//                                                 phi_Z_recip,
//                                                 common_grad_term_1,
//                                                 L_Omega_double[c],
//                                                 prob[c],
//                                                 prob_recip,
//                                                 prob_rowwise_prod_temp,
//                                                 dphi_over_L, // y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
//                                                 dZ_dmu_neg,     // y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,  
//                                                 z_grad_term,
//                                                 grad_prob,
//                                                 prod_container_or_inc_array,
//                                                 derivs_chain_container_vec,
//                                                 Model_args_for_chunk);
//               
//               u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests).array() += u_grad_array_CM_chunk_block.array();
//               
//               const int start_index = 1 + (chunk_size_orig * n_tests * chunk_counter);
//               const int length = chunk_size * n_tests;
//               
//               if (c == n_class - 1) {
//                   
//                   //// update output vector once all u_grad computations are done
//                   out_mat.segment(start_index, length) = u_grad_array_CM_chunk.reshaped();
//                   
//                   //// account for unconstrained -> constrained transformations and Jacobian adjustments
//                   fn_MVP_nuisance_first_deriv(  du_wrt_duu_chunk,
//                                                 u_vec_chunk, u_unc_vec_chunk, Model_args_for_chunk);
//                   
//                   fn_MVP_nuisance_deriv_of_log_det_J(    d_J_wrt_duu_chunk,
//                                                          u_vec_chunk, u_unc_vec_chunk, du_wrt_duu_chunk, Model_args_for_chunk);
//                   
//                   out_mat.segment(start_index, length).array() *= du_wrt_duu_chunk.array();
//                   out_mat.segment(start_index, length).array() += d_J_wrt_duu_chunk.array();
//                 
//               }
//           
//         }
//         /////////////////////////////////////////////////////////////////////////// Grad of intercepts / coefficients (beta's)
//         if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only") ) {
//           
//               Eigen::Matrix<int, -1, 1> n_cov_vec_c = n_covariates_per_outcome_vec.row(c).transpose();
//               
//               fn_MVP_compute_coefficients_grad_v3(    c,
//                                                       beta_grad_array[c],
//                                                       X[c],
//                                                       n_cov_vec_c,
//                                                       chunk_size_orig * chunk_counter,
//                                                       // chunk_counter,
//                                                       n_covariates_max,
//                                                       common_grad_term_1,
//                                                       L_Omega_double[c],
//                                                       prob[c],
//                                                       prob_recip,
//                                                       prob_rowwise_prod_temp,
//                                                       ////
//                                                       dphi_over_L,
//                                                       dZ_dmu_neg,
//                                                       ////
//                                                       z_grad_term,
//                                                       grad_prob,
//                                                       prod_container_or_inc_array,
//                                                       derivs_chain_container_vec,
//                                                       true,  ///   compute_final_scalar_grad,
//                                                       Model_args_for_chunk);
//           
//         }
//         /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of L_Omega ('s)
//         if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only") ) {
//           
//               fn_MOVP_compute_L_Omega_grad_v3(   U_Omega_grad_array[c],  // direct to shared (serial)
//                                                  common_grad_term_1, 
//                                                  L_Omega_double[c],
//                                                  prob[c],
//                                                  prob_recip,
//                                                  // Bound_Z[c], 
//                                                  Z_std_norm[c],
//                                                  prob_rowwise_prod_temp,
//                                                  ////
//                                                  dphi_over_L,   //// ---- ordinal-only ---- y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
//                                                  dZ_dmu_neg,    //// ---- ordinal-only ---- y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
//                                                  dphi_times_bz, //// ---- ordinal-only + NEW
//                                                  dZ_times_bz,   //// ---- ordinal-only + NEW
//                                                  ////
//                                                  z_grad_term, 
//                                                  grad_prob, 
//                                                  prod_container_or_inc_array,
//                                                  derivs_chain_container_vec, 
//                                                  true,
//                                                  Model_args_for_chunk);
//           
//         }
//         /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of Cutpoints (MVOP/ordinal only!!)
//         if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "cutpoints_only") ) {
//           
//               
//               for (int t = 0; t < n_tests; ++t) {
//                     
//                     const int t_ord = ord_idx_of_test(t);
//                     if (t_ord < 0) continue;
//                     const int K_t = n_thr_per_ord_test(t_ord) + 1;  // n_cat
//             
//               // for (int t_ord = 0; t_ord < n_ordinal_tests; ++t_ord) {
//               //       
//               //       int t = n_binary_tests + t_ord;
//               //       int K_t = n_thr_per_ord_test(t_ord) + 1;  // n_cat
//                     
//                     fn_MVOP_compute_cutpoint_grad(     c, 
//                                                        cutpoint_grad_array[c],  // (n_cutpoints_max × n_ordinal_tests) 
//                                                        t, 
//                                                        t_ord, 
//                                                        K_t,
//                                                        common_grad_term_1,
//                                                        L_Omega_double[c],
//                                                        prob[c],
//                                                        prob_recip, 
//                                                        prob_rowwise_prod_temp,
//                                                        phi_Bound_Z,
//                                                        phi_Upper_Bound_Z,
//                                                        phi_Z_recip,
//                                                        u_array,
//                                                        y_chunk,
//                                                        ////
//                                                        dphi_over_L,
//                                                        dZ_dmu_neg,
//                                                        ////
//                                                        z_grad_term,
//                                                        grad_prob,
//                                                        prod_container_or_inc_array,
//                                                        derivs_chain_container_vec,
//                                                        dphi_direct_Cj, //// ---- ordinal-only
//                                                        dZ_direct_Cj, //// ---- ordinal-only
//                                                        true,
//                                                        Model_args_for_chunk);
//                 
//               }
//           
//         }
//         /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Prev grad (LC-MVP/LC-MVOP only)
//         if (n_class > 1) { /// prevelance only estimated for latent class models
//           
//             if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only" ) ) {
//               
//                 // rowwise_prod = prob[c].rowwise().prod();
//                 // rowwise_prod.array() = prob_n_recip.array() * rowwise_prod.array();
//                 // double prev_grad = rowwise_prod.sum();
//                 // prev_grad_vec(c)  +=  prev_grad;
//                 
//                 fn_MVP_prev_multi_pop_accumulate_grad( prob[c], 
//                                                        prob_n_recip,
//                                                        pop_ind,
//                                                        chunk_size_orig * chunk_counter, 
//                                                        chunk_size,
//                                                        n_pops, 
//                                                        c, 
//                                                        prev_grad_mat,
//                                                        rowwise_prod);
//               
//             }
//           
//         }
//         
//       } // end of c loop (gradient)
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
// //////////////////////////////////////////////////------------------------------------------------------------------------------------------------------------------------
// inline void fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial(  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat ,
//                                                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
//                                                                                 const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
//                                                                                 const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
//                                                                                 const std::string &grad_option,
//                                                                                 const Model_fn_args_struct &Model_args_as_cpp_struct,
//                                                                                 std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs
// ) {
//   
//         out_mat.setZero(); //// set log_prob and grad vec to zero at the start (only do this on outer fns, not inner/likelihood fns)
//         
//         //// important params
//         const int N = y_ref.rows();
//         const int n_tests = y_ref.cols();
//         const int n_us = theta_us_vec_ref.rows()  ;
//         const int n_params_main =  theta_main_vec_ref.rows()  ;
//         const int n_params = n_params_main + n_us;
//         
//         //////////////  access elements from struct and read
//         const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>  &X =  Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0];
//         
//         const bool exclude_priors = Model_args_as_cpp_struct.Model_args_bools(0);
//         const bool CI =             Model_args_as_cpp_struct.Model_args_bools(1);
//         const bool corr_force_positive = Model_args_as_cpp_struct.Model_args_bools(2);
//         const bool corr_prior_beta = Model_args_as_cpp_struct.Model_args_bools(3);
//         const bool corr_prior_norm = Model_args_as_cpp_struct.Model_args_bools(4);
//         const bool handle_numerical_issues = Model_args_as_cpp_struct.Model_args_bools(5);
//         const bool skip_checks_exp =   Model_args_as_cpp_struct.Model_args_bools(6);
//         const bool skip_checks_log =   Model_args_as_cpp_struct.Model_args_bools(7);
//         const bool skip_checks_lse =   Model_args_as_cpp_struct.Model_args_bools(8);
//         const bool skip_checks_tanh =  Model_args_as_cpp_struct.Model_args_bools(9);
//         const bool skip_checks_Phi =  Model_args_as_cpp_struct.Model_args_bools(10);
//         const bool skip_checks_log_Phi = Model_args_as_cpp_struct.Model_args_bools(11);
//         const bool skip_checks_inv_Phi = Model_args_as_cpp_struct.Model_args_bools(12);
//         const bool skip_checks_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_bools(13);
//         const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);
//         
//         const int n_cores = Model_args_as_cpp_struct.Model_args_ints(0);
//         const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
//         const int ub_threshold_phi_approx = Model_args_as_cpp_struct.Model_args_ints(2);
//         const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);
//         
//         // const double prev_prior_a = Model_args_as_cpp_struct.Model_args_doubles(0);
//         // const double prev_prior_b = Model_args_as_cpp_struct.Model_args_doubles(1);
//         const double overflow_threshold  = Model_args_as_cpp_struct.Model_args_doubles(0);
//         const double underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(1);
//         const double C_raw_lower = Model_args_as_cpp_struct.Model_args_doubles(2); //// ---- ordinal-only
//         const double C_raw_upper = Model_args_as_cpp_struct.Model_args_doubles(3); //// ---- ordinal-only
//         
//         std::string vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
//         const std::string &Phi_type = Model_args_as_cpp_struct.Model_args_strings(1);
//         const std::string &inv_Phi_type = Model_args_as_cpp_struct.Model_args_strings(2);
//         std::string vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);
//         std::string vect_type_log = Model_args_as_cpp_struct.Model_args_strings(4);
//         std::string vect_type_lse = Model_args_as_cpp_struct.Model_args_strings(5);
//         std::string vect_type_tanh = Model_args_as_cpp_struct.Model_args_strings(6);
//         std::string vect_type_Phi = Model_args_as_cpp_struct.Model_args_strings(7);
//         std::string vect_type_log_Phi = Model_args_as_cpp_struct.Model_args_strings(8);
//         std::string vect_type_inv_Phi = Model_args_as_cpp_struct.Model_args_strings(9);
//         std::string vect_type_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_strings(10);
//         const std::string J_grad_option =  Model_args_as_cpp_struct.Model_args_strings(11); // for "num_diff" or "analytical" or "autodiff" (for complex Jacobian)
//         const std::string nuisance_transformation =   Model_args_as_cpp_struct.Model_args_strings(12);
//         
//         ///// load vectors / matrices
//         const Eigen::Matrix<double, -1, 1> &lkj_cholesky_eta = Model_args_as_cpp_struct.Model_args_col_vecs_double[0];
//         const Eigen::Matrix<double, -1, 1> &prev_prior_a     = Model_args_as_cpp_struct.Model_args_col_vecs_double[1]; //// ---- for mult-pops
//         const Eigen::Matrix<double, -1, 1> &prev_prior_b     = Model_args_as_cpp_struct.Model_args_col_vecs_double[2]; //// ---- for mult-pops
//         
//         const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];
//         
//         // const Eigen::Matrix<double, -1, -1> &LT_b_priors_shape  = Model_args_as_cpp_struct.Model_args_mats_double[0];
//         // const Eigen::Matrix<double, -1, -1> &LT_b_priors_scale  = Model_args_as_cpp_struct.Model_args_mats_double[1];
//         // const Eigen::Matrix<double, -1, -1> &LT_known_bs_indicator = Model_args_as_cpp_struct.Model_args_mats_double[2];
//         // const Eigen::Matrix<double, -1, -1> &LT_known_bs_values = Model_args_as_cpp_struct.Model_args_mats_double[3];
//         
//         const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_coeffs_mean = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0];
//         const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_coeffs_sd   =  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1];
//         ////
//         const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_for_corr_a   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2];
//         const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_for_corr_b   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3];
//         std::vector<Eigen::Matrix<double, -1, -1>>   lb_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4];
//         const std::vector<Eigen::Matrix<double, -1, -1>>   &ub_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5];
//         const std::vector<Eigen::Matrix<double, -1, -1>>   &known_values    = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6];
//         
//         const std::vector<Eigen::Matrix<int, -1, -1 >> &known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];
//         
//         //// Override lb_corr if corr_force_positive is TRUE:
//         if (corr_force_positive == true) { 
//           for (int c = 0; c < n_class; ++c) {
//             for (int i = 1; i < n_tests; ++i) {
//               for (int j = 0; j < i; ++j) {
//                 lb_corr[c](i, j) = 0.0;
//               }
//             }
//           }
//         }
//         
//         //// ---- Ordinal-only params:
//         // const int n_binary_tests       = Model_args_as_cpp_struct.Model_args_ints(4); //// ---- ordinal-only
//         // const int n_ordinal_tests      = Model_args_as_cpp_struct.Model_args_ints(5); //// ---- ordinal-only
//         //// ---- NEW: per-test category counts in SLOT order (2 == binary). Single source of truth; types may interleave:
//         const Eigen::Matrix<int, -1, 1> &n_cat_per_test = Model_args_as_cpp_struct.Model_args_col_vecs_int[3];
//         ////
//         Eigen::Matrix<int, -1, 1> ord_idx_of_test(n_tests);   //// -1 for binary slots; else slot-order ordinal index
//         int n_binary_tests = 0;
//         int n_ordinal_tests = 0;
//         for (int t = 0; t < n_tests; ++t) {
//           if (n_cat_per_test(t) > 2) { ord_idx_of_test(t) = n_ordinal_tests; ++n_ordinal_tests; }
//           else                       { ord_idx_of_test(t) = -1;              ++n_binary_tests;  }
//         }
//         
//         const Eigen::Matrix<int, -1, 1> &n_cat_per_ord_test = Model_args_as_cpp_struct.Model_args_col_vecs_int[0]; //// ---- ordinal-only
//         const Eigen::Matrix<int, -1, 1> &n_thr_per_ord_test = Model_args_as_cpp_struct.Model_args_col_vecs_int[1]; //// ---- ordinal-only
//         
//         const int n_cutpoints_total = n_class * n_thr_per_ord_test.sum(); //// ---- ordinal-only
//         
//         //// Flattened Dirichlet concentration params (stacked: alpha for test 0, then test 1, etc):
//         // const Eigen::Matrix<double, -1, -1> &prior_dirichlet_alpha = Model_args_as_cpp_struct.Model_args_mats_double[4]; //// ---- ordinal-only
//         const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_dirichlet_alpha    = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[7]; //// ---- ordinal-only
//         
//         //// ---- Other prev params:
//         const int n_pops = Model_args_as_cpp_struct.Model_args_ints(6);  //// length = N ---- multi-pops
//         const Eigen::Matrix<int, -1, 1> &pop_ind = Model_args_as_cpp_struct.Model_args_col_vecs_int[2];  // 0-indexed, length N //// ---- multi-pops
//         
//         //////////////
//         const int n_corrs =  n_class * n_tests * (n_tests - 1) * 0.5;
//         
//         int n_covariates_total, n_covariates_max;
//         int n_covariates_total_nd, n_covariates_total_d;
//         int n_covariates_max_nd, n_covariates_max_d;
//         
//         if (n_class > 1)  {
//           
//               n_covariates_total_nd = n_covariates_per_outcome_vec.row(0).sum();
//               n_covariates_total_d = n_covariates_per_outcome_vec.row(1).sum();
//               n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
//               
//               n_covariates_max_nd = n_covariates_per_outcome_vec.row(0).maxCoeff();
//               n_covariates_max_d = n_covariates_per_outcome_vec.row(1).maxCoeff();
//               n_covariates_max = std::max(n_covariates_max_nd, n_covariates_max_d);
//           
//         } else {
//               
//               n_covariates_total = n_covariates_per_outcome_vec.sum();
//               n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();
//           
//         }
//         
//         const double sqrt_2_pi_recip = 1.0 / sqrt(2.0 * M_PI);
//         const double sqrt_2_recip = 1.0 / stan::math::sqrt(2.0);
//         const double minus_sqrt_2_recip = -sqrt_2_recip;
//         const double a = 0.07056;
//         const double b = 1.5976;
//         const double a_times_3 = 3.0 * 0.07056;
//         const double s = 1.0 / 1.702;
//         const double Inf = std::numeric_limits<double>::infinity();
//         
//         //// ---- determine chunk size --------------------------------------------------
//         const int desired_n_chunks = n_chunks;
//         
//         int vec_size;
//         if (vect_type == "AVX512")      vec_size = 8;
//         else if (vect_type == "AVX2")   vec_size = 4;
//         else if (vect_type == "AVX")    vec_size = 2;
//         else                            vec_size = 1;
//         
//         ChunkSizeInfo chunk_size_info = calculate_chunk_sizes( N, 
//                                                                vec_size,
//                                                                desired_n_chunks);
//         
//         // int chunk_size = chunk_size_info.chunk_size;
//         const int chunk_size_orig = chunk_size_info.chunk_size_orig;
//         const int normal_chunk_size = chunk_size_info.normal_chunk_size;
//         int last_chunk_size = chunk_size_info.last_chunk_size;
//         const int n_total_chunks = chunk_size_info.n_total_chunks;
//         int n_full_chunks = chunk_size_info.n_full_chunks;
//         
//         // // After computing chunk_size_orig:
//         // if (chunk_size_orig % vec_size != 0) {
//         //   // Can't use SIMD on this chunk — treat as remainder
//         //   n_full_chunks = n_total_chunks - 1;  // last chunk uses scalar fallback
//         //   last_chunk_size = chunk_size_orig;    // processed with "Stan" SIMD
//         // }
//         
//         //////////////  --------------------------------------------------------------------------------------------------------------------------------------
//         ////
//         //// ---- Corrs (doubles):
//         ////
//         const Eigen::Matrix<double, -1, 1>  Omega_raw_vec_double = theta_main_vec_ref.head(n_corrs); 
//         ////
//         //// ---- Coeffs (doubles):
//         ////
//         std::vector<Eigen::Matrix<double, -1, -1>> beta_double_array = vec_of_mats(n_covariates_max, n_tests, n_class);
//         
//         {
//           int i = n_corrs;
//           for (int c = 0; c < n_class; ++c) {
//             for (int t = 0; t < n_tests; ++t) {
//               for (int k = 0; k < n_covariates_per_outcome_vec(c, t); ++k) {
//                 beta_double_array[c](k, t) = theta_main_vec_ref(i);
//                 i += 1;
//               }
//             }
//           }
//         }
//         // //// ---- Identification constraint (label-switch guard): D+ must have the HIGHER intercept on
//         // ////      the (first) binary test. Outside this half-space the posterior is -Inf. Implemented as
//         // ////      a hard rejection rather than a reparameterisation, so the priors on the two intercepts
//         // ////      are exactly as specified, restricted to beta_d > beta_nd (cf. Stan: target += -Inf).
//         // ////      The gradient is irrelevant at a rejected point; NaN-free zeros are fine.
//         // ////
//         // if (n_class > 1 && n_binary_tests > 0) {
//         //       const double beta_bin_nd = beta_double_array[0](0, 0);
//         //       const double beta_bin_d  = beta_double_array[1](0, 0);
//         //       if (!(beta_bin_d > beta_bin_nd)) {
//         //             out_mat.setZero();
//         //             out_mat(0) = -std::numeric_limits<double>::infinity();
//         //             // stan::math::recover_memory_nested();
//         //             return;
//         //       }
//         // }
//         ////
//         //// ---- Prev (doubles):
//         ////
//         Eigen::Matrix<double, -1, 1> u_prev_raw(n_pops);
//         if (n_class > 1) {
//           for (int g = 0; g < n_pops; ++g) {
//             u_prev_raw(g) = theta_main_vec_ref(n_corrs + n_covariates_total + g);
//           }
//         }
//         // double u_prev_diseased = 0.00001;
//         // if (n_class > 1) u_prev_diseased = theta_main_vec_ref(n_corrs + n_covariates_total);
//         //////////////  --------------------------------------------------------------------------------------------------------------------------------------
//         ////
//         //// ---- Cutpoints (doubles):
//         ////
//         // ====================================================================
//         // ORDINAL ADDITION #1: unconstrained -> C_raw (BOUNDED) -> C, prior + Jacobians
//         //
//         // Two-stage transform now:
//         //   (a) theta -> C_raw   via  lb_ub:  C_raw = L + (U - L)*0.5*(1 + tanh(theta))
//         //       => C_raw[0]   (the FIRST cutpoint)   is bounded to (L, U)
//         //       => C_raw[k>0] (the LOG-GAPS)         bounded => gaps in (exp(L), exp(U))
//         //   (b) C_raw -> C       via  log-differences (UNCHANGED)
//         // ====================================================================
//         const int n_cutpoints_max = (n_ordinal_tests > 0) ? n_thr_per_ord_test.maxCoeff() : 0;
//         ////
//         const double C_raw_range = C_raw_upper - C_raw_lower;
//         const double log_half_range = std::log(0.5 * C_raw_range);
//         ////
//         std::vector<Eigen::Matrix<double, -1, -1>> C_raw       = vec_of_mats(n_cutpoints_max, n_ordinal_tests, n_class);
//         std::vector<Eigen::Matrix<double, -1, -1>> C           = vec_of_mats(n_cutpoints_max, n_ordinal_tests, n_class);
//         std::vector<Eigen::Matrix<double, -1, -1>> dC_raw_dunc = vec_of_mats(n_cutpoints_max, n_ordinal_tests, n_class);
//         ////
//         double log_det_J_unc_to_C_raw_double = 0.0;
//         {
//           int i = n_corrs + n_covariates_total + n_pops;
//           for (int c = 0; c < n_class; ++c) {
//             for (int t_ord = 0; t_ord < n_ordinal_tests; ++t_ord) {
//               for (int k = 0; k < n_thr_per_ord_test(t_ord); ++k) {
//                 
//                     const double C_unc_k      = theta_main_vec_ref(i);
//                     const double tanh_C_unc_k = std::tanh(C_unc_k);
//                     ////
//                     C_raw[c](k, t_ord)       = C_raw_lower + C_raw_range * 0.5 * (1.0 + tanh_C_unc_k);
//                     dC_raw_dunc[c](k, t_ord) = C_raw_range * 0.5 * (1.0 - tanh_C_unc_k * tanh_C_unc_k);
//                     ////
//                     //// log|dC_raw/dtheta| = log(0.5*range) + log(1 - tanh^2)
//                     //// log(1 - tanh^2(x)) = 2*(log2 - |x| - log1p(exp(-2|x|)))  <- stable for large |x|
//                     ////
//                     const double abs_C_unc_k = std::abs(C_unc_k);
//                     const double log_1m_tanh_sq = 2.0 * (0.6931471805599453 - abs_C_unc_k - std::log1p(std::exp(-2.0 * abs_C_unc_k)));
//                     log_det_J_unc_to_C_raw_double += log_half_range + log_1m_tanh_sq;
//                     ////
//                     i += 1;
//                 
//               }
//             } 
//           }
//         }
//         
//         double prior_density_induced_Dirichlet_double = 0.0;
//         double log_det_J_C_raw_to_C_double = 0.0;
//         
//         //// C_raw -> C transform + Jacobian + induced Dirichlet prior  (UNCHANGED from here down)
//         for (int c = 0; c < n_class; ++c) {
//           
//             for (int t_ord = 0; t_ord < n_ordinal_tests; ++t_ord) {
//               
//                   int n_thr_t = n_thr_per_ord_test(t_ord);
//                   int n_cat_t = n_thr_t + 1;
//                   
//                   //// ordered cutpoints: C[0] = C_raw[0], C[k] = C[k-1] + exp(C_raw[k])
//                   C[c](0, t_ord) = C_raw[c](0, t_ord);
//                   for (int k = 1; k < n_thr_t; ++k) {
//                     C[c](k, t_ord) = C[c](k - 1, t_ord) + stan::math::exp(C_raw[c](k, t_ord));
//                   } 
//                   
//                   //// Jacobian for C_raw -> C: log|det(J)| = sum_{k>=1} C_raw[k]
//                   for (int k = 1; k < n_thr_t; ++k) {
//                     log_det_J_C_raw_to_C_double += C_raw[c](k, t_ord);
//                   } 
//                   
//                   //// Induced Dirichlet: p_k = Phi(C_k) - Phi(C_{k-1}), then Dirichlet(alpha)
//                   int t = n_binary_tests + t_ord;
//                   double anchor = prior_coeffs_mean[c](0, t); // prior of intercepts as anchor
//                   ////
//                   Eigen::Matrix<double, -1, 1> cumul_probs(n_thr_t);
//                   for (int k = 0; k < n_thr_t; ++k) {
//                     cumul_probs(k) = stan::math::Phi(C[c](k, t_ord) - anchor);
//                   } 
//                   
//                   Eigen::Matrix<double, -1, 1> p_ord(n_cat_t);
//                   p_ord(0) = cumul_probs(0);
//                   for (int k = 1; k < n_thr_t; ++k) {
//                     p_ord(k) = cumul_probs(k) - cumul_probs(k - 1);
//                   }
//                   p_ord(n_cat_t - 1) = 1.0 - cumul_probs(n_thr_t - 1);
//                   
//                   Eigen::Matrix<double, -1, 1> alpha_t = prior_dirichlet_alpha[c].col(t_ord).head(n_cat_t);
//                   ////
//                   double alpha_sum = alpha_t.sum();
//                   prior_density_induced_Dirichlet_double += stan::math::lgamma(alpha_sum);
//                   ////
//                   for (int k = 0; k < n_cat_t; ++k) {
//                     prior_density_induced_Dirichlet_double -= stan::math::lgamma(alpha_t(k));
//                     double coeff = alpha_t(k) - 1.0;
//                     if (std::abs(coeff) > 1e-15) {
//                       prior_density_induced_Dirichlet_double += coeff * stan::math::log(p_ord(k));
//                     }
//                   }
//                   ////
//                   for (int k = 0; k < n_thr_t; ++k) {
//                     prior_density_induced_Dirichlet_double += stan::math::std_normal_lpdf(C[c](k, t_ord) - anchor);
//                   }
//               
//             }
//           
//         }
//         // ====================================================================
//         // ==== END ORDINAL ADDITION #1 ====
//         // ====================================================================
//         ////
//         //// ---- Omega:
//         ////
//         double prior_densities_L_Omega_double = 0.0;
//         double log_det_J_L_Omega_double = 0.0;
//         Eigen::Matrix<double, -1, 1>  grad_Omega_raw_priors_and_log_det_J(n_corrs);
//         ////
//         int dim_choose_2 = n_tests * (n_tests - 1) * 0.5 ;
//         std::vector<Eigen::Matrix<double, -1, -1>> deriv_L_wrt_unc_full = vec_of_mats_double(dim_choose_2 + n_tests, dim_choose_2, n_class);
//         std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_double = vec_of_mats_double(n_tests, n_tests, n_class);
//         std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_recip_double = vec_of_mats_double(n_tests, n_tests, n_class);
//         std::vector<Eigen::Matrix<double, -1, -1>> Omega_double = vec_of_mats_double(n_tests, n_tests, n_class);
//         std::vector<Eigen::Matrix<double, -1, -1>> Omega_unconstrained_double = fn_convert_std_vec_of_corrs_to_3d_array_double(Eigen_vec_to_std_vec(Omega_raw_vec_double),
//                                                                                                                                n_tests,
//                                                                                                                                n_class);
//         ////
//         for (int c = 0; c < n_class; ++c) {
//           
//               Eigen::Matrix<double, -1, -1>  Chol_Schur_outs_dbl =  Pinkney_corr_master_dbl(  n_tests,
//                                                                                               lb_corr[c],
//                                                                                               ub_corr[c],
//                                                                                               Omega_unconstrained_double[c],
//                                                                                               known_values_indicator[c],
//                                                                                               known_values[c]);
//               L_Omega_double[c] = Chol_Schur_outs_dbl.block(1, 0, n_tests, n_tests);
//               Omega_double[c]   = L_Omega_double[c] * L_Omega_double[c].transpose();
//               ////
//               L_Omega_recip_double[c] = Eigen::Matrix<double, -1, -1>::Zero(n_tests, n_tests);
//               for (int i = 1; i < n_tests; ++i) {
//                 for (int j = 0; j < i; ++j) {  // Only lower triangular
//                   if (std::abs(L_Omega_double[c](i, j)) > 1e-10) {  // Avoid near-zero
//                     L_Omega_recip_double[c](i, j) = 1.0 / L_Omega_double[c](i, j);
//                   } else {
//                     L_Omega_recip_double[c](i, j) = 0.0;  // or handle differently
//                   }
//                 }
//               }
//               
//         }
//         ////
//         //// ---- Compute L_Omega derivatives gradient using finite differences:
//         ////
//         Eigen::Matrix<double, -1, 1> grad_log_det_J(n_corrs);
//         grad_log_det_J.setZero();
//         ////
//         if (J_grad_option == "num_diff") {
//           
//                     int global_idx = 0;  // Cumulative counter for all classes
//                     
//                     for (int c = 0; c < n_class; ++c) { 
//                               
//                             //// Helper lambda: unconstrained -> un-permuted L
//                             auto get_L_orig = [&](const Eigen::Matrix<double, -1, -1> &Omega_unc) {
//                               
//                                   auto out = Pinkney_corr_master_dbl(   n_tests,
//                                                                         lb_corr[c],
//                                                                         ub_corr[c],
//                                                                         Omega_unc,
//                                                                         known_values_indicator[c],
//                                                                         known_values[c]);
//                                   return out.block(1, 0, n_tests, n_tests).eval();
//                               
//                             };
//                             
//                             // // Base
//                             // L_Omega_double[c] = get_L_orig( Omega_unconstrained_double[c]);
//                             
//                             // Finite differences
//                             int cnt_2 = 0;
//                             for (int i = 1; i < n_tests; i++) {
//                               
//                                   for (int j = 0; j < i; j++) {
//                                     
//                                         double epsilon = std::max(1e-8, 1e-6 * std::abs( Omega_unconstrained_double[c](i, j)));
//                                         if (std::abs( Omega_unconstrained_double[c](i, j)) > 2.0) epsilon = 1e-4;
//                                         
//                                         Omega_unconstrained_double[c](i, j) += epsilon;
//                                         Eigen::Matrix<double, -1, -1> L_perturbed = get_L_orig(Omega_unconstrained_double[c]);
//                                         Omega_unconstrained_double[c](i, j) -= epsilon;
//                                         
//                                         int cnt_1 = 0;
//                                         for (int k = 0; k < n_tests; k++) {
//                                           for (int l = 0; l <= k; l++) {
//                                             double deriv = (L_perturbed(k, l) - L_Omega_double[c](k, l)) / epsilon;
//                                             deriv_L_wrt_unc_full[c](cnt_1, cnt_2) = std::isfinite(deriv) ? deriv : 0.0;
//                                             cnt_1++;
//                                           }
//                                         }
//                                         
//                                         cnt_2++;
//                                         global_idx++;
//                                     
//                                   }
//                               
//                             }
//                         
//                     }
//           
//         } else if (J_grad_option == "analytical") { 
//           
//           // for (int c = 0; c < n_class; ++c) {                    
//           //   deriv_L_wrt_unc_full[c] = compute_L_Omega_derivatives_analytical( Omega_unconstrained_double[c],
//           //                                                                     L_Omega_double[c],
//           //                                                                     lb_corr[c], 
//           //                                                                     ub_corr[c],
//           //                                                                     known_values_indicator[c],
//           //                                                                     known_values[c],
//           //                                                                     n_tests);
//           // }
//           
//         }
//         ////
//         Eigen::Matrix<double, -1, 1> grad_C_raw_priors_and_log_det_J_double(n_cutpoints_total);
//         ////
//         Eigen::Matrix<double, -1, 1> grad_prev_raw(n_pops);
//         double prior_densities_prev_double = 0.0;
//         // double log_det_J_prev_double = 0.0; //// ---- multi-pops
//         // double grad_prev_raw_priors_and_log_det_J = 0.0; //// ---- multi-pops
//         double log_det_J_prev_from_AD = 0.0;
//         
//         {    ///////////   -------------------  start of AD block  ---------------------------------------------------------------------------------------------------------------
//           
//           stan::math::start_nested();  ////////////////////////
//           ////
//           stan::math::var target_AD = 0.0;
//           ////
//           //// ---- Cutpoints (var's):
//           ////
//           // ====================================================================
//           // ORDINAL ADDITION #1: Unpack C_raw, transform to C, prior + Jacobian
//           // ====================================================================
//           const int n_cutpoints_max = (n_ordinal_tests > 0) ? n_thr_per_ord_test.maxCoeff() : 0;
//           ////
//           std::vector<Eigen::Matrix<stan::math::var, -1, -1>> C_raw_var = vec_of_mats_var(n_cutpoints_max, n_ordinal_tests, n_class);
//           std::vector<Eigen::Matrix<stan::math::var, -1, -1>> C_var     = vec_of_mats_var(n_cutpoints_max, n_ordinal_tests, n_class);
//           ////
//           Eigen::Matrix<stan::math::var, -1, 1> C_unc_vec_var(n_cutpoints_total);   //// <- adj() taken wrt THIS now
//           {
//             int i = n_corrs + n_covariates_total + n_pops;
//             int j = 0;
//             for (int c = 0; c < n_class; ++c) {
//               for (int t_ord = 0; t_ord < n_ordinal_tests; ++t_ord) {
//                 for (int k = 0; k < n_thr_per_ord_test(t_ord); ++k) {
//                       C_unc_vec_var(j) = stan::math::to_var(theta_main_vec_ref(i));
//                       ////
//                       stan::math::var tanh_C_unc_vec_var = stan::math::tanh(C_unc_vec_var(j));
//                       C_raw_var[c](k, t_ord)  = C_raw_lower + C_raw_range * 0.5 * (1.0 + tanh_C_unc_vec_var);
//                       ////
//                       //// tanh Jacobian (gradient side; the VALUE is in log_det_J_unc_to_C_raw_double):
//                       target_AD += stan::math::log(0.5 * C_raw_range) + stan::math::log1m(stan::math::square(tanh_C_unc_vec_var));
//                       ////
//                       i += 1;
//                       j += 1;
//                 }
//               } 
//             }
//           }
//           ////
//           stan::math::var prior_density_induced_Dirichlet = 0.0;
//           stan::math::var log_det_J_C_raw_to_C = 0.0;
//           ////
//           {
//               //// C_raw -> C transform + Jacobian + induced Dirichlet prior
//               for (int c = 0; c < n_class; ++c) {
//                 
//                   for (int t_ord = 0; t_ord < n_ordinal_tests; ++t_ord) {
//                     
//                         int n_thr_t = n_thr_per_ord_test(t_ord);
//                         int n_cat_t = n_thr_t + 1;
//                         
//                         //// ordered cutpoints: C[0] = C_raw[0], C[k] = C[k-1] + exp(C_raw[k])
//                         C_var[c](0, t_ord) = C_raw_var[c](0, t_ord);
//                         for (int k = 1; k < n_thr_t; ++k) {
//                           C_var[c](k, t_ord) = C_var[c](k - 1, t_ord) + stan::math::exp(C_raw_var[c](k, t_ord));
//                         } 
//                         
//                         //// Jacobian for C_raw -> C: log|det(J)| = sum_{k>=1} C_raw[k]
//                         for (int k = 1; k < n_thr_t; ++k) {
//                           log_det_J_C_raw_to_C += C_raw_var[c](k, t_ord);
//                         } 
//                         
//                         //// Induced Dirichlet: p_k = Phi(C_k) - Phi(C_{k-1}), then Dirichlet(alpha)
//                         int t = n_binary_tests + t_ord;
//                         stan::math::var anchor = stan::math::to_var(prior_coeffs_mean[c](0, t)); // prior of intercepts as anchor
//                         ////
//                         Eigen::Matrix<stan::math::var, -1, 1> cumul_probs(n_thr_t);
//                         for (int k = 0; k < n_thr_t; ++k) {
//                           cumul_probs(k) = stan::math::Phi(C_var[c](k, t_ord) - anchor);
//                         } 
//                         
//                         Eigen::Matrix<stan::math::var, -1, 1> p_ord(n_cat_t);
//                         p_ord(0) = cumul_probs(0);
//                         for (int k = 1; k < n_thr_t; ++k) {
//                           p_ord(k) = cumul_probs(k) - cumul_probs(k - 1);
//                         }
//                         p_ord(n_cat_t - 1) = 1.0 - cumul_probs(n_thr_t - 1);
//                         
//                         //// Dirichlet log-density: sum (alpha_k - 1) * log(p_k) + lgamma(sum(alpha)) - sum(lgamma(alpha_k))
//                         Eigen::Matrix<double, -1, 1> alpha_t = prior_dirichlet_alpha[c].col(t_ord).head(n_cat_t);
//                         double alpha_sum = alpha_t.sum();
//                         prior_density_induced_Dirichlet += stan::math::lgamma(alpha_sum);
//                         ////
//                         if (prior_dirichlet_alpha[c].col(t_ord).head(n_cat_t).isOnes() == false) {
//                           for (int k = 0; k < n_cat_t; ++k) {
//                             prior_density_induced_Dirichlet -= stan::math::lgamma(alpha_t(k));
//                             double coeff = alpha_t(k) - 1.0;
//                             if (std::abs(coeff) > 1e-15) {
//                               prior_density_induced_Dirichlet += coeff * stan::math::log(p_ord(k));
//                             }
//                           }
//                         } 
//                         ////
//                         //// Jacobian for C -> p (induced Dirichlet): sum log(phi(C_k))  i.e. std_normal_lpdf
//                         ////
//                         for (int k = 0; k < n_thr_t; ++k) {
//                           prior_density_induced_Dirichlet += stan::math::std_normal_lpdf(C_var[c](k, t_ord) - anchor);
//                         }
//                         
//                   }
//                 
//               }
//               
//               target_AD += log_det_J_C_raw_to_C + prior_density_induced_Dirichlet; //// ----
//           }
//           // ====================================================================
//           // ==== END ORDINAL ADDITION #1 ====
//           // ====================================================================
//           // std::vector<Eigen::Matrix<stan::math::var, -1, -1>> C_raw_var = vec_of_mats_var(n_cutpoints_max, n_ordinal_tests, n_class);
//           Eigen::Matrix<stan::math::var, -1, 1> C_raw_vec_var(n_cutpoints_total);
//           {
//             int j = 0;
//             for (int c = 0; c < n_class; ++c) {
//               for (int t_ord = 0; t_ord < n_ordinal_tests; ++t_ord) {
//                 for (int k = 0; k < n_thr_per_ord_test(t_ord); ++k) {
//                   C_raw_vec_var(j) = C_raw_var[c](k, t_ord);
//                   j += 1;
//                 }
//               }
//             }
//           }
//           //  
//           //  const bool softplus = false;
//           //  std::vector<Eigen::Matrix<stan::math::var, -1, -1>> C_var = vec_of_mats_var(n_cutpoints_max, n_ordinal_tests, n_class);
//           //  for (int c = 0; c < n_class; ++c) {
//           //    for (int t = 0; t < n_ordinal_tests; ++t) {
//           //      
//           //      int n_thr_t = n_thr_per_ord_test(t);
//           //      Eigen::Matrix<stan::math::var, -1, 1> C_raw_t_var = C_raw_var[c].col(t).head(n_thr_t);
//           //      C_var[c].col(t).head(n_thr_t) = construct_C_var(C_raw_t_var, softplus);
//           //      
//           //    }
//           //  }
//           
//           // ////
//           // //// Cutpoints (priors + Jacobian) - var's: ----------------------------------------------------------------------------------------------------------------------
//           // ////
//           // stan::math::var log_det_J_C = 0.0;
//           // ////
//           // //// Jacobian for raw_C -> C transformation:
//           // ////
//           // for (int c = 0; c < n_class; ++c) {
//           //   for (int t = 0; t < n_ordinal_tests; ++t) {
//           //     
//           //     int n_thr_t = n_thr_per_ord_test(t);
//           //     log_det_J_C += raw_C_to_C_log_det_J_lp_var(C_raw_var[c].col(t).head(n_thr_t), softplus);
//           //     
//           //   }
//           // }
//           // target_AD += log_det_J_C;
//           
//           ////  std::cout << "MVOP: Random checkpoint #9" << std::endl; std::cout.flush();
//           
//           // ////
//           // //// Cutpoint priors (Induced-Dirichlet):
//           // ////
//           // // prior_dirichlet_alpha = ....;
//           // {
//           //   stan::math::var log_prior_density_C = 0.0;
//           //   for (int c = 0; c < n_class; ++c) {
//           //     for (int t = 0; t < n_ordinal_tests; ++t) {
//           //       
//           //       int n_thr_t = n_thr_per_ord_test(t);
//           //       int n_cat_t = n_thr_t + 1;
//           //       
//           //       Eigen::Matrix<stan::math::var, -1, 1> ID_cumul_probs_t_var = stan::math::Phi(C_var[c].col(t).head(n_thr_t));
//           //       Eigen::Matrix<stan::math::var, -1, 1> ID_ord_probs_t_var   = cumul_probs_to_ord_probs_var(ID_cumul_probs_t_var);
//           //       ////
//           //       Eigen::Matrix<double, -1, 1> prior_dirichlet_alpha_t = prior_dirichlet_alpha.col(t).head(n_cat_t);
//           //       ////
//           //       log_prior_density_C += induced_dirichlet_given_C_lpdf_var( ID_ord_probs_t_var,
//           //                                                                  C_var[c].col(t).head(n_thr_t), 
//           //                                                                  prior_dirichlet_alpha_t,
//           //                                                                  true);
//           //       
//           //     }
//           //     
//           //   }
//           //   target_AD += log_prior_density_C;
//           // }
//           // ////////////////////////////////////////////////////////////
//           // target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
//           // grad_C_raw_priors_and_log_det_J_double =  C_raw_vec_var.adj();    // differentiating WRT this
//           // out_mat.segment(1 + n_us + n_corrs + n_covariates_total, n_cutpoints_total) =  grad_C_raw_priors_and_log_det_J_double ;   //// add grad constribution to output
//           // stan::math::set_zero_all_adjoints_nested();
//           
//           // ////////////////////////////////////////////////////////////
//           // target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
//           // grad_C_raw_priors_and_log_det_J_double =  C_raw_vec_var.adj();    // differentiating WRT this
//           // int cutpoint_start = 1 + n_us + n_corrs + n_covariates_total + (n_class - 1);
//           // ////
//           // // 1. After AD writes to out_mat:
//           // if (grad_C_raw_priors_and_log_det_J_double.hasNaN()) {
//           //   std::cout << "NaN SOURCE: AD block, positions: ";
//           //   for (int i = 0; i < n_cutpoints_total; ++i)
//           //     if (std::isnan(grad_C_raw_priors_and_log_det_J_double(i))) std::cout << i << " ";
//           //     std::cout << std::endl; std::cout.flush();
//           // }
//           // out_mat.segment(cutpoint_start, n_cutpoints_total) = grad_C_raw_priors_and_log_det_J_double;
//           // stan::math::set_zero_all_adjoints_nested();
//           // // out_mat.segment(cutpoint_start, n_cutpoints_total) = grad_C_raw_priors_and_log_det_J_double;
//           // // stan::math::set_zero_all_adjoints_nested();
//           ////
//           ////////////////////////////////////////////////////////////
//           target_AD.grad();
//           grad_C_raw_priors_and_log_det_J_double = C_unc_vec_var.adj();
//           int cutpoint_start = 1 + n_us + n_corrs + n_covariates_total + n_pops;
//           ////
//           out_mat.segment(cutpoint_start, n_cutpoints_total) = grad_C_raw_priors_and_log_det_J_double;
//           stan::math::set_zero_all_adjoints_nested();
//           ////////////////////////////////////////////////////////////
//           ////
//           //// L_Omega /  correlation params - var's:  ----------------------------------------------------------------------------------------------------------------------
//           ////
//           Eigen::Matrix<stan::math::var, -1, 1  >  Omega_raw_vec_var =  stan::math::to_var(Omega_raw_vec_double) ;
//           std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_unconstrained_var = fn_convert_std_vec_of_corrs_to_3d_array_var(Eigen_vec_to_std_vec_var(Omega_raw_vec_var),
//                                                                                                                                        n_tests,
//                                                                                                                                        n_class);
//           std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  L_Omega_var = vec_of_mats_var(n_tests, n_tests, n_class);
//           std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_var   = vec_of_mats_var(n_tests, n_tests, n_class);
//           {
//               stan::math::var log_det_J_L_Omega = 0.0;
//               
//               for (int c = 0; c < n_class; ++c) {
//                     
//                     Eigen::Matrix<stan::math::var, -1, -1>  Chol_Schur_outs =  Pinkney_corr_master(  n_tests,
//                                                                                                      lb_corr[c],
//                                                                                                      ub_corr[c],
//                                                                                                      Omega_unconstrained_var[c],
//                                                                                                      known_values_indicator[c],
//                                                                                                      known_values[c]);
//                     L_Omega_var[c] = Chol_Schur_outs.block(1, 0, n_tests, n_tests);
//                     target_AD         += Chol_Schur_outs(0, 0);
//                     log_det_J_L_Omega += Chol_Schur_outs(0, 0); // now can set prior directly on Omega (as this is Jacobian adjustment)
//                     Omega_var[c] = L_Omega_var[c] * L_Omega_var[c].transpose();
//                     
//                     for (int i = 1; i < n_tests; ++i) {    ////   for (i in 2:n_tests) {
//                       for (int j = 0; j < i; ++j) {   ////   for (j in 1:(i - 1)) {
//                         if (known_values_indicator[c](i, j) == 1) {
//                           stan::math::var known_val_prior_ij = stan::math::normal_lpdf( Omega_var[c](i, j), 0.0, 10.0 ); //// to ensure any corr's we aren't estimating dont cause divergences
//                           target_AD += known_val_prior_ij;
//                           prior_densities_L_Omega_double += known_val_prior_ij.val();   //// value now reaches lp (was grad-only)
//                         }
//                       }
//                     }
//                 
//               }
//               
//               log_det_J_L_Omega_double += log_det_J_L_Omega.val();
//           }
//           
//           {
//               stan::math::var prior_densities_L_Omega = 0.0;
//               
//               for (int c = 0; c < n_class; ++c) {
//                 
//                     if ( (corr_prior_beta == false)   &&  (corr_prior_norm == false) ) {
//                       prior_densities_L_Omega +=  stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c], lkj_cholesky_eta(c)) ;
//                     } else if ( (corr_prior_beta == true)   &&  (corr_prior_norm == false) ) { 
//                       for (int i = 1; i < n_tests; i++) {
//                         for (int j = 0; j < i; j++) {
//                           prior_densities_L_Omega +=  stan::math::beta_lpdf(  (Omega_var[c](i, j) + 1)/2, prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
//                         }
//                       }
//                       //  Jacobian for  Omega -> L_Omega transformation for prior log-densities (since both LKJ and truncated normal prior densities are in terms of Omega, not L_Omega)
//                       Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
//                       for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i + 1) ) * stan::math::log(L_Omega_var[c](i, i));
//                       prior_densities_L_Omega  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
//                     } else if  ( (corr_prior_beta == false)   &&  (corr_prior_norm == true) ) {
//                       for (int i = 1; i < n_tests; i++) {
//                         for (int j = 0; j < i; j++) {
//                           prior_densities_L_Omega +=  stan::math::normal_lpdf(  Omega_var[c](i, j), prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
//                         }
//                       }
//                       Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
//                       for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i + 1) ) * stan::math::log(L_Omega_var[c](i, i));
//                       prior_densities_L_Omega  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
//                     }
//                 
//               }
//               
//               target_AD += prior_densities_L_Omega;
//               prior_densities_L_Omega_double += prior_densities_L_Omega.val();
//               
//               ////////////////////////////////////////////////////////////
//               target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
//               grad_Omega_raw_priors_and_log_det_J =  Omega_raw_vec_var.adj();    // differentiating WRT this
//               // if (J_grad_option != "autodiff") {
//               //   grad_Omega_raw_priors_and_log_det_J.array() += grad_log_det_J.array();
//               // } 
//               out_mat.segment(1 + n_us, n_corrs) =  grad_Omega_raw_priors_and_log_det_J ;   //// add grad constribution to output
//               stan::math::set_zero_all_adjoints_nested();
//               ////////////////////////////////////////////////////////////
//           }
//           
//           /////////////  prev stuff  ---- vars
//           {
//             if (n_class > 1) {  //// if latent class //// ----
//               
//               fn_MVP_prev_multi_pop_AD(  u_prev_raw, 
//                                          prev_prior_a, 
//                                          prev_prior_b,
//                                          n_pops,
//                                          prior_densities_prev_double, //// ---- computing this
//                                          log_det_J_prev_from_AD, //// ---- computing this
//                                          grad_prev_raw //// ---- computing this
//               ); 
//               
//               // Write to output (n_pops positions instead of 1)
//               int prev_start = 1 + n_us + n_corrs + n_covariates_total;
//               out_mat.segment(prev_start, n_pops) = grad_prev_raw;
//               
//             }
//             
//           }
//           ////////////////////////////////////////////////////////////
//           if (J_grad_option == "autodiff") {
//             for (int c = 0; c < n_class; ++c) {
//               int cnt_1 = 0;
//               for (int k = 0; k < n_tests; k++) {
//                 for (int l = 0; l < k + 1; l++) {
//                   (  L_Omega_var[c](k, l)).grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
//                   int cnt_2 = 0;
//                   for (int i = 1; i < n_tests; i++) {
//                     for (int j = 0; j < i; j++) {
//                       deriv_L_wrt_unc_full[c](cnt_1, cnt_2)  =   Omega_unconstrained_var[c](i, j).adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
//                       cnt_2 += 1;
//                     }
//                   }
//                   stan::math::set_zero_all_adjoints_nested();
//                   cnt_1 += 1;
//                 }
//               }
//             }
//           }
//           
//           ///////////////// get cholesky factor's (lower-triangular) of corr matrices
//           // convert to 3d var array
//           for (int c = 0; c < n_class; ++c) {
//             for (int t2 = 0; t2 < n_tests; ++t2) { //// col-major storage
//               for (int t1 = 0; t1 < n_tests; ++t1) {
//                 L_Omega_double[c](t1, t2) =   L_Omega_var[c](t1, t2).val()  ;
//                 L_Omega_recip_double[c](t1, t2) =   1.0 / L_Omega_double[c](t1, t2) ;
//               }
//             }
//           }
//           
//           stan::math::recover_memory_nested();  //////////////////////////////////////////
//           
//         }   //////////////////////////  end of local AD block  ------------------------------------------------------------------------------------------------------------------------------------------------------------
//         
//         /////////////  prev stuff (multi-pop)
//         Eigen::Matrix<double, -1, -1> prev_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
//         Eigen::Matrix<double, -1, -1> log_prev_mat_small = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class);
//         Eigen::Matrix<double, -1, 1>  tanh_u_prev_vec(n_pops);
//         Eigen::Matrix<double, -1, 1>  deriv_p_wrt_u_vec(n_pops);
//         double log_det_J_prev_double_total = 0.0;
//         ////
//         if (n_class > 1) {
//           
//           for (int g = 0; g < n_pops; ++g) {
//             tanh_u_prev_vec(g) = stan::math::tanh(u_prev_raw(g));
//             double prev_g = 0.5 * (tanh_u_prev_vec(g) + 1.0); 
//             prev_mat(g, 1) = prev_g;
//             prev_mat(g, 0) = 1.0 - prev_g;
//             deriv_p_wrt_u_vec(g) = 0.5 * (1.0 - tanh_u_prev_vec(g) * tanh_u_prev_vec(g));
//             log_det_J_prev_double_total +=  stan::math::log(deriv_p_wrt_u_vec(g));
//           }
//           ////
//           log_prev_mat_small = stan::math::log(prev_mat);
//           
//         }
//         ///////////////////////////////////////////////////////////////////////// prior densities
//         double prior_densities = 0.0;
//         
//         if (exclude_priors == false) {
//           
//           ///////////////////// priors for coeffs
//           double prior_densities_coeffs_double = 0.0;
//           for (int c = 0; c < n_class; c++) {
//             for (int t = 0; t < n_tests; t++) {
//               for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++) {
//                 prior_densities_coeffs_double += stan::math::normal_lpdf(beta_double_array[c](k, t), prior_coeffs_mean[c](k, t), prior_coeffs_sd[c](k, t));
//               }
//             }
//           }
//           
//           prior_densities += prior_densities_coeffs_double;
//           prior_densities += prior_densities_L_Omega_double;
//           prior_densities += prior_densities_prev_double;
//           prior_densities += prior_density_induced_Dirichlet_double;
//           
//         }
//         
//         ////////  ------- likelihood function  -----------------------------------------------------------------------------------------------------------------------------------
//         double log_prob_out = 0.0;
//         
//         // Jacobian adjustments (none needed for coeffs as unconstrained - so only for L_Omega -> Omega and u_prev -> prev, and the one for u's is computed in the likelihood)
//         const double log_det_J_main = log_det_J_prev_double_total + log_det_J_L_Omega_double + log_det_J_C_raw_to_C_double + log_det_J_unc_to_C_raw_double;
//         ////
//         //// Global (shared) gradient accumulators — will be written to by reduction
//         std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array = vec_of_mats<double>(n_covariates_max, n_tests, n_class);
//         std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array = vec_of_mats<double>(n_tests, n_tests, n_class);
//         Eigen::Matrix<double, -1, -1> prev_grad_mat = Eigen::Matrix<double, -1, -1>::Zero(n_pops, n_class); // --------
//         ////
//         double log_jac_u = 0.0;
//         ////
//         std::vector<Eigen::Matrix<double, -1, -1>> cutpoint_grad_array = vec_of_mats<double>(n_cutpoints_max, n_ordinal_tests, n_class); //// ordinal-only
//         ////
//         // ============================================================================
//         // CHUNK LOOP - one shared per-chunk routine (fn_lp_grad_MVOP_LC_Pinkney_NoLog_process_chunk).
//         // Full chunks receive the SIMD-configured struct; the remainder chunk receives a "Stan"
//         // (scalar) COPY of the struct, so it is genuinely scalar end-to-end (this replaces the
//         // previous duplicated full-chunks + last-chunk blocks AND bakes in the remainder-SIMD fix).
//         // ============================================================================
//         LC_MVP_workspace_struct &ws = LC_MVP_ws_structs[0];
//         ws.reset_sizes();
//         
//         for (int nc = 0; nc < n_full_chunks; nc++) {
//           
//               fn_lp_grad_MVOP_LC_Pinkney_NoLog_process_chunk(            out_mat,
//                                                                          theta_us_vec_ref,
//                                                                          y_ref,
//                                                                          grad_option,
//                                                                          nc,
//                                                                          normal_chunk_size,
//                                                                          chunk_size_orig,
//                                                                          N,
//                                                                          n_params,
//                                                                          n_tests,
//                                                                          n_class,
//                                                                          n_binary_tests,
//                                                                          n_ordinal_tests,
//                                                                          ord_idx_of_test, //// ----
//                                                                          n_covariates_max,
//                                                                          n_pops,
//                                                                          n_covariates_per_outcome_vec,
//                                                                          n_cat_per_ord_test,
//                                                                          n_thr_per_ord_test,
//                                                                          pop_ind,
//                                                                          X,
//                                                                          beta_double_array,
//                                                                          C,
//                                                                          L_Omega_double,
//                                                                          L_Omega_recip_double,
//                                                                          prev_mat,
//                                                                          log_prev_mat_small,
//                                                                          log_jac_u,
//                                                                          beta_grad_array,
//                                                                          U_Omega_grad_array,
//                                                                          cutpoint_grad_array,
//                                                                          prev_grad_mat,
//                                                                          ws,
//                                                                          Model_args_as_cpp_struct);
//           
//         }
//         
//         // ---- LAST CHUNK (remainder) - scalar ("Stan") fallback end-to-end: ----
//         if ((n_full_chunks < n_total_chunks) && (last_chunk_size > 0)) {
//           
//               Model_fn_args_struct Model_args_last_chunk = Model_args_as_cpp_struct;
//               Model_args_last_chunk.Model_args_strings(0)  = "Stan";  // vect_type
//               for (int s_i = 3; s_i <= 10; ++s_i)  Model_args_last_chunk.Model_args_strings(s_i) = "Stan";
//               
//               fn_lp_grad_MVOP_LC_Pinkney_NoLog_process_chunk(            out_mat,
//                                                                          theta_us_vec_ref,
//                                                                          y_ref,
//                                                                          grad_option,
//                                                                          n_full_chunks,
//                                                                          last_chunk_size,
//                                                                          chunk_size_orig,
//                                                                          N,
//                                                                          n_params,
//                                                                          n_tests,
//                                                                          n_class,
//                                                                          n_binary_tests,
//                                                                          n_ordinal_tests,
//                                                                          ord_idx_of_test, //// ----
//                                                                          n_covariates_max,
//                                                                          n_pops,
//                                                                          n_covariates_per_outcome_vec,
//                                                                          n_cat_per_ord_test,
//                                                                          n_thr_per_ord_test,
//                                                                          pop_ind,
//                                                                          X,
//                                                                          beta_double_array,
//                                                                          C,
//                                                                          L_Omega_double,
//                                                                          L_Omega_recip_double,
//                                                                          prev_mat,
//                                                                          log_prev_mat_small,
//                                                                          log_jac_u,
//                                                                          beta_grad_array,
//                                                                          U_Omega_grad_array,
//                                                                          cutpoint_grad_array,
//                                                                          prev_grad_mat,
//                                                                          ws,
//                                                                          Model_args_last_chunk);
//           
//         }
//         ////
//         //// After the chunk loop, chain-rule cutpoint grads through C_raw -> C:
//         ////
//         int out_idx = 1 + n_us + n_corrs + n_covariates_total + n_pops;
//         ////
//         for (int c = 0; c < n_class; ++c) {
//           
//             for (int t_ord = 0; t_ord < n_ordinal_tests; ++t_ord) {
//                 
//                 int n_thr_t = n_thr_per_ord_test(t_ord);
//                 Eigen::Matrix<double, -1, 1> grad_wrt_C = cutpoint_grad_array[c].col(t_ord).head(n_thr_t);
//                 Eigen::Matrix<double, -1, 1> grad_wrt_C_raw = chain_rule_C_to_C_raw(grad_wrt_C, C_raw[c].col(t_ord).head(n_thr_t));
//                 ////
//                 //// ---- NEW: chain through the bounding transform  d(C_raw)/d(theta):
//                 grad_wrt_C_raw.array() *= dC_raw_dunc[c].col(t_ord).head(n_thr_t).array();
//                 ////
//                 out_mat.segment(out_idx, n_thr_t) += grad_wrt_C_raw;
//                 out_idx += n_thr_t;
//               
//             }
//           
//         }
//         
//         // ============================================================================
//         // Post-loop: prevalence gradient, log_prob, output assembly
//         // ============================================================================
//         // Eigen::Matrix<double, -1, 1> prev_unconstrained_grad_vec = Eigen::Matrix<double, -1, 1>::Zero(n_class);
//         Eigen::Matrix<double, -1, 1> prev_unconstrained_grad_vec_out = Eigen::Matrix<double, -1, 1>::Zero(n_pops); //// ----
//         
//         ////////////////////////  --------------------------------------------------------------------------
//         // if (n_class > 1) {
//         //   for (int c = 0; c < n_class; c++) {
//         //     prev_unconstrained_grad_vec(c)  =   prev_grad_vec(c) * deriv_p_wrt_pu_double ;
//         //   }
//         //   prev_unconstrained_grad_vec(0) = prev_unconstrained_grad_vec(1) - prev_unconstrained_grad_vec(0) - 2 * tanh_u_prev[1];
//         //   prev_unconstrained_grad_vec_out(0) = prev_unconstrained_grad_vec(0);
//         // }
//         if (n_class > 1) {
//           for (int g = 0; g < n_pops; ++g) { //// ----
//             double lik_grad = (prev_grad_mat(g, 1) - prev_grad_mat(g, 0)) * deriv_p_wrt_u_vec(g);
//             double jac_grad = -2.0 * tanh_u_prev_vec(g);
//             prev_unconstrained_grad_vec_out(g) = lik_grad + jac_grad;
//           }
//         }
//         
//         ////////////////////////  --------------------------------------------------------------------------
//         log_prob_out +=  out_mat.tail(N).sum();  ////  log_lik
//         log_prob_out +=  log_jac_u;
//         if (exclude_priors == false)  log_prob_out += prior_densities;
//         log_prob_out +=  log_det_J_main ; // log_jac_p_double;
//         
//         Eigen::Matrix<double, -1, 1> beta_grad_vec = Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);
//         {
//           int i = 0;
//           for (int c = 0; c < n_class; c++ ) {
//             for (int t = 0; t < n_tests; t++) {
//               for (int k = 0; k <  n_covariates_per_outcome_vec(c, t); k++) {
//                 beta_grad_vec(i) = beta_grad_array[c](k, t);
//                 i += 1;
//               }
//             }
//           }
//         }
//         
//         // Pack L_Omega gradients and chain-rule through deriv_L_wrt_unc_full
//         Eigen::Matrix<double, -1, 1> L_Omega_grad_vec(n_corrs + (n_class * n_tests));
//         Eigen::Matrix<double, -1, 1> U_Omega_grad_vec(n_corrs);
//         
//         {
//           int i = 0;
//           for (int c = 0; c < n_class; c++) {
//             for (int t1 = 0; t1 < n_tests; t1++) {
//               for (int t2 = 0; t2 <  t1 + 1; t2++) {
//                 L_Omega_grad_vec(i) = U_Omega_grad_array[c](t1, t2);
//                 i += 1;
//               }
//             }
//           }
//         }
//         
//         Eigen::Matrix<double, -1, 1>  grad_wrt_L_Omega_nd(dim_choose_2 + n_tests);
//         Eigen::Matrix<double, -1, 1>  grad_wrt_L_Omega_d(dim_choose_2 + n_tests);
//         
//         if (n_class > 1) {
//           grad_wrt_L_Omega_nd =  L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
//           grad_wrt_L_Omega_d =   L_Omega_grad_vec.segment(dim_choose_2 + n_tests, dim_choose_2 + n_tests);
//           ////
//           U_Omega_grad_vec.head(dim_choose_2) =  ( grad_wrt_L_Omega_nd.transpose() * deriv_L_wrt_unc_full[0]  ).transpose();
//           U_Omega_grad_vec.segment(dim_choose_2, dim_choose_2) =   ( grad_wrt_L_Omega_d.transpose()  *  deriv_L_wrt_unc_full[1] ).transpose();
//           ////
//         } else {
//           grad_wrt_L_Omega_nd =   L_Omega_grad_vec.head(dim_choose_2 + n_tests);
//           U_Omega_grad_vec.head(dim_choose_2) =  ( grad_wrt_L_Omega_nd.transpose() * deriv_L_wrt_unc_full[0] ).transpose();
//         }
//         
//         // ---- Final output assembly ----
//         ////////////////////////////  outputs // add log grad and sign stuff';///////////////
//         out_mat(0) =  log_prob_out;
//         out_mat.segment(1 + n_us, n_corrs) += U_Omega_grad_vec ;
//         out_mat.segment(1 + n_us + n_corrs, n_covariates_total) += beta_grad_vec ;  /// no Jacobian needed
//         ////
//         if (n_class > 1) {
//           // out_mat(1 + n_us + n_corrs + n_covariates_total) += prev_unconstrained_grad_vec_out(0);
//           const int prev_start_final = 1 + n_us + n_corrs + n_covariates_total;
//           out_mat.segment(prev_start_final, n_pops) += prev_unconstrained_grad_vec_out;
//         }
//         
//         // add derivative of normal prior density to beta / coeffs gradient
//         // ---- Prior gradient contribution to coefficients ----
//         {
//           int i = n_us + n_corrs + 1; /// + 1 because first element is the log_prob !!
//           for (int c = 0; c < n_class; c++) {
//             for (int t = 0; t < n_tests; t++) {
//               for (int k = 0; k <  n_covariates_per_outcome_vec(c, t); k++) {
//                 if (exclude_priors == false) {
//                   out_mat(i) += - ((beta_double_array[c](k, t) - prior_coeffs_mean[c](k, t)) / prior_coeffs_sd[c](k, t) ) * (1.0 / prior_coeffs_sd[c](k, t) ) ;
//                   i += 1;
//                 }
//               }
//             }
//           }
//         }
//         
// }
// 
// 



