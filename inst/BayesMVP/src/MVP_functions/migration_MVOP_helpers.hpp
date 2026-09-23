

//// migration_MVOP_helpers_and_MVP_driver.hpp
////
//// How to apply fn_dispatch_templated.hpp to the code you pasted.
////
//// RULE: only functions that CALL fn_EIGEN_double change. In what you pasted that is:
////   - fn_MVOP_compute_lp_GHK_cols          (rewritten below, complete)
////   - fn_MVOP_compute_phi_Bound_Z_cols      (rewritten below, complete)
////   - fn_MVP_compute_lp_GHK_cols            (the MVP one, rewritten below, complete)
////   - the driver fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl
////     (calls it once for prob_n; exact edits in PART D)
////
//// These do NOT call fn_EIGEN_double and need NO change at all:
////   fn_MVOP_grad_prep, fn_MVOP_compute_cutpoint_grad, fn_MOVP_compute_L_Omega_grad_v3,
////   chain_rule_C_to_C_raw, and all the fn_MVP_compute_*_grad_* functions.
////
//// Helpers you did not paste but which the driver calls and which DO call fn_EIGEN_double
//// (same recipe as PART A/B, one at a time):
////   fn_MVP_compute_phi_Z_recip_cols, fn_MVP_compute_phi_Bound_Z_cols, fn_MVP_grad_prep (if it does),
////   fn_MVP_compute_nuisance, fn_MVP_compute_nuisance_log_jac_u, fn_MVP_nuisance_first_deriv,
////   fn_MVP_nuisance_deriv_of_log_det_J, log_sum_exp_general.
////
//// Until ALL of them are migrated, the driver still passes Model_args_as_cpp_struct to the
//// un-migrated ones and they keep working through the old string path. Migrate incrementally,
//// build and run the self-test after each.

#pragma once
// #include "MVP_helpers_migrated.hpp"

//// =====================================================================================
//// PART 0: (REMOVED — KernelChoice, kernel_choice_from_args, apply_col_inplace, Phi_col_inplace,
////          inv_Phi_col_inplace all live in MVP_helpers_migrated.hpp; include that instead.)

//// =====================================================================================
//// PART A: fn_MVOP_compute_lp_GHK_cols — rewritten. Same outputs, same maths, no strings,
////         no temporaries. Note the pattern: copy the input column INTO the output column,
////         then transform the output column in place.
//// =====================================================================================
template <Vec vec>
inline void fn_MVOP_compute_lp_GHK_cols_T(   const int t,
                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> y_chunk,
                                           const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array,
                                           Eigen::Ref<Eigen::Matrix<double, -1, -1>> Upper_Bound_Z,
                                           const bool is_binary,
                                           const KernelChoice &k
) {

  ////
  //// ---- 2026-09-22 (assistant, approved change): for exact Phi (Phi_type = "Phi") the right-side seam is removed by reflection.
  ////
  //// Binary: identical to fn_MVP_compute_lp_GHK_cols_T (see the derivation there): prob = Phi(sign_y * B), sign_y = 1 - 2y,
  ////         q = (y - (2y-1) u) prob, Z = sign_y * Phi^{-1}(q).
  //// Ordinal: when BOTH bounds are positive (lower bound lb > 0), Phi(ub) - Phi(lb) is a difference of two numbers near 1
  ////         (catastrophic cancellation near lb = +7.5, just inside overflow_threshold). Using Phi(x) = 1 - Phi(-x):
  ////           reflect  = (lb > 0);  a = reflect ? -ub : lb;  b = reflect ? -lb : ub     (a < b; a = -Inf when ub = +Inf)
  ////           prob     = Phi(b) - Phi(a)                       [ = Phi(-lb) - Phi(-ub) = Phi(ub) - Phi(lb) when reflecting ]
  ////           q        = Phi(a) + w prob,  w = reflect ? (1 - u) : u
  ////                      [ reflecting: q = u Phi(-ub) + (1 - u) Phi(-lb) = 1 - (Phi(lb) + u (Phi(ub) - Phi(lb))) = 1 - Phi_Z ]
  ////           Z        = reflect ? -Phi^{-1}(q) : Phi^{-1}(q),  Phi_Z = reflect ? 1 - q : q
  ////           Phi(lb)  = reflect ? 1 - Phi(b) : Phi(a)       (stored for completeness; only the Phi_approx derivative reads it)
  ////         Non-reflected rows use exactly the old operations. The gradient code is unchanged: it uses the analytic
  ////         derivatives phi(lb), phi(ub) (from the bounds) and 1/phi(Z) (from Z), which do not depend on how prob was formed.
  ////
  if (!k.Phi_approx) {
    
        if (is_binary) {
          
              prob.col(t).array() = (1.0 - 2.0 * y_chunk.col(t).array()) * Bound_Z.col(t).array();
              Phi_col_inplace<vec>(prob, t, k);
              Bound_U_Phi_Bound_Z.col(t).array() = y_chunk.col(t).array()
                                                 + (1.0 - 2.0 * y_chunk.col(t).array()) * prob.col(t).array();
              Z_std_norm.col(t).array() = (y_chunk.col(t).array() - (2.0 * y_chunk.col(t).array() - 1.0) * u_array.col(t).array())
                                        * prob.col(t).array();                                                       //// q
              Phi_Z.col(t).array() = y_chunk.col(t).array()
                                   + (1.0 - 2.0 * y_chunk.col(t).array()) * Z_std_norm.col(t).array();
              inv_Phi_col_inplace<vec>(Z_std_norm, t, k);
              Z_std_norm.col(t).array() *= (1.0 - 2.0 * y_chunk.col(t).array());
              y1_log_prob.col(t) = prob.col(t);
              apply_col_inplace<vec, Fn::log>(y1_log_prob, t);
          
        } else {
          
              const Eigen::Array<bool, -1, 1> reflect_mask_col = (Bound_Z.col(t).array() > 0.0);                   //// lb > 0 (then ub > lb > 0)
              //// Phi(a) -> Bound_U_Phi_Bound_Z (temporarily), Phi(b) -> prob:
              Bound_U_Phi_Bound_Z.col(t).array() = reflect_mask_col.select(-Upper_Bound_Z.col(t).array(), Bound_Z.col(t).array());
              Phi_col_inplace<vec>(Bound_U_Phi_Bound_Z, t, k);
              prob.col(t).array() = reflect_mask_col.select(-Bound_Z.col(t).array(), Upper_Bound_Z.col(t).array());
              Phi_col_inplace<vec>(prob, t, k);
              prob.col(t).array() -= Bound_U_Phi_Bound_Z.col(t).array();                                               //// prob = Phi(b) - Phi(a)
              //// q = Phi(a) + w prob:
              Z_std_norm.col(t).array() = Bound_U_Phi_Bound_Z.col(t).array()
                                        + reflect_mask_col.select(1.0 - u_array.col(t).array(), u_array.col(t).array()) * prob.col(t).array();
              Phi_Z.col(t).array() = reflect_mask_col.select(1.0 - Z_std_norm.col(t).array(), Z_std_norm.col(t).array());
              //// Phi(lb) = reflect ? 1 - Phi(b) = 1 - (Phi(a) + prob) : Phi(a):
              Bound_U_Phi_Bound_Z.col(t).array() = reflect_mask_col.select(1.0 - (Bound_U_Phi_Bound_Z.col(t).array() + prob.col(t).array()),
                                                                           Bound_U_Phi_Bound_Z.col(t).array());
              inv_Phi_col_inplace<vec>(Z_std_norm, t, k);
              Z_std_norm.col(t).array() = reflect_mask_col.select(-Z_std_norm.col(t).array(), Z_std_norm.col(t).array());
              y1_log_prob.col(t) = prob.col(t);
              apply_col_inplace<vec, Fn::log>(y1_log_prob, t);
          
        }
        
        return;
    
  }

  //// ---- Phi_approx setting: original code below, unchanged ----
  if (is_binary) {

        //// Phi(BZ):
        Bound_U_Phi_Bound_Z.col(t) = Bound_Z.col(t);
        Phi_col_inplace<vec>(Bound_U_Phi_Bound_Z, t, k);

        //// Phi_Z = y*Phi(BZ) + (y - Phi(BZ)) * (2y-1) * u      [(2y-1) written inline, no y_sign temp]
        Phi_Z.col(t).array() = y_chunk.col(t).array() * Bound_U_Phi_Bound_Z.col(t).array()
                             + (y_chunk.col(t).array() - Bound_U_Phi_Bound_Z.col(t).array())
                             * ((2.0 * y_chunk.col(t).array() - 1.0) * u_array.col(t).array());

        //// Z = inv_Phi(Phi_Z):
        Z_std_norm.col(t) = Phi_Z.col(t);
        inv_Phi_col_inplace<vec>(Z_std_norm, t, k);

        //// prob = y*(1 - Phi(BZ)) + (y-1)*Phi(BZ)*(2y-1):
        prob.col(t).array() = y_chunk.col(t).array() * (1.0 - Bound_U_Phi_Bound_Z.col(t).array())
                            + (y_chunk.col(t).array() - 1.0) * Bound_U_Phi_Bound_Z.col(t).array()
                            * (2.0 * y_chunk.col(t).array() - 1.0);

        //// log(prob):
        y1_log_prob.col(t) = prob.col(t);
        apply_col_inplace<vec, Fn::log>(y1_log_prob, t);

  } else {

        //// Phi(lower_bz) -> Bound_U_Phi_Bound_Z:
        Bound_U_Phi_Bound_Z.col(t) = Bound_Z.col(t);
        Phi_col_inplace<vec>(Bound_U_Phi_Bound_Z, t, k);

        //// prob = Phi(upper_bz) - Phi(lower_bz). Use prob.col(t) itself as the scratch for Phi(ub):
        prob.col(t) = Upper_Bound_Z.col(t);
        Phi_col_inplace<vec>(prob, t, k);
        prob.col(t).array() -= Bound_U_Phi_Bound_Z.col(t).array();

        //// Phi(Z) = Phi(lb) + u * prob:
        Phi_Z.col(t).array() = Bound_U_Phi_Bound_Z.col(t).array() + u_array.col(t).array() * prob.col(t).array();

        //// Z = inv_Phi(Phi(Z)):
        Z_std_norm.col(t) = Phi_Z.col(t);
        inv_Phi_col_inplace<vec>(Z_std_norm, t, k);

        //// log(prob):
        y1_log_prob.col(t) = prob.col(t);
        apply_col_inplace<vec, Fn::log>(y1_log_prob, t);

  }

}

//// =====================================================================================
//// PART B: fn_MVOP_compute_phi_Bound_Z_cols — rewritten.
//// =====================================================================================
template <Vec vec>
inline void fn_MVOP_compute_phi_Bound_Z_cols_T(  const int t,
                                               Eigen::Matrix<double, -1, -1> &phi_Bound_Z,
                                               Eigen::Matrix<double, -1, -1> &phi_Upper_Bound_Z_out,
                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Phi_Upper_Bound_Z,
                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Bound_Z,
                                               const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Upper_Bound_Z,
                                               const bool is_binary,
                                               const KernelChoice &k
) {

  const double sqrt_2_pi_recip = 1.0 / std::sqrt(2.0 * M_PI);
  const double a_times_3 = 3.0 * 0.07056;
  const double b = 1.5976;

  //// phi(lower_bz) or phi(BZ):
  if (k.Phi_approx) {
        //// d/dx Phi_approx(x) = Phi_approx(x) (1 - Phi_approx(x)) (3 a x^2 + b)
        phi_Bound_Z.col(t).array() = (a_times_3 * Bound_Z.col(t).array().square() + b)
                                   * Bound_U_Phi_Bound_Z.col(t).array()
                                   * (1.0 - Bound_U_Phi_Bound_Z.col(t).array());
  } else {
        //// phi(x) = exp(-x^2/2) / sqrt(2 pi):  write -x^2/2 into the output column, exp it in place, scale.
        phi_Bound_Z.col(t).array() = -0.5 * Bound_Z.col(t).array().square();
        apply_col_inplace<vec, Fn::exp>(phi_Bound_Z, t);
        phi_Bound_Z.col(t).array() *= sqrt_2_pi_recip;
  }

  //// Ordinal: also phi(upper_bz):
  if (!is_binary) {
        if (k.Phi_approx) {
              phi_Upper_Bound_Z_out.col(t).array() = (a_times_3 * Upper_Bound_Z.col(t).array().square() + b)
                                                   * Phi_Upper_Bound_Z.col(t).array()
                                                   * (1.0 - Phi_Upper_Bound_Z.col(t).array());
        } else {
              phi_Upper_Bound_Z_out.col(t).array() = -0.5 * Upper_Bound_Z.col(t).array().square();
              apply_col_inplace<vec, Fn::exp>(phi_Upper_Bound_Z_out, t);
              phi_Upper_Bound_Z_out.col(t).array() *= sqrt_2_pi_recip;
        }
  }

}




// //// =====================================================================================
// //// PART C: fn_MVP_compute_lp_GHK_cols (the binary-only MVP version the driver calls).
// //// =====================================================================================
// template <Vec vec>
// inline void fn_MVP_compute_lp_GHK_cols_T(   const int t,
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
//   Bound_U_Phi_Bound_Z.col(t) = Bound_Z.col(t);
//   Phi_col_inplace<vec>(Bound_U_Phi_Bound_Z, t, k);
// 
//   Phi_Z.col(t).array() = y_chunk.col(t).array() * Bound_U_Phi_Bound_Z.col(t).array()
//                        + (y_chunk.col(t).array() - Bound_U_Phi_Bound_Z.col(t).array())
//                        * ((2.0 * y_chunk.col(t).array() - 1.0) * u_array.col(t).array());
// 
//   Z_std_norm.col(t) = Phi_Z.col(t);
//   inv_Phi_col_inplace<vec>(Z_std_norm, t, k);
// 
//   prob.col(t).array() = y_chunk.col(t).array() * (1.0 - Bound_U_Phi_Bound_Z.col(t).array())
//                       + (y_chunk.col(t).array() - 1.0) * Bound_U_Phi_Bound_Z.col(t).array()
//                       * (2.0 * y_chunk.col(t).array() - 1.0);
// 
//   y1_log_prob.col(t) = prob.col(t);
//   apply_col_inplace<vec, Fn::log>(y1_log_prob, t);
// }

//// =====================================================================================
//// PART D: the driver. Exact edits to
////   fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl
//// (numbered so you can do them in order and build after each).
//// =====================================================================================
////
//// D1. Make it a template and add a non-template wrapper that does the ONE runtime dispatch.
////     Rename your existing function to ..._serial_impl_T and put `template <Vec vec>` in
////     front of it:
////
////       template <Vec vec>
////       inline void fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl_T( ...same args... )
////
////     then add, AFTER it:
////
////       inline void fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl(
////               Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
////               const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
////               const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
////               const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
////               const std::string &grad_option,
////               const Model_fn_args_struct &Model_args_as_cpp_struct,
////               std::vector<LC_MVP_workspace_struct> &LC_MVP_ws_structs) {
////         const Vec vec = vec_from_string(Model_args_as_cpp_struct.Model_args_strings(0));
////         if (vec == Vec::AVX512)
////           fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl_T<Vec::AVX512>(out_mat, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct, LC_MVP_ws_structs);
////         else
////           fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process_serial_impl_T<Vec::Scalar>(out_mat, theta_main_vec_ref, theta_us_vec_ref, y_ref, grad_option, Model_args_as_cpp_struct, LC_MVP_ws_structs);
////       }
////
////     Every caller of ..._serial_impl keeps compiling unchanged.
////
//// D2. At the top of the template, right after `const int n_chunks = ...;`, add:
////
////       const KernelChoice kchoice = kernel_choice_from_args(Model_args_as_cpp_struct);
////
//// D3. DELETE these declarations (they are the string dispatch and nothing needs them now):
////
////       std::string vect_type_exp = ...;                      (strings 3..10, all nine of them)
////       std::string vect_type_log = ...;
////       std::string vect_type_lse = ...;
////       std::string vect_type_tanh = ...;
////       std::string vect_type_Phi = ...;
////       std::string vect_type_log_Phi = ...;
////       std::string vect_type_inv_Phi = ...;
////       std::string vect_type_inv_Phi_approx_from_logit_prob = ...;
////
////     KEEP `vect_type` (strings(0)) — it is still used for vec_size in calculate_chunk_sizes.
////     Inside the chunk block, DELETE the nine `std::string vt_* = vect_type_*;` copies.
////
//// D4. In the chunk loop, replace the helper calls:
////
////       fn_MVP_compute_lp_GHK_cols(t, Bound_U_Phi_Bound_Z[c], Phi_Z[c], Z_std_norm[c], prob[c],
////                                  y1_log_prob, Bound_Z[c], y_chunk, u_array, Model_args_as_cpp_struct);
////     becomes
////       fn_MVP_compute_lp_GHK_cols<vec>(t, Bound_U_Phi_Bound_Z[c], Phi_Z[c], Z_std_norm[c], prob[c],
////                                       y1_log_prob, Bound_Z[c], y_chunk, u_array, kchoice);
////
////     and
////
////       prob_n = fn_EIGEN_double(log_lik_chunk, "exp", vt_exp);
////     becomes
////       prob_n = log_lik_chunk;
////       apply_inplace<vec, Fn::exp>(prob_n);
////
////     log_sum_exp_general(lp_array, vt_exp, vt_log, log_sum_result, container_max_logs)
////     becomes (once you have migrated log_sum_exp_general to the same pattern)
////       log_sum_exp_general<vec>(lp_array, log_sum_result, container_max_logs);
////     until then leave it and keep ONE `const std::string vt_exp/vt_log` pair for it.
////
////     fn_MVP_compute_phi_Z_recip_cols / fn_MVP_compute_phi_Bound_Z_cols / fn_MVP_grad_prep /
////     fn_MVP_compute_nuisance* : leave the calls exactly as they are until each helper is
////     migrated; then swap to the <vec> version the same way.
////
//// D5. Collapse the duplicated LAST-CHUNK block. With the masked tail, the remainder chunk
////     runs the SAME instantiation as the full chunks, so the entire
////       "// LAST CHUNK (remainder) — processed serially with fallback SIMD" block
////     (the Model_args_last_chunk copy, the nine "Stan" string overrides, the duplicated
////     class/test/grad loops) is deleted. What remains is the workspace resize, which you
////     fold into the main loop by iterating to n_total_chunks instead of n_full_chunks:
////
////       for (int nc = 0; nc < n_total_chunks; nc++) {
////
////         const int chunk_counter = nc;
////         const bool is_last = (nc == n_full_chunks);          //// true only for the remainder chunk
////         const int chunk_size = is_last ? last_chunk_size : normal_chunk_size;
////
////         if (is_last) {
////           //// resize the workspace ONCE for the remainder (this is the block of ws.*.resize(...)
////           //// calls you already have; keep it verbatim, it just moves here).
////           ws.resize_to(last_chunk_size, n_tests, n_class);     //// or the inline resize list
////         }
////
////         //// ... the existing chunk body, UNCHANGED, using `chunk_size` ...
////       }
////
////     Note `n_full_chunks < n_total_chunks` iff last_chunk_size > 0, so `is_last` is only
////     ever true when there is a remainder; no extra guard needed.
////     Model_args_as_cpp_struct is passed to the un-migrated helpers for ALL chunks now —
////     which is correct, because the masked tail means the AVX path handles any length.
////
////     (If you prefer not to touch the loop structure yet: keep the last-chunk block but
////     delete the Model_args_last_chunk copy and the "Stan" overrides, and pass
////     Model_args_as_cpp_struct + kchoice instead. Same result, more code left behind.)
////
//// D6. Remove the two `fn_MVP_compute_nuisance`-family calls' dependence on Model_args_last_chunk
////     the moment those helpers are migrated (they will take <vec> and nothing else).
////
//// ORDER OF WORK
////   1. Add fn_dispatch_templated.hpp + PART 0. Build. Export dispatch_self_test() to R, run it.
////   2. Add PART C (fn_MVP_compute_lp_GHK_cols<vec>) alongside the old one (different signature,
////      they coexist). Do D1, D2, D4 (just the GHK call + prob_n). Build. Run a model, compare lp
////      against the AD path at one theta — they must agree to what they agreed before.
////   3. D3, D5. Build, same check.
////   4. Migrate the remaining helpers one by one (phi_Z_recip, phi_Bound_Z, nuisance, jac_u,
////      first_deriv, log_det_J, log_sum_exp_general), building after each.
////   5. Same for the MVOP driver with PART A/B.

