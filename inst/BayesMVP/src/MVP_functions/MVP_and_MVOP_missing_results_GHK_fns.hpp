//// MVP_and_MVOP_missing_results_GHK_fns.hpp
////
//// Missing test results (coded y = -1) in the MVP-based models with manual gradients (binary LC-MVP / MVP and
//// LC-MVOP / MVOP), by the GHK method:
////
////   - the latent value of a missing result is the UNTRUNCATED GHK draw from its own nuisance value u,
////       Z = F^{-1}(u),
////     with F the CDF of the Phi_type in use (Phi or Phi_approx; the kernel the GHK step itself uses,
////     inv_Phi_col_inplace with the same KernelChoice, so the types are never mixed);
////   - Z keeps its place in the conditioning chain: it enters the bounds of the later tests through L_Omega, and
////     the gradients of the later tests flow back to it (and so to its u and to L_Omega) through the existing
////     reverse recursion;
////   - there is NO likelihood term: prob = 1, log(prob) = 0;
////   - the log-Jacobian term of u stays (the nuisance transform is applied to every u, missing or not).
////
//// This is exactly the GHK step of an observed result whose truncation interval is the whole real line, i.e.
//// lower bound -Inf and upper bound +Inf:
////   binary tests (the reflected form of fn_MVP_compute_lp_GHK_cols_T) - with y = 1 and Phi(Bound_Z) = 0:
////       prob = 1,  Phi(Bound_Z) = 0,  Phi_Z = u,  Z = F^{-1}(u),  phi(Bound_Z) = 0,
////     so the binary gradient formulas (which only see prob, y_sign = 2y - 1, y - y_sign u, phi(Bound_Z) and
////     1 / phi(Z)) give, for the missing entry:
////       d prob / d Bound_Z = 0           (y_sign phi(Bound_Z) = 0: no dependence on the linear predictor, the
////                                         coefficients or the earlier Z's through the bound),
////       d Z / d Bound_Z    = 0           ((y - y_sign u) phi(Bound_Z) / phi(Z) = 0),
////       d Z / d u          = 1 / phi(Z)  (prob / phi(Z) with prob = 1).
////     Bound_Z itself stays finite (it is only ever multiplied by phi(Bound_Z) = 0), so no Inf x 0 appears.
////   ordinal tests - the existing code already handles infinite bounds (category 1 and category K): with BOTH
////     bounds infinite, prob = Phi(+Inf) - Phi(-Inf) = 1, q = u, Z = F^{-1}(u), phi(bounds) = 0, and every
////     cutpoint gradient term of the entry is exactly 0 (each is a multiple of phi(lower) or phi(upper)).
////
//// Nothing here is called unless a chunk has at least one missing result: data without any y = -1 run the
//// current code path unchanged (the only extra work is one pass of integer comparisons over the chunk of y,
//// fn_MVP_rows_of_y_have_a_missing_result).
////
//// Requires: MVP_helpers_migrated.hpp (KernelChoice, inv_Phi_col_inplace) and fn_dispatch_templated.hpp (Vec).

#pragma once

#include <Eigen/Dense>
#include <limits>




//// -------------------------------------------------------------------------------------
//// 1. Whether rows row_start, ..., row_start + n_rows - 1 of y have at least one missing result (y < 0).
////    One pass of integer comparisons; decides once per chunk whether the missing-result path is taken.
//// -------------------------------------------------------------------------------------
inline bool fn_MVP_rows_of_y_have_a_missing_result(  const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                     const int row_start,
                                                     const int n_rows
) {

        return (y_ref.middleRows(row_start, n_rows).array() < 0).any();

}




//// -------------------------------------------------------------------------------------
//// 2. The indicator of the missing results of a chunk (1.0 where y < 0, else 0.0), and the recoding of every
////    missing result to y = 1:
////      - binary tests: y_sign = 1 and y - y_sign u = 1 - u, which the binary gradient formulas only ever
////        multiply by phi(Bound_Z) = 0 (fn_MVP_set_phi_Bound_Z_col_of_missing_results_to_zero); the recoded
////        value never reaches the likelihood (fn_MVP_set_GHK_col_of_missing_results_to_the_untruncated_draw);
////      - ordinal tests: category 1, so that every index into the cutpoints stays valid; the bounds of the entry
////        are then set to -Inf and +Inf (fn_MVOP_set_bounds_of_missing_results_to_minus_and_plus_infinity), so
////        that its phi(bounds) = 0 and its cutpoint gradient terms are exactly 0.
//// -------------------------------------------------------------------------------------
inline void fn_MVP_missing_result_indicator_and_recode_missing_results_to_y_equals_1(
                                     Eigen::Matrix<double, -1, -1> &y_chunk,             //// updated
                                     Eigen::Matrix<double, -1, -1> &is_missing_result    //// output
) {

        is_missing_result = (y_chunk.array() < 0.0).cast<double>();
        y_chunk.array() = (is_missing_result.array() > 0.5).select(1.0, y_chunk.array());

}




//// -------------------------------------------------------------------------------------
//// 3. The untruncated GHK draws Z = F^{-1}(u) of a chunk, column by column, for the columns that have at least
////    one missing result (the other columns are not read). They depend on u only: computed once per chunk, for
////    every class. F^{-1} is the inverse the GHK step uses (inv_Phi_col_inplace with the same KernelChoice:
////    exact inv_Phi for Phi_type = "Phi", inv_Phi_approx for Phi_type = "Phi_approx").
//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_MVP_compute_untruncated_GHK_Z_of_missing_results_T(
                                     Eigen::Matrix<double, -1, -1> &Z_untruncated,          //// output
                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array,
                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> is_missing_result,
                                     const KernelChoice &k
) {

        const int n_tests = static_cast<int>(u_array.cols());
        Z_untruncated = u_array;
        for (int t = 0; t < n_tests; t++) {
          if ((is_missing_result.col(t).array() > 0.5).any()) inv_Phi_col_inplace<vec>(Z_untruncated, t, k);
        }

}




//// -------------------------------------------------------------------------------------
//// 4. After the GHK step of column t (fn_MVP_compute_lp_GHK_cols_T, i.e. a binary test), the missing entries
////    of the column are set to the untruncated draw (the GHK step with Phi(Bound_Z) = 0, see the top):
////      prob = 1, log(prob) = 0, Phi(Bound_Z) = 0, Phi_Z = u, Z = F^{-1}(u).
////    Column-wise selects (SIMD blends), no loop over the rows. Must run BEFORE Z of column t is used for the
////    increments of the later tests and before phi(Z) of column t is formed.
//// -------------------------------------------------------------------------------------
inline void fn_MVP_set_GHK_col_of_missing_results_to_the_untruncated_draw(
                                     const int t,
                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> is_missing_result,
                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> u_array,
                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Z_untruncated,
                                     Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z,
                                     Eigen::Ref<Eigen::Matrix<double, -1, -1>> Phi_Z,
                                     Eigen::Ref<Eigen::Matrix<double, -1, -1>> Z_std_norm,
                                     Eigen::Ref<Eigen::Matrix<double, -1, -1>> prob,
                                     Eigen::Ref<Eigen::Matrix<double, -1, -1>> y1_log_prob
) {

        const auto result_is_missing = (is_missing_result.col(t).array() > 0.5);
        prob.col(t).array()                = result_is_missing.select(1.0, prob.col(t).array());
        y1_log_prob.col(t).array()         = result_is_missing.select(0.0, y1_log_prob.col(t).array());
        Bound_U_Phi_Bound_Z.col(t).array() = result_is_missing.select(0.0, Bound_U_Phi_Bound_Z.col(t).array());
        Phi_Z.col(t).array()               = result_is_missing.select(u_array.col(t).array(),
                                                                      Phi_Z.col(t).array());
        Z_std_norm.col(t).array()          = result_is_missing.select(Z_untruncated.col(t).array(),
                                                                      Z_std_norm.col(t).array());

}




//// -------------------------------------------------------------------------------------
//// 5. phi(Bound_Z) = 0 for the missing entries of column t (binary test): the derivative of the GHK step of a
////    missing result with respect to its bound (Phi(Bound_Z) is the constant 0 there). Must run after
////    fn_MVP_compute_phi_Bound_Z_cols_T and before phi(Bound_Z) is used (or its log is taken).
//// -------------------------------------------------------------------------------------
inline void fn_MVP_set_phi_Bound_Z_col_of_missing_results_to_zero(
                                     const int t,
                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> is_missing_result,
                                     Eigen::Ref<Eigen::Matrix<double, -1, -1>> phi_Bound_Z
) {

        phi_Bound_Z.col(t).array() = (is_missing_result.col(t).array() > 0.5)
                                     .select(0.0, phi_Bound_Z.col(t).array());

}




//// -------------------------------------------------------------------------------------
//// 6. Over- / underflow masks of the log-scale (PartialLog) path of the binary LC-MVP: a missing entry is never
////    a tail entry (its prob is exactly 1). Zeroes the mask of the missing entries of column t.
//// -------------------------------------------------------------------------------------
inline void fn_MVP_remove_missing_results_from_tail_mask(
                                     const int t,
                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> is_missing_result,
                                     Eigen::Matrix<int, -1, 1> &tail_mask
) {

        tail_mask.array() = (is_missing_result.col(t).array() > 0.5).select(0, tail_mask.array());

}





//// -------------------------------------------------------------------------------------
//// 7. Ordinal test t: both truncation bounds of the missing entries infinite (lower -Inf, upper +Inf), i.e. the
////    untruncated draw; the ordinal GHK step (fn_MVOP_compute_lp_GHK_cols_T) then gives prob = 1, q = u and
////    Z = F^{-1}(u), and the existing guards give phi(bounds) = 0 and x phi(x) = 0 for the infinite bounds.
//// -------------------------------------------------------------------------------------
inline void fn_MVOP_set_bounds_of_missing_results_to_minus_and_plus_infinity(
                                     const int t,
                                     const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> is_missing_result,
                                     Eigen::Ref<Eigen::Matrix<double, -1, -1>> Bound_Z,
                                     Eigen::Ref<Eigen::Matrix<double, -1, -1>> Upper_Bound_Z
) {

        const double Inf = std::numeric_limits<double>::infinity();
        const auto result_is_missing = (is_missing_result.col(t).array() > 0.5);
        Bound_Z.col(t).array()       = result_is_missing.select(-Inf, Bound_Z.col(t).array());
        Upper_Bound_Z.col(t).array() = result_is_missing.select( Inf, Upper_Bound_Z.col(t).array());

}






















