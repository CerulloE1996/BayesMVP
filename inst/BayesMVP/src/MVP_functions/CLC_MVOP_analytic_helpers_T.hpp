#pragma once

//// -------------------------------------------------------------------------------------
//// CLC_MVOP_analytic_helpers_T.hpp
////
//// Pure-double analytic forward and reverse helpers for CLC-MVOP priors, cutpoints,
//// and Pinkney bounded LDL correlation transforms, replacing per-leapfrog-step
//// Stan Math autodiff tape passes.
//// -------------------------------------------------------------------------------------

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include "MVOP_PartialLog_helper_fns.hpp"

namespace clc_mvop_detail {

static constexpr double CLC_ANALYTIC_LOG_2 = 0.693147180559945309417232121458176568;
static constexpr double CLC_ANALYTIC_HALF_LOG_2PI = 0.918938533204672741780329736405617640;

#if BMVP_HAS_AVX512
static constexpr Vec CLC_ANALYTIC_VEC = Vec::AVX512;
#elif BMVP_HAS_AVX2
static constexpr Vec CLC_ANALYTIC_VEC = Vec::AVX2;
#else
static constexpr Vec CLC_ANALYTIC_VEC = Vec::Scalar;
#endif

//// For a fixed pooled curve q, b is the second class's weighted cumulative mass.
//// The first class's mass q - b is ordered by the same interval constraints.
inline bool clc_split_pooled_cutpoint(
      const double q_left,
      const double q,
      const double q_right,
      const double b_left,
      const double b_right,
      const double u,
      const double one_minus_u,
      double &b,
      double &fraction,
      double &one_minus_fraction,
      double &width,
      double &d_b_d_centre,
      double &d_b_d_lower,
      double &d_b_d_upper
) {

      const double pooled_width = q_right - q_left;
      const double b_width = b_right - b_left;
      if (!(q > q_left && q < q_right && b_width > 0.0 && b_width < pooled_width)) return false;

      const double lower_candidate = b_right - (q_right - q);
      const double upper_candidate = b_left + (q - q_left);
      const double lower = (lower_candidate > b_left) ? lower_candidate : b_left;
      const double upper = (upper_candidate < b_right) ? upper_candidate : b_right;
      const double centre = b_left + ((q - q_left) / pooled_width) * b_width;
      const double left_width = centre - lower;
      const double right_width = upper - centre;
      if (!(left_width > 0.0 && right_width > 0.0 && u > 0.0 && one_minus_u > 0.0)) return false;

      //// The median splits the intervening mass proportionally; the full interval remains accessible.
      const double scale = (left_width > right_width) ? left_width : right_width;
      const double numerator = u * (left_width / scale);
      const double complement = one_minus_u * (right_width / scale);
      const double denominator = numerator + complement;
      fraction = numerator / denominator;
      one_minus_fraction = complement / denominator;
      width = upper - lower;
      b = (fraction <= one_minus_fraction) ? (lower + width * fraction)
                                         : (upper - width * one_minus_fraction);

      const double width_over_scale = width / scale;
      const double lower_term = (width_over_scale * one_minus_fraction) * (u / denominator);
      const double upper_term = (width_over_scale * fraction) * (one_minus_u / denominator);
      d_b_d_centre = lower_term + upper_term;
      d_b_d_lower = one_minus_fraction - lower_term;
      d_b_d_upper = fraction - upper_term;
      return (b > lower && b < upper);

}

//// Reverse after accumulating the derivatives from all cut-offs split using these endpoints.
inline void clc_split_pooled_cutpoint_reverse(
      const double q_left,
      const double q,
      const double q_right,
      const double b_left,
      const double b_right,
      const double fraction,
      const double one_minus_fraction,
      const double width,
      const double d_b_d_centre,
      const double d_b_d_lower,
      const double d_b_d_upper,
      const double d_b,
      double &d_q_left,
      double &d_q,
      double &d_q_right,
      double &d_b_left,
      double &d_b_right,
      double &d_split_logit
) {

      const double pooled_width = q_right - q_left;
      const double position = (q - q_left) / pooled_width;
      const double d_centre = d_b * d_b_d_centre;
      const double position_scale = d_centre * ((b_right - b_left) / pooled_width);
      const double d_lower = d_b * d_b_d_lower;
      const double d_upper = d_b * d_b_d_upper;

      d_q += position_scale;
      d_q_left -= (1.0 - position) * position_scale;
      d_q_right -= position * position_scale;
      d_b_left += (1.0 - position) * d_centre;
      d_b_right += position * d_centre;
      d_split_logit += width * fraction * one_minus_fraction * d_b;

      if (b_right - (q_right - q) > b_left) {
            d_b_right += d_lower;
            d_q_right -= d_lower;
            d_q += d_lower;
      } else {
            d_b_left += d_lower;
      }

      if (b_left + (q - q_left) < b_right) {
            d_b_left += d_upper;
            d_q_left -= d_upper;
            d_q += d_upper;
      } else {
            d_b_right += d_upper;
      }

}

//// Use indices 1..n_cutoffs; the end masses at 0 and n_cutoffs + 1 are supplied by the caller.
//// split_steps has six columns of length n_cutoffs + 2 and is allocated before sampling.
inline bool clc_split_pooled_cutpoint_curve(
      const int n_cutoffs,
      const int n_anchors,
      const int *anchors,
      const double *q,
      const double *u,
      const double *one_minus_u,
      double *b,
      double *split_steps
) {

      const int stride = n_cutoffs + 2;
      const int last = n_cutoffs + 1;
      if (n_anchors < 0 || n_anchors > n_cutoffs) return false;

      const auto split = [&](const int left, const int k, const int right) {
            return clc_split_pooled_cutpoint(q[left], q[k], q[right], b[left], b[right], u[k], one_minus_u[k],
                                            b[k], split_steps[k], split_steps[stride + k],
                                            split_steps[2 * stride + k], split_steps[3 * stride + k],
                                            split_steps[4 * stride + k], split_steps[5 * stride + k]);
      };

      int left = 0;
      for (int j = 0; j < n_anchors; ++j) {
            const int k = anchors[j];
            if (!(k > left && k < last) || !split(left, k, last)) return false;
            left = k;
      }
      left = 0;
      for (int j = 0; j <= n_anchors; ++j) {
            const int right = (j < n_anchors) ? anchors[j] : last;
            for (int k = left + 1; k < right; ++k) {
                  if (!split(k - 1, k, right)) return false;
            }
            left = right;
      }
      return true;

}

//// d_b is seeded with the two class-curve contributions before reversing the split.
inline void clc_split_pooled_cutpoint_curve_reverse(
      const int n_cutoffs,
      const int n_anchors,
      const int *anchors,
      const double *q,
      const double *b,
      const double *split_steps,
      double *d_q,
      double *d_b,
      double *d_split_logit
) {

      const int stride = n_cutoffs + 2;
      const int last = n_cutoffs + 1;
      const auto reverse = [&](const int left, const int k, const int right) {
            clc_split_pooled_cutpoint_reverse(q[left], q[k], q[right], b[left], b[right],
                                              split_steps[k], split_steps[stride + k],
                                              split_steps[2 * stride + k], split_steps[3 * stride + k],
                                              split_steps[4 * stride + k], split_steps[5 * stride + k], d_b[k],
                                              d_q[left], d_q[k], d_q[right], d_b[left], d_b[right],
                                              d_split_logit[k]);
      };

      int right = last;
      for (int j = n_anchors - 1; j >= -1; --j) {
            const int left = (j >= 0) ? anchors[j] : 0;
            for (int k = right - 1; k > left; --k) reverse(k - 1, k, right);
            right = left;
      }
      for (int j = n_anchors - 1; j >= 0; --j) {
            const int left = (j > 0) ? anchors[j - 1] : 0;
            reverse(left, anchors[j], last);
      }

}

//// -------------------------------------------------------------------------------------
//// Buffers of the analytic prior helpers, sized once before sampling and reused
//// -------------------------------------------------------------------------------------
static constexpr int CLC_SURVEY_TILE_LEN = 256;
static constexpr int CLC_SURVEY_N_SLOTS = 23;

struct CLC_analytic_workspace {
      int n_slot_len = 0;
      Eigen::Matrix<double, -1, 1> survey_thr_buf;
      Eigen::Matrix<int, -1, 1> survey_thr_int_buf;
      Eigen::Matrix<double, -1, 1> survey_tile_buf;
      Eigen::Matrix<double, -1, 1> log_Phi_pair_buf;
      Eigen::Matrix<double, -1, 1> cutpoint_buf;
      Eigen::Matrix<double, -1, 1> chosen_buf;
      Eigen::Matrix<double, -1, 1> Se_Sp_buf;
      Eigen::Matrix<double, -1, 1> batch_buf;
      Eigen::Matrix<double, -1, -1> bar_lower_buf;
      Eigen::Matrix<double, -1, 1> bar_diag_buf;
      Eigen::Matrix<double, -1, -1> cholesky_work_buf;

      void ensure(
            const int n_cutpoints_max,
            const int n_chosen_max,
            const int n_tests
      ) {
            const int n_batch_max = 24 * n_chosen_max;
            if (n_slot_len != std::max(n_cutpoints_max, 1)) {
                  n_slot_len = std::max(n_cutpoints_max, 1);
                  survey_thr_buf.resize(CLC_SURVEY_N_SLOTS * n_slot_len);
                  survey_thr_int_buf.resize(2 * n_slot_len);
                  survey_tile_buf.resize(5 * CLC_SURVEY_TILE_LEN);
            }
            const int n_pair_max = std::max(
                  std::max(
                        n_cutpoints_max,
                        n_batch_max
                  ),
                  1
            );
            if (log_Phi_pair_buf.rows() != 2 * n_pair_max) {
                  log_Phi_pair_buf.resize(2 * n_pair_max);
                  cutpoint_buf.resize(6 * std::max(n_cutpoints_max, 1) + 1);
                  chosen_buf.resize(31 * std::max(n_chosen_max, 1) + 1);
                  Se_Sp_buf.resize(4 * std::max(n_chosen_max, 1));
                  batch_buf.resize(6 * std::max(n_batch_max, 1));
            }
            if (bar_lower_buf.rows() != n_tests) {
                  bar_lower_buf.resize(
                        n_tests,
                        n_tests
                  );
                  bar_diag_buf.resize(n_tests);
                  cholesky_work_buf.resize(
                        n_tests,
                        n_tests
                  );
            }
      }
};

inline void clc_apply_log_Phi_pair_vec(
      const double *x_in,
      double *lcdf_out,
      double *lccdf_out,
      const int n_elems,
      CLC_analytic_workspace &analytic_ws
) {
      //// exact log-Phi of x and of -x in one call
      const int n_total = 2 * n_elems;
      double *pair_ptr = analytic_ws.log_Phi_pair_buf.data();
      for (int idx = 0; idx < n_elems; idx++) {
            pair_ptr[idx] = x_in[idx];
            pair_ptr[n_elems + idx] = -x_in[idx];
      }
      fn_apply_log_Phi_exact_raw_T<CLC_ANALYTIC_VEC>(
            pair_ptr,
            n_total
      );
      for (int idx = 0; idx < n_elems; idx++) {
            lcdf_out[idx] = pair_ptr[idx];
            lccdf_out[idx] = pair_ptr[n_elems + idx];
      }
}

inline double clc_do_lkj_constant_dbl(
      const double eta_val,
      const int dim_size
) {
      if (dim_size <= 1) {
            return 0.0;
      }
      return stan::math::do_lkj_constant(
            eta_val,
            static_cast<unsigned int>(dim_size)
      );
}

//// -------------------------------------------------------------------------------------
//// Pure-double log-scale induced-Dirichlet value and analytic gradient w.r.t. cutpoints
//// -------------------------------------------------------------------------------------
inline double clc_induced_dirichlet_log_scale_val_grad(
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> cutpoints_vec,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> alpha_vec,
      const bool include_dirichlet_const,
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> d_cutpoints_out,
      CLC_analytic_workspace &analytic_ws
) {
      const int n_thr = static_cast<int>(cutpoints_vec.rows());
      double *log_cdf_ptr = analytic_ws.cutpoint_buf.data();
      double *log_ccdf_ptr = log_cdf_ptr + n_thr;
      double *log_pdf_ptr = log_ccdf_ptr + n_thr;
      double *log_p_ptr = log_pdf_ptr + n_thr;

      clc_apply_log_Phi_pair_vec(
            cutpoints_vec.data(),
            log_cdf_ptr,
            log_ccdf_ptr,
            n_thr,
            analytic_ws
      );
      double log_prob = 0.0;
      for (int thr_index = 0; thr_index < n_thr; thr_index++) {
            const double c_val = cutpoints_vec(thr_index);
            log_pdf_ptr[thr_index] = -0.5 * c_val * c_val - CLC_ANALYTIC_HALF_LOG_2PI;
            log_prob += log_pdf_ptr[thr_index];
      }

      log_p_ptr[0] = log_cdf_ptr[0];
      log_p_ptr[n_thr] = log_ccdf_ptr[n_thr - 1];
      for (int thr_index = 1; thr_index < n_thr; thr_index++) {
            if (cutpoints_vec(thr_index - 1) >= 0.0) {
                  log_p_ptr[thr_index] = sl_log_diff_exp(log_ccdf_ptr[thr_index - 1], log_ccdf_ptr[thr_index], false);
            } else {
                  log_p_ptr[thr_index] = sl_log_diff_exp(log_cdf_ptr[thr_index], log_cdf_ptr[thr_index - 1], false);
            }
      }

      if (include_dirichlet_const) {
            double sum_alpha = 0.0;
            double sum_lgamma = 0.0;
            for (int cat_index = 0; cat_index <= n_thr; cat_index++) {
                  const double a_cur = alpha_vec(cat_index);
                  sum_alpha += a_cur;
                  sum_lgamma += std::lgamma(a_cur);
            }
            log_prob += std::lgamma(sum_alpha) - sum_lgamma;
      }

      for (int cat_index = 0; cat_index <= n_thr; cat_index++) {
            const double a_m1 = alpha_vec(cat_index) - 1.0;
            if (a_m1 != 0.0) {
                  log_prob += a_m1 * log_p_ptr[cat_index];
            }
      }

      double *ratio_ptr = log_p_ptr + (n_thr + 1);
      for (int thr_index = 0; thr_index < n_thr; thr_index++) {
            ratio_ptr[thr_index] = log_pdf_ptr[thr_index] - log_p_ptr[thr_index];
            ratio_ptr[n_thr + thr_index] = log_pdf_ptr[thr_index] - log_p_ptr[thr_index + 1];
      }
      apply_raw<CLC_ANALYTIC_VEC, Fn::exp, true>(
            ratio_ptr,
            2 * n_thr
      );
      for (int thr_index = 0; thr_index < n_thr; thr_index++) {
            double grad_val = -cutpoints_vec(thr_index);
            const double a_left_m1 = alpha_vec(thr_index) - 1.0;
            if (a_left_m1 != 0.0) {
                  grad_val += a_left_m1 * ratio_ptr[thr_index];
            }
            const double a_right_m1 = alpha_vec(thr_index + 1) - 1.0;
            if (a_right_m1 != 0.0) {
                  grad_val -= a_right_m1 * ratio_ptr[n_thr + thr_index];
            }
            d_cutpoints_out(thr_index) += grad_val;
      }

      return log_prob;
}

//// -------------------------------------------------------------------------------------
//// Pure-double 24-node Laplace-centred Gauss-Hermite baseline marginal value and
//// analytic gradient w.r.t. x_chosen
//// -------------------------------------------------------------------------------------
inline double clc_log_baseline_marginal_val_grad(
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> x_chosen_vec,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> alpha_seg,
      const double v0_val,
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> d_x_out,
      CLC_analytic_workspace &analytic_ws
) {
      const int n_chosen = static_cast<int>(x_chosen_vec.rows());
      if (n_chosen == 0) {
            return 0.0;
      }
      if (v0_val <= 1e-15) {
            return clc_induced_dirichlet_log_scale_val_grad(
                  x_chosen_vec,
                  alpha_seg,
                  true,
                  d_x_out,
                  analytic_ws
            );
      }

      static const double gh_t[24] = {
            -8.50780351919527433e+00, -7.43789066602167104e+00, -6.54167500509862876e+00, -5.73274717525120181e+00,
            -4.97804137463911367e+00, -4.26038360501990354e+00, -3.56930676407355918e+00, -2.89772864322331181e+00,
            -2.24046785169175244e+00, -1.59348042981641846e+00, -9.53421922932108146e-01, -3.17370096629455034e-01,
             3.17370096629453258e-01,  9.53421922932111698e-01,  1.59348042981641846e+00,  2.24046785169175156e+00,
             2.89772864322331269e+00,  3.56930676407356007e+00,  4.26038360501990709e+00,  4.97804137463911989e+00,
             5.73274717525120003e+00,  6.54167500509863320e+00,  7.43789066602166216e+00,  8.50780351919525657e+00
      };
      static const double gh_lw[24] = {
            -3.69042806607304854e+01, -2.86212344883108756e+01, -2.24843031451895179e+01, -1.76020197436324572e+01,
            -1.36185797828409356e+01, -1.03468101270751411e+01, -7.67407824125704963e+00, -5.52732585071174043e+00,
            -3.85723446710271611e+00, -2.63012623460007200e+00, -1.82350086207048112e+00, -1.42349743041213506e+00,
            -1.42349743041215793e+00, -1.82350086207048467e+00, -2.63012623460007866e+00, -3.85723446710268059e+00,
            -5.52732585071170757e+00, -7.67407824125703453e+00, -1.03468101270751269e+01, -1.36185797828409267e+01,
            -1.76020197436324182e+01, -2.24843031451895001e+01, -2.86212344883107441e+01, -3.69042806607303291e+01
      };

      double sum_seg = 0.0;
      double sum_lgamma = 0.0;
      for (int seg_index = 0; seg_index <= n_chosen; seg_index++) {
            const double a_cur = alpha_seg(seg_index);
            sum_seg += a_cur;
            sum_lgamma += std::lgamma(a_cur);
      }
      const double lg_const = std::lgamma(sum_seg) - sum_lgamma;
      const double inv_v0 = 1.0 / v0_val;

      double b_star = 0.0;
      double cum_w = 0.0;
      for (int chosen_index = 0; chosen_index < n_chosen; chosen_index++) {
            cum_w += alpha_seg(chosen_index);
            const double p_mid = cum_w / sum_seg;
            const double q_mid = stan::math::inv_Phi(p_mid);
            b_star += (q_mid - x_chosen_vec(chosen_index));
      }
      b_star /= static_cast<double>(n_chosen);

      double *z_ptr = analytic_ws.chosen_buf.data();
      double *lcdf_ptr = z_ptr + n_chosen;
      double *lccdf_ptr = lcdf_ptr + n_chosen;
      double *lpdf_ptr = lccdf_ptr + n_chosen;
      double *lq_ptr = lpdf_ptr + n_chosen;
      double *rown_ptr = lq_ptr + (n_chosen + 1);
      double *rnext_ptr = rown_ptr + n_chosen;
      double *node_grad_ptr = rnext_ptr + n_chosen;

      double sig_star = 1.0;
      for (int newton_iter = 0; newton_iter < 11; newton_iter++) {
            for (int chosen_index = 0; chosen_index < n_chosen; chosen_index++) {
                  const double z_val = x_chosen_vec(chosen_index) + b_star;
                  z_ptr[chosen_index] = z_val;
                  lpdf_ptr[chosen_index] = -0.5 * z_val * z_val - CLC_ANALYTIC_HALF_LOG_2PI;
            }
            clc_apply_log_Phi_pair_vec(
                  z_ptr,
                  lcdf_ptr,
                  lccdf_ptr,
                  n_chosen,
                  analytic_ws
            );
            lq_ptr[0] = lcdf_ptr[0];
            lq_ptr[n_chosen] = lccdf_ptr[n_chosen - 1];
            for (int chosen_index = 1; chosen_index < n_chosen; chosen_index++) {
                  if (z_ptr[chosen_index - 1] >= 0.0) {
                        lq_ptr[chosen_index] = sl_log_diff_exp(lccdf_ptr[chosen_index - 1], lccdf_ptr[chosen_index], false);
                  } else {
                        lq_ptr[chosen_index] = sl_log_diff_exp(lcdf_ptr[chosen_index], lcdf_ptr[chosen_index - 1], false);
                  }
            }
            for (int chosen_index = 0; chosen_index < n_chosen; chosen_index++) {
                  rown_ptr[chosen_index] = lpdf_ptr[chosen_index] - lq_ptr[chosen_index];
                  rnext_ptr[chosen_index] = lpdf_ptr[chosen_index] - lq_ptr[chosen_index + 1];
            }
            apply_raw<CLC_ANALYTIC_VEC, Fn::exp, true>(
                  rown_ptr,
                  n_chosen
            );
            apply_raw<CLC_ANALYTIC_VEC, Fn::exp, true>(
                  rnext_ptr,
                  n_chosen
            );
            double sum_dg = 0.0;
            double sum_d2 = -inv_v0;
            for (int chosen_index = 0; chosen_index < n_chosen; chosen_index++) {
                  sum_dg += (alpha_seg(chosen_index) - 1.0) * rown_ptr[chosen_index]
                            - (alpha_seg(chosen_index + 1) - 1.0) * rnext_ptr[chosen_index]
                            - z_ptr[chosen_index];
                  sum_d2 -= 1.0;
            }
            if (alpha_seg(0) != 1.0) {
                  sum_d2 += (alpha_seg(0) - 1.0)
                            * (-z_ptr[0] * rown_ptr[0] - rown_ptr[0] * rown_ptr[0]);
            }
            if (alpha_seg(n_chosen) != 1.0) {
                  sum_d2 += (alpha_seg(n_chosen) - 1.0)
                            * (z_ptr[n_chosen - 1] * rnext_ptr[n_chosen - 1]
                               - rnext_ptr[n_chosen - 1] * rnext_ptr[n_chosen - 1]);
            }
            for (int chosen_index = 1; chosen_index < n_chosen; chosen_index++) {
                  if (alpha_seg(chosen_index) != 1.0) {
                        const double diff_ratio = rown_ptr[chosen_index] - rnext_ptr[chosen_index - 1];
                        sum_d2 += (alpha_seg(chosen_index) - 1.0)
                                  * (-z_ptr[chosen_index] * rown_ptr[chosen_index]
                                     + z_ptr[chosen_index - 1] * rnext_ptr[chosen_index - 1]
                                     - diff_ratio * diff_ratio);
                  }
            }
            if (newton_iter == 10) {
                  sig_star = 1.0 / std::sqrt(-sum_d2);
            } else {
                  const double dg_db = -b_star * inv_v0 + sum_dg;
                  double step_val = -dg_db / sum_d2;
                  if (step_val > 2.0) {
                        step_val = 2.0;
                  } else if (step_val < -2.0) {
                        step_val = -2.0;
                  }
                  b_star += step_val;
            }
      }

      const int n_batch = 24 * n_chosen;
      double *z_batch_ptr = analytic_ws.batch_buf.data();
      double *lcdf_batch_ptr = z_batch_ptr + n_batch;
      double *lccdf_batch_ptr = lcdf_batch_ptr + n_batch;
      double *lpdf_batch_ptr = lccdf_batch_ptr + n_batch;
      double *exp_batch_ptr = lpdf_batch_ptr + n_batch;
      for (int node_index = 0; node_index < 24; node_index++) {
            const double b_node = b_star + sig_star * gh_t[node_index];
            const int row_off = node_index * n_chosen;
            for (int chosen_index = 0; chosen_index < n_chosen; chosen_index++) {
                  const double z_val = x_chosen_vec(chosen_index) + b_node;
                  z_batch_ptr[row_off + chosen_index] = z_val;
                  lpdf_batch_ptr[row_off + chosen_index] = -0.5 * z_val * z_val - CLC_ANALYTIC_HALF_LOG_2PI;
            }
      }
      clc_apply_log_Phi_pair_vec(
            z_batch_ptr,
            lcdf_batch_ptr,
            lccdf_batch_ptr,
            n_batch,
            analytic_ws
      );
      double lw_terms[24];
      double max_lw = -std::numeric_limits<double>::infinity();
      for (int node_index = 0; node_index < 24; node_index++) {
            const double b_node = b_star + sig_star * gh_t[node_index];
            const int row_off = node_index * n_chosen;
            const double *z_row = z_batch_ptr + row_off;
            const double *lcdf_row = lcdf_batch_ptr + row_off;
            const double *lccdf_row = lccdf_batch_ptr + row_off;
            const double *lpdf_row = lpdf_batch_ptr + row_off;
            double sum_log_pdf = 0.0;
            for (int chosen_index = 0; chosen_index < n_chosen; chosen_index++) {
                  sum_log_pdf += lpdf_row[chosen_index];
            }
            lq_ptr[0] = lcdf_row[0];
            lq_ptr[n_chosen] = lccdf_row[n_chosen - 1];
            for (int chosen_index = 1; chosen_index < n_chosen; chosen_index++) {
                  if (z_row[chosen_index - 1] >= 0.0) {
                        lq_ptr[chosen_index] = sl_log_diff_exp(
                              lccdf_row[chosen_index - 1],
                              lccdf_row[chosen_index],
                              false
                        );
                  } else {
                        lq_ptr[chosen_index] = sl_log_diff_exp(
                              lcdf_row[chosen_index],
                              lcdf_row[chosen_index - 1],
                              false
                        );
                  }
            }
            double val_node = -0.5 * b_node * b_node * inv_v0 + lg_const + sum_log_pdf;
            for (int seg_index = 0; seg_index <= n_chosen; seg_index++) {
                  const double a_m1 = alpha_seg(seg_index) - 1.0;
                  if (a_m1 != 0.0) {
                        val_node += a_m1 * lq_ptr[seg_index];
                  }
            }
            const double lw_cur = gh_lw[node_index] + val_node + 0.5 * gh_t[node_index] * gh_t[node_index];
            lw_terms[node_index] = lw_cur;
            if (lw_cur > max_lw) {
                  max_lw = lw_cur;
            }
            for (int chosen_index = 0; chosen_index < n_chosen; chosen_index++) {
                  exp_batch_ptr[row_off + chosen_index] = lpdf_row[chosen_index] - lq_ptr[chosen_index];
                  exp_batch_ptr[n_batch + row_off + chosen_index] = lpdf_row[chosen_index] - lq_ptr[chosen_index + 1];
            }
      }
      apply_raw<CLC_ANALYTIC_VEC, Fn::exp, true>(
            exp_batch_ptr,
            2 * n_batch
      );
      for (int node_index = 0; node_index < 24; node_index++) {
            const int row_off = node_index * n_chosen;
            double *g_row = node_grad_ptr + row_off;
            for (int chosen_index = 0; chosen_index < n_chosen; chosen_index++) {
                  double g_cur = -z_batch_ptr[row_off + chosen_index];
                  const double a_left_m1 = alpha_seg(chosen_index) - 1.0;
                  if (a_left_m1 != 0.0) {
                        g_cur += a_left_m1 * exp_batch_ptr[row_off + chosen_index];
                  }
                  const double a_right_m1 = alpha_seg(chosen_index + 1) - 1.0;
                  if (a_right_m1 != 0.0) {
                        g_cur -= a_right_m1 * exp_batch_ptr[n_batch + row_off + chosen_index];
                  }
                  g_row[chosen_index] = g_cur;
            }
      }

      double w_node[24];
      for (int node_index = 0; node_index < 24; node_index++) {
            w_node[node_index] = lw_terms[node_index] - max_lw;
      }
      apply_raw<CLC_ANALYTIC_VEC, Fn::exp, true>(
            w_node,
            24
      );
      double sum_w = 0.0;
      for (int node_index = 0; node_index < 24; node_index++) {
            sum_w += w_node[node_index];
      }
      const double inv_sum_w = 1.0 / sum_w;
      for (int node_index = 0; node_index < 24; node_index++) {
            const double wn = w_node[node_index] * inv_sum_w;
            const double *g_row = node_grad_ptr + node_index * n_chosen;
            for (int chosen_index = 0; chosen_index < n_chosen; chosen_index++) {
                  d_x_out(chosen_index) += wn * g_row[chosen_index];
            }
      }

      double log_scalars[3] = { v0_val, sig_star, sum_w };
      apply_raw<CLC_ANALYTIC_VEC, Fn::log, true>(
            log_scalars,
            3
      );
      return -0.5 * log_scalars[0] + log_scalars[1] + max_lw + log_scalars[2];
}

//// -------------------------------------------------------------------------------------
//// Pure-double Pinkney bounded LDL correlation transform (forward + saved tape + reverse)
//// -------------------------------------------------------------------------------------
struct PinkneyLDLTape {
      int dim_size = 0;
      int n_free = 0;
      Eigen::Matrix<double, -1, -1> lower_tri;
      Eigen::Matrix<double, -1, 1> diag_vec;
      Eigen::Matrix<double, -1, 1> sqrt_diag_vec;
      Eigen::Matrix<double, -1, -1> rem_before;
      Eigen::Matrix<double, -1, -1> pd_bound;
      Eigen::Matrix<double, -1, -1> entry_val;
      Eigen::Matrix<double, -1, -1> range_val;
      Eigen::Matrix<double, -1, -1> tanh_val;
      Eigen::Matrix<int, -1, -1> clamp_lo;
      Eigen::Matrix<int, -1, -1> clamp_up;
      Eigen::Matrix<int, -1, -1> free_idx;
      Eigen::Matrix<double, -1, -1> L_Omega;

      void ensure(
            const int dim_req
      ) {
            if ((dim_size != dim_req) || (lower_tri.rows() != dim_req)) {
                  dim_size = dim_req;
                  lower_tri.resize(dim_req, dim_req);
                  diag_vec.resize(dim_req);
                  sqrt_diag_vec.resize(dim_req);
                  rem_before.resize(dim_req, dim_req);
                  pd_bound.resize(dim_req, dim_req);
                  entry_val.resize(dim_req, dim_req);
                  range_val.resize(dim_req, dim_req);
                  tanh_val.resize(dim_req, dim_req);
                  clamp_lo.resize(dim_req, dim_req);
                  clamp_up.resize(dim_req, dim_req);
                  free_idx.resize(dim_req, dim_req);
                  L_Omega.resize(dim_req, dim_req);
            }
      }
};

inline void clc_pinkney_LDL_bounds_free_forward_double(
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> raw_vec,
      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> lb_corr,
      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> ub_corr,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> known_ind,
      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> known_val,
      PinkneyLDLTape &tape_out,
      double &log_jac
) {
      const int dim_size = static_cast<int>(lb_corr.rows());
      tape_out.ensure(dim_size);
      tape_out.lower_tri.setZero();
      for (int diag_index = 0; diag_index < dim_size; diag_index++) {
            tape_out.lower_tri(diag_index, diag_index) = 1.0;
      }
      tape_out.diag_vec(0) = 1.0;
      int free_counter = 0;

      for (int row_index = 1; row_index < dim_size; row_index++) {
            double remainder_val = 1.0;
            for (int col_index = 0; col_index < row_index; col_index++) {
                  double dot_val = 0.0;
                  for (int inner_index = 0; inner_index < col_index; inner_index++) {
                        dot_val += tape_out.lower_tri(col_index, inner_index)
                                   * tape_out.diag_vec(inner_index)
                                   * tape_out.lower_tri(row_index, inner_index);
                  }
                  const double d_col = tape_out.diag_vec(col_index);
                  const double pd_b = std::sqrt(remainder_val * d_col);
                  tape_out.rem_before(row_index, col_index) = remainder_val;
                  tape_out.pd_bound(row_index, col_index) = pd_b;

                  double entry_cur = 0.0;
                  if (known_ind(row_index, col_index) == 1) {
                        entry_cur = known_val(row_index, col_index) - dot_val;
                        if (std::abs(entry_cur) >= pd_b) {
                              throw std::domain_error("CLC-MVOP: fixed correlation incompatible with positive definiteness.");
                        }
                        tape_out.free_idx(row_index, col_index) = -1;
                  } else {
                        const double low_cand = lb_corr(row_index, col_index) - dot_val;
                        const double up_cand = ub_corr(row_index, col_index) - dot_val;
                        const int use_low_cand = (low_cand > -pd_b) ? 1 : 0;
                        const int use_up_cand = (up_cand < pd_b) ? 1 : 0;
                        const double low_val = (use_low_cand == 1) ? low_cand : (-pd_b);
                        const double up_val = (use_up_cand == 1) ? up_cand : pd_b;
                        if (low_val >= up_val) {
                              throw std::domain_error("CLC-MVOP: empty correlation interval.");
                        }
                        const double unc_corr_val = raw_vec(free_counter);
                        tape_out.free_idx(row_index, col_index) = free_counter;
                        free_counter++;
                        const double range_cur = up_val - low_val;
                        double tanh_buf[1] = { unc_corr_val };
                        apply_raw<CLC_ANALYTIC_VEC, Fn::tanh, true>(tanh_buf, 1);
                        const double tanh_cur = tanh_buf[0];
                        tape_out.clamp_lo(row_index, col_index) = use_low_cand;
                        tape_out.clamp_up(row_index, col_index) = use_up_cand;
                        tape_out.range_val(row_index, col_index) = range_cur;
                        tape_out.tanh_val(row_index, col_index) = tanh_cur;

                        const double abs_u = (unc_corr_val >= 0.0) ? unc_corr_val : (-unc_corr_val);
                        double l1p_buf[1] = { -2.0 * abs_u };
                        apply_raw<CLC_ANALYTIC_VEC, Fn::log1p_exp, true>(l1p_buf, 1);
                        double log2_buf[2] = { range_cur, d_col };
                        apply_raw<CLC_ANALYTIC_VEC, Fn::log, true>(log2_buf, 2);
                        log_jac += log2_buf[0]
                                   + CLC_ANALYTIC_LOG_2
                                   - 2.0 * abs_u
                                   - 2.0 * l1p_buf[0];
                        entry_cur = low_val + range_cur * 0.5 * (1.0 + tanh_cur);
                        log_jac += -0.5 * log2_buf[1];
                  }
                  tape_out.entry_val(row_index, col_index) = entry_cur;
                  tape_out.lower_tri(row_index, col_index) = entry_cur / d_col;
                  remainder_val -= (entry_cur * entry_cur) / d_col;
            }
            tape_out.diag_vec(row_index) = remainder_val;
      }
      tape_out.n_free = free_counter;

      tape_out.L_Omega.setZero();
      for (int col_index = 0; col_index < dim_size; col_index++) {
            tape_out.sqrt_diag_vec(col_index) = std::sqrt(tape_out.diag_vec(col_index));
      }
      for (int row_index = 0; row_index < dim_size; row_index++) {
            for (int col_index = 0; col_index <= row_index; col_index++) {
                  tape_out.L_Omega(row_index, col_index) =
                        tape_out.lower_tri(row_index, col_index) * tape_out.sqrt_diag_vec(col_index);
            }
      }
}

inline void clc_pinkney_LDL_bounds_free_reverse_double(
      const PinkneyLDLTape &tape_in,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> known_ind,
      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> bar_L_Omega,
      const double bar_log_jac,
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> bar_raw_out,
      CLC_analytic_workspace &analytic_ws
) {
      const int dim_size = tape_in.dim_size;
      Eigen::Ref<Eigen::Matrix<double, -1, -1>> bar_lower = analytic_ws.bar_lower_buf.topLeftCorner(
            dim_size,
            dim_size
      );
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> bar_diag = analytic_ws.bar_diag_buf.head(dim_size);
      bar_lower.setZero();
      bar_diag.setZero();

      for (int row_index = 0; row_index < dim_size; row_index++) {
            for (int col_index = 0; col_index <= row_index; col_index++) {
                  const double b_L = bar_L_Omega(row_index, col_index);
                  if (b_L != 0.0) {
                        const double sq_d = tape_in.sqrt_diag_vec(col_index);
                        bar_lower(row_index, col_index) += b_L * sq_d;
                        bar_diag(col_index) += b_L * tape_in.lower_tri(row_index, col_index) / (2.0 * sq_d);
                  }
            }
      }

      for (int row_index = dim_size - 1; row_index >= 1; row_index--) {
            double bar_rem = bar_diag(row_index);
            for (int col_index = row_index - 1; col_index >= 0; col_index--) {
                  const double d_col = tape_in.diag_vec(col_index);
                  const double inv_d = 1.0 / d_col;
                  const double inv_d2 = inv_d * inv_d;
                  const double entry_cur = tape_in.entry_val(row_index, col_index);

                  double bar_entry = -2.0 * bar_rem * entry_cur * inv_d;
                  bar_diag(col_index) += bar_rem * (entry_cur * entry_cur) * inv_d2;

                  const double b_lo = bar_lower(row_index, col_index);
                  bar_entry += b_lo * inv_d;
                  bar_diag(col_index) -= b_lo * entry_cur * inv_d2;

                  double bar_dot = 0.0;
                  if (known_ind(row_index, col_index) == 1) {
                        bar_dot = -bar_entry;
                  } else {
                        const int f_idx = tape_in.free_idx(row_index, col_index);
                        const double range_cur = tape_in.range_val(row_index, col_index);
                        const double tanh_cur = tape_in.tanh_val(row_index, col_index);
                        const double s_val = 0.5 * (1.0 + tanh_cur);
                        const double sech2_val = 1.0 - tanh_cur * tanh_cur;

                        bar_diag(col_index) += -0.5 * bar_log_jac * inv_d;
                        bar_raw_out(f_idx) += bar_log_jac * (-2.0 * tanh_cur)
                                              + bar_entry * range_cur * 0.5 * sech2_val;
                        const double bar_range = bar_log_jac / range_cur + bar_entry * s_val;
                        const double bar_up = bar_range;
                        const double bar_low = bar_entry - bar_range;

                        double bar_pd = 0.0;
                        if (tape_in.clamp_lo(row_index, col_index) == 1) {
                              bar_dot -= bar_low;
                        } else {
                              bar_pd -= bar_low;
                        }
                        if (tape_in.clamp_up(row_index, col_index) == 1) {
                              bar_dot -= bar_up;
                        } else {
                              bar_pd += bar_up;
                        }
                        const double pd_b = tape_in.pd_bound(row_index, col_index);
                        const double rem_b = tape_in.rem_before(row_index, col_index);
                        const double inv_2pd = 0.5 / pd_b;
                        bar_rem += bar_pd * d_col * inv_2pd;
                        bar_diag(col_index) += bar_pd * rem_b * inv_2pd;
                  }

                  if (bar_dot != 0.0) {
                        for (int inner_index = 0; inner_index < col_index; inner_index++) {
                              const double d_in = tape_in.diag_vec(inner_index);
                              const double l_ci = tape_in.lower_tri(col_index, inner_index);
                              const double l_ri = tape_in.lower_tri(row_index, inner_index);
                              bar_lower(col_index, inner_index) += bar_dot * d_in * l_ri;
                              bar_lower(row_index, inner_index) += bar_dot * d_in * l_ci;
                              bar_diag(inner_index) += bar_dot * l_ci * l_ri;
                        }
                  }
            }
      }
}

//// -------------------------------------------------------------------------------------
//// Pure-double Cholesky forward and exact reverse (symmetric bar_Omega_out)
//// -------------------------------------------------------------------------------------
inline void clc_cholesky_decompose_forward_double(
      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> Omega_mat,
      Eigen::Ref<Eigen::Matrix<double, -1, -1>> L_out
) {
      const int dim_size = static_cast<int>(Omega_mat.rows());
      L_out.setZero();
      for (int row_index = 0; row_index < dim_size; row_index++) {
            for (int col_index = 0; col_index < row_index; col_index++) {
                  double sum_val = Omega_mat(row_index, col_index);
                  for (int p_idx = 0; p_idx < col_index; p_idx++) {
                        sum_val -= L_out(row_index, p_idx) * L_out(col_index, p_idx);
                  }
                  L_out(row_index, col_index) = sum_val / L_out(col_index, col_index);
            }
            double diag_rem = Omega_mat(row_index, row_index);
            for (int p_idx = 0; p_idx < row_index; p_idx++) {
                  diag_rem -= L_out(row_index, p_idx) * L_out(row_index, p_idx);
            }
            if (!(diag_rem > 0.0)) {
                  throw std::domain_error("CLC-MVOP: correlation matrix is not positive definite.");
            }
            L_out(row_index, row_index) = std::sqrt(diag_rem);
      }
}

inline void clc_cholesky_decompose_reverse_double(
      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> L_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, -1>> bar_L_in,
      Eigen::Ref<Eigen::Matrix<double, -1, -1>> bar_Omega_out,
      CLC_analytic_workspace &analytic_ws
) {
      const int dim_size = static_cast<int>(L_mat.rows());
      Eigen::Ref<Eigen::Matrix<double, -1, -1>> work_L = analytic_ws.cholesky_work_buf.topLeftCorner(
            dim_size,
            dim_size
      );
      work_L = bar_L_in;
      bar_Omega_out.setZero();

      for (int row_index = dim_size - 1; row_index >= 0; row_index--) {
            const double bar_s = work_L(row_index, row_index) / (2.0 * L_mat(row_index, row_index));
            bar_Omega_out(row_index, row_index) += bar_s;
            for (int p_idx = 0; p_idx < row_index; p_idx++) {
                  work_L(row_index, p_idx) -= 2.0 * bar_s * L_mat(row_index, p_idx);
            }
            for (int col_index = row_index - 1; col_index >= 0; col_index--) {
                  const double t_val = work_L(row_index, col_index) / L_mat(col_index, col_index);
                  bar_Omega_out(row_index, col_index) += 0.5 * t_val;
                  bar_Omega_out(col_index, row_index) += 0.5 * t_val;
                  work_L(col_index, col_index) -= t_val * L_mat(row_index, col_index);
                  for (int p_idx = 0; p_idx < col_index; p_idx++) {
                        work_L(row_index, p_idx) -= t_val * L_mat(col_index, p_idx);
                        work_L(col_index, p_idx) -= t_val * L_mat(row_index, p_idx);
                  }
            }
      }
}

} //// namespace clc_mvop_detail
