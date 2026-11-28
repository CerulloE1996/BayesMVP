//// -------------------------------------------------------------------------------------
//// CLC_MVOP_lp_grad_common_T.hpp
////
//// Co-morbidity Latent Class Multivariate Ordinal Probit (CLC-MVOP) for BayesMVP.
////
//// Prepares per-class constrained parameters, priors, and Jacobians before the shared
//// MVOP GHK chunk loop, and chains accumulated likelihood gradients back to theta_main
//// after the chunk loop.
//// -------------------------------------------------------------------------------------

#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <stan/math/rev.hpp>
#include <NicoSIMD/math/nico_double_probability_reference.hpp>
#include <NicoSIMD/stan_external_binned_weighted_mean_Phi_across_cutpoints.hpp>

#include "MVOP_Se_Sp_prior_fns.hpp"
#include "../general_functions/var_fns.hpp"
#include "CLC_MVOP_analytic_helpers_T.hpp"


namespace clc_mvop_detail {

static constexpr double CLC_LOG_2 = 0.693147180559945309417232121458176568;
static constexpr double CLC_HALF_LOG_2PI = 0.918938533204672741780329736405617640;
static constexpr double CLC_INV_SQRT_2PI = 0.398942280401432677939946059934381868;
static constexpr double CLC_INV_SQRT_2 = 0.707106781186547524400844362104849039;

inline double dbl_normal_lpdf(
      const double input_val,
      const double mean_val,
      const double sd_val
) {
      const double std_z = (input_val - mean_val) / sd_val;
      return -CLC_HALF_LOG_2PI - std::log(sd_val) - 0.5 * std_z * std_z;
}

//// -------------------------------------------------------------------------------------
//// Normalised baseline induced-Dirichlet marginal log-density at chosen cutoffs
//// Uses 24-node Laplace-centred Gauss-Hermite quadrature (probit, stan::math::var).
//// -------------------------------------------------------------------------------------
inline stan::math::var clc_log_baseline_marginal_var(
      const Eigen::Matrix<stan::math::var, -1, 1> &x_chosen_var,
      const Eigen::Matrix<double, -1, 1> &alpha_seg,
      const double v0_val
) {
      const int n_chosen = static_cast<int>(x_chosen_var.rows());
      if (n_chosen == 0) {
            return stan::math::var(0.0);
      }
      if (v0_val <= 1e-15) {
            Eigen::Matrix<stan::math::var, -1, 1> cumul_probs_var(n_chosen);
            for (int idx = 0; idx < n_chosen; idx++) {
                  cumul_probs_var(idx) = stan::math::Phi(x_chosen_var(idx));
            }
            const Eigen::Matrix<stan::math::var, -1, 1> p_ord_var = cumul_probs_to_ord_probs_var(
                  cumul_probs_var
            );
            Eigen::Matrix<stan::math::var, -1, 1> alpha_seg_var(alpha_seg.rows());
            for (int idx = 0; idx < alpha_seg.rows(); idx++) {
                  alpha_seg_var(idx) = alpha_seg(idx);
            }
            return induced_dirichlet_given_C_lpdf_var(
                  p_ord_var,
                  x_chosen_var,
                  alpha_seg_var,
                  true
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
            sum_seg += alpha_seg(seg_index);
            sum_lgamma += std::lgamma(alpha_seg(seg_index));
      }
      const double lg_const = std::lgamma(sum_seg) - sum_lgamma;
      const double inv_v0 = 1.0 / v0_val;

      stan::math::var b_star = 0.0;
      double cum_w = 0.0;
      for (int chosen_index = 0; chosen_index < n_chosen; chosen_index++) {
            cum_w += alpha_seg(chosen_index);
            const double p_mid = cum_w / sum_seg;
            const double q_mid = stan::math::inv_Phi(p_mid);
            b_star += (q_mid - x_chosen_var(chosen_index));
      }
      b_star /= static_cast<double>(n_chosen);

      stan::math::var sig_star = 1.0;
      Eigen::Matrix<stan::math::var, -1, 1> z_vec(n_chosen);
      Eigen::Matrix<stan::math::var, -1, 1> log_cdf(n_chosen);
      Eigen::Matrix<stan::math::var, -1, 1> log_ccdf(n_chosen);
      Eigen::Matrix<stan::math::var, -1, 1> log_pdf(n_chosen);
      Eigen::Matrix<stan::math::var, -1, 1> log_q(n_chosen + 1);
      Eigen::Matrix<stan::math::var, -1, 1> ratio_own(n_chosen);
      Eigen::Matrix<stan::math::var, -1, 1> ratio_next(n_chosen);

      for (int newton_iter = 0; newton_iter < 11; newton_iter++) {
            for (int chosen_index = 0; chosen_index < n_chosen; chosen_index++) {
                  z_vec(chosen_index) = x_chosen_var(chosen_index) + b_star;
                  log_cdf(chosen_index) = stan::math::std_normal_lcdf(z_vec(chosen_index));
                  log_ccdf(chosen_index) = stan::math::std_normal_lcdf(-z_vec(chosen_index));
                  log_pdf(chosen_index) = -0.5 * stan::math::square(z_vec(chosen_index)) - CLC_HALF_LOG_2PI;
            }
            log_q(0) = log_cdf(0);
            log_q(n_chosen) = log_ccdf(n_chosen - 1);
            for (int chosen_index = 1; chosen_index < n_chosen; chosen_index++) {
                  if (z_vec(chosen_index - 1).val() >= 0.0) {
                        log_q(chosen_index) = stan::math::log_diff_exp(
                              log_ccdf(chosen_index - 1),
                              log_ccdf(chosen_index)
                        );
                  } else {
                        log_q(chosen_index) = stan::math::log_diff_exp(
                              log_cdf(chosen_index),
                              log_cdf(chosen_index - 1)
                        );
                  }
            }
            for (int chosen_index = 0; chosen_index < n_chosen; chosen_index++) {
                  ratio_own(chosen_index) = stan::math::exp(log_pdf(chosen_index) - log_q(chosen_index));
                  ratio_next(chosen_index) = stan::math::exp(log_pdf(chosen_index) - log_q(chosen_index + 1));
            }
            stan::math::var sum_dg = 0.0;
            stan::math::var sum_d2 = -inv_v0;
            for (int chosen_index = 0; chosen_index < n_chosen; chosen_index++) {
                  sum_dg += (alpha_seg(chosen_index) - 1.0) * ratio_own(chosen_index)
                            - (alpha_seg(chosen_index + 1) - 1.0) * ratio_next(chosen_index)
                            - z_vec(chosen_index);
                  sum_d2 -= 1.0;
            }
            if (alpha_seg(0) != 1.0) {
                  sum_d2 += (alpha_seg(0) - 1.0)
                            * (-z_vec(0) * ratio_own(0) - stan::math::square(ratio_own(0)));
            }
            if (alpha_seg(n_chosen) != 1.0) {
                  sum_d2 += (alpha_seg(n_chosen) - 1.0)
                            * (z_vec(n_chosen - 1) * ratio_next(n_chosen - 1)
                               - stan::math::square(ratio_next(n_chosen - 1)));
            }
            for (int chosen_index = 1; chosen_index < n_chosen; chosen_index++) {
                  if (alpha_seg(chosen_index) != 1.0) {
                        const stan::math::var diff_ratio = ratio_own(chosen_index) - ratio_next(chosen_index - 1);
                        sum_d2 += (alpha_seg(chosen_index) - 1.0)
                                  * (-z_vec(chosen_index) * ratio_own(chosen_index)
                                     + z_vec(chosen_index - 1) * ratio_next(chosen_index - 1)
                                     - stan::math::square(diff_ratio));
                  }
            }
            if (newton_iter == 10) {
                  sig_star = stan::math::inv_sqrt(-sum_d2);
            } else {
                  const stan::math::var dg_db = -b_star * inv_v0 + sum_dg;
                  stan::math::var step_val = -dg_db / sum_d2;
                  if (step_val.val() > 2.0) {
                        step_val = 2.0;
                  } else if (step_val.val() < -2.0) {
                        step_val = -2.0;
                  }
                  b_star += step_val;
            }
      }

      Eigen::Matrix<stan::math::var, -1, 1> lw_terms(24);
      for (int node_index = 0; node_index < 24; node_index++) {
            const stan::math::var b_node = b_star + sig_star * gh_t[node_index];
            stan::math::var sum_log_pdf = 0.0;
            for (int chosen_index = 0; chosen_index < n_chosen; chosen_index++) {
                  z_vec(chosen_index) = x_chosen_var(chosen_index) + b_node;
                  log_cdf(chosen_index) = stan::math::std_normal_lcdf(z_vec(chosen_index));
                  log_ccdf(chosen_index) = stan::math::std_normal_lcdf(-z_vec(chosen_index));
                  log_pdf(chosen_index) = -0.5 * stan::math::square(z_vec(chosen_index)) - CLC_HALF_LOG_2PI;
                  sum_log_pdf += log_pdf(chosen_index);
            }
            log_q(0) = log_cdf(0);
            log_q(n_chosen) = log_ccdf(n_chosen - 1);
            for (int chosen_index = 1; chosen_index < n_chosen; chosen_index++) {
                  if (z_vec(chosen_index - 1).val() >= 0.0) {
                        log_q(chosen_index) = stan::math::log_diff_exp(
                              log_ccdf(chosen_index - 1),
                              log_ccdf(chosen_index)
                        );
                  } else {
                        log_q(chosen_index) = stan::math::log_diff_exp(
                              log_cdf(chosen_index),
                              log_cdf(chosen_index - 1)
                        );
                  }
            }
            stan::math::var val_node = -0.5 * stan::math::square(b_node) * inv_v0 + lg_const + sum_log_pdf;
            for (int seg_index = 0; seg_index <= n_chosen; seg_index++) {
                  if (alpha_seg(seg_index) != 1.0) {
                        val_node += (alpha_seg(seg_index) - 1.0) * log_q(seg_index);
                  }
            }
            lw_terms(node_index) = gh_lw[node_index] + val_node + 0.5 * gh_t[node_index] * gh_t[node_index];
      }
      return -0.5 * std::log(v0_val) + stan::math::log(sig_star) + stan::math::log_sum_exp(lw_terms);
}

//// -------------------------------------------------------------------------------------
//// Tiled AVX-512 / AVX2 population-standardised induced-Dirichlet + Se/Sp prior for one
//// (class, ordinal test) across all cutpoints k = 0 .. n_thr - 1.
//// Computes exact log-density and chains exact analytic pure-double gradients to C_t,
//// Xbeta_std, beta_0, and log_prev_std_c with zero n_std x n_thr heap allocations and
//// zero Stan Math autodiff tape passes.
//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void clc_pop_all_cutpoints_prior_one_class_test_T(
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> C_t_vec,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> Xbeta_std,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> log_weight_raw,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> alpha_ct,
      const double x0_val,
      const double beta_0_val,
      const double prior_beta_0_mean,
      const double prior_beta_0_sd,
      const bool apply_Se_Sp,
      const std::vector<int> &chosen_idx,
      const Eigen::Matrix<double, -1, 1> &alpha_seg,
      const double *d_z_external_chosen,
      double &lp_out,
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> d_C_t_out,
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> d_Xbeta_std_out,
      double &d_beta_0_out,
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> d_log_prev_std_c_out,
      Eigen::Matrix<double, -1, 1> &norm_w_buf,
      Eigen::Matrix<double, -1, 1> &log_wn_buf,
      Eigen::Matrix<double, -1, 1> &d_norm_w_buf,
      nico::math::nico_binned_weighted_mean_Phi_workspace &binned_Phi_ws,
      CLC_analytic_workspace &analytic_ws
) {
      const int n_thr = static_cast<int>(C_t_vec.rows());
      const int n_std = static_cast<int>(Xbeta_std.rows());
      static constexpr int TILE_LEN = CLC_SURVEY_TILE_LEN;

      if (norm_w_buf.rows() != n_std) {
            norm_w_buf.resize(n_std);
            log_wn_buf.resize(n_std);
      }
      if (d_norm_w_buf.rows() != n_std) {
            d_norm_w_buf.resize(n_std);
      }
      //// d_norm_w_buf also holds log(weight) + log_Phi of every profile at a cutpoint in the tail
      Eigen::Matrix<double, -1, 1> &log_w_plus_log_Phi_buf = d_norm_w_buf;

      //// Normalise survey weights for this class
      const double max_lw = log_weight_raw.maxCoeff();
      norm_w_buf.array() = log_weight_raw.array() - max_lw;
      apply_inplace<vec, Fn::exp>(norm_w_buf);
      const double sum_w = norm_w_buf.sum();
      const double log_total_w = max_lw + std::log(sum_w);
      const double inv_sum_w = 1.0 / sum_w;
      norm_w_buf.array() *= inv_sum_w;
      log_wn_buf.array() = log_weight_raw.array() - log_total_w;

      const double *Xbeta_ptr = Xbeta_std.data();
      const double *norm_w_ptr = norm_w_buf.data();
      const double *log_wn_ptr = log_wn_buf.data();

      double *z_pop_ptr = analytic_ws.survey_thr_buf.data() + 0 * analytic_ws.n_slot_len;
      double *log_J_ptr = analytic_ws.survey_thr_buf.data() + 1 * analytic_ws.n_slot_len;
      double *max_exp_ptr = analytic_ws.survey_thr_buf.data() + 2 * analytic_ws.n_slot_len;
      double *inv_sum_scaled_ptr = analytic_ws.survey_thr_buf.data() + 3 * analytic_ws.n_slot_len;
      double *sum_w_exp_ptr = analytic_ws.survey_thr_buf.data() + 4 * analytic_ws.n_slot_len;
      double *p_pop_ptr = analytic_ws.survey_thr_buf.data() + 5 * analytic_ws.n_slot_len;
      double *q_pop_ptr = analytic_ws.survey_thr_buf.data() + 6 * analytic_ws.n_slot_len;
      double *log_p_ptr = analytic_ws.survey_thr_buf.data() + 7 * analytic_ws.n_slot_len;
      double *log_q_ptr = analytic_ws.survey_thr_buf.data() + 8 * analytic_ws.n_slot_len;
      int *tail_mode_ptr = analytic_ws.survey_thr_int_buf.data() + 0 * analytic_ws.n_slot_len;
      int *used_log_shift_ptr = analytic_ws.survey_thr_int_buf.data() + 1 * analytic_ws.n_slot_len;

      for (int thr_index = 0; thr_index < n_thr; thr_index++) {
            p_pop_ptr[thr_index] = 0.0;
            q_pop_ptr[thr_index] = 0.0;
            sum_w_exp_ptr[thr_index] = 0.0;
            max_exp_ptr[thr_index] = -std::numeric_limits<double>::infinity();
            used_log_shift_ptr[thr_index] = 0;
      }

      double *m_tile = analytic_ws.survey_tile_buf.data() + 0 * TILE_LEN;
      double *t_tile = analytic_ws.survey_tile_buf.data() + 1 * TILE_LEN;
      double *a_tile = analytic_ws.survey_tile_buf.data() + 2 * TILE_LEN;
      double *p_tile = analytic_ws.survey_tile_buf.data() + 3 * TILE_LEN;
      double *q_tile = analytic_ws.survey_tile_buf.data() + 4 * TILE_LEN;

      //// Pass 1: use fused NicoSIMD nico_binned_weighted_mean_Phi_across_cutpoints when span fits in bins
      const bool used_binned_Phi = nico::math::nico_binned_weighted_mean_Phi_across_cutpoints_forward_T<vec>(
            C_t_vec,
            Xbeta_std,
            norm_w_buf,
            binned_Phi_ws,
            p_pop_ptr,
            q_pop_ptr,
            sum_w_exp_ptr
      );
      if (!used_binned_Phi) {
            for (int tile_start = 0; tile_start < n_std; tile_start += TILE_LEN) {
                  const int cur_len = ((tile_start + TILE_LEN) <= n_std) ? TILE_LEN : (n_std - tile_start);
                  Eigen::Map<const Eigen::Array<double, -1, 1>> Xbeta_tile_map(
                        Xbeta_ptr + tile_start,
                        cur_len
                  );
                  Eigen::Map<const Eigen::Array<double, -1, 1>> w_tile_map(
                        norm_w_ptr + tile_start,
                        cur_len
                  );
                  Eigen::Map<Eigen::Array<double, -1, 1>> m_tile_map(
                        m_tile,
                        cur_len
                  );
                  Eigen::Map<Eigen::Array<double, -1, 1>> t_tile_map(
                        t_tile,
                        cur_len
                  );
                  Eigen::Map<Eigen::Array<double, -1, 1>> a_tile_map(
                        a_tile,
                        cur_len
                  );
                  Eigen::Map<Eigen::Array<double, -1, 1>> p_tile_map(
                        p_tile,
                        cur_len
                  );
                  Eigen::Map<Eigen::Array<double, -1, 1>> q_tile_map(
                        q_tile,
                        cur_len
                  );
                  for (int thr_index = 0; thr_index < n_thr; thr_index++) {
                        const double C_k = C_t_vec(thr_index);
                        m_tile_map = C_k - Xbeta_tile_map;
                        t_tile_map = -m_tile_map.abs();
                        a_tile_map = -0.5 * m_tile_map.square();
                        apply_raw<vec, Fn::Phi, true>(
                              t_tile,
                              cur_len
                        );
                        apply_raw<vec, Fn::exp, true>(
                              a_tile,
                              cur_len
                        );
                        p_tile_map = (m_tile_map <= 0.0).select(
                              t_tile_map,
                              1.0 - t_tile_map
                        );
                        q_tile_map = (m_tile_map <= 0.0).select(
                              1.0 - t_tile_map,
                              t_tile_map
                        );
                        p_pop_ptr[thr_index] += (w_tile_map * p_tile_map).sum();
                        q_pop_ptr[thr_index] += (w_tile_map * q_tile_map).sum();
                        sum_w_exp_ptr[thr_index] += (w_tile_map * a_tile_map).sum();
                  }
            }
      }

      //// log of the weighted Gaussian sum and inv_Phi of the smaller tail probability of every cutpoint, by the vector kernels
      double *log_sum_w_exp_ptr = analytic_ws.survey_thr_buf.data() + 9 * analytic_ws.n_slot_len;
      double *z_of_smaller_tail_ptr = analytic_ws.survey_thr_buf.data() + 10 * analytic_ws.n_slot_len;
      for (int thr_index = 0; thr_index < n_thr; thr_index++) {
            log_sum_w_exp_ptr[thr_index] = (sum_w_exp_ptr[thr_index] > 1e-250) ? sum_w_exp_ptr[thr_index] : 1.0;
            const double smaller_tail = (p_pop_ptr[thr_index] <= 0.5) ? p_pop_ptr[thr_index] : q_pop_ptr[thr_index];
            z_of_smaller_tail_ptr[thr_index] = (smaller_tail < 1e-12) ? 0.5 : smaller_tail;
      }
      apply_raw<vec, Fn::log, true>(
            log_sum_w_exp_ptr,
            n_thr
      );
      apply_raw<vec, Fn::inv_Phi, true>(
            z_of_smaller_tail_ptr,
            n_thr
      );

      for (int thr_index = 0; thr_index < n_thr; thr_index++) {
            const double C_k = C_t_vec(thr_index);
            const double p_pop = p_pop_ptr[thr_index];
            const double q_pop = q_pop_ptr[thr_index];
            double log_sum_w_phi_unnorm = 0.0;
            if (sum_w_exp_ptr[thr_index] > 1e-250) {
                  used_log_shift_ptr[thr_index] = 0;
                  log_sum_w_phi_unnorm = log_sum_w_exp_ptr[thr_index];
                  inv_sum_scaled_ptr[thr_index] = 1.0 / sum_w_exp_ptr[thr_index];
            } else {
                  used_log_shift_ptr[thr_index] = 1;
                  double max_exp = -std::numeric_limits<double>::infinity();
                  for (int prof_index = 0; prof_index < n_std; prof_index++) {
                        const double m_val = C_k - Xbeta_ptr[prof_index];
                        const double exp_cur = log_wn_ptr[prof_index] - 0.5 * m_val * m_val;
                        if (exp_cur > max_exp) {
                              max_exp = exp_cur;
                        }
                  }
                  double sum_scaled = 0.0;
                  for (int tile_start = 0; tile_start < n_std; tile_start += TILE_LEN) {
                        const int cur_len = ((tile_start + TILE_LEN) <= n_std) ? TILE_LEN : (n_std - tile_start);
                        for (int idx = 0; idx < cur_len; idx++) {
                              const double m_val = C_k - Xbeta_ptr[tile_start + idx];
                              a_tile[idx] = log_wn_ptr[tile_start + idx] - 0.5 * m_val * m_val - max_exp;
                        }
                        apply_raw<vec, Fn::exp, true>(
                              a_tile,
                              cur_len
                        );
                        for (int idx = 0; idx < cur_len; idx++) {
                              sum_scaled += a_tile[idx];
                        }
                  }
                  max_exp_ptr[thr_index] = max_exp;
                  inv_sum_scaled_ptr[thr_index] = 1.0 / sum_scaled;
                  log_sum_w_phi_unnorm = max_exp + std::log(sum_scaled);
            }

            double z_cur = 0.0;
            int tail_mode = 0;
            double log_p_cur = 0.0;
            double log_q_cur = 0.0;
            if (p_pop < 1e-12) {
                  tail_mode = -1;
                  log_w_plus_log_Phi_buf.array() = C_k - Xbeta_std.array();
                  fn_apply_log_Phi_exact_inplace_T<vec>(log_w_plus_log_Phi_buf);
                  log_w_plus_log_Phi_buf.array() += log_wn_buf.array();
                  const double max_lp = log_w_plus_log_Phi_buf.maxCoeff();
                  log_w_plus_log_Phi_buf.array() -= max_lp;
                  apply_inplace<vec, Fn::exp>(log_w_plus_log_Phi_buf);
                  const double sum_ep = log_w_plus_log_Phi_buf.sum();
                  log_p_cur = max_lp + std::log(sum_ep);
                  log_q_cur = std::log1p(-std::exp(log_p_cur));
                  z_cur = nico::math::reference::nico_inv_Phi_from_log_p(
                        log_p_cur,
                        log_q_cur
                  );
            } else if (q_pop < 1e-12) {
                  tail_mode = 1;
                  log_w_plus_log_Phi_buf.array() = Xbeta_std.array() - C_k;
                  fn_apply_log_Phi_exact_inplace_T<vec>(log_w_plus_log_Phi_buf);
                  log_w_plus_log_Phi_buf.array() += log_wn_buf.array();
                  const double max_lq = log_w_plus_log_Phi_buf.maxCoeff();
                  log_w_plus_log_Phi_buf.array() -= max_lq;
                  apply_inplace<vec, Fn::exp>(log_w_plus_log_Phi_buf);
                  const double sum_eq = log_w_plus_log_Phi_buf.sum();
                  log_q_cur = max_lq + std::log(sum_eq);
                  log_p_cur = std::log1p(-std::exp(log_q_cur));
                  z_cur = nico::math::reference::nico_inv_Phi_from_log_p(
                        log_p_cur,
                        log_q_cur
                  );
            } else if (p_pop <= 0.5) {
                  z_cur = z_of_smaller_tail_ptr[thr_index];
            } else {
                  z_cur = -z_of_smaller_tail_ptr[thr_index];
            }

            z_pop_ptr[thr_index] = z_cur;
            log_J_ptr[thr_index] = log_sum_w_phi_unnorm + 0.5 * z_cur * z_cur;
            log_p_ptr[thr_index] = log_p_cur;
            log_q_ptr[thr_index] = log_q_cur;
            tail_mode_ptr[thr_index] = tail_mode;
      }

      //// Pure-double analytic prior density and gradient w.r.t. z_pop_raw and beta_0
      double *C_eff_ptr = analytic_ws.survey_thr_buf.data() + 11 * analytic_ws.n_slot_len;
      double *d_z_ptr = analytic_ws.survey_thr_buf.data() + 12 * analytic_ws.n_slot_len;
      const double x0_shift = x0_val * (beta_0_val - prior_beta_0_mean);
      for (int thr_index = 0; thr_index < n_thr; thr_index++) {
            C_eff_ptr[thr_index] = z_pop_ptr[thr_index] + x0_shift;
            d_z_ptr[thr_index] = 0.0;
      }
      Eigen::Map<Eigen::Matrix<double, -1, 1>> C_eff_map(
            C_eff_ptr,
            n_thr
      );
      Eigen::Map<Eigen::Matrix<double, -1, 1>> d_z_map(
            d_z_ptr,
            n_thr
      );
      double lp_prior_terms = clc_induced_dirichlet_log_scale_val_grad(
            C_eff_map,
            alpha_ct,
            false,
            d_z_map,
            analytic_ws
      );
      const double d_lp_d_beta_0 = x0_val * d_z_map.sum();

      if (apply_Se_Sp && !chosen_idx.empty()) {
            const int n_chosen = static_cast<int>(chosen_idx.size());
            double *z_ch_ptr = analytic_ws.survey_thr_buf.data() + 13 * analytic_ws.n_slot_len;
            double *d_marg_ptr = analytic_ws.survey_thr_buf.data() + 14 * analytic_ws.n_slot_len;
            double *d_sesp_ptr = analytic_ws.survey_thr_buf.data() + 15 * analytic_ws.n_slot_len;
            for (int chosen_pos = 0; chosen_pos < n_chosen; chosen_pos++) {
                  z_ch_ptr[chosen_pos] = z_pop_ptr[chosen_idx[chosen_pos]];
                  d_marg_ptr[chosen_pos] = 0.0;
                  d_sesp_ptr[chosen_pos] = 0.0;
            }
            Eigen::Map<Eigen::Matrix<double, -1, 1>> z_ch_map(
                  z_ch_ptr,
                  n_chosen
            );
            Eigen::Map<Eigen::Matrix<double, -1, 1>> d_marg_map(
                  d_marg_ptr,
                  n_chosen
            );
            const double v_beta_0 = (x0_val * prior_beta_0_sd) * (x0_val * prior_beta_0_sd);
            lp_prior_terms -= clc_log_baseline_marginal_val_grad(
                  z_ch_map,
                  alpha_seg,
                  v_beta_0,
                  d_marg_map,
                  analytic_ws
            );
            //// the prior on the marginal Se / Sp (across the two classes with / without the other condition): its log density
            //// is added once per test outside this kernel, and only its derivative w.r.t. this class's z comes in here
            for (int chosen_pos = 0; chosen_pos < n_chosen; chosen_pos++) {
                  d_sesp_ptr[chosen_pos] = d_z_external_chosen[chosen_pos];
            }
            for (int chosen_pos = 0; chosen_pos < n_chosen; chosen_pos++) {
                  d_z_ptr[chosen_idx[chosen_pos]] += (d_sesp_ptr[chosen_pos] - d_marg_ptr[chosen_pos]);
            }
      }

      double sum_log_J = 0.0;
      for (int thr_index = 0; thr_index < n_thr; thr_index++) {
            sum_log_J += log_J_ptr[thr_index];
      }
      lp_out += lp_prior_terms + sum_log_J;
      d_beta_0_out += d_lp_d_beta_0;

      //// Pass 2 (outer tile in L1 across all normal thresholds): analytic chain rule to C_t, Xbeta_std, log_prev_std_c
      double *d_Xbeta_ptr = d_Xbeta_std_out.data();
      double *d_log_prev_ptr = d_log_prev_std_c_out.data();

      double *grad_z_total_ptr = analytic_ws.survey_thr_buf.data() + 16 * analytic_ws.n_slot_len;
      double *grad_z_exp_lj_ptr = analytic_ws.survey_thr_buf.data() + 17 * analytic_ws.n_slot_len;
      double *grad_z_over_phi_z_ptr = analytic_ws.survey_thr_buf.data() + 18 * analytic_ws.n_slot_len;
      double *d_C_sum_ptr = analytic_ws.survey_thr_buf.data() + 19 * analytic_ws.n_slot_len;

      //// exp(log_J) and exp(0.5 * z^2) of every cutpoint, by the vector kernel
      for (int thr_index = 0; thr_index < n_thr; thr_index++) {
            log_sum_w_exp_ptr[thr_index] = log_J_ptr[thr_index];
            z_of_smaller_tail_ptr[thr_index] = 0.5 * z_pop_ptr[thr_index] * z_pop_ptr[thr_index];
      }
      apply_raw<vec, Fn::exp, true>(
            log_sum_w_exp_ptr,
            n_thr
      );
      apply_raw<vec, Fn::exp, true>(
            z_of_smaller_tail_ptr,
            n_thr
      );
      const double *exp_of_log_J_ptr = log_sum_w_exp_ptr;
      const double *exp_of_half_z_squared_ptr = z_of_smaller_tail_ptr;

      bool has_tail_thr = false;
      bool has_normal_thr = false;
      for (int thr_index = 0; thr_index < n_thr; thr_index++) {
            const double z_cur = z_pop_ptr[thr_index];
            const double grad_z_total = d_z_ptr[thr_index] + z_cur;
            grad_z_total_ptr[thr_index] = grad_z_total;
            grad_z_exp_lj_ptr[thr_index] = grad_z_total * exp_of_log_J_ptr[thr_index];
            d_C_sum_ptr[thr_index] = 0.0;
            if (tail_mode_ptr[thr_index] == 0) {
                  has_normal_thr = true;
                  const double inv_phi_z = exp_of_half_z_squared_ptr[thr_index] / CLC_INV_SQRT_2PI;
                  grad_z_over_phi_z_ptr[thr_index] = grad_z_total * inv_phi_z;
            } else {
                  has_tail_thr = true;
                  grad_z_over_phi_z_ptr[thr_index] = 0.0;
            }
      }

      if (has_normal_thr) {
            bool run_tiled_normal_pass = !used_binned_Phi;
            if (used_binned_Phi) {
                  double *adj_lower_ptr = analytic_ws.survey_thr_buf.data() + 20 * analytic_ws.n_slot_len;
                  double *adj_upper_ptr = analytic_ws.survey_thr_buf.data() + 21 * analytic_ws.n_slot_len;
                  double *adj_gaussian_ptr = analytic_ws.survey_thr_buf.data() + 22 * analytic_ws.n_slot_len;
                  bool any_binned_normal_thr = false;
                  for (int thr_index = 0; thr_index < n_thr; thr_index++) {
                        if ((tail_mode_ptr[thr_index] == 0) && (used_log_shift_ptr[thr_index] == 0)) {
                              any_binned_normal_thr = true;
                              const bool use_upper = (p_pop_ptr[thr_index] > 0.5);
                              adj_lower_ptr[thr_index] = use_upper ? 0.0 : grad_z_over_phi_z_ptr[thr_index];
                              adj_upper_ptr[thr_index] = use_upper ? (-grad_z_over_phi_z_ptr[thr_index]) : 0.0;
                              adj_gaussian_ptr[thr_index] = inv_sum_scaled_ptr[thr_index];
                        } else {
                              adj_lower_ptr[thr_index] = 0.0;
                              adj_upper_ptr[thr_index] = 0.0;
                              adj_gaussian_ptr[thr_index] = 0.0;
                              if ((tail_mode_ptr[thr_index] == 0) && (used_log_shift_ptr[thr_index] != 0)) {
                                    run_tiled_normal_pass = true;
                              }
                        }
                  }
                  if (any_binned_normal_thr) {
                        if (d_norm_w_buf.rows() != n_std) {
                              d_norm_w_buf.resize(n_std);
                        }
                        d_norm_w_buf.setZero();
                        nico::math::nico_binned_weighted_mean_Phi_across_cutpoints_backward_T<vec>(
                              C_t_vec,
                              norm_w_buf,
                              adj_lower_ptr,
                              adj_upper_ptr,
                              adj_gaussian_ptr,
                              binned_Phi_ws,
                              d_C_sum_ptr,
                              d_Xbeta_std_out,
                              d_norm_w_buf
                        );
                        const double weighted_mean_d_norm_w =
                              (norm_w_buf.array() * d_norm_w_buf.array()).sum();
                        d_log_prev_std_c_out.array() +=
                              norm_w_buf.array() * (d_norm_w_buf.array() - weighted_mean_d_norm_w);
                  }
            }
            if (run_tiled_normal_pass) {
                  for (int tile_start = 0; tile_start < n_std; tile_start += TILE_LEN) {
                        const int cur_len = ((tile_start + TILE_LEN) <= n_std) ? TILE_LEN : (n_std - tile_start);
                        Eigen::Map<const Eigen::Array<double, -1, 1>> Xbeta_tile_map(
                              Xbeta_ptr + tile_start,
                              cur_len
                        );
                        Eigen::Map<const Eigen::Array<double, -1, 1>> w_tile_map(
                              norm_w_ptr + tile_start,
                              cur_len
                        );
                        Eigen::Map<const Eigen::Array<double, -1, 1>> lwn_tile_map(
                              log_wn_ptr + tile_start,
                              cur_len
                        );
                        Eigen::Map<Eigen::Array<double, -1, 1>> d_Xb_tile_map(
                              d_Xbeta_ptr + tile_start,
                              cur_len
                        );
                        Eigen::Map<Eigen::Array<double, -1, 1>> d_lp_tile_map(
                              d_log_prev_ptr + tile_start,
                              cur_len
                        );
                        Eigen::Map<Eigen::Array<double, -1, 1>> m_tile_map(
                              m_tile,
                              cur_len
                        );
                        Eigen::Map<Eigen::Array<double, -1, 1>> t_tile_map(
                              t_tile,
                              cur_len
                        );
                        Eigen::Map<Eigen::Array<double, -1, 1>> a_tile_map(
                              a_tile,
                              cur_len
                        );
                        Eigen::Map<Eigen::Array<double, -1, 1>> p_tile_map(
                              p_tile,
                              cur_len
                        );
                        Eigen::Map<Eigen::Array<double, -1, 1>> q_tile_map(
                              q_tile,
                              cur_len
                        );

                        for (int thr_index = 0; thr_index < n_thr; thr_index++) {
                              if (tail_mode_ptr[thr_index] != 0) {
                                    continue;
                              }
                              if (used_binned_Phi && (used_log_shift_ptr[thr_index] == 0)) {
                                    continue;
                              }
                              const double C_k = C_t_vec(thr_index);
                              const double inv_sum_scaled = inv_sum_scaled_ptr[thr_index];
                              const double grad_z_exp_lj = grad_z_exp_lj_ptr[thr_index];
                              const double grad_z_over_phi_z = grad_z_over_phi_z_ptr[thr_index];
                              const double p_pop = p_pop_ptr[thr_index];
                              const double q_pop = q_pop_ptr[thr_index];
                              const bool use_upper = (p_pop > 0.5);
                              const bool use_shift = (used_log_shift_ptr[thr_index] != 0);
                              const double max_exp = max_exp_ptr[thr_index];

                              m_tile_map = C_k - Xbeta_tile_map;
                              t_tile_map = -m_tile_map.abs();
                              apply_raw<vec, Fn::Phi, true>(
                                    t_tile,
                                    cur_len
                              );
                              if (!use_shift) {
                                    a_tile_map = -0.5 * m_tile_map.square();
                                    apply_raw<vec, Fn::exp, true>(
                                          a_tile,
                                          cur_len
                                    );
                                    a_tile_map = w_tile_map * a_tile_map * inv_sum_scaled;
                              } else {
                                    a_tile_map = lwn_tile_map - 0.5 * m_tile_map.square() - max_exp;
                                    apply_raw<vec, Fn::exp, true>(
                                          a_tile,
                                          cur_len
                                    );
                                    a_tile_map *= inv_sum_scaled;
                              }
                              q_tile_map = a_tile_map * (grad_z_exp_lj - m_tile_map);
                              d_C_sum_ptr[thr_index] += q_tile_map.sum();
                              d_Xb_tile_map -= q_tile_map;
                              if (!use_upper) {
                                    p_tile_map = (m_tile_map <= 0.0).select(
                                          t_tile_map,
                                          1.0 - t_tile_map
                                    );
                                    d_lp_tile_map += grad_z_over_phi_z * w_tile_map * (p_tile_map - p_pop)
                                                     + (a_tile_map - w_tile_map);
                              } else {
                                    p_tile_map = (m_tile_map <= 0.0).select(
                                          1.0 - t_tile_map,
                                          t_tile_map
                                    );
                                    d_lp_tile_map += grad_z_over_phi_z * w_tile_map * (q_pop - p_tile_map)
                                                     + (a_tile_map - w_tile_map);
                              }
                        }
                  }
            }
      }

      if (has_tail_thr) {
            for (int thr_index = 0; thr_index < n_thr; thr_index++) {
                  const int tail_mode = tail_mode_ptr[thr_index];
                  if (tail_mode == 0) {
                        continue;
                  }
                  const double C_k = C_t_vec(thr_index);
                  const double z_cur = z_pop_ptr[thr_index];
                  const double max_exp = max_exp_ptr[thr_index];
                  const double inv_sum_scaled = inv_sum_scaled_ptr[thr_index];
                  const double grad_z_total = grad_z_total_ptr[thr_index];
                  const double grad_z_exp_lj = grad_z_exp_lj_ptr[thr_index];
                  const bool use_shift = (used_log_shift_ptr[thr_index] != 0);
                  const double log_inv_phi_z = 0.5 * z_cur * z_cur + CLC_HALF_LOG_2PI;
                  double d_C_sum = 0.0;

                  //// weight * Phi / phi(z) of every profile at this cutpoint, by the vector kernels
                  if (tail_mode == -1) {
                        log_w_plus_log_Phi_buf.array() = C_k - Xbeta_std.array();
                  } else {
                        log_w_plus_log_Phi_buf.array() = Xbeta_std.array() - C_k;
                  }
                  fn_apply_log_Phi_exact_inplace_T<vec>(log_w_plus_log_Phi_buf);
                  log_w_plus_log_Phi_buf.array() += log_wn_buf.array() + log_inv_phi_z;
                  apply_inplace<vec, Fn::exp>(log_w_plus_log_Phi_buf);
                  const double *w_Phi_over_phi_z_ptr = log_w_plus_log_Phi_buf.data();
                  const double log_tail_prob = (tail_mode == -1) ? log_p_ptr[thr_index] : log_q_ptr[thr_index];
                  const double tail_prob_over_phi_z = stan::math::exp(log_tail_prob + log_inv_phi_z);

                  for (int tile_start = 0; tile_start < n_std; tile_start += TILE_LEN) {
                        const int cur_len = ((tile_start + TILE_LEN) <= n_std) ? TILE_LEN : (n_std - tile_start);
                        for (int idx = 0; idx < cur_len; idx++) {
                              const double m_val = C_k - Xbeta_ptr[tile_start + idx];
                              m_tile[idx] = m_val;
                              a_tile[idx] = use_shift
                                    ? (log_wn_ptr[tile_start + idx] - 0.5 * m_val * m_val - max_exp)
                                    : (-0.5 * m_val * m_val);
                        }
                        apply_raw<vec, Fn::exp, true>(
                              a_tile,
                              cur_len
                        );
                        if (tail_mode == -1) {
                              for (int idx = 0; idx < cur_len; idx++) {
                                    const int prof_index = tile_start + idx;
                                    const double m_val = m_tile[idx];
                                    const double w_val = norm_w_ptr[prof_index];
                                    const double a_val = use_shift
                                          ? (a_tile[idx] * inv_sum_scaled)
                                          : (w_val * a_tile[idx] * inv_sum_scaled);
                                    const double d_m = a_val * (grad_z_exp_lj - m_val);
                                    d_C_sum += d_m;
                                    d_Xbeta_ptr[prof_index] -= d_m;
                                    const double d_w = grad_z_total * (
                                          w_Phi_over_phi_z_ptr[prof_index]
                                          - w_val * tail_prob_over_phi_z
                                    );
                                    d_log_prev_ptr[prof_index] += d_w + (a_val - w_val);
                              }
                        } else {
                              for (int idx = 0; idx < cur_len; idx++) {
                                    const int prof_index = tile_start + idx;
                                    const double m_val = m_tile[idx];
                                    const double w_val = norm_w_ptr[prof_index];
                                    const double a_val = use_shift
                                          ? (a_tile[idx] * inv_sum_scaled)
                                          : (w_val * a_tile[idx] * inv_sum_scaled);
                                    const double d_m = a_val * (grad_z_exp_lj - m_val);
                                    d_C_sum += d_m;
                                    d_Xbeta_ptr[prof_index] -= d_m;
                                    const double d_w = grad_z_total * (
                                          w_val * tail_prob_over_phi_z
                                          - w_Phi_over_phi_z_ptr[prof_index]
                                    );
                                    d_log_prev_ptr[prof_index] += d_w + (a_val - w_val);
                              }
                        }
                  }
                  d_C_sum_ptr[thr_index] += d_C_sum;
            }
      }

      for (int thr_index = 0; thr_index < n_thr; thr_index++) {
            d_C_t_out(thr_index) += d_C_sum_ptr[thr_index];
      }
}

} //// namespace clc_mvop_detail


//// -------------------------------------------------------------------------------------
//// CLC_MVOP_prep: state prepared once before the shared MVOP GHK chunk loop
//// -------------------------------------------------------------------------------------
struct CLC_MVOP_prep {
      //// dimensions
      int n_obs;
      int n_tests;
      int n_us;
      int n_params_main;
      int n_params;
      int n_class;
      int n_pops;
      int n_binary_tests;
      int n_ordinal_tests;
      int n_total_C_per_class;
      int n_cutpoints_max;
      int n_covariates_max;
      int n_covariates_total;
      int n_prevalence_covariates;
      int n_standardisation;
      int pool_correlations_across_classes;
      int use_pop_prev_prior;
      int use_Se_Sp_prior;
      //// use_Se_Sp_prior = 1: the user's prior on the probit gap (class with minus class without the other condition),
      //// [0] = Se, [1] = Sp, each [chosen cutoff j, ordinal test] (Model_args_vecs_of_mats_double[12] means, [13] SDs)
      const std::vector<Eigen::Matrix<double, -1, -1>> *gap_link_mean = nullptr;
      const std::vector<Eigen::Matrix<double, -1, -1>> *gap_link_SD = nullptr;
      int use_label_constraint;
      int n_chosen_cutoffs_max;
      bool exclude_priors;
      bool CI;
      bool corr_force_positive;
      bool prior_only;
      bool use_pop_Se_Sp_prior;
      bool has_y_upper;
      bool label_constraint_violated;
      double overflow_threshold;
      double underflow_threshold;
      double C_raw_lower;
      double C_raw_upper;
      double perfect_ref_M;

      //// correlation counts
      Eigen::Matrix<int, -1, 1> n_corrs_per_class;
      Eigen::Matrix<int, -1, 1> corr_start_index;
      int n_corrs_total;
      int n_corrs_global;
      int n_lambda;

      //// unconstrained offsets in theta_main (0-based within theta_main)
      int off_Omega;
      int off_Omega_global;
      int off_lambda_class;
      int off_beta_vec;
      int off_beta_prev_A;
      int off_beta_prev_B;
      int off_z_logit_A;
      int off_z_logit_B;
      int off_mu_logit_A;
      int off_mu_logit_B;
      int off_sigma_logit_A;
      int off_sigma_logit_B;
      int off_log_OR;
      int off_C_unc;

      //// chunking
      int chunk_size_orig;
      int normal_chunk_size;
      int last_chunk_size;
      int n_total_chunks;
      int n_full_chunks;
      KernelChoice kchoice;

      //// pointers into Model_args (valid for the lifetime of the lp_grad call)
      const Eigen::Matrix<int, -1, 1> *n_cat_per_ord_test;
      const Eigen::Matrix<int, -1, 1> *n_thr_per_ord_test;
      const Eigen::Matrix<int, -1, 1> *pop_ind;
      const Eigen::Matrix<int, -1, 1> *test_condition;
      const Eigen::Matrix<int, -1, 1> *is_perfect_ref;
      const Eigen::Matrix<int, -1, 1> *std_index;
      const Eigen::Matrix<int, -1, 1> *pop_std;
      const Eigen::Matrix<int, -1, 1> *reference_test_fitted;
      const Eigen::Matrix<int, -1, 1> *cutoff_category;
      const Eigen::Matrix<int, -1, -1> *n_covs_per_outcome;
      const Eigen::Matrix<int, -1, -1> *class_D;
      const Eigen::Matrix<int, -1, -1> *y_upper_mat;
      const Eigen::Matrix<double, -1, -1> *X_prev;
      const Eigen::Matrix<double, -1, -1> *X_prev_std;
      const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> *X_class_test;
      const Eigen::Matrix<double, -1, 1> *std_w;

      //// which stored covariate matrix each class and each test reads
      bool X_is_the_same_matrix_for_every_class_and_test;
      Eigen::Matrix<int, -1, 1> class_index_of_X_to_read;
      Eigen::Matrix<int, -1, 1> test_index_of_X_to_read;

      //// derived slot and class maps
      Eigen::Matrix<int, -1, 1> ord_idx_of_test;
      Eigen::Matrix<int, -1, 1> fitted_test_of_ord_slot;
      Eigen::Matrix<int, -1, 1> ord_start_index;
      int class_with_neither_condition;
      int class_with_only_condition[2];
      Se_Sp_prior_args_struct Se_Sp_args;

      //// constrained parameters used by the GHK chunk loop
      std::vector<Eigen::Matrix<double, -1, -1>> beta_4;
      std::vector<Eigen::Matrix<double, -1, -1>> C_raw_4;
      std::vector<Eigen::Matrix<double, -1, -1>> C_4;
      std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_double;
      std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_recip_double;
      std::vector<Eigen::Matrix<double, -1, -1>> lb_corr_actual;

      //// prevalence state
      Eigen::Matrix<double, -1, -1> prev_subject;
      Eigen::Matrix<double, -1, -1> log_prev_subject;
      Eigen::Matrix<double, -1, 1> eta_A_all;
      Eigen::Matrix<double, -1, 1> eta_B_all;
      Eigen::Matrix<double, -1, 1> eta_q_all;
      Eigen::Matrix<double, -1, 1> pA_all;
      Eigen::Matrix<double, -1, 1> pB_all;
      Eigen::Matrix<double, -1, 1> q_all;
      Eigen::Matrix<double, -1, 1> one_m_pA_all;
      Eigen::Matrix<double, -1, 1> one_m_pB_all;
      Eigen::Matrix<double, -1, 1> one_m_q_all;
      Eigen::Matrix<double, -1, 1> one_minus_prob_all_buf;
      Eigen::Matrix<double, -1, 1> log_prob_all_buf;
      Eigen::Matrix<double, -1, 1> log_one_minus_prob_all_buf;
      Eigen::Matrix<double, -1, 1> z_A;
      Eigen::Matrix<double, -1, 1> z_B;
      Eigen::Matrix<double, -1, 1> beta_prev_A;
      Eigen::Matrix<double, -1, 1> beta_prev_B;
      double log_OR;
      double mu_A;
      double mu_B;
      double sigma_A;
      double sigma_B;
      //// inv_logit(mu_A), inv_logit(mu_B) of the Pinkney partially-centered population logits (NicoSIMD), the size of
      //// each population, and w and tau_wt of each population (A in 0 .. n_pops - 1, B in n_pops .. 2 n_pops - 1)
      double pcp_p_A;
      double pcp_p_B;
      Eigen::Matrix<double, -1, 1, 0, 50, 1> pcp_n_of_pop;
      Eigen::Matrix<double, -1, 1, 0, 50, 1> pcp_w;
      Eigen::Matrix<double, -1, 1, 0, 50, 1> pcp_tau_wt;
      Eigen::Matrix<double, -1, 1, 0, 50, 1> pcp_log_tau_wt;
      Eigen::Matrix<double, -1, 1, 0, 50, 1> pcp_eta;
      Eigen::Matrix<double, -1, 1, 0, 50, 1> pcp_prob;
      Eigen::Matrix<double, -1, 1, 0, 50, 1> pcp_one_m_prob;
      Eigen::Matrix<double, -1, 1, 0, 50, 1> pcp_log_prob;
      Eigen::Matrix<double, -1, 1, 0, 50, 1> pcp_log_one_m_prob;

      //// accumulated log-prob and prior/Jacobian gradients w.r.t. theta_main
      double log_prob_prior_and_jac;
      Eigen::Matrix<double, -1, 1> grad_main_prior_and_jac;
      std::vector<Eigen::Matrix<double, -1, -1>> cutpoint_grad_4_from_prior;
      std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_4_from_prior;
      Eigen::Matrix<double, -1, -1> d_lp_d_log_prev_std_from_pop_prior;

      //// pure-double correlation tapes and intermediates
      clc_mvop_detail::PinkneyLDLTape corr_global_tape;
      std::vector<clc_mvop_detail::PinkneyLDLTape> corr_class_tapes;
      Eigen::Matrix<double, -1, -1> Omega_global_double;
      std::vector<Eigen::Matrix<double, -1, -1>> Omega_dep_double;
      Eigen::Matrix<double, -1, 1> lambda_val_double;
      Eigen::Matrix<double, -1, 1> lambda_unc_double;
      Eigen::Matrix<double, -1, 1> lambda_one_m_val_buf;
      Eigen::Matrix<double, -1, 1> lambda_log_val_buf;
      Eigen::Matrix<double, -1, 1> lambda_log_one_m_val_buf;
      Eigen::Matrix<double, -1, -1> Omega_class_buf;
      clc_mvop_detail::CLC_analytic_workspace analytic_ws;
      Eigen::Matrix<double, -1, 1> prior_only_u_unc_buf;
      Eigen::Matrix<double, -1, 1> prior_only_u_buf;
      Eigen::Matrix<double, -1, 1> prior_only_du_wrt_duu_buf;
      Eigen::Matrix<double, -1, 1> prior_only_d_J_wrt_duu_buf;
      Eigen::Matrix<double, -1, -1> corr_bar_Omega_global_buf;
      Eigen::Matrix<double, -1, -1> corr_bar_Omega_class_buf;
      Eigen::Matrix<double, -1, -1> corr_bar_L_dep_buf;
      Eigen::Matrix<double, -1, -1> corr_bar_L_Omega_buf;
      Eigen::Matrix<double, -1, -1> corr_product_buf;
      Eigen::Matrix<double, -1, -1> corr_bar_L_global_buf;

      //// content-verified Se/Sp prior args snapshot (never keyed by pointer address)
      bool se_sp_cache_valid = false;
      std::vector<Eigen::Matrix<int, -1, -1>> se_sp_cache_chosen_cutoffs;
      std::vector<Eigen::Matrix<double, -1, -1>> se_sp_cache_A_user;
      std::vector<Eigen::Matrix<double, -1, -1>> se_sp_cache_beta_a;
      std::vector<Eigen::Matrix<double, -1, -1>> se_sp_cache_beta_b;
      std::vector<Eigen::Matrix<double, -1, -1>> se_sp_cache_link_mean;
      std::vector<Eigen::Matrix<double, -1, -1>> se_sp_cache_link_SD;
      std::vector<Eigen::Matrix<double, -1, -1>> se_sp_cache_alpha;
      Eigen::Matrix<int, -1, 1> se_sp_cache_n_thr;
      Eigen::Matrix<int, -1, 1> se_sp_cache_fitted_test;
      Eigen::Matrix<int, -1, 1> se_sp_cache_test_cond;
      Eigen::Matrix<int, -1, -1> se_sp_cache_class_D;

      //// reusable per-step work buffers (avoid per-step heap allocations)
      Eigen::Matrix<double, -1, 1> pA_std_buf;
      Eigen::Matrix<double, -1, 1> one_m_pA_std_buf;
      Eigen::Matrix<double, -1, 1> pB_std_buf;
      Eigen::Matrix<double, -1, 1> one_m_pB_std_buf;
      Eigen::Matrix<double, -1, 1> pQ_std_buf;
      Eigen::Matrix<double, -1, 1> one_m_pQ_std_buf;
      Eigen::Matrix<double, -1, 1> d_eta_A_std_buf;
      Eigen::Matrix<double, -1, 1> d_eta_B_std_buf;
      Eigen::Matrix<double, -1, 1> log_std_w_buf;
      Eigen::Matrix<double, -1, 1> log_weight_c_buf;
      Eigen::Matrix<double, -1, 1> Xbeta_std_buf;
      Eigen::Matrix<double, -1, 1> d_Xbeta_std_buf;
      Eigen::Matrix<double, -1, 1> d_C_t_buf;
      Eigen::Matrix<double, -1, 1> norm_w_buf;
      Eigen::Matrix<double, -1, 1> log_wn_buf;
      Eigen::Matrix<double, -1, 1> Xbeta_all_obs_buf;
      Eigen::Matrix<double, -1, 1> d_Xbeta_all_obs_buf;
      Eigen::Matrix<double, -1, 1> prev_eta_A_chunk_buf;
      Eigen::Matrix<double, -1, 1> prev_eta_B_chunk_buf;
      Eigen::Matrix<double, -1, 1> prev_eta_q_chunk_buf;
      Eigen::Matrix<double, -1, 1> prev_prob_chunk_buf;
      Eigen::Matrix<double, -1, 1> prev_one_m_prob_chunk_buf;
      Eigen::Matrix<double, -1, 1> prev_log_prob_chunk_buf;
      Eigen::Matrix<double, -1, 1> prev_log_one_m_prob_chunk_buf;
      Eigen::Matrix<double, -1, 1> prev_d_eta_A_chunk_buf;
      Eigen::Matrix<double, -1, 1> prev_d_eta_B_chunk_buf;
      Eigen::Matrix<double, -1, -1> d_lp_d_log_prev_total_buf;
      Eigen::Matrix<double, -1, 1> d_eta_A_vec_buf;
      Eigen::Matrix<double, -1, 1> d_eta_B_vec_buf;
      Eigen::Matrix<double, -1, 1> grad_main_buf;
      Eigen::Matrix<double, -1, 1> int_A_buf;
      Eigen::Matrix<double, -1, 1> int_B_buf;
      bool std_index_is_contiguous;
      Eigen::Matrix<double, -1, 1> d_norm_w_buf;
      //// tanh, log1p(exp(-2 * |.|)) and exp(C_raw) of every unconstrained cutpoint, in the order of theta
      Eigen::Matrix<double, -1, 1> tanh_of_C_unc_buf;
      Eigen::Matrix<double, -1, 1> log1p_exp_of_minus_2_abs_C_unc_buf;
      Eigen::Matrix<double, -1, 1> exp_of_C_raw_buf;
      nico::math::nico_binned_weighted_mean_Phi_workspace binned_Phi_ws;

};


//// -------------------------------------------------------------------------------------
//// inv_logit(eta), 1 - inv_logit(eta) and their logs for a whole vector of eta
//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_CLC_MVOP_inv_logit_and_its_logs_T(
      const Eigen::Matrix<double, -1, 1> &eta_vec,
      Eigen::Matrix<double, -1, 1> &prob_vec,
      Eigen::Matrix<double, -1, 1> &one_minus_prob_vec,
      Eigen::Matrix<double, -1, 1> &log_prob_vec,
      Eigen::Matrix<double, -1, 1> &log_one_minus_prob_vec
) {
      //// exp(-|eta|)
      one_minus_prob_vec.array() = -eta_vec.array().abs();
      apply_inplace<vec, Fn::exp>(one_minus_prob_vec);
      //// 1 + exp(-|eta|), then the larger and the smaller of the two probabilities
      log_prob_vec.array() = 1.0 + one_minus_prob_vec.array();
      prob_vec.array() = 1.0 / log_prob_vec.array();
      log_one_minus_prob_vec.array() = one_minus_prob_vec.array() * prob_vec.array();
      one_minus_prob_vec.array() = (eta_vec.array() >= 0.0).select(
            log_one_minus_prob_vec.array(),
            prob_vec.array()
      );
      prob_vec.array() = (eta_vec.array() >= 0.0).select(
            prob_vec.array(),
            log_one_minus_prob_vec.array()
      );
      //// log(1 + exp(-|eta|)), then the two logs
      apply_inplace<vec, Fn::log>(log_prob_vec);
      log_one_minus_prob_vec.array() = (-eta_vec.array()).min(0.0) - log_prob_vec.array();
      log_prob_vec.array() = eta_vec.array().min(0.0) - log_prob_vec.array();
}


//// -------------------------------------------------------------------------------------
//// Chunked per-subject prevalences + population prevalence prior using persistent workspace buffers only
//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_CLC_MVOP_inv_logit_and_its_logs_chunk_T(const double *eta_ptr,
                                                       double *prob_ptr,
                                                       double *one_minus_prob_ptr,
                                                       double *log_prob_ptr,
                                                       double *log_one_minus_prob_ptr,
                                                       const int cur_len) {
      Eigen::Map<const Eigen::Array<double, -1, 1>> eta_map(eta_ptr,
                                                            cur_len);
      Eigen::Map<Eigen::Array<double, -1, 1>> prob_map(prob_ptr,
                                                       cur_len);
      Eigen::Map<Eigen::Array<double, -1, 1>> one_m_map(one_minus_prob_ptr,
                                                        cur_len);
      Eigen::Map<Eigen::Array<double, -1, 1>> log_p_map(log_prob_ptr,
                                                        cur_len);
      Eigen::Map<Eigen::Array<double, -1, 1>> log_1mp_map(log_one_minus_prob_ptr,
                                                          cur_len);
      one_m_map = -eta_map.abs();
      apply_raw<vec, Fn::exp, true>(one_minus_prob_ptr,
                                    cur_len);
      log_p_map = 1.0 + one_m_map;
      prob_map = 1.0 / log_p_map;
      log_1mp_map = one_m_map * prob_map;
      one_m_map = (eta_map >= 0.0).select(log_1mp_map,
                                          prob_map);
      prob_map = (eta_map >= 0.0).select(prob_map,
                                         log_1mp_map);
      apply_raw<vec, Fn::log, true>(log_prob_ptr,
                                    cur_len);
      log_1mp_map = (-eta_map).min(0.0) - log_p_map;
      log_p_map = eta_map.min(0.0) - log_p_map;
}


template <Vec vec>
inline void fn_CLC_MVOP_chunked_prev_and_pop_prior_T(CLC_MVOP_prep &prep,
                                                     const Model_fn_args_struct &Model_args_as_cpp_struct) {
      static constexpr int PREV_CHUNK_ROWS = 256;
      const int n_obs = prep.n_obs;
      const int n_prev_covs = prep.n_prevalence_covariates;
      const int chunk_cap = (n_obs < PREV_CHUNK_ROWS) ? n_obs : PREV_CHUNK_ROWS;
      if (prep.prev_eta_A_chunk_buf.rows() < chunk_cap) {
            prep.prev_eta_A_chunk_buf.resize(chunk_cap);
            prep.prev_eta_B_chunk_buf.resize(chunk_cap);
            prep.prev_eta_q_chunk_buf.resize(chunk_cap);
            prep.prev_prob_chunk_buf.resize(chunk_cap);
            prep.prev_one_m_prob_chunk_buf.resize(chunk_cap);
            prep.prev_log_prob_chunk_buf.resize(chunk_cap);
            prep.prev_log_one_m_prob_chunk_buf.resize(chunk_cap);
            prep.prev_d_eta_A_chunk_buf.resize(chunk_cap);
            prep.prev_d_eta_B_chunk_buf.resize(chunk_cap);
      }
      if (prep.pA_all.rows() != n_obs) {
            prep.pA_all.resize(n_obs);
            prep.pB_all.resize(n_obs);
            prep.q_all.resize(n_obs);
            prep.log_prev_subject.resize(n_obs,
                                         prep.n_class);
            prep.prev_subject.resize(n_obs,
                                     prep.n_class);
      }
      if (prep.int_A_buf.rows() != prep.n_pops) {
            prep.int_A_buf.resize(prep.n_pops);
            prep.int_B_buf.resize(prep.n_pops);
      }
      //// Pinkney partially-centered population logits: log-Jacobian -sum log tau_wt (tau_wt set at the unpack), with
      //// his NicoSIMD log at this vect_type
      prep.pcp_log_tau_wt = prep.pcp_tau_wt;
      apply_raw<vec, Fn::log, true>(prep.pcp_log_tau_wt.data(),
                                    static_cast<int>(prep.pcp_log_tau_wt.rows()));
      prep.log_prob_prior_and_jac -= prep.pcp_log_tau_wt.sum();
      prep.int_A_buf.array() = prep.mu_A + prep.sigma_A * prep.z_A.array();
      prep.int_B_buf.array() = prep.mu_B + prep.sigma_B * prep.z_B.array();

      double *eta_A_ptr = prep.prev_eta_A_chunk_buf.data();
      double *eta_B_ptr = prep.prev_eta_B_chunk_buf.data();
      double *eta_q_ptr = prep.prev_eta_q_chunk_buf.data();
      double *prob_ptr = prep.prev_prob_chunk_buf.data();
      double *one_m_ptr = prep.prev_one_m_prob_chunk_buf.data();
      double *log_p_ptr = prep.prev_log_prob_chunk_buf.data();
      double *log_1mp_ptr = prep.prev_log_one_m_prob_chunk_buf.data();

      for (int row_start = 0; row_start < n_obs; row_start += PREV_CHUNK_ROWS) {
            const int cur_len = ((row_start + PREV_CHUNK_ROWS) <= n_obs) ? PREV_CHUNK_ROWS : (n_obs - row_start);
            for (int offset_idx = 0; offset_idx < cur_len; offset_idx++) {
                  const int pop_index = (*prep.pop_ind)(row_start + offset_idx);
                  eta_A_ptr[offset_idx] = prep.int_A_buf(pop_index);
                  eta_B_ptr[offset_idx] = prep.int_B_buf(pop_index);
            }
            if (n_prev_covs > 0) {
                  Eigen::Map<Eigen::Matrix<double, -1, 1>> eta_A_vec_map(eta_A_ptr,
                                                                         cur_len);
                  Eigen::Map<Eigen::Matrix<double, -1, 1>> eta_B_vec_map(eta_B_ptr,
                                                                         cur_len);
                  eta_A_vec_map.noalias() += prep.X_prev->block(row_start,
                                                                0,
                                                                cur_len,
                                                                n_prev_covs) * prep.beta_prev_A;
                  eta_B_vec_map.noalias() += prep.X_prev->block(row_start,
                                                                0,
                                                                cur_len,
                                                                n_prev_covs) * prep.beta_prev_B;
            }
            for (int offset_idx = 0; offset_idx < cur_len; offset_idx++) {
                  eta_q_ptr[offset_idx] = eta_B_ptr[offset_idx] + prep.log_OR;
            }

            //// condition A: classes 0 and 2 have 1 - Pr(A), classes 1 and 3 have Pr(A)
            fn_CLC_MVOP_inv_logit_and_its_logs_chunk_T<vec>(eta_A_ptr,
                                                            prob_ptr,
                                                            one_m_ptr,
                                                            log_p_ptr,
                                                            log_1mp_ptr,
                                                            cur_len);
            Eigen::Map<const Eigen::Array<double, -1, 1>> prob_map(prob_ptr,
                                                                   cur_len);
            Eigen::Map<const Eigen::Array<double, -1, 1>> one_m_map(one_m_ptr,
                                                                    cur_len);
            Eigen::Map<const Eigen::Array<double, -1, 1>> log_p_map(log_p_ptr,
                                                                    cur_len);
            Eigen::Map<const Eigen::Array<double, -1, 1>> log_1mp_map(log_1mp_ptr,
                                                                      cur_len);
            prep.pA_all.segment(row_start,
                                cur_len).array() = prob_map;
            prep.log_prev_subject.col(0).segment(row_start,
                                                 cur_len).array() = log_1mp_map;
            prep.log_prev_subject.col(1).segment(row_start,
                                                 cur_len).array() = log_p_map;
            prep.log_prev_subject.col(2).segment(row_start,
                                                 cur_len).array() = log_1mp_map;
            prep.log_prev_subject.col(3).segment(row_start,
                                                 cur_len).array() = log_p_map;
            prep.prev_subject.col(0).segment(row_start,
                                             cur_len).array() = one_m_map;
            prep.prev_subject.col(1).segment(row_start,
                                             cur_len).array() = prob_map;
            prep.prev_subject.col(2).segment(row_start,
                                             cur_len).array() = one_m_map;
            prep.prev_subject.col(3).segment(row_start,
                                             cur_len).array() = prob_map;

            //// condition B given not A: classes 0 and 2
            fn_CLC_MVOP_inv_logit_and_its_logs_chunk_T<vec>(eta_B_ptr,
                                                            prob_ptr,
                                                            one_m_ptr,
                                                            log_p_ptr,
                                                            log_1mp_ptr,
                                                            cur_len);
            prep.pB_all.segment(row_start,
                                cur_len).array() = prob_map;
            prep.log_prev_subject.col(0).segment(row_start,
                                                 cur_len).array() += log_1mp_map;
            prep.log_prev_subject.col(2).segment(row_start,
                                                 cur_len).array() += log_p_map;
            prep.prev_subject.col(0).segment(row_start,
                                             cur_len).array() *= one_m_map;
            prep.prev_subject.col(2).segment(row_start,
                                             cur_len).array() *= prob_map;

            //// condition B given A: classes 1 and 3
            fn_CLC_MVOP_inv_logit_and_its_logs_chunk_T<vec>(eta_q_ptr,
                                                            prob_ptr,
                                                            one_m_ptr,
                                                            log_p_ptr,
                                                            log_1mp_ptr,
                                                            cur_len);
            prep.q_all.segment(row_start,
                               cur_len).array() = prob_map;
            prep.log_prev_subject.col(1).segment(row_start,
                                                 cur_len).array() += log_1mp_map;
            prep.log_prev_subject.col(3).segment(row_start,
                                                 cur_len).array() += log_p_map;
            prep.prev_subject.col(1).segment(row_start,
                                             cur_len).array() *= one_m_map;
            prep.prev_subject.col(3).segment(row_start,
                                             cur_len).array() *= prob_map;
      }

      if (!prep.exclude_priors) {
            const int n_std = prep.n_standardisation;
            double sum_w_a = 0.0;
            double sum_w_1ma = 0.0;
            double sum_w_aq = 0.0;
            double sum_w_a1mq = 0.0;
            double sum_w_jac_Q = 0.0;
            double sum_w_jac_A = 0.0;
            double sum_w_1mab = 0.0;
            double sum_w_1ma1mb = 0.0;
            double sum_w_jac_B = 0.0;

            for (int prof_index = 0; prof_index < n_std; prof_index++) {
                  const int subj_idx = (*prep.std_index)(prof_index);
                  const double w_cur = (*prep.std_w)(prof_index);
                  const double pA_val = prep.pA_all(subj_idx);
                  const double one_m_pA = 1.0 - pA_val;
                  const double pB_val = prep.pB_all(subj_idx);
                  const double one_m_pB = 1.0 - pB_val;
                  const double pQ_val = prep.q_all(subj_idx);
                  const double one_m_pQ = 1.0 - pQ_val;

                  const double w_a = w_cur * pA_val;
                  const double w_1ma = w_cur * one_m_pA;
                  sum_w_a += w_a;
                  sum_w_1ma += w_1ma;
                  sum_w_aq += w_a * pQ_val;
                  sum_w_a1mq += w_a * one_m_pQ;
                  sum_w_jac_Q += w_a * pQ_val * one_m_pQ;
                  if (prep.use_pop_prev_prior >= 1) {
                        sum_w_jac_A += w_cur * pA_val * one_m_pA;
                        sum_w_1mab += w_1ma * pB_val;
                        sum_w_1ma1mb += w_1ma * one_m_pB;
                        sum_w_jac_B += w_1ma * pB_val * one_m_pB;
                  }
            }

            //// use_pop_prev_prior: 1 = the second prior is on Pr(B) (population), 2 = it is on Pr(B | not A) (population),
            //// so that the priors on Pr(A), Pr(B | A) and Pr(B | not A) are on variation-independent quantities and
            //// each stated prior is the real (implied) prior.
            //// 3 = as 2, but the third prior (Model_args_doubles 15 / 16) is on Pr(A | B) (population) in place of Pr(B | A):
            ////     Pr(A | B) = S1 / (S1 + S2), S1 = sum_w_aq (both), S2 = sum_w_1mab (B without A); logit Pr(A | B) = log S1 - log S2,
            ////     whose derivative in log_OR is sum_jac_Q / S1 (S2 does not depend on log_OR), so the Jacobian term is
            ////     log_sum_jac_Q - log_sum_aq; the (mu_A, mu_B, log_OR) Jacobian stays triangular as for 2.
            const bool prior_on_B_given_not_A = (prep.use_pop_prev_prior >= 2);
            const bool prior_on_A_given_B = (prep.use_pop_prev_prior == 3);
            const double sum_B = prior_on_B_given_not_A ? sum_w_1mab : (sum_w_1mab + sum_w_aq);
            const double sum_1mB = prior_on_B_given_not_A ? sum_w_1ma1mb : (sum_w_1ma1mb + sum_w_a1mq);
            double log_sums[9] = {
                  sum_w_a,
                  sum_w_aq,
                  sum_w_a1mq,
                  sum_w_jac_Q,
                  (prep.use_pop_prev_prior >= 1) ? sum_w_1ma : 1.0,
                  (prep.use_pop_prev_prior >= 1) ? sum_w_jac_A : 1.0,
                  (prep.use_pop_prev_prior >= 1) ? sum_B : 1.0,
                  (prep.use_pop_prev_prior >= 1) ? sum_1mB : 1.0,
                  (prep.use_pop_prev_prior >= 1) ? sum_w_jac_B : 1.0
            };
            apply_raw<vec, Fn::log, true>(log_sums,
                                          (prep.use_pop_prev_prior >= 1) ? 9 : 4);
            const double log_sum_a = log_sums[0];
            const double log_sum_aq = log_sums[1];
            const double log_sum_a1mq = log_sums[2];
            const double log_sum_jac_Q = log_sums[3];
            //// logit_Q: logit Pr(B | A) (use_pop_prev_prior 0 to 2) or logit Pr(A | B) (use_pop_prev_prior 3); log_sums[6] = log S2 when 3
            const double logit_Q = prior_on_A_given_B ? (log_sum_aq - log_sums[6]) : (log_sum_aq - log_sum_a1mq);
            const double prior_Q_mean = Model_args_as_cpp_struct.Model_args_doubles(15);
            const double prior_Q_sd = Model_args_as_cpp_struct.Model_args_doubles(16);

            prep.log_prob_prior_and_jac += clc_mvop_detail::dbl_normal_lpdf(logit_Q,
                                                                            prior_Q_mean,
                                                                            prior_Q_sd)
                                           + (prior_on_A_given_B ? (log_sum_jac_Q - log_sum_aq)
                                                                 : (- log_sum_aq - log_sum_a1mq + log_sum_a + log_sum_jac_Q));

            const double g_logit_Q = -(logit_Q - prior_Q_mean) / (prior_Q_sd * prior_Q_sd);
            double c_sum_a = prior_on_A_given_B ? 0.0 : (1.0 / sum_w_a);
            double c_sum_1ma = 0.0;
            double c_sum_aq = (g_logit_Q - 1.0) / sum_w_aq;
            double c_sum_a1mq = prior_on_A_given_B ? 0.0 : ((-g_logit_Q - 1.0) / sum_w_a1mq);
            const double c_sum_jac_Q = 1.0 / sum_w_jac_Q;
            double c_sum_jac_A = 0.0;
            double c_sum_1mab = 0.0;
            double c_sum_1ma1mb = 0.0;
            double c_sum_jac_B = 0.0;

            if (prep.use_pop_prev_prior >= 1) {
                  const double log_sum_1ma = log_sums[4];
                  const double log_sum_jac_A = log_sums[5];
                  const double log_sum_B = log_sums[6];
                  const double log_sum_1mB = log_sums[7];
                  const double log_sum_jac_B = log_sums[8];
                  const double logit_A = log_sum_a - log_sum_1ma;
                  const double logit_B = log_sum_B - log_sum_1mB;
                  const double prior_A_mean = Model_args_as_cpp_struct.Model_args_doubles(9);
                  const double prior_A_sd = Model_args_as_cpp_struct.Model_args_doubles(10);
                  const double prior_B_mean = Model_args_as_cpp_struct.Model_args_doubles(11);
                  const double prior_B_sd = Model_args_as_cpp_struct.Model_args_doubles(12);

                  prep.log_prob_prior_and_jac += clc_mvop_detail::dbl_normal_lpdf(logit_A,
                                                                                  prior_A_mean,
                                                                                  prior_A_sd) - log_sum_a - log_sum_1ma + log_sum_jac_A;
                  //// with the prior on Pr(B | not A): d logit(Pr(B | not A)) / d mu_B = sum_jac_B * sum_1ma / (sum_1mab * sum_1ma1mb)
                  prep.log_prob_prior_and_jac += clc_mvop_detail::dbl_normal_lpdf(logit_B,
                                                                                  prior_B_mean,
                                                                                  prior_B_sd) - log_sum_B - log_sum_1mB + log_sum_jac_B
                                                  + (prior_on_B_given_not_A ? log_sum_1ma : 0.0);

                  const double g_logit_A = -(logit_A - prior_A_mean) / (prior_A_sd * prior_A_sd);
                  const double g_logit_B = -(logit_B - prior_B_mean) / (prior_B_sd * prior_B_sd);
                  c_sum_a += (g_logit_A - 1.0) / sum_w_a;
                  c_sum_1ma += (-g_logit_A - 1.0) / sum_w_1ma;
                  c_sum_jac_A = 1.0 / sum_w_jac_A;
                  const double c_sum_B = (g_logit_B - 1.0) / sum_B;
                  const double c_sum_1mB = (-g_logit_B - 1.0) / sum_1mB;
                  c_sum_1mab = c_sum_B;
                  c_sum_1ma1mb = c_sum_1mB;
                  if (prior_on_A_given_B) {
                        c_sum_1mab += -g_logit_Q / sum_w_1mab;   //// d normal(log S1 - log S2) / d S2
                  }
                  if (prior_on_B_given_not_A) {
                        c_sum_1ma += 1.0 / sum_w_1ma;
                  } else {
                        c_sum_aq += c_sum_B;
                        c_sum_a1mq += c_sum_1mB;
                  }
                  c_sum_jac_B = 1.0 / sum_w_jac_B;
            }

            double *d_eta_A_ptr = prep.prev_d_eta_A_chunk_buf.data();
            double *d_eta_B_ptr = prep.prev_d_eta_B_chunk_buf.data();
            double d_log_OR_prior = 0.0;
            double sum_d_eta_A = 0.0;
            double sum_d_eta_B = 0.0;
            for (int prof_start = 0; prof_start < n_std; prof_start += PREV_CHUNK_ROWS) {
                  const int cur_len = ((prof_start + PREV_CHUNK_ROWS) <= n_std) ? PREV_CHUNK_ROWS : (n_std - prof_start);
                  for (int offset_idx = 0; offset_idx < cur_len; offset_idx++) {
                        const int prof_index = prof_start + offset_idx;
                        const int subj_idx = (*prep.std_index)(prof_index);
                        const double w_cur = (*prep.std_w)(prof_index);
                        const double pA_val = prep.pA_all(subj_idx);
                        const double one_m_pA = 1.0 - pA_val;
                        const double pB_val = prep.pB_all(subj_idx);
                        const double one_m_pB = 1.0 - pB_val;
                        const double pQ_val = prep.q_all(subj_idx);
                        const double one_m_pQ = 1.0 - pQ_val;

                        const double vA = pA_val * one_m_pA;
                        const double vB = pB_val * one_m_pB;
                        const double vQ = pQ_val * one_m_pQ;
                        const double d_eta_a = w_cur * vA * (c_sum_a
                                                             - c_sum_1ma
                                                             + c_sum_aq * pQ_val
                                                             + c_sum_a1mq * one_m_pQ
                                                             + c_sum_jac_Q * vQ
                                                             + c_sum_jac_A * (one_m_pA - pA_val)
                                                             - c_sum_1mab * pB_val
                                                             - c_sum_1ma1mb * one_m_pB
                                                             - c_sum_jac_B * vB);
                        const double d_eta_q = w_cur * pA_val * vQ * (c_sum_aq
                                                                      - c_sum_a1mq
                                                                      + c_sum_jac_Q * (one_m_pQ - pQ_val));
                        const double d_eta_b_only = w_cur * one_m_pA * vB * (c_sum_1mab
                                                                             - c_sum_1ma1mb
                                                                             + c_sum_jac_B * (one_m_pB - pB_val));
                        const double d_eta_b_tot = d_eta_b_only + d_eta_q;
                        d_eta_A_ptr[offset_idx] = d_eta_a;
                        d_eta_B_ptr[offset_idx] = d_eta_b_tot;
                        d_log_OR_prior += d_eta_q;
                        sum_d_eta_A += d_eta_a;
                        sum_d_eta_B += d_eta_b_tot;

                        const int pop_index = (*prep.pop_std)(prof_index);
                        prep.grad_main_prior_and_jac(prep.off_z_logit_A + pop_index) += prep.sigma_A * d_eta_a;
                        prep.grad_main_prior_and_jac(prep.off_sigma_logit_A) += prep.sigma_A * prep.z_A(pop_index) * d_eta_a;
                        prep.grad_main_prior_and_jac(prep.off_z_logit_B + pop_index) += prep.sigma_B * d_eta_b_tot;
                        prep.grad_main_prior_and_jac(prep.off_sigma_logit_B) += prep.sigma_B * prep.z_B(pop_index) * d_eta_b_tot;
                  }
                  if (n_prev_covs > 0) {
                        Eigen::Map<const Eigen::Matrix<double, -1, 1>> d_eta_A_map(d_eta_A_ptr,
                                                                                   cur_len);
                        Eigen::Map<const Eigen::Matrix<double, -1, 1>> d_eta_B_map(d_eta_B_ptr,
                                                                                   cur_len);
                        prep.grad_main_prior_and_jac.segment(prep.off_beta_prev_A,
                                                             n_prev_covs).noalias() += prep.X_prev_std->block(prof_start,
                                                                                                              0,
                                                                                                              cur_len,
                                                                                                              n_prev_covs).transpose() * d_eta_A_map;
                        prep.grad_main_prior_and_jac.segment(prep.off_beta_prev_B,
                                                             n_prev_covs).noalias() += prep.X_prev_std->block(prof_start,
                                                                                                              0,
                                                                                                              cur_len,
                                                                                                              n_prev_covs).transpose() * d_eta_B_map;
                  }
            }
            prep.grad_main_prior_and_jac(prep.off_log_OR) += d_log_OR_prior;
            prep.grad_main_prior_and_jac(prep.off_mu_logit_A) += sum_d_eta_A;
            prep.grad_main_prior_and_jac(prep.off_mu_logit_B) += sum_d_eta_B;
      }
}


//// -------------------------------------------------------------------------------------
//// Pure-double correlation block forward (Pinkney LDL, pooling, Cholesky, LKJ, Beta)
//// -------------------------------------------------------------------------------------
inline void fn_CLC_MVOP_corr_forward_double(
      CLC_MVOP_prep &prep,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      const Model_fn_args_struct &model_args,
      const Vec vec_of_vect_type
) {
      const int n_class = prep.n_class;
      const int n_tests = prep.n_tests;
      const std::vector<Eigen::Matrix<double, -1, -1>> &ub_corr = model_args.Model_args_vecs_of_mats_double[3];
      const std::vector<Eigen::Matrix<double, -1, -1>> &known_val = model_args.Model_args_vecs_of_mats_double[4];
      const std::vector<Eigen::Matrix<int, -1, -1>> &known_ind = model_args.Model_args_vecs_of_mats_int[0];
      const Eigen::Matrix<double, -1, 1> &prior_LKJ = model_args.Model_args_col_vecs_double[0];

      if (prep.lambda_val_double.rows() != prep.n_lambda) {
            prep.lambda_val_double.resize(prep.n_lambda);
            prep.lambda_unc_double.resize(prep.n_lambda);
      }
      if (static_cast<int>(prep.corr_class_tapes.size()) != n_class) {
            prep.corr_class_tapes.resize(static_cast<std::size_t>(n_class));
            prep.Omega_dep_double.resize(static_cast<std::size_t>(n_class));
            prep.L_Omega_double.resize(static_cast<std::size_t>(n_class));
            prep.L_Omega_recip_double.resize(static_cast<std::size_t>(n_class));
      }
      if (prep.Omega_global_double.rows() != n_tests) {
            prep.Omega_global_double.resize(
                  n_tests,
                  n_tests
            );
      }

      double lp_corr_jac = 0.0;
      double lp_corr_prior = 0.0;

      if (prep.n_lambda > 0) {
            if (prep.lambda_one_m_val_buf.rows() != prep.n_lambda) {
                  prep.lambda_one_m_val_buf.resize(prep.n_lambda);
                  prep.lambda_log_val_buf.resize(prep.n_lambda);
                  prep.lambda_log_one_m_val_buf.resize(prep.n_lambda);
            }
            prep.lambda_unc_double = theta_main_vec_ref.segment(
                  prep.off_lambda_class,
                  prep.n_lambda
            );
            DISPATCH_VEC(
                  vec_of_vect_type,
                  fn_CLC_MVOP_inv_logit_and_its_logs_T,
                  prep.lambda_unc_double,
                  prep.lambda_val_double,
                  prep.lambda_one_m_val_buf,
                  prep.lambda_log_val_buf,
                  prep.lambda_log_one_m_val_buf
            );
            lp_corr_jac += (prep.lambda_log_val_buf.array() + prep.lambda_log_one_m_val_buf.array()).sum();
      }

      if (prep.pool_correlations_across_classes == 1) {
            clc_mvop_detail::clc_pinkney_LDL_bounds_free_forward_double(
                  theta_main_vec_ref.segment(
                        prep.off_Omega_global,
                        prep.n_corrs_global
                  ),
                  prep.lb_corr_actual[0],
                  ub_corr[0],
                  known_ind[0],
                  known_val[0],
                  prep.corr_global_tape,
                  lp_corr_jac
            );
            prep.Omega_global_double.noalias() =
                  prep.corr_global_tape.L_Omega * prep.corr_global_tape.L_Omega.transpose();
            const double eta_global = model_args.Model_args_doubles(5);
            if (!prep.exclude_priors) {
                  lp_corr_prior += clc_mvop_detail::clc_do_lkj_constant_dbl(
                        eta_global,
                        n_tests
                  );
                  for (int row_index = 1; row_index < n_tests; row_index++) {
                        const double exp_k = static_cast<double>(n_tests - 1 - row_index) + 2.0 * eta_global - 2.0;
                        lp_corr_prior += exp_k * std::log(prep.corr_global_tape.L_Omega(row_index, row_index));
                  }
            }
      }

      const double prior_LKJ_dep = model_args.Model_args_doubles(6);
      const double lambda_alpha = model_args.Model_args_doubles(7);
      const double lambda_beta = model_args.Model_args_doubles(8);
      const double log_beta_lambda = std::lgamma(lambda_alpha)
                                     + std::lgamma(lambda_beta)
                                     - std::lgamma(lambda_alpha + lambda_beta);

      if ((prep.Omega_class_buf.rows() != n_tests) || (prep.Omega_class_buf.cols() != n_tests)) {
            prep.Omega_class_buf.resize(
                  n_tests,
                  n_tests
            );
      }
      Eigen::Matrix<double, -1, -1> &Omega_class = prep.Omega_class_buf;

      for (int class_index = 0; class_index < n_class; class_index++) {
            const int n_corrs_class = prep.n_corrs_per_class(class_index);
            clc_mvop_detail::clc_pinkney_LDL_bounds_free_forward_double(
                  theta_main_vec_ref.segment(
                        prep.off_Omega + prep.corr_start_index(class_index),
                        n_corrs_class
                  ),
                  prep.lb_corr_actual[class_index],
                  ub_corr[class_index],
                  known_ind[class_index],
                  known_val[class_index],
                  prep.corr_class_tapes[class_index],
                  lp_corr_jac
            );

            if ((prep.L_Omega_double[class_index].rows() != n_tests)
                || (prep.L_Omega_double[class_index].cols() != n_tests)) {
                  prep.L_Omega_double[class_index].resize(
                        n_tests,
                        n_tests
                  );
                  prep.L_Omega_recip_double[class_index].resize(
                        n_tests,
                        n_tests
                  );
                  prep.Omega_dep_double[class_index].resize(
                        n_tests,
                        n_tests
                  );
            }

            if (prep.pool_correlations_across_classes == 1) {
                  prep.Omega_dep_double[class_index].noalias() =
                        prep.corr_class_tapes[class_index].L_Omega
                        * prep.corr_class_tapes[class_index].L_Omega.transpose();
                  const double lam_c = prep.lambda_val_double(class_index);
                  Omega_class.noalias() =
                        (1.0 - lam_c) * prep.Omega_global_double
                        + lam_c * prep.Omega_dep_double[class_index];
                  clc_mvop_detail::clc_cholesky_decompose_forward_double(
                        Omega_class,
                        prep.L_Omega_double[class_index]
                  );
                  if (!prep.exclude_priors) {
                        lp_corr_prior += clc_mvop_detail::clc_do_lkj_constant_dbl(
                              prior_LKJ_dep,
                              n_tests
                        );
                        for (int row_index = 1; row_index < n_tests; row_index++) {
                              const double exp_k = static_cast<double>(n_tests - 1 - row_index)
                                                   + 2.0 * prior_LKJ_dep - 2.0;
                              lp_corr_prior += exp_k * std::log(prep.corr_class_tapes[class_index].L_Omega(row_index, row_index));
                        }
                        lp_corr_prior += (lambda_alpha - 1.0) * prep.lambda_log_val_buf(class_index)
                                         + (lambda_beta - 1.0) * prep.lambda_log_one_m_val_buf(class_index)
                                         - log_beta_lambda;
                  }
            } else {
                  prep.L_Omega_double[class_index] = prep.corr_class_tapes[class_index].L_Omega;
                  if (!prep.exclude_priors) {
                        const double eta_c = prior_LKJ(class_index);
                        lp_corr_prior += clc_mvop_detail::clc_do_lkj_constant_dbl(
                              eta_c,
                              n_tests
                        );
                        for (int row_index = 1; row_index < n_tests; row_index++) {
                              const double exp_k = static_cast<double>(n_tests - 1 - row_index) + 2.0 * eta_c - 2.0;
                              lp_corr_prior += exp_k * std::log(prep.L_Omega_double[class_index](row_index, row_index));
                        }
                  }
            }

            prep.L_Omega_recip_double[class_index].setZero();
            for (int test_index = 0; test_index < n_tests; test_index++) {
                  prep.L_Omega_recip_double[class_index](test_index, test_index) =
                        1.0 / prep.L_Omega_double[class_index](test_index, test_index);
            }
      }

      prep.log_prob_prior_and_jac += lp_corr_jac;
      if (!prep.exclude_priors) {
            prep.log_prob_prior_and_jac += lp_corr_prior;
      }
}

//// -------------------------------------------------------------------------------------
//// Pure-double correlation block reverse combining prior, Jacobian, and likelihood VJP
//// -------------------------------------------------------------------------------------
inline void fn_CLC_MVOP_corr_backward_double(
      CLC_MVOP_prep &prep,
      const Model_fn_args_struct &model_args,
      const std::vector<Eigen::Matrix<double, -1, -1>> &U_Omega_grad_array,
      const bool include_lik_vjp,
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> grad_main_out
) {
      const int n_class = prep.n_class;
      const int n_tests = prep.n_tests;
      const std::vector<Eigen::Matrix<int, -1, -1>> &known_ind = model_args.Model_args_vecs_of_mats_int[0];
      const Eigen::Matrix<double, -1, 1> &prior_LKJ = model_args.Model_args_col_vecs_double[0];
      const double eta_global = model_args.Model_args_doubles(5);
      const double prior_LKJ_dep = model_args.Model_args_doubles(6);
      const double lambda_alpha = model_args.Model_args_doubles(7);
      const double lambda_beta = model_args.Model_args_doubles(8);

      if ((prep.corr_bar_Omega_global_buf.rows() != n_tests) || (prep.corr_bar_Omega_global_buf.cols() != n_tests)) {
            prep.corr_bar_Omega_global_buf.resize(
                  n_tests,
                  n_tests
            );
            prep.corr_bar_Omega_class_buf.resize(
                  n_tests,
                  n_tests
            );
            prep.corr_bar_L_dep_buf.resize(
                  n_tests,
                  n_tests
            );
            prep.corr_bar_L_Omega_buf.resize(
                  n_tests,
                  n_tests
            );
            prep.corr_product_buf.resize(
                  n_tests,
                  n_tests
            );
            prep.corr_bar_L_global_buf.resize(
                  n_tests,
                  n_tests
            );
      }
      Eigen::Matrix<double, -1, -1> &bar_Omega_global = prep.corr_bar_Omega_global_buf;
      Eigen::Matrix<double, -1, -1> &bar_Omega_class = prep.corr_bar_Omega_class_buf;
      Eigen::Matrix<double, -1, -1> &bar_L_dep = prep.corr_bar_L_dep_buf;
      Eigen::Matrix<double, -1, -1> &bar_L_Omega = prep.corr_bar_L_Omega_buf;
      Eigen::Matrix<double, -1, -1> &product_buf = prep.corr_product_buf;
      bar_Omega_global.setZero();

      for (int class_index = 0; class_index < n_class; class_index++) {
            bar_L_dep.setZero();
            if (prep.pool_correlations_across_classes == 1) {
                  const double lam_c = prep.lambda_val_double(class_index);
                  double bar_lam_c = 0.0;
                  if (include_lik_vjp) {
                        bar_L_Omega.setZero();
                        bool any_nz = false;
                        for (int row_index = 0; row_index < n_tests; row_index++) {
                              for (int col_index = 0; col_index <= row_index; col_index++) {
                                    const double g_val = U_Omega_grad_array[class_index](row_index, col_index);
                                    if (g_val != 0.0) {
                                          bar_L_Omega(row_index, col_index) = g_val;
                                          any_nz = true;
                                    }
                              }
                        }
                        if (any_nz) {
                              clc_mvop_detail::clc_cholesky_decompose_reverse_double(
                                    prep.L_Omega_double[class_index],
                                    bar_L_Omega,
                                    bar_Omega_class,
                                    prep.analytic_ws
                              );
                              bar_lam_c += (
                                    bar_Omega_class.array()
                                    * (prep.Omega_dep_double[class_index].array() - prep.Omega_global_double.array())
                              ).sum();
                              bar_Omega_global.noalias() += (1.0 - lam_c) * bar_Omega_class;
                              product_buf.noalias() = bar_Omega_class * prep.corr_class_tapes[class_index].L_Omega;
                              product_buf *= (2.0 * lam_c);
                              for (int row_index = 0; row_index < n_tests; row_index++) {
                                    for (int col_index = 0; col_index <= row_index; col_index++) {
                                          bar_L_dep(row_index, col_index) += product_buf(row_index, col_index);
                                    }
                              }
                        }
                  }
                  if (!prep.exclude_priors) {
                        for (int row_index = 1; row_index < n_tests; row_index++) {
                              const double exp_k = static_cast<double>(n_tests - 1 - row_index)
                                                   + 2.0 * prior_LKJ_dep - 2.0;
                              bar_L_dep(row_index, row_index) +=
                                    exp_k / prep.corr_class_tapes[class_index].L_Omega(row_index, row_index);
                        }
                  }
                  const double d_lam_du = lam_c * (1.0 - lam_c);
                  double bar_u_lam = (1.0 - 2.0 * lam_c) + bar_lam_c * d_lam_du;
                  if (!prep.exclude_priors) {
                        bar_u_lam += (lambda_alpha - 1.0) * (1.0 - lam_c) - (lambda_beta - 1.0) * lam_c;
                  }
                  grad_main_out(prep.off_lambda_class + class_index) += bar_u_lam;
            } else {
                  if (include_lik_vjp) {
                        for (int row_index = 0; row_index < n_tests; row_index++) {
                              for (int col_index = 0; col_index <= row_index; col_index++) {
                                    bar_L_dep(row_index, col_index) += U_Omega_grad_array[class_index](row_index, col_index);
                              }
                        }
                  }
                  if (!prep.exclude_priors) {
                        const double eta_c = prior_LKJ(class_index);
                        for (int row_index = 1; row_index < n_tests; row_index++) {
                              const double exp_k = static_cast<double>(n_tests - 1 - row_index) + 2.0 * eta_c - 2.0;
                              bar_L_dep(row_index, row_index) +=
                                    exp_k / prep.L_Omega_double[class_index](row_index, row_index);
                        }
                  }
            }

            const int n_corrs_class = prep.n_corrs_per_class(class_index);
            if (n_corrs_class > 0) {
                  clc_mvop_detail::clc_pinkney_LDL_bounds_free_reverse_double(
                        prep.corr_class_tapes[class_index],
                        known_ind[class_index],
                        bar_L_dep,
                        1.0,
                        grad_main_out.segment(
                              prep.off_Omega + prep.corr_start_index(class_index),
                              n_corrs_class
                        ),
                        prep.analytic_ws
                  );
            }
      }

      if (prep.pool_correlations_across_classes == 1) {
            Eigen::Matrix<double, -1, -1> &bar_L_global = prep.corr_bar_L_global_buf;
            bar_L_global.setZero();
            product_buf.noalias() = bar_Omega_global * prep.corr_global_tape.L_Omega;
            product_buf *= 2.0;
            for (int row_index = 0; row_index < n_tests; row_index++) {
                  for (int col_index = 0; col_index <= row_index; col_index++) {
                        bar_L_global(row_index, col_index) = product_buf(row_index, col_index);
                  }
            }
            if (!prep.exclude_priors) {
                  for (int row_index = 1; row_index < n_tests; row_index++) {
                        const double exp_k = static_cast<double>(n_tests - 1 - row_index) + 2.0 * eta_global - 2.0;
                        bar_L_global(row_index, row_index) += exp_k / prep.corr_global_tape.L_Omega(row_index, row_index);
                  }
            }
            if (prep.n_corrs_global > 0) {
                  clc_mvop_detail::clc_pinkney_LDL_bounds_free_reverse_double(
                        prep.corr_global_tape,
                        known_ind[0],
                        bar_L_global,
                        1.0,
                        grad_main_out.segment(
                              prep.off_Omega_global,
                              prep.n_corrs_global
                        ),
                        prep.analytic_ws
                  );
            }
      }
}


//// -------------------------------------------------------------------------------------
//// Prepare CLC-MVOP parameters, priors, and Jacobians into a reusable prep struct
//// -------------------------------------------------------------------------------------
//// -------------------------------------------------------------------------------------
//// tanh(C_unc), log1p(exp(-2 * |C_unc|)) and exp(C_raw) of every cutpoint, by the vector kernels
//// -------------------------------------------------------------------------------------
template <Vec vec>
inline void fn_CLC_MVOP_tanh_log1p_exp_and_exp_of_the_cutpoints_T(
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> C_unc_vec,
      const double C_raw_lower,
      const double C_range,
      Eigen::Matrix<double, -1, 1> &tanh_of_C_unc_buf,
      Eigen::Matrix<double, -1, 1> &log1p_exp_of_minus_2_abs_C_unc_buf,
      Eigen::Matrix<double, -1, 1> &exp_of_C_raw_buf
) {
      const int n_cutpoints = static_cast<int>(C_unc_vec.rows());
      if (tanh_of_C_unc_buf.rows() != n_cutpoints) {
            tanh_of_C_unc_buf.resize(n_cutpoints);
            log1p_exp_of_minus_2_abs_C_unc_buf.resize(n_cutpoints);
            exp_of_C_raw_buf.resize(n_cutpoints);
      }
      tanh_of_C_unc_buf = C_unc_vec;
      apply_inplace<vec, Fn::tanh>(tanh_of_C_unc_buf);
      log1p_exp_of_minus_2_abs_C_unc_buf.array() = -2.0 * C_unc_vec.array().abs();
      apply_inplace<vec, Fn::log1p_exp>(log1p_exp_of_minus_2_abs_C_unc_buf);
      exp_of_C_raw_buf.array() = C_raw_lower + C_range * 0.5 * (1.0 + tanh_of_C_unc_buf.array());
      apply_inplace<vec, Fn::exp>(exp_of_C_raw_buf);
}




//// -------------------------------------------------------------------------------------
//// log survey weight of every standardisation profile in one class: log(w_i) + log Pr(class | profile i), into prep.log_weight_c_buf
//// -------------------------------------------------------------------------------------
inline void fn_CLC_MVOP_survey_log_weight_one_class(CLC_MVOP_prep &prep,
                                                    const int class_index,
                                                    const int n_std) {
      if (prep.std_index_is_contiguous) {
            prep.log_weight_c_buf.array() = prep.log_std_w_buf.array()
                                            + prep.log_prev_subject.col(class_index).head(n_std).array();
      } else {
            const double *log_prev_col = prep.log_prev_subject.col(class_index).data();
            const double *log_sw_ptr = prep.log_std_w_buf.data();
            double *log_wc_ptr = prep.log_weight_c_buf.data();
            const int *std_idx_ptr = prep.std_index->data();
            for (int prof_index = 0; prof_index < n_std; prof_index++) {
                  log_wc_ptr[prof_index] = log_sw_ptr[prof_index] + log_prev_col[std_idx_ptr[prof_index]];
            }
      }
}

//// -------------------------------------------------------------------------------------
//// linear predictor of every standardisation profile for one (class, test), into prep.Xbeta_std_buf
//// -------------------------------------------------------------------------------------
inline void fn_CLC_MVOP_survey_Xbeta_one_class_test(CLC_MVOP_prep &prep,
                                                    const Eigen::Matrix<double, -1, -1> &X_ct,
                                                    const int class_index,
                                                    const int test_index,
                                                    const int n_covs_ct,
                                                    const int n_std) {
      if (n_covs_ct > 0) {
            if (prep.std_index_is_contiguous) {
                  prep.Xbeta_std_buf.noalias() = X_ct.topLeftCorner(n_std,
                                                                    n_covs_ct) * prep.beta_4[class_index].col(test_index).head(n_covs_ct);
            } else {
                  //// linear predictor of every subject from the shared design matrix, then the survey rows
                  prep.Xbeta_all_obs_buf.noalias() = X_ct.topLeftCorner(prep.n_obs,
                                                                        n_covs_ct) * prep.beta_4[class_index].col(test_index).head(n_covs_ct);
                  const double *Xbeta_all_obs_ptr = prep.Xbeta_all_obs_buf.data();
                  const int *std_index_ptr = prep.std_index->data();
                  double *Xbeta_std_ptr = prep.Xbeta_std_buf.data();
                  for (int prof_index = 0; prof_index < n_std; prof_index++) {
                        Xbeta_std_ptr[prof_index] = Xbeta_all_obs_ptr[std_index_ptr[prof_index]];
                  }
            }
      } else {
            prep.Xbeta_std_buf.setZero();
      }
}

//// -------------------------------------------------------------------------------------
//// The Se / Sp prior (use_Se_Sp_prior = 1), the MetaOrdDTA way, on the MARGINAL Se / Sp of ONE ordinal test at each chosen
//// cutoff: always Jensen-adjusted (population-averaged over the covariates) and always with the gap prior. For Se and for Sp,
//// a = the class without and b = the class with the other condition:
////   y_c = probit of the population-averaged Se (or Sp) of class c, pi = W_b / (W_a + W_b), W_c = sum_i w_i Pr(class c | profile i),
////   M = inv_Phi((1 - pi) Phi(y_a) + pi Phi(y_b)),   D = y_b - y_a ~ N(mean_D, sd_D),
////   log |d(M, D) / d(y_a, y_b)| = log((1 - pi) phi(y_a) + pi phi(y_b)) - log phi(M).
//// At any number of rising chosen cutoffs: M ~ N(mean_M, sd_M), from the usual link slot of class a.
//// The prior of D is its own data input, prior_Se_Sp_gap_link_mean_SD (prep.gap_link_mean / gap_link_SD, [Se / Sp](j, ordinal test)),
//// on the probit of Pr(positive): with minus without the other condition, so a positive gap means more positives in the class
//// with it (for Sp: gap = -(D), as Pr(positive) = 1 - Sp). Each class's z is the z of clc_pop_all_cutpoints_prior_one_class_test_T
//// (inv_Phi of the population-averaged Pr(score below the cutoff)), so y_c = s z_c with s = -1 for Se (classes with the test's own
//// condition) and +1 for Sp; the stored link means carry the same s.
//// Out: the log density (into prep.log_prob_prior_and_jac); its derivative w.r.t. every class's z at the chosen cutoffs, in the
//// workspace's Se_Sp_buf at class * n_slot_len + j; and w.r.t. log W of every class (d_lp_d_log_W, added to).
//// Every class's z and every per-cutoff value: the workspace's chosen_buf, slot k at k * n_slot_len + j (slots 0 to 15).
//// prep.d_Xbeta_std_buf (n_std, zeroed again before each per-class kernel) is used as the scratch vector of Phi.
//// -------------------------------------------------------------------------------------
inline void fn_CLC_MVOP_marginal_Se_Sp_prior(CLC_MVOP_prep &prep,
                                             const int ord_index,
                                             double *d_lp_d_log_W) {
      const int n_std = prep.n_standardisation;
      const int n_slot = prep.analytic_ws.n_slot_len;
      if ((prep.n_class != 4) || (prep.n_ordinal_tests > 4)) {
            throw std::invalid_argument("CLC-MVOP: the marginal Se / Sp prior needs 4 classes and at most 4 ordinal tests.");
      }
      double *z_marg = prep.analytic_ws.chosen_buf.data();
      double *z_tail = z_marg + 4 * n_slot;
      double *M_ptr = z_marg + 5 * n_slot;
      double *g_M_ptr = z_marg + 6 * n_slot;
      double *s_M_ptr = z_marg + 7 * n_slot;
      double *phi_a_ptr = z_marg + 8 * n_slot;
      double *phi_b_ptr = z_marg + 9 * n_slot;
      double *d_Phi_ptr = z_marg + 10 * n_slot;
      double *y_ab_ptr = z_marg + 11 * n_slot;
      double *log_sd_M_ptr = z_marg + 12 * n_slot;
      double *log_sd_D_ptr = z_marg + 13 * n_slot;
      double *log_Q_ptr = z_marg + 14 * n_slot;
      double *phi_M_ptr = z_marg + 15 * n_slot;
      double *d_z_marg_ext = prep.analytic_ws.Se_Sp_buf.data();
      for (int idx = 0; idx < 4 * n_slot; idx++) {
            z_marg[idx] = 0.0;
            d_z_marg_ext[idx] = 0.0;
      }
      const int test_index = prep.fitted_test_of_ord_slot(ord_index);
      const int n_thr_test = (*prep.n_thr_per_ord_test)(ord_index);
      double log_W[4];
      for (int class_index = 0; class_index < 4; class_index++) {
            fn_CLC_MVOP_survey_log_weight_one_class(prep,
                                                    class_index,
                                                    n_std);
            const double max_lw = prep.log_weight_c_buf.head(n_std).maxCoeff();
            prep.norm_w_buf.array() = prep.log_weight_c_buf.head(n_std).array() - max_lw;
            apply_inplace<clc_mvop_detail::CLC_ANALYTIC_VEC, Fn::exp>(prep.norm_w_buf);
            double sum_w_and_log[1] = { prep.norm_w_buf.sum() };
            prep.norm_w_buf.array() *= (1.0 / sum_w_and_log[0]);
            apply_raw<clc_mvop_detail::CLC_ANALYTIC_VEC, Fn::log, true>(sum_w_and_log,
                                                       1);
            log_W[class_index] = max_lw + sum_w_and_log[0];
            const int n_chosen = prep.Se_Sp_args.n_chosen[class_index](ord_index);
            if (n_chosen == 0) {
                  continue;
            }
            const int n_covs_ct = (*prep.n_covs_per_outcome)(class_index, test_index);
            const Eigen::Matrix<double, -1, -1> &X_ct =
                  (*prep.X_class_test)[prep.class_index_of_X_to_read(class_index)][prep.test_index_of_X_to_read(test_index)];
            fn_CLC_MVOP_survey_Xbeta_one_class_test(prep,
                                                    X_ct,
                                                    class_index,
                                                    test_index,
                                                    n_covs_ct,
                                                    n_std);
            //// the population-averaged Pr(score below each cutoff), the same way as the per-class kernel (binned when it fits)
            double *p_pop_ptr = prep.analytic_ws.survey_thr_buf.data() + 5 * n_slot;
            double *q_pop_ptr = prep.analytic_ws.survey_thr_buf.data() + 6 * n_slot;
            double *sum_w_exp_ptr = prep.analytic_ws.survey_thr_buf.data() + 4 * n_slot;
            for (int thr_index = 0; thr_index < n_thr_test; thr_index++) {
                  p_pop_ptr[thr_index] = 0.0;
                  q_pop_ptr[thr_index] = 0.0;
                  sum_w_exp_ptr[thr_index] = 0.0;
            }
            const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> C_t_ref = prep.C_4[class_index].col(ord_index).head(n_thr_test);
            const bool used_binned_Phi = nico::math::nico_binned_weighted_mean_Phi_across_cutpoints_forward_T<clc_mvop_detail::CLC_ANALYTIC_VEC>(C_t_ref,
                                                                                                                                 prep.Xbeta_std_buf,
                                                                                                                                 prep.norm_w_buf,
                                                                                                                                 prep.binned_Phi_ws,
                                                                                                                                 p_pop_ptr,
                                                                                                                                 q_pop_ptr,
                                                                                                                                 sum_w_exp_ptr);
            for (int j = 0; j < n_chosen; j++) {
                  const int thr_index = prep.Se_Sp_args.chosen[class_index][ord_index][j];
                  if (!used_binned_Phi) {
                        prep.d_Xbeta_std_buf.array() = prep.C_4[class_index](thr_index, ord_index) - prep.Xbeta_std_buf.array();
                        apply_inplace<clc_mvop_detail::CLC_ANALYTIC_VEC, Fn::Phi>(prep.d_Xbeta_std_buf);
                        p_pop_ptr[thr_index] = prep.norm_w_buf.dot(prep.d_Xbeta_std_buf);
                        q_pop_ptr[thr_index] = 1.0 - p_pop_ptr[thr_index];
                  }
                  z_tail[j] = (p_pop_ptr[thr_index] <= 0.5) ? p_pop_ptr[thr_index] : q_pop_ptr[thr_index];
            }
            apply_raw<clc_mvop_detail::CLC_ANALYTIC_VEC, Fn::inv_Phi, true>(z_tail,
                                                           n_chosen);
            for (int j = 0; j < n_chosen; j++) {
                  const int thr_index = prep.Se_Sp_args.chosen[class_index][ord_index][j];
                  z_marg[class_index * n_slot + j] = (p_pop_ptr[thr_index] <= 0.5) ? z_tail[j] : -z_tail[j];
            }
      }

      double lp_marg = 0.0;
      const int cond_index = (*prep.test_condition)(test_index);
      const int other_index = 1 - cond_index;
      for (int acc_index = 0; acc_index < 2; acc_index++) {
            //// acc_index 0 = Se (the classes with the test's own condition), 1 = Sp (the classes without it)
            const int own_val = (acc_index == 0) ? 1 : 0;
            int class_a = -1;
            int class_b = -1;
            for (int class_index = 0; class_index < 4; class_index++) {
                  if ((*prep.class_D)(class_index, cond_index) == own_val) {
                        if ((*prep.class_D)(class_index, other_index) == 0) {
                              class_a = class_index;
                        } else {
                              class_b = class_index;
                        }
                  }
            }
            const int n_chosen = prep.Se_Sp_args.n_chosen[class_a](ord_index);
            if ((n_chosen == 0) || (prep.Se_Sp_args.n_chosen[class_b](ord_index) != n_chosen)) {
                  continue;
            }
            const double s_val = (own_val == 1) ? -1.0 : 1.0;
            double pi_val = 0.0;
            //// pass 1: the marginal M at every chosen cutoff, and the values of the Jacobian
            for (int j = 0; j < n_chosen; j++) {
                  const double y_a = s_val * z_marg[class_a * n_slot + j];
                  const double y_b = s_val * z_marg[class_b * n_slot + j];
                  double Phi_vals[4] = { y_a, y_b, -y_a, -y_b };
                  apply_raw<clc_mvop_detail::CLC_ANALYTIC_VEC, Fn::Phi, true>(Phi_vals,
                                                             4);
                  double exp_vals[3] = { log_W[class_a] - log_W[class_b], -0.5 * y_a * y_a, -0.5 * y_b * y_b };
                  apply_raw<clc_mvop_detail::CLC_ANALYTIC_VEC, Fn::exp, true>(exp_vals,
                                                             3);
                  pi_val = 1.0 / (1.0 + exp_vals[0]);
                  phi_a_ptr[j] = exp_vals[1] * clc_mvop_detail::CLC_INV_SQRT_2PI;
                  phi_b_ptr[j] = exp_vals[2] * clc_mvop_detail::CLC_INV_SQRT_2PI;
                  d_Phi_ptr[j] = Phi_vals[1] - Phi_vals[0];
                  y_ab_ptr[j] = y_b - y_a;
                  const double P_val = (1.0 - pi_val) * Phi_vals[0] + pi_val * Phi_vals[1];
                  const double one_m_P = (1.0 - pi_val) * Phi_vals[2] + pi_val * Phi_vals[3];
                  double M_tail[1] = { (P_val <= 0.5) ? P_val : one_m_P };
                  apply_raw<clc_mvop_detail::CLC_ANALYTIC_VEC, Fn::inv_Phi, true>(M_tail,
                                                                 1);
                  M_ptr[j] = (P_val <= 0.5) ? M_tail[0] : -M_tail[0];
                  s_M_ptr[j] = s_val * M_ptr[j];
                  g_M_ptr[j] = 0.0;
                  log_sd_M_ptr[j] = prep.Se_Sp_args.link_mean_SD[class_a][ord_index](j, 1);
                  log_sd_D_ptr[j] = (*prep.gap_link_SD)[acc_index](j, ord_index);
                  log_Q_ptr[j] = (1.0 - pi_val) * phi_a_ptr[j] + pi_val * phi_b_ptr[j];
                  phi_M_ptr[j] = -0.5 * M_ptr[j] * M_ptr[j];
            }
            apply_raw<clc_mvop_detail::CLC_ANALYTIC_VEC, Fn::log, true>(log_sd_M_ptr,
                                                       n_chosen);
            apply_raw<clc_mvop_detail::CLC_ANALYTIC_VEC, Fn::log, true>(log_sd_D_ptr,
                                                       n_chosen);
            apply_raw<clc_mvop_detail::CLC_ANALYTIC_VEC, Fn::log, true>(log_Q_ptr,
                                                       n_chosen);
            apply_raw<clc_mvop_detail::CLC_ANALYTIC_VEC, Fn::exp, true>(phi_M_ptr,
                                                       n_chosen);
            //// the normal prior on the probit of the marginal Se / Sp, and its derivative g_M = d log p(M) / d M
            for (int j = 0; j < n_chosen; j++) {
                  const double mean_M = s_val * prep.Se_Sp_args.link_mean_SD[class_a][ord_index](j, 0);
                  const double sd_M = prep.Se_Sp_args.link_mean_SD[class_a][ord_index](j, 1);
                  const double u_M = (M_ptr[j] - mean_M) / sd_M;
                  lp_marg += -0.5 * u_M * u_M - log_sd_M_ptr[j] - clc_mvop_detail::CLC_HALF_LOG_2PI;
                  g_M_ptr[j] = -u_M / sd_M;
            }
            //// pass 2: the gap prior, the Jacobian and the chain rule to y_a, y_b and pi
            for (int j = 0; j < n_chosen; j++) {
                  const double y_a = s_val * z_marg[class_a * n_slot + j];
                  const double y_b = s_val * z_marg[class_b * n_slot + j];
                  //// the user's gap prior (its own data input), on the probit of Pr(positive): positive = the class with the other
                  //// condition screens positive more. For Se that is D itself; for Sp, Pr(positive) = 1 - Sp, so D = -gap.
                  const double mean_D = (acc_index == 0) ? (*prep.gap_link_mean)[acc_index](j, ord_index)
                                                         : -(*prep.gap_link_mean)[acc_index](j, ord_index);
                  const double sd_D = (*prep.gap_link_SD)[acc_index](j, ord_index);
                  const double M_val = M_ptr[j];
                  const double phi_a = phi_a_ptr[j];
                  const double phi_b = phi_b_ptr[j];
                  const double phi_M = phi_M_ptr[j] * clc_mvop_detail::CLC_INV_SQRT_2PI;
                  const double Q_val = (1.0 - pi_val) * phi_a + pi_val * phi_b;
                  const double u_D = (y_ab_ptr[j] - mean_D) / sd_D;
                  //// N(D; mean_D, sd_D) + log Q - log phi(M): the two half log 2 pi cancel
                  lp_marg += -0.5 * u_D * u_D - log_sd_D_ptr[j] + log_Q_ptr[j] + 0.5 * M_val * M_val;
                  const double g_D = -u_D / sd_D;
                  const double c_M = g_M_ptr[j] + M_val;
                  const double inv_phi_M = 1.0 / phi_M;
                  const double inv_Q = 1.0 / Q_val;
                  const double d_lp_d_y_a = c_M * (1.0 - pi_val) * phi_a * inv_phi_M - g_D - (1.0 - pi_val) * y_a * phi_a * inv_Q;
                  const double d_lp_d_y_b = c_M * pi_val * phi_b * inv_phi_M + g_D - pi_val * y_b * phi_b * inv_Q;
                  const double d_lp_d_pi = c_M * d_Phi_ptr[j] * inv_phi_M + (phi_b - phi_a) * inv_Q;
                  d_z_marg_ext[class_a * n_slot + j] += s_val * d_lp_d_y_a;
                  d_z_marg_ext[class_b * n_slot + j] += s_val * d_lp_d_y_b;
                  d_lp_d_log_W[class_b] += d_lp_d_pi * pi_val * (1.0 - pi_val);
                  d_lp_d_log_W[class_a] -= d_lp_d_pi * pi_val * (1.0 - pi_val);
            }
      }
      prep.log_prob_prior_and_jac += lp_marg;
}




inline void fn_CLC_MVOP_prep_inplace(
      CLC_MVOP_prep &prep,
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const Model_fn_args_struct &Model_args_as_cpp_struct,
      const int n_chunks_multiplier
) {
      out_mat.setZero();

      prep.n_obs = static_cast<int>(y_ref.rows());
      prep.n_tests = static_cast<int>(y_ref.cols());
      prep.n_us = static_cast<int>(theta_us_vec_ref.rows());
      prep.n_params_main = static_cast<int>(theta_main_vec_ref.rows());
      prep.n_params = prep.n_params_main + prep.n_us;
      prep.label_constraint_violated = false;

      //// Boolean flags
      prep.exclude_priors = Model_args_as_cpp_struct.Model_args_bools(0);
      prep.CI = Model_args_as_cpp_struct.Model_args_bools(1);
      prep.corr_force_positive = Model_args_as_cpp_struct.Model_args_bools(2);
      prep.prior_only = (Model_args_as_cpp_struct.Model_args_bools.rows() > 15)
                        ? Model_args_as_cpp_struct.Model_args_bools(15)
                        : false;
      prep.use_pop_Se_Sp_prior = (Model_args_as_cpp_struct.Model_args_bools.rows() > 16)
                                 ? Model_args_as_cpp_struct.Model_args_bools(16)
                                 : false;
      prep.has_y_upper = (Model_args_as_cpp_struct.Model_args_bools.rows() > 17)
                         ? Model_args_as_cpp_struct.Model_args_bools(17)
                         : false;
      prep.X_is_the_same_matrix_for_every_class_and_test = (Model_args_as_cpp_struct.Model_args_bools.rows() > 18)
                                                           ? Model_args_as_cpp_struct.Model_args_bools(18)
                                                           : false;

      //// Integer metadata
      prep.n_class = Model_args_as_cpp_struct.Model_args_ints(1);
      const int n_chunks_base = Model_args_as_cpp_struct.Model_args_ints(3);
      prep.n_binary_tests = Model_args_as_cpp_struct.Model_args_ints(4);
      prep.n_ordinal_tests = Model_args_as_cpp_struct.Model_args_ints(5);
      prep.n_pops = Model_args_as_cpp_struct.Model_args_ints(6);
      //// the only two options: use_Se_Sp_prior = 0, the induced Dirichlet on every class's cutpoints and nothing else; or
      //// use_Se_Sp_prior = 1, the induced accuracy prior on the MARGINAL Se / Sp, always Jensen-adjusted and always with the gap prior
      prep.use_Se_Sp_prior = Model_args_as_cpp_struct.Model_args_ints(7);
      if ((prep.use_Se_Sp_prior != 0) && (prep.use_Se_Sp_prior != 1)) {
            throw std::invalid_argument("CLC-MVOP: use_Se_Sp_prior must be 0 (the induced Dirichlet only) or 1 (the prior on the marginal Se / Sp).");
      }
      if (prep.use_Se_Sp_prior == 1) {
            if (!prep.use_pop_Se_Sp_prior) {
                  throw std::invalid_argument("CLC-MVOP: the prior on the marginal Se / Sp is always Jensen-adjusted: use_pop_Se_Sp_prior must be 1.");
            }
            if (Model_args_as_cpp_struct.Model_args_vecs_of_mats_double.size() < 14) {
                  throw std::invalid_argument("CLC-MVOP: the prior on the marginal Se / Sp needs the gap prior (prior_Se_Sp_gap_link_mean_SD) in Model_args_vecs_of_mats_double[12] and [13].");
            }
            prep.gap_link_mean = &Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[12];
            prep.gap_link_SD = &Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[13];
      }
      prep.n_covariates_max = Model_args_as_cpp_struct.Model_args_ints(9);
      prep.n_prevalence_covariates = Model_args_as_cpp_struct.Model_args_ints(10);
      prep.pool_correlations_across_classes = Model_args_as_cpp_struct.Model_args_ints(11);
      prep.use_pop_prev_prior = Model_args_as_cpp_struct.Model_args_ints(12);
      prep.n_standardisation = Model_args_as_cpp_struct.Model_args_ints(13);
      prep.use_label_constraint = Model_args_as_cpp_struct.Model_args_ints(14);
      prep.n_chosen_cutoffs_max = Model_args_as_cpp_struct.Model_args_ints(15);

      if (prep.n_class != 4) {
            throw std::invalid_argument("CLC-MVOP: requires n_class == 4.");
      }

      //// Doubles
      prep.overflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(0);
      prep.underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(1);
      prep.C_raw_lower = Model_args_as_cpp_struct.Model_args_doubles(2);
      prep.C_raw_upper = Model_args_as_cpp_struct.Model_args_doubles(3);
      prep.perfect_ref_M = Model_args_as_cpp_struct.Model_args_doubles(4);

      //// Pointers into Model_args
      prep.n_cat_per_ord_test = &Model_args_as_cpp_struct.Model_args_col_vecs_int[0];
      prep.n_thr_per_ord_test = &Model_args_as_cpp_struct.Model_args_col_vecs_int[1];
      prep.pop_ind = &Model_args_as_cpp_struct.Model_args_col_vecs_int[2];
      const Eigen::Matrix<int, -1, 1> &n_cat_per_test = Model_args_as_cpp_struct.Model_args_col_vecs_int[3];
      prep.test_condition = &Model_args_as_cpp_struct.Model_args_col_vecs_int[4];
      prep.is_perfect_ref = &Model_args_as_cpp_struct.Model_args_col_vecs_int[5];
      prep.std_index = &Model_args_as_cpp_struct.Model_args_col_vecs_int[6];
      prep.pop_std = &Model_args_as_cpp_struct.Model_args_col_vecs_int[7];
      prep.reference_test_fitted = &Model_args_as_cpp_struct.Model_args_col_vecs_int[8];
      prep.cutoff_category = &Model_args_as_cpp_struct.Model_args_col_vecs_int[9];
      prep.n_covs_per_outcome = &Model_args_as_cpp_struct.Model_args_mats_int[0];
      prep.class_D = &Model_args_as_cpp_struct.Model_args_mats_int[1];
      prep.y_upper_mat = &Model_args_as_cpp_struct.Model_args_mats_int[2];
      prep.X_prev = &Model_args_as_cpp_struct.Model_args_mats_double[0];
      prep.X_prev_std = &Model_args_as_cpp_struct.Model_args_mats_double[1];
      prep.std_w = &Model_args_as_cpp_struct.Model_args_col_vecs_double[5];
      prep.X_class_test = &Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0];

      //// every class and test reads the first stored covariate matrix when all of them are the same matrix
      if (prep.class_index_of_X_to_read.rows() != prep.n_class) {
            prep.class_index_of_X_to_read.resize(prep.n_class);
      }
      if (prep.test_index_of_X_to_read.rows() != prep.n_tests) {
            prep.test_index_of_X_to_read.resize(prep.n_tests);
      }
      for (int class_index = 0; class_index < prep.n_class; class_index++) {
            prep.class_index_of_X_to_read(class_index) = prep.X_is_the_same_matrix_for_every_class_and_test ? 0 : class_index;
      }
      for (int test_index = 0; test_index < prep.n_tests; test_index++) {
            prep.test_index_of_X_to_read(test_index) = prep.X_is_the_same_matrix_for_every_class_and_test ? 0 : test_index;
      }

      //// Slot maps
      prep.ord_idx_of_test.resize(prep.n_tests);
      prep.fitted_test_of_ord_slot.resize(std::max(1, prep.n_ordinal_tests));
      int ord_counter = 0;
      for (int test_index = 0; test_index < prep.n_tests; test_index++) {
            if (n_cat_per_test(test_index) > 2) {
                  prep.ord_idx_of_test(test_index) = ord_counter;
                  prep.fitted_test_of_ord_slot(ord_counter) = test_index;
                  ord_counter++;
            } else {
                  prep.ord_idx_of_test(test_index) = -1;
            }
      }
      prep.n_ordinal_tests = ord_counter;
      prep.n_binary_tests = prep.n_tests - ord_counter;
      prep.ord_start_index.resize(std::max(1, prep.n_ordinal_tests));
      int pos_accum = 0;
      for (int ord_index = 0; ord_index < prep.n_ordinal_tests; ord_index++) {
            prep.ord_start_index(ord_index) = pos_accum;
            pos_accum += (*prep.n_thr_per_ord_test)(ord_index);
      }
      prep.n_total_C_per_class = pos_accum;
      prep.n_cutpoints_max = (prep.n_ordinal_tests > 0)
                             ? prep.n_thr_per_ord_test->head(prep.n_ordinal_tests).maxCoeff()
                             : 0;
      prep.analytic_ws.ensure(
            prep.n_cutpoints_max,
            prep.n_cutpoints_max,
            prep.n_tests
      );

      //// Class-to-condition maps for label constraint
      prep.class_with_neither_condition = 0;
      prep.class_with_only_condition[0] = 1;
      prep.class_with_only_condition[1] = 2;
      for (int class_index = 0; class_index < prep.n_class; class_index++) {
            const int d0 = (*prep.class_D)(class_index, 0);
            const int d1 = (*prep.class_D)(class_index, 1);
            if ((d0 == 0) && (d1 == 0)) {
                  prep.class_with_neither_condition = class_index;
            } else if ((d0 == 1) && (d1 == 0)) {
                  prep.class_with_only_condition[0] = class_index;
            } else if ((d0 == 0) && (d1 == 1)) {
                  prep.class_with_only_condition[1] = class_index;
            }
      }

      //// Correlation counts and actual lower bounds
      const std::vector<Eigen::Matrix<int, -1, -1>> &known_ind =
            Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];
      const std::vector<Eigen::Matrix<double, -1, -1>> &lb_corr_src =
            Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2];
      if (static_cast<int>(prep.lb_corr_actual.size()) != prep.n_class) {
            prep.lb_corr_actual.resize(static_cast<std::size_t>(prep.n_class));
      }
      for (int class_index = 0; class_index < prep.n_class; class_index++) {
            if ((prep.lb_corr_actual[class_index].rows() != lb_corr_src[class_index].rows())
                || (prep.lb_corr_actual[class_index].cols() != lb_corr_src[class_index].cols())) {
                  prep.lb_corr_actual[class_index].resize(
                        lb_corr_src[class_index].rows(),
                        lb_corr_src[class_index].cols()
                  );
            }
            prep.lb_corr_actual[class_index] = lb_corr_src[class_index];
            if (prep.corr_force_positive) {
                  for (int row_index = 1; row_index < prep.n_tests; row_index++) {
                        for (int col_index = 0; col_index < row_index; col_index++) {
                              prep.lb_corr_actual[class_index](row_index, col_index) = 0.0;
                        }
                  }
            }
      }
      if (prep.n_corrs_per_class.rows() != prep.n_class) {
            prep.n_corrs_per_class.resize(prep.n_class);
      }
      prep.n_corrs_per_class.setZero();
      prep.corr_start_index.resize(prep.n_class);
      prep.n_corrs_total = 0;
      for (int class_index = 0; class_index < prep.n_class; class_index++) {
            prep.corr_start_index(class_index) = prep.n_corrs_total;
            for (int row_index = 1; row_index < prep.n_tests; row_index++) {
                  for (int col_index = 0; col_index < row_index; col_index++) {
                        if (known_ind[class_index](row_index, col_index) == 0) {
                              prep.n_corrs_per_class(class_index) += 1;
                        }
                  }
            }
            prep.n_corrs_total += prep.n_corrs_per_class(class_index);
      }
      prep.n_corrs_global = (prep.pool_correlations_across_classes == 1)
                            ? prep.n_corrs_per_class(0)
                            : 0;
      prep.n_lambda = (prep.pool_correlations_across_classes == 1) ? prep.n_class : 0;

      prep.n_covariates_total = 0;
      for (int class_index = 0; class_index < prep.n_class; class_index++) {
            for (int test_index = 0; test_index < prep.n_tests; test_index++) {
                  prep.n_covariates_total += (*prep.n_covs_per_outcome)(class_index, test_index);
            }
      }

      //// Unconstrained theta_main offsets (exact match to CLC_MVOP_cpp_skeleton.stan)
      int offset_cur = 0;
      prep.off_Omega = offset_cur;
      offset_cur += prep.n_corrs_total;
      prep.off_Omega_global = offset_cur;
      offset_cur += prep.n_corrs_global;
      prep.off_lambda_class = offset_cur;
      offset_cur += prep.n_lambda;
      prep.off_beta_vec = offset_cur;
      offset_cur += prep.n_covariates_total;
      prep.off_beta_prev_A = offset_cur;
      offset_cur += prep.n_prevalence_covariates;
      prep.off_beta_prev_B = offset_cur;
      offset_cur += prep.n_prevalence_covariates;
      prep.off_z_logit_A = offset_cur;
      offset_cur += prep.n_pops;
      prep.off_z_logit_B = offset_cur;
      offset_cur += prep.n_pops;
      prep.off_mu_logit_A = offset_cur;
      offset_cur += 1;
      prep.off_mu_logit_B = offset_cur;
      offset_cur += 1;
      prep.off_sigma_logit_A = offset_cur;
      offset_cur += 1;
      prep.off_sigma_logit_B = offset_cur;
      offset_cur += 1;
      prep.off_log_OR = offset_cur;
      offset_cur += 1;
      prep.off_C_unc = offset_cur;
      offset_cur += prep.n_class * prep.n_total_C_per_class;
      if (offset_cur != prep.n_params_main) {
            throw std::invalid_argument(
                  "CLC-MVOP: theta_main length mismatch (expected "
                  + std::to_string(offset_cur)
                  + ", got "
                  + std::to_string(prep.n_params_main)
                  + ")."
            );
      }

      //// Kernel choice and chunk sizes
      const std::string &vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
      prep.kchoice = kernel_choice_from_args(
            Model_args_as_cpp_struct
      );
      int vec_size = 1;
      if (vect_type == "AVX512") {
            vec_size = 8;
      } else if (vect_type == "AVX2") {
            vec_size = 4;
      } else if (vect_type == "AVX") {
            vec_size = 2;
      }
      const int n_chunks_eff = std::max(
            1,
            n_chunks_base * std::max(1, n_chunks_multiplier)
      );
      const ChunkSizeInfo chunk_info = calculate_chunk_sizes(
            prep.n_obs,
            vec_size,
            n_chunks_eff
      );
      prep.chunk_size_orig = chunk_info.chunk_size_orig;
      prep.normal_chunk_size = chunk_info.normal_chunk_size;
      prep.last_chunk_size = chunk_info.last_chunk_size;
      prep.n_total_chunks = chunk_info.n_total_chunks;
      prep.n_full_chunks = chunk_info.n_full_chunks;

      //// Se/Sp prior metadata (built via fn_Se_Sp_prior_args whenever any prior input value changes)
      if ((prep.use_Se_Sp_prior == 1) && (prep.n_ordinal_tests > 0)) {
            const std::vector<Eigen::Matrix<int, -1, -1>> &in_chosen =
                  Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[1];
            const std::vector<Eigen::Matrix<double, -1, -1>> &in_A =
                  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6];
            const std::vector<Eigen::Matrix<double, -1, -1>> &in_ba =
                  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[7];
            const std::vector<Eigen::Matrix<double, -1, -1>> &in_bb =
                  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[8];
            const std::vector<Eigen::Matrix<double, -1, -1>> &in_lm =
                  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[9];
            const std::vector<Eigen::Matrix<double, -1, -1>> &in_ls =
                  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[10];
            const std::vector<Eigen::Matrix<double, -1, -1>> &in_al =
                  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5];
            const bool cache_matches = prep.se_sp_cache_valid
                  && (prep.se_sp_cache_chosen_cutoffs == in_chosen)
                  && (prep.se_sp_cache_A_user == in_A)
                  && (prep.se_sp_cache_beta_a == in_ba)
                  && (prep.se_sp_cache_beta_b == in_bb)
                  && (prep.se_sp_cache_link_mean == in_lm)
                  && (prep.se_sp_cache_link_SD == in_ls)
                  && (prep.se_sp_cache_alpha == in_al)
                  && (prep.se_sp_cache_n_thr == *prep.n_thr_per_ord_test)
                  && (prep.se_sp_cache_fitted_test == prep.fitted_test_of_ord_slot)
                  && (prep.se_sp_cache_test_cond == *prep.test_condition)
                  && (prep.se_sp_cache_class_D == *prep.class_D);
            if (!cache_matches) {
                  prep.Se_Sp_args = fn_Se_Sp_prior_args(
                        in_chosen,
                        in_A,
                        in_ba,
                        in_bb,
                        in_lm,
                        in_ls,
                        in_al,
                        *prep.n_thr_per_ord_test,
                        0);
                  for (int class_index = 0; class_index < prep.n_class; class_index++) {
                        for (int ord_index = 0; ord_index < prep.n_ordinal_tests; ord_index++) {
                              if (prep.Se_Sp_args.n_chosen[class_index](ord_index) == 0) {
                                    continue;
                              }
                              const int test_index = prep.fitted_test_of_ord_slot(ord_index);
                              const int cond_index = (*prep.test_condition)(test_index);
                              const bool has_own_condition = ((*prep.class_D)(class_index, cond_index) == 1);
                              if (has_own_condition != (class_index == 1)) {
                                    prep.Se_Sp_args.link_mean_SD[class_index][ord_index].col(0) =
                                          -prep.Se_Sp_args.link_mean_SD[class_index][ord_index].col(0);
                              }
                        }
                  }
                  prep.se_sp_cache_chosen_cutoffs = in_chosen;
                  prep.se_sp_cache_A_user = in_A;
                  prep.se_sp_cache_beta_a = in_ba;
                  prep.se_sp_cache_beta_b = in_bb;
                  prep.se_sp_cache_link_mean = in_lm;
                  prep.se_sp_cache_link_SD = in_ls;
                  prep.se_sp_cache_alpha = in_al;
                  prep.se_sp_cache_n_thr = *prep.n_thr_per_ord_test;
                  prep.se_sp_cache_fitted_test = prep.fitted_test_of_ord_slot;
                  prep.se_sp_cache_test_cond = *prep.test_condition;
                  prep.se_sp_cache_class_D = *prep.class_D;
                  prep.se_sp_cache_valid = true;
            }
      }

      prep.std_index_is_contiguous = true;
      for (int prof_index = 0; prof_index < prep.n_standardisation; prof_index++) {
            if ((*prep.std_index)(prof_index) != prof_index) {
                  prep.std_index_is_contiguous = false;
                  break;
            }
      }

      prep.log_prob_prior_and_jac = 0.0;
      if (prep.grad_main_prior_and_jac.rows() != prep.n_params_main) {
            prep.grad_main_prior_and_jac.resize(prep.n_params_main);
      }
      prep.grad_main_prior_and_jac.setZero();

      const int K_alloc = std::max(1, prep.n_covariates_max);
      const int cut_rows_alloc = std::max(1, prep.n_cutpoints_max);
      const int cut_cols_alloc = std::max(1, prep.n_ordinal_tests);
      if (static_cast<int>(prep.beta_4.size()) != prep.n_class) {
            prep.beta_4.resize(static_cast<std::size_t>(prep.n_class));
            prep.beta_grad_4_from_prior.resize(static_cast<std::size_t>(prep.n_class));
            prep.C_raw_4.resize(static_cast<std::size_t>(prep.n_class));
            prep.C_4.resize(static_cast<std::size_t>(prep.n_class));
            prep.cutpoint_grad_4_from_prior.resize(static_cast<std::size_t>(prep.n_class));
      }
      for (int class_index = 0; class_index < prep.n_class; class_index++) {
            if ((prep.beta_4[class_index].rows() != K_alloc)
                || (prep.beta_4[class_index].cols() != prep.n_tests)) {
                  prep.beta_4[class_index].resize(K_alloc, prep.n_tests);
                  prep.beta_grad_4_from_prior[class_index].resize(K_alloc, prep.n_tests);
            }
            prep.beta_4[class_index].setZero();
            prep.beta_grad_4_from_prior[class_index].setZero();

            if ((prep.C_4[class_index].rows() != cut_rows_alloc)
                || (prep.C_4[class_index].cols() != cut_cols_alloc)) {
                  prep.C_raw_4[class_index].resize(cut_rows_alloc, cut_cols_alloc);
                  prep.C_4[class_index].resize(cut_rows_alloc, cut_cols_alloc);
                  prep.cutpoint_grad_4_from_prior[class_index].resize(cut_rows_alloc, cut_cols_alloc);
            }
            prep.C_raw_4[class_index].setZero();
            prep.C_4[class_index].setZero();
            prep.cutpoint_grad_4_from_prior[class_index].setZero();
      }
      if ((prep.d_lp_d_log_prev_std_from_pop_prior.rows() != prep.n_standardisation)
          || (prep.d_lp_d_log_prev_std_from_pop_prior.cols() != prep.n_class)) {
            prep.d_lp_d_log_prev_std_from_pop_prior.resize(
                  prep.n_standardisation,
                  prep.n_class
            );
      }
      prep.d_lp_d_log_prev_std_from_pop_prior.setZero();

      //// -------------------------------------------------------------------------------------
      //// 1. Correlations (pure-double Pinkney LDL, pooling, Cholesky, LKJ, Beta, and Jacobians)
      //// -------------------------------------------------------------------------------------
      const Vec vec_of_vect_type = vec_from_string(vect_type);
      fn_CLC_MVOP_corr_forward_double(
            prep,
            theta_main_vec_ref,
            Model_args_as_cpp_struct,
            vec_of_vect_type
      );

      //// -------------------------------------------------------------------------------------
      //// 2. Per-class regression coefficients beta[c, k, t] and Gaussian priors
      //// -------------------------------------------------------------------------------------
      const std::vector<Eigen::Matrix<double, -1, -1>> &prior_beta_mean =
            Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0];
      const std::vector<Eigen::Matrix<double, -1, -1>> &prior_beta_sd =
            Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1];
      const std::vector<Eigen::Matrix<double, -1, -1>> &baseline_case =
            Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[11];

      int beta_cursor = 0;
      for (int class_index = 0; class_index < prep.n_class; class_index++) {
            for (int test_index = 0; test_index < prep.n_tests; test_index++) {
                  const int n_covs_ct = (*prep.n_covs_per_outcome)(class_index, test_index);
                  const bool is_perf = ((*prep.is_perfect_ref)(test_index) == 1);
                  for (int cov_index = 0; cov_index < n_covs_ct; cov_index++) {
                        const double val_cur = theta_main_vec_ref(prep.off_beta_vec + beta_cursor);
                        if (is_perf) {
                              if (!prep.exclude_priors) {
                                    prep.log_prob_prior_and_jac += clc_mvop_detail::dbl_normal_lpdf(
                                          val_cur,
                                          0.0,
                                          1.0
                                    );
                                    prep.grad_main_prior_and_jac(prep.off_beta_vec + beta_cursor) += -val_cur;
                              }
                        } else {
                              prep.beta_4[class_index](cov_index, test_index) = val_cur;
                              if (!prep.exclude_priors) {
                                    const double mean_cur = prior_beta_mean[class_index](cov_index, test_index);
                                    const double sd_cur = prior_beta_sd[class_index](cov_index, test_index);
                                    prep.log_prob_prior_and_jac += clc_mvop_detail::dbl_normal_lpdf(
                                          val_cur,
                                          mean_cur,
                                          sd_cur
                                    );
                                    prep.grad_main_prior_and_jac(prep.off_beta_vec + beta_cursor) +=
                                          -(val_cur - mean_cur) / (sd_cur * sd_cur);
                              }
                        }
                        beta_cursor++;
                  }
                  if (is_perf) {
                        const int cond_index = (*prep.test_condition)(test_index);
                        prep.beta_4[class_index](0, test_index) =
                              ((*prep.class_D)(class_index, cond_index) == 1)
                              ? prep.perfect_ref_M
                              : (-prep.perfect_ref_M);
                        for (int cov_index = 1; cov_index < K_alloc; cov_index++) {
                              prep.beta_4[class_index](cov_index, test_index) = 0.0;
                        }
                  }
            }
      }

      //// -------------------------------------------------------------------------------------
      //// 3. Per-class ordered cutpoints C_unc -> C_raw (tanh lb_ub) -> C (log-difference)
      //// -------------------------------------------------------------------------------------
      if (prep.n_ordinal_tests > 0) {
            const double C_range = prep.C_raw_upper - prep.C_raw_lower;
            const double log_C_range = stan::math::log(C_range);
            DISPATCH_VEC(
                  vec_from_string(vect_type),
                  fn_CLC_MVOP_tanh_log1p_exp_and_exp_of_the_cutpoints_T,
                  theta_main_vec_ref.segment(prep.off_C_unc, prep.n_class * prep.n_total_C_per_class),
                  prep.C_raw_lower,
                  C_range,
                  prep.tanh_of_C_unc_buf,
                  prep.log1p_exp_of_minus_2_abs_C_unc_buf,
                  prep.exp_of_C_raw_buf
            );
            for (int class_index = 0; class_index < prep.n_class; class_index++) {
                  const int class_base = prep.off_C_unc + class_index * prep.n_total_C_per_class;
                  for (int ord_index = 0; ord_index < prep.n_ordinal_tests; ord_index++) {
                        const int n_thr_test = (*prep.n_thr_per_ord_test)(ord_index);
                        const int ord_base = class_base + prep.ord_start_index(ord_index);
                        for (int thr_index = 0; thr_index < n_thr_test; thr_index++) {
                              const double unc_val = theta_main_vec_ref(ord_base + thr_index);
                              const int cutpoint_index = ord_base + thr_index - prep.off_C_unc;
                              const double tanh_val = prep.tanh_of_C_unc_buf(cutpoint_index);
                              const double C_raw_val = prep.C_raw_lower + C_range * 0.5 * (1.0 + tanh_val);
                              prep.C_raw_4[class_index](thr_index, ord_index) = C_raw_val;
                              const double abs_unc_val = (unc_val >= 0.0) ? unc_val : (-unc_val);
                              prep.log_prob_prior_and_jac += log_C_range
                                                             + clc_mvop_detail::CLC_LOG_2
                                                             - 2.0 * abs_unc_val
                                                             - 2.0 * prep.log1p_exp_of_minus_2_abs_C_unc_buf(cutpoint_index);
                              //// d/du log(sech^2(u)) = -2 * tanh(u)
                              prep.grad_main_prior_and_jac(ord_base + thr_index) += -2.0 * tanh_val;

                              if (thr_index == 0) {
                                    prep.C_4[class_index](0, ord_index) = C_raw_val;
                              } else {
                                    prep.C_4[class_index](thr_index, ord_index) =
                                          prep.C_4[class_index](thr_index - 1, ord_index) + prep.exp_of_C_raw_buf(cutpoint_index);
                                    //// Log-difference Jacobian += C_raw_val; d(C_raw_val)/du = 0.5 * C_range * (1 - tanh^2(u))
                                    prep.log_prob_prior_and_jac += C_raw_val;
                                    prep.grad_main_prior_and_jac(ord_base + thr_index) +=
                                          0.5 * C_range * (1.0 - tanh_val * tanh_val);
                              }
                        }
                  }
            }
      }

      //// -------------------------------------------------------------------------------------
      //// 4. Optional label-ordering truncation constraint
      //// -------------------------------------------------------------------------------------
      if (prep.use_label_constraint == 1) {
            //// every test (the references and the screens): at its cut-off, the class with only the test's own condition
            //// has a linear predictor at least that of the class with neither condition (no test inverted); a hard
            //// constraint (the chains start inside it, from the runner's starting values)
            for (int test_index = 0; test_index < prep.n_tests; test_index++) {
                  const int cond_index = (*prep.test_condition)(test_index);
                  const int class_pair[2] = {
                        prep.class_with_only_condition[cond_index],
                        prep.class_with_neither_condition
                  };
                  double metric_val[2] = { 0.0, 0.0 };
                  for (int pair_pos = 0; pair_pos < 2; pair_pos++) {
                        const int class_index = class_pair[pair_pos];
                        const int n_covs_ct = (*prep.n_covs_per_outcome)(class_index, test_index);
                        double xb_base = 0.0;
                        for (int cov_index = 0; cov_index < n_covs_ct; cov_index++) {
                              xb_base += baseline_case[class_index](cov_index, test_index)
                                         * prep.beta_4[class_index](cov_index, test_index);
                        }
                        const int ord_index = prep.ord_idx_of_test(test_index);
                        if (ord_index >= 0) {
                              const int cut_idx = (*prep.cutoff_category)(test_index) - 2;
                              xb_base -= prep.C_4[class_index](cut_idx, ord_index);
                        }
                        metric_val[pair_pos] = xb_base;
                  }
                  if (metric_val[0] < metric_val[1]) {
                        prep.label_constraint_violated = true;
                        return;
                  }
            }
      }

      //// -------------------------------------------------------------------------------------
      //// 5. Prevalence model (2 x 2 conditional factorial + population standardisation prior)
      //// -------------------------------------------------------------------------------------
      prep.beta_prev_A.resize(prep.n_prevalence_covariates);
      prep.beta_prev_B.resize(prep.n_prevalence_covariates);
      const Eigen::Matrix<double, -1, 1> &prior_bpA_mean = Model_args_as_cpp_struct.Model_args_col_vecs_double[1];
      const Eigen::Matrix<double, -1, 1> &prior_bpA_sd = Model_args_as_cpp_struct.Model_args_col_vecs_double[2];
      const Eigen::Matrix<double, -1, 1> &prior_bpB_mean = Model_args_as_cpp_struct.Model_args_col_vecs_double[3];
      const Eigen::Matrix<double, -1, 1> &prior_bpB_sd = Model_args_as_cpp_struct.Model_args_col_vecs_double[4];
      for (int cov_index = 0; cov_index < prep.n_prevalence_covariates; cov_index++) {
            const double val_A = theta_main_vec_ref(prep.off_beta_prev_A + cov_index);
            const double val_B = theta_main_vec_ref(prep.off_beta_prev_B + cov_index);
            prep.beta_prev_A(cov_index) = val_A;
            prep.beta_prev_B(cov_index) = val_B;
            if (!prep.exclude_priors) {
                  prep.log_prob_prior_and_jac += clc_mvop_detail::dbl_normal_lpdf(
                        val_A,
                        prior_bpA_mean(cov_index),
                        prior_bpA_sd(cov_index)
                  );
                  prep.grad_main_prior_and_jac(prep.off_beta_prev_A + cov_index) +=
                        -(val_A - prior_bpA_mean(cov_index)) / (prior_bpA_sd(cov_index) * prior_bpA_sd(cov_index));
                  prep.log_prob_prior_and_jac += clc_mvop_detail::dbl_normal_lpdf(
                        val_B,
                        prior_bpB_mean(cov_index),
                        prior_bpB_sd(cov_index)
                  );
                  prep.grad_main_prior_and_jac(prep.off_beta_prev_B + cov_index) +=
                        -(val_B - prior_bpB_mean(cov_index)) / (prior_bpB_sd(cov_index) * prior_bpB_sd(cov_index));
            }
      }

      prep.z_A = theta_main_vec_ref.segment(
            prep.off_z_logit_A,
            prep.n_pops
      );
      prep.z_B = theta_main_vec_ref.segment(
            prep.off_z_logit_B,
            prep.n_pops
      );
      prep.mu_A = theta_main_vec_ref(prep.off_mu_logit_A);
      prep.mu_B = theta_main_vec_ref(prep.off_mu_logit_B);
      const double log_sig_A = theta_main_vec_ref(prep.off_sigma_logit_A);
      const double log_sig_B = theta_main_vec_ref(prep.off_sigma_logit_B);
      prep.sigma_A = std::exp(log_sig_A);
      prep.sigma_B = std::exp(log_sig_B);
      ////
      //// ---- Pinkney partially-centered population logits (as CLC_MVOP_PartialLog_v2_NicoSIMD_AVX512_fused.stan):
      ////      theta_main holds z_pcp at off_z_logit_*; w = 1 / (1 + n_g p (1 - p) sigma^2), p = inv_logit(mu);
      ////      tau_wt = sigma - w (sigma - 1); prep.z_A / prep.z_B become z = (z_pcp - mu (1 - w)) / tau_wt, the non-centered
      ////      values the rest of the code uses (their N(0, 1) prior is unchanged); log-Jacobian -sum log tau_wt (always);
      ////      the gradient w.r.t. (z_pcp, mu, log sigma) is formed at the end of fn_CLC_MVOP_assemble.
      ////
      prep.z_A = theta_main_vec_ref.segment(
            prep.off_z_logit_A,
            prep.n_pops
      );
      prep.z_B = theta_main_vec_ref.segment(
            prep.off_z_logit_B,
            prep.n_pops
      );
      {
            if (2 * prep.n_pops > 50) {
                  throw std::domain_error("CLC_MVOP: the Pinkney buffers hold 2 * n_pops <= 50 values (Rule 0); n_pops = "
                                          + std::to_string(prep.n_pops));
            }
            //// inv_logit(mu_A), inv_logit(mu_B) with his NicoSIMD kernel (vect_type: AVX512, AVX2 or Scalar)
            prep.pcp_eta.resize(2);
            prep.pcp_prob.resize(2);
            prep.pcp_one_m_prob.resize(2);
            prep.pcp_log_prob.resize(2);
            prep.pcp_log_one_m_prob.resize(2);
            prep.pcp_eta(0) = prep.mu_A;
            prep.pcp_eta(1) = prep.mu_B;
            DISPATCH_VEC(
                  vec_of_vect_type,
                  fn_CLC_MVOP_inv_logit_and_its_logs_chunk_T,
                  prep.pcp_eta.data(),
                  prep.pcp_prob.data(),
                  prep.pcp_one_m_prob.data(),
                  prep.pcp_log_prob.data(),
                  prep.pcp_log_one_m_prob.data(),
                  2
            );
            prep.pcp_p_A = prep.pcp_prob(0);
            prep.pcp_p_B = prep.pcp_prob(1);
            //// size of each population: one pass over the rows
            prep.pcp_n_of_pop.resize(prep.n_pops);
            prep.pcp_n_of_pop.setZero();
            for (int row_index = 0; row_index < prep.n_obs; row_index++) {
                  prep.pcp_n_of_pop((*prep.pop_ind)(row_index)) += 1.0;
            }
            prep.pcp_w.resize(2 * prep.n_pops);
            prep.pcp_tau_wt.resize(2 * prep.n_pops);
            prep.pcp_log_tau_wt.resize(2 * prep.n_pops);
            const double pq_A = prep.pcp_p_A * prep.pcp_one_m_prob(0);
            const double pq_B = prep.pcp_p_B * prep.pcp_one_m_prob(1);
            for (int pop_index = 0; pop_index < prep.n_pops; pop_index++) {
                  const double n_of_pop = prep.pcp_n_of_pop(pop_index);
                  const double w_A = 1.0 / (1.0 + n_of_pop * pq_A * prep.sigma_A * prep.sigma_A);
                  const double w_B = 1.0 / (1.0 + n_of_pop * pq_B * prep.sigma_B * prep.sigma_B);
                  const double tau_wt_A = prep.sigma_A - w_A * (prep.sigma_A - 1.0);
                  const double tau_wt_B = prep.sigma_B - w_B * (prep.sigma_B - 1.0);
                  prep.pcp_w(pop_index) = w_A;
                  prep.pcp_w(prep.n_pops + pop_index) = w_B;
                  prep.pcp_tau_wt(pop_index) = tau_wt_A;
                  prep.pcp_tau_wt(prep.n_pops + pop_index) = tau_wt_B;
                  prep.z_A(pop_index) = (theta_main_vec_ref(prep.off_z_logit_A + pop_index) - prep.mu_A * (1.0 - w_A)) / tau_wt_A;
                  prep.z_B(pop_index) = (theta_main_vec_ref(prep.off_z_logit_B + pop_index) - prep.mu_B * (1.0 - w_B)) / tau_wt_B;
            }
            //// (the log-Jacobian -sum log tau_wt is added in fn_CLC_MVOP_chunked_prev_and_pop_prior_T<vec>, NicoSIMD log)
      }
      //// ---- logit Pr(B | A) intercept is sampled at off_log_OR; log_OR = mu_logit_B_given_A - mu_logit_B (Jacobian 1).
      //// ---- use_pop_prev_prior = 3: log_OR >= 0 (B never lowers the chance of A, and so A never lowers the chance of B):
      ////      mu_logit_B_given_A has the lower bound mu_logit_B (as in the skeleton), so the sampled value is
      ////      u = log(mu_logit_B_given_A - mu_logit_B), log_OR = exp(u) = Pr / (1 - Pr) with Pr = inv_logit(u) (his NicoSIMD
      ////      kernel), log-Jacobian u; its gradient is done in fn_CLC_MVOP_assemble.
      if (prep.use_pop_prev_prior == 3) {
            const double u_log_OR = theta_main_vec_ref(prep.off_log_OR);
            double p_u = 0.0;
            double one_m_p_u = 0.0;
            double log_p_u = 0.0;
            double log_one_m_p_u = 0.0;
            DISPATCH_VEC(
                  vec_of_vect_type,
                  fn_CLC_MVOP_inv_logit_and_its_logs_chunk_T,
                  &u_log_OR,
                  &p_u,
                  &one_m_p_u,
                  &log_p_u,
                  &log_one_m_p_u,
                  1
            );
            prep.log_OR = p_u / one_m_p_u;
            prep.log_prob_prior_and_jac += u_log_OR;
      } else {
            prep.log_OR = theta_main_vec_ref(prep.off_log_OR) - prep.mu_B;
      }

      //// Jacobians for positive sigma_A and sigma_B
      prep.log_prob_prior_and_jac += log_sig_A + log_sig_B;
      prep.grad_main_prior_and_jac(prep.off_sigma_logit_A) += 1.0;
      prep.grad_main_prior_and_jac(prep.off_sigma_logit_B) += 1.0;

      if (!prep.exclude_priors) {
            for (int pop_index = 0; pop_index < prep.n_pops; pop_index++) {
                  prep.log_prob_prior_and_jac += clc_mvop_detail::dbl_normal_lpdf(
                        prep.z_A(pop_index),
                        0.0,
                        1.0
                  );
                  prep.grad_main_prior_and_jac(prep.off_z_logit_A + pop_index) += -prep.z_A(pop_index);
                  prep.log_prob_prior_and_jac += clc_mvop_detail::dbl_normal_lpdf(
                        prep.z_B(pop_index),
                        0.0,
                        1.0
                  );
                  prep.grad_main_prior_and_jac(prep.off_z_logit_B + pop_index) += -prep.z_B(pop_index);
            }
            const double prior_sig_A_sd = Model_args_as_cpp_struct.Model_args_doubles(13);
            const double prior_sig_B_sd = Model_args_as_cpp_struct.Model_args_doubles(14);
            prep.log_prob_prior_and_jac += clc_mvop_detail::dbl_normal_lpdf(
                  prep.sigma_A,
                  0.0,
                  prior_sig_A_sd
            );
            prep.grad_main_prior_and_jac(prep.off_sigma_logit_A) +=
                  -(prep.sigma_A * prep.sigma_A) / (prior_sig_A_sd * prior_sig_A_sd);
            prep.log_prob_prior_and_jac += clc_mvop_detail::dbl_normal_lpdf(
                  prep.sigma_B,
                  0.0,
                  prior_sig_B_sd
            );
            prep.grad_main_prior_and_jac(prep.off_sigma_logit_B) +=
                  -(prep.sigma_B * prep.sigma_B) / (prior_sig_B_sd * prior_sig_B_sd);

            if (prep.use_pop_prev_prior == 0) {
                  const double prior_muA_m = Model_args_as_cpp_struct.Model_args_doubles(9);
                  const double prior_muA_s = Model_args_as_cpp_struct.Model_args_doubles(10);
                  const double prior_muB_m = Model_args_as_cpp_struct.Model_args_doubles(11);
                  const double prior_muB_s = Model_args_as_cpp_struct.Model_args_doubles(12);
                  prep.log_prob_prior_and_jac += clc_mvop_detail::dbl_normal_lpdf(
                        prep.mu_A,
                        prior_muA_m,
                        prior_muA_s
                  );
                  prep.grad_main_prior_and_jac(prep.off_mu_logit_A) +=
                        -(prep.mu_A - prior_muA_m) / (prior_muA_s * prior_muA_s);
                  prep.log_prob_prior_and_jac += clc_mvop_detail::dbl_normal_lpdf(
                        prep.mu_B,
                        prior_muB_m,
                        prior_muB_s
                  );
                  prep.grad_main_prior_and_jac(prep.off_mu_logit_B) +=
                        -(prep.mu_B - prior_muB_m) / (prior_muB_s * prior_muB_s);
            }
      }

      //// Per-subject log prevalences, prevalences, and population prevalence prior (chunked in workspace)
      DISPATCH_VEC(vec_of_vect_type,
                   fn_CLC_MVOP_chunked_prev_and_pop_prior_T,
                   prep,
                   Model_args_as_cpp_struct);

      //// -------------------------------------------------------------------------------------
      //// 6. Induced-Dirichlet and optional Se/Sp cutpoint priors (baseline OR population-averaged)
      //// -------------------------------------------------------------------------------------
      if (!prep.exclude_priors && (prep.n_ordinal_tests > 0)) {
            const std::vector<Eigen::Matrix<double, -1, -1>> &prior_dir_alpha =
                  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5];

            if (prep.use_pop_Se_Sp_prior) {
                  const Vec vec_simd = vec_from_string(vect_type);
                  const int n_std = prep.n_standardisation;
                  if (prep.log_std_w_buf.rows() != n_std) {
                        prep.log_std_w_buf.resize(n_std);
                        prep.log_weight_c_buf.resize(n_std);
                        prep.Xbeta_std_buf.resize(n_std);
                        prep.d_Xbeta_std_buf.resize(n_std);
                  }
                  prep.log_std_w_buf.array() = prep.std_w->head(n_std).array().log();
                  if (!prep.std_index_is_contiguous && (prep.Xbeta_all_obs_buf.rows() != prep.n_obs)) {
                        prep.Xbeta_all_obs_buf.resize(prep.n_obs);
                        prep.d_Xbeta_all_obs_buf.resize(prep.n_obs);
                  }

                  const std::vector<int> empty_chosen;
                  const Eigen::Matrix<double, -1, 1> empty_vec;
                  const Eigen::Matrix<double, -1, -1> empty_mat;

                  const bool prior_on_marginal = (prep.use_Se_Sp_prior == 1);
                  for (int ord_index = 0; ord_index < prep.n_ordinal_tests; ord_index++) {
                        //// the prior on the marginal Se / Sp of this test (and its derivative w.r.t. every class's z and class weight)
                        //// is worked out first, since each per-class kernel needs its part of the derivative before its own chain rule
                        double d_lp_d_log_W[4] = { 0.0, 0.0, 0.0, 0.0 };
                        if (prior_on_marginal) {
                              fn_CLC_MVOP_marginal_Se_Sp_prior(prep,
                                                               ord_index,
                                                               d_lp_d_log_W);
                        }

                        for (int class_index = 0; class_index < prep.n_class; class_index++) {
                              fn_CLC_MVOP_survey_log_weight_one_class(prep,
                                                                      class_index,
                                                                      n_std);
                              const int test_index = prep.fitted_test_of_ord_slot(ord_index);
                              const int n_thr_test = (*prep.n_thr_per_ord_test)(ord_index);
                              const int n_cat_test = n_thr_test + 1;
                              const int n_covs_ct = (*prep.n_covs_per_outcome)(class_index, test_index);
                              const bool is_perf = ((*prep.is_perfect_ref)(test_index) == 1);
                              const Eigen::Matrix<double, -1, -1> &X_ct =
                                    (*prep.X_class_test)[prep.class_index_of_X_to_read(class_index)][prep.test_index_of_X_to_read(test_index)];

                              const int pair_index = class_index * prep.n_ordinal_tests + ord_index;
                              fn_CLC_MVOP_survey_Xbeta_one_class_test(prep,
                                                                      X_ct,
                                                                      class_index,
                                                                      test_index,
                                                                      n_covs_ct,
                                                                      n_std);

                              const bool apply_se_sp = (prep.use_Se_Sp_prior == 1)
                                                       && (prep.Se_Sp_args.n_chosen[class_index](ord_index) > 0);
                              const std::vector<int> &chosen_ref = apply_se_sp
                                    ? prep.Se_Sp_args.chosen[class_index][ord_index]
                                    : empty_chosen;
                              const Eigen::Matrix<double, -1, 1> &alpha_seg_ref = apply_se_sp
                                    ? prep.Se_Sp_args.alpha_seg[class_index][ord_index]
                                    : empty_vec;

                              if (prep.d_C_t_buf.rows() < n_thr_test) {
                                    prep.d_C_t_buf.resize(n_thr_test);
                              }
                              prep.d_C_t_buf.head(n_thr_test).setZero();
                              prep.d_Xbeta_std_buf.setZero();
                              double d_beta_0 = 0.0;

                              const double x0_val = baseline_case[class_index](0, test_index);
                              const double beta_0_val = prep.beta_4[class_index](0, test_index);
                              const double p_b0_mean = prior_beta_mean[class_index](0, test_index);
                              const double p_b0_sd = prior_beta_sd[class_index](0, test_index);

                              DISPATCH_VEC(
                                    vec_simd,
                                    clc_mvop_detail::clc_pop_all_cutpoints_prior_one_class_test_T,
                                    prep.C_4[class_index].col(ord_index).head(n_thr_test),
                                    prep.Xbeta_std_buf,
                                    prep.log_weight_c_buf,
                                    prior_dir_alpha[class_index].col(ord_index).head(n_cat_test),
                                    x0_val,
                                    beta_0_val,
                                    p_b0_mean,
                                    p_b0_sd,
                                    apply_se_sp,
                                    chosen_ref,
                                    alpha_seg_ref,
                                    prior_on_marginal ? prep.analytic_ws.Se_Sp_buf.data() + class_index * prep.analytic_ws.n_slot_len : nullptr,
                                    prep.log_prob_prior_and_jac,
                                    prep.d_C_t_buf.head(n_thr_test),
                                    prep.d_Xbeta_std_buf,
                                    d_beta_0,
                                    prep.d_lp_d_log_prev_std_from_pop_prior.col(class_index),
                                    prep.norm_w_buf,
                                    prep.log_wn_buf,
                                    prep.d_norm_w_buf,
                                    prep.binned_Phi_ws,
                                    prep.analytic_ws
                              );
                              //// the marginal Se / Sp prior's derivative (this test's part) w.r.t. log W of this class, onto every profile:
                              //// d log W / d log Pr(class | profile i) = the normalised weight of profile i, which the kernel leaves in norm_w_buf
                              if (prior_on_marginal) {
                                    prep.d_lp_d_log_prev_std_from_pop_prior.col(class_index).array() += d_lp_d_log_W[class_index] * prep.norm_w_buf.array();
                              }

                              prep.cutpoint_grad_4_from_prior[class_index].col(ord_index).head(n_thr_test) +=
                                    prep.d_C_t_buf.head(n_thr_test);
                              if (!is_perf && (n_covs_ct > 0)) {
                                    prep.beta_grad_4_from_prior[class_index](0, test_index) += d_beta_0;
                                    if (prep.std_index_is_contiguous) {
                                          prep.beta_grad_4_from_prior[class_index].col(test_index).head(n_covs_ct).noalias() +=
                                                X_ct.topLeftCorner(
                                                      n_std,
                                                      n_covs_ct
                                                ).transpose()
                                                * prep.d_Xbeta_std_buf;
                                    } else {
                                          //// derivatives of the survey rows added onto their subjects
                                          prep.d_Xbeta_all_obs_buf.setZero();
                                          const double *d_Xbeta_std_ptr = prep.d_Xbeta_std_buf.data();
                                          const int *std_index_ptr = prep.std_index->data();
                                          double *d_Xbeta_all_obs_ptr = prep.d_Xbeta_all_obs_buf.data();
                                          for (int prof_index = 0; prof_index < n_std; prof_index++) {
                                                d_Xbeta_all_obs_ptr[std_index_ptr[prof_index]] += d_Xbeta_std_ptr[prof_index];
                                          }
                                          prep.beta_grad_4_from_prior[class_index].col(test_index).head(n_covs_ct).noalias() +=
                                                X_ct.topLeftCorner(
                                                      prep.n_obs,
                                                      n_covs_ct
                                                ).transpose()
                                                * prep.d_Xbeta_all_obs_buf;
                                    }
                              }
                        }
                  }
            } else {
                  //// Baseline-case induced-Dirichlet only (use_Se_Sp_prior = 0; the Se / Sp prior is always Jensen-adjusted)
                  for (int class_index = 0; class_index < prep.n_class; class_index++) {
                        for (int ord_index = 0; ord_index < prep.n_ordinal_tests; ord_index++) {
                              const int test_index = prep.fitted_test_of_ord_slot(ord_index);
                              const int n_thr_test = (*prep.n_thr_per_ord_test)(ord_index);
                              const int n_cat_test = n_thr_test + 1;
                              const int n_covs_ct = (*prep.n_covs_per_outcome)(class_index, test_index);

                              double baseline_loc_prior = 0.0;
                              double v0_val = 0.0;
                              for (int cov_index = 0; cov_index < n_covs_ct; cov_index++) {
                                    const double b_val = baseline_case[class_index](cov_index, test_index);
                                    baseline_loc_prior += b_val * prior_beta_mean[class_index](cov_index, test_index);
                                    const double term_sd = b_val * prior_beta_sd[class_index](cov_index, test_index);
                                    v0_val += term_sd * term_sd;
                              }

                              Eigen::Matrix<double, -1, 1> C_centred(n_thr_test);
                              Eigen::Matrix<double, -1, 1> d_C_cur = Eigen::Matrix<double, -1, 1>::Zero(n_thr_test);
                              for (int thr_index = 0; thr_index < n_thr_test; thr_index++) {
                                    C_centred(thr_index) = prep.C_4[class_index](thr_index, ord_index) - baseline_loc_prior;
                              }
                              prep.log_prob_prior_and_jac += clc_mvop_detail::clc_induced_dirichlet_log_scale_val_grad(
                                    C_centred,
                                    prior_dir_alpha[class_index].col(ord_index).head(n_cat_test),
                                    false,
                                    d_C_cur,
                                    prep.analytic_ws
                              );
                              prep.cutpoint_grad_4_from_prior[class_index].col(ord_index).head(n_thr_test) += d_C_cur;
                        }
                  }
            }
      }
}

inline CLC_MVOP_prep fn_CLC_MVOP_prep(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
      const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
      const Model_fn_args_struct &Model_args_as_cpp_struct,
      const int n_chunks_multiplier
) {
      CLC_MVOP_prep prep;
      fn_CLC_MVOP_prep_inplace(
            prep,
            out_mat,
            theta_main_vec_ref,
            theta_us_vec_ref,
            y_ref,
            Model_args_as_cpp_struct,
            n_chunks_multiplier
      );
      return prep;
}


//// -------------------------------------------------------------------------------------
//// Chain accumulated likelihood + prior gradients back to theta_main and finalise out_mat
//// -------------------------------------------------------------------------------------
inline void fn_CLC_MVOP_assemble(
      Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat,
      const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
      CLC_MVOP_prep &prep,
      const Model_fn_args_struct &Model_args_as_cpp_struct,
      const double log_jac_u,
      const std::vector<Eigen::Matrix<double, -1, -1>> &beta_4_grad_lik,
      const std::vector<Eigen::Matrix<double, -1, -1>> &U_Omega_grad_array,
      const std::vector<Eigen::Matrix<double, -1, -1>> &cutpoint_grad_4_lik,
      const Eigen::Matrix<double, -1, -1> &d_lp_d_log_prev_subject
) {
      const int n_obs = prep.n_obs;
      const int n_tests = prep.n_tests;
      const int n_class = prep.n_class;
      const int n_params = prep.n_params;

      double log_lik_total = 0.0;
      if (!prep.prior_only) {
            log_lik_total = out_mat.segment(
                  1 + n_params,
                  n_obs
            ).sum();
      }
      out_mat(0) = log_lik_total + log_jac_u + prep.log_prob_prior_and_jac;

      if (prep.grad_main_buf.rows() != prep.n_params_main) {
            prep.grad_main_buf.resize(prep.n_params_main);
      }
      Eigen::Matrix<double, -1, 1> &grad_main = prep.grad_main_buf;
      grad_main = prep.grad_main_prior_and_jac;

      //// -------------------------------------------------------------------------------------
      //// 1. Correlations: pure-double analytic reverse (prior + Jacobian + likelihood VJP)
      //// -------------------------------------------------------------------------------------
      const bool include_corr_lik = (!prep.prior_only && !prep.CI);
      fn_CLC_MVOP_corr_backward_double(
            prep,
            Model_args_as_cpp_struct,
            U_Omega_grad_array,
            include_corr_lik,
            grad_main
      );

      //// -------------------------------------------------------------------------------------
      //// 2. Per-class regression coefficients beta_vec
      //// -------------------------------------------------------------------------------------
      int beta_cursor = 0;
      for (int class_index = 0; class_index < n_class; class_index++) {
            for (int test_index = 0; test_index < n_tests; test_index++) {
                  const int n_covs_ct = (*prep.n_covs_per_outcome)(class_index, test_index);
                  const bool is_perf = ((*prep.is_perfect_ref)(test_index) == 1);
                  for (int cov_index = 0; cov_index < n_covs_ct; cov_index++) {
                        if (!is_perf) {
                              grad_main(prep.off_beta_vec + beta_cursor) +=
                                    beta_4_grad_lik[class_index](cov_index, test_index)
                                    + prep.beta_grad_4_from_prior[class_index](cov_index, test_index);
                        }
                        beta_cursor++;
                  }
            }
      }

      //// -------------------------------------------------------------------------------------
      //// 3. Per-class cutpoints: C -> C_raw (log-difference) -> C_unc (tanh lb_ub)
      //// -------------------------------------------------------------------------------------
      if (prep.n_ordinal_tests > 0) {
            const double C_range = prep.C_raw_upper - prep.C_raw_lower;
            for (int class_index = 0; class_index < n_class; class_index++) {
                  const int class_base = prep.off_C_unc + class_index * prep.n_total_C_per_class;
                  for (int ord_index = 0; ord_index < prep.n_ordinal_tests; ord_index++) {
                        const int n_thr_test = (*prep.n_thr_per_ord_test)(ord_index);
                        const int ord_base = class_base + prep.ord_start_index(ord_index);
                        double suffix_sum = 0.0;
                        for (int thr_index = n_thr_test - 1; thr_index >= 0; thr_index--) {
                              suffix_sum += cutpoint_grad_4_lik[class_index](thr_index, ord_index)
                                            + prep.cutpoint_grad_4_from_prior[class_index](thr_index, ord_index);
                              const int cutpoint_index = ord_base + thr_index - prep.off_C_unc;
                              const double d_C_raw = (thr_index == 0)
                                    ? suffix_sum
                                    : (suffix_sum * prep.exp_of_C_raw_buf(cutpoint_index));
                              const double tanh_val = prep.tanh_of_C_unc_buf(cutpoint_index);
                              const double d_C_raw_du = 0.5 * C_range * (1.0 - tanh_val * tanh_val);
                              grad_main(ord_base + thr_index) += d_C_raw * d_C_raw_du;
                        }
                  }
            }
      }

      //// -------------------------------------------------------------------------------------
      //// 4. Prevalence chain rule from d_lp_d_log_prev (likelihood + population Se/Sp prior)
      //// -------------------------------------------------------------------------------------
      if ((prep.d_lp_d_log_prev_total_buf.rows() != n_obs)
          || (prep.d_lp_d_log_prev_total_buf.cols() != n_class)) {
            prep.d_lp_d_log_prev_total_buf.resize(
                  n_obs,
                  n_class
            );
      }
      Eigen::Matrix<double, -1, -1> &d_lp_d_log_prev_total = prep.d_lp_d_log_prev_total_buf;
      d_lp_d_log_prev_total = d_lp_d_log_prev_subject;
      if (!prep.exclude_priors && prep.use_pop_Se_Sp_prior && (prep.n_ordinal_tests > 0)) {
            if (prep.std_index_is_contiguous) {
                  d_lp_d_log_prev_total.topRows(prep.n_standardisation) +=
                        prep.d_lp_d_log_prev_std_from_pop_prior;
            } else {
                  for (int prof_index = 0; prof_index < prep.n_standardisation; prof_index++) {
                        const int subj_idx = (*prep.std_index)(prof_index);
                        for (int class_index = 0; class_index < n_class; class_index++) {
                              d_lp_d_log_prev_total(subj_idx, class_index) +=
                                    prep.d_lp_d_log_prev_std_from_pop_prior(prof_index, class_index);
                        }
                  }
            }
      }

      if (d_lp_d_log_prev_total.cwiseAbs().maxCoeff() > 0.0) {
            if (prep.d_eta_A_vec_buf.rows() != n_obs) {
                  prep.d_eta_A_vec_buf.resize(n_obs);
                  prep.d_eta_B_vec_buf.resize(n_obs);
            }
            Eigen::Matrix<double, -1, 1> &d_eta_A_vec = prep.d_eta_A_vec_buf;
            Eigen::Matrix<double, -1, 1> &d_eta_B_vec = prep.d_eta_B_vec_buf;
            double d_log_OR_lik = 0.0;
            for (int row_index = 0; row_index < n_obs; row_index++) {
                  const double w00 = d_lp_d_log_prev_total(row_index, 0);
                  const double w10 = d_lp_d_log_prev_total(row_index, 1);
                  const double w01 = d_lp_d_log_prev_total(row_index, 2);
                  const double w11 = d_lp_d_log_prev_total(row_index, 3);
                  const double pA_val = prep.pA_all(row_index);
                  const double pB_val = prep.pB_all(row_index);
                  const double q_val = prep.q_all(row_index);
                  const double d_eta_A = (w10 + w11) * (1.0 - pA_val) - (w00 + w01) * pA_val;
                  const double d_eta_B_only = w01 * (1.0 - pB_val) - w00 * pB_val;
                  const double d_eta_q = w11 * (1.0 - q_val) - w10 * q_val;
                  d_eta_A_vec(row_index) = d_eta_A;
                  d_eta_B_vec(row_index) = d_eta_B_only + d_eta_q;
                  d_log_OR_lik += d_eta_q;
            }
            grad_main(prep.off_log_OR) += d_log_OR_lik;
            grad_main(prep.off_mu_logit_A) += d_eta_A_vec.sum();
            grad_main(prep.off_mu_logit_B) += d_eta_B_vec.sum();
            for (int row_index = 0; row_index < n_obs; row_index++) {
                  const int pop_index = (*prep.pop_ind)(row_index);
                  grad_main(prep.off_z_logit_A + pop_index) += prep.sigma_A * d_eta_A_vec(row_index);
                  grad_main(prep.off_sigma_logit_A) += prep.sigma_A * prep.z_A(pop_index) * d_eta_A_vec(row_index);
                  grad_main(prep.off_z_logit_B + pop_index) += prep.sigma_B * d_eta_B_vec(row_index);
                  grad_main(prep.off_sigma_logit_B) += prep.sigma_B * prep.z_B(pop_index) * d_eta_B_vec(row_index);
            }
            if (prep.n_prevalence_covariates > 0) {
                  grad_main.segment(
                        prep.off_beta_prev_A,
                        prep.n_prevalence_covariates
                  ).noalias() += prep.X_prev->leftCols(prep.n_prevalence_covariates).transpose() * d_eta_A_vec;
                  grad_main.segment(
                        prep.off_beta_prev_B,
                        prep.n_prevalence_covariates
                  ).noalias() += prep.X_prev->leftCols(prep.n_prevalence_covariates).transpose() * d_eta_B_vec;
            }
      }

      //// -------------------------------------------------------------------------------------
      //// 5. Pinkney partially-centered population logits and logit Pr(B | A) (as the .stan file): grad_main holds the
      ////    gradient w.r.t. the non-centered z (at off_z_logit_*) and log_OR (at off_log_OR); the sampled values are
      ////    z_pcp and mu_logit_B_given_A. z = (z_pcp - mu (1 - w)) / tau_wt with w = w(mu, sigma), so
      ////    dz / dz_pcp = 1 / tau_wt, dz / dmu and dz / dlog sigma below, plus the gradient of the -sum log tau_wt Jacobian;
      ////    log_OR = mu_logit_B_given_A - mu_logit_B, so d / dmu_logit_B gets -d / dlog_OR.
      //// -------------------------------------------------------------------------------------
      for (int class_AB = 0; class_AB < 2; class_AB++) {
            const int off_z = (class_AB == 0) ? prep.off_z_logit_A : prep.off_z_logit_B;
            const int off_mu = (class_AB == 0) ? prep.off_mu_logit_A : prep.off_mu_logit_B;
            const int off_ls = (class_AB == 0) ? prep.off_sigma_logit_A : prep.off_sigma_logit_B;
            const double mu_val = (class_AB == 0) ? prep.mu_A : prep.mu_B;
            const double s_val = (class_AB == 0) ? prep.sigma_A : prep.sigma_B;
            const double p_val = (class_AB == 0) ? prep.pcp_p_A : prep.pcp_p_B;
            const double pq = p_val * (1.0 - p_val);
            for (int pop_index = 0; pop_index < prep.n_pops; pop_index++) {
                  const double n_of_pop = prep.pcp_n_of_pop(pop_index);
                  const double w_val = prep.pcp_w(class_AB * prep.n_pops + pop_index);
                  const double dw_dmu = -w_val * w_val * n_of_pop * s_val * s_val * pq * (1.0 - 2.0 * p_val);
                  const double dw_ds = -w_val * w_val * 2.0 * n_of_pop * pq * s_val;
                  const double tau_wt = prep.pcp_tau_wt(class_AB * prep.n_pops + pop_index);
                  const double dtau_dmu = -(s_val - 1.0) * dw_dmu;
                  const double dtau_ds = 1.0 - w_val - (s_val - 1.0) * dw_ds;
                  const double z_nc = (class_AB == 0) ? prep.z_A(pop_index) : prep.z_B(pop_index);
                  const double dz_dmu = (-(1.0 - w_val) + mu_val * dw_dmu - z_nc * dtau_dmu) / tau_wt;
                  const double dz_dls = s_val * (mu_val * dw_ds - z_nc * dtau_ds) / tau_wt;
                  const double g_z_nc = grad_main(off_z + pop_index);
                  grad_main(off_mu) += g_z_nc * dz_dmu - dtau_dmu / tau_wt;
                  grad_main(off_ls) += g_z_nc * dz_dls - s_val * dtau_ds / tau_wt;
                  grad_main(off_z + pop_index) = g_z_nc / tau_wt;
            }
      }
      if (prep.use_pop_prev_prior == 3) {
            //// log_OR = exp(u) does not depend on mu_logit_B: d / du = log_OR * d / dlog_OR, plus 1 from the log-Jacobian u
            grad_main(prep.off_log_OR) = prep.log_OR * grad_main(prep.off_log_OR) + 1.0;
      } else {
            grad_main(prep.off_mu_logit_B) -= grad_main(prep.off_log_OR);
      }

      out_mat.segment(
            1 + prep.n_us,
            prep.n_params_main
      ) = grad_main;
}
