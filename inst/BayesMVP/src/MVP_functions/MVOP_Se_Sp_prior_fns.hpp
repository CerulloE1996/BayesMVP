#pragma once

#ifndef MVOP_SE_SP_PRIOR_FNS_HPP
#define MVOP_SE_SP_PRIOR_FNS_HPP


#include <stan/math/rev.hpp>
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>




////
//// ---- Se/Sp prior data of every class c and ordinal test t_ord: the chosen cutoffs (0-based, -1 = unused),
//// ---- Dirichlet A (type 1), Beta a / b (types 2-3), link-scale mean / SD (type 4) and alpha_seg (alpha summed
//// ---- between chosen cutoffs):
////
struct Se_Sp_prior_args_struct {

        std::vector<Eigen::Matrix<int, -1, 1>> n_chosen;
        std::vector<std::vector<std::vector<int>>> chosen;
        std::vector<std::vector<Eigen::Matrix<double, -1, 1>>> A;
        std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> beta_ab;
        std::vector<std::vector<Eigen::Matrix<double, -1, -1>>> link_mean_SD;
        std::vector<std::vector<Eigen::Matrix<double, -1, 1>>> alpha_seg;

};




inline Se_Sp_prior_args_struct fn_Se_Sp_prior_args(const std::vector<Eigen::Matrix<int, -1, -1>> &chosen_cutoffs,
                                                   const std::vector<Eigen::Matrix<double, -1, -1>> &A_user,
                                                   const std::vector<Eigen::Matrix<double, -1, -1>> &beta_a,
                                                   const std::vector<Eigen::Matrix<double, -1, -1>> &beta_b,
                                                   const std::vector<Eigen::Matrix<double, -1, -1>> &link_mean,
                                                   const std::vector<Eigen::Matrix<double, -1, -1>> &link_SD,
                                                   const std::vector<Eigen::Matrix<double, -1, -1>> &alpha,
                                                   const Eigen::Matrix<int, -1, 1> &n_thr_per_ord_test,
                                                   const int prior_type) {


        const int n_class = static_cast<int>(chosen_cutoffs.size());
        const int n_ordinal_tests = static_cast<int>(n_thr_per_ord_test.rows());
        Se_Sp_prior_args_struct S;
        S.n_chosen.resize(n_class);
        S.chosen.resize(n_class);
        S.A.resize(n_class);
        S.beta_ab.resize(n_class);
        S.link_mean_SD.resize(n_class);
        S.alpha_seg.resize(n_class);
        ////
        for (int c = 0; c < n_class; c++) {

             S.n_chosen[c] = Eigen::Matrix<int, -1, 1>::Zero(n_ordinal_tests);
             S.chosen[c].resize(n_ordinal_tests);
             S.A[c].resize(n_ordinal_tests);
             S.beta_ab[c].resize(n_ordinal_tests);
             S.link_mean_SD[c].resize(n_ordinal_tests);
             S.alpha_seg[c].resize(n_ordinal_tests);
             for (int t_ord = 0; t_ord < n_ordinal_tests; t_ord++) {

                  const int n_cat_t = n_thr_per_ord_test(t_ord) + 1;
                  int m = 0;
                  for (int j = 0; j < chosen_cutoffs[c].rows(); j++) {
                       if (chosen_cutoffs[c](j, t_ord) >= 0) {
                            S.chosen[c][t_ord].push_back(chosen_cutoffs[c](j, t_ord));
                            ++m;
                       }
                  }
                  S.n_chosen[c](t_ord) = m;
                  if (m == 0) continue;
                  S.A[c][t_ord] = A_user[c].col(t_ord).head(m + 1);
                  S.beta_ab[c][t_ord].resize(m, 2);
                  S.beta_ab[c][t_ord].col(0) = beta_a[c].col(t_ord).head(m);
                  S.beta_ab[c][t_ord].col(1) = beta_b[c].col(t_ord).head(m);
                  ////
                  //// ---- mean / SD of link(Sp) (c = 0) and link(Se) (c = 1):
                  ////
                  S.link_mean_SD[c][t_ord].resize(m, 2);
                  S.link_mean_SD[c][t_ord].col(0) = link_mean[c].col(t_ord).head(m);
                  S.link_mean_SD[c][t_ord].col(1) = link_SD[c].col(t_ord).head(m);
                  ////
                  //// ---- P is 1 - Se at c = 1: the link mean is negated, and type 2's Beta on Se swapped:
                  ////
                  if (c == 1) {
                       S.link_mean_SD[c][t_ord].col(0) = - S.link_mean_SD[c][t_ord].col(0);
                       if (prior_type == 2) {
                            S.beta_ab[c][t_ord].col(0) = beta_b[c].col(t_ord).head(m);
                            S.beta_ab[c][t_ord].col(1) = beta_a[c].col(t_ord).head(m);
                       }
                  }
                  ////
                  //// ---- sums of alpha over the segments between chosen cutoffs:
                  ////
                  S.alpha_seg[c][t_ord] = Eigen::Matrix<double, -1, 1>::Zero(m + 1);
                  int k_prev = 0;
                  for (int j = 0; j < m; j++) {
                       const int k_j = S.chosen[c][t_ord][j] + 1;
                       if (k_j <= k_prev) throw std::domain_error("chosen cutoffs must rise");
                       S.alpha_seg[c][t_ord](j) = alpha[c].col(t_ord).segment(k_prev, k_j - k_prev).sum();
                       k_prev = k_j;
                  }
                  S.alpha_seg[c][t_ord](m) = alpha[c].col(t_ord).segment(k_prev, n_cat_t - k_prev).sum();

             }

        }
        return S;


}



////
//// ---- Induced Dirichlet with the Dirichlet of the segment sums (between chosen cutoffs) swapped for
//// ---- a prior on baseline Sp (c = 0) or 1 - Se (c = 1) at the chosen cutoffs (probit link):
////
inline stan::math::var
fn_induced_dirichlet_with_Se_Sp_prior_lpdf(const Eigen::Matrix<stan::math::var, -1, 1> &C_centred,
                                           const Eigen::Matrix<stan::math::var, -1, 1> &C_baseline,
                                           const Eigen::Matrix<double, -1, 1> &alpha,
                                           const Eigen::Matrix<double, -1, 1> &alpha_seg,
                                           const std::vector<int> &chosen,
                                           const int prior_type,
                                           const Eigen::Matrix<double, -1, 1> &A_user,
                                           const Eigen::Matrix<double, -1, -1> &beta_ab,
                                           const Eigen::Matrix<double, -1, -1> &link_mean_SD) {


        const int n_thr = C_centred.rows();
        const int m = static_cast<int>(chosen.size());
        ////
        Eigen::Matrix<stan::math::var, -1, 1> P(n_thr);
        for (int k = 0; k < n_thr; k++) P(k) = stan::math::Phi(C_centred(k));
        Eigen::Matrix<stan::math::var, -1, 1> p_ord(n_thr + 1);
        p_ord(0) = P(0);
        for (int k = 1; k < n_thr; k++) p_ord(k) = P(k) - P(k - 1);
        p_ord(n_thr) = 1.0 - P(n_thr - 1);
        ////
        Eigen::Matrix<stan::math::var, -1, 1> P_chosen(m);
        Eigen::Matrix<stan::math::var, -1, 1> P_anchor(m);
        for (int j = 0; j < m; j++) {
             P_chosen(j) = stan::math::Phi(C_baseline(chosen[j]));
             P_anchor(j) = P(chosen[j]);
        }
        Eigen::Matrix<stan::math::var, -1, 1> q(m + 1);
        Eigen::Matrix<stan::math::var, -1, 1> q_anchor(m + 1);
        q(0) = P_chosen(0);
        q_anchor(0) = P_anchor(0);
        for (int j = 1; j < m; j++) {
             q(j) = P_chosen(j) - P_chosen(j - 1);
             q_anchor(j) = P_anchor(j) - P_anchor(j - 1);
        }
        q(m) = 1.0 - P_chosen(m - 1);
        q_anchor(m) = 1.0 - P_anchor(m - 1);
        ////
        //// ---- induced Dirichlet minus the Dirichlet of the segment sums:
        ////
        stan::math::var log_prob = 0.0;
        for (int k = 0; k < n_thr; k++) log_prob += stan::math::std_normal_lpdf(C_centred(k));
        log_prob += stan::math::dirichlet_lpdf(p_ord, alpha);
        log_prob += - stan::math::dirichlet_lpdf(q_anchor, alpha_seg);
        ////
        //// ---- Jacobian between selected anchor and baseline probabilities:
        ////
        for (int j = 0; j < m; j++) {
             const int k = chosen[j];
             log_prob += - stan::math::std_normal_lpdf(C_centred(k));
             if (prior_type != 4) {
                  log_prob += stan::math::std_normal_lpdf(C_baseline(k));
             }
        }
        ////
        //// ---- prior on P at the chosen cutoffs:
        ////
        if (prior_type == 1) {
             //// Dirichlet on the segment sums:
             log_prob += stan::math::dirichlet_lpdf(q, A_user);
        } else if (prior_type == 2) {
             //// Beta on each P, restricted to rising P (normaliser constant):
             for (int j = 0; j < m; j++) log_prob += stan::math::beta_lpdf(P_chosen(j),
                                                                           beta_ab(j, 0),
                                                                           beta_ab(j, 1));
        } else if (prior_type == 4) {
             //// Normal on the link scale, restricted to rising P:
             for (int j = 0; j < m; j++) log_prob += stan::math::normal_lpdf(C_baseline(chosen[j]),
                                                                             link_mean_SD(j, 0),
                                                                             link_mean_SD(j, 1));
        } else {
             //// Beta on each (P_j - P_{j-1}) / (1 - P_{j-1}):
             for (int j = 0; j < m; j++) {
                  stan::math::var P_prev = (j == 0) ? stan::math::var(0.0) : P_chosen(j - 1);
                  log_prob += stan::math::beta_lpdf(q(j) / (1.0 - P_prev),
                                                    beta_ab(j, 0),
                                                    beta_ab(j, 1));
                  log_prob += - stan::math::log1m(P_prev);
             }
        }
        return log_prob;


}


#endif






















