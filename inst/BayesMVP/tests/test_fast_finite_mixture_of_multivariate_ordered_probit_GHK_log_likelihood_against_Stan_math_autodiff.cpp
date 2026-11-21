//
// ==== test_fast_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood_against_Stan_math_autodiff.cpp ===
//
// ---- The value and gradient of BayesMVP's nico_LC_MVP_MVOP_GHK_log_lik
//      (stan_external_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood.hpp) against the same
//      likelihood written with Stan math's own functions and differentiated by Stan math's autodiff, at random
//      points of random problems: C = 1, 2 and 4 classes; binary-only and mixed binary / ordinal tests; with and
//      without missing results, interval-censored results and covariates (coefficient sets shared by classes, as
//      in the four-class LC-MVOP, and covariates by set, as in the two-class LC-MVOP); a GHK order that is not
//      the data order; Phi_type 1 and 2; u = Phi(u_raw) and u = (1 + tanh(u_raw)) / 2; ordinary points and
//      points with large linear predictors (rows in the log-scale tails); problems with narrow intervals.
//      Prints the largest relative difference of the value, and the largest difference of the gradient relative
//      to the largest |gradient| of the problem, for each problem.
//
// Usage (CPUs 180-191, lowest priority):
//      bash run_test_fast_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood.sh
//

#include "stan_external_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood.hpp"

#include <cstdio>
#include <random>

using stan::math::var;

//// ---- the reference: Stan math's own functions, one subject, class and step at a time
struct problem_t {
    int M, T, C, Phi_type, nuisance_transformation, n_sets_B, n_sets_C, n_sets_X;
    std::vector<int> test_is_binary, n_categories, first_cutpoint, order;
    std::vector<std::vector<int>> category, upper_category, coefficient_set, cutpoint_set, n_covariates;
    std::vector<std::vector<Eigen::MatrixXd>> covariates;
    double overflow_threshold, underflow_threshold;
};

//// (log Phi(x) = log(erfc(-x / sqrt(2)) / 2), whose autodiff derivative is exact; Stan math's own
////  std_normal_lcdf takes its derivative from the approximation 7.1.26 of Abramowitz and Stegun, accurate to
////  about 1.5e-7 only)
var log_F_reference(const var& x, const int Phi_type) {
    if (Phi_type == 1) return stan::math::log(0.5 * stan::math::erfc(-x * 0.70710678118654752440));
    return stan::math::log_inv_logit(0.07056 * x * x * x + 1.5976 * x);
}

var inverse_of_F_from_logs_reference(const var& log_F, const var& log_1m_F, const int Phi_type) {
    if (Phi_type == 1) {
        if (log_F.val() < log_1m_F.val()) return stan::math::std_normal_log_qf(log_F);
        return -stan::math::std_normal_log_qf(log_1m_F);
    }
    const var logit = log_F - log_1m_F;
    return 5.494448514153059 * stan::math::sinh(stan::math::asinh(0.34176618822627863 * logit) / 3.0);
}

var reference_log_likelihood(const problem_t& p, const std::vector<Eigen::Matrix<var, 1, -1>>& u_raw,
                             const Eigen::Matrix<var, -1, -1>& intercept,
                             const std::vector<Eigen::Matrix<var, -1, -1>>& coefficients,
                             const std::vector<Eigen::Matrix<var, -1, 1>>& cutpoints,
                             const std::vector<Eigen::Matrix<var, -1, -1>>& L_Omega,
                             const Eigen::Matrix<var, -1, -1>& log_class_probability) {
    var total = 0.0;
    const double infinity = std::numeric_limits<double>::infinity();
    for (int m = 0; m < p.M; ++m) {
        std::vector<var> log_u(p.T), log_1m_u(p.T);
        for (int j = 0; j < p.T; ++j) {
            const var x = u_raw[m](j);
            if (p.nuisance_transformation == 1) {
                log_u[j] = log_F_reference(x, 1);
                log_1m_u[j] = log_F_reference(-x, 1);
                total += -0.5 * x * x - 0.5 * std::log(2.0 * stan::math::pi());
            } else {
                log_u[j] = stan::math::log_inv_logit(2.0 * x);
                log_1m_u[j] = stan::math::log_inv_logit(-2.0 * x);
                total += std::log(2.0) + log_u[j] + log_1m_u[j];
            }
        }
        std::vector<var> lp(p.C);
        for (int c = 0; c < p.C; ++c) {
            lp[c] = log_class_probability(m, c);
            std::vector<var> Z(p.T);
            for (int j = 0; j < p.T; ++j) {
                const int t = p.order[j] - 1;
                if (p.category[m][t] < 0) {
                    Z[j] = inverse_of_F_from_logs_reference(log_u[j], log_1m_u[j], p.Phi_type);
                    continue;
                }
                var eta = intercept(c, t);
                if (p.n_sets_X > 0) {
                    const int s = p.coefficient_set[c][t] - 1;
                    const int s_X = (p.n_sets_X == 1) ? 0 : s;
                    for (int q = 0; q < p.n_covariates[s_X][t]; ++q) {
                        eta += p.covariates[s_X][t](m, q) * coefficients[s](q, t);
                    }
                }
                for (int k = 0; k < j; ++k) eta += L_Omega[c](j, k) * Z[k];
                const var s = 1.0 / L_Omega[c](j, j);
                bool a_finite = true, b_finite = true;
                var a = 0.0, b = 0.0;
                if (p.test_is_binary[t] == 1) {
                    if (p.category[m][t] == 1) { a = -eta * s; b_finite = false; }
                    else                       { b = -eta * s; a_finite = false; }
                } else {
                    const int k = p.category[m][t], k_upper = p.upper_category[m][t], K = p.n_categories[t];
                    const int first = p.first_cutpoint[t] - 1;
                    const auto& C_vec = cutpoints[p.cutpoint_set[c][t] - 1];
                    if (k == 1) a_finite = false; else a = (C_vec(first + k - 2) - eta) * s;
                    if (k_upper == K) b_finite = false; else b = (C_vec(first + k_upper - 1) - eta) * s;
                }
                const double a_value = a_finite ? a.val() : -infinity;
                const double b_value = b_finite ? b.val() : infinity;
                var log_F_a = a_finite ? log_F_reference(a, p.Phi_type) : var(-infinity);
                var log_F_b = b_finite ? log_F_reference(b, p.Phi_type) : var(0.0);
                var log_1m_F_a = a_finite ? log_F_reference(-a, p.Phi_type) : var(0.0);
                var log_1m_F_b = b_finite ? log_F_reference(-b, p.Phi_type) : var(-infinity);
                var log_mass, log_F_Z, log_1m_F_Z;
                if (b_value < 0.0) {
                    log_mass = a_finite ? stan::math::log_diff_exp(log_F_b, log_F_a) : log_F_b;
                    log_F_Z = a_finite ? stan::math::log_sum_exp(log_1m_u[j] + log_F_a, log_u[j] + log_F_b)
                                       : log_u[j] + log_F_b;
                    log_1m_F_Z = stan::math::log1m_exp(log_F_Z);
                } else if (a_value > 0.0) {
                    log_mass = b_finite ? stan::math::log_diff_exp(log_1m_F_a, log_1m_F_b) : log_1m_F_a;
                    log_1m_F_Z = b_finite ? stan::math::log_sum_exp(log_1m_u[j] + log_1m_F_a,
                                                                    log_u[j] + log_1m_F_b)
                                          : log_1m_u[j] + log_1m_F_a;
                    log_F_Z = stan::math::log1m_exp(log_1m_F_Z);
                } else {
                    //// (a <= 0 <= b: neither F(a) nor 1 - F(b) is near 1)
                    const var F_a = a_finite ? stan::math::exp(log_F_a) : var(0.0);
                    const var one_minus_F_b = b_finite ? stan::math::exp(log_1m_F_b) : var(0.0);
                    log_mass = stan::math::log1m(F_a + one_minus_F_b);
                    log_F_Z = a_finite ? stan::math::log_sum_exp(log_1m_u[j] + log_F_a, log_u[j] + log_F_b)
                                       : log_u[j] + log_F_b;
                    log_1m_F_Z = b_finite ? stan::math::log_sum_exp(log_1m_u[j] + log_1m_F_a,
                                                                    log_u[j] + log_1m_F_b)
                                          : log_1m_u[j] + log_1m_F_a;
                }
                Z[j] = inverse_of_F_from_logs_reference(log_F_Z, log_1m_F_Z, p.Phi_type);
                lp[c] += log_mass;
            }
        }
        total += stan::math::log_sum_exp(lp);
    }
    return total;
}

int main() {
    std::printf("SIMD lanes of BayesMVP's kernels in this build: %d\n", BMVP_LANES);
    std::mt19937 generator(1);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    int problem_index = 0;
    double worst_value = 0.0, worst_gradient = 0.0;
    for (int C : {1, 2, 4}) {
    for (int Phi_type : {1, 2}) {
    for (int nuisance_transformation : {1, 2}) {
    for (int kind_of_tests : {0, 1}) {             // 0: binary only; 1: binary and ordinal
    for (int with_missing_censoring_and_covariates : {0, 1}) {
    for (double scale_of_linear_predictor : {1.0, 6.0}) {   // 6: many rows in the log-scale tails
    for (int narrow : {0, 1}) {
        if (narrow == 1 && (kind_of_tests == 0 || scale_of_linear_predictor > 1.0)) continue;
        problem_t p;
        p.M = 37; p.T = 4; p.C = C; p.Phi_type = Phi_type; p.nuisance_transformation = nuisance_transformation;
        p.overflow_threshold = 2.5; p.underflow_threshold = -2.5;
        p.test_is_binary = (kind_of_tests == 0) ? std::vector<int>{1, 1, 1, 1} : std::vector<int>{0, 1, 0, 1};
        p.n_categories.assign(p.T, 2);
        p.first_cutpoint.assign(p.T, 1);
        int n_cutpoints = 0;
        for (int t = 0; t < p.T; ++t) {
            if (p.test_is_binary[t] == 1) continue;
            p.n_categories[t] = (t == 0) ? 5 : 3;
            p.first_cutpoint[t] = n_cutpoints + 1;
            n_cutpoints += p.n_categories[t] - 1;
        }
        p.order = {3, 1, 4, 2};
        p.n_sets_C = 2;
        p.n_sets_B = 2;
        p.n_sets_X = with_missing_censoring_and_covariates ? ((C == 2) ? 2 : 1) : 0;
        p.coefficient_set.assign(C, std::vector<int>(p.T, 1));
        p.cutpoint_set.assign(C, std::vector<int>(p.T, 1));
        for (int c = 0; c < C; ++c) for (int t = 0; t < p.T; ++t) {
            p.coefficient_set[c][t] = (c + t) % 2 + 1;
            p.cutpoint_set[c][t] = (c * 3 + t) % 2 + 1;
        }
        const int P = 3;
        p.covariates.assign(p.n_sets_X, std::vector<Eigen::MatrixXd>(p.T));
        p.n_covariates.assign(p.n_sets_X, std::vector<int>(p.T, 0));
        for (int s = 0; s < p.n_sets_X; ++s) for (int t = 0; t < p.T; ++t) {
            p.covariates[s][t] = Eigen::MatrixXd(p.M, P);
            for (int m = 0; m < p.M; ++m) for (int q = 0; q < P; ++q) p.covariates[s][t](m, q) = normal(generator);
            p.n_covariates[s][t] = 1 + (s + t) % P;
        }
        p.category.assign(p.M, std::vector<int>(p.T, 0));
        p.upper_category = p.category;
        for (int m = 0; m < p.M; ++m) for (int t = 0; t < p.T; ++t) {
            const int K = p.n_categories[t];
            int k = (p.test_is_binary[t] == 1) ? static_cast<int>(uniform(generator) < 0.5)
                                               : 1 + static_cast<int>(uniform(generator) * K);
            int k_upper = k;
            if (with_missing_censoring_and_covariates && p.test_is_binary[t] != 1 && uniform(generator) < 0.2) {
                k_upper = std::min(K, k + 1);
            }
            if (with_missing_censoring_and_covariates && uniform(generator) < 0.25) k = -1;
            //// (narrow: half of the first ordinal test's results in its narrow category 3)
            if (narrow == 1 && t == 0 && uniform(generator) < 0.5) { k = 3; k_upper = 3; }
            p.category[m][t] = k;
            p.upper_category[m][t] = k_upper;
        }
        //// ---- the parameters
        std::vector<Eigen::Matrix<var, 1, -1>> u_raw(p.M, Eigen::Matrix<var, 1, -1>(p.T));
        for (int m = 0; m < p.M; ++m) for (int j = 0; j < p.T; ++j) u_raw[m](j) = 1.3 * normal(generator);
        Eigen::Matrix<var, -1, -1> intercept(C, p.T);
        for (int c = 0; c < C; ++c) for (int t = 0; t < p.T; ++t) {
            intercept(c, t) = scale_of_linear_predictor * normal(generator);
        }
        std::vector<Eigen::Matrix<var, -1, -1>> coefficients(p.n_sets_B, Eigen::Matrix<var, -1, -1>(P, p.T));
        for (int s = 0; s < p.n_sets_B; ++s) for (int q = 0; q < P; ++q) for (int t = 0; t < p.T; ++t) {
            coefficients[s](q, t) = 0.5 * normal(generator);
        }
        std::vector<Eigen::Matrix<var, -1, 1>> cutpoints(p.n_sets_C, Eigen::Matrix<var, -1, 1>(n_cutpoints));
        for (int s = 0; s < p.n_sets_C; ++s) {
            for (int t = 0; t < p.T; ++t) {
                if (p.test_is_binary[t] == 1) continue;
                double value = -1.0 + 0.3 * normal(generator);
                for (int k = 0; k < p.n_categories[t] - 1; ++k) {
                    cutpoints[s](p.first_cutpoint[t] - 1 + k) = value;
                    //// (narrow: category 3 of the first ordinal test, between cutpoints 2 and 3, of width 1e-7)
                    value += (narrow == 1 && t == 0 && k == 1) ? 1e-7 : 0.5 + uniform(generator);
                }
            }
        }
        std::vector<Eigen::Matrix<var, -1, -1>> L_Omega(C, Eigen::Matrix<var, -1, -1>(p.T, p.T));
        for (int c = 0; c < C; ++c) {
            Eigen::MatrixXd A(p.T, p.T);
            for (int i = 0; i < p.T * p.T; ++i) A(i) = normal(generator);
            Eigen::MatrixXd Omega = A * A.transpose() + 0.5 * Eigen::MatrixXd::Identity(p.T, p.T);
            const Eigen::VectorXd d = Omega.diagonal().cwiseSqrt().cwiseInverse();
            Omega = d.asDiagonal() * Omega * d.asDiagonal();
            const Eigen::MatrixXd L = Omega.llt().matrixL();
            for (int i = 0; i < p.T; ++i) for (int k = 0; k < p.T; ++k) L_Omega[c](i, k) = L(i, k);
        }
        Eigen::Matrix<var, -1, -1> log_class_probability(p.M, C);
        for (int m = 0; m < p.M; ++m) {
            Eigen::VectorXd w(C);
            for (int c = 0; c < C; ++c) w(c) = normal(generator);
            const double log_sum = stan::math::log_sum_exp(w);
            for (int c = 0; c < C; ++c) log_class_probability(m, c) = w(c) - log_sum;
        }
        //// ---- the gradient of each: every parameter in one vector
        std::vector<var> parameters;
        for (auto& row : u_raw) for (int j = 0; j < p.T; ++j) parameters.push_back(row(j));
        for (int i = 0; i < intercept.size(); ++i) parameters.push_back(intercept(i));
        for (auto& x : coefficients) for (int i = 0; i < x.size(); ++i) parameters.push_back(x(i));
        for (auto& x : cutpoints) for (int i = 0; i < x.size(); ++i) parameters.push_back(x(i));
        for (auto& x : L_Omega) for (int i = 0; i < x.size(); ++i) parameters.push_back(x(i));
        for (int i = 0; i < log_class_probability.size(); ++i) parameters.push_back(log_class_probability(i));
        var fused = nico_LC_MVP_MVOP_GHK_log_lik(
              u_raw, nuisance_transformation, intercept, coefficients, p.coefficient_set, p.covariates,
              p.n_covariates, cutpoints, p.cutpoint_set, p.first_cutpoint, L_Omega, log_class_probability,
              p.category, p.upper_category, p.test_is_binary, p.n_categories, p.order, Phi_type,
              p.overflow_threshold, p.underflow_threshold);
        std::vector<double> gradient_of_fused;
        fused.grad(parameters, gradient_of_fused);
        stan::math::set_zero_all_adjoints();
        var reference = reference_log_likelihood(p, u_raw, intercept, coefficients, cutpoints, L_Omega,
                                                 log_class_probability);
        std::vector<double> gradient_of_reference;
        reference.grad(parameters, gradient_of_reference);
        double largest_gradient = 0.0, largest_difference = 0.0;
        for (std::size_t i = 0; i < parameters.size(); ++i) {
            largest_gradient = std::max(largest_gradient, std::fabs(gradient_of_reference[i]));
            largest_difference = std::max(largest_difference,
                                          std::fabs(gradient_of_fused[i] - gradient_of_reference[i]));
        }
        const double relative_value = std::fabs(fused.val() - reference.val()) / std::fabs(reference.val());
        const double relative_gradient = largest_difference / largest_gradient;
        stan::math::recover_memory();
        if (narrow == 0) {
            worst_value = std::max(worst_value, relative_value);
            worst_gradient = std::max(worst_gradient, relative_gradient);
        }
        std::printf("problem %2d: C %d Phi_type %d u_%s %s %s scale %.0f%s: value %.12g reference %.12g "
                    "relative %.2e; gradient max|diff| / max|grad| %.2e (max|grad| %.3g)\n",
                    problem_index, C, Phi_type, nuisance_transformation == 1 ? "Phi " : "tanh",
                    kind_of_tests == 0 ? "binary" : "mixed ",
                    with_missing_censoring_and_covariates ? "missing+censored+covariates" : "complete",
                    scale_of_linear_predictor, narrow ? " narrow" : "", fused.val(), reference.val(),
                    relative_value, relative_gradient, largest_gradient);
        problem_index += 1;
    }}}}}}}
    std::printf("largest relative difference of the value (problems without narrow intervals): %.2e; "
                "of the gradient: %.2e\n", worst_value, worst_gradient);
    return 0;
}























