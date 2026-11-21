#include "../inst/include/BayesMVP/stan_external_fused_affine_compositions.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using stan::math::var;
using vector_v = Eigen::Matrix<var, Eigen::Dynamic, 1>;

double largest_value_error = 0.0;
double largest_gradient_error = 0.0;
int number_of_values = 0;
int number_of_gradients = 0;

void record(const double actual, const double reference, const bool gradient = false) {
    if (!std::isfinite(actual) || !std::isfinite(reference)) {
        throw std::runtime_error("Non-finite result in a finite correctness case.");
    }
    const double error = std::abs(actual - reference)
                         / std::max(1.0, std::max(std::abs(actual), std::abs(reference)));
    if (gradient) {
        largest_gradient_error = std::max(largest_gradient_error, error);
        ++number_of_gradients;
    } else {
        largest_value_error = std::max(largest_value_error, error);
        ++number_of_values;
    }
}

void compare(var actual, var reference, std::vector<var> parameters) {
    record(actual.val(), reference.val());
    std::vector<double> actual_gradient;
    std::vector<double> reference_gradient;
    stan::math::set_zero_all_adjoints();
    actual.grad(parameters, actual_gradient);
    stan::math::set_zero_all_adjoints();
    reference.grad(parameters, reference_gradient);
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        record(actual_gradient[i], reference_gradient[i], true);
    }
}

template <typename T_weight, typename T_beta, typename T_scale, typename T_z, typename T_offset>
auto prefix_reference(const T_weight& linear, const T_weight& prefix, const Eigen::MatrixXd& x,
                      const T_beta& beta, const T_scale& scale, const T_z& z, const T_offset& offset) {
    using scalar = stan::return_type_t<T_weight, T_beta, T_scale, T_z, T_offset>;
    Eigen::Matrix<scalar, Eigen::Dynamic, 1> eta = x * beta + scale * z;
    scalar result = 0.0;
    scalar running = offset;
    for (Eigen::Index i = 0; i < z.size(); ++i) {
        running = stan::math::log_sum_exp(running, eta[i]);
        result += linear[i] * eta[i] + prefix[i] * running;
    }
    return result;
}

void prefix_case(const int n, const double offset_value, const bool extreme) {
    Eigen::MatrixXd x(n, 2);
    vector_v beta(2), z(n), linear(n), prefix(n);
    beta << 0.31, -0.19;
    std::vector<var> parameters{beta[0], beta[1]};
    for (int i = 0; i < n; ++i) {
        x(i, 0) = std::sin(0.17 * i);
        x(i, 1) = std::cos(0.23 * i);
        z[i] = extreme ? (i == 0 ? -900.0 : 900.0) : std::sin(0.37 * i);
        linear[i] = 0.3 * std::cos(0.41 * i);
        prefix[i] = -0.7 + 0.2 * std::sin(0.43 * i);
        parameters.push_back(z[i]);
        parameters.push_back(linear[i]);
        parameters.push_back(prefix[i]);
    }
    var scale = extreme ? 1.0 : -0.83;
    var offset = offset_value;
    parameters.push_back(scale);
    if (std::isfinite(offset_value)) parameters.push_back(offset);
    const var actual = nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(
            linear, prefix, x, beta, scale, z, offset);
    const var reference = prefix_reference(linear, prefix, x, beta, scale, z, offset);
    compare(actual, reference, parameters);
    const Eigen::VectorXd linear_data = stan::math::value_of(linear);
    const Eigen::VectorXd prefix_data = stan::math::value_of(prefix);
    compare(nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(
                    linear_data, prefix_data, x, beta, scale, z, offset),
            prefix_reference(linear_data, prefix_data, x, beta, scale, z, offset), parameters);
    const Eigen::VectorXd beta_data = stan::math::value_of(beta);
    const Eigen::VectorXd z_data = stan::math::value_of(z);
    record(nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(
                   linear_data, prefix_data, x, beta_data, scale.val(), z_data, offset_value),
            prefix_reference(linear_data, prefix_data, x, beta_data, scale.val(), z_data, offset_value));
    stan::math::recover_memory();
}

template <bool propto>
void normal_case(const int n) {
    vector_v y(n), mu(n), predictor(n);
    var intercept = 0.27;
    var slope = -0.43;
    std::vector<var> parameters{intercept, slope};
    for (int i = 0; i < n; ++i) {
        y[i] = std::sin(0.21 * i);
        mu[i] = 0.1 * std::cos(0.13 * i);
        predictor[i] = std::cos(0.31 * i);
        parameters.push_back(y[i]);
        parameters.push_back(mu[i]);
        parameters.push_back(predictor[i]);
    }
    vector_v sigma = stan::math::exp(stan::math::add(intercept, slope * predictor));
    compare(nico_normal_affine_log_scale_lpdf<propto>(y, mu, intercept, slope, predictor),
            stan::math::normal_lpdf<propto>(y, mu, sigma), parameters);
    const Eigen::VectorXd y_data = stan::math::value_of(y);
    const Eigen::VectorXd predictor_data = stan::math::value_of(predictor);
    vector_v scales = stan::math::exp(stan::math::add(intercept, slope.val() * predictor_data));
    compare(nico_normal_affine_log_scale_lpdf<propto>(y_data, 0.0, intercept, slope.val(), predictor_data),
            stan::math::normal_lpdf<propto>(y_data, 0.0, scales), parameters);
    const Eigen::VectorXd sigma_data = stan::math::value_of(sigma);
    compare(nico_normal_affine_log_scale_lpdf<propto>(y, 0.0, intercept.val(), slope.val(), predictor_data),
            stan::math::normal_lpdf<propto>(y, 0.0, sigma_data), parameters);
    record(nico_normal_affine_log_scale_lpdf<propto>(y_data, 0.0,
                                                   intercept.val(), slope.val(), predictor_data),
            stan::math::normal_lpdf<propto>(y_data, 0.0, sigma_data));
    stan::math::recover_memory();
}

void alias_case() {
    var shared = 0.23;
    vector_v y = vector_v::Constant(9, shared);
    vector_v scales = stan::math::exp(stan::math::add(shared, shared * y));
    compare(nico_normal_affine_log_scale_lpdf(y, shared, shared, shared, y),
            stan::math::normal_lpdf(y, shared, scales), {shared});
    Eigen::MatrixXd x = Eigen::MatrixXd::Ones(9, 1);
    vector_v beta = vector_v::Constant(1, shared);
    compare(nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(
                    y, y, x, beta, shared, y, shared),
            prefix_reference(y, y, x, beta, shared, y, shared), {shared});
    stan::math::recover_memory();
}

int main() {
    for (const int n : {1, 3, 4, 7, 8, 9, 31, 64, 257, 25000}) {
        prefix_case(n, -0.37, false);
        prefix_case(n, -std::numeric_limits<double>::infinity(), false);
        normal_case<false>(n);
        normal_case<true>(n);
    }
    prefix_case(9, -std::numeric_limits<double>::infinity(), true);
    alias_case();
    std::cout << "values=" << number_of_values << " gradients=" << number_of_gradients
              << " largest_scaled_value_error=" << largest_value_error
              << " largest_scaled_gradient_error=" << largest_gradient_error << '\n';
    return (largest_value_error <= 1e-9 && largest_gradient_error <= 1e-6) ? 0 : 1;
}





















