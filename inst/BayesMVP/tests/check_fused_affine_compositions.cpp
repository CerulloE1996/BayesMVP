#include "../inst/include/BayesMVP/stan_external_fused_affine_compositions.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>


double largest_value_error = 0.0;
double largest_gradient_error = 0.0;
int number_of_values = 0;
int number_of_gradients = 0;

void record(const double actual,
            const double reference,
            const bool gradient) {
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

void compare(stan::math::var actual,
             stan::math::var reference,
             std::vector<stan::math::var> parameters) {
    record( actual.val(),
            reference.val(),
            false);
    std::vector<double> actual_gradient;
    std::vector<double> reference_gradient;
    stan::math::set_zero_all_adjoints();
    actual.grad(parameters, actual_gradient);
    stan::math::set_zero_all_adjoints();
    reference.grad(parameters, reference_gradient);
    for (std::size_t i = 0; i < parameters.size(); i++) {
        record(actual_gradient[i], reference_gradient[i], true);
    }
}

template <typename T_weight, typename T_beta, typename T_scale, typename T_z, typename T_offset>
auto prefix_reference(const T_weight& linear,
                      const T_weight& prefix,
                      const Eigen::Matrix<double, -1, -1>& x,
                      const T_beta& beta,
                      const T_scale& scale,
                      const T_z& z,
                      const T_offset& offset) {
    Eigen::Matrix<stan::return_type_t<T_weight, T_beta, T_scale, T_z, T_offset>, -1, 1> eta = x * beta + scale * z;
    stan::return_type_t<T_weight, T_beta, T_scale, T_z, T_offset> result = 0.0;
    stan::return_type_t<T_weight, T_beta, T_scale, T_z, T_offset> running = offset;
    for (Eigen::Index i = 0; i < z.size(); i++) {
        running = stan::math::log_sum_exp(running, eta[i]);
        result += linear[i] * eta[i] + prefix[i] * running;
    }
    return result;
}

void prefix_case(const int n,
                 const double offset_value,
                 const bool extreme) {
    Eigen::Matrix<double, -1, -1> x(n, 2);
    Eigen::Matrix<stan::math::var, -1, 1> beta(2);
    Eigen::Matrix<stan::math::var, -1, 1> z(n);
    Eigen::Matrix<stan::math::var, -1, 1> linear(n);
    Eigen::Matrix<stan::math::var, -1, 1> prefix(n);
    beta << 0.31, -0.19;
    std::vector<stan::math::var> parameters{beta[0], beta[1]};
    for (int i = 0; i < n; i++) {
        x(i, 0) = std::sin(0.17 * i);
        x(i, 1) = std::cos(0.23 * i);
        z[i] = extreme ? (i == 0 ? -900.0 : 900.0) : std::sin(0.37 * i);
        linear[i] = 0.3 * std::cos(0.41 * i);
        prefix[i] = -0.7 + 0.2 * std::sin(0.43 * i);
        parameters.push_back(z[i]);
        parameters.push_back(linear[i]);
        parameters.push_back(prefix[i]);
    }
    stan::math::var scale = extreme ? 1.0 : -0.83;
    stan::math::var offset = offset_value;
    parameters.push_back(scale);
    if (std::isfinite(offset_value)) parameters.push_back(offset);
    const stan::math::var actual = nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(
            linear, prefix, x, beta, scale, z, offset);
    const stan::math::var reference = prefix_reference(linear, prefix, x, beta, scale, z, offset);
    compare(actual, reference, parameters);
    const Eigen::Matrix<double, -1, 1> linear_data = stan::math::value_of(linear);
    const Eigen::Matrix<double, -1, 1> prefix_data = stan::math::value_of(prefix);
    compare(nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(
                    linear_data, prefix_data, x, beta, scale, z, offset),
            prefix_reference(linear_data, prefix_data, x, beta, scale, z, offset), parameters);
    const Eigen::Matrix<double, -1, 1> beta_data = stan::math::value_of(beta);
    const Eigen::Matrix<double, -1, 1> z_data = stan::math::value_of(z);
    record(nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(
                   linear_data, prefix_data, x, beta_data, scale.val(), z_data, offset_value),
            prefix_reference(linear_data, prefix_data, x, beta_data, scale.val(), z_data, offset_value),
           false);
    stan::math::recover_memory();
}

template <bool propto>
void normal_case(const int n) {
    Eigen::Matrix<stan::math::var, -1, 1> y(n);
    Eigen::Matrix<stan::math::var, -1, 1> mu(n);
    Eigen::Matrix<stan::math::var, -1, 1> predictor(n);
    stan::math::var intercept = 0.27;
    stan::math::var slope = -0.43;
    std::vector<stan::math::var> parameters{intercept, slope};
    for (int i = 0; i < n; i++) {
        y[i] = std::sin(0.21 * i);
        mu[i] = 0.1 * std::cos(0.13 * i);
        predictor[i] = std::cos(0.31 * i);
        parameters.push_back(y[i]);
        parameters.push_back(mu[i]);
        parameters.push_back(predictor[i]);
    }
    Eigen::Matrix<stan::math::var, -1, 1> sigma = stan::math::exp(stan::math::add(intercept, slope * predictor));
    compare(nico_normal_affine_log_scale_lpdf<propto>( y,
                                                       mu,
                                                       intercept,
                                                       slope,
                                                       predictor,
                                                       nullptr),
            stan::math::normal_lpdf<propto>(y, mu, sigma), parameters);
    const Eigen::Matrix<double, -1, 1> y_data = stan::math::value_of(y);
    const Eigen::Matrix<double, -1, 1> predictor_data = stan::math::value_of(predictor);
    Eigen::Matrix<stan::math::var, -1, 1> scales = stan::math::exp(stan::math::add(intercept, slope.val() * predictor_data));
    compare(nico_normal_affine_log_scale_lpdf<propto>( y_data,
                                                       0.0,
                                                       intercept,
                                                       slope.val(),
                                                       predictor_data,
                                                       nullptr),
            stan::math::normal_lpdf<propto>(y_data, 0.0, scales), parameters);
    const Eigen::Matrix<double, -1, 1> sigma_data = stan::math::value_of(sigma);
    compare(nico_normal_affine_log_scale_lpdf<propto>( y,
                                                       0.0,
                                                       intercept.val(),
                                                       slope.val(),
                                                       predictor_data,
                                                       nullptr),
            stan::math::normal_lpdf<propto>(y, 0.0, sigma_data), parameters);
    record(nico_normal_affine_log_scale_lpdf<propto>( y_data,
                                                      0.0,
                                                      intercept.val(),
                                                      slope.val(),
                                                      predictor_data,
                                                      nullptr),
            stan::math::normal_lpdf<propto>(y_data, 0.0, sigma_data),
           false);
    stan::math::recover_memory();
}

void alias_case() {
    stan::math::var shared = 0.23;
    Eigen::Matrix<stan::math::var, -1, 1> y = Eigen::Matrix<stan::math::var, -1, 1>::Constant(9, shared);
    Eigen::Matrix<stan::math::var, -1, 1> scales = stan::math::exp(stan::math::add(shared, shared * y));
    compare(nico_normal_affine_log_scale_lpdf( y,
                                               shared,
                                               shared,
                                               shared,
                                               y,
                                               nullptr),
            stan::math::normal_lpdf(y, shared, scales), {shared});
    Eigen::Matrix<double, -1, -1> x = Eigen::Matrix<double, -1, -1>::Ones(9, 1);
    Eigen::Matrix<stan::math::var, -1, 1> beta = Eigen::Matrix<stan::math::var, -1, 1>::Constant(1, shared);
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





















