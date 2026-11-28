#include "../inst/include/BayesMVP/stan_external_affine_block_reductions.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>


double largest_value_error = 0.0;
double largest_gradient_error = 0.0;
int number_of_values = 0;
int number_of_gradients = 0;

void record(double actual,
            double reference,
            bool gradient) {
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

template <typename T_z, typename T_start, typename T_a, typename T_b>
auto recurrence_reference(const T_z& z,
                          const T_start& start,
                          const T_a& a,
                          const T_b& b) {
    stan::return_type_t<T_z, T_start, T_a, T_b> value = start;
    for (Eigen::Index i = 0; i < z.size(); i++) value = a * value + b * z[i];
    return value;
}

void recurrence_case(int n,
                     double coefficient,
                     double scale) {
    Eigen::Matrix<stan::math::var, -1, 1> z(n);
    stan::math::var start = 0.31;
    stan::math::var a = coefficient;
    stan::math::var b = scale;
    std::vector<stan::math::var> parameters{start, a, b};
    for (int i = 0; i < n; i++) {
        z[i] = std::sin(0.17 * i) + 0.2 * std::cos(0.13 * i);
        parameters.push_back(z[i]);
    }
    compare(nico_last_value_of_affine_recurrence( z,
                                                  start,
                                                  a,
                                                  b,
                                                  nullptr),
            recurrence_reference(z, start, a, b), parameters);
    const Eigen::Matrix<double, -1, 1> z_data = stan::math::value_of(z);
    compare(nico_last_value_of_affine_recurrence( z_data,
                                                  start,
                                                  a,
                                                  b,
                                                  nullptr),
            recurrence_reference(z_data, start, a, b), parameters);
    compare(nico_last_value_of_affine_recurrence( z,
                                                  start.val(),
                                                  a.val(),
                                                  b.val(),
                                                  nullptr),
            recurrence_reference(z, start.val(), a.val(), b.val()), parameters);
    record(nico_last_value_of_affine_recurrence( z_data,
                                                 start.val(),
                                                 a.val(),
                                                 b.val(),
                                                 nullptr),
            recurrence_reference(z_data, start.val(), a.val(), b.val()),
           false);
    stan::math::recover_memory();
}

void log_sum_exp_case(int n,
                      bool extreme) {
    Eigen::Matrix<double, -1, -1> x(n, 2);
    Eigen::Matrix<stan::math::var, -1, 1> beta(2);
    Eigen::Matrix<stan::math::var, -1, 1> z(n);
    beta << 0.23, -0.41;
    stan::math::var scale = 0.83;
    std::vector<stan::math::var> parameters{beta[0], beta[1], scale};
    for (int i = 0; i < n; i++) {
        x(i, 0) = std::sin(0.17 * i);
        x(i, 1) = std::cos(0.31 * i);
        z[i] = extreme ? (i == 0 ? -900.0 : 900.0) : std::sin(0.13 * i);
        parameters.push_back(z[i]);
    }
    Eigen::Matrix<stan::math::var, -1, 1> eta = x * beta + scale * z;
    compare(nico_log_sum_exp_of_affine_predictor( x,
                                                  beta,
                                                  scale,
                                                  z,
                                                  nullptr),
            stan::math::log_sum_exp(eta), parameters);
    const Eigen::Matrix<double, -1, 1> beta_data = stan::math::value_of(beta);
    const Eigen::Matrix<double, -1, 1> z_data = stan::math::value_of(z);
    Eigen::Matrix<stan::math::var, -1, 1> eta_of_data_z = x * beta + scale * z_data;
    compare(nico_log_sum_exp_of_affine_predictor( x,
                                                  beta,
                                                  scale,
                                                  z_data,
                                                  nullptr),
            stan::math::log_sum_exp(eta_of_data_z), parameters);
    Eigen::Matrix<stan::math::var, -1, 1> eta_of_data_coefficients = x * beta_data + scale.val() * z;
    compare(nico_log_sum_exp_of_affine_predictor( x,
                                                  beta_data,
                                                  scale.val(),
                                                  z,
                                                  nullptr),
            stan::math::log_sum_exp(eta_of_data_coefficients), parameters);
    const Eigen::Matrix<double, -1, 1> eta_data = x * beta_data + scale.val() * z_data;
    record(nico_log_sum_exp_of_affine_predictor( x,
                                                 beta_data,
                                                 scale.val(),
                                                 z_data,
                                                 nullptr),
            stan::math::log_sum_exp(eta_data),
           false);
    stan::math::recover_memory();
}

void alias_and_empty_cases() {
    stan::math::var shared = 0.23;
    Eigen::Matrix<stan::math::var, -1, 1> z = Eigen::Matrix<stan::math::var, -1, 1>::Constant(17, shared);
    compare(nico_last_value_of_affine_recurrence( z,
                                                  shared,
                                                  shared,
                                                  shared,
                                                  nullptr),
            recurrence_reference(z, shared, shared, shared), {shared});
    Eigen::Matrix<double, -1, -1> x = Eigen::Matrix<double, -1, -1>::Ones(17, 1);
    Eigen::Matrix<stan::math::var, -1, 1> beta = Eigen::Matrix<stan::math::var, -1, 1>::Constant(1, shared);
    Eigen::Matrix<stan::math::var, -1, 1> eta = x * beta + shared * z;
    compare(nico_log_sum_exp_of_affine_predictor( x,
                                                  beta,
                                                  shared,
                                                  z,
                                                  nullptr),
            stan::math::log_sum_exp(eta), {shared});
    Eigen::Matrix<double, -1, -1> empty_x(0, 1);
    Eigen::Matrix<stan::math::var, -1, 1> empty_z(0);
    stan::math::var empty_lse = nico_log_sum_exp_of_affine_predictor( empty_x,
                                                                      beta,
                                                                      shared,
                                                                      empty_z,
                                                                      nullptr);
    if (!std::isinf(empty_lse.val()) || empty_lse.val() >= 0) {
        throw std::runtime_error("The log sum of an empty affine predictor must be negative infinity.");
    }
    std::vector<stan::math::var> parameters{shared};
    std::vector<double> gradient;
    stan::math::set_zero_all_adjoints();
    empty_lse.grad(parameters, gradient);
    record(gradient[0], 0.0, true);
    stan::math::recover_memory();
}

template <bool propto, typename T_y, typename T_mu, typename T_z, typename T_initial, typename T_a,
          typename T_b, typename T_c, typename T_d, typename T_prior_mean, typename T_prior_sd>
auto joint_reference(const T_y& y,
                     const T_mu& mu,
                     const T_z& z,
                     const T_initial& initial,
                     const T_a& a,
                     const T_b& b,
                     const T_c& c,
                     const T_d& d,
                     const T_prior_mean& prior_mean,
                     const T_prior_sd& prior_sd) {
    Eigen::Matrix<stan::return_type_t<T_z, T_initial, T_a, T_b>, -1, 1> states(z.size());
    stan::scalar_seq_view<T_b> scale_sequence(b);
    stan::return_type_t<T_z, T_initial, T_a, T_b> state = initial;
    for (Eigen::Index i = 0; i < z.size(); i++) {
        state = a * state + scale_sequence[i] * z[i];
        states[i] = state;
    }
    const auto log_scales = stan::math::add(c, stan::math::multiply(d, states));
    const auto scales = stan::math::exp(log_scales);
    return stan::math::normal_lpdf<propto>(y, mu, scales)
             + stan::math::normal_lpdf<propto>(z, prior_mean, prior_sd);
}

template <bool propto>
void joint_case(int n,
                double coefficient) {
    Eigen::Matrix<stan::math::var, -1, 1> y(n);
    Eigen::Matrix<stan::math::var, -1, 1> mu(n);
    Eigen::Matrix<stan::math::var, -1, 1> z(n);
    Eigen::Matrix<stan::math::var, -1, 1> b_vector(n);
    stan::math::var initial = 0.11;
    stan::math::var a = coefficient;
    stan::math::var b = 1.2;
    stan::math::var c = -0.23;
    stan::math::var d = 0.07;
    stan::math::var prior_mean = 0.17;
    stan::math::var prior_sd = 1.13;
    std::vector<stan::math::var> parameters{initial, a, b, c, d, prior_mean, prior_sd};
    for (int i = 0; i < n; i++) {
        y[i] = std::sin(0.21 * i);
        mu[i] = 0.1 * std::cos(0.13 * i);
        z[i] = std::sin(0.17 * i) + 0.2 * std::cos(0.13 * i);
        b_vector[i] = 0.7 + 0.1 * std::sin(0.11 * i);
        parameters.push_back(y[i]);
        parameters.push_back(mu[i]);
        parameters.push_back(z[i]);
        parameters.push_back(b_vector[i]);
    }
    compare(nico_normal_affine_recurrence_log_scale_joint_lpdf<propto>( y,
                                                                        mu,
                                                                        z,
                                                                        initial,
                                                                        a,
                                                                        b,
                                                                        c,
                                                                        d,
                                                                        prior_mean,
                                                                        prior_sd,
                                                                        nullptr),
            joint_reference<propto>(y, mu, z, initial, a, b, c, d, prior_mean, prior_sd), parameters);
    compare(nico_normal_affine_recurrence_log_scale_joint_lpdf<propto>( y,
                                                                        mu,
                                                                        z,
                                                                        initial,
                                                                        a,
                                                                        b_vector,
                                                                        c,
                                                                        d,
                                                                        prior_mean,
                                                                        prior_sd,
                                                                        nullptr),
            joint_reference<propto>(y, mu, z, initial, a, b_vector, c, d, prior_mean, prior_sd), parameters);
    const Eigen::Matrix<double, -1, 1> y_data = stan::math::value_of(y);
    const Eigen::Matrix<double, -1, 1> z_data = stan::math::value_of(z);
    compare(nico_normal_affine_recurrence_log_scale_joint_lpdf<propto>( y_data,
                                                                        0.0,
                                                                        z,
                                                                        initial,
                                                                        a,
                                                                        b,
                                                                        c,
                                                                        d,
                                                                        0.0,
                                                                        1.0,
                                                                        nullptr),
            joint_reference<propto>(y_data, 0.0, z, initial, a, b, c, d, 0.0, 1.0), parameters);
    compare(nico_normal_affine_recurrence_log_scale_joint_lpdf<propto>( y_data,
                                                                        0.0,
                                                                        z_data,
                                                                        initial.val(),
                                                                        a.val(),
                                                                        b.val(),
                                                                        c.val(),
                                                                        d.val(),
                                                                        prior_mean,
                                                                        prior_sd,
                                                                        nullptr),
            joint_reference<propto>(y_data, 0.0, z_data, initial.val(), a.val(), b.val(), c.val(), d.val(),
                                      prior_mean, prior_sd), parameters);
    record(nico_normal_affine_recurrence_log_scale_joint_lpdf<propto>( y_data,
                                                                       0.0,
                                                                       z_data,
                                                                       initial.val(),
                                                                       a.val(),
                                                                       b.val(),
                                                                       c.val(),
                                                                       d.val(),
                                                                       0.0,
                                                                       1.0,
                                                                       nullptr),
            joint_reference<propto>(y_data, 0.0, z_data, initial.val(), a.val(), b.val(), c.val(), d.val(),
                                      0.0, 1.0),
           false);
    stan::math::recover_memory();
}

template <typename T_z, typename T_initial, typename T_a, typename T_b, typename T_first>
auto positions_reference(const std::vector<T_z>& z,
                         const std::vector<int>& positions,
                         const T_initial& initial,
                         const T_a& a,
                         const T_b& b,
                         const T_first& first) {
    Eigen::Matrix<stan::return_type_t<T_z, T_initial, T_a, T_b, T_first>, -1, 1> result(positions.size());
    stan::return_type_t<T_z, T_initial, T_a, T_b, T_first> state = initial;
    std::size_t selected = 0;
    for (std::size_t i = 0; i < z.size(); i++) {
        state = a * state + (i == 0 ? first : b) * z[i];
        if (selected < positions.size() && static_cast<int>(i + 1) == positions[selected]) {
            result[selected++] = state;
        }
    }
    return result;
}

void positions_case(int n,
                    double coefficient) {
    std::vector<stan::math::var> z(n);
    std::vector<double> z_data(n);
    stan::math::var initial = 0.11;
    stan::math::var a = coefficient;
    stan::math::var b = 0.7;
    stan::math::var first = 1.3;
    std::vector<stan::math::var> parameters{initial, a, b, first};
    for (int i = 0; i < n; i++) {
        z_data[i] = std::sin(0.17 * i) + 0.2 * std::cos(0.13 * i);
        z[i] = z_data[i];
        parameters.push_back(z[i]);
    }
    std::vector<int> positions;
    if (n > 0) positions = {1, std::max(1, n / 2), n};
    std::sort(positions.begin(), positions.end());
    positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
    const Eigen::Matrix<double, -1, 1> weights = Eigen::Matrix<double, -1, 1>::LinSpaced(positions.size(), 0.3, 1.2);
    compare(stan::math::dot_product(weights, nico_affine_recurrence_at_positions( z,
                                                                                  positions,
                                                                                  initial,
                                                                                  a,
                                                                                  b,
                                                                                  first,
                                                                                  nullptr)),
            stan::math::dot_product(weights, positions_reference(z, positions, initial, a, b, first)),
            parameters);
    compare(stan::math::dot_product(weights, nico_affine_recurrence_at_positions( z_data,
                                                                                  positions,
                                                                                  initial,
                                                                                  a,
                                                                                  b,
                                                                                  first,
                                                                                  nullptr)),
            stan::math::dot_product(weights, positions_reference(z_data, positions, initial, a, b, first)),
            parameters);
    record(stan::math::dot_product(weights, nico_affine_recurrence_at_positions( z_data,
                                                                                 positions,
                                                                                 initial.val(),
                                                                                 a.val(),
                                                                                 b.val(),
                                                                                 first.val(),
                                                                                 nullptr)),
            stan::math::dot_product(weights, positions_reference(z_data, positions,
                                         initial.val(), a.val(), b.val(), first.val())),
           false);
    stan::math::recover_memory();
}

template <typename T_beta, typename T_scale, typename T_z>
auto segmented_reference(const Eigen::Matrix<double, -1, -1>& x,
                         const T_beta& beta,
                         const T_scale& scale,
                         const std::vector<T_z>& z,
                         const std::vector<int>& index,
                         const std::vector<int>& first,
                         const std::vector<int>& last) {
    Eigen::Matrix<stan::return_type_t<T_beta, T_scale, T_z>, -1, 1> result(first.size());
    for (std::size_t b = 0; b < first.size(); b++) {
        const int start = first[b] - 1;
        const int length = last[b] - first[b] + 1;
        Eigen::Matrix<stan::return_type_t<T_beta, T_scale, T_z>, -1, 1> values(length);
        for (int i = 0; i < length; i++) values[i] = z[index[start + i] - 1];
        const auto eta = stan::math::add(stan::math::multiply(x.middleRows(start, length), beta),
                                       stan::math::multiply(scale, values));
        result[b] = stan::math::log_sum_exp(eta);
    }
    return result;
}

void segmented_case(int n,
                    int columns) {
    const int groups = std::max(1, n / 3 + 1);
    std::vector<stan::math::var> z(groups);
    std::vector<double> z_data(groups);
    std::vector<int> index(n);
    Eigen::Matrix<double, -1, -1> x(n, columns);
    Eigen::Matrix<stan::math::var, -1, 1> beta(columns);
    stan::math::var scale = -0.7;
    std::vector<stan::math::var> parameters{scale};
    for (int j = 0; j < columns; j++) {
        beta[j] = 0.17 * (j + 1);
        parameters.push_back(beta[j]);
    }
    for (int j = 0; j < groups; j++) {
        z_data[j] = std::sin(0.11 * j);
        z[j] = j % 3 == 1 ? z[j - 1] : stan::math::var(z_data[j]);
        z_data[j] = z[j].val();
        parameters.push_back(z[j]);
    }
    for (int i = 0; i < n; i++) {
        index[i] = i % groups + 1;
        for (int j = 0; j < columns; j++) x(i, j) = std::cos(0.13 * i * (j + 1));
    }
    std::vector<int> first;
    std::vector<int> last;
    if (n > 0) {
        first = {1, 1, std::max(1, n / 2)};
        last = {1, n, n};
    }
    const Eigen::Matrix<double, -1, 1> weights = Eigen::Matrix<double, -1, 1>::LinSpaced(first.size(), -0.3, 1.2);
    compare(stan::math::dot_product(weights,
                nico_log_sum_exp_of_indexed_affine_predictor_in_segments( x,
                                                                          beta,
                                                                          scale,
                                                                          z,
                                                                          index,
                                                                          first,
                                                                          last,
                                                                          nullptr)),
            stan::math::dot_product(weights, segmented_reference(x, beta, scale, z, index, first, last)),
            parameters);
    compare(stan::math::dot_product(weights,
                nico_log_sum_exp_of_indexed_affine_predictor_in_segments( x,
                                                                          beta,
                                                                          scale,
                                                                          z_data,
                                                                          index,
                                                                          first,
                                                                          last,
                                                                          nullptr)),
            stan::math::dot_product(weights, segmented_reference(x, beta, scale, z_data, index, first, last)),
            parameters);
    const Eigen::Matrix<double, -1, 1> beta_data = stan::math::value_of(beta);
    record(stan::math::dot_product(weights,
               nico_log_sum_exp_of_indexed_affine_predictor_in_segments( x,
                                                                         beta_data,
                                                                         scale.val(),
                                                                         z_data,
                                                                         index,
                                                                         first,
                                                                         last,
                                                                         nullptr)),
            stan::math::dot_product(weights,
                segmented_reference(x, beta_data, scale.val(), z_data, index, first, last)),
           false);
    stan::math::recover_memory();
}

void positions_alias_case() {
    stan::math::var shared = 0.23;
    std::vector<stan::math::var> z(17, shared);
    const std::vector<int> positions{1, 3, 8, 17};
    const Eigen::Matrix<double, -1, 1> weights = Eigen::Matrix<double, -1, 1>::LinSpaced(4, -0.7, 1.1);
    compare(stan::math::dot_product(weights, nico_affine_recurrence_at_positions( z,
                                                                                  positions,
                                                                                  shared,
                                                                                  shared,
                                                                                  shared,
                                                                                  shared,
                                                                                  nullptr)),
            stan::math::dot_product(weights, positions_reference(z, positions, shared, shared, shared, shared)),
            {shared});
    stan::math::recover_memory();
}

void joint_alias_case() {
    stan::math::var shared = 0.23;
    Eigen::Matrix<stan::math::var, -1, 1> values = Eigen::Matrix<stan::math::var, -1, 1>::Constant(17, shared);
    compare(nico_normal_affine_recurrence_log_scale_joint_lpdf( values,
                                                                shared,
                                                                values,
                                                                shared,
                                                                shared,
                                                                values,
                                                                shared,
                                                                shared,
                                                                shared,
                                                                shared,
                                                                nullptr),
            joint_reference<false>(values, shared, values, shared, shared, values,
                                     shared, shared, shared, shared), {shared});
    stan::math::recover_memory();
}

void positions_many_blocks_case(int n,
                                double coefficient) {
    //// the boundary pass as the SV model calls it: a position at the end of every chunk but the last
    std::vector<stan::math::var> z(n);
    stan::math::var initial = 0.0;
    stan::math::var a = coefficient;
    stan::math::var b = 0.7;
    stan::math::var first = 1.3;
    std::vector<stan::math::var> parameters{a, b, first};
    for (int i = 0; i < n; i++) {
        z[i] = std::sin(0.17 * i) + 0.2 * std::cos(0.13 * i);
        parameters.push_back(z[i]);
    }
    std::vector<int> positions;
    for (int position = 1 + n / 128; position < n; position += 1 + n / 128) positions.push_back(position);
    if (positions.empty()) return;
    const Eigen::Matrix<double, -1, 1> weights = Eigen::Matrix<double, -1, 1>::LinSpaced(positions.size(), -0.9, 1.4);
    compare(stan::math::dot_product(weights, nico_affine_recurrence_at_positions( z,
                                                                                  positions,
                                                                                  initial,
                                                                                  a,
                                                                                  b,
                                                                                  first,
                                                                                  nullptr)),
            stan::math::dot_product(weights, positions_reference(z, positions, initial, a, b, first)),
            parameters);
    stan::math::recover_memory();
}
int main() {
    for (int n : {0, 1, 3, 7, 8, 9, 31, 257, 25000}) {
        for (double coefficient : {-0.999, -0.7, 0.0, 0.43, 0.999}) {
            for (double scale : {-1.3, 0.0, 1.2}) recurrence_case(n, coefficient, scale);
        }
        if (n > 0) log_sum_exp_case(n, false);
        segmented_case(n, 0);
        segmented_case(n, 2);
        for (double coefficient : {-0.93, 0.0, 0.87}) {
            positions_case(n, coefficient);
            joint_case<false>(n, coefficient);
            joint_case<true>(n, coefficient);
        }
    }
    for (int n : {257, 5000, 25000}) {
        for (double coefficient : {-0.93, 0.0, 0.87, 0.999}) positions_many_blocks_case(n, coefficient);
    }
    log_sum_exp_case(31, true);
    alias_and_empty_cases();
    joint_alias_case();
    positions_alias_case();
    std::cout << "values=" << number_of_values << " gradients=" << number_of_gradients
              << " largest_scaled_value_error=" << largest_value_error
              << " largest_scaled_gradient_error=" << largest_gradient_error << '\n';
    return (largest_value_error <= 1e-9 && largest_gradient_error <= 1e-6) ? 0 : 1;
}

















