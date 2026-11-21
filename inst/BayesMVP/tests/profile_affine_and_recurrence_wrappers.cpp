#include "../inst/include/BayesMVP/stan_external_affine_block_reductions.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using clock_of_profile = std::chrono::steady_clock;

struct profile_barrier {
    std::mutex mutex;
    std::condition_variable condition;
    int ready = 0;
    int finished = 0;
    bool start = false;
    bool release = false;
    clock_of_profile::time_point deadline;
};

template <typename T_vector, typename T_scalar>
stan::return_type_t<T_vector, T_scalar> fn_evaluate(
        const std::string& function, bool use_AVX, const T_vector& z, const T_vector& beta,
        const T_scalar& scale, const T_scalar& coefficient, const T_scalar& initial,
        const T_scalar& intercept, const Eigen::MatrixXd& x, const Eigen::VectorXd& y,
        const Eigen::VectorXd& linear_weights, const Eigen::VectorXd& prefix_weights) {

    if (function == "affine_log_sum_exp") {
        if (use_AVX) return nico_log_sum_exp_of_affine_predictor(x, beta, scale, z);
        const auto eta = stan::math::add(stan::math::multiply(x, beta), stan::math::multiply(scale, z));
        return stan::math::log_sum_exp(eta);
    }
    if (function == "affine_weighted_prefix") {
        if (use_AVX) return nico_dot_products_of_affine_predictor_and_log_cumulative_sum_exp(
            linear_weights, prefix_weights, x, beta, scale, z, initial);
        const auto eta = stan::math::add(stan::math::multiply(x, beta), stan::math::multiply(scale, z));
        const auto terms = stan::math::exp(eta);
        auto sums = stan::math::cumulative_sum(terms).eval();
        sums.array() += stan::math::exp(initial);
        return stan::math::dot_product(linear_weights, eta)
                 + stan::math::dot_product(prefix_weights, stan::math::log(sums));
    }
    if (function == "normal_recurrence_joint" && use_AVX) {
        return nico_normal_affine_recurrence_log_scale_joint_lpdf<false>(
            y, 0.0, z, initial, coefficient, scale, intercept, 0.5, 0.0, 1.0);
    }
    using scalar = stan::scalar_type_t<T_vector>;
    Eigen::Matrix<scalar, Eigen::Dynamic, 1> states(z.size());
    if (use_AVX) {
        const auto shocks = stan::math::multiply(scale, z);
        states = nico_AR1_recursion(shocks, initial, coefficient);
        return nico_normal_affine_log_scale_lpdf<false>(y, 0.0, intercept, 0.5, states)
                 + nico_normal_lpdf<false>(z, 0.0, 1.0, nullptr);
    }
    scalar state = initial;
    for (Eigen::Index i = 0; i < z.size(); ++i) {
        state = coefficient * state + scale * z[i];
        states[i] = state;
    }
    const auto log_scale = stan::math::add(intercept, stan::math::multiply(0.5, states));
    return stan::math::normal_lpdf<false>(y, 0.0, stan::math::exp(log_scale))
             + stan::math::normal_lpdf<false>(z, 0.0, 1.0);
}

template <bool autodiff>
void fn_profile_worker(int cpu, int n, int chunks, const std::string& function,
                       bool use_AVX, profile_barrier& barrier, double& elapsed,
                       unsigned long& calls, std::string& error) {
    try {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu, &mask);
        if (sched_setaffinity(0, sizeof(mask), &mask) != 0) throw std::runtime_error("CPU pin failed.");
        stan::math::ChainableStack stack;
        using scalar = std::conditional_t<autodiff, stan::math::var, double>;
        Eigen::Matrix<scalar, Eigen::Dynamic, 1> z(n), beta(2);
        Eigen::MatrixXd x(n, 2);
        Eigen::VectorXd y(n), linear_weights(n), prefix_weights(n);
        for (int i = 0; i < n; ++i) {
            z[i] = 0.2 * std::sin(0.13 * i);
            x(i, 0) = 1.0;
            x(i, 1) = std::cos(0.17 * i);
            y[i] = std::sin(0.19 * i);
            linear_weights[i] = i % 3 == 0 ? 1.0 : 0.0;
            prefix_weights[i] = -linear_weights[i];
        }
        beta << 0.2, -0.1;
        scalar scale = 0.3;
        scalar coefficient = 0.8;
        scalar initial = 0.1;
        scalar intercept = -0.5;
        const auto one_call = [&]() {
            stan::math::nested_rev_autodiff nested;
            scalar target = 0.0;
            for (int chunk = 0; chunk < chunks; ++chunk) {
                const int first = n * chunk / chunks;
                const int length = n * (chunk + 1) / chunks - first;
                const Eigen::Matrix<scalar, Eigen::Dynamic, 1> z_of_chunk = z.segment(first, length);
                const Eigen::MatrixXd x_of_chunk = x.middleRows(first, length);
                target += fn_evaluate(function, use_AVX, z_of_chunk, beta, scale, coefficient,
                                      initial, intercept, x_of_chunk, y.segment(first, length),
                                      linear_weights.segment(first, length),
                                      prefix_weights.segment(first, length));
            }
            if constexpr (autodiff) {
                stan::math::set_zero_all_adjoints();
                target.grad();
                const double value = target.val() + z[0].adj();
                asm volatile("" : : "g"(value) : "memory");
            } else {
                asm volatile("" : : "g"(target) : "memory");
            }
        };
        one_call();
        {
            std::unique_lock<std::mutex> lock(barrier.mutex);
            ++barrier.ready;
            barrier.condition.notify_all();
            barrier.condition.wait(lock, [&]() { return barrier.start; });
        }
        const auto started = clock_of_profile::now();
        do {
            one_call();
            ++calls;
        } while (clock_of_profile::now() < barrier.deadline);
        elapsed = std::chrono::duration<double>(clock_of_profile::now() - started).count();
        {
            std::unique_lock<std::mutex> lock(barrier.mutex);
            ++barrier.finished;
            barrier.condition.notify_all();
            barrier.condition.wait(lock, [&]() { return barrier.release; });
        }
        stan::math::recover_memory();
    } catch (const std::exception& failure) {
        error = failure.what();
        std::lock_guard<std::mutex> lock(barrier.mutex);
        ++barrier.ready;
        ++barrier.finished;
        barrier.condition.notify_all();
    }
}

void fn_snapshot() {
    const char* pmc_pid = std::getenv("PMC_STAT_PID");
    if (!pmc_pid) throw std::runtime_error("Run this diagnostic under pmc_stat.");
    if (kill(static_cast<pid_t>(std::stoi(pmc_pid)), SIGUSR1) != 0) {
        throw std::runtime_error("The hardware-counter snapshot failed.");
    }
    usleep(100000);
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: profile <function> <Stan|AVX> <double|autodiff> <N> <chunks> <CPUs>\n";
        return 2;
    }
    const std::string function = argv[1];
    const bool use_AVX = std::string(argv[2]) == "AVX";
    const bool autodiff = std::string(argv[3]) == "autodiff";
    const int n = std::stoi(argv[4]);
    const int chunks = std::stoi(argv[5]);
    const int cpus = std::stoi(argv[6]);
    if (n < 1 || chunks < 1 || chunks > n || cpus < 1) return 2;
    if (function != "affine_log_sum_exp" && function != "affine_weighted_prefix"
        && function != "normal_recurrence_joint" && function != "normal_recurrence_composite") return 2;
    profile_barrier barrier;
    std::vector<std::thread> workers;
    std::vector<double> seconds(cpus);
    std::vector<unsigned long> calls(cpus);
    std::vector<std::string> errors(cpus);
    for (int cpu = 0; cpu < cpus; ++cpu) {
        if (autodiff) workers.emplace_back(fn_profile_worker<true>, cpu, n, chunks,
            std::cref(function), use_AVX, std::ref(barrier), std::ref(seconds[cpu]),
            std::ref(calls[cpu]), std::ref(errors[cpu]));
        else workers.emplace_back(fn_profile_worker<false>, cpu, n, chunks,
            std::cref(function), use_AVX, std::ref(barrier), std::ref(seconds[cpu]),
            std::ref(calls[cpu]), std::ref(errors[cpu]));
    }
    {
        std::unique_lock<std::mutex> lock(barrier.mutex);
        barrier.condition.wait(lock, [&]() { return barrier.ready == cpus; });
    }
    fn_snapshot();
    {
        std::lock_guard<std::mutex> lock(barrier.mutex);
        barrier.deadline = clock_of_profile::now() + std::chrono::milliseconds(750);
        barrier.start = true;
        barrier.condition.notify_all();
    }
    {
        std::unique_lock<std::mutex> lock(barrier.mutex);
        barrier.condition.wait(lock, [&]() { return barrier.finished == cpus; });
    }
    fn_snapshot();
    {
        std::lock_guard<std::mutex> lock(barrier.mutex);
        barrier.release = true;
        barrier.condition.notify_all();
    }
    for (auto& worker : workers) worker.join();
    for (const auto& error : errors) if (!error.empty()) {
        std::cerr << error << '\n';
        return 1;
    }
    unsigned long total_calls = 0;
    for (auto count : calls) total_calls += count;
    const double elapsed = *std::max_element(seconds.begin(), seconds.end());
    std::cout << std::setprecision(12) << function << ',' << (use_AVX ? "AVX" : "Stan") << ','
              << (autodiff ? "autodiff" : "double") << ',' << n << ',' << chunks << ',' << cpus << ','
              << BMVP_LANES << ',' << total_calls << ',' << elapsed << ','
              << total_calls / elapsed << '\n';
}



















