#pragma once
//
// -| --------- BayesMVP AVX-512 / AVX2 special functions, exposed to Stan as external functions ------------------------------
//
// Vector -> vector versions of the special functions that dominate the LC-MVOP likelihood, computed
// with the BayesMVP SIMD kernels (fast_and_approx_AVX512_fns.hpp / fast_and_approx_AVX2_fns.hpp)
// instead of Stan's scalar implementations. Each function has
//
//     * a PRIM overload  (Eigen column vector of double)  -> Eigen::VectorXd
//     * a REV  overload  (Eigen column vector of var)     -> Eigen::Matrix<var, -1, 1>
//
// v4 (this file):
//
//   * SELF-CONTAINED. Only the two kernel files are included, with the three macros they need
//     defined here. v1-v3 included BayesMVP's initial_config / SIMD_config / eigen_config /
//     double_fns.hpp, and double_fns.hpp puts `using namespace Eigen; using namespace stan::math;`
//     and Eigen macro redefinitions into the WHOLE model translation unit (the user header is
//     force-included before the generated model code). Nothing of BayesMVP's leaks into the model now.
//
//   * LEAN TAPE. Forward pass: operand values gathered into thread-local scratch, kernel applied,
//     result varis placement-new'ed as ONE contiguous block on the arena and registered on the
//     nochain stack (their chain() is empty). Only the operand vari pointers are stored. Per element:
//     8 (operand ptr) + 24 (vari) bytes on the arena; nothing else. Stan's own path for
//     Eigen::Matrix<var> input (apply_scalar_unary -> one callback vari per element) is 32 bytes
//     per element plus one virtual call per element in the reverse pass.
//     Reverse pass: one callback for the whole vector; it re-reads operand and result values,
//     recomputes F'(x) with the kernels into scratch, and scatters adj(x) += adj(f) * F'(x).
//
//   In isolation (stan_ext/microbench, 96 threads, length-200 vectors) this is 1.1x (log) to 12x
//   (inv_Phi_approx_from_logit_prob) faster than Stan's built-ins, and 10-30% faster than v3.
//
// Declared in the Stan functions block as e.g.   vector fast_Phi(vector x);   (no body) and compiled
// with --allow-undefined and this file as the user header (cmdstanr: user_header = ...;
// BridgeStan / BayesMVP: stanc_args --allow-undefined + make_args USER_HEADER=...).
//
// Functions:  fast_Phi, fast_log_Phi, fast_inv_Phi, fast_exp, fast_log, fast_log1m, fast_log1p,
//             fast_log1p_exp, fast_log1m_exp, fast_inv_logit, fast_log_inv_logit, fast_Phi_approx,
//             fast_log_Phi_approx, fast_inv_Phi_approx, fast_inv_Phi_approx_from_logit_prob,
//             bmvp_simd_lanes (diagnostic: 8 = AVX-512 kernels, 4 = AVX2 kernels)
//
// Kernel dispatch: AVX-512 (8 lanes) when compiled with -mavx512f -mavx512vl -mavx512dq, else AVX2
// (4 lanes) with -mavx2; anything less is a compile error (a silently scalar build is what we do NOT
// want to benchmark). Compiling with -DBAYESMVP_FORCE_AVX2 selects the 4-lane AVX2 kernels
// even when AVX-512 is enabled (e.g. BridgeStan make argument CPPFLAGS_OPTIM=-DBAYESMVP_FORCE_AVX2), so an
// AVX-512 machine can run the AVX2 arm with every other compile flag unchanged; bmvp_simd_lanes() then
// returns 4, which callers check against the requested backend. Tail handling: a partial final block goes through a padded stack buffer whose
// spare lanes repeat the last valid element; only the valid lanes are written back.
//
// -----------------------------------------------------------------------------------------------------------------------------

#include <stan/math/rev.hpp>
#include <stan/math/prim.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <new>
#include <ostream>
#include <vector>

// ---- the three macros the kernel files use (same definitions as BayesMVP's initial_config / SIMD_config)
#ifndef ALWAYS_INLINE
#  if defined(__GNUC__) || defined(__clang__)
#    define ALWAYS_INLINE __attribute__((always_inline)) inline
#  else
#    define ALWAYS_INLINE inline
#  endif
#endif
#ifndef VECTORCALL
#  define VECTORCALL
#endif
#ifndef ALIGN64
#  define ALIGN64 alignas(64)
#endif
#ifndef ALIGN32
#  define ALIGN32 alignas(32)
#endif

// ---- the four global constants the kernel files read (same values as BayesMVP's double_fns.hpp)
#ifndef BMVP_STAN_EXT_KERNEL_CONSTANTS
#define BMVP_STAN_EXT_KERNEL_CONSTANTS
static constexpr double NORMAL_TAIL_THRESH  = 2.5;
static constexpr int    MILLS_CF_DEPTH      = 40;
static constexpr double NEG_HALF_LOG_2PI    = -0.91893853320467274178;   // -0.5*log(2*pi)
static constexpr double INV_SQRT_2PI        =  0.39894228040143267794;   //  1/sqrt(2*pi)
#endif

// ---- the SIMD kernels (self-contained: <immintrin.h>, <cmath>, the macros and constants above). Absolute paths: no -I needed.
#include "math/fast_and_approx_AVX2_fns.hpp"
#include "math/fast_and_approx_AVX512_fns.hpp"

#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512DQ__) && !defined(BAYESMVP_FORCE_AVX2)
#  define BMVP_LANES 8
#  define BMVP_KERNEL_NAME(stem) stem##_AVX512
#elif defined(__AVX2__)
#  define BMVP_LANES 4
#  define BMVP_KERNEL_NAME(stem) stem##_AVX2
#else
#  error "BayesMVP_AVX_stan_fns.hpp: build with -mavx512f -mavx512vl -mavx512dq (or at least -mavx2); see BridgeStan make/local"
#endif

namespace bmvp_stan_ext {

using stan::math::var;
using stan::math::vari;
using VecVar = Eigen::Matrix<var, -1, 1>;

// ---- constants
static constexpr double INV_SQRT_2PI = 0.398942280401432677939946059934;
static constexpr double SQRT_2PI     = 2.506628274631000502415765284811;
static constexpr double LN2          = 0.693147180559945309417232121458;
static constexpr double PHI_APPROX_A = 0.07056;   // Stan's Phi_approx: inv_logit(0.07056 x^3 + 1.5976 x)
static constexpr double PHI_APPROX_B = 1.5976;
static constexpr double INV_PHI_APPROX_C1 = 0.34176618822627863;   // the model's inv_Phi_approx: C2 * sinh(asinh(C1 * L) / 3)
static constexpr double INV_PHI_APPROX_C2 = 5.494448514153059;

// ---- generic elementwise application on a raw buffer, in place. Kernels are passed as callables
// (the BayesMVP kernels are always_inline and cannot be called through function pointers).
#if BMVP_LANES == 8
using simd_t = __m512d;
inline simd_t simd_loadu(const double* p) { return _mm512_loadu_pd(p); }
inline void   simd_storeu(double* p, simd_t v) { _mm512_storeu_pd(p, v); }
inline simd_t simd_load(const double* p) { return _mm512_load_pd(p); }
inline void   simd_store(double* p, simd_t v) { _mm512_store_pd(p, v); }
#else
using simd_t = __m256d;
inline simd_t simd_loadu(const double* p) { return _mm256_loadu_pd(p); }
inline void   simd_storeu(double* p, simd_t v) { _mm256_storeu_pd(p, v); }
inline simd_t simd_load(const double* p) { return _mm256_load_pd(p); }
inline void   simd_store(double* p, simd_t v) { _mm256_store_pd(p, v); }
#endif

template <typename K>
inline void apply_kernel(double* p, const int n, K k) {
    int i = 0;
    for (; i + BMVP_LANES <= n; i += BMVP_LANES) simd_storeu(p + i, k(simd_loadu(p + i)));
    if (i < n) {
        alignas(64) double buf[BMVP_LANES];
        const int rem = n - i;
        for (int j = 0; j < BMVP_LANES; ++j) buf[j] = p[i + (j < rem ? j : rem - 1)];
        simd_store(buf, k(simd_load(buf)));
        for (int j = 0; j < rem; ++j) p[i + j] = buf[j];
    }
}

#define BMVP_K(stem) [](const simd_t v) { return BMVP_KERNEL_NAME(stem)(v); }

// ---- in-place kernels on a buffer of n doubles
inline void k_Phi(double* p, int n)            { apply_kernel(p, n, BMVP_K(fast_Phi)); }
inline void k_log_Phi(double* p, int n)        { apply_kernel(p, n, BMVP_K(fast_log_Phi)); }
inline void k_dlog_Phi_dx(double* p, int n)    { apply_kernel(p, n, BMVP_K(fast_dlog_Phi_dx)); }
inline void k_inv_Phi(double* p, int n)        { apply_kernel(p, n, BMVP_K(fast_inv_Phi_wo_checks)); }
inline void k_exp(double* p, int n)            { apply_kernel(p, n, BMVP_K(fast_exp_1)); }
inline void k_log(double* p, int n)            { apply_kernel(p, n, BMVP_K(fast_log_1)); }
inline void k_log1m(double* p, int n)          { apply_kernel(p, n, BMVP_K(fast_log1m_1)); }
inline void k_log1p(double* p, int n)          { apply_kernel(p, n, BMVP_K(fast_log1p_1)); }
inline void k_log1p_exp(double* p, int n)      { apply_kernel(p, n, BMVP_K(fast_log1p_exp_1)); }
inline void k_inv_logit(double* p, int n)      { apply_kernel(p, n, BMVP_K(fast_inv_logit)); }
inline void k_log_inv_logit(double* p, int n)  { apply_kernel(p, n, BMVP_K(fast_log_inv_logit)); }
inline void k_Phi_approx(double* p, int n)     { apply_kernel(p, n, BMVP_K(fast_Phi_approx)); }
inline void k_log_Phi_approx(double* p, int n) { apply_kernel(p, n, BMVP_K(fast_log_Phi_approx)); }
inline void k_inv_Phi_approx(double* p, int n) { apply_kernel(p, n, BMVP_K(fast_inv_Phi_approx)); }
inline void k_inv_Phi_approx_from_logit_prob(double* p, int n) { apply_kernel(p, n, BMVP_K(fast_inv_Phi_approx_from_logit_prob)); }
#undef BMVP_K

inline void copy_n(const double* x, int n, double* out) { for (int i = 0; i < n; ++i) out[i] = x[i]; }

// ---- VALUE functions:      f <- F(x)            (x read-only; f written; both length n; f != x)
// ---- DERIVATIVE functions: d <- F'(x)           given the operand values x AND the result values f,
//                                                  so the reverse pass can recompute cheaply.

inline void v_Phi(const double* x, int n, double* f) { copy_n(x, n, f); k_Phi(f, n); }
inline void d_Phi(const double* x, const double*, int n, double* d) {
    for (int i = 0; i < n; ++i) d[i] = -0.5 * x[i] * x[i];
    k_exp(d, n);
    for (int i = 0; i < n; ++i) d[i] *= INV_SQRT_2PI;                          // phi(x)
}

inline void v_log_Phi(const double* x, int n, double* f) { copy_n(x, n, f); k_log_Phi(f, n); }
inline void d_log_Phi(const double* x, const double*, int n, double* d) { copy_n(x, n, d); k_dlog_Phi_dx(d, n); }

inline void v_inv_Phi(const double* p, int n, double* f) { copy_n(p, n, f); k_inv_Phi(f, n); }
inline void d_inv_Phi(const double*, const double* z, int n, double* d) {
    for (int i = 0; i < n; ++i) d[i] = 0.5 * z[i] * z[i];
    k_exp(d, n);
    for (int i = 0; i < n; ++i) d[i] *= SQRT_2PI;                               // 1 / phi(z)
}

inline void v_exp(const double* x, int n, double* f) { copy_n(x, n, f); k_exp(f, n); }
inline void d_exp(const double*, const double* f, int n, double* d) { copy_n(f, n, d); }

inline void v_log(const double* x, int n, double* f) { copy_n(x, n, f); k_log(f, n); }
inline void d_log(const double* x, const double*, int n, double* d) { for (int i = 0; i < n; ++i) d[i] = 1.0 / x[i]; }

inline void v_log1m(const double* x, int n, double* f) { copy_n(x, n, f); k_log1m(f, n); }
inline void d_log1m(const double* x, const double*, int n, double* d) { for (int i = 0; i < n; ++i) d[i] = -1.0 / (1.0 - x[i]); }

inline void v_log1p(const double* x, int n, double* f) { copy_n(x, n, f); k_log1p(f, n); }
inline void d_log1p(const double* x, const double*, int n, double* d) { for (int i = 0; i < n; ++i) d[i] = 1.0 / (1.0 + x[i]); }

inline void v_log1p_exp(const double* x, int n, double* f) { copy_n(x, n, f); k_log1p_exp(f, n); }
inline void d_log1p_exp(const double* x, const double*, int n, double* d) { copy_n(x, n, d); k_inv_logit(d, n); }

// log1m_exp(x) = log(1 - exp(x)), x <= 0: log1p(-exp(x)) via the kernels for x < -ln 2; nearer 0
// that loses digits, so those lanes use the scalar log(-expm1(x)).
inline void v_log1m_exp(const double* x, int n, double* f) {
    copy_n(x, n, f); k_exp(f, n);
    for (int i = 0; i < n; ++i) f[i] = -f[i];
    k_log1p(f, n);
    for (int i = 0; i < n; ++i) if (x[i] >= -LN2) f[i] = std::log(-std::expm1(x[i]));
}
inline void d_log1m_exp(const double* x, const double* f, int n, double* d) {
    for (int i = 0; i < n; ++i) d[i] = x[i] - f[i];
    k_exp(d, n);
    for (int i = 0; i < n; ++i) d[i] = -d[i];                                   // -exp(x) / (1 - exp(x))
}

inline void v_inv_logit(const double* x, int n, double* f) { copy_n(x, n, f); k_inv_logit(f, n); }
inline void d_inv_logit(const double*, const double* f, int n, double* d) { for (int i = 0; i < n; ++i) d[i] = f[i] * (1.0 - f[i]); }

inline void v_log_inv_logit(const double* x, int n, double* f) { copy_n(x, n, f); k_log_inv_logit(f, n); }
inline void d_log_inv_logit(const double* x, const double*, int n, double* d) {
    for (int i = 0; i < n; ++i) d[i] = -x[i];
    k_inv_logit(d, n);                                                           // 1 - inv_logit(x)
}

inline double phi_approx_dpoly(double x) { return 3.0 * PHI_APPROX_A * x * x + PHI_APPROX_B; }

inline void v_Phi_approx(const double* x, int n, double* f) { copy_n(x, n, f); k_Phi_approx(f, n); }
inline void d_Phi_approx(const double* x, const double* f, int n, double* d) {
    for (int i = 0; i < n; ++i) d[i] = f[i] * (1.0 - f[i]) * phi_approx_dpoly(x[i]);
}

inline void v_log_Phi_approx(const double* x, int n, double* f) { copy_n(x, n, f); k_log_Phi_approx(f, n); }
inline void d_log_Phi_approx(const double* x, const double* f, int n, double* d) {
    copy_n(f, n, d); k_exp(d, n);                                                // p = exp(f)
    for (int i = 0; i < n; ++i) d[i] = (1.0 - d[i]) * phi_approx_dpoly(x[i]);
}

// z = C2 * sinh(asinh(C1 * L) / 3): dz/dL = (C2 C1 / 3) * sqrt(1 + (z / C2)^2) / sqrt(1 + (C1 L)^2)
inline double inv_phi_approx_dz_dL(double L, double z) {
    const double zc = z / INV_PHI_APPROX_C2;
    const double Lc = INV_PHI_APPROX_C1 * L;
    return (INV_PHI_APPROX_C2 * INV_PHI_APPROX_C1 / 3.0) * std::sqrt(1.0 + zc * zc) / std::sqrt(1.0 + Lc * Lc);
}

inline void v_inv_Phi_approx(const double* p, int n, double* f) { copy_n(p, n, f); k_inv_Phi_approx(f, n); }
inline void d_inv_Phi_approx(const double* p, const double* z, int n, double* d) {
    for (int i = 0; i < n; ++i) {
        const double L = std::log(p[i] / (1.0 - p[i]));
        d[i] = inv_phi_approx_dz_dL(L, z[i]) / (p[i] * (1.0 - p[i]));
    }
}

inline void v_inv_Phi_approx_from_logit_prob(const double* lp, int n, double* f) { copy_n(lp, n, f); k_inv_Phi_approx_from_logit_prob(f, n); }
inline void d_inv_Phi_approx_from_logit_prob(const double* lp, const double* z, int n, double* d) {
    for (int i = 0; i < n; ++i) d[i] = inv_phi_approx_dz_dL(lp[i], z[i]);
}

// ---- thread-local scratch: reused across calls, so no malloc after warm-up and it stays in cache
struct scratch_t {
    std::vector<double> a, b, c;
    void ensure(int n) { if ((int)a.size() < n) { a.resize(n); b.resize(n); c.resize(n); } }
};
inline scratch_t& scratch() { thread_local scratch_t s; return s; }

// ---- the rev-mode wrapper (see the header comment). Arena per element: 8 (operand vari*) + 24 (result vari).
// Select dependencies at compile time: never reread values which the derivative does not use.
// Keep 'both' as the default for existing callers and the matched pre-optimisation benchmark.
// Unused derivative arguments are nullptr; only the declared buffers are populated.
enum class derivative_values { operand, result, both };
template <derivative_values needed = derivative_values::both, typename T, typename V, typename D, typename InPlace = std::nullptr_t>
inline VecVar rev_apply(const T& x_in, V v, D d, InPlace in_place = nullptr) {
    const int n = static_cast<int>(x_in.size());
    auto& mem = stan::math::ChainableStack::instance_->memalloc_;
    vari** xp = mem.alloc_array<vari*>(n);
    scratch_t& sc = scratch(); sc.ensure(n);
    double* xb = sc.a.data(); double* fb = sc.b.data();
    for (int i = 0; i < n; ++i) { xp[i] = x_in.coeff(i).vi_; xb[i] = xp[i]->val_; }
    if constexpr (std::is_same<InPlace, std::nullptr_t>::value) {
        v(xb, n, fb);
    } else {
        in_place(xb, n);
        fb = xb;
    }
    vari* rv = mem.alloc_array<vari>(n);
    VecVar res(n);
    for (int i = 0; i < n; ++i) {
        ::new (static_cast<void*>(rv + i)) vari(fb[i], false);                    // nochain stack: chain() is empty
        res.coeffRef(i) = var(rv + i);
    }
    stan::math::reverse_pass_callback([xp, rv, n, d]() {
        scratch_t& s2 = scratch(); s2.ensure(n);
        double* xb2 = s2.a.data(); double* fb2 = s2.b.data(); double* db2 = s2.c.data();
        if constexpr (needed == derivative_values::both) {
            for (int i = 0; i < n; ++i) { xb2[i] = xp[i]->val_; fb2[i] = rv[i].val_; }
        } else if constexpr (needed == derivative_values::operand) {
            for (int i = 0; i < n; ++i) xb2[i] = xp[i]->val_;
            fb2 = nullptr;
        } else {
            for (int i = 0; i < n; ++i) fb2[i] = rv[i].val_;
            xb2 = nullptr;
        }
        d(xb2, fb2, n, db2);
        for (int i = 0; i < n; ++i) xp[i]->adj_ += rv[i].adj_ * db2[i];
    });
    return res;
}

// ---- Specialised log / inv_Phi wrappers: same kernels, derivatives and arena ownership as rev_apply.
// Gather once and transform in place. log's reverse pass needs only operand values; inv_Phi's needs
// only result values. Keep adjoint accumulation scalar and ordered: different inputs can alias one vari.
template <bool inverse_normal, typename T>
inline VecVar rev_apply_log_or_inv_Phi(const T& input) {
    const int n = static_cast<int>(input.size());
    auto& arena = stan::math::ChainableStack::instance_->memalloc_;
    vari** input_nodes = arena.alloc_array<vari*>(n);
    scratch_t& buffers = scratch(); buffers.ensure(n);
    double* values = buffers.a.data();
    for (int i = 0; i < n; ++i) {
        input_nodes[i] = input.coeff(i).vi_;
        values[i] = input_nodes[i]->val_;
    }
    if constexpr (inverse_normal) k_inv_Phi(values, n);
    else                          k_log(values, n);
    vari* result_nodes = arena.alloc_array<vari>(n);
    VecVar result(n);
    for (int i = 0; i < n; ++i) {
        ::new (static_cast<void*>(result_nodes + i)) vari(values[i], false);
        result.coeffRef(i) = var(result_nodes + i);
    }
    stan::math::reverse_pass_callback([input_nodes, result_nodes, n]() {
        const auto update_block = [input_nodes, result_nodes](int first, int count) {
            alignas(64) double derivatives[BMVP_LANES];
            if constexpr (!inverse_normal) {
                for (int lane = 0; lane < BMVP_LANES; ++lane)
                    derivatives[lane] = input_nodes[first + (lane < count ? lane : count - 1)]->val_;
#if BMVP_LANES == 8
                simd_store(derivatives, _mm512_div_pd(_mm512_set1_pd(1.0), simd_load(derivatives)));
#else
                simd_store(derivatives, _mm256_div_pd(_mm256_set1_pd(1.0), simd_load(derivatives)));
#endif
            } else {
                alignas(64) double result_values[BMVP_LANES];
                for (int lane = 0; lane < BMVP_LANES; ++lane)
                    result_values[lane] = result_nodes[first + (lane < count ? lane : count - 1)].val_;
                d_inv_Phi(nullptr, result_values, BMVP_LANES, derivatives);
            }
            for (int lane = 0; lane < count; ++lane)
                input_nodes[first + lane]->adj_ += result_nodes[first + lane].adj_ * derivatives[lane];
        };
        // A constant full-block size lets the compiler remove padding/index tests from the hot loop.
        int first = 0;
        for (; first + BMVP_LANES <= n; first += BMVP_LANES) update_block(first, BMVP_LANES);
        if (first < n) update_block(first, n - first);
    });
    return result;
}

// These two value-only paths can apply the kernel directly to the returned Eigen vector.
template <bool inverse_normal, typename T>
inline Eigen::VectorXd prim_apply_log_or_inv_Phi(const T& input) {
    Eigen::VectorXd result = input.template cast<double>();
    if constexpr (inverse_normal) k_inv_Phi(result.data(), static_cast<int>(result.size()));
    else                          k_log(result.data(), static_cast<int>(result.size()));
    return result;
}

// ---- the prim (double) path: one output allocation, same as Stan's own vectorised functions.
template <typename T, typename V, typename InPlace = std::nullptr_t>
inline Eigen::VectorXd prim_apply(const T& x_in, V v, InPlace in_place = nullptr) {
    const int n = static_cast<int>(x_in.size());
    if constexpr (!std::is_same<InPlace, std::nullptr_t>::value) {
        Eigen::VectorXd f = x_in.template cast<double>();
        in_place(f.data(), n);
        return f;
    }
    scratch_t& sc = scratch(); sc.ensure(n);
    double* xb = sc.a.data();
    for (int i = 0; i < n; ++i) xb[i] = x_in.coeff(i);
    Eigen::VectorXd f(n);
    v(xb, n, f.data());
    return f;
}

}  // namespace bmvp_stan_ext

// ---- the Stan-facing functions (global scope; stanc calls them as  fast_Phi(x, pstream__)).
#define BMVP_STAN_EXTERNAL(NAME, V, D, NEEDED, IN_PLACE)                                                       \
template <typename T, stan::require_eigen_col_vector_vt<std::is_arithmetic, T>* = nullptr>                     \
inline Eigen::VectorXd NAME(const T& x, std::ostream* /*pstream__*/) {                                          \
    return bmvp_stan_ext::prim_apply(x, [](const double* a, int n, double* f) { bmvp_stan_ext::V(a, n, f); }, IN_PLACE); \
}                                                                                                                \
template <typename T, stan::require_eigen_col_vector_vt<stan::is_var, T>* = nullptr>                           \
inline Eigen::Matrix<stan::math::var, -1, 1> NAME(const T& x, std::ostream* /*pstream__*/) {                    \
    return bmvp_stan_ext::rev_apply<bmvp_stan_ext::derivative_values::NEEDED>(                                 \
                                       x, [](const double* a, int n, double* f) { bmvp_stan_ext::V(a, n, f); }, \
                                       [](const double* a, const double* f, int n, double* d) { bmvp_stan_ext::D(a, f, n, d); }, IN_PLACE); \
}

#define BMVP_IN_PLACE(K) [](double* values, int n) { bmvp_stan_ext::K(values, n); }
BMVP_STAN_EXTERNAL(fast_Phi, v_Phi, d_Phi, operand, BMVP_IN_PLACE(k_Phi))
BMVP_STAN_EXTERNAL(fast_log_Phi, v_log_Phi, d_log_Phi, operand, BMVP_IN_PLACE(k_log_Phi))
BMVP_STAN_EXTERNAL(fast_exp, v_exp, d_exp, result, BMVP_IN_PLACE(k_exp))
BMVP_STAN_EXTERNAL(fast_log1m, v_log1m, d_log1m, operand, BMVP_IN_PLACE(k_log1m))
BMVP_STAN_EXTERNAL(fast_log1p, v_log1p, d_log1p, operand, BMVP_IN_PLACE(k_log1p))
BMVP_STAN_EXTERNAL(fast_log1p_exp, v_log1p_exp, d_log1p_exp, operand, BMVP_IN_PLACE(k_log1p_exp))
// log1m_exp's near-zero repair still needs the original x: do not overwrite it with exp(x).
BMVP_STAN_EXTERNAL(fast_log1m_exp, v_log1m_exp, d_log1m_exp, both, nullptr)
BMVP_STAN_EXTERNAL(fast_inv_logit, v_inv_logit, d_inv_logit, result, BMVP_IN_PLACE(k_inv_logit))
BMVP_STAN_EXTERNAL(fast_log_inv_logit, v_log_inv_logit, d_log_inv_logit, operand, BMVP_IN_PLACE(k_log_inv_logit))
// Keep the copy path here: in-place evaluation regressed complete-call timings in the matched benchmark.
BMVP_STAN_EXTERNAL(fast_Phi_approx, v_Phi_approx, d_Phi_approx, both, nullptr)
BMVP_STAN_EXTERNAL(fast_log_Phi_approx, v_log_Phi_approx, d_log_Phi_approx, both, BMVP_IN_PLACE(k_log_Phi_approx))
BMVP_STAN_EXTERNAL(fast_inv_Phi_approx, v_inv_Phi_approx, d_inv_Phi_approx, both, BMVP_IN_PLACE(k_inv_Phi_approx))
BMVP_STAN_EXTERNAL(fast_inv_Phi_approx_from_logit_prob, v_inv_Phi_approx_from_logit_prob, d_inv_Phi_approx_from_logit_prob, both, BMVP_IN_PLACE(k_inv_Phi_approx_from_logit_prob))

#undef BMVP_STAN_EXTERNAL
#undef BMVP_IN_PLACE

template <typename T, stan::require_eigen_col_vector_vt<std::is_arithmetic, T>* = nullptr>
inline Eigen::VectorXd fast_log(const T& input, std::ostream* /*pstream__*/) {
    return bmvp_stan_ext::prim_apply_log_or_inv_Phi<false>(input);
}
template <typename T, stan::require_eigen_col_vector_vt<stan::is_var, T>* = nullptr>
inline bmvp_stan_ext::VecVar fast_log(const T& input, std::ostream* /*pstream__*/) {
    return bmvp_stan_ext::rev_apply_log_or_inv_Phi<false>(input);
}
template <typename T, stan::require_eigen_col_vector_vt<std::is_arithmetic, T>* = nullptr>
inline Eigen::VectorXd fast_inv_Phi(const T& input, std::ostream* /*pstream__*/) {
    return bmvp_stan_ext::prim_apply_log_or_inv_Phi<true>(input);
}
template <typename T, stan::require_eigen_col_vector_vt<stan::is_var, T>* = nullptr>
inline bmvp_stan_ext::VecVar fast_inv_Phi(const T& input, std::ostream* /*pstream__*/) {
    return bmvp_stan_ext::rev_apply_log_or_inv_Phi<true>(input);
}

// ---- diagnostic: which kernel path this build took. Declare in Stan as   real bmvp_simd_lanes();
inline double bmvp_simd_lanes(std::ostream* /*pstream__*/) { return static_cast<double>(BMVP_LANES); }
