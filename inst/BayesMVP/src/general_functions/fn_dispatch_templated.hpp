
#pragma once
////
//// fn_dispatch_templated.hpp  (v2: AVX512 + AVX2 + Scalar)
////
//// Compile-time dispatch for the element-wise math kernels. Replaces the runtime std::string
//// dispatch in fn_EIGEN_double / fn_EIGEN_Ref_double / fn_process_Ref_double_AVX.
////
////   apply_inplace<VEC, Fn::exp>(x);                 // x overwritten in place, any length
////   apply_col_inplace<VEC, Fn::log>(M, t);          // one column of a col-major matrix
////   Eigen::VectorXd y = apply_copy<VEC, Fn::Phi>(expr);
////
//// VEC is Vec::AVX512, Vec::AVX2 or Vec::Scalar, chosen ONCE per driver call with
//// vec_from_string() and threaded down as a template parameter. Use DISPATCH_VEC(...) in the
//// non-template wrappers so all three instantiations are covered in one line.
////
//// vec_from_string() resolves the string with fn_BayesMVP_SIMD_lane_width_for_vect_type()
//// (fns_SIMD_and_wrappers/fn_SIMD_level_resolver.hpp), the same function the string dispatch
//// (fn_EIGEN_Ref_double) uses: "AVX512" -> Vec::AVX512, "AVX2" -> Vec::AVX2 (the genuine 4-lane
//// kernels, also on AVX-512 builds), "Stan"/"Loop" -> Vec::Scalar.
//// a level that is not compiled in (e.g. AVX512 on an AVX2-only laptop) now THROWS.
//// It used to fall back silently (AVX512 -> AVX2 -> Scalar), so an R-side vect_type = "AVX512" on
//// an AVX2 machine ran AVX2 without saying so (the R front end now stops before sampling in that
//// case). apply_raw<>'s compile-time fallback below is kept only so that DISPATCH_VEC can
//// instantiate all three levels; vec_from_string never selects a level that is not compiled.
////
//// REQUIRES: fast_and_approx_AVX512_fns.hpp and fast_and_approx_AVX2_fns.hpp (the fast_*_AVX512
////           / fast_*_AVX2 kernels) included first, and stan/math for the scalar path.
////

#include <immintrin.h>
#include <Eigen/Dense>
#include <stan/math/prim.hpp>
#include <cmath>
#include <string>
#include <stdexcept>
#include <iostream>

#include "fns_SIMD_and_wrappers/fn_SIMD_level_resolver.hpp"

//// one definition of "compiled", shared with the string dispatch (same conditions as before:
//// AVX512F+VL+DQ for the 8-lane kernels, AVX2+FMA for the 4-lane kernels).
#define BMVP_HAS_AVX512 BAYESMVP_COMPILED_AVX512_KERNELS
#define BMVP_HAS_AVX2   BAYESMVP_COMPILED_AVX2_KERNELS

//// =====================================================================================
//// 1. Enums
//// =====================================================================================
enum class Fn {
  exp, log, log1p, log1m, log1p_exp,
  logit, inv_logit, log_inv_logit,
  Phi, Phi_approx, log_Phi_approx,
  inv_Phi, inv_Phi_approx, inv_Phi_approx_from_logit_prob,
  tanh
};

enum class Vec { AVX512, AVX2, Scalar };

//// Best level actually compiled into this binary:
inline Vec best_compiled_vec() {
#if BMVP_HAS_AVX512
  return Vec::AVX512;
#elif BMVP_HAS_AVX2
  return Vec::AVX2;
#else
  return Vec::Scalar;
#endif
}

//// Translate the Model_args string to the enum ONCE per driver call.
//// resolved by fn_BayesMVP_SIMD_lane_width_for_vect_type(), which THROWS for a level that is not compiled and for
//// unknown strings (including "AVX", which used to map silently to Scalar; the R front end already rejects it). Previously a
//// request for a level that was not compiled was clamped silently to the best compiled level.
inline Vec vec_from_string(const std::string &vect_type) {
  const int SIMD_lane_width = fn_BayesMVP_SIMD_lane_width_for_vect_type(vect_type);
  if (SIMD_lane_width == 8) return Vec::AVX512;
  if (SIMD_lane_width == 4) return Vec::AVX2;
  return Vec::Scalar;
}

inline int vec_width(Vec v) { return v == Vec::AVX512 ? 8 : (v == Vec::AVX2 ? 4 : 1); }

//// One-line three-way dispatch for the non-template driver wrappers:
////   DISPATCH_VEC(vec, fn_lp_grad_..._T, out_mat, theta, ...);
#define DISPATCH_VEC(VEC_RT, FN_T, ...)                                         \
  do {                                                                         \
    switch (VEC_RT) {                                                          \
      case Vec::AVX512: FN_T<Vec::AVX512>(__VA_ARGS__); break;                 \
      case Vec::AVX2:   FN_T<Vec::AVX2>(__VA_ARGS__);   break;                 \
      default:          FN_T<Vec::Scalar>(__VA_ARGS__); break;                 \
    }                                                                          \
  } while (0)




//// =====================================================================================
//// 2. AVX-512 kernels (8 doubles). `checks` = true: range-checked; false: *_wo_checks_*
//// =====================================================================================
#if BMVP_HAS_AVX512
template <Fn fn, bool checks> ALWAYS_INLINE __m512d kernel_AVX512(__m512d x);
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::exp, true >(__m512d x) { return fast_exp_1_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::exp, false>(__m512d x) { return fast_exp_1_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::log, true >(__m512d x) { return fast_log_1_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::log, false>(__m512d x) { return fast_log_1_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::log1p, true >(__m512d x) { return fast_log1p_1_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::log1p, false>(__m512d x) { return fast_log1p_1_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::log1m, true >(__m512d x) { return fast_log1m_1_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::log1m, false>(__m512d x) { return fast_log1m_1_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::log1p_exp, true >(__m512d x) { return fast_log1p_exp_1_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::log1p_exp, false>(__m512d x) { return fast_log1p_exp_1_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::logit, true >(__m512d x) { return fast_logit_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::logit, false>(__m512d x) { return fast_logit_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::inv_logit, true >(__m512d x) { return fast_inv_logit_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::inv_logit, false>(__m512d x) { return fast_inv_logit_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::log_inv_logit, true >(__m512d x) { return fast_log_inv_logit_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::log_inv_logit, false>(__m512d x) { return fast_log_inv_logit_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::Phi, true >(__m512d x) { return fast_Phi_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::Phi, false>(__m512d x) { return fast_Phi_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::Phi_approx, true >(__m512d x) { return fast_Phi_approx_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::Phi_approx, false>(__m512d x) { return fast_Phi_approx_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::log_Phi_approx, true >(__m512d x) { return fast_log_Phi_approx_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::log_Phi_approx, false>(__m512d x) { return fast_log_Phi_approx_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::inv_Phi, true >(__m512d x) { return fast_inv_Phi_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::inv_Phi, false>(__m512d x) { return fast_inv_Phi_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::inv_Phi_approx, true >(__m512d x) { return fast_inv_Phi_approx_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::inv_Phi_approx, false>(__m512d x) { return fast_inv_Phi_approx_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::inv_Phi_approx_from_logit_prob, true >(__m512d x) { return fast_inv_Phi_approx_from_logit_prob_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::inv_Phi_approx_from_logit_prob, false>(__m512d x) { return fast_inv_Phi_approx_from_logit_prob_wo_checks_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::tanh, true >(__m512d x) { return fast_tanh_AVX512(x); }
template <> ALWAYS_INLINE __m512d kernel_AVX512<Fn::tanh, false>(__m512d x) { return fast_tanh_wo_checks_AVX512(x); }
#endif




//// =====================================================================================
//// 3. AVX2 kernels (4 doubles). Same names as the AVX512 ones with _AVX2 — if any of your
////    AVX2 kernels is named differently, edit that ONE line; the compiler will tell you which.
//// =====================================================================================
#if BMVP_HAS_AVX2
template <Fn fn, bool checks> ALWAYS_INLINE __m256d kernel_AVX2(__m256d x);
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::exp, true >(__m256d x) { return fast_exp_1_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::exp, false>(__m256d x) { return fast_exp_1_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::log, true >(__m256d x) { return fast_log_1_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::log, false>(__m256d x) { return fast_log_1_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::log1p, true >(__m256d x) { return fast_log1p_1_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::log1p, false>(__m256d x) { return fast_log1p_1_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::log1m, true >(__m256d x) { return fast_log1m_1_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::log1m, false>(__m256d x) { return fast_log1m_1_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::log1p_exp, true >(__m256d x) { return fast_log1p_exp_1_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::log1p_exp, false>(__m256d x) { return fast_log1p_exp_1_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::logit, true >(__m256d x) { return fast_logit_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::logit, false>(__m256d x) { return fast_logit_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::inv_logit, true >(__m256d x) { return fast_inv_logit_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::inv_logit, false>(__m256d x) { return fast_inv_logit_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::log_inv_logit, true >(__m256d x) { return fast_log_inv_logit_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::log_inv_logit, false>(__m256d x) { return fast_log_inv_logit_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::Phi, true >(__m256d x) { return fast_Phi_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::Phi, false>(__m256d x) { return fast_Phi_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::Phi_approx, true >(__m256d x) { return fast_Phi_approx_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::Phi_approx, false>(__m256d x) { return fast_Phi_approx_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::log_Phi_approx, true >(__m256d x) { return fast_log_Phi_approx_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::log_Phi_approx, false>(__m256d x) { return fast_log_Phi_approx_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::inv_Phi, true >(__m256d x) { return fast_inv_Phi_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::inv_Phi, false>(__m256d x) { return fast_inv_Phi_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::inv_Phi_approx, true >(__m256d x) { return fast_inv_Phi_approx_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::inv_Phi_approx, false>(__m256d x) { return fast_inv_Phi_approx_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::inv_Phi_approx_from_logit_prob, true >(__m256d x) { return fast_inv_Phi_approx_from_logit_prob_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::inv_Phi_approx_from_logit_prob, false>(__m256d x) { return fast_inv_Phi_approx_from_logit_prob_wo_checks_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::tanh, true >(__m256d x) { return fast_tanh_AVX2(x); }
template <> ALWAYS_INLINE __m256d kernel_AVX2<Fn::tanh, false>(__m256d x) { return fast_tanh_wo_checks_AVX2(x); }
#endif




//// =====================================================================================
//// 4. Scalar kernels. Same formulas as the SIMD *approx* kernels; stan::math for exact ones.
//// =====================================================================================
template <Fn fn> ALWAYS_INLINE double kernel_scalar(double x);
template <> ALWAYS_INLINE double kernel_scalar<Fn::exp>(double x)           { return stan::math::exp(x); }
template <> ALWAYS_INLINE double kernel_scalar<Fn::log>(double x)           { return stan::math::log(x); }
template <> ALWAYS_INLINE double kernel_scalar<Fn::log1p>(double x)         { return stan::math::log1p(x); }
template <> ALWAYS_INLINE double kernel_scalar<Fn::log1m>(double x)         { return stan::math::log1m(x); }
template <> ALWAYS_INLINE double kernel_scalar<Fn::log1p_exp>(double x)     { return stan::math::log1p_exp(x); }
template <> ALWAYS_INLINE double kernel_scalar<Fn::logit>(double x)         { return stan::math::logit(x); }
template <> ALWAYS_INLINE double kernel_scalar<Fn::inv_logit>(double x)     { return stan::math::inv_logit(x); }
template <> ALWAYS_INLINE double kernel_scalar<Fn::log_inv_logit>(double x) { return stan::math::log_inv_logit(x); }
template <> ALWAYS_INLINE double kernel_scalar<Fn::Phi>(double x)           { return stan::math::Phi(x); }
template <> ALWAYS_INLINE double kernel_scalar<Fn::Phi_approx>(double x)    { return stan::math::inv_logit(x * (0.07056 * x * x + 1.5976)); }
template <> ALWAYS_INLINE double kernel_scalar<Fn::log_Phi_approx>(double x){ return stan::math::log_inv_logit(x * (0.07056 * x * x + 1.5976)); }
template <> ALWAYS_INLINE double kernel_scalar<Fn::inv_Phi>(double x)       { return stan::math::inv_Phi(x); }
template <> ALWAYS_INLINE double kernel_scalar<Fn::inv_Phi_approx>(double p) {
  return 5.494448514153059 * std::sinh(std::asinh(-0.34176618822627863 * std::log(1.0 / p - 1.0)) / 3.0);
}
template <> ALWAYS_INLINE double kernel_scalar<Fn::inv_Phi_approx_from_logit_prob>(double lp) {
  return 5.494448514153059 * std::sinh(std::asinh(0.34176618822627863 * lp) / 3.0);
}
template <> ALWAYS_INLINE double kernel_scalar<Fn::tanh>(double x)          { return std::tanh(x); }




//// =====================================================================================
//// 5. Loops on n contiguous doubles at p, in place. Masked tails => any length works.
//// =====================================================================================
#if BMVP_HAS_AVX512
template <Fn fn, bool checks>
inline void apply_AVX512_raw(double *p, const int n) {
  int i = 0;
  for (; i + 8 <= n; i += 8) _mm512_storeu_pd(p + i, kernel_AVX512<fn, checks>(_mm512_loadu_pd(p + i)));
  if (i < n) {
    const __mmask8 m = static_cast<__mmask8>((1u << (n - i)) - 1u);
    _mm512_mask_storeu_pd(p + i, m, kernel_AVX512<fn, checks>(_mm512_maskz_loadu_pd(m, p + i)));
  }
}
#endif

#if BMVP_HAS_AVX2
ALWAYS_INLINE __m256i avx2_tail_mask(const int n_rem) {   //// n_rem in 1..3; lane on iff top bit set
  return _mm256_setr_epi64x(n_rem > 0 ? -1LL : 0LL, n_rem > 1 ? -1LL : 0LL, n_rem > 2 ? -1LL : 0LL, 0LL);
}
template <Fn fn, bool checks>
inline void apply_AVX2_raw(double *p, const int n) {
  int i = 0;
  for (; i + 4 <= n; i += 4) _mm256_storeu_pd(p + i, kernel_AVX2<fn, checks>(_mm256_loadu_pd(p + i)));
  if (i < n) {
    const __m256i m = avx2_tail_mask(n - i);
    _mm256_maskstore_pd(p + i, m, kernel_AVX2<fn, checks>(_mm256_maskload_pd(p + i, m)));   //// maskload zero-fills off lanes
  }
}
#endif

template <Fn fn>
inline void apply_scalar_raw(double *p, const int n) { for (int i = 0; i < n; ++i) p[i] = kernel_scalar<fn>(p[i]); }




//// Entry point on raw memory: the level is picked at compile time; a level not compiled in
//// falls back to the next one down.
template <Vec vec, Fn fn, bool checks = true>
inline void apply_raw(double *p, const int n) {
  if constexpr (vec == Vec::AVX512) {
    #if BMVP_HAS_AVX512
      apply_AVX512_raw<fn, checks>(p, n);
    #elif BMVP_HAS_AVX2
      apply_AVX2_raw<fn, checks>(p, n);
    #else
      apply_scalar_raw<fn>(p, n);
    #endif
  } else if constexpr (vec == Vec::AVX2) {
    #if BMVP_HAS_AVX2
      apply_AVX2_raw<fn, checks>(p, n);
    #else
      apply_scalar_raw<fn>(p, n);
    #endif
  } else {
    apply_scalar_raw<fn>(p, n);
  }
}




//// =====================================================================================
//// 6. Eigen conveniences
//// =====================================================================================
template <Vec vec, Fn fn, bool checks = true, typename Derived>
inline void apply_inplace(Eigen::PlainObjectBase<Derived> &x) {
  apply_raw<vec, fn, checks>(x.data(), static_cast<int>(x.size()));
}
template <Vec vec, Fn fn, bool checks = true, typename T>
inline void apply_inplace(Eigen::Ref<T> x) {
  if (x.innerStride() != 1) throw std::runtime_error("apply_inplace: non-unit inner stride");
  apply_raw<vec, fn, checks>(x.data(), static_cast<int>(x.size()));
}
//// One column of ANY column-major Eigen object (Matrix, Ref, Block) — contiguous, no copy.
template <Vec vec, Fn fn, bool checks = true, typename Derived>
inline void apply_col_inplace(Eigen::DenseBase<Derived> &M, const int col) {
  auto column = M.derived().col(col);
  apply_raw<vec, fn, checks>(column.data(), static_cast<int>(M.rows()));
}
//// Copying version for expression inputs (blocks, indexed views, arithmetic results). ALLOCATES.
template <Vec vec, Fn fn, bool checks = true, typename Derived>
inline Eigen::Matrix<double, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
apply_copy(const Eigen::MatrixBase<Derived> &x) {
  Eigen::Matrix<double, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> out = x;
  apply_raw<vec, fn, checks>(out.data(), static_cast<int>(out.size()));
  return out;
}
template <Vec vec, Fn fn, bool checks = true, typename Derived>
inline Eigen::Matrix<double, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
apply_copy(const Eigen::ArrayBase<Derived> &x) { return apply_copy<vec, fn, checks>(x.matrix()); }




//// =====================================================================================
//// 7. Self-test: max |SIMD - scalar| per Fn for every level compiled in. Run once per machine.
////    Expected: ~1e-14 for exp/log/logit/tanh/inv_Phi, ~1e-7 for Phi (Abramowitz-Stegun).
//// =====================================================================================
template <Vec vec, Fn fn>
inline double dispatch_self_test_one(const double lo, const double hi, const int n = 1001) {
  Eigen::Matrix<double, -1, 1> x = Eigen::Matrix<double, -1, 1>::LinSpaced(n, lo, hi);
  Eigen::Matrix<double, -1, 1> a = x, s = x;
  apply_inplace<vec, fn>(a);
  apply_inplace<Vec::Scalar, fn>(s);
  return (a - s).cwiseAbs().maxCoeff();
}
template <Vec vec>
inline void dispatch_self_test_level(const char *name) {
  std::cout << "---- " << name << " ----\n";
  std::cout << "exp            " << dispatch_self_test_one<vec, Fn::exp>(-50.0, 50.0)   << "\n";
  std::cout << "log            " << dispatch_self_test_one<vec, Fn::log>(1e-12, 1e6)    << "\n";
  std::cout << "log1p          " << dispatch_self_test_one<vec, Fn::log1p>(-0.999, 100) << "\n";
  std::cout << "log1m          " << dispatch_self_test_one<vec, Fn::log1m>(-100, 0.999) << "\n";
  std::cout << "log1p_exp      " << dispatch_self_test_one<vec, Fn::log1p_exp>(-50, 50) << "\n";
  std::cout << "logit          " << dispatch_self_test_one<vec, Fn::logit>(1e-9, 1-1e-9)<< "\n";
  std::cout << "inv_logit      " << dispatch_self_test_one<vec, Fn::inv_logit>(-40, 40) << "\n";
  std::cout << "log_inv_logit  " << dispatch_self_test_one<vec, Fn::log_inv_logit>(-40, 40) << "\n";
  std::cout << "Phi            " << dispatch_self_test_one<vec, Fn::Phi>(-8, 8)         << "\n";
  std::cout << "Phi_approx     " << dispatch_self_test_one<vec, Fn::Phi_approx>(-8, 8)  << "\n";
  std::cout << "log_Phi_approx " << dispatch_self_test_one<vec, Fn::log_Phi_approx>(-30, 8) << "\n";
  std::cout << "inv_Phi        " << dispatch_self_test_one<vec, Fn::inv_Phi>(1e-6, 1-1e-6) << "\n";
  std::cout << "inv_Phi_approx " << dispatch_self_test_one<vec, Fn::inv_Phi_approx>(1e-6, 1-1e-6) << "\n";
  std::cout << "inv_Phi_apx_lgt" << dispatch_self_test_one<vec, Fn::inv_Phi_approx_from_logit_prob>(-15, 15) << "\n";
  std::cout << "tanh           " << dispatch_self_test_one<vec, Fn::tanh>(-20, 20)      << "\n";
}
inline void dispatch_self_test() {
#if BMVP_HAS_AVX512
  dispatch_self_test_level<Vec::AVX512>("AVX512");
#else
  std::cout << "AVX512: not compiled in\n";
#endif
#if BMVP_HAS_AVX2
  dispatch_self_test_level<Vec::AVX2>("AVX2");
#else
  std::cout << "AVX2: not compiled in\n";
#endif
  //// tail test: a length that is not a multiple of 8 or 4, through the best level
  Eigen::Matrix<double, -1, 1> x = Eigen::Matrix<double, -1, 1>::LinSpaced(13, -3.0, 3.0), s = x;
  apply_inplace<Vec::AVX512, Fn::exp>(x);  apply_inplace<Vec::Scalar, Fn::exp>(s);
  std::cout << "tail (n=13)    " << (x - s).cwiseAbs().maxCoeff() << "\n";
}



