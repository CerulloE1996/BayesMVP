#pragma once

//// (x86-64 only: ARM CPUs have no AVX)
//// The AVX2 kernels below are compiled on AVX-512 builds too, so that vect_type = "AVX2" runs the 4-lane kernels
//// there instead of falling through fn_log_sum_exp_2d_double below to Stan's scalar log_sum_exp.
// Eigen::Matrix<double, -1, 2> &x,
//// a SIMD level this build did not compile THROWS (as in fn_EIGEN_Ref_double;
//// fn_SIMD_level_resolver.hpp), instead of silently falling through to the Stan return at the bottom of
//// this function. Plain string comparisons (no extra per-call cost).

#include <NicoSIMD/kernel_dispatch_and_vector_loops_of_the_BayesMVP_manual_gradient_models/fns_SIMD_and_wrappers/fn_wrappers_log_sum_exp_SIMD.hpp>






















