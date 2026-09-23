
#pragma once

#ifndef FN_SIMD_LEVEL_RESOLVER_HPP
#define FN_SIMD_LEVEL_RESOLVER_HPP


#include <string>
#include <stdexcept>


//// -------------------------------------------------------------------------------------------------------------------------------------------------------------
//// ---- Which SIMD kernel sets this build compiles, and which one a vect_type string selects (added 2026-09-22):
////
////      Until 2026-09-22 the native models' string dispatch (fn_EIGEN_Ref_double -> fn_process_Ref_double_AVX) compiled ONE
////      SIMD level: on an AVX-512 build a vect_type = "AVX2" request ran the 8-lane AVX-512 kernels (and the R front end then
////      stopped on "AVX2"), and fn_log_sum_exp_2d_double silently ran Stan's scalar log_sum_exp for it. The AVX2 kernel
////      header (BayesMVP/math/fast_and_approx_AVX2_fns.hpp) was already compiled on every AVX2-capable build, AVX-512 ones
////      included; only the dispatch ignored it. Now every dispatch layer resolves vect_type through
////      fn_BayesMVP_SIMD_lane_width_for_vect_type() below:
////        "AVX512"        -> 8 lanes (the __m512d kernels, fast_*_AVX512),
////        "AVX2"          -> 4 lanes (the __m256d kernels, fast_*_AVX2; on AVX2-only AND on AVX-512 builds),
////        "Stan" / "Loop" -> 1 (scalar).
////      A level this build did not compile (e.g. "AVX512" on an AVX2-only laptop build) THROWS instead of silently running
////      another level. The R front end (fn_check_native_model_vect_types_and_Phi_types) stops before sampling in that case;
////      the throw is the second line of defence.
////      The dispatch layers that use it: fn_EIGEN_Ref_double (fn_wrappers_overall.hpp), fn_log_sum_exp_2d_double
////      (fn_wrappers_log_sum_exp_SIMD.hpp) and vec_from_string (fn_dispatch_templated.hpp). The exported
////      Rcpp_BayesMVP_SIMD_lane_width_for_vect_type() (Rcpp_SIMD_dispatch_info.cpp) returns this same function's value, so R
////      can confirm which kernel width a vect_type string will actually run.
////      NOTE: vect_type selects BayesMVP's own kernels only. Eigen / Stan-math code in the same binary is vectorised at the
////      build's compile level (AVX-512 on the Local_HPC) whatever vect_type is.
////
//// The two conditions below are the guards of the kernel headers themselves (fast_and_approx_AVX512_fns.hpp needs AVX512F+VL+DQ;
//// the AVX2 kernels use FMA intrinsics, hence __FMA__, as fn_dispatch_templated.hpp already required).
////
#if defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__)
    #define BAYESMVP_COMPILED_AVX512_KERNELS 1
#else
    #define BAYESMVP_COMPILED_AVX512_KERNELS 0
#endif

#if defined(__AVX2__) && defined(__FMA__)
    #define BAYESMVP_COMPILED_AVX2_KERNELS 1
#else
    #define BAYESMVP_COMPILED_AVX2_KERNELS 0
#endif




//// Highest SIMD level compiled into this build ("AVX512", "AVX2" or "none"). The default vect_type of the native models is
//// this level (R: default_vect_type_for_native_models in init_hard_coded_model_args), so defaults are unchanged.
inline std::string fn_BayesMVP_best_compiled_SIMD_level() {

      #if BAYESMVP_COMPILED_AVX512_KERNELS
            return "AVX512";
      #elif BAYESMVP_COMPILED_AVX2_KERNELS
            return "AVX2";
      #else
            return "none";
      #endif

}




//// Message used when a SIMD level is requested that this build did not compile.
inline std::string fn_BayesMVP_SIMD_level_not_compiled_message(const std::string &vect_type) {

      std::string compiled_SIMD_levels_as_text = "";
      #if BAYESMVP_COMPILED_AVX512_KERNELS
            compiled_SIMD_levels_as_text += "AVX512 ";
      #endif
      #if BAYESMVP_COMPILED_AVX2_KERNELS
            compiled_SIMD_levels_as_text += "AVX2 ";
      #endif
      if (compiled_SIMD_levels_as_text.empty()) compiled_SIMD_levels_as_text = "none ";

      return std::string("BayesMVP: vect_type = '") + vect_type + "' was requested, but this BayesMVP build does not contain the " +
             ((vect_type == "AVX512") ? "AVX-512 (8-lane)" : "AVX2 (4-lane)") +
             " kernels (compiled SIMD levels: " + compiled_SIMD_levels_as_text +
             "). Use a compiled level, 'Stan' or 'Loop', or rebuild BayesMVP on a machine whose compiler flags provide " + vect_type +
             " (an AVX-512 build contains both the AVX-512 and the AVX2 kernels).";

}




//// Lane width that BayesMVP's SIMD dispatch uses for vect_type: 8 = AVX-512 kernels, 4 = AVX2 kernels, 1 = scalar ("Stan", "Loop").
//// Throws std::invalid_argument for a SIMD level this build did not compile and for any other string (no silent substitution).
//// Called once per dispatch call (never per element).
inline int fn_BayesMVP_SIMD_lane_width_for_vect_type(const std::string &vect_type) {

      if (vect_type == "AVX512") {
            #if BAYESMVP_COMPILED_AVX512_KERNELS
                  return 8;
            #else
                  throw std::invalid_argument(fn_BayesMVP_SIMD_level_not_compiled_message(vect_type));
            #endif
      }

      if (vect_type == "AVX2") {
            #if BAYESMVP_COMPILED_AVX2_KERNELS
                  return 4;
            #else
                  throw std::invalid_argument(fn_BayesMVP_SIMD_level_not_compiled_message(vect_type));
            #endif
      }

      if ((vect_type == "Stan") || (vect_type == "Loop")) {
            return 1;
      }

      throw std::invalid_argument(std::string("BayesMVP: unknown vect_type '") + vect_type +
                                  "' (accepted: 'AVX512', 'AVX2', 'Stan', 'Loop'; the SIMD levels only if compiled into this build).");

}




#endif

