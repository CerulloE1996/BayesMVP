
#include <Rcpp.h>

#include <string>
#include <vector>

#include "general_functions/fns_SIMD_and_wrappers/fn_SIMD_level_resolver.hpp"

// [[Rcpp::plugins(cpp17)]]


//// ---- SIMD dispatch information for the R front end (added 2026-09-22).
////
////      This translation unit is compiled with the same Makevars flags as main.cpp, so the BAYESMVP_COMPILED_* macros describe the
////      SAME build as the likelihood code, and fn_BayesMVP_SIMD_lane_width_for_vect_type() is the SAME inline function that
////      fn_EIGEN_Ref_double (fn_wrappers_overall.hpp), fn_log_sum_exp_2d_double (fn_wrappers_log_sum_exp_SIMD.hpp) and
////      vec_from_string (fn_dispatch_templated.hpp) switch on. So the lane width reported here is the kernel width those
////      dispatchers run for a vect_type string: 8 = AVX-512 kernels, 4 = AVX2 kernels, 1 = scalar ("Stan", "Loop").
////      Used by fn_check_native_model_vect_types_and_Phi_types (R_fns_init_hard_coded_models.R) and the vect_type tests.




//// Rcpp_BayesMVP_SIMD_lane_width_for_vect_type
////
//// Lane width that BayesMVP's native-model SIMD dispatch uses for a vect_type string: 8 (AVX-512 kernels), 4 (AVX2 kernels) or
//// 1 (scalar: "Stan", "Loop"). Stops with an error for a SIMD level this build did not compile and for unknown strings.
// [[Rcpp::export]]
int Rcpp_BayesMVP_SIMD_lane_width_for_vect_type(const std::string vect_type) {

      return fn_BayesMVP_SIMD_lane_width_for_vect_type(vect_type);

}




//// Rcpp_BayesMVP_compiled_SIMD_levels
////
//// SIMD kernel sets compiled into this BayesMVP build, highest first: c("AVX512", "AVX2") on an AVX-512 build (it compiles both),
//// "AVX2" on an AVX2-only build, character(0) if neither.
// [[Rcpp::export]]
Rcpp::CharacterVector Rcpp_BayesMVP_compiled_SIMD_levels() {

      std::vector<std::string> compiled_SIMD_levels;

      #if BAYESMVP_COMPILED_AVX512_KERNELS
            compiled_SIMD_levels.push_back("AVX512");
      #endif
      #if BAYESMVP_COMPILED_AVX2_KERNELS
            compiled_SIMD_levels.push_back("AVX2");
      #endif

      return Rcpp::wrap(compiled_SIMD_levels);

}

