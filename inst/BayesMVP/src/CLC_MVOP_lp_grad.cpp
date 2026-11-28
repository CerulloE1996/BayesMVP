//// -------------------------------------------------------------------------------------
//// CLC_MVOP_lp_grad.cpp
////
//// Dedicated compilation unit for the four-class Co-morbidity Latent Class Multivariate
//// Ordinal Probit (CLC_MVOP) manual-gradient and autodiff log-density evaluators.
//// -------------------------------------------------------------------------------------

#define NICOSTAN_WITH_BUILTINS 1
#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#define ENABLE_OPEN_MP 1

#if ENABLE_OPEN_MP == 1
      #if __has_include("omp.h")
            #include "omp.h"
      #endif
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static bool g_debug_cutpoint_grads = false;

//// -------------------------------------------------------------------------------------
//// Stan Math includes
//// -------------------------------------------------------------------------------------
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <stan/math.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/prim/err/check_pos_definite.hpp>
#include <stan/math/prim/err/check_square.hpp>
#include <stan/math/prim/err/check_symmetric.hpp>
#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/sqrt.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/transpose.hpp>
#include <stan/math/prim/fun/dot_product.hpp>
#include <stan/math/prim/fun/norm2.hpp>
#include <stan/math/prim/fun/diagonal.hpp>
#include <stan/math/prim/fun/eigenvalues_sym.hpp>
#include <stan/math/prim/fun/diag_post_multiply.hpp>
#include <stan/math/prim/prob/multi_normal_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/lkj_corr_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/beta_lpdf.hpp>

#undef OUT
#include <RcppEigen.h>
#include <unsupported/Eigen/SpecialFunctions>

//// -------------------------------------------------------------------------------------
//// Runtime configuration and SIMD wrappers
//// -------------------------------------------------------------------------------------
#include <NicoStan/runtime/initial_config.hpp>
#include <NicoStan/runtime/SIMD_config.hpp>
#include <NicoStan/runtime/eigen_config.hpp>

#include <RcppParallel.h>

#include "general_functions/var_fns.hpp"
#include "general_functions/double_fns.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_Stan.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_Loop.hpp"
#include "general_functions/fns_SIMD_and_wrappers/nico_and_approx_AVX2_fns.hpp"
#include "general_functions/fns_SIMD_and_wrappers/nico_and_approx_AVX512_fns.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_SIMD_AVX_general.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_overall.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_log_sum_exp_dbl.hpp"
#include "general_functions/fns_SIMD_and_wrappers/fn_wrappers_log_sum_exp_SIMD.hpp"
#include "general_functions/fn_dispatch_templated.hpp"

#include <NicoStan/runtime/general_functions/array_creators_Eigen_fns.hpp>
#include <NicoStan/runtime/general_functions/array_creators_other_fns.hpp>
#include <NicoStan/runtime/general_functions/structures.hpp>
#include <NicoStan/runtime/general_functions/message_of_the_first_failed_gradient_of_each_chain.hpp>
#include <NicoStan/runtime/general_functions/classes.hpp>

#include <Rcpp.h>

#include <NicoStan/runtime/general_functions/misc_helper_fns_1.hpp>
#include <NicoStan/runtime/general_functions/misc_helper_fns_2.hpp>

//// -------------------------------------------------------------------------------------
//// Shared MVP/MVOP helpers and CLC_MVOP evaluators
//// -------------------------------------------------------------------------------------
#include "MVP_functions/MVP_manual_trans_and_J_fns.hpp"
#include "MVP_functions/MVP_helpers_migrated.hpp"
#include "MVP_functions/MVP_manual_grad_calc_fns.hpp"
#include "MVP_functions/MVP_log_scale_grad_calc_fns.hpp"
#include "MVP_functions/MVP_log_scale_grad_calc_fns_T.hpp"
#include "MVP_functions/MVP_and_MVOP_helper_fns.hpp"
#include "MVP_functions/migration_MVOP_helpers.hpp"
#include "MVP_functions/MVOP_manual_grad_calc_fns.hpp"
#include "MVP_functions/MVOP_PartialLog_helper_fns.hpp"
#include "MVP_functions/MVP_and_MVOP_missing_results_GHK_fns.hpp"
#include "MVP_functions/PartialLog_nuisance_chunk_relayout.hpp"
#include "MVP_functions/MVOP_lp_grad_common_T.hpp"
#include "MVP_functions/MVOP_lp_grad_MD_AD_fns.hpp"

#include "MVP_functions/CLC_MVOP_lp_grad_decl.hpp"
#include "MVP_functions/CLC_MVOP_lp_grad_MD_AD_fns.hpp"
