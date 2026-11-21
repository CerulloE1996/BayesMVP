//// stan_external_functions.hpp
////
//// ---- Moved into NicoSIMD, where every SIMD function lives: this header only includes NicoSIMD's of the
////      same name, stan_external_functions_with_the_fast_names_of_BayesMVP.hpp,
//// whose maths is that of the copy kept here before, token for token.
////      The copy is on the 20 TB drive (Work_backup_cumulative).
////
#pragma once
#if __has_include(<NicoSIMD/stan_external_functions_and_the_fused_ordered_logistic_glm_GP_marginal_and_tensor_product_glm.hpp>)
//// (the layer that gave old names to the nico_ functions is gone: every call is by its nico_ name)
#include <NicoSIMD/stan_external_functions_and_the_fused_ordered_logistic_glm_GP_marginal_and_tensor_product_glm.hpp>
#elif __has_include("../../../NicoSIMD/include/NicoSIMD/stan_external_functions_and_the_fused_ordered_logistic_glm_GP_marginal_and_tensor_product_glm.hpp")
//// (no include path to NicoSIMD: the installed NicoSIMD beside this package in the same library)
//// (the layer that gave old names to the nico_ functions is gone: every call is by its nico_ name)
#include "../../../NicoSIMD/include/NicoSIMD/stan_external_functions_and_the_fused_ordered_logistic_glm_GP_marginal_and_tensor_product_glm.hpp"
#else
//// (the development tree: NicoSIMD in R_packages/NicoStan/inst/NicoSIMD/inst/include/)
//// (the layer that gave old names to the nico_ functions is gone: every call is by its nico_ name)
#include "../../../../../../NicoStan/inst/NicoSIMD/inst/include/NicoSIMD/stan_external_functions_and_the_fused_ordered_logistic_glm_GP_marginal_and_tensor_product_glm.hpp"
#endif






















