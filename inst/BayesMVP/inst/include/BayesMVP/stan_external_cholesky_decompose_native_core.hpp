//// stan_external_cholesky_decompose_native_core.hpp
////
//// ---- Moved into NicoSIMD, where every SIMD function lives: this header only includes NicoSIMD's of the
////      same name, stan_external_cholesky_decompose_native_core.hpp,
//// whose maths is that of the copy kept here before, token for token.
////      The copy is on the 20 TB drive (Work_backup_cumulative).
////
#pragma once
#if __has_include(<NicoSIMD/stan_external_cholesky_decompose_native_core.hpp>)
//// (the layer that gave old names to the nico_ functions is gone: every call is by its nico_ name)
#include <NicoSIMD/stan_external_cholesky_decompose_native_core.hpp>
#elif __has_include("../../../NicoSIMD/include/NicoSIMD/stan_external_cholesky_decompose_native_core.hpp")
//// (no include path to NicoSIMD: the installed NicoSIMD beside this package in the same library)
//// (the layer that gave old names to the nico_ functions is gone: every call is by its nico_ name)
#include "../../../NicoSIMD/include/NicoSIMD/stan_external_cholesky_decompose_native_core.hpp"
#else
//// (the development tree: NicoSIMD in R_packages/NicoStan/inst/NicoSIMD/inst/include/)
//// (the layer that gave old names to the nico_ functions is gone: every call is by its nico_ name)
#include "../../../../../../NicoStan/inst/NicoSIMD/inst/include/NicoSIMD/stan_external_cholesky_decompose_native_core.hpp"
#endif






















