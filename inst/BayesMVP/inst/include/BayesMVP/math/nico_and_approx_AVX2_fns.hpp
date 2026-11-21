//// nico_and_approx_AVX2_fns.hpp
////
//// ---- Moved into NicoSIMD, where every SIMD function lives: this header only includes NicoSIMD's of the
////      same name, math/nico_and_approx_AVX2_fns.hpp,
//// whose maths is that of the copy kept here before, token for token.
////      The copy is on the 20 TB drive (Work_backup_cumulative).
////
#pragma once
#if __has_include(<NicoSIMD/math/nico_and_approx_AVX2_fns.hpp>)
#include <NicoSIMD/math/nico_and_approx_AVX2_fns.hpp>
#elif __has_include("../../../../NicoSIMD/include/NicoSIMD/math/nico_and_approx_AVX2_fns.hpp")
//// (no include path to NicoSIMD: the installed NicoSIMD beside this package in the same library)
#include "../../../../NicoSIMD/include/NicoSIMD/math/nico_and_approx_AVX2_fns.hpp"
#else
//// (the development tree: NicoSIMD in R_packages/NicoStan/inst/NicoSIMD/inst/include/)
#include "../../../../../../../NicoStan/inst/NicoSIMD/inst/include/NicoSIMD/math/nico_and_approx_AVX2_fns.hpp"
#endif






















