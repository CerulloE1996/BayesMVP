#!/bin/bash
##
## ==== run_test_fast_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood.sh =======================
##
## ---- Builds and runs the C++ test of BayesMVP's fused GHK log likelihood of a finite mixture of multivariate
##      (ordered) probit models against Stan math's autodiff, with the AVX-512 kernels and with the AVX2 kernels
##      (the same AVX-512 compiler flags; AVX2 differs only by -DBAYESMVP_FORCE_AVX2), on CPUs 180-191 at the
##      lowest priority. Output: the test's lines, in outputs/ next to this script.
##
## Usage:   bash run_test_fast_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood.sh
##
tests_folder="$(cd "$(dirname "$0")" && pwd)"
include_folder="${tests_folder}/../inst/include/BayesMVP"
runtime="${HOME}/.cache/R/NicoStan/nicostan_page_release_bridgestan_826be39ccb9d277649e3a52798ce46f5"
stan_math="${runtime}/stan/lib/stan_math"
name="test_fast_finite_mixture_of_multivariate_ordered_probit_GHK_log_likelihood_against_Stan_math_autodiff"
mkdir -p "${tests_folder}/outputs"
for kernels in AVX512 AVX2; do
      extra_flag=""
      if [ "${kernels}" = "AVX2" ]; then extra_flag="-DBAYESMVP_FORCE_AVX2"; fi
      chrt -i 0 nice -n 19 ionice -c 3 taskset -c 180-191 \
            g++ -std=c++17 -O3 -march=native -mtune=native -mfma -mavx2 -mavx512f -mavx512vl -mavx512dq \
                ${extra_flag} -DBOOST_DISABLE_ASSERTS -D_REENTRANT -Wno-ignored-attributes \
                -I "${include_folder}" -I "${stan_math}" -I "${stan_math}/lib/eigen_3.4.0" \
                -I "${stan_math}/lib/boost_1.84.0" -I "${stan_math}/lib/sundials_6.1.1/include" \
                -I "${stan_math}/lib/tbb_2020.3/include" \
                "${tests_folder}/${name}.cpp" -o "${tests_folder}/outputs/${name}_${kernels}" \
                -L "${stan_math}/lib/tbb" -Wl,-rpath,"${stan_math}/lib/tbb" -ltbb || exit 1
      chrt -i 0 nice -n 19 ionice -c 3 taskset -c 180-191 "${tests_folder}/outputs/${name}_${kernels}" \
            | tee "${tests_folder}/outputs/${name}_${kernels}.txt"
done























