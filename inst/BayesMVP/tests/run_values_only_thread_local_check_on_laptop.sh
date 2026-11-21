#!/usr/bin/env bash

set -euo pipefail

{
    workspace=/home/enzo/Documents/Work/PhD_work
    snapshot=$workspace/validation_staging/values_only_thread_local_snapshot
    stan_math=/home/enzo/.cache/R/NicoStan/nicostan_page_release_bridgestan_b95ebabcf0e8ae8f27e363466244a5f1
    stan_math=$stan_math/stan/lib/stan_math
    pmc=$workspace/Alg_paper_analysis/paper_1_chunking_and_parallel_scalability/mechanism_study/pmc_stat
}

until ! kill -0 125627 2>/dev/null; do sleep 5; done
if ps -C R,Rscript,pmc_stat,cc1plus,g++ -o pid= | grep -q '[0-9]'; then
    echo 'Another laptop job is active; the values-only check was not started.'
    exit 1
fi

flags=( -std=c++17 -O3 -march=native -mtune=native -mavx2 -DBAYESMVP_FORCE_AVX2
        -DSTAN_THREADS -D_REENTRANT -pthread
        -I "$stan_math" -I "$stan_math/lib/eigen_3.4.0" -I "$stan_math/lib/boost_1.84.0"
        -I "$stan_math/lib/sundials_6.1.1/include" -I "$stan_math/lib/tbb_2020.3/include" )
links=( -L "$stan_math/lib/tbb" -Wl,-rpath,"$stan_math/lib/tbb" -ltbb )

build_jobs=
for version in before after; do
    chrt -i 0 nice -n 19 ionice -c 3 taskset -c 0-15 g++ "${flags[@]}" \
        "$snapshot/$version/tests/profile_affine_and_recurrence_wrappers.cpp" \
        "${links[@]}" -o "$snapshot/$version/profile" > "$snapshot/$version/build.log" 2>&1 &
    build_jobs="$build_jobs $!"
done
chrt -i 0 nice -n 19 ionice -c 3 taskset -c 0-15 g++ "${flags[@]}" \
    "$snapshot/after/tests/check_fused_affine_compositions.cpp" "${links[@]}" \
    -o "$snapshot/after/check" > "$snapshot/after/build_check.log" 2>&1 &
build_jobs="$build_jobs $!"
failed=0
for build_job in $build_jobs; do wait "$build_job" || failed=1; done
if test "$failed" != 0; then exit 1; fi
chrt -i 0 nice -n 19 taskset -c 0-15 "$snapshot/after/check" > "$snapshot/after/check.log" 2>&1

output=$snapshot/results
if test -e "$output"; then echo 'Results already exist; preserving them.'; exit 1; fi
mkdir "$output"
for n in 10000 25000; do
    for chunks in 1 16; do
        for repetition in 1 2 3; do
            for method in Stan before after; do
                version=$method
                math=AVX
                if test "$method" = Stan; then version=after; math=Stan; fi
                case_name=double_N${n}_chunks${chunks}_${method}_repeat${repetition}
                chrt -i 0 nice -n 19 ionice -c 3 taskset -c 0-15 "$pmc" \
                    "$output/${case_name}_counters.csv" "$case_name" -- \
                    "$snapshot/$version/profile" affine_weighted_prefix "$math" double "$n" "$chunks" 16 \
                    > "$output/${case_name}.log" 2>&1
            done
        done
    done
done
echo 'VALUES ONLY THREAD LOCAL CHECK FINISHED'






















