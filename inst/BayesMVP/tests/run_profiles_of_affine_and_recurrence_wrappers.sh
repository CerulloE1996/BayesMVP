#!/usr/bin/env bash

set -euo pipefail

{
    workspace=/home/enzocerullo/Documents/Work/PhD_work
    profile_binaries=$workspace/validation_staging/avx-sol-repair-20261015-auUpzo/affine_block_reductions
    profile_folder=$profile_binaries/wrapper_profile_full_load_attempt1
    mechanism_folder=$workspace/Alg_paper_analysis/paper_1_chunking_and_parallel_scalability/mechanism_study
    pmc_tool=$mechanism_folder/pmc_stat
}

if ps -C R,Rscript,pmc_stat,cc1plus,g++,clang++ -o pid= | rg -q '[0-9]'; then
    echo 'Another job is active; wrapper profile not started.'
    exit 1
fi
if test -e "$profile_folder"; then
    echo 'Profile directory already exists; preserving it.'
    exit 1
fi
mkdir "$profile_folder"

for function_name in affine_weighted_prefix normal_recurrence_composite \
                     normal_recurrence_joint affine_log_sum_exp; do
    for data_type in double autodiff; do
        for data_N in 10000 25000; do
            for num_chunks in 1 16; do
                for repeat_index in 1 2 3; do
                    for math_backend in Stan AVX2 AVX512; do
                        simd_backend=AVX2
                        profile_math=Stan
                        if test "$math_backend" != Stan; then
                            simd_backend=$math_backend
                            profile_math=AVX
                        fi
                        case_name=${function_name}_${data_type}_N${data_N}_chunks${num_chunks}
                        case_name=${case_name}_${math_backend}_repeat${repeat_index}
                        profile_program=$profile_binaries/profile_wrappers_${simd_backend}_attempt2
                        chrt -i 0 nice -n 19 ionice -c 3 taskset -c 0-179 "$pmc_tool" \
                            "$profile_folder/${case_name}_counters.csv" "$case_name" -- \
                            "$profile_program" "$function_name" "$profile_math" \
                            "$data_type" "$data_N" "$num_chunks" 180 \
                            > "$profile_folder/${case_name}.log" 2>&1
                    done
                done
            done
        done
    done
    printf '%s function %s finished\n' "$(date -Is)" "$function_name"
done






















