#!/usr/bin/env bash
##
## ---- run_decomposition.sh ------------------------------------------------------------------------------------------------------
##
## Runs the efficiency-decomposition arms ONE AT A TIME (runs share the machine, so never in parallel),
## each in a fresh R process, then prints the summary table.
##
## Usage:   ./run_decomposition.sh                                    ## all arms, seeds 1 2 3
##          ARMS_TO_RUN="H G" SEEDS_TO_RUN="1 2 3" ./run_decomposition.sh
##          OUTPUT_ROOT_DIR=/some/dir LEGACY_LIBRARIES_DIR=~/BayesMVP_legacy_variant_libs ./run_decomposition.sh
##
## Every arm: N = 10000 (the 09-17 dataset), your integrator (kick_flow_kick), 25/25 chunks, test order
## 5 4 6 1 3 2, tau_initial = 2pi, learning_rate_initial = 0.15 - i.e. the 09-17 configuration - and they
## differ ONLY in:
##
##   arm   C++ build        metric_estimator     tau_ramp   what it tests
##   D     legacy_both      chain_mean           staged     EXACT reproduction of your saved 09-17 runs (they used the staged ramp)
##   D0    legacy_both      chain_mean           original   the same, with your original ramp
##   E0    installed        chain_mean           original   current (correct) C++, same settings
##   F     installed        pooled               original   current C++ + Sigma-scale metric
##   G     legacy_both      pooled               original   is 'pooled' itself worse than chain_mean was?
##   H     installed        chain_mean_scaled    original   current C++ + chain_mean x n_chains_burnin
##   A0    legacy_rng       chain_mean           original   current bound, old RNG  -> effect of the RNG fix alone
##   B0    legacy_bound     chain_mean           original   current RNG, old bound  -> effect of the bound fix alone
##
## The legacy_* builds come from ./build_legacy_variants.sh (run that first, ~6 min).
##
set -uo pipefail
THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEGACY_LIBRARIES_DIR="${LEGACY_LIBRARIES_DIR:-$HOME/BayesMVP_legacy_variant_libs}"
OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-$HOME/Documents/Work/PhD_work/Alg_paper_analysis/1_appendix_pilot_studies/ps_7_basic_MCMC_settings_BayesMVP/outputs/DGM_3/efficiency_decomposition_2026_09_18}"
ARMS_TO_RUN="${ARMS_TO_RUN:-D D0 E0 F G H A0 B0}"
SEEDS_TO_RUN="${SEEDS_TO_RUN:-1 2 3}"
SETTINGS_SHARED_BY_ALL_ARMS="num_chunks_burnin=25 num_chunks_sampling=25 test_perm_override=546132 diffusion_HMC_integrator=kick_flow_kick tau_initial=2pi"

fn_settings_for_arm() {
  case "$1" in
    D)  echo "BayesMVP_library_path=$LEGACY_LIBRARIES_DIR/legacy_both   metric_estimator=chain_mean          tau_ramp=staged" ;;
    D0) echo "BayesMVP_library_path=$LEGACY_LIBRARIES_DIR/legacy_both   metric_estimator=chain_mean          tau_ramp=original" ;;
    E0) echo "BayesMVP_library_path=installed                           metric_estimator=chain_mean          tau_ramp=original" ;;
    F)  echo "BayesMVP_library_path=installed                           metric_estimator=pooled              tau_ramp=original" ;;
    G)  echo "BayesMVP_library_path=$LEGACY_LIBRARIES_DIR/legacy_both   metric_estimator=pooled              tau_ramp=original" ;;
    H)  echo "BayesMVP_library_path=installed                           metric_estimator=chain_mean_scaled   tau_ramp=original" ;;
    A0) echo "BayesMVP_library_path=$LEGACY_LIBRARIES_DIR/legacy_rng    metric_estimator=chain_mean          tau_ramp=original" ;;
    B0) echo "BayesMVP_library_path=$LEGACY_LIBRARIES_DIR/legacy_bound  metric_estimator=chain_mean          tau_ramp=original" ;;
    *)  echo "UNKNOWN_ARM" ;;
  esac
}

## ---- check every requested arm is known and its C++ build exists, BEFORE starting any run:
for arm_name in $ARMS_TO_RUN; do
  settings_for_arm="$(fn_settings_for_arm "$arm_name")"
  [ "$settings_for_arm" = "UNKNOWN_ARM" ] && { echo "unknown arm: $arm_name"; exit 1; }
  BayesMVP_library_path="$(echo "$settings_for_arm" | sed -n 's/.*BayesMVP_library_path=\([^ ]*\).*/\1/p')"
  if [ "$BayesMVP_library_path" != "installed" ] && ! ls "$BayesMVP_library_path"/BayesMVP/libs/*.so >/dev/null 2>&1; then
    echo "arm $arm_name needs the build in $BayesMVP_library_path - run ./build_legacy_variants.sh first"; exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT_DIR/logs"
## seeds in the OUTER loop, so every arm has at least one result early on:
for seed_index in $SEEDS_TO_RUN; do
  for arm_name in $ARMS_TO_RUN; do
    log_file="$OUTPUT_ROOT_DIR/logs/${arm_name}_seed${seed_index}.txt"
    echo "== $(date +%H:%M:%S) start $arm_name seed ${seed_index}000"
    ## R_main_v5.R builds its paths from $PWD, so always start R from $HOME:
    ( cd "$HOME" && Rscript "$THIS_SCRIPT_DIR/run_one_arm.R" arm_name="$arm_name" seed_index="$seed_index" \
          output_root_dir="$OUTPUT_ROOT_DIR" $(fn_settings_for_arm "$arm_name") $SETTINGS_SHARED_BY_ALL_ARMS ) > "$log_file" 2>&1
    exit_status=$?
    echo "   $(date +%H:%M:%S) $arm_name seed ${seed_index}000 exit=$exit_status | $(grep -hE 'loaded from|% divergences =  |ESS/sec =|min_ESS =' "$log_file" | grep -v '^%' | sed 's#.*loaded from: ##' | tr '\n' ' ')"
  done
done

Rscript "$THIS_SCRIPT_DIR/summarise_decomposition.R" "$OUTPUT_ROOT_DIR"
