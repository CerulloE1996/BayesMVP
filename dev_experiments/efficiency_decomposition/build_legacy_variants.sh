#!/usr/bin/env bash
##
## ---- build_legacy_variants.sh --------------------------------------------------------------------------------------------------
##
## Builds "legacy" variants of the BayesMVP C++ from the CURRENT source, each into its own library
## folder, so the old behaviour can be run side by side with the installed package.
## The installed package is NOT touched.
##
##   legacy_rng     burn-in RNG seeding as before 2026-09-18 00:17 BST: each burn-in chain was reseeded
##                  with (chain seed + iteration), so chain k at iteration i+1 reused chain k+1's seed
##                  from iteration i (overlapping random-number streams across the burn-in chains).
##   legacy_bound   Pinkney LDL positivity bound as before 2026-09-18 00:33 BST: sqrt(l_ij_old) * D(j),
##                  too tight by a factor sqrt(D(j)) (the correct bound is sqrt(l_ij_old * D(j))).
##   legacy_both    both of the above = the C++ behind every run up to and including 2026-09-17.
##
## Usage:   ./build_legacy_variants.sh [LEGACY_LIBRARIES_DIR] [--check]
##          LEGACY_LIBRARIES_DIR defaults to ~/BayesMVP_legacy_variant_libs
##          --check   only verifies that every patch site is where it should be (no build)
##
## Every patch site is counted before it is changed. If the source has moved on, this stops rather
## than building something that is not what its name says.
##
set -euo pipefail

BAYESMVP_SOURCE_DIR="${BAYESMVP_SOURCE_DIR:-$HOME/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP}"
LEGACY_LIBRARIES_DIR="$HOME/BayesMVP_legacy_variant_libs"
ONLY_CHECK_PATCH_SITES=0
for command_line_arg in "$@"; do
  case "$command_line_arg" in
    --check) ONLY_CHECK_PATCH_SITES=1 ;;
    *)       LEGACY_LIBRARIES_DIR="$command_line_arg" ;;
  esac
done

## ---- the two patches:
LEGACY_RNG_PATCH_FILE="src/MCMC/EHMC_single_threaded_samp_fns.hpp"
LEGACY_RNG_CURRENT_CODE='                     if (burnin_indicator) {'
LEGACY_RNG_REPLACEMENT_CODE='                     if (false) { // LEGACY BUILD: pre-2026-09-18 burn-in seeding (the else-branch below)'
LEGACY_BOUND_PATCH_FILE="src/general_functions/var_fns.hpp"
LEGACY_BOUND_CURRENT_CODE='stan::math::sqrt(l_ij_old * D(j-1))'
LEGACY_BOUND_REPLACEMENT_CODE='stan::math::sqrt(l_ij_old) * D(j-1)'

fn_count_occurrences() { grep -cF -- "$1" "$2" || true; }

n_legacy_rng_patch_sites=$(fn_count_occurrences "$LEGACY_RNG_CURRENT_CODE" "$BAYESMVP_SOURCE_DIR/$LEGACY_RNG_PATCH_FILE")
n_legacy_bound_patch_sites=$(fn_count_occurrences "$LEGACY_BOUND_CURRENT_CODE" "$BAYESMVP_SOURCE_DIR/$LEGACY_BOUND_PATCH_FILE")
echo "patch sites in current source: RNG = $n_legacy_rng_patch_sites (expect 1), bound = $n_legacy_bound_patch_sites (expect 4: double + var, lower + upper)"
[ "$n_legacy_rng_patch_sites" -eq 1 ]   || { echo "ERROR: RNG patch site not found exactly once in $LEGACY_RNG_PATCH_FILE"; exit 1; }
[ "$n_legacy_bound_patch_sites" -eq 4 ] || { echo "ERROR: bound patch sites not found exactly 4 times in $LEGACY_BOUND_PATCH_FILE"; exit 1; }
[ "$ONLY_CHECK_PATCH_SITES" -eq 1 ] && { echo "check only: OK"; exit 0; }

## this script deletes $LEGACY_LIBRARIES_DIR/<variant> before each build, so never let it be empty or /:
case "$LEGACY_LIBRARIES_DIR" in ""|"/") echo "ERROR: refusing LEGACY_LIBRARIES_DIR='$LEGACY_LIBRARIES_DIR'"; exit 1 ;; esac

INSTALLED_BAYESMVP_DIR="$(Rscript -e 'cat(find.package("BayesMVP"))')"   ## find.package does not load the DLL
TEMPORARY_BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMPORARY_BUILD_DIR"' EXIT
mkdir -p "$LEGACY_LIBRARIES_DIR"
echo "installed package (left untouched): $INSTALLED_BAYESMVP_DIR"
echo "variant libraries -> $LEGACY_LIBRARIES_DIR"

fn_apply_patch() {   ## args: file, current code, replacement code, expected number of sites
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import sys
file_path, current_code, replacement_code, expected_n_sites = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
file_text = open(file_path).read()
assert file_text.count(current_code) == expected_n_sites, (file_path, file_text.count(current_code))
open(file_path, "w").write(file_text.replace(current_code, replacement_code))
PY
}

fn_build_one_legacy_variant() {
  local variant_name="$1" use_legacy_rng="$2" use_legacy_bound="$3"
  local variant_source_dir="$TEMPORARY_BUILD_DIR/$variant_name/BayesMVP"
  ## start from an EMPTY target library: if R CMD INSTALL fails it restores the previous install,
  ## which would otherwise look like a successful build
  rm -rf "$LEGACY_LIBRARIES_DIR/$variant_name"
  mkdir -p "$TEMPORARY_BUILD_DIR/$variant_name" "$LEGACY_LIBRARIES_DIR/$variant_name"
  cp -r "$BAYESMVP_SOURCE_DIR" "$variant_source_dir"
  find "$variant_source_dir/src" \( -name '*.o' -o -name '*.so' \) -delete
  if [ "$use_legacy_rng" -eq 1 ]; then
    fn_apply_patch "$variant_source_dir/$LEGACY_RNG_PATCH_FILE" "$LEGACY_RNG_CURRENT_CODE" "$LEGACY_RNG_REPLACEMENT_CODE" 1
  fi
  if [ "$use_legacy_bound" -eq 1 ]; then
    fn_apply_patch "$variant_source_dir/$LEGACY_BOUND_PATCH_FILE" "$LEGACY_BOUND_CURRENT_CODE" "$LEGACY_BOUND_REPLACEMENT_CODE" 4
  fi
  echo "[$variant_name] building (legacy RNG = $use_legacy_rng, legacy bound = $use_legacy_bound) ..."
  ( cd "$TEMPORARY_BUILD_DIR/$variant_name" && MAKEFLAGS=-j12 R CMD INSTALL --no-docs --no-html --no-help --no-test-load --preclean \
        -l "$LEGACY_LIBRARIES_DIR/$variant_name" BayesMVP > "$LEGACY_LIBRARIES_DIR/build_$variant_name.log" 2>&1 ) || return 1
  ## re-use the already-compiled BridgeStan skeleton(s), so no variant pays a model compile at run time:
  if ls "$INSTALLED_BAYESMVP_DIR"/stan_models/*.so >/dev/null 2>&1; then
    ## plain cp (NOT -p): the copy must be newer than the installed .stan, or BridgeStan's make recompiles it
    cp "$INSTALLED_BAYESMVP_DIR"/stan_models/*.so "$LEGACY_LIBRARIES_DIR/$variant_name/BayesMVP/stan_models/"
  fi
  echo "[$variant_name] finished"
}

fn_build_one_legacy_variant legacy_rng   1 0 &   build_process_id_legacy_rng=$!
fn_build_one_legacy_variant legacy_bound 0 1 &   build_process_id_legacy_bound=$!
fn_build_one_legacy_variant legacy_both  1 1 &   build_process_id_legacy_both=$!

## a bare 'wait' always returns 0, so wait on each build separately and check its own exit status:
any_build_failed=0
for variant_name in legacy_rng legacy_bound legacy_both; do
  build_process_id_variable="build_process_id_$variant_name"
  if wait "${!build_process_id_variable}" && ls "$LEGACY_LIBRARIES_DIR/$variant_name/BayesMVP/libs/"*.so >/dev/null 2>&1; then
    echo "OK      $variant_name"
  else
    echo "FAILED  $variant_name (see $LEGACY_LIBRARIES_DIR/build_$variant_name.log)"; any_build_failed=1
  fi
done
[ "$any_build_failed" -eq 0 ] || exit 1
echo "all variants built in $LEGACY_LIBRARIES_DIR (logs: $LEGACY_LIBRARIES_DIR/build_*.log)"
