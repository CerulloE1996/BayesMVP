# Efficiency decomposition (2026-09-18)

Why ESS/grad fell from about 16–18 (your runs up to 09-17) to about 7–11 after the 09-18 changes, for your integrator (kick_flow_kick) at N = 10000.

## Running it

```bash
cd ~/Documents/Work/PhD_work/R_packages/BayesMVP/dev_experiments/efficiency_decomposition
./build_legacy_variants.sh                                   # once, ~6 min: old-C++ builds into ~/BayesMVP_legacy_variant_libs
./run_decomposition.sh                                       # all 8 arms x seeds 1-3, one run at a time (~40 min), then the table
ARMS_TO_RUN="H F" SEEDS_TO_RUN="1 2 3" ./run_decomposition.sh # a subset
Rscript summarise_decomposition.R <output_root_dir>          # re-print the table at any time
```

- Results are written to `Alg_paper_analysis/.../outputs/DGM_3/efficiency_decomposition_2026_09_18/<arm>_seed<k>/`, with logs in `logs/`.
- Your installed BayesMVP is never modified.
- The legacy builds live in their own library folders, and only the runs that name them use them.

**In RStudio:** open `run_one_arm.R`, edit the defaults block at the top, and **Source** the whole file. This only works for arms that use the installed build (E0, F, H). A session that has already loaded BayesMVP can't switch to a legacy build, so the script stops if you try.

## Arms

Every arm uses the 09-17 configuration: the 09-17 N = 10000 dataset, 25/25 chunks, test order 5 4 6 1 3 2, LR 0.05, 250 burn-in, 100 sampling iterations, `learning_rate_initial` 0.15 and tau_initial 2π. The arms differ only in these columns:

| arm | C++ | metric_estimator | tau_ramp | tests |
|---|---|---|---|---|
| D  | legacy_both  | chain_mean        | staged   | exact reproduction of the saved 09-17 runs |
| D0 | legacy_both  | chain_mean        | original | the same, with your original ramp |
| E0 | current      | chain_mean        | original | the current (correct) C++ with the same settings |
| F  | current      | pooled            | original | a Σ-scale metric on the current C++ |
| G  | legacy_both  | pooled            | original | whether `pooled` itself is worse than `chain_mean` was |
| H  | current      | chain_mean_scaled | original | `chain_mean` × n_chains_burnin on the current C++ |
| A0 | legacy_rng   | chain_mean        | original | the RNG fix alone (old RNG, correct bound) |
| B0 | legacy_bound | chain_mean        | original | the bound fix alone (current RNG, old bound) |

`legacy_rng` restores the burn-in seeding used before 09-18 00:17 BST, where the burn-in chains reused each other's seeds one iteration apart. `legacy_bound` restores the Pinkney bound used before 09-18 00:33, `sqrt(l)·D(j)`, which is too tight. Both are known defects: the legacy builds exist only to explain the old numbers.

**Caveat for D, D0, G and B0 (old bound):** their Omega / L_Omega summaries come from the current Stan skeleton, which uses the correct bound, so ignore those posterior summaries. ESS, ESS/grad, Rhat and the other efficiency numbers are unaffected: D reproduces the saved 09-17 values exactly.

## Each run is recorded

Each run writes `run_record.rds` in its folder, naming the one file it produced, and the summariser uses only that file. If ps7's resume check skips a run because a file with the same name already exists, the run stops with an error instead of reusing old results. `tau_ramp` and several other options are not encoded in the file name.

A non-default `learning_rate_initial=...` gets its own folder, e.g. `H_Li0.1_seed1`, so different configurations are never mixed.

## New package options (usable from ps7 as `settings$...`)

- `tau_ramp = "original"` (default: the ramp used before 09-17 12:17) or `"staged"`.
- `metric_estimator = "chain_mean_scaled"`: the `chain_mean` estimate × `n_chains_burnin`. Filename token `_Ms`.
- `test_perm_override = c(5,4,6,1,3,2)`: fixes the test order. The pre-burn-in still runs. Filename token `_tp546132`. It stops with an error if reordering is off, rather than being silently ignored.
- ps7 respects `settings$num_chunks_burnin` / `num_chunks_sampling` when you set them. Before 09-17 20:58, every N = 10000 run used 25/25, whatever its filename said.
