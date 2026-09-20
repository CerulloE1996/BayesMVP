# APMS application and simulation-study fitting benchmarks

Open `Autism_BPD_paper/R_benchmark_APMS_full_fits.R` in RStudio. It compares full fits using the same prepared APMS data and priors. It uses the fitting engines behind the two bridges, rather than independently sourcing both long scripts and risking different targets.

1. Edit `MODELS_TO_BENCHMARK`, `REPETITION_SEEDS`, `THREAD_BUDGET`, and the configuration lists. Existing CmdStanR settings are copied from `R_bridge_apms07_to_DGM.R`; BayesMVP settings come from the current BayesMVP bridge. They intentionally retain their separate chain counts and iteration counts.
2. Set BayesMVP arguments directly in `fn_sampler_settings_APMS_BayesMVP(...)`, which creates `bayesmvp_sampler_settings` in the runner. For example, pass `n_burnin = 125`, `tau_ramp = "original"` and `burnin_TBB_pool_equals_n_chains = FALSE` alongside each other. APMS uses external Stan models: the package forces this setting to `FALSE`, even if `TRUE` is supplied, so nested TBB likelihood work can use the full chain-count times WCP thread budget. For built-in models, the automatic default is `TRUE`, and an explicit `FALSE` remains available. There is no nested `sampler_options` list. The regular bridge uses the same function. `summary_options` remains separate because it controls post-processing, not sampling. Explicit `NULL` values are retained; unknown arguments fail before fitting. Omitted `M_decay_scale` is calculated from the requested adaptation length, and omitted `n_superchains` follows `n_chains_sampling`.
3. Set `RUN_BENCHMARK <- TRUE` and source the driver. Each configuration/model/seed runs in a fresh R worker. Threads within a fit retain their requested layout. Different configurations run sequentially, so they do not compete with each other for CPU resources.
4. Set `RUN_BENCHMARK <- FALSE` to load completed and failed results without fitting anything. The final block also works for incomplete sweeps. Re-running with `resume = TRUE` skips completed cells with identical fingerprints and retries failed cells. Interrupting a fit leaves earlier checkpoints intact.

The newly exposed R6 controls require installing the updated **inner** BayesMVP package through your usual workflow before enabling them. No C++ algorithm changes were made for this interface update. Defaults are preserved. Available controls include shared tau, the TBB burn-in pool, nuisance-centre estimation/freezing, tau objective/ramp/weighting, fixed trajectories in leapfrog units, pre-burn-in controls, burn-in profiling, log-likelihood trace storage, and the existing metric/integrator settings.

`J_grad_option` belongs to the bespoke built-in likelihood, not this external `.stan` model. It is not silently accepted as a Stan-model option. Stan differentiates the model's Jacobian code through its own autodiff path.

`n_burnin` is the total main burn-in length; there is no separate post-adaptation early-stop option. For `n_burnin = 125`, the default ramp is `clip_iter = 20`, with the tau handover at `clip_iter_tau = 50`. The 250-iteration schedule retains its handover at 125. The existing additional tau resets are unchanged: 50, 51, 52, 60 and 61 for the 125-iteration schedule.

## Configuration choices

The starting list contains CmdStanR and BayesMVP, each with plain Stan functions or the existing AVX externals. Copy a named configuration to test other sampler settings, chunks, or compiler options. Both engines support a user header. The driver preserves the current `Phi_type`; AVX comparisons require a vectorised likelihood path (`0` or `2`).

For CmdStanR, `num_chunks = NULL` retains the prepared data's grainsize. Set an explicit positive number to control `ceil(N / num_chunks)`. For BayesMVP, set `num_chunks_burnin` and `num_chunks_sampling` independently in its sampler settings; the sampler recalculates grainsize per phase. One reduce_sum slice still uses the reduce_sum program. To compare the truly unsliced program, select `LC_MVOP_4class_joint_v1.stan` explicitly. Chunk controls have no effect in a Stan program which does not use grainsize.

Compiler options are explicit per configuration. Existing toolchain `make/local` settings still apply. In particular, enabling AVX2 flags alone does not disable AVX-512 flags already in `make/local`; inspect the resulting build flags when comparing instruction sets. Source, AVX kernel and toolchain fingerprints isolate compiled artifacts in the benchmark's `builds/` directory. Original Stan files and compiled models are not overwritten.

## Reading the timings and diagnostics

- `full_fit_seconds`: initial values, compilation/model initialisation, warmup, sampling, generated quantities, engine summaries, draw extraction, and the shared comparison diagnostics.
- `setup_seconds`: compilation/model initialisation, reported together because BayesMVP performs them in the same constructor. `binary_existed_before` distinguishes fresh from previously built model paths.
- `sample_call_seconds`, `engine_summary_seconds`, `comparison_seconds`: component wall times. `fit_excluding_setup_seconds` removes only setup; it still includes warmup and postprocessing.
- `job_wall_seconds`: parent-observed worker launch, package/helper loading, the fit, result transfer and worker shutdown. Final RDS checkpoint serialization is outside this timer. Common data preparation is recorded once in the saved prepared-input bundle, rather than charged differently to each engine.
- `native_burnin_seconds` / `native_sampling_seconds`: engine-reported diagnostics. CmdStanR uses the maximum per-chain values; these are not phase wall times when chains are queued. Use the measured full-fit/job wall times to compare end-to-end cost.
- `min_ESS_bulk`, `min_ESS_tail`, `max_rhat`: recomputed with **posterior** for both engines, on the same requested application quantities. Constant structural entries are excluded; non-finite diagnostics fail the gate. Each saved result includes the per-variable diagnostic table and monitored names for review.
- `diagnostics_pass`: the supplied minimum bulk ESS / maximum R-hat thresholds, complete diagnostics, and zero sampling divergences. It is a computational screen, not proof of convergence or posterior equivalence.
- `ESS_per_sec_full_fit`: minimum bulk ESS divided by measured full-fit wall time. Compare within the same model and `input_fingerprint`; review posterior summaries as well as efficiency. Different priors/data, build options or settings create distinct files.

RDS files retain configuration, input data, summaries, timings, session information, and optionally posterior draws. CSV files are retained for CmdStanR. No live R6 object or BridgeStan pointer is serialized. A compact `benchmark_summary.csv` is refreshed after each cell. The runner preserves full standardisation and generated-quantities settings; it does not borrow the reduced outputs used by the gradient microbenchmark.

The runner does not execute the later application-table/DGM-export sections of either bridge. It benchmarks fitting and diagnostics; the saved summaries/draws are available for downstream application or simulation work. Independent prior scenarios can be supplied as prepared data in separate runs; their different fingerprints prevent accidental reuse.
