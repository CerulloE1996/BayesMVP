# RUN_BayesMVP — rebuilding, installing and running the updated BayesMVP + APMS 2007 bridge

This document covers:

1. What changed in this deliverable (short version — see the change log for the full list).
2. Rebuilding / installing the updated package.
3. The external-Stan (BridgeStan) interface: nuisance-block convention, supported and
   unsupported configurations.
4. Running the APMS 2007 four-class autism/BPD model through BayesMVP.
5. Reproducible test commands and known limitations.

---

## 1. What changed

* The `MVP_model` R6 class now matches the underlying functions: constructor inputs are
  preserved during deferred initialisation, `$sample()` forwards every required sampler /
  storage / threading / metric argument, object state (`init_object`, `n_nuisance`,
  `n_params_main`, `is_compiled`) is refreshed after sampling, and `$summary()` uses the
  actual storage mode used by sampling.
* The external-Stan path now counts **unconstrained** dimensions (via BridgeStan), identifies
  the **first declared** parameter using `stanc --info` compiler metadata (a zero-length first
  declaration is handled), preserves the **complete** external-model initialisation
  (unconstrain first, split nuisance/main second), and supports **partial** inits by
  completing missing entries with model defaults.
* The C++ `n_nuisance <= 10` shortcut that silently dropped small nuisance blocks from the
  Stan log-density/gradient has been removed. The complete `[nuisance | main]` vector is
  always evaluated.
* Nuisance draws are now **stored** (RAM or disk, matching the log-lik storage mode) and used
  to **reconstruct complete unconstrained draws** before calling BridgeStan's constrain.
  Constrained main / transformed / generated-quantity indexing and naming now use constrained
  metadata (correct for simplexes and other dimension-changing constraints).
* Models **without** a nuisance block are supported: `sample_nuisance = FALSE` means every
  declared parameter is main (`n_nuisance = 0`), no dummy coordinates are introduced, and all
  coordinates are sampled.

---

## 2. Rebuild / install

Environment used for verification: R 4.3.3, g++ 11.4, CmdStan 2.37.0
(`/home/enzocerullo/.cmdstan/cmdstan-2.37.0`), BridgeStan R 2.6.2
(`/home/enzocerullo/.bridgestan/bridgestan-2.6.2`), 192 cores, 377 GB RAM.

```bash
cd /home/enzocerullo/Documents/Work/PhD_work/R_packages

## build the source tarball
R CMD build BayesMVP

## install (compiles the C++ from src/ - this is the long step; expect 10-40 min)
R CMD INSTALL --preclean --no-multiarch BayesMVP_*.tar.gz
```

In an R session:

```r
library(BayesMVP)
## make sure the BridgeStan environment variable points at an installed bridgestan:
BayesMVP:::bridgestan_path()   # should print ~/.bridgestan/bridgestan-2.6.2
```

Notes:

* The package **must** be called `BayesMVP` (both the outer wrapper and the inner package at
  `inst/BayesMVP`). The previous `BayesMVP` folder was archived as
  `R_packages/BayesMVPold/` with package name `BayesMVPold`.
* `stanc_args` are Stan **compiler** arguments (e.g. `list("--O1")`). `make_args` are
  **C++/make** arguments for compiling the generated C++ (e.g. `list("STAN_THREADS=true")` for
  `reduce_sum` models). They are distinct and are forwarded separately through
  `MVP_model$new(..., stanc_args = ..., make_args = ...)` and `initialise_model()`.

---

## 3. External-Stan interface: the nuisance-block convention

BayesMVP's convention for user-supplied Stan models:

* The **nuisance** (high-dimensional latent) block is the **FIRST declaration** of the
  `parameters` block. Its **unconstrained** dimension is detected automatically — you do not
  count coordinates and you do not have to rename it (any name works; `u_raw` is just the name
  used by the bundled examples).
* `sample_nuisance = TRUE`: the first declaration is the nuisance block. The bridge uses the
  **joint dual sampler** (`partitioned_HMC = FALSE`, `diffusion_HMC = TRUE`), which samples the
  nuisance block with the diffusion-pathspace step and the main block with standard HMC in ONE
  leapfrog — this is the paper's algorithm. `partitioned_HMC = TRUE` (separate nuisance/main
  updates) remains available but is not the bridge default.
* `diffusion_HMC_integrator`: the joint-diffusion ordering, `"kick_flow_kick"` (default) or
  `"flow_kick_flow"` (Alenlöv–Doucet–Lindsten ordering). Unsupported combinations are rejected.
* `sample_nuisance = FALSE`: **no nuisance block exists** — every declared parameter belongs
  to the main block (`n_nuisance = 0`), and the complete unconstrained vector is sampled
  jointly. No dummy declaration or source modification is required.
  This is **distinct** from "a nuisance declaration exists but has zero length": that case is
  `sample_nuisance = TRUE` with a first declaration that evaluates to size 0 (e.g. an empty
  array under a switched-off branch); detection then yields `n_nuisance = 0`, nuisance-only
  updates are skipped automatically, and all main coordinates are sampled.
* `n_nuisance_override = <integer>` overrides the detection (never required).

Compatibility note (dimensional vs algorithmic): detecting the first declaration and its
unconstrained size is purely dimensional. Whether a given sampler configuration is
**statistically** appropriate is a separate question: diffusion-pathspace HMC is designed for
high-dimensional latent blocks that are (approximately) Gaussian on the unconstrained scale
under the target density. BayesMVP does not silently alter the target density, the Jacobian
handling (BridgeStan computes the full log-density + gradient with `jacobian = TRUE`), or the
reference measure. Accepting an arbitrary first declaration proves nothing about that
mathematical validity; inspect the model and the diffusion assumptions before trusting a run.

---

## 4. Running the APMS 2007 model

Files (delivered with this package under `APMS_bridge/`):

* `R_bridge_APMS2007_BayesMVP.R` — the full bridge: reads `apms07arch.dta`, builds the four
  ordinal tests and the phase-one grouping (unchanged from the original bridge), builds the
  Stan data, runs the prior check (its own subset configuration) and fits
  M3/M4/M5/M6 through the updated BayesMVP interface, then produces the application table and
  the DGM parameter sets.
* `R_fn_APMS_BayesMVP.R` — the BayesMVP-specific helpers (named-list sampler settings, inits,
  convergence gate, prior check, application table, DGM extraction).

The Stan model `LC_MVOP_4class_joint_v1_reduce_sum.stan` is used **as supplied** — the bridge
does not modify it and does not provide a manual likelihood.

Run it:

```r
## install BayesMVP first (section 2), then:
source("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/APMS_bridge/R_bridge_APMS2007_BayesMVP.R")
```

Key settings and their mapping to the established CmdStanR settings:

| CmdStanR (established)            | BayesMVP equivalent                          | Notes |
|-----------------------------------|----------------------------------------------|-------|
| `iter_warmup = 250`               | `n_burnin = 250`                             | same warm-up length |
| `iter_sampling = 100`             | `n_iter = 100`                               | same post-warm-up draws |
| `adapt_delta = 0.65`              | `adapt_delta = 0.65`                         | same Metropolis target acceptance |
| `max_treedepth = 8`               | `max_L = 256` (= 2^8)                        | bound on max leapfrog steps; **not** an exact NUTS equivalent |
| `chains = 16`                     | `n_chains_burnin = n_chains_sampling = 16`   | |
| `threads_per_chain = 1`           | `n_threads_WCP_burnin = n_threads_WCP_sampling = 1` | no within-chain threading |
| `seed = 123`                      | `seed = 123`                                 | |
| `RUN_PARALLEL` / `N_CORES = 64`   | same `fn_chains_for_parallel` rule           | 15 chains/model for 4 concurrent models |

These are explicit mappings in `fn_sampler_settings_APMS_BayesMVP()`; they are **not** claimed
to be exact equivalents of cmdstanr's NUTS settings. The prior check settings
(`PRIOR_CHECK_CHAINS = 4`, `PRIOR_CHECK_WARMUP = 200`, `PRIOR_CHECK_SAMPLING = 200`,
`PRIOR_CHECK_ROWS = 1000`) are applied **only** to the prior-only fit, and the posterior
subset (`subset = TRUE`, `N_subset = 1000`, stratified PPS) is controlled independently.

`fn_fit_APMS_model_BayesMVP()` validates the settings first and refuses to run unless
`partitioned_HMC`, `diffusion_HMC`, `M_decay_type`, `M_decay_power` and `use_disk` are supplied
explicitly (no silent hard-coded defaults; `M_decay_scale = NULL` is allowed and means
`n_adapt / 5`). `"flow_kick_flow"` additionally requires `diffusion_HMC = TRUE` and
`partitioned_HMC = FALSE`.

Nothing is replaced with `detectCores()`, parallelism is not disabled, iteration counts are
not capped, and the joint diffusion HMC + metric + decay choices are preserved as listed
in `fn_sampler_settings_APMS_BayesMVP()`.

Output: `apms07_fit_and_DGM_params_using_BayesMVP.rds` in the APMS paper directory, plus
per-run artifacts under `BayesMVP_out/`.

---

## 5. Tests

Test scripts are delivered in `APMS_bridge/tests/`:

```bash
cd /home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/APMS_bridge/tests

Rscript test_01_detect_nuisance.R        # detection: standard, simplex-first, zero-length first decl, no-nuisance
Rscript test_02_r6_mocks.R               # R6 deferred init / forwarding / state refresh (mocked init fns)
Rscript test_03_bridgestan_init.R        # REAL BridgeStan compile + init: complete & partial inits
Rscript test_04_small_smoke.R            # real package sampling, tiny runs: nuisance-present, no-nuisance, simplex
Rscript test_05_reconstruction.R         # constrain reconstruction vs BridgeStan on known unconstrained draws
Rscript test_06_lp_grad.R                # log-density / gradient agreement against BridgeStan
Rscript test_07_flow_kick_flow.R         # joint FKF vs KFK: selector validation + posterior-equivalence smoke
```

Notes:

* Tests 1–2 run without a compiled package (pure R / mocks).
* Tests 3–6 need the installed package + BridgeStan; tests 4–6 are smoke tests that verify
  **execution**, not convergence or statistical validity (reduced settings live in
  `tests/fixtures/settings_smoke.R` — the application settings are untouched).
* Test 7 needs the **rebuilt** native package (the `diffusion_HMC_integrator` selector lives in
  the C++ sampler). It skips cleanly (exit 0) if the installed package predates the selector.

---

## 6. Known limitations / untested paths

* `fn_compute_param_constrain_from_trace_v2` (the legacy, sequential constrain entry point) is
  unchanged and does not accept a nuisance trace; the active path is
  `fn_compute_param_constrain_from_trace_parallel`.
* OpenMP sampling supports RAM traces only (the OpenMP entry point has no `use_disk` option,
  as before); disk-mode nuisance storage is exercised on the RcppParallel path.
* The "nuisance block exists but its updates should be skipped while still partitioning"
  combination for **external Stan** models is not a first-class mode: `sample_nuisance = FALSE`
  means "no nuisance block" (all main). Built-in models keep their historical dummy-coordinate
  behaviour unchanged.
* The bridge's parallel path was source-inspected and mock-tested; an end-to-end PSOCK run was
  not executed (see the final report).
* Zero-length `data`-dependent declarations are handled through `stanc --info` metadata; if
  stanc is unavailable, detection falls back to the first unconstrained name with a warning
  (zero-length first declarations cannot then be distinguished).
