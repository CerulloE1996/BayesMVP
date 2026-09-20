# Optional joint flow–kick–flow integrator

The inner package (`inst/BayesMVP`) now accepts
`diffusion_HMC_integrator = "flow_kick_flow"` through the R6 `$sample()` method
and `R_fn_sample_model()`. The default is `"kick_flow_kick"`.

For an existing complete sampling argument list:

```r
sampling_args$diffusion_HMC <- TRUE
sampling_args$partitioned_HMC <- FALSE
sampling_args$diffusion_HMC_integrator <- "flow_kick_flow"
do.call(what = model$sample, args = sampling_args)
```

Flow–kick–flow requires a nonempty nuisance block. The warmup entry point
validates the selector and these flags. Direct native callers supply the
selector in `EHMC_args_as_Rcpp_List`; older lists lacking it default to
kick–flow–kick. Native callers must also select joint diffusion sampling.

## Implementation

The source was ported from `BayesMVPsim_mini (3).zip`, specifically
`src/MCMC/EHMC_dual_flow_kick_flow_fns.hpp` and the accompanying selector
changes. Archive `(4)` does not contain that new header.

The port is **flat**: four separate integrator functions (dense/diagonal main
metric × general/identity nuisance metric), matching the existing
kick–flow–kick dispatch style exactly — no templates, no shared-code helper.
The fast path uses the same condition as the existing KFK dispatch
(`theta_hat_us_vec.isZero(1e-5) && M_us_vec.isOnes(1e-5)`). Nonfinite density
or gradient returns early, and the existing wrapper's divergence handling
rejects the trajectory as usual.

For reference precision A = I and centre c, the nuisance residual kick is
`h * M_inv_us * (grad_log_pi + u - c)`. Its exact flow has frequency
`1 / sqrt(M_us)`, with the matching `sqrt(M_us)` and reciprocal factors.
The main block uses ordinary drift and its mass-preconditioned gradient kick.
Mass and centre remain fixed within a trajectory.

### What M_us means (worth one sentence in the methods paper)

`M_us` sets the **oscillator frequency**, not the reference covariance: the
exact flow is for the reference `N(theta_hat_us, I)`. The kick
`M_inv_us * (grad_log_pi + (u - theta_hat_us))` is correct as written —
with BridgeStan's `jacobian = TRUE`, `grad_log_pi` already contains the
`-u` prior term from the implicit standard normal, so adding
`(u - theta_hat_us)` leaves only the non-Gaussian residual
(`grad_log_lik - theta_hat_us`). This is a valid symmetric splitting for any
positive-definite `M_us`; `M_us` is not the same object as Beskos's `C`.

Selection occurs inside `fn_diffusion_HMC_dual_single_iter_InPlace_process`,
so initial step-size search, warmup and sampling use the selected ordering.
Velocity generation, trajectory-length randomization and the joint MH energy
remain in that existing wrapper. The original kick–flow–kick integrators
remain the default route.

The returned sampling object records the selector, as does the summary's
`HMC_info`, with a backward-compatible default for older results.

## Cost and comparison

With the current combined density/gradient interface, a completed L-step
flow–kick–flow trajectory uses L midpoint evaluations and one endpoint
evaluation. The next trajectory reuses that endpoint evaluation when accepted,
or the saved starting evaluation when rejected. Thus consecutive FKF
trajectories use L + 1 evaluations each, matching the existing KFK wrapper's
L + 1 evaluations.

A fresh chain/batch needs one extra initial evaluation before reuse is possible
(the first FKF trajectory costs L + 2). Cache validity is scoped to the
multi-iteration sampling call, or retained per chain in persistent built-in
model warmup. Resetting warmup positions invalidates the cache; changing only
mass, centre, step size or path length does not. The initial step-size search
reuses its existing initial evaluation. Exceptions invalidate the cache.

External Stan models are reloaded on each persistent warmup call in the
existing code, so that path conservatively reevaluates its initial state.
Fresh one-iteration legacy calls and direct calls without a validity flag also
reevaluate. No cache is shared between chains or separate model loads.

Existing L-based gradient-work summaries are estimates; they do not count
the setup or endpoint evaluations exactly. Use elapsed-time comparisons or
explicitly account for evaluation counts when comparing ESS per gradient.

Both orderings are symmetric second-order compositions. No efficiency
advantage is assumed. Use fresh warmup for adapted comparisons.

The archive cites Alenlöv, Doucet and Lindsten (2021), *Pseudo-Marginal
Hamiltonian Monte Carlo*, section 2.4, equations (16)–(19), for this ordering:
https://jmlr.org/papers/volume22/19-486/19-486.pdf

## Rebuild and verification

Rebuild/reinstall the inner native package using your usual RStudio workflow;
sourcing R files alone cannot update the C++ integrator. Regenerate help pages
with your usual `devtools::document()` step.

The archive describes standalone numerical and sanitizer tests performed by
its author. Those results are not verification of this adapted implementation.
No installation or MCMC run has been performed in this editing session.

Checks performed in this session:
- All edited R files parsed successfully with `Rscript`.
- The C++ header (flat port) passed a C++17 `g++ -fsyntax-only` check against
  Eigen 3.4.0, instantiating all four flat integrators with stub declarations
  for package types and the gradient callback. This is a syntax/type check,
  not a full package build or numerical validation.
- Source inspection confirmed that both warmup calls forward the selector,
  sampling takes the returned warmup argument list, persistent warmup copies
  the complete sampler-argument struct, and initial step-size search calls
  the joint wrapper containing the new dispatch.
- After the cache change, the actual joint-sampler header and actual HMCResult
  class passed a C++17 syntax/type check with stub model/gradient/RNG
  declarations, instantiating both the cached and backward-compatible wrapper
  signatures. Cache behavior has been source-reviewed but not runtime-tested.
- The spurious unconditional "nuisance file not open for chain!" print in the
  sampling worker constructor was removed, and `store_iteration()` now only
  checks/writes the nuisance stream when nuisance is actually sampled.

## APMS bridge integration (Claude review feedback)

- The bridge settings now default to the paper's algorithm:
  `partitioned_HMC = FALSE`, `diffusion_HMC = TRUE`, with
  `diffusion_HMC_integrator = "kick_flow_kick"` (selectable).
- `fn_validate_settings_APMS_BayesMVP()` refuses to run unless
  `partitioned_HMC`, `diffusion_HMC`, `M_decay_type`, `M_decay_power` and
  `use_disk` are supplied explicitly; `M_decay_scale = NULL` stays allowed.
  `"flow_kick_flow"` additionally requires `diffusion_HMC = TRUE` and
  `partitioned_HMC = FALSE`.
- `fn_fit_APMS_model_BayesMVP()` validates settings first and forwards the
  integrator selector through the R6 `$sample()` call.
- New test: `APMS_bridge/tests/test_07_flow_kick_flow.R`.
