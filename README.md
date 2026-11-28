# BayesMVP

BayesMVP is the specialised model extension for [NicoStan](https://github.com/CerulloE1996/NicoStan). NicoStan supplies the general Stan-compatible sampler, warm-up, trajectory-length adaptation and diagnostics. BayesMVP adds native manually implemented gradients and model-specific C++ code for multivariate probit models.

BayesMVP was originally a standalone R package (and NicoStan was formerly called "BayesMVP"); however, it has now been split into 2 R packages: [NicoStan](https://github.com/CerulloE1996/NicoStan), for the general Stan interface (which relies heavily on [BridgeStan](https://roualdes.us/bridgestan/latest/)), and BayesMVP, which is now an extension to NicoStan, specifically for the extremely efficient fitting of multivariate probit (MVP) based models (including the latent class MVP [LC-MVP] and the latent trait model, for both binary and/or ordinal outcomes/tests).

The extension contains:

- **MVP:** the multivariate probit model for correlated binary outcomes.
- **2LC-MVP:** the two-class latent-class multivariate probit model, selected with `Model_type = "LC_MVP"`.
- **MVOP:** the multivariate ordinal probit model.
- **2LC-MVOP:** the two-class latent-class multivariate ordinal probit model.
- **Two-class latent trait:** the two-class latent-trait model, selected with `Model_type = "latent_trait"`.
- **CLC-MVOP:** the co-morbidity latent class multivariate ordinal probit model, for two conditions that can co-occur, selected with `Model_type = "CLC_MVOP"` (see [below](#the-clc-mvop-co-morbidity-model)).

These models are used for correlated binary and ordinal outcomes, including diagnostic and screening test-accuracy models without a perfect reference standard. The ordinal latent-class model is described in [Cerullo et al. (2022)](https://doi.org/10.1002/jrsm.1567), and the latent-class MVP versus latent-trait simulation study is reported in [Cerullo et al. (2025)](https://arxiv.org/abs/2509.18489v1).

## Efficiency compared with Stan and Mplus

The table below shows the estimated time needed to reach a target minimum effective sample size (ESS) - i.e., the ESS target must be met by every sensitivity, specificity and prevalence parameter - for the LC-MVP model, fitted to simulated binary data (6 tests) based on real, publicly available COVID-19 data, on a 96-core AMD EPYC 9654 server (see our upcoming paper for full details).

| $N$ | Target min. ESS | NicoStan + BayesMVP | Stan (NUTS, via cmdstanr) | Mplus (PX-Gibbs) | Speed-up vs. Stan | Speed-up vs. Mplus |
|---|---|---|---|---|---|---|
| 500 | 7,000 | 7.66 s | 45.6 s | - | ~6× | - |
| 2,500 | 2,500 | 9.89 s | 3,025 s (50.4 min) | - | ~300-400× | - |
| 10,000 | 1,000 | 12.20 s | 21,440 s (357.3 min) | 550 s (9.2 min) | ~1,500-2,000× | ~40-50× |

Note that these times include burnin; the NicoStan + BayesMVP times also include computing the posterior summaries.

Furthermore, for our ordinal example dataset (LC-MVOP), which is based on a real dataset on three tests to screen and/or diagnose depression (specifically, the MINI as the imperfect gold standard, and the PHQ-9 and CES-D-10 as the two ordinal tests), with N = 5,000, NicoStan/BayesMVP was **over 500× more efficient than Stan** and **over 1000× more efficient than Mplus**. The latter is because Mplus's Gibbs-based algorithm struggles greatly with ordinal outcomes, and Mplus does not let you fit ordinal outcomes with more than 10 categories; hence, we had to group categories together just to attempt to measure its efficiency.

## How BayesMVP computes the log-posterior and its gradient

BayesMVP uses manually derived gradients, hand-coded in C++, for all of its models (i.e., rather than relying on automatic differentiation); more specifically, each log-posterior/gradient evaluation uses up to three levels:

1. **Hand-coded gradient (standard scale):** the fastest option, and the one used for almost every evaluation.
2. **Hand-coded gradient on the log scale:** if the first evaluation returns a non-finite value (e.g., due to numerical underflow/overflow in the tails of the normal CDF), the evaluation is repeated at exactly the same position using log-scale versions of the same manually derived gradients, which are more numerically stable. This is on by default (`multi_attempts = TRUE`). Note that the log-scale gradients are not yet available for the latent trait model (which still uses the hand-coded standard-scale gradient first).
3. **Automatic differentiation:** as a final backup, the gradient is computed using reverse-mode autodiff from the Stan math C++ library. This is on by default for the latent trait model (where it is the only backup, until its log-scale gradients are finished), and can be switched on for the other models via `options(BayesMVP_autodiff_fallback = TRUE)`.

Note that the backups only change how the gradient is computed at that position (i.e., they do not change the target distribution), so the sampler itself is unaffected.

## What BayesMVP adds

- Manually derived likelihood gradients, hand-coded in C++, for the MVP, LC-MVP, MVOP, LC-MVOP and latent trait models, with log-scale and autodiff backups (see [above](#how-bayesmvp-computes-the-log-posterior-and-its-gradient)).
- Cache-aware chunking of the likelihood/gradient evaluation (`num_chunks`, chosen automatically by default), which greatly improves parallel scaling at large N, as well as within-chain parallelism (WCP) through NicoStan.
- Covariates (`X`; intercept-only by default), and multiple populations/studies (`n_pops`, `pop`), each with their own disease prevalence (and Beta prior).
- Correlation matrices use the flexible Cholesky parameterisation of [Sean Pinkney](https://github.com/spinkney) ([Pinkney, 2024](https://arxiv.org/abs/2405.07286)) (the default, `corr_param = "Sean"`), which also allows the correlations to be constrained to be positive (`corr_force_positive = TRUE`), as well as custom element-specific bounds (`lb_corr`, `ub_corr`); additionally, known/fixed correlations (`known_values_indicator_list`, `known_values_list`) and a choice of correlation priors are supported.
- Ordinal and mixed binary/ordinal outcomes (MVOP, LC-MVOP), with any number of categories per test (`n_cat_per_ord_test`).
- A choice of nuisance-parameter transformations (`nuisance_transformation = "Phi"`, `"Phi_approx"`, `"tanh"` or `"inv_logit"`), and of the normal CDF (`Phi_type = "Phi"` or `"Phi_approx"`).
- Numerically stable tails: beyond the `overflow_threshold` / `underflow_threshold` (±7.5 by default), the normal CDF and its inverse are computed on the log scale.
- Manually implemented likelihood gradients for the specialised MVP, latent-class MVP, MVOP and latent-trait implementations.
- Native C++ implementations built on Eigen, Stan Math, RcppParallel and the NicoStan sampler core.
- Reusable AVX2 and AVX-512 mathematical functions for vectorised exponentials, logarithms, normal distribution functions and related kernels.
- The general Stan external-function header at:

  ```text
  inst/BayesMVP/inst/include/BayesMVP/stan_external_functions.hpp
  ```

- Stan model files and examples for using the specialised functions through CmdStanR or NicoStan.

The AVX functions are general external kernels. They are also used by the native BayesMVP implementations where the selected processor supports the required instruction set. The compiled implementation reports the detected SIMD lane count.

## The CLC-MVOP (co-morbidity) model

`Model_type = "CLC_MVOP"` implements the **co-morbidity latent class multivariate ordinal probit (CLC-MVOP)** model for jointly estimating the true prevalence, co-occurrence, and diagnostic/screening accuracy of **two conditions that can co-occur** (e.g., autism and borderline personality disorder, or heart disease and dementia), without assuming any test is a perfect gold standard.

### Model structure & latent classes
- **Four factorial latent classes:** $d \in \{(0,0), (1,0), (0,1), (1,1)\}$ representing neither condition, condition A only, condition B only, and both conditions.
- **Mixed binary and ordinal tests:** supports any combination of binary tests ($K_t = 2$) and ordinal tests with arbitrary numbers of categories ($K_t \ge 3$), without collapsing ordinal scales.
- **Own-status accuracy & cross-loadings:** each test measures its designated target condition with own-status location parameters (for binary tests) or stochastically ordered cut-points $C^{[1]} \le C^{[0]}$ (for ordinal tests). Optional **cross-loadings** (`allow_cross_loading`) allow the *other* condition to shift a test's latent mean (separately by own status, with symmetric or asymmetric priors), plus optional covariate-by-other-condition cross-loading interactions (`n_cross_cov`) and optional perfect-reference pinning (`is_perfect_ref`).
- **Covariates on both prevalence and test accuracy:** supports subject-level covariates (`X_prevalence`) on both conditions' prevalences and test-specific covariates (`X`) on measurement accuracy, either shared across disease statuses (`covariate_effects_by_own_status = 0`) or separate by own status (`covariate_effects_by_own_status = 1`, enabling differential item functioning on sensitivity vs. specificity).
- **Survey missingness & interval censoring:** handles both two-phase (trailing) and arbitrary item/test missingness cleanly under the GHK likelihood, and natively supports **interval-censored ordinal scores** (`y_upper_when_interval_censored`, e.g., partially answered questionnaires whose true total lies in $[y_{\text{lo}}, y_{\text{hi}}]$).

### Prevalence & co-occurrence parameterisations
- **Dirichlet stratum simplex (`use_OR_prior = 0`):** independent Dirichlet priors on the 4-class simplex per population/stratum.
- **Hierarchical logit-normal margins + conditional $\Pr(B \mid A)$ (`use_OR_prior = 1, use_cond_prior = 1`):** non-centred stratum-level logit-normal effects for A and B (`mu_logit_A/B`, `sigma_logit_A/B`, `z_logit_A/B`) plus prevalence covariates, linked via a shared comorbidity log-odds parameter with an exact change-of-variables prior directly on the survey-standardised conditional probability $\Pr(B \mid A)$, and optionally (`use_pop_prev_prior = 1`) an exact $3 \times 3$ Jacobian prior directly on the survey-standardised population prevalences $\mathrm{prev}_A$, $\mathrm{prev}_B$, and $\Pr(B \mid A)$.
- **Plackett bivariate copula (`use_OR_prior = 1, use_cond_prior = 0`):** exact numerically stable [Plackett (1965)](https://doi.org/10.1080/01621459.1965.10480807) $2 \times 2$ cell probabilities with either a shared (`log_OR_shared = 1`) or stratum-specific (`log_OR_shared = 0`) odds ratio.

### Test-accuracy & cut-point priors
- **Induced-Dirichlet cut-point priors:** places coherent induced-Dirichlet priors on ordinal cut-points for each own-status set.
- **Multi-cutoff `Se`/`Sp` priors (`use_Se_Sp_prior = 1`):** supports four prior families (`Se_Sp_prior_type = 1..4`: Dirichlet, Beta, logit-normal, probit-normal) on any chosen subset of binary tests and ordinal cut-offs.
- **Survey-standardised population `Se`/`Sp` priors (`use_pop_Se_Sp_prior = 1`):** places the probit-normal `Se`/`Sp` priors directly on the weighted survey-standardised population scale (with exact 1-D diagonal Jacobian and SIMD-accelerated evaluation over standardisation profiles), eliminating probit covariate-variance attenuation without anchoring priors at $X = 0$.

### Within-class correlation structures
All non-independent modes use the bounded Cholesky parameterisation of [Pinkney (2024)](https://arxiv.org/abs/2405.07286), supporting positive constraints (`corr_force_positive = 1`), custom element-wise bounds (`lb_corr`, `ub_corr`), and arbitrary fixed/known correlations (`known_values_indicator_list`, `known_values_list`):
- **P1 (single shared matrix, `pool_correlations_across_classes = 2`):** one correlation matrix shared across all four classes.
- **P2 (partial pooling, `pool_correlations_across_classes = 1`):** a global correlation matrix combined with class-specific departures via class-level mixing weights $\lambda_c \in (0, 1)$, $\Omega_c = (1 - \lambda_c)\Omega_{\text{global}} + \lambda_c \Omega_{\text{dep}, c}$.
- **P3 (separate matrices, `pool_correlations_across_classes = 0`):** an independent correlation matrix for each of the four classes.
- **P4 (class 1 own + classes 2-4 shared, `pool_correlations_across_classes = 3`):** separate correlation matrix for the neither-condition class $(0,0)$ and one shared matrix across the three morbid classes $(1,0), (0,1), (1,1)$.
- **CI (conditional independence):** when all off-diagonal correlations are fixed to $0$, BayesMVP automatically switches to an exact closed-form zero-nuisance path (`n_nuisance = 0`).

### Computation, test ordering & usage
- **Shared hand-coded gradients & chunking:** reuses BayesMVP's tiled AVX-512/AVX2 GHK forward/backward kernels, standard-scale (`NoLog`) and log-scale (`PartialLog`) hand-coded gradients, cache-aware chunking, and OpenMP within-chain parallelism (WCP).
- **Custom `latent_test_order`:** supports arbitrary latent GHK test permutations (e.g., `c(4, 3, 2, 1)`) whilst automatically unpermuting constrained parameters, correlation matrices, and accuracy summaries back to the original observed test order.
- **Summaries & LOO:** `$summary()` computes constrained parameters, transformed parameters (`Omega`, `Omega_global`, `C_vec_tp`, `beta`), survey-standardised generated quantities (`prev`, `prev_A_overall`, `prev_B_overall`, `Pr_B_given_A_overall`, `Pr_A_given_B_overall`, `Se_bin`/`Sp_bin`, `Se_ord`/`Sp_ord`), and per-observation `log_lik` traces for PSIS-LOO.

```r
library(BayesMVP)

mod <- MVP_model$new(
  Model_type = "CLC_MVOP",
  sample_nuisance = TRUE,
  model_args_list = list(
    stan_data = clc_stan_data,
    vect_type = "AVX512",
    Phi_type = "Phi_approx",
    inv_Phi_type = "inv_Phi_approx",
    nuisance_transformation = "Phi",
    use_pop_Se_Sp_prior = 1L
  )
)

fit <- mod$sample(
  partitioned_HMC = TRUE,
  diffusion_HMC = TRUE,
  n_chains_burnin = 8L,
  n_chains_sampling = 8L,
  n_burnin = 500L,
  n_iter = 1000L,
  store_log_lik_trace = TRUE,
  init_lists_per_chain = init_lists
)

res <- fit$summary(
  save_log_lik_trace = TRUE,
  compute_transformed_parameters = TRUE,
  compute_generated_quantities = TRUE
)
```

## Automatic test (column) reordering

For the latent class models (LC-MVP and LC-MVOP), BayesMVP chooses the order of the tests
(i.e., the columns of `y`) automatically, before the main burnin;
this is on by default (`reorder_cols_MVP = TRUE`;
set `reorder_cols_MVP = FALSE` to keep the original order).
The model itself does not depend on the order of the tests;
however, the unconstrained parameters which HMC samples do - and hence so does the efficiency of HMC.

**How it works:**

- A short pre-burnin (by default 125 iterations; `pre_burnin_n_iter`), on the data in the original order,
gives rough estimates of the within-class correlations, the probit intercepts
and (for ordinal tests) the cutpoints.
- The most nearly deterministic test is placed last;
i.e., for the LC-MVP, the test whose probit linear predictors are furthest from zero (summed over the two latent
classes), and, for the LC-MVOP, the test whose tail categories sit furthest from the class-conditional latent mean
(which, for a binary test, is the same measure).
- The other tests then follow the greedy correlation order: the most strongly correlated pair first,
then, one at a time, the test with the largest sum of squared correlations with the tests already placed.
- The old greedy-only rule remains available via
`test_order_rule_for_reorder_cols_MVP = "greedy_correlation_order"`
(which is also the rule used for user-supplied Stan models).
- The summaries and traces of the model parameters (e.g., the coefficients, correlations, cutpoints,
sensitivities and specificities) are returned in the original test order.

**Why it works:**

- In the GHK (truncated-normal) construction, the truncation bounds of each test depend on the latent values
of every test placed before it.
- For a nearly deterministic test, each rare discordant result has a latent value deep in the tail
(i.e., $\Phi^{-1}(u \cdot p)$, with a small $p$),
which is a steep, strongly non-linear function of the parameters;
this curvature then propagates to the bounds of every later test -
whereas, placed last, no other test depends on it.
- The order also sets the order of the correlation parameterisation
([Pinkney, 2024](https://arxiv.org/abs/2405.07286));
hence, placing the most strongly correlated tests first means that the largest dependencies are carried by the
first columns of the Cholesky factor.

**Evidence:**

- Binary LC-MVP (N = 10,000): with the old greedy-only rule, the position of two tests which are only weakly
correlated with the others was decided by the noise of the pre-burnin, so the chosen order changed between runs;
the slower order - with the most nearly deterministic test fifth, rather than last -
gave 0.59× the min ESS per 1000 gradient evaluations,
and needed 1.60× the gradient evaluations to reach the target min ESS
(over 173 comparisons within the cells of our pilot-study grid;
and 0.45× and 2.15×, respectively, when we fixed both orders as a control).
- Binary LC-MVP (N = 10,000): the new rule chose the same (faster) order in all 599 logged pre-burnins of a
600-fit grid; furthermore, when we changed the simulated data so that the greedy-only rule placed the most extreme
test first, its order needed 2.67× the gradient evaluations to reach the target min ESS
(0.33× the min ESS per 1000 gradient evaluations) of the new rule's order.
- Ordinal LC-MVOP (one binary and two ordinal tests, N = 5,000): the rule placed the binary test last
(order 2-3-1) in all 15 of our checks (5 seeds × 3 chunk counts);
in a 36-fit test of all six orders, the two orders with the binary test last needed 0.67-0.75× the gradient
evaluations of the input order (1-2-3) to reach the target min ESS,
and had 1.37-1.56× its min ESS per 1000 gradient evaluations.

## Installation order

BayesMVP depends on NicoStan. Install NicoStan first, then BayesMVP.

For installation from GitHub, start with a fresh R session and a writable package library:

```r
install.packages(c("remotes", "devtools"))
remotes::install_github(repo = "CerulloE1996/NicoStan", upgrade = "never")
NicoStan::install_NicoStan()
## Restart R, then install BayesMVP from the main branch:
remotes::install_github(repo = "CerulloE1996/BayesMVP", ref = "main", upgrade = "never")
BayesMVP::install_BayesMVP()
## Restart R, then load library(BayesMVP).
```

For a local source checkout, the local installer checks NicoStan and installs or repairs it automatically when needed, then compiles BayesMVP against it.

The source tree includes one local administrator installer at:

```text
inst/examples/BayesMVP_admin_install.R
```

Run this directly in the RStudio console after placing the NicoStan and BayesMVP source directories next to one another in `R_packages`:

```r
source("path/to/R_packages/BayesMVP/inst/examples/BayesMVP_admin_install.R")
```

The same command installs or reinstalls the package. It launches a clean R process automatically and leaves the calling session's namespaces alone. You can also rerun `run_BayesMVP_admin_install()` after sourcing. There is no need to open a terminal or disable workspace restoration to start the installation.

The installer:

- locates the local BayesMVP source tree without hard-coded workstation or laptop paths;
- checks NicoStan's version, native library, exported headers and required R exports, repairing a missing, outdated or incomplete installation from the adjacent source tree;
- loads the outer installer only in a temporary build directory, which is removed afterwards;
- compiles the inner BayesMVP package into your selected R library, retaining R's normal installation rollback if compilation fails;
- preserves your library paths and optional compiler flags in the clean build process;
- verifies the required BayesMVP exports before returning.

For explicit library paths, set `BAYESMVP_INSTALL_LIB` and `NICOSTAN_INSTALL_LIB` before sourcing the installer. Alternatively, disable automatic execution before sourcing, then call the function with your arguments:

```r
options(BayesMVP.admin.autorun = FALSE)
source("path/to/R_packages/BayesMVP/inst/examples/BayesMVP_admin_install.R")
run_BayesMVP_admin_install(
    source_root = "path/to/R_packages/BayesMVP",
    lib = "/path/to/bayesmvp-library",
    nicostan_lib = "/path/to/nicostan-library"
)
```

Both packages use `lib` by default; `nicostan_lib` or `NICOSTAN_INSTALL_LIB` can select a separate dependency library when explicitly needed. The build uses the ordinary compiler and CPU detection by default. Pass optional named `CUSTOM_FLAGS` to `run_BayesMVP_admin_install()`. If a previous build was already loaded, restart R after installation to use the new build. On Windows, a DLL locked by an open R session may require that session to be restarted before replacement.

## Using the external AVX functions

When compiling a compatible Stan model with CmdStanR, use the exported header supplied by the installed extension:

```r
header <- system.file(
    "include", "BayesMVP", "stan_external_functions.hpp",
    package = "BayesMVP"
)

model <- cmdstanr::cmdstan_model(
    stan_file = "path/to/model.stan",
    user_header = header,
    force_recompile = TRUE
)
```

The Stan model must declare the external functions that it calls. The general NicoStan model examples use the same header through their `math_backend = "AVX2"` or `math_backend = "AVX512"` choices.

## Relationship with NicoStan

BayesMVP is the specialised extension. NicoStan remains the general package and can fit any Stan model. The dependency direction is:

```text
NicoStan sampler and diagnostics
            ↓
BayesMVP manual-gradient and AVX model extension
```

The two-class latent-class MVP and latent-trait implementations were used in the simulation study by Cerullo et al. (2025). The mixed binary and ordinal test-accuracy models are described by Cerullo et al. (2022).

## License

BayesMVP is licensed under GPL-3. See [LICENSE](LICENSE) for the complete licence text.

