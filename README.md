# BayesMVP

BayesMVP is the specialised model extension for [NicoStan](https://github.com/CerulloE1996/NicoStan). NicoStan supplies the general Stan-compatible sampler, warm-up, trajectory-length adaptation and diagnostics. BayesMVP adds native manually implemented gradients and model-specific C++ code for multivariate probit models.

BayesMVP was originally a standalone R package (and NicoStan was formerly called "BayesMVP"); however, it has now been split into 2 R packages: [NicoStan](https://github.com/CerulloE1996/NicoStan), for the general Stan interface (which relies heavily on [BridgeStan](https://roualdes.us/bridgestan/latest/)), and BayesMVP, which is now an extension to NicoStan, specifically for the extremely efficient fitting of multivariate probit (MVP) based models (including the latent class MVP [LC-MVP] and the latent trait model, for both binary and/or ordinal outcomes/tests).

The extension contains:

- **MVP:** the multivariate probit model for correlated binary outcomes.
- **2LC-MVP:** the two-class latent-class multivariate probit model, selected with `Model_type = "LC_MVP"`.
- **MVOP:** the multivariate ordinal probit model.
- **2LC-MVOP:** the two-class latent-class multivariate ordinal probit model.
- **Two-class latent trait:** the two-class latent-trait model, selected with `Model_type = "latent_trait"`.

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
- Correlation matrices use the flexible Cholesky parameterisation of [Sean Pinkney](https://github.com/spinkney) ([Pinkney, 2024](https://arxiv.org/abs/2405.07286)) (the default, `corr_param = "Sean"`), which also allows the correlations to be constrained to be positive (`corr_force_positive = TRUE`); additionally, known/fixed correlations (`known_values_list`) and a choice of correlation priors are supported.
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

