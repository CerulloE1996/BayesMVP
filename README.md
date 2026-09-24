# BayesMVP

BayesMVP is the specialised model extension for [NicoStan](https://github.com/CerulloE1996/NicoStan). NicoStan supplies the general Stan-compatible sampler, warm-up, trajectory-length adaptation and diagnostics. BayesMVP adds native manually implemented gradients and model-specific C++ code for multivariate probit models.

The extension contains:

- **MVP:** the multivariate probit model for correlated binary outcomes.
- **2LC-MVP:** the two-class latent-class multivariate probit model, selected with `Model_type = "LC_MVP"`.
- **MVOP:** the multivariate ordinal probit model.
- **2LC-MVOP:** the two-class latent-class multivariate ordinal probit model.
- **Two-class latent trait:** the two-class latent-trait model, selected with `Model_type = "latent_trait"`.

These models are used for correlated binary and ordinal outcomes, including diagnostic and screening test-accuracy models without a perfect reference standard. The ordinal latent-class model is described in [Cerullo et al. (2022)](https://doi.org/10.1002/jrsm.1567), and the latent-class MVP versus latent-trait simulation study is reported in [Cerullo et al. (2025)](https://arxiv.org/abs/2509.18489v1).

## What BayesMVP adds

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

