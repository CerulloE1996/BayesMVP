# Development source snapshot - 20 September 2026

This branch brings the current local BayesMVP development source into the existing GitHub repository. It is a development snapshot, not a validated release.

The outer R package retains the existing installation wrapper. The sampler implementation is in `inst/BayesMVP/`, including the R6 interface, R adaptation code, C++ kernels, Stan source models, and generated R documentation. `APMS_bridge/` contains development bridge code and test sources; `dev_experiments/` contains development experiment scripts.

Files copied from the working package retain their existing contents and experimental settings. The public root README is retained because it contains alpha-status caveats absent from the local copy. Its method descriptions and the package metadata have not been reviewed for this snapshot and may be outdated.

Local assistant configuration, chat exports, cached Stan data, generated test results, compiled package objects, and compiled example/test models are excluded. Bundled TBB and dummy-model libraries required by the existing installation code are retained. Files removed or replaced in the local development source are also removed or replaced here.

The APMS bridge and experiment scripts contain workstation-specific paths and depend on models/helpers outside this repository. Publishing their source does not make those workflows self-contained. In particular, external APMS fused/AVX model variants are not bundled by this sync.

Validation for this snapshot is limited to source parsing and static consistency checks. Both packages' R source directories parse, every exported name has a top-level definition, and all 114 Rd documentation files parse without errors or warnings. Across the full development tree, 97 of 100 R files parse. The three failures are existing scratch/debug scripts containing console output:

- `Eg_2a_mnl_LC_MVP_debug.R`, line 991.
- `inst/BayesMVP/src/MVP_functions/ewafqegf.R`, line 6.
- `inst/examples/KEEP_DEBUG_Eg_2_mnl_LC_MVP_and_latent_trait_LINUX.R`, line 706.

These files are preserved as development source; they are not loaded from either package's `R/` directory. No package installation, C++ rebuild, MCMC run, numerical-equivalence test, or performance benchmark is implied. The existing package version has not been changed.
