# BayesMVP inner package

This is the compiled inner package for the [BayesMVP extension](../../README.md).
Install NicoStan first. BayesMVP then adds manually implemented gradients and
specialised native implementations for:

- MVP;
- two-class LC-MVP;
- MVOP;
- two-class LC-MVOP;
- the two-class latent-trait model.

The reusable AVX2 and AVX-512 external functions are exported through
`inst/include/BayesMVP/stan_external_functions.hpp` in the outer source tree.
The outer administrator installer compiles this inner package in a fresh R
process after checking the NicoStan dependency and its exported headers.
