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

Approximate-CDF model paths using `Phi_approx` / `inv_Phi_approx` are experimental and are not supported for production native fits. Finite-difference checks found gradient failures when these settings were forced into the native MVOP and LC-MVOP `NoLog` paths. The native defaults remain `Phi` / `inv_Phi`, and the R interface rejects approximate-CDF model settings. This limitation concerns the model CDF/inverse-CDF choice; `nuisance_transformation = "Phi_approx"` is a separate setting.
