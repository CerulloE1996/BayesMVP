BayesMVP uses diffusion-pathspace adaptive Hamiltonian Monte Carlo (HMC) to efficiently sample the 
multivariate probit model (MVP) - which is used to model correlated binary data - as well as the latent class MVP
model (LC-MVP) and the latent trait model, which are commonly used in medical applications to model diagnostic 
and/or screening test accuracy data. 

In addition, it can also **sample any user-supplied Stan model**, and performs best for models with 
a high-dimensional latent variable vector (or "nuisance parameters"). 

BayesMVP makes use of two state-of-the-art HMC algorithms. For the burnin phase, it uses an algorithm which 
is based on the recently proposed **SNAPER-HMC** (Sountsov et al, 2022). For the sampling (i.e., post-burnin)
phase, it uses standard HMC _(with randomized path length)_ to sample the main model parameters, and then 
it samples the nuisance parameters using **diffusion-pathspace HMC** (Beskos et al, 2013). 

Furthermore, specifically for the three built-in models (i.e. the MVP, LC_MVP, and latent_trait), 
it achieves rapid sampling by using: 
manually-derived gradients,
chunking, 
and (on systems with AVX-512 and/or AVX2) fast approximate, vectorised (SIMD) math functions. 

Users can also use the optimised manual-gradient lp_grad functions for the 3 built-in models with Stan directly 
(via the cmdstanr R package) by downloading/installing the R package, and  then, when you compile your Stan model 
with cmdstanr using cmdstan_model(), use the user_header argument as follows: 

      ## path to Stan model
      file <- file.path(pkg_dir, "inst/stan_models/LC_MVP_bin_w_mnl_cpp_grad_v1.stan") 
      ## path to the C++ .hpp header file
      path_to_cpp_user_header <- file.path(pkg_dir, "src/lp_grad_fn_for_Stan.hpp") 
      ## compile model together with the C++ functions
      mod <- cmdstan_model(file,  force_recompile = TRUE, user_header = path_to_cpp_user_header) 




References:

SNAPER-HMC:  https://arxiv.org/abs/2110.11576v1

Diffusion-pathspace HMC: https://www.sciencedirect.com/science/article/pii/S0304414912002621
