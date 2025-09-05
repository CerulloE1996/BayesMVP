# BayesMVP [v0.1 - Under Active Development]

**🚧 Early release - Breaking changes expected until v1.0**

### Current Status: Alpha
- Core functionality should be mostly working.
- Successfully used in research (10,000+ model fits for LC-MVP and latent_trait models).
- ChESSR-HMC burnin algorithm working & stable on tested built-in models & selected Stan models.
- SNAPER-HMC burnin algorithm is not consistent/stable yet.
- API still evolving.
- Limited error handling.
- Actively improving.

BayesMVP uses a highly-efficient, recently proposed state-of-the-art HMC algorithm called SNAPER Hamiltonian Monte Carlo (SNAPER-HMC; Sountsov & Hoffman et al, 2022) to sample the posterior distribution. 
Furthermore, depending on the model (and if the user enables this experimenal option), it also makes use of diffusion-pathspace HMC (Beskos et al, 2013) for models which have high-dimensional guassian latent variables. 

BayesMVP was designed specifcially to efficiently sample the following models:
- The multivariate probit model (MVP)
- The latent class MVP model (LC-MVP), (currently) with 2 latent classes. 
- THe latent trait latent class model, (currently) with 2 latent classes.

The MVP model is used generally to model correlated binary data across various fields, whereas the latter two more complex latent class models tend to be used more specifically in medical
applications to model diagnostic and screening test accuracy data without the presence of a perfect reference (or "gold standard") test. 

In addition to these "built-in" models, BayesMVP can also **sample any user-supplied Stan model**. However, BayesMVP will perform best for models which 
have high-dimensional guassian latent variables (or "nuisance parameters"), particularly when the diffusion-pathspace
HMC option is enabled. That being said, even for models without such guassian latent variables, BayesMVP can sometimes 
still outperform Stan substantially thanks to the burn-in algorithm it uses = which is based on SNAPER-HMC - an algorithm 
which is generally much more efficient than the NUTS-based algorithm Stan uses during adaptation. However, this will be highly
model-dependent, and as it currently standard BayesMVP has mostly only been formally tested on models based on the 3 "built-in" 
models (i.e. the MVP, LC-MVP and the latent trait model), hence the performance and efficiency outcomes for other models 
(especially for models without high-dimensional nuisance parameters) still needs to be evaluated. 

BayesMVP makes use of two state-of-the-art HMC algorithms:

- For the burnin phase, it uses an algorithm which is based on the recently proposed **SNAPER-HMC** (Sountsov et al, 2022). BayesMVP will use this algorithm for all models, including user-supplied Stan models. 
- For the sampling (i.e., post-burnin) phase, it uses standard HMC _(with randomized path length)_ to sample the main model parameters, and then, it samples the nuisance parameters using the **diffusion-pathspace HMC** algorithm (Beskos et al, 2013), which is an algorithm that is specifically designed to sample models which have a high-dimensional guassian latent variables, such as multivariate probit models. Note that the latter feature of BayesMVP is **optional** and is considered an "experimental" feature of the R package. 

Furthermore, specifically for the three built-in models (i.e. the MVP, LC_MVP, and latent_trait), 
it achieves particularly rapid sampling (despite using full MCMC - not approximate Bayes methods) by using a veriety of techniques, including:
- (i) Manually-derived gradients (all implemented in C++ using the Eigen and stan::math libraries),
- (ii) "Chunking" - a technique where large objects (e.g. Eigen::Matrix objects in C++) are partitioned into smaller blocks to prevent so-called "cache misses", and:
- (iii) Fast approximate, vectorised (or "SIMD") math functions (on systems with AVX-512 and/or AVX2 CPU instruction sets - on systems without either of these instruction sets, BayesMVP will use a combination of the Eigen and stan::math C++ libraries for vectorisation).  

--------------------------------------------------------------------------------------------------------------------------------------
Before installing BayesMVP, first you must first install **cmdstanr** as well as the **bridgestan** R packages. 

**To install cmdstanr:**

To install the cmdstanr R package, the following worked on both my Linux and Windows systems:

```
        #### Install the cmdstanr "outer" R package:
        remotes::install_github("stan-dev/cmdstanr", force = TRUE)
        #### Load cmdstanr R outer package:
        require(cmdstanr) 
        #### Then, install the latest version of CmdStan:
        install_cmdstan(cores = parallel::detectCores(),
                        overwrite = TRUE,
                        cpp_options = list("STAN_MODEL_LDFLAGS" = "-shared",   "CXXFLAGS" = "-fPIC"))
```
               
   

**To install bridgestan:**

Please also see the guide here: https://roualdes.github.io/bridgestan/latest/languages/r.html

To install the bridgestan R package, the following worked on both my Linux and Windows systems:

```
remotes::install_github("https://github.com/roualdes/bridgestan", subdir="R")
#### Load bridgestan:
require(bridgestan)
```


Then, you should be able to install BayesMVP by doing the following:

```
## First remove any possible package fragments:
## Find user_pkg_install_dir:
user_pkg_install_dir <- Sys.getenv("R_LIBS_USER")
print(paste("user_pkg_install_dir = ", user_pkg_install_dir))
##
## Find pkg_install_path + pkg_temp_install_path:
pkg_install_path <- file.path(user_pkg_install_dir, "BayesMVP")
pkg_temp_install_path <- file.path(user_pkg_install_dir, "00LOCK-BayesMVP")
##
## Remove any (possible) BayesMVP package fragments:
remove.packages("BayesMVP")
unlink(pkg_install_path, recursive = TRUE, force = TRUE)
unlink(pkg_temp_install_path, recursive = TRUE, force = TRUE)
##
## First install OUTER package:
remotes::install_github("https://github.com/CerulloE1996/BayesMVP", force = TRUE, upgrade = "never")
## Then restart R session:
rstudioapi::restartSession()
## Then install INNTER (i.e. the "real") package:
require(BayesMVP)
BayesMVP::install_BayesMVP()
require(BayesMVP)
```


--------------------------------------------------------------------------------------------------------------------------------------


Users can also use the optimised manual-gradient lp_grad functions for the 3 built-in models with Stan directly 
(via the cmdstanr R package) by downloading/installing the R package, and  then, when you compile your Stan model 
with cmdstanr using cmdstan_model(), use the user_header argument as follows: 

      ## path to Stan model
      file <- file.path(pkg_dir, "inst/stan_models/LC_MVP_bin_w_mnl_cpp_grad_v1.stan") 
      ## path to the C++ .hpp header file
      path_to_cpp_user_header <- file.path(pkg_dir, "src/lp_grad_fn_for_Stan.hpp") 
      ## compile model together with the C++ functions
      mod <- cmdstan_model(file,  force_recompile = TRUE, user_header = path_to_cpp_user_header) 




--------------------------------------------------------------------------------------------------------------------------------------



**References:**

SNAPER-HMC:  https://arxiv.org/abs/2110.11576v1

Diffusion-pathspace HMC: https://www.sciencedirect.com/science/article/pii/S0304414912002621
