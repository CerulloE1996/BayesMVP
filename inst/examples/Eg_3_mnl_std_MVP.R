

########## -------- EXAMPLE 3 -------------------------------------------------------------------------------------------------------------------------------- 
## Running the BUILT-IN (i.e. manual gradients) MVP model
## Uses simulated data.





####  ---- 1. Install BayesMVP (from GitHub) - SKIP THIS STEP IF INSTALLED: -----------------------------------------------------------
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

# require(BayesMVP)
# CUSTOM_FLAGS <- list()
# install_BayesMVP(CUSTOM_FLAGS = list())
# require(BayesMVP) 





####  ---- 2. Set BayesMVP example path and set working directory:  --------------------------------------------------------------------
{
  user_dir_outs <- BayesMVP:::set_pkg_example_path_and_wd()
  ## Set paths:
  user_root_dir <- user_dir_outs$user_root_dir
  user_BayesMVP_dir <- user_dir_outs$user_BayesMVP_dir
  pkg_example_path <- user_dir_outs$pkg_example_path
}




####  ---- 3. Set options   ------------------------------------------------------------------------------------------------------------
{
  options(scipen = 99999)
  options(max.print = 1000000000)
  options(mc.cores = parallel::detectCores())
}




####  ---- 4. Now run the example:   ---------------------------------------------------------------------------------------------------
require(BayesMVP)

## Function to check BayesMVP AVX support 
BayesMVP:::detect_vectorization_support()


Model_type <- "MVP"




# ------------------------------  Set basic specs / options
{
  options(scipen = 99999)
  options(max.print = 1000000)
  options(mc.cores = parallel::detectCores())
  numerical_diff_e = 0.001
  
  Model_type <- "MVP"
  set.seed(1, kind = "L'Ecuyer-CMRG")
  
  N <-  5000
  
  n_outcomes <- n_tests <- 4
  
  n_class <- 1 # NON-LCM model so only 1 class. 
  
  n_chains = round(parallel::detectCores()/2)
  n_iter = 500
}

 


# ------------------------------- SIMULATE data (assumed n_outcomes = 4 and non-LCM - modify for other values)
{
  
# n_covariates_per_outcome_vec <-  vector("list", length = n_class)
# n_covariates_per_outcome_vec[[1]] <- rep(NA, n_outcomes)
n_covariates_per_outcome_vec <- array(0, dim = c(n_class, n_outcomes))
# first 3 outcomes have SINGLE BINARY COVARIATE and a SINGLE CONTINUOUS COVARIATE  [put cts covs FIRST and then binary covs SECOND and then cat covs THIRD]
n_covariates_per_outcome_vec[1, 1:3] <-  2
X_for_outcomes_w_single_bin_and_single_cts_cov <-  array(dim  = c(3, N, n_covariates_per_outcome_vec[1, 1]))

for (t in 1:3) {  ######  TEMP (change to cat w/ 3 categories!!)
  set.seed(t, kind = "L'Ecuyer-CMRG")
  X_for_outcomes_w_single_bin_and_single_cts_cov[t, 1:N, 1] <-  rnorm(n = N, mean = 1, sd = 1)  # cts covariates
  X_for_outcomes_w_single_bin_and_single_cts_cov[t, 1:N, 2] <-  rbinom(n = N, size = 1, prob = 0.25) # binary covariates
}
set.seed(1, kind = "L'Ecuyer-CMRG") # reset seed to 1

true_betas_for_outcome_1 <- c(-2.50, +0.25)
true_betas_for_outcome_2 <- true_betas_for_outcome_3  <- true_betas_for_outcome_1

# 3rd outcome has 1 CATEGORICAL COVARIATE w/ 3 categories (remember - MVP  has NO INTERCEPT if there are covariates)
# n_covariates_per_outcome_vec[[1]][4] <-    1
# X_for_outcome_w_single_cat_cov <- rbinom(n = N, size = 1, prob = 0.75)
# true_betas_for_outcome_4_cat_0_vs_cat_1 <- -0.50
# true_betas_for_outcome_4_cat_0_vs_cat_2 <- +0.50
 


# 4th outcome has 8 CONTINUOUS COVARIATES
n_covariates_per_outcome_vec[1, 4] <-    8
X_for_outcome_w_8_continuous_covs <- array(dim = c(N, n_covariates_per_outcome_vec[1, 4]))


for (k in 1:n_covariates_per_outcome_vec[1, 4]) {
  set.seed(k, kind = "L'Ecuyer-CMRG")
   kk = k - n_covariates_per_outcome_vec[1, 4]/2
   X_for_outcome_w_8_continuous_covs[1:N, k] <- rnorm(n = N, mean =  0.25*kk, sd = 1.0)
}
set.seed(1, kind = "L'Ecuyer-CMRG") # reset seed to 1

true_betas_for_outcome_4 <- c(-0.5, 1.5, 0.5, -0.25, -1.0, 0.5, 0, 0)

true_betas_all_outcomes_vec <- c(true_betas_for_outcome_1,  true_betas_for_outcome_2, true_betas_for_outcome_3, true_betas_for_outcome_4) # , true_betas_for_outcome_5)

# now make X (FULL covariate data array)
n_covariates_max <- max(n_covariates_per_outcome_vec)
X <- list(vector("list", n_tests))

str(X_for_outcomes_w_single_bin_and_single_cts_cov[1,,])

X[[1]][[1]] <- X_for_outcomes_w_single_bin_and_single_cts_cov[1,,]
X[[1]][[2]] <- X_for_outcomes_w_single_bin_and_single_cts_cov[2,,]
X[[1]][[3]] <- X_for_outcomes_w_single_bin_and_single_cts_cov[3,,]
X[[1]][[4]] <- X_for_outcome_w_8_continuous_covs

# check X correct format (should be a list of lists, and then each inner element is a matrix with N columns)
str(X)

# X  <- array(999999, dim = c(n_tests, n_covariates_max, N))
# X[1:3, 1:n_covariates_per_outcome_vec[1, 1], 1:N] <- X_for_outcomes_w_single_bin_and_single_cts_cov
# X[4, 1:n_covariates_per_outcome_vec[1, 4], 1:N] <- t(X_for_outcome_w_8_continuous_covs)


true_beta_vals <- array(999999, dim = c(n_tests, n_covariates_max))

true_beta_vals[1, 1:n_covariates_per_outcome_vec[1, 1]] <- true_betas_for_outcome_1
true_beta_vals[2, 1:n_covariates_per_outcome_vec[1, 2]] <- true_betas_for_outcome_2
true_beta_vals[3, 1:n_covariates_per_outcome_vec[1, 3]] <- true_betas_for_outcome_3
true_beta_vals[4, 1:n_covariates_per_outcome_vec[1, 4]] <- true_betas_for_outcome_4
## true_beta_vals[5, 1:n_covariates_per_outcome_vec[5]] <- true_betas_for_outcome_5


true_Xbeta_vals <- array(dim = c(N, n_tests))
for (t in 1:n_tests) {
  true_Xbeta_vals[1:N, t] <-  (X[[1]][[t]][1:N, 1:n_covariates_per_outcome_vec[1, t]]) %*% (true_beta_vals[t, 1:n_covariates_per_outcome_vec[1, t]])
}
 
 
}

## str(X)




# ------------------------------  generate data
{
  
  y_master_list_seed_123_datasets <- list()
  
  N_datasets <- 123
  
  y_list <- list()
  
  for (ii in 1:N_datasets) {
    
    df_sim_seed <- ii
    
    set.seed(df_sim_seed, kind = "L'Ecuyer-CMRG")
    
    Sigma_highly_varied <- matrix(c(1,  0,     0,        0,       
                                    0,  1,     0.50,     0.25,    
                                    0,  0.50,  1,        0.40,   
                                    0,  0.25,  0.40,     1    ),   
                                  n_tests, n_tests)
 
  
      
      Sigma  <- 1.5  * Sigma_highly_varied # / 2
      diag(Sigma) <- rep(1, n_tests)
    
    
    latent_results <- LaplacesDemon::rmvn(n = N, mu = (true_Xbeta_vals), Sigma = Sigma)
    results <- ifelse(latent_results > 0, 1, 0)
    y <- results
    
    print(Sigma)

    y_list[[ii]] <- y
    
    
  }
  
  if (N == 500)   y_master_list_seed_123_datasets[[1]] <- y_list[[123]]
  if (N == 1000)  y_master_list_seed_123_datasets[[2]] <- y_list[[123]]
  if (N == 2500)  y_master_list_seed_123_datasets[[3]] <- y_list[[123]]
  if (N == 5000)  y_master_list_seed_123_datasets[[4]] <- y_list[[123]]
  if (N == 12500) y_master_list_seed_123_datasets[[5]] <- y_list[[123]]
  if (N == 25000) y_master_list_seed_123_datasets[[6]] <- y_list[[123]]
  
}




# -------------------------------  Set PRIORS
# any priors not specified will be set to defaults. 
{
  
  beta_prior_mean <- beta_prior_sd  <-  vector("list", length = n_class)
  beta_prior_mean[[1]] <- array(0, dim = c(n_covariates_max, n_tests))
  beta_prior_sd[[1]] <- array(5, dim = c(n_covariates_max, n_tests))
  
  lkj_cholesky_eta <- 2
  corr_force_positive <- FALSE
  
}





 
 

# ------------------------------  Set INITIAL VALUES ("inits") -----------------------------------------------------------------------
# any inits not specified will be set to defaults. 

{
  
{
  
  init_vals_beta <-  vector("list", length = n_class)
  init_vals_beta[[1]] <- array(0, dim = c(n_tests, n_covariates_max))
  init_vals_beta[[1]] <-  ifelse(true_beta_vals == 999999.0, 0, true_beta_vals)
  init_vals_beta[[1]] <- t(init_vals_beta[[1]])
  
  # make beta vector (make sure to loop through class first, then outcome/test, then n_covs_per_outcome)
  beta_vec_init <- c()
  counter <- 1
  for (c in 1 : n_class) {
      for (t in 1:n_tests) {
        for (k in 1:n_covariates_per_outcome_vec[c, t]) {
          beta_vec_init[counter] <- init_vals_beta[[c]][k, t]
          counter <- counter + 1
        }
      }
  }
  
  ### init values for correlations 
  k_choose_2 <- choose(n_tests, 2)
  km1_choose_2 = 0.5 * (n_tests - 2) * (n_tests - 1)
  known_num = 0
  
  for (i in 1:n_class) {
    off_raw<-  (c(rep(0.01, km1_choose_2 - known_num)))
    col_one_raw <-  (c(rep(0.01, n_tests - 1)))
  }
  
  ## init values for nuisance parameters 
  u_raw <- array(0.01, dim = c(N, n_tests))
  
}


## init values list 
init = list(
  u_raw = (u_raw),
  beta_vec = beta_vec_init,
  off_raw = off_raw,
  col_one_raw =  col_one_raw
)

## number of parameters
n_params_main <-   sum(n_covariates_per_outcome_vec) + choose(n_tests, 2)
n_nuisance <- N * n_tests

}





## -----------  initialise model / inits etc
# based on (informal) testing, more than 8 burnin chains seems unnecessary 
# and probably not worth the extra overhead (even on a 96-core AMD EPYC Genoa CPU)
n_chains_burnin <- min(8, parallel::detectCores())
init_lists_per_chain <- rep(list(init), n_chains_burnin) 


model_args_list <- list(         n_covariates_per_outcome_mat = n_covariates_per_outcome_vec,  
                                 num_chunks =  1,# BayesMVP:::find_num_chunks_MVP(N, n_tests),
                                 X = X,
                                 lkj_cholesky_eta = lkj_cholesky_eta,
                                 corr_force_positive= corr_force_positive,
                                 prior_coeffs_mean_mat = beta_prior_mean,
                                 prior_coeffs_sd_mat =    beta_prior_sd)

model_args_list$prior_coeffs_mean_mat

###  -----------  Compile + initialise the model using "MVP_model$new(...)" 
# ?BayesMVP::MVP_model
model_obj <- BayesMVP::MVP_model$new(   Model_type = Model_type,
                                        y = y,
                                        N = N,
                                        model_args_list = model_args_list, # this arg is only needed for BUILT-IN (not Stan) models
                                        init_lists_per_chain = init_lists_per_chain,
                                        sample_nuisance = TRUE,
                                        n_chains_burnin = n_chains_burnin,
                                        n_params_main = n_params_main,
                                        n_nuisance = n_nuisance)



# mat <- init_model_and_vals_object$init_model_object$Model_args_as_Rcpp_List$Model_args_mats_int[[1]]
# is.matrix(mat)
   


## ----------- Set basic sampler settings
{
  ### seed <- 123
  n_chains_sampling <- max(64, parallel::detectCores() / 2)
  n_superchains <- min(8, parallel::detectCores() / 2)  ## round(n_chains_sampling / n_chains_burnin) # Each superchain is a "group" or "nest" of chains. If using ~8 chains or less, set this to 1. 
  n_iter <- 1000                         
  n_burnin <- 500
  n_nuisance_to_track <- 10 # set to some small number (< 10) if don't care about making inference on nuisance params (which is most of the time!)
}

#### ------ sample model using "  model_obj$sample()" --------- 
##  NOTE: The model can also use "model_obj$sample()" to update the model.
##
##  For example, if using the same model but a new/different dataset (so new y and N, and n_nuisance is needed), use:
##  model_obj$sample(y = y, N = N, n_nuisance = n_nuisance, ...)
##
##  model_args_list can also be updated. 
##  For example, say the aim was to force correlations to be positive:
model_args_list$corr_force_positive <- FALSE


# lkj_cholesky_eta_mat <- matrix(lkj_cholesky_eta)
# model_args_list$lkj_cholesky_eta <- lkj_cholesky_eta

n_tests
choose(4, 2)



## To run standard HMC, do:
partitioned_HMC <- FALSE ;    diffusion_HMC <- FALSE
## To run * partitioned * HMC (i.e. sample nuisance and main params. seperately), do:
# partitioned_HMC <- TRUE ;     diffusion_HMC <- FALSE # fine
## To run partitioned * and * diffusion HMC (i.e., nuisance params. sampled using diffusion-pathspace HMC), do:
# partitioned_HMC <- TRUE ;    diffusion_HMC <- TRUE  # fine


model_samples <-  model_obj$sample(  partitioned_HMC = partitioned_HMC,
                                     diffusion_HMC = diffusion_HMC,
                                     seed = 1,
                                     n_burnin = n_burnin,
                                     n_iter = n_iter,
                                     n_chains_sampling = n_chains_sampling,
                                     n_superchains = n_superchains,
                                     ## Some other arguments:
                                     # y = y,
                                     # N = N,
                                     # n_params_main = n_params_main,
                                     # n_nuisance = n_nuisance,
                                     # init_lists_per_chain = init_lists_per_chain,
                                     # n_chains_burnin = n_chains_burnin,
                                     model_args_list = model_args_list,
                                     ## Some other SAMPLER / MCMC arguments:
                                     # sample_nuisance = TRUE,
                                     force_autodiff = FALSE,
                                     force_PartialLog = FALSE,
                                     multi_attempts = FALSE,
                                     adapt_delta = 0.80,
                                     learning_rate = 0.05,
                                     # metric_shape_main = "dense",
                                     # metric_type_main = "Hessian",
                                     # tau_mult = 2.0,
                                     # clip_iter = 25,
                                     # interval_width_main = 50,
                                     # ratio_M_us = 0.25,
                                     # ratio_M_main = 0.25,
                                      parallel_method = "RcppParallel",
                                      vect_type = "Stan",
                                     # vect_type = "AVX512",
                                     # vect_type = "AVX2",
                                     n_nuisance_to_track = n_nuisance_to_track)   







#### --- MODEL RESULTS SUMMARY + DIAGNOSTICS -------------------------------------------------------------
# after fitting, call the "summary()" method to compute + extract e.g. model summaries + traces + plotting methods 
# model_fit <- model_samples$summary() # to call "summary()" w/ default options json_file_path
require(bridgestan)
model_fit <- model_samples$summary(save_log_lik_trace = TRUE, 
                                   # compute_nested_rhat = FALSE,
                                   compute_generated_quantities = FALSE, ## We don't have any gen_quantities for std-MVP
                                   # compute_transformed_parameters = FALSE
) 


# ?BayesMVP::MVP_model


# extract # divergences + % of sampling iterations which have divergences
model_fit$get_divergences()

# HMC_info <- model_fit$get_HMC_info()
# L_main <- HMC_info$tau_main/HMC_info$eps_main
# L_us <- HMC_info$tau_us/HMC_info$eps_us
# 
# n_grad_evals_main_sampling <- n_iter * n_chains_sampling * L_main
# n_grad_evals_us_sampling <- n_iter * n_chains_sampling * L_us
# n_grad_evals_total_sampling <- n_grad_evals_us_sampling + n_grad_evals_main_sampling
# 
# Min_ESS <- model_fit$get_efficiency_metrics()$Min_ESS_main
# 
# 
# 
# ESS_per_grad_total_sampling <- Min_ESS / (n_grad_evals_total_sampling / 2) ; ESS_per_grad_total_sampling
# 1.32 / (ESS_per_grad_total_sampling * 1000)
# 


###### --- TRACE PLOTS  ----------------------------------------------------------------------------------
# trace_plots_all <- model_samples$plot_traces() # if want the trace for all parameters 
trace_plots <- model_fit$plot_traces(params = c("beta", "Omega"), 
                                     batch_size = 6)

# parameters can be extracted using: "trace_plots$param_name()". 
# For example:
# display each panel for beta and Omega ("batch_size" controls the # of plots per panel. Default = 9)
trace_plots$beta[[1]] # 1st panel
trace_plots$beta[[2]] # 2nd panel
trace_plots$beta[[3]] # 3rd (and last) panel

# display each panel for Omega ("batch_size" controls the # of plots per panel.  Default = 9)
trace_plots$Omega[[1]] # 1st panel
trace_plots$Omega[[2]] # 2nd panel
trace_plots$Omega[[3]] # 3rd (and last) panel



###### --- POSTERIOR DENSITY PLOTS -------------------------------------------------------------------------
# density_plots_all <- model_samples$plot_densities() # if want the densities for all parameters 
# Let's plot the densities for: sensitivity, specificity, and prevalence 
density_plots <- model_fit$plot_densities( params = c("beta", "Omega"), 
                                         batch_size = 6)

# parameters can be extracted using: "density_plots$param_name()". 
# For example:
# display each panel for beta and Omega ("batch_size" controls the # of plots per panel. Default = 9)
density_plots$beta[[1]] # 1st panel
density_plots$beta[[2]] # 2nd panel
density_plots$beta[[3]] # 3rd (and last) panel

# display each panel for Omega ("batch_size" controls the # of plots per panel.  Default = 9)
density_plots$Omega[[1]] # 1st panel
density_plots$Omega[[2]] # 2nd panel
density_plots$Omega[[3]] # 3rd (and last) panel




###### --- OTHER FEATURES -------------------------------------------------------------------------
## The "model_summary" object (created using the "$summary()" method) contains many useful objects. 
## For example:
require(dplyr)
# nice summary tibble for main parameters, includes ESS/Rhat, etc
model_fit$get_summary_main() %>% print(n = 50) 
# nice summary tibble for transformed parameters, includes ESS/Rhat, etc
model_fit$get_summary_transformed() %>% print(n = 150) 
# nice summary tibble for generated quantities, includes ESS/Rhat, etc (for LC-MVP this includes Se/Sp/prevalence)
model_fit$get_summary_generated_quantities () %>% print(n = 150) 

## users can also easily use the "posterior" R package to compute their own statistics. 
## For example:
# let's say we want to compute something not included in the default
# "$summary()" method of BayesMVP, such as tail-ESS.
# The code can use the posterior R package to compute this:
require(posterior)  
## first extract the trace array object (note: already in a posterior-compatible format!)
posterior_draws <- model_fit$get_posterior_draws()
# then compute tail-ESS using posterior::ess_tail:
posterior::ess_tail(posterior_draws[,,"beta[1,1]"])

## The traces can also be returned as tibbles (stored in seperate tibbles for main params, 
## transformed params, and generates quantities) using the "$get_posterior_draws_as_tibbles()" method:
tibble_traces  <- model_fit$get_posterior_draws_as_tibbles()
tibble_trace_main <- tibble_traces$trace_as_tibble_main_params
tibble_trace_transformed_params <- tibble_traces$trace_as_tibble_transformed_params
tibble_trace_generated_quantities <- tibble_traces$trace_as_tibble_generated_quantities

## Model run time / efficiency information can also be extracted using the "$get_efficiency_metrics()" method:
model_efficiency_metrics <- model_fit$get_efficiency_metrics()
time_burnin <- model_efficiency_metrics$time_burnin  ; time_burnin
time_sampling <- model_efficiency_metrics$time_sampling ; time_sampling
time_total_MCMC <- model_efficiency_metrics$time_total_MCMC  ; time_total_MCMC
time_total_inc_summaries <- model_efficiency_metrics$time_total_inc_summaries ; time_total_inc_summaries # note this includes time to compute R-hat, etc 

# The code can also extract some more specific efficiency info, again using the "$get_efficiency_metrics()" method:
Min_ESS_main_params <- model_efficiency_metrics$Min_ESS_main   ; Min_ESS_main_params
Min_ESS_per_sec_sampling <- model_efficiency_metrics$Min_ESS_per_sec_samp ; Min_ESS_per_sec_sampling
Min_ESS_per_sec_overall <- model_efficiency_metrics$Min_ESS_per_sec_total ; Min_ESS_per_sec_overall

Min_ESS_per_grad <- model_efficiency_metrics$Min_ESS_per_grad_sampling ; Min_ESS_per_grad
grad_evals_per_sec <- model_efficiency_metrics$grad_evals_per_sec ; grad_evals_per_sec

## extract the "time to X ESS" - these are very useful for knowing how long to
#  run the model for. 
est_time_to_100_ESS <- model_efficiency_metrics$est_time_to_100_ESS_inc_summaries ; est_time_to_100_ESS
est_time_to_1000_ESS <- model_efficiency_metrics$est_time_to_1000_ESS_inc_summaries ; est_time_to_1000_ESS
est_time_to_10000_ESS <- model_efficiency_metrics$est_time_to_10000_ESS_inc_summaries; est_time_to_10000_ESS

##  The "model_samples$time_to_ESS()" method can also estimate 
## "time to X ESS" for general X:
##  For example let's say we determined our target (min) ESS to be ~5000:
est_time_5000_ESS <- model_fit$time_to_target_ESS(target_ESS = 5000) ; est_time_5000_ESS
est_time_5000_ESS

### The log_lik trace can also be extracted (note: for Stan models this will only work )
log_lik_trace <- model_fit$get_log_lik_trace()
str(log_lik_trace) # will be NULL unless "save_log_lik_trace = TRUE" is specified in the "$summary()" method
## can then use log_lik_trace e.g. to compute LOO-IC using the loo package 






















