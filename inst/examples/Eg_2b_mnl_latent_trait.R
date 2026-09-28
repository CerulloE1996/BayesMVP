
########## -------- EXAMPLE 2(b) --------------------------------------------------------------------------------------------------------- 
## Running the BUILT-IN (i.e. manual gradients) latent_trait model (e.g., for the analysis of test accuracy data without a gold standard).
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
user_dir_outs <- BayesMVP:::set_pkg_example_path_and_wd()
## Set paths:
user_root_dir <- user_dir_outs$user_root_dir
user_BayesMVP_dir <- user_dir_outs$user_BayesMVP_dir
pkg_example_path <- user_dir_outs$pkg_example_path




####  ---- 3. Set options   ------------------------------------------------------------------------------------------------------------
options(scipen = 99999)
options(max.print = 1000000000)
options(mc.cores = parallel::detectCores())


 
  
####  ---- 4. Now run the example:   ---------------------------------------------------------------------------------------------------
require(BayesMVP)

## Function to check BayesMVP AVX support 
BayesMVP::detect_vectorization_support()


 

 

## Model_type <- "LC_MVP"
Model_type <- "latent_trait"

source(file.path(pkg_example_path, "BayesMVP_LC_MVP_prep.R"))



###  ------  select sample size (one of: 500, 1000, 2500, 5000, 12500, 25000)
N <- 500



{
   
  
{
  if (N == 500)   y <-  y_master_list_seed_123_datasets[[1]]  
  if (N == 1000)  y <-  y_master_list_seed_123_datasets[[2]]  
  if (N == 2500)  y <-  y_master_list_seed_123_datasets[[3]]  
  if (N == 5000)  y <-  y_master_list_seed_123_datasets[[4]] 
  if (N == 12500) y <-  y_master_list_seed_123_datasets[[5]] 
  if (N == 25000) y <-  y_master_list_seed_123_datasets[[6]]  
}


  ## Set important variables 
    n_tests <- ncol(y)
    n_class <- 2
    n_covariates <- 1
    n_covariates_max <- 1
    n_nuisance <- N * n_tests
    n_params_main <-  choose(n_tests, 2) * 2 + 1 + n_tests * n_covariates * 2

  
    seed_dataset <- 123 
    
    n_covariates_per_outcome_mat <- array(n_covariates_max, dim = c(n_class, n_tests))

  # ## X is user-supplied
  # n_covariates_per_outcome_mat <- array(1, dim = c(n_class, n_tests))
  # X_per_class <- array(1, dim = c(n_tests, 1, N))
  # X_list <- list()
  # 
  # for (c in 1:n_class) {
  #   X_list[[c]] <- list()
  #   for (t in 1:n_tests) {
  #     X_list[[c]][[t]] <- array(999999, dim = c(N, n_covariates_per_outcome_mat[c, t] ))
  #     for (k in 1:n_covariates_per_outcome_mat[c, t] ) {
  #       for (n in 1:N) {
  #         X_list[[c]][[t]][n,  k] <- X_per_class[t, k, n]
  #       }
  #     }
  #   }
  # }
  # 
  # 
  # X <- X_list
  # # check X right format 
  # str(X)
  
  prior_a_mean  <-  vector("list", length = n_class)
  prior_a_sd <-  vector("list", length = n_class)

  prior_a_mean_mat  <-   array(0,  dim = c(n_covariates_max, n_tests))
  prior_a_sd_mat  <-    array(1,  dim = c(n_covariates_max, n_tests))

  for (c in 1:n_class) {
    prior_a_mean[[c]] <- prior_a_mean_mat
    prior_a_sd[[c]] <- prior_a_sd_mat
  }

  # intercepts / coeffs prior means
  prior_a_mean[[1]][1, 1] <- -2.10
  prior_a_sd[[1]][1, 1] <- 0.45

  prior_a_mean[[2]][1, 1] <- +0.40
  prior_a_sd[[2]][1, 1] <-  0.375
  
  k_choose_2 <- choose(n_tests, 2)
  km1_choose_2 = 0.5 * (n_tests - 2) * (n_tests - 1)
  known_num = 0
  
  beta_vec_init <- rep(0, n_class * n_tests)
  beta_vec_init[1:n_tests] <- - 1   # tests 1-T, class 1 (D-)
  beta_vec_init[(n_tests + 1):(2*n_tests)] <- + 1   # tests 1-T, class 1 (D+)
  
  ## Set inits
  n_obs <- N * n_tests
  prev_est <- sum(y) / (n_obs)
  p_raw <- atanh(2*prev_est - 1) 
  
  u_raw <- array(0.01, dim = c(N, n_tests))
    
          LT_a_vec_init <- rep(0, n_class * n_tests)
          LT_a_vec_init[1:n_tests] <- - 1   # tests 1-T, class 1 (D-)
          LT_a_vec_init[(n_tests + 1):(2*n_tests)] <- + 1   # tests 1-T, class 1 (D+)
          
          init = list(
            u_raw = (u_raw),
            p_raw =  (-0.6931472), # equiv to 0.20 on p
            LT_a_vec = LT_a_vec_init,
            LT_b_raw_vec = rep(-3.00, n_class * n_tests) 
          )
 
  ## Set n_params_main
  n_params_main <- (n_class - 1) + sum(unlist(n_covariates_per_outcome_mat)) + n_class * n_tests 
  
  
}

  



## -----------  initialise model / inits etc
# based on (informal) testing, more than 8 burnin chains seems unnecessary 
# and probably not worth the extra overhead (even on a 96-core AMD EPYC Genoa CPU)
  n_chains_burnin <- 8
  init_lists_per_chain <- rep(list(init), n_chains_burnin) 
  
   
  
  ## make model_args_list (note: Stan models don't need this)
    
    # prior_a_mean[[1]][,] <- 0
    # prior_a_mean[[2]][,] <- 0
    # prior_a_sd[[2]][,] <- 1
    # prior_a_sd[[2]][,] <- 1
    
    model_args_list <- list(         n_covariates_per_outcome_mat = n_covariates_per_outcome_mat,  
                                     num_chunks =   BayesMVP:::find_num_chunks_MVP(N, n_tests),
                                     LT_b_priors_shape = LT_b_priors_shape,
                                     LT_b_priors_scale = LT_b_priors_scale,
                                     prior_coeffs_mean_mat = prior_a_mean,
                                     prior_coeffs_sd_mat =    prior_a_sd, 
                                     prev_prior_a = 1,  # show how to change this later
                                     prev_prior_b = 1   # show how to change this later
                            ) 

  
 
  
 #   source(file.path(pkg_dir, "examples/load_R_packages.R"))
  
  #  setup_env()
  
  ###  -----------  Compile + initialise the model using "MVP_model$new(...)" 
  model_obj <- BayesMVP::MVP_model$new(   Model_type = Model_type,
                                          y = y,
                                          N = N,
                                          model_args_list = model_args_list, # this arg is only needed for BUILT-IN (not Stan) models
                                          init_lists_per_chain = init_lists_per_chain,
                                          sample_nuisance = TRUE,
                                          n_chains_burnin = n_chains_burnin,
                                          n_params_main = n_params_main,
                                          n_nuisance = n_nuisance)
  
  
  
  
  ## ----------- Set soe basic sampler settings
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
  ##  for example, for example, to to change the prior for disease prevalence to be informative s.t. prev ~ beta(5, 10). 
  ##  I could do this by modifying model_args_list:
  model_args_list$prev_prior_a <-  5
  model_args_list$prev_prior_b <-  10
  
 
  
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
                                      ## Optional arguments:
                                      # y = y,
                                      # N = N,
                                      # n_params_main = n_params_main,
                                      # n_nuisance = n_nuisance,
                                      # init_lists_per_chain = init_lists_per_chain,
                                      # n_chains_burnin = n_chains_burnin,
                                      model_args_list = model_args_list,
                                      ## Optional SAMPLER / MCMC arguments:
                                      # sample_nuisance = TRUE,
                                      # force_autodiff = FALSE,
                                      # force_PartialLog = FALSE,
                                      # multi_attempts = TRUE,
                                      adapt_delta = 0.80,
                                      learning_rate = 0.05,
                                      # metric_shape_main = "dense",
                                      # metric_type_main = "Hessian",
                                      # tau_mult = 2.0,
                                      # clip_iter = 25,
                                      # interval_width_main = 50,
                                      # ratio_M_us = 0.25,
                                      # ratio_M_main = 0.25,
                                      # parallel_method = "RcppParallel",
                                      # vect_type = "Stan",
                                      # vect_type = "AVX512",
                                      # vect_type = "AVX2",
                                      n_nuisance_to_track = n_nuisance_to_track)   


 
  
  #### --- MODEL RESULTS SUMMARY + DIAGNOSTICS -------------------------------------------------------------
  # after fitting, call the "summary()" method to compute + extract e.g. model summaries + traces + plotting methods 
  # model_fit <- model_samples$summary() # to call "summary()" w/ default options 
  require(bridgestan)
  model_fit <- model_samples$summary(save_log_lik_trace = FALSE, 
                                     compute_nested_rhat = FALSE
                                    # compute_transformed_parameters = FALSE
                                     ) 
  

   
  # extract # divergences + % of sampling iterations which have divergences
  model_fit$get_divergences()
  

  ###### --- TRACE PLOTS  ----------------------------------------------------------------------------------
  # trace_plots_all <- model_samples$plot_traces() # if want the trace for all parameters 
  trace_plots <- model_fit$plot_traces(params = c("beta", "Omega", "p"), 
                                       batch_size = 12)
  
  # parameters can be extracted using: "trace$param_name()". 
  # For example:
  # display each panel for beta and Omega ("batch_size" controls the # of plots per panel. Default = 9)
  trace_plots$beta[[1]] # 1st (and only) panel
  
  
  # display each panel for Omega ("batch_size" controls the # of plots per panel.  Default = 9)
  trace_plots$Omega[[1]] # 1st panel
  trace_plots$Omega[[2]] # 2nd panel
  trace_plots$Omega[[3]] # 3rd panel
  trace_plots$Omega[[4]] # 4th panel
  trace_plots$Omega[[5]] # 5th (and last) panel
  
 
  
  ###### --- POSTERIOR DENSITY PLOTS -------------------------------------------------------------------------
  # density_plots_all <- model_samples$plot_densities() # if want the densities for all parameters 
  # Let's plot the densities for: sensitivity, specificity, and prevalence 
  density_plots <- model_fit$plot_densities(params = c("Se_bin", "Sp_bin", "p"), 
                                            batch_size = 12)
  
  
  # parameters can be extracted using: "trace$param_name()". 
  # For example:
  # display each panel for beta and Omega ("batch_size" controls the # of plots per panel. Default = 9)
  density_plots$Se[[1]] # Se - 1st (and only) panel
  density_plots$Sp[[1]] # Sp - 1st (and only) panel
  density_plots$p[[1]] # p (prevelance) - 1st (and only) panel
  
  
  
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
  posterior::ess_tail(posterior_draws[,,"Se_bin[1]"])
  
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
  
  
  
  
  
  
  
  














