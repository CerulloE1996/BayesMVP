
########## -------- EXAMPLE 4 --------------------------------------------------------------------------------------------------------- 
## Running the Stan (i.e. autodiff gradients) BASIC EXAMPLE model (univariate logistic regression)
## Uses simulated data.
## Uses .stan model file (comes bundles with the BayesMVP R package). 
## NOTE: the purpose of this specific example is to show that BayesMVP is  capable of running * any * Stan model, even
## models WITHOUT any high-dimensional latent variables / nuisance parameters. 
## However, BayesMVP will not necessarily be faster than Stan for these models (sometimes it might even be slower - e.g. if N is small). 
## Sometimes though, it may still be much faster than Stan - this is because BayesMVP uses a different algorithm during the 
## burnin/warmup phease, which is based on the recently proposed SNAPER-HMC (Sountsov et al, 2023), as opposed to Stan which uses
## NUTS-HMC (Hoffman et al, 2014) for the adaptation. 


# 
# ####  ---- 1. Install BayesMVP (from GitHub) - SKIP THIS STEP IF INSTALLED: -----------------------------------------------------------
# ## First remove any possible package fragments:
# ## Find user_pkg_install_dir:
# user_pkg_install_dir <- Sys.getenv("R_LIBS_USER")
# print(paste("user_pkg_install_dir = ", user_pkg_install_dir))
# ##
# ## Find pkg_install_path + pkg_temp_install_path:
# pkg_install_path <- file.path(user_pkg_install_dir, "BayesMVP")
# pkg_temp_install_path <- file.path(user_pkg_install_dir, "00LOCK-BayesMVP") 
# ##
# ## Remove any (possible) BayesMVP package fragments:
# remove.packages("BayesMVP")
# unlink(pkg_install_path, recursive = TRUE, force = TRUE)
# unlink(pkg_temp_install_path, recursive = TRUE, force = TRUE)
# ##
# ## First install OUTER package:
# remotes::install_github("https://github.com/CerulloE1996/BayesMVP", force = TRUE, upgrade = "never")
# ## Then restart R session:
# rstudioapi::restartSession()
# ## Then install INNTER (i.e. the "real") package:
# require(BayesMVP)
# BayesMVP::install_BayesMVP()
# require(BayesMVP)
# 
# require(BayesMVP)
# CUSTOM_FLAGS <- list()
# install_BayesMVP(CUSTOM_FLAGS = list())
# require(BayesMVP)





####  ---- 2. Set BayesMVP example path and set working directory:  --------------------------------------------------------------------
require(RcppParallel)
require(BayesMVP)
require(cmdstanr)
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

 
Model_type <- "Stan"  # specify Model_type as "Stan" if using a Stan model

# source(file.path(pkg_example_path, "load_R_packages.R"))

 
 
 
# - | ----------  prepare Stan data and inits --------------------------------------------------------------------

{
      ### input N to use 
      N <- 10000

      ### generate data
      n_covs <- 10
      X <- array(NA, dim = c(N, n_covs))

      set.seed(1)
      for (i in 1:n_covs) {
         #  X[1:N, i]  <-  rnorm(n = N, mean = 1, sd = 1)  # cts covariates
           X[1:N, i]  <-  rlogis(n = N)  # cts covariates
      }
      
      true_intercept <- 0.0
      true_beta_vals <- c(-1.0, -0.50, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5) # , rep(1, 20))
      length(true_beta_vals)
      true_Xbeta_vals <- X %*%  t(t(true_beta_vals))
      
       # latent_results <- rnorm(n = N, mean =  true_intercept + true_Xbeta_vals, sd = 1) # for probit regression
       latent_results <- rlogis(n = N, location =  true_intercept + true_Xbeta_vals) # for logistic regression
      
      ## simulated dataset
      y <- ifelse(latent_results > 0, 1, 0)
  
      ### Stan model data list 
      stan_data = list(  N =  N, 
                         y = c(y),
                         n_covs = n_covs, 
                         X = X)
      
      ### Stan model file (put path to your Stan model)
      file <- (file.path(user_BayesMVP_dir, "stan_models/basic_logistic.stan"))
      mod <- cmdstan_model(file)
      
      # # OR using the version w/ fast math C++ functions:
      # file <- (file.path(pkg_dir, "inst/stan_models/basic_logistic_w_fast_fns.stan"))
      # ## and then input the path to the corresponding C++ files:
      # path_to_cpp_user_header <- file.path(pkg_dir, "src/approx_cpp_fns_for_Stan.hpp")
      
}
 
      
      
      mod <- cmdstan_model(file, 
                          # force_recompile = TRUE,
                          # user_header = path_to_cpp_user_header,
                           cpp_options = list(
                                               "CXXFLAGS =    -O3  -march=native  -mtune=native -fPIC -D_REENTRANT  -mfma  -mavx512f -mavx512vl -mavx512dq" ,
                                               "CPPFLAGS =    -O3  -march=native  -mtune=native -fPIC -D_REENTRANT  -mfma  -mavx512f -mavx512vl -mavx512dq"
                                               ))



 
{
        
      #### Stan initial values list 
      alpha <- 0.001
      ##  beta <- rep(0.001, n_covs)
      beta <- true_beta_vals
      
      init = list(
        alpha = alpha, 
        beta = beta,
        u_raw = rep(0.001, N)
      )
      
      Stan_init_list <- init
      
      # # make lists of lists for inits 
      # n_chains <- max(64, parallel::detectCores() / 2)
      ##
      n_chains <- 8
      ##
      init_lists_per_chain <- rep(list(Stan_init_list), n_chains) 
      
      n_burnin <- 500
      n_iter <- 1000
      
}
      
      
      
### run model using Stan directly (cmdstanr) first 
{
            tictoc::tic()
        
            Stan_model_out <- mod$sample(data = stan_data, 
                                         seed = 3,
                                         refresh = 50,
                                         init = init_lists_per_chain,
                                         chains = n_chains,
                                         parallel_chains = n_chains,
                                         max_treedepth = 10,
                                         iter_warmup = n_burnin,
                                         iter_sampling = n_iter)
          
            
            try({
              {
                print(tictoc::toc(log = TRUE))
                log.txt <- tictoc::tic.log(format = TRUE)
                tictoc::tic.clearlog()
                total_time <- unlist(log.txt)
              }
            })
            
            try({
              total_time <- as.numeric( substr(start = 0, stop = 100,  strsplit(  strsplit(total_time, "[:]")[[1]], "[s]")  [[1]][1] ) )
            })
        
}
      
      
      (Stan_model_out$time())
      
     ###  Stan_model_out$unconstrain_variables()
      
      
      ### results summary
      Stan_model_results <- Stan_model_out$summary()
      print(Stan_model_results, n = 100)
      
      
      ### some basic efficiency stats 
{
  
        Stan_sampler_diagnostics <- as.array(Stan_model_out$sampler_diagnostics())
        Stan_treedepth_array <- Stan_sampler_diagnostics[,,1]
        Stan_avg_treedepth_during_sampling <- sum(Stan_treedepth_array) / length(Stan_treedepth_array)
        Stan_avg_L_during_sampling <- 2^Stan_avg_treedepth_during_sampling
        Stan_n_grad_evals_sampling <- n_chains * n_iter * Stan_avg_L_during_sampling
        
        Stan_times_per_chain <- Stan_model_out$time()
        time_sampling <- max(Stan_times_per_chain$chains$sampling)
        
        Stan_min_ESS <- min(round(Stan_model_results$ess_bulk[(0 + 1):(0 + n_covs - 1)]))
   
        Stan_min_ESS_per_sec <-  Stan_min_ESS / total_time ; Stan_min_ESS_per_sec
        Stan_min_ESS_per_sec_sampling <-  Stan_min_ESS / time_sampling ; Stan_min_ESS_per_sec_sampling
        Stan_min_ESS_per_grad_sampling <-   ( Stan_min_ESS / Stan_n_grad_evals_sampling )
        Stan_min_ESS_per_grad_sampling_x_1000 <- 1000 * Stan_min_ESS_per_grad_sampling
        Stan_grad_evals_per_sec <- Stan_min_ESS_per_sec_sampling /  Stan_min_ESS_per_grad_sampling 
        Stan_grad_evals_per_sec_div_1000 <- Stan_grad_evals_per_sec / 1000
        
        
        print(paste("Stan_min_ESS = ", round(Stan_min_ESS_per_sec, 0)))
        print(paste("Stan_min_ESS_per_sec = ", round(Stan_min_ESS_per_sec, 2)))
        print(paste("Stan_min_ESS_per_sec_sampling = ", round(Stan_min_ESS_per_sec_sampling, 2)))
        print(paste("Stan_min_ESS_per_grad_sampling_x_1000 = ", round(Stan_min_ESS_per_grad_sampling_x_1000, 2)))
        print(paste("Stan_grad_evals_per_sec_div_1000 = ", round(Stan_grad_evals_per_sec_div_1000, 2)))
        
}
 
      # 
      # [1] "Stan_min_ESS =  888"
      # [1] "Stan_min_ESS_per_sec =  887.67"
      # [1] "Stan_min_ESS_per_sec_sampling =  2026.34"
      # [1] "Stan_min_ESS_per_grad_sampling_x_1000 =  24.45"
      # [1] "Stan_grad_evals_per_sec_div_1000 =  82.88"
      

    
    sample_nuisance <- FALSE
    
    
    ## make lists of lists for inits 
    n_chains_burnin <- 8 
    init_lists_per_chain <- rep(list(Stan_init_list), n_chains_burnin) 
   
    ## Define # of (raw) model parameters
    n_params_main <- n_covs + 1
    
    n_nuisance <- 9 # dummry variable (just set to something small e.g. between 5-10 - this example model doesn't have any nuisance parameters)
    
      
    ## set Stan model file path for your Stan model (replace with your path)
    Stan_model_file_path <- system.file("stan_models/basic_logistic.stan", package = "BayesMVP")
    
    ## make y a matrix first (matrix w/ 1 col)
    y <- matrix(data = c(y), ncol = 1)
    
    ###  -----------  Compile + initialise the model using "MVP_model$new(...)" 
    model_obj <- BayesMVP::MVP_model$new(   Model_type =  "Stan",
                                            y = y,
                                            N = N,
                                            ##  model_args_list = model_args_list, # this arg is only needed for BUILT-IN (not Stan) models
                                            Stan_data_list = stan_data,
                                            Stan_model_file_path = Stan_model_file_path,
                                            init_lists_per_chain = init_lists_per_chain,
                                            sample_nuisance = sample_nuisance,
                                            n_chains_burnin = n_chains_burnin,
                                            n_params_main = n_params_main,
                                            n_nuisance = n_nuisance)
    
    ## ----------- Set basic sampler settings
    {
      seed <- 1
      n_chains_sampling <- max(64, parallel::detectCores() / 2)
      n_superchains <- min(8, parallel::detectCores() / 2)  ## round(n_chains_sampling / n_chains_burnin) # Each superchain is a "group" or "nest" of chains. If using ~8 chains or less, set this to 1. 
      n_iter <- 1000                                 
      n_burnin <- 500
      ## n_nuisance_to_track <- n_nuisance # set to some small number (< 10) if don't care about making inference on nuisance params (which is most of the time!)
    }
    
    
    ## Since this model (i.e., basic logistic reg.) does NOT have any nuisance parameters, we will set:
    ## partitioned_HMC = FALSE - this is because partitioned HMC is only available if sample_nuisance == TRUE
    ## (since we sample the nuisance params and main params seperately)
    partitioned_HMC <- FALSE ;    diffusion_HMC <- FALSE
    
    
    recip <- function(x) { 
      
      if (is.vector(x)) {
        
        recip_x <- 1.0 / x
        return(recip_x)
        
      } else if (is.matrix(x)) { 
        
        diag_of_x <- diag(x)
        recip_diag_x <- 1.0 / diag_of_x
        recip_mat_x <- diag(recip_diag_x)
        return(recip_mat_x)
        
      } else {
        
        stop("ERROR: x must be either a vector or a matrix")
        
      }
      
    }
    
    ##
    model_samples <-  model_obj$sample(  partitioned_HMC = partitioned_HMC,
                                         diffusion_HMC = diffusion_HMC,
                                         seed = seed,
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
                                         # model_args_list = model_args_list,
                                         ## Some other SAMPLER / MCMC arguments:
                                         # sample_nuisance = TRUE,
                                         adapt_delta = 0.80,
                                         learning_rate = 0.05,
                                         ## metric_shape_main = "diag",
                                         ## metric_type_main = "Hessian",
                                         ## tau_mult = 2.0,
                                         clip_iter = 25,
                                         ## interval_width_main = 50,
                                         ## ratio_M_us = 0.25,
                                         ##  ratio_M_main = 0.25,
                                         ## parallel_method = "RcppParallel"
                                         )   
    
    compile <- TRUE
    Stan_cpp_user_header <- NULL
    Stan_cpp_flags = NULL
    ##
    diffusion_HMC = diffusion_HMC
    seed = seed
    n_burnin = 500
    n_iter = n_iter
    n_chains_sampling = n_chains_sampling
    n_superchains = n_superchains
    ## Some other arguments:
    y = y
    N = N
    n_params_main = n_params_main
    n_nuisance = n_nuisance
    init_lists_per_chain = init_lists_per_chain
    n_chains_burnin = n_chains_burnin
    model_args_list = NULL
    # Some other SAMPLER / MCMC arguments:
    sample_nuisance = FALSE
    adapt_delta = 0.80
    learning_rate = 0.05
    metric_shape_main = "diag"
    metric_type_main = "Hessian"
    tau_mult = 2.0
    clip_iter = 25
    interval_width_main = 50
    ratio_M_us = 0.25
    ratio_M_main = 0.25
    parallel_method = "RcppParallel"
    ##
    LR_main <- 0.05
    LR_us <- 0.05
    manual_tau <- FALSE
    ##
    n_chains_burnin <- n_chains
    ##
    ## Helper function:
    if_null_then_set_to <- function(x, set_to_this_if_null) { 
      if (is.null(x)) {
        return(set_to_this_if_null)
      }
      return(x)
    }
    ##
    gap <- n_adapt <- NULL
    n_adapt <- if_null_then_set_to(n_adapt, n_burnin - round(n_burnin/10))
    gap <- if_null_then_set_to(gap, clip_iter  + round(n_adapt / 5))
    interval_width_nuisance <- interval_width_main <- NULL
    interval_width_main <- if_null_then_set_to(interval_width_main, round(n_burnin/10))
    interval_width_nuisance <- if_null_then_set_to(interval_width_nuisance, round(n_burnin/10))
    metric_shape_nuisance <- "diag"
    metric_type_nuisance = "Empirical"
    ##
    force_autodiff = FALSE
    force_PartialLog = FALSE
    multi_attempts = TRUE
    ##
    max_eps_main <- max_eps_us <- 100
    max_L = 1024
    ##
    # init_object <- model_obj$init_object
    Stan_data_list = stan_data
    
    
    
    n_chains_sampling = n_chains_sampling
    n_superchains = n_superchains
    n_params_main = n_params_main
    n_nuisance = n_nuisance
    theta_main_vectors_all_chains_input_from_R
    theta_us_vectors_all_chains_input_from_R



    
    #### --- MODEL RESULTS SUMMARY + DIAGNOSTICS -------------------------------------------------------------
    # after fitting, call the "summary()" method to compute + extract e.g. model summaries + traces + plotting methods 
    # model_fit <- model_samples$summary() # to call "summary()" w/ default options 
    require(bridgestan)
    model_fit <- model_samples$summary(save_log_lik_trace = FALSE, 
                                       compute_nested_rhat = FALSE,
                                       compute_transformed_parameters = FALSE,
                                       compute_generated_quantities = FALSE
    ) 
    
    
    
    #     "Stan_min_ESS_per_sec =  846.8"
    # [1] "Stan_min_ESS_per_sec_sampling =  1855.57"
    # [1] "Stan_min_ESS_per_grad_sampling_x_1000 =  23.52"
    # [1] "Stan_grad_evals_per_sec_div_1000 =  78.89"
    
    
  
    # extract # divergences + % of sampling iterations which have divergences
    model_fit$get_divergences()
    
    ###### --- TRACE PLOTS  ----------------------------------------------------------------------------------
    # trace_plots_all <- model_samples$plot_traces() # if want the trace for all parameters 
    trace_plots <- model_fit$plot_traces(params = c("alpha", "beta"), 
                                         batch_size = 8)
    
    ## Display trace plot for alpha:
    trace_plots$alpha[[1]] # 1st (and only) panel
    ## Display trace plots for beta:
    trace_plots$beta[[1]] # 1st panel
    trace_plots$beta[[2]] # 1st panel
    
    ###### --- POSTERIOR DENSITY PLOTS -------------------------------------------------------------------------
    # density_plots_all <- model_samples$plot_densities() # if want the densities for all parameters 
    # Let's plot the densities for: sensitivity, specificity, and prevalence 
    density_plots <- model_fit$plot_densities(params = c("alpha", "beta"), 
                                              batch_size = 8)
    
    ## Display density plot for alpha:
    density_plots$alpha[[1]] # 1st (and only) panel
    ## Display density plots for beta:
    density_plots$beta[[1]] # 1st panel
    density_plots$beta[[2]] # 1st panel
    
    
    
    
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
    # We can just use the posterior R package to compute this:
    require(posterior)  
    ## first extract the trace array object (note: already in a posterior-compatible format!)
    posterior_draws <- model_fit$get_posterior_draws()
    ## then compute tail-ESS using posterior::ess_tail:
    posterior::ess_tail(posterior_draws[,,"Se_bin[1]"])
    
    ## You can also get the traces as tibbles (stored in seperate tibbles for main params, 
    ## transformed params, and generates quantities) using the "$get_posterior_draws_as_tibbles()" method:
    tibble_traces  <- model_fit$get_posterior_draws_as_tibbles()
    tibble_trace_main <- tibble_traces$trace_as_tibble_main_params
    tibble_trace_transformed_params <- tibble_traces$trace_as_tibble_transformed_params
    tibble_trace_generated_quantities <- tibble_traces$trace_as_tibble_generated_quantities
    
    ## You can also easily extract model run time / efficiency information using the "$get_efficiency_metrics()" method:
    model_efficiency_metrics <- model_fit$get_efficiency_metrics()
    time_burnin <- model_efficiency_metrics$time_burnin  ; time_burnin
    time_sampling <- model_efficiency_metrics$time_sampling ; time_sampling
    time_total_MCMC <- model_efficiency_metrics$time_total_MCMC  ; time_total_MCMC
    time_total_inc_summaries <- model_efficiency_metrics$time_total_inc_summaries ; time_total_inc_summaries # note this includes time to compute R-hat, etc 
    
    # We can also extract some more specific efficiency info, again using the "$get_efficiency_metrics()" method:
    Min_ESS_main_params <- model_efficiency_metrics$Min_ESS_main   ; Min_ESS_main_params
    Min_ESS_per_sec_sampling <- model_efficiency_metrics$Min_ESS_per_sec_samp ; Min_ESS_per_sec_sampling
    Min_ESS_per_sec_overall <- model_efficiency_metrics$Min_ESS_per_sec_total ; Min_ESS_per_sec_overall
    
    Min_ESS_per_grad <- model_efficiency_metrics$Min_ESS_per_grad_sampling ; Min_ESS_per_grad
    grad_evals_per_sec <- model_efficiency_metrics$grad_evals_per_sec ; grad_evals_per_sec
    
    ## extract the "time to X ESS" - these are very useful for knowing how long to
    #  run your model for. 
    est_time_to_100_ESS <- model_efficiency_metrics$est_time_to_100_ESS_inc_summaries ; est_time_to_100_ESS
    est_time_to_1000_ESS <- model_efficiency_metrics$est_time_to_1000_ESS_inc_summaries ; est_time_to_1000_ESS
    est_time_to_10000_ESS <- model_efficiency_metrics$est_time_to_10000_ESS_inc_summaries; est_time_to_10000_ESS
    
    ##  You can also use the "model_samples$time_to_ESS()" method to estimate 
    ## "time to X ESS" for general X:
    ##  For example let's say we determined our target (min) ESS to be ~5000:
    est_time_5000_ESS <- model_fit$time_to_target_ESS(target_ESS = 5000) ; est_time_5000_ESS
    est_time_5000_ESS
    
    ### You can also extract the log_lik trace (note: for Stan models this will only work if "log_lik" transformed parameter is defined in Stan model)
    log_lik_trace <- model_fit$get_log_lik_trace()
    str(log_lik_trace) # will be NULL unless you specify  "save_log_lik_trace = TRUE" in the "$summary()" method
    ## can then use log_lik_trace e.g. to compute LOO-IC using the loo package 
    
    
    
    
    
    
    
    
    
    
    
    
    
    

  
  