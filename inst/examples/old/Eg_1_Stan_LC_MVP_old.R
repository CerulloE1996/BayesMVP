


########## -------- EXAMPLE 1 --------------------------------------------------------------------------------------------------------- 
## Running the Stan (i.e. autodiff gradients) LC-MVP model (e.g., for the analysis of test accuracy data without a gold standard).
## Uses simulated data.
## Uses .stan model file (comes bundles with the BayesMVP R package).




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
BayesMVP:::detect_vectorization_support()



     
   
## first sprecify model type (for Stan models see other Stan example file)
Model_type <- "Stan"

 
source(file.path(pkg_example_path, "BayesMVP_LC_MVP_prep.R"))
 

# 
# - | ----------  prepare Stan data and inits --------------------------------------------------------------------



## select N to use 
N <- 500
    
    
    {
      
      if (N == 500)     dataset_index = 1
      if (N == 1000)    dataset_index = 2
      if (N == 2500)    dataset_index = 3
      if (N == 5000)    dataset_index = 4
      if (N == 12500)   dataset_index = 5
      if (N == 25000)   dataset_index = 6
      
      y = y_master_list_seed_123_datasets [[dataset_index]]
      
      
      ## Set important variables
      n_tests <- ncol(y)
      n_class <- 2
      n_covariates <- 1
      n_nuisance <- N * n_tests
      n_params_main <-  choose(n_tests, 2) * 2 + 1 + n_tests * n_covariates * 2
      
      
      
      N_sims <- 123
      
    ###  model_type <- "LC_MVP"
      fully_vectorised <- 1  ;   GHK_comp_method <- 2 ;       handle_numerical_issues <- 1 ;     overflow_threshold <- +5 ;    underflow_threshold <- -5  #  MAIN MODEL SETTINGS 
      
       Phi_type <- 2 # using Phi_approx and inv_Phi_approx - in  Stan these are slower than Phi() and inv_Phi() !!!!!!!! (as log/exp are slow)
      #  Phi_type <- 1 # using Phi and inv_Phi
      
      corr_param <- "Sean" ; prior_lkj <- c(12, 3) ;  corr_prior_beta =  0 ;  corr_force_positive <-  0  ;  prior_param_3 <- 0 ; uniform_indicator <- 0 
      #  corr_param <- "Sean" ; prior_lkj <- c(10, 2) ;  corr_prior_beta =  0 ;  corr_force_positive <-  0  ;  prior_param_3 <- 0 ; uniform_indicator <- 0 
      
      CI <- 0
      prior_only <-  0
      corr_prior_norm = 0
      gamma_indicator <-  0
      skew_norm_indicator <- 0
      
      prior_lkj_skewed_diseased <-   prior_lkj
      prior_lkj_skewed_non_diseased <-  prior_lkj
      
      #  tailored_corr_priors <- TRUE # priors which are more consistent with posteior (should recover parameters better, especially for lower N and in the smaller latent class)
      tailored_corr_priors <- FALSE # priors which are more consistent with posteior (should recover parameters better, especially for lower N and in the smaller latent class)
      
      n_class <- 2
      n_covariates <- 1
      n_tests <-   n_tests 
      n_ordinal_tests <- 0
      n_binary_tests <- n_tests
      Thr =   rep(1, n_tests)
      #N <- nrow(y)
      
      prior_a_mean <-   array(0,  dim = c(n_class, n_tests, n_covariates))
      prior_a_sd  <-    array(1,  dim = c(n_class, n_tests, n_covariates))
      
      
      
      # intercepts / coeffs prior means
      prior_a_mean[1,1,1] <- -2.10
      prior_a_sd[1,1,1] <- 0.45
      
      prior_a_mean[2,1,1] <- +0.40
      prior_a_sd[2,1,1] <-  0.375
      
      n_pops <- 1
      group <- rep(1, N)
    
      
      n_covs_per_outcome = array(1, dim = c(n_class, n_tests))
      n_covariates_total_nd  =    (sum( (n_covs_per_outcome[1,])));
      n_covariates_total_d   =     (sum( (n_covs_per_outcome[2,])));
      n_covariates_total  =       n_covariates_total_nd + n_covariates_total_d;
      
      k_choose_2   = (n_tests * (n_tests - 1)) / 2;
      km1_choose_2 =  ((n_tests - 1) * (n_tests - 2)) / 2;
      
      i = 123
      
      
      n_covariates_max <- 1
      n_covariates_max_nd <- 1
      n_covariates_max_d <- 1
      
      X_nd <- list()
      for(t in 1:n_tests) {
        X_nd[[t]] <- matrix(1, nrow=N, ncol=n_covariates_max_nd)
      }
      
      X_d <- list()
      for(t in 1:n_tests) {
        X_d[[t]] <- matrix(1, nrow=N, ncol=n_covariates_max_nd)
      }
      
      
      {
        
        
        # if (prior_only == 0 ) { 
        stan_data = list(  N =   (N), # length(corr_prior_dist_LKJ_4), #  N, 
                           n_tests = n_tests,
                           y = (y_master_list_seed_123_datasets[[dataset_index]]),
                           n_class = 2,
                           n_pops = 1, #  n_pops,
                           pop = group,
                           n_covariates_max_nd = n_covariates_max_nd,
                           n_covariates_max_d = n_covariates_max_d,
                           n_covariates_max = n_covariates_max,
                           X_nd = X_nd,
                           X_d =  X_d,
                           n_covs_per_outcome = n_covs_per_outcome, 
                           corr_force_positive = corr_force_positive,
                           known_num = 0,
                           overflow_threshold = overflow_threshold,
                           underflow_threshold = underflow_threshold,
                           prior_only = prior_only, ######
                           prior_beta_mean =   prior_a_mean ,
                           prior_beta_sd  =   prior_a_sd ,
                           prior_LKJ = prior_lkj, 
                           prior_p_alpha = array(rep(5, n_pops)),
                           prior_p_beta =  array(rep(10, n_pops)),
                           Phi_type = Phi_type,
                           handle_numerical_issues = handle_numerical_issues,
                           fully_vectorised = fully_vectorised,
                           corr_prior_beta = corr_prior_beta,
                           corr_prior_norm = corr_prior_norm,
                           k_choose_2 = k_choose_2,
                           km1_choose_2 = km1_choose_2,
                           GHK_comp_method = GHK_comp_method)
        
        # stan_data_list[[i]] <- stan_data
        
      }
      
      
      ## set Stan model file path for your Stan model (replace with your path)
      ##  Stan_model_file_path <- normalizePath(file.path("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/BayesMVP\\stan_models\\LC_MVP_bin_PartialLog_v5.stan"))
     ## Stan_model_file_path <- normalizePath((file.path("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP/inst/stan_models/LC_MVP_bin_PartialLog_v5.stan")))
      
      ## set Stan model file path for your Stan model (replace with your path)
      Stan_model_file_path <- system.file("stan_models/LC_MVP_bin_PartialLog_v5.stan", package = "BayesMVP")
      # -------------------------------------------------------------------------
      
      u_raw <- array(0.01, dim = c(N, n_tests))
      
      km1_choose_2 = 0.5 * (n_tests - 2) * (n_tests - 1)
      known_num = 0
      
      beta_vec_init <- rep(0.01, n_class * n_tests)
      beta_vec_init[1:n_tests] <- - 1   # tests 1-T, class 1 (D-)
      beta_vec_init[(n_tests + 1):(2*n_tests)] <- + 1   # tests 1-T, class 1 (D+)
      
      off_raw <- list()
      col_one_raw <- list()
      
      for (i in 1:2) {
        off_raw[[i]] <-  (c(rep(0.01, km1_choose_2 - known_num)))
        col_one_raw[[i]] <-  (c(rep(0.01, n_tests - 1)))
      }
      
      init = list(
        u_raw = (u_raw),
        p_raw =  (-0.6931472), # equiv to 0.20 on p
        beta_vec = beta_vec_init,
        off_raw = off_raw,
        col_one_raw =  col_one_raw
        #L_Omega_raw = array(0.01, dim = c(n_class, choose(n_tests, 2)))
      )
      
      Stan_init_list <- init
      
      
      stan_data$prior_beta_mean
      stan_data$prior_beta_sd
      stan_data$prior_LKJ
      stan_data$Phi_type
 
      
      
    }
    
  
    
    ## make lists of lists for inits 
    n_chains_burnin <- 8 
    init_lists_per_chain <- rep(list(Stan_init_list), n_chains_burnin) 

    
    ###  -----------  Compile + initialise the model using "MVP_model$new(...)" 
    model_obj <- BayesMVP::MVP_model$new(   Model_type =  "Stan",
                                            y = y,
                                            N = N,
                                            ##  X = NULL,
                                            ##  model_args_list = model_args_list, # this arg is only needed for BUILT-IN (not Stan) models
                                            Stan_data_list = stan_data,
                                            Stan_model_file_path = Stan_model_file_path,
                                            init_lists_per_chain = init_lists_per_chain,
                                            sample_nuisance = TRUE,
                                            n_chains_burnin = n_chains_burnin,
                                            n_params_main = n_params_main,
                                            n_nuisance = n_nuisance)

 
    
    ## ----------- Set basic sampler settings
    {
      ### seed <- 123
      n_chains_sampling <- max(64, parallel::detectCores() / 2)
      n_superchains <- min(8, parallel::detectCores() / 2)  ## round(n_chains_sampling / n_chains_burnin) # Each superchain is a "group" or "nest" of chains. If using ~8 chains or less, set this to 1. 
      n_iter <- 1000                                 
      n_burnin <- 500
      n_nuisance_to_track <- 10 # set to some small number (< 10) if don't care about making inference on nuisance params (which is most of the time!)
    }
    
    

    
    ## sample / run model
    model_samples <-  model_obj$sample(  partitioned_HMC = TRUE,
                                         diffusion_HMC = TRUE,
                                         seed = 1,
                                         n_burnin = n_burnin,
                                         n_iter = n_iter,
                                         n_chains_sampling = n_chains_sampling,
                                         n_superchains = n_superchains,
                                         ## Some other optional arguments:
                                         Stan_data_list = stan_data,
                                         Stan_model_file_path = Stan_model_file_path,
                                         # y = y,
                                         # N = N,
                                         # n_params_main = n_params_main,
                                         # n_nuisance = n_nuisance,
                                         # init_lists_per_chain = init_lists_per_chain,
                                         # n_chains_burnin = n_chains_burnin,
                                         ## model_args_list = model_args_list, # this arg is only needed for BUILT-IN (not Stan) models
                                         adapt_delta = 0.80,
                                         learning_rate = 0.05,
                                         n_nuisance_to_track = n_nuisance_to_track)  
 
  
  
    

    
    
    #### --- MODEL RESULTS SUMMARY + DIAGNOSTICS -------------------------------------------------------------
    # after fitting, call the "summary()" method to compute + extract e.g. model summaries + traces + plotting methods 
    # model_fit <- model_samples$summary() # to call "summary()" w/ default options 
    require(bridgestan)
    model_fit <- model_samples$summary(save_log_lik_trace = FALSE, 
                                       compute_nested_rhat = FALSE
                                       # compute_transformed_parameters = FALSE
    ) 
    
    
    # x <- matrix(runif(n = 100, min = 0, max = 1))
    # BayesMVP:::Rcpp_wrapper_EIGEN_double(x = x, fn = "log", vect_type = "AVX2", skip_checks = FALSE) - 
    # BayesMVP:::Rcpp_wrapper_EIGEN_double(x = x, fn = "log", vect_type = "AVX512", skip_checks = FALSE)
    # 
    # 
    
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
    trace_plots <- model_fit$plot_traces(params = c("beta", "Omega", "p"), 
                                         batch_size = 12)
    
    # you can extract parameters by doing: "trace$param_name()". 
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
    
    
    # you can extract parameters by doing: "trace$param_name()". 
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
    
    
    
    
    
    
    
    
    
    
    
    # {
    #   
    #   set.seed(1)
    #   
    #   init_model_and_vals_object <- model_obj$init_object
    #   
    #   
    #   {
    #     # sample_obj <- model_samples$model_results
    #     #model_results = sample_obj
    #     init_object <- model_obj$init_object
    #     compute_main_params = TRUE
    #     compute_generated_quantities = TRUE
    #     compute_transformed_parameters = TRUE
    #     save_log_lik_trace = TRUE
    #     save_nuisance_trace = FALSE
    #   }
    #   
    #   options(scipen = 99999)
    #   
    #   
    #   
    #   
    #   n_us <- N*n_tests
    #   n_params <- n_us + n_params_main
    #   index_us <- 1:n_us
    #   index_main <- (1 + n_us):n_params
    #   
    #   if (Model_type == "latent_trait") {
    #     n_corrs <- n_tests * 2
    #   } else {
    #     n_corrs <- 2 * choose(n_tests, 2)
    #   }
    #   
    #   theta_vec <- rep(0.01, n_params)
    #   
    #   Model_args_as_Rcpp_List <- init_model_and_vals_object$Model_args_as_Rcpp_List
    # 
    #   
    # }
    # 
    # 
    # 
    # 
    # ## Rcpp::sourceCpp("~/Documents/Work/PhD_work/R_packages/BayesMVP/src/main.cpp")
    # 
    # ##  dyn.load("C:\\Users\\enzoc\\Documents\\BayesMVP\\inst\\dummy_stan_model_win_model.dll")
    # 
    # #   tic()
    # # # for (i in 1:1000)
    # # 
    # # init_model_and_vals_object$model_so_file
    # # 
    # # Model_args_as_Rcpp_List$model_so_file <-   "C:\\Users\\enzoc\\AppData\\Local\\R\\win-library\\4.4\\BayesMVP\\stan_models\\LC_MVP_bin_PartialLog_v5_model.so"
    # # Model_args_as_Rcpp_List$json_file_path <- init_model_and_vals_object$json_file_path# "C:\\Users\\enzoc\\AppData\\Local\\R\\win-library\\4.4\\BayesMVP\\stan_models\\LC_MVP_bin_PartialLog_v5_model.dll"
    # #   
    # 
    # Model_args_as_Rcpp_List$model_so_file
    # Model_args_as_Rcpp_List$json_file_path
    # 
    # BayesMVP::detect_vectorization_support()
    # 
    # 
    # require(BayesMVP)
    # 
    # lp_grad_outs <-   BayesMVP:::safe_test_wrapper_1 ( theta_vec = theta_vec,
    #                                         index_main = index_main,
    #                                         index_us = index_us,
    #                                         y = y,
    #                                         Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
    #                                         
    #                                         expr = {
    #                                           
    #                                           require(BayesMVP)
    # 
    #                                           
    #                                           
    #                                           BayesMVP::Rcpp_wrapper_fn_lp_grad( Model_type = "Stan",
    #                                                                              force_autodiff = TRUE,
    #                                                                              force_PartialLog = TRUE,
    #                                                                              multi_attempts = FALSE,
    #                                                                              theta_main_vec = theta_vec[index_main],
    #                                                                              theta_us_vec = theta_vec[index_us],
    #                                                                              y = y,
    #                                                                              grad_option = "all",
    #                                                                              Model_args_as_Rcpp_List = Model_args_as_Rcpp_List) 
    #                                           
    #                                         } )
    # 
    # 
    # 
    # lp_grad_outs
    # # lp_grad_outs$job
    # 
    # 
    # 
    
    
    
    
    
    
    
    
  
  
  