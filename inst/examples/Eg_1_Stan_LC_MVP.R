


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
    options(warning.length = 8170)
}




####  ---- 4. Now run the example:   ---------------------------------------------------------------------------------------------------
source(file.path(getwd(), "load_R_packages.R"))
require(BayesMVP)

## Function to check BayesMVP AVX support 
BayesMVP:::detect_vectorization_support()

## Simulate data (for N = 500)
{
  source(file.path(getwd(), "R_fn_load_data_binary_LC_MVP_sim.R"))
  N <- 500
  ## Call the fn to simulate binary data:
  data_sim_outs <- simulate_binary_LC_MVP_data(N_vec = N, seed = 123, DGP = 5)
  ## Extract dataset (y):
  y <- data_sim_outs$y_binary_list[[1]]
}

     
   
## first sprecify model type (for Stan models see other Stan example file)
Model_type <- "Stan"



 
    
    {
      
      ## Set important variables
      n_tests <- ncol(y)
      n_class <- 2
      n_covariates <- 1
      n_nuisance <- N * n_tests
      n_corrs <- n_class * choose(n_tests, 2)
      ## Intercept-only:
      n_covariates_max <- 1
      n_covariates_max_nd <- 1
      n_covariates_max_d <- 1
      n_covs_per_outcome = array(1, dim = c(n_class, n_tests))
      n_covariates_total_nd  =    (sum( (n_covs_per_outcome[1,])));
      n_covariates_total_d   =     (sum( (n_covs_per_outcome[2,])));
      n_covariates_total  =       n_covariates_total_nd + n_covariates_total_d;
      ##
      n_params_main <- n_corrs + n_covariates_total + 1
 
      fully_vectorised <- 1  ;   GHK_comp_method <- 2 ;       handle_numerical_issues <- 1 ;     overflow_threshold <- +5 ;    underflow_threshold <- -5  #  MAIN MODEL SETTINGS 
      
       # Phi_type <- 2 # using Phi_approx and inv_Phi_approx - in  Stan these are slower than Phi() and inv_Phi() !!!!!!!! (as log/exp are slow)
       Phi_type <- 1 # using Phi and inv_Phi
      
      corr_param <- "Sean" ; prior_LKJ <- matrix(c(12, 3), ncol = 1) ;  corr_prior_beta =  0 ;  corr_force_positive <-  0  ;  prior_param_3 <- 0 ; uniform_indicator <- 0 
 
      prior_only <-  0
 
      {
        prior_a_mean <-   array(0,  dim = c(n_class, n_tests, n_covariates_max))
        prior_a_sd  <-    array(1,  dim = c(n_class, n_tests, n_covariates_max))
        ##
        ## intercepts / coeffs prior means
        prior_a_mean[1,1,1] <- -2.10
        prior_a_sd[1,1,1] <- 0.45
        ##
        prior_a_mean[2,1,1] <- +0.40
        prior_a_sd[2,1,1] <-  0.375
        ## As lists of mats (needed for C++ manual-grad models:
        prior_a_mean_as_list <- prior_a_sd_as_list <- list()
        for (c in 1:n_class) {
          prior_a_mean_as_list[[c]] <- matrix(prior_a_mean[c,,], ncol = n_tests, nrow = n_covariates_max)
          prior_a_sd_as_list[[c]] <-   matrix(prior_a_sd[c,,],   ncol = n_tests, nrow = n_covariates_max)
        }
      }
      
      n_pops <- 1
      group <- rep(1, N)
 
      X_nd <- X_d <- list()
      for(t in 1:n_tests) {
        X_nd[[t]] <- matrix(1, nrow=N, ncol=n_covariates_max_nd)
        X_d[[t]] <- matrix(1, nrow=N, ncol=n_covariates_max_nd)
      }
      
      {
        
        
        # if (prior_only == 0 ) { 
        Stan_data_list = list(  N =   N, # length(corr_prior_dist_LKJ_4), #  N, 
                           n_tests = n_tests,
                           y = y,
                           n_class = n_class,
                           n_pops = n_pops, 
                           pop = c(rep(1, N)),
                           ##
                           n_covariates_max_nd = n_covariates_max_nd,
                           n_covariates_max_d = n_covariates_max_d,
                           n_covariates_max = n_covariates_max,
                           X_nd = X_nd,
                           X_d =  X_d,
                           ##
                           n_covs_per_outcome = n_covs_per_outcome, 
                           ##
                           corr_force_positive = corr_force_positive,
                           known_num = 0,
                           ##
                           overflow_threshold = overflow_threshold,
                           underflow_threshold = underflow_threshold,
                           ##
                           prior_only = prior_only,  
                           prior_beta_mean =   prior_a_mean_as_list ,
                           prior_beta_sd  =   prior_a_sd_as_list ,
                           prior_LKJ = prior_LKJ, 
                           prior_p_alpha = matrix(rep(5, n_pops), ncol = 1),
                           prior_p_beta =  matrix(rep(10, n_pops), ncol = 1),
                           ##
                           Phi_type = Phi_type,
                           handle_numerical_issues = handle_numerical_issues,
                           fully_vectorised = fully_vectorised)
                         
 
        
      }
  
      ##  ------- Set inits:
      {
          u_raw <- array(0.01, dim = c(N, n_tests))
          
          k_choose_2   = (n_tests * (n_tests - 1)) / 2;
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
          
          Stan_init_list = list(
            u_raw = (u_raw),
            p_raw =   array(-0.6931472), # equiv to 0.20 on p
            beta_vec = beta_vec_init,
            off_raw = off_raw,
            col_one_raw =  col_one_raw
            #L_Omega_raw = array(0.01, dim = c(n_class, choose(n_tests, 2)))
          )
      }
      
      Stan_data_list_save_1 <- Stan_data_list
      
    }

# 
# # ##
# Stan_data_list_save_1$n_class
# Stan_data_list_save_1$n_pops
# Stan_data_list_save_1$pop
# Stan_data_list_save_1$n_covariates_max_nd
# Stan_data_list_save_1$n_covariates_max_d
# Stan_data_list_save_1$n_covariates_max
# str(Stan_data_list_save_1$X_nd)
# str(Stan_data_list_save_1$X_d)
# ##
# Stan_data_list_save_1$n_covs_per_outcome
# ##
# Stan_data_list_save_1$corr_force_positive
# Stan_data_list_save_1$known_num
# ##
# Stan_data_list_save_1$overflow_threshold
# Stan_data_list_save_1$underflow_threshold
# ##
# Stan_data_list_save_1$prior_only
# Stan_data_list_save_1$prior_beta_mean
# Stan_data_list_save_1$prior_beta_sd
# Stan_data_list_save_1$prior_LKJ
# Stan_data_list_save_1$prior_p_alpha
# Stan_data_list_save_1$prior_p_beta
# ##
# Stan_data_list_save_1$Phi_type
# Stan_data_list_save_1$handle_numerical_issues
# Stan_data_list_save_1$fully_vectorised
# # ##
# 
# 
# str(Stan_init_list$u_raw)
# str(Stan_init_list$p_raw)
# str(Stan_init_list$beta_vec)
# str(Stan_init_list$off_raw)
# str(Stan_init_list$col_one_raw)

     ##  prior_beta_mean; position=1; dims declared=(2,5,1); dims found=(2,1,5) 

    # ##
    # Stan_init_list$p_raw <- array(-0.6931472)
    # Stan_data_list$prior_p_alpha <- array(5)
    # Stan_data_list$prior_p_beta <- array(10)
    # ##
    str(Stan_data_list$prior_LKJ)
    str(Stan_data_list$prior_p_alpha)
    str(Stan_data_list$prior_p_beta)
    str(Stan_init_list$p_raw)
    
    ## make lists of lists for inits 
    n_chains_burnin <- 8 
    init_lists_per_chain <- rep(list(Stan_init_list), n_chains_burnin) 
    
    ## set Stan model file path for your Stan model (replace with your path)
    Stan_model_file_path <- system.file("stan_models/LC_MVP_bin_PartialLog_v5.stan", package = "BayesMVP")
    #### Stan_model_file_path <- system.file("stan_models/PO_LC_MVP_bin.stan", package = "BayesMVP")
    
    ## Stan_model_file_path <- file.path(getwd(), "stan_models/LC_MVP_bin_PartialLog_v5.stan")
    
    
    
   ## mod
    
    ###  -----------  Compile + initialise the model using "MVP_model$new(...)" 
    model_obj <- BayesMVP::MVP_model$new(   Model_type =  "Stan",
                                            y = y,
                                            N = N,
                                            ##  X = NULL,
                                            ##  model_args_list = model_args_list, # this arg is only needed for BUILT-IN (not Stan) models
                                            Stan_data_list = Stan_data_list,
                                            Stan_model_file_path = Stan_model_file_path,
                                            init_lists_per_chain = init_lists_per_chain,
                                            sample_nuisance = TRUE,
                                            n_chains_burnin = n_chains_burnin,
                                            n_params_main = n_params_main,
                                            n_nuisance = n_nuisance)
    
    
    init_object <-  BayesMVP:::initialise_model( Model_type = "Stan",
                                                    stream = 123,
                                                    sample_nuisance = TRUE,
                                                    n_nuisance_override = NULL,
                                                    model_args_list = NULL,
                                                    compile = TRUE,
                                                    force_recompile = FALSE,
                                                    cmdstanr_model_fit_obj = NULL,
                                                    ##
                                                    Stan_model_file_path = Stan_model_file_path,
                                                    Stan_data_list = Stan_data_list,
                                                    Stan_cpp_user_header = NULL,
                                                    Stan_cpp_flags = NULL,
                                                    stanc_args = NULL)
 
    
    
    ## ----------- Set basic sampler settings
    {
      ### seed <- 123
      n_chains_sampling <- min(64, parallel::detectCores())
      n_superchains <- min(8, parallel::detectCores())  ## round(n_chains_sampling / n_chains_burnin) # Each superchain is a "group" or "nest" of chains. If using ~8 chains or less, set this to 1. 
      n_iter <- 500                                 
      n_burnin <- 500
      n_nuisance_to_track <- 10 # set to some small number (< 10) if don't care about making inference on nuisance params (which is most of the time!)
    }
    
    
   ##   n_burnin <- 10
 
    
    # Stan_data_list$prior_p_alpha <- array(Stan_data_list$prior_p_alpha)
    # Stan_data_list$prior_p_alpha <- array(Stan_data_list$prior_p_beta)
    # 
    # str( Stan_data_list$prior_p_alpha)
    # 
    # ### convert_Stan_data_list_to_JSON(Stan_data_list = Stan_data_list)
    
    #### cmdstanr::write_stan_json(data = Stan_data_list, file = json_file_path)
  
    ## sample / run model
    model_samples <-  model_obj$sample(  partitioned_HMC = TRUE,
                                         diffusion_HMC = TRUE,
                                         seed = 1,
                                         n_burnin = n_burnin,
                                         n_iter = n_iter,
                                         n_chains_sampling = n_chains_sampling,
                                         n_superchains = n_superchains,
                                         ## Some other optional arguments:
                                         Stan_data_list = Stan_data_list,
                                         Stan_model_file_path = Stan_model_file_path,
                                         y = y,
                                         N = N,
                                         n_params_main = n_params_main,
                                         n_nuisance = n_nuisance,
                                         init_lists_per_chain = init_lists_per_chain,
                                         n_chains_burnin = n_chains_burnin,
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
    
    

    # extract # divergences + % of sampling iterations which have divergences
    model_fit$get_divergences()
    

    
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  
  
  
