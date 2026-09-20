
########## -------- EXAMPLE 2(a) --------------------------------------------------------------------------------------------------------- 
## Running the BUILT-IN (i.e. manual gradients) LC-MVP model (e.g., for the analysis of test accuracy data without a gold standard).
## Uses simulated data.
 


 
# # ####  ---- 1. Install BayesMVP (from GitHub) - SKIP THIS STEP IF INSTALLED: -----------------------------------------------------------
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

# require(BayesMVP)
# CUSTOM_FLAGS <- list()
# install_BayesMVP(CUSTOM_FLAGS = list())
# require(BayesMVP)




# # Install if needed
# remotes::install_github("ashbaldry/ttscli")
# library(ttscli)
# 
# # Speak a number
# speak_text("1028")
# 
# install.packages("text2speech")
# require(text2speech)
# 
# text2speech::
#   
#   speak("1028")
# 
# install.packages("speech")
# require(speech)
# 
# speak("1028")
# 
# 



# ?BayesMVP::MVP_model






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
source(file.path(getwd(), "load_R_packages.R"))
require(BayesMVP)

## Function to check BayesMVP AVX support 
BayesMVP:::detect_vectorization_support()





## Simulate data (for N = 500)
{
      source(file.path(getwd(), "R_fn_load_data_binary_LC_MVP_sim.R"))
      N <- 2500
      ## Call the fn to simulate binary data:
      data_sim_outs <- simulate_binary_LC_MVP_data(N_vec = N, 
                                                   seed = 123, 
                                                   DGP = 5)
      ## Extract dataset (y):
      y <- data_sim_outs$y_binary_list[[1]]
      
      true_Se_vec <- data_sim_outs$Se_true_observed_list[[1]]
      true_Sp_vec <- data_sim_outs$Sp_true_observed_list[[1]]
      true_prev <- data_sim_outs$prev_true_observed_list[[1]]
}


 

Model_type <- "LC_MVP"
 

 

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
  n_covariates_per_outcome_mat <- array(n_covariates_max, dim = c(n_class, n_tests))
  n_covariates_total_nd  =    (sum( (n_covariates_per_outcome_mat[1,])));
  n_covariates_total_d   =     (sum( (n_covariates_per_outcome_mat[2,])));
  n_covariates_total  =       n_covariates_total_nd + n_covariates_total_d;
  ##
  n_params_main <- n_corrs + n_covariates_total + 1
 
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
  ##
  # {
  #   prior_a_mean  <-  vector("list", length = n_class)
  #   prior_a_sd <-  vector("list", length = n_class)
  # 
  #   for (c in 1:n_class) {
  #     prior_a_mean[[c]] <-  array(0,  dim = c(n_covariates_max, n_tests))
  #     prior_a_sd[[c]] <- array(1,  dim = c(n_covariates_max, n_tests))
  #   }
  # 
  #   # intercepts / coeffs prior means
  #   prior_a_mean[[1]][1, 1] <- -2.10
  #   prior_a_sd[[1]][1, 1] <- 0.45
  # 
  #   prior_a_mean[[2]][1, 1] <- +0.40
  #   prior_a_sd[[2]][1, 1] <-  0.375
  # }
    
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
    
    init = list(
      u_raw = (u_raw),
      p_raw =    array(-0.6931472), # equiv to 0.20 on p
      beta_vec = beta_vec_init,
      off_raw = off_raw,
      col_one_raw =  col_one_raw
      #L_Omega_raw = array(0.01, dim = c(n_class, choose(n_tests, 2)))
    )
  }
 
}


    ## -----------  initialise model / inits etc
    # based on (informal) testing, more than 8 burnin chains seems unnecessary 
    # and probably not worth the extra overhead (even on a 96-core AMD EPYC Genoa CPU)
    n_chains_burnin <- min(16, parallel::detectCores()) 
    init_lists_per_chain <- rep(list(init), n_chains_burnin) 
    
     
    ## make model_args_list (note: Stan models don't need this)
    model_args_list  <- list(       n_class = n_class,
                                    lkj_cholesky_eta =  matrix(c(12, 3), ncol = 1), 
                                    n_covariates_per_outcome_mat = n_covariates_per_outcome_mat,  
                                    #X = X, # only needed if want to include covariates
                                    num_chunks =   find_num_chunks_MVP(N, n_tests),
                                    prior_coeffs_mean_mat = prior_a_mean_as_list,
                                    prior_coeffs_sd_mat =    prior_a_sd_as_list, 
                                    prev_prior_a =  matrix(1, ncol = 1), # show how to change this later
                                    prev_prior_b =  matrix(1, ncol = 1)  # show how to change this later
                             )
    
    model_args_list_save <- model_args_list
    
    model_args_list$num_chunks
    
    # {
    #   str(model_args_list$lkj_cholesky_eta)
    #   str(model_args_list$prev_prior_a)
    #   str(model_args_list$prev_prior_b)
    #   str(init_lists_per_chain[[1]]$p_raw)
    # }

   
  
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

    
    
    
    
    
    
    
    
    
 
    {
      ##  NOTE: You can also use "model_obj$sample()" to update the model.
      ##
      ##  For example, if using the same model but a new/different dataset (so new y and N, and n_nuisance needed), you can do:
      ##  model_obj$sample(y = y, N = N, n_nuisance = n_nuisance, ...)
      ##
      ##  You can also update model_args_list. 
      ##  For example, let's say I wanted to change the prior for disease prevalence to be informative s.t. prev ~ beta(5, 10). 
      ##  I could do this by modifying model_args_list:
      model_args_list$prev_prior_a <-  matrix(5, ncol = 1)
      model_args_list$prev_prior_b <-  matrix(10, ncol = 1) ## 10
      
      
      # ## To run standard HMC, do:
      partitioned_HMC <- FALSE ;    diffusion_HMC <- FALSE
      # ## To run * partitioned * HMC (i.e. sample nuisance and main params. seperately), do:
      #  partitioned_HMC <- TRUE ;     diffusion_HMC <- FALSE # fine
      # ## To run partitioned * and * diffusion HMC (i.e., nuisance params. sampled using diffusion-pathspace HMC), do:
      # partitioned_HMC <- TRUE ;    diffusion_HMC <- TRUE  # fine
      
      ## To use manual tau (path length):
      #### manual_tau <- TRUE ;    tau_if_manual <- c(0.50, 1.0) # NOTE: first element in "tau_if_manual" is tau_main and second is tau_us
      ## To use automaticn tau (path length) adaptation (using SNAPER-HMC):
      manual_tau <- FALSE ;   tau_if_manual <- NA
      
      force_autodiff <- FALSE
      force_PartialLog <- FALSE
      multi_attempts <- TRUE
      
      n_runs <- 5
    }
    
    
    
    
    ## ----------- Set some basic sampler settings
    {
            #### seed <- 123
            n_chains_sampling <- min(64, parallel::detectCores()) ## min(64, parallel::detectCores())
            n_superchains     <- min(8, parallel::detectCores())  ## round(n_chains_sampling / n_chains_burnin) # Each superchain is a "group" or "nest" of chains. If using ~8 chains or less, set this to 1. 
            n_iter <-   1000                                 
            n_burnin <- 500
            n_nuisance_to_track <- n_nuisance # set to some small number (< 10) if don't care about making inference on nuisance params (which is most of the time!)
            ##
            #### learning_rate <- 0.025
            learning_rate <- 0.05
            ##
            #  metric_shape_main <- "diag"    ; metric_type_main <- "Empirical"  # works w/ part == TRUE (N = 500)  + w/ part == FALSE (N = 500)
            # metric_shape_main <- "dense"   ; metric_type_main <- "Empirical"   # works w/ part == TRUE (N = 500) + w/ part == FALSE (N = 500)
            # metric_shape_main <- "diag"    ; metric_type_main <- "Hessian"      # works w/ part == TRUE (N = 500) + w/ part == FALSE (N = 500)
             metric_shape_main <- "dense"  ; metric_type_main <- "Hessian"      # works w/ part == TRUE (N = 500) + w/ part == FALSE (N = 500)
             ##
            if (parallel::detectCores() < 64) { 
              vect_type <- "AVX2"
            } else { 
              vect_type <- "AVX512"
            }
            ##
            tau_mult = 1.60
            ##
            clip_iter = round(n_burnin/10) ; clip_iter
            ##
            
            if (partitioned_HMC == TRUE) {
            
                          
                          if (metric_type_main == "Hessian") { 

                                   #ratio_M_main <- 0.25
                                   ratio_M_main <- 0.50
                                   ## ratio_M_main <- 0.75
                                   ##
                                   ratio_M_us   <- 0.25
                                   ##
                                   interval_width_main <- 50
                                   ##
                                   interval_width_nuisance <- interval_width_main
                               
                          } else if (metric_type_main == "Empirical") { 
                            
                                    if (metric_shape_main == "diag") { ## Seems to work well for N = 500 w/ the following settings:
                                      
                                                interval_width_main <- 50
                                                ##
                                                ratio_M_main <- 0.50
                                                ##
                                                ratio_M_us   <- 0.25
                                                ##
                                                interval_width_nuisance <- interval_width_main
                                               
                                    } else if (metric_shape_main == "dense") { 
                                      
                                                interval_width_main <- 50
                                                ##
                                                ## ratio_M_main <- 0.75
                                                ratio_M_main <- 0.50
                                                ##
                                                ratio_M_us   <- 0.25
                                                ##
                                                interval_width_nuisance <- interval_width_main
                                      
                                    }
                                  
                          }
                          ##
                          ## ratio_M_us   <- 0
                          # ##
                          # interval_width_nuisance <- interval_width_main
                          # # # interval_width_nuisance <- 1
                          # #   interval_width_nuisance <- 5
                          # # # interval_width_nuisance <- 10
                          # # # interval_width_nuisance <- 50
                          # ##
                          # # ratio_M_main <- 0.25
                          # # ratio_M_main <- 0.50
                          # ratio_M_main = 0.75 ## NOTE: if set to 0, you will get a UNIT metric for these params. 
                          # # ratio_M_main = 0.80 ## NOTE: if set to 0, you will get a UNIT metric for these params. 
                          # # ratio_M_main = 0.90 ## NOTE: if set to 0, you will get a UNIT metric for these params.
                          # # ratio_M_main = 0.95 ## NOTE: if set to 0, you will get a UNIT metric for these params.
                          # #### ratio_M_main = 1.00 ## NOTE: if set to 0, you will get a UNIT metric for these params.
                          # ## For nuisance metric:
                          # #### ratio_M_us =   0.75 ## NOTE: if set to 0, you will get a UNIT metric for these params. 
                          # # ratio_M_us <-  0.50 ## NOTE: if set to 0, you will get a UNIT metric for these params. 
                          # ratio_M_us <-  0.25
                          # ## ratio_M_us <-  0
            
            } else if (partitioned_HMC == FALSE) {
                
                        if (metric_type_main == "Empirical") { 
                        
                              ratio_M_main <- 0.50
                              ##
                              ratio_M_us   <- 0.50
                              ##
                              interval_width_main <- 1
                              ##
                              interval_width_nuisance <- interval_width_main
                              
                        } else if (metric_type_main == "Hessian") {
                          
                              ratio_M_main <- 0.50
                              ##
                              ratio_M_us   <- 0.50
                              ##
                              interval_width_main <- 5
                              ##
                              interval_width_nuisance <- interval_width_main
                          
                        }
              
              
            }
    }
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    #### ------ sample model using "  model_obj$sample()" --------- 
    {
      
    {
    
      tau_main_vec <- eps_main_vec <- L_main_vec <- c()
      tau_us_vec <- eps_us_vec <- L_us_vec <- c()
      any_divs_indicator_vec <- pct_divs_vec <- total_time_vec <-  c()
      Min_ESS_per_sec_sampling_vec <- c()
      Min_ESS_per_grad_sampling_vec <- c()
    
    for (seed in 1:n_runs) {
      
        i <- seed
      
        {
            
            set.seed(seed)
            require(dqrng)
            ## Set dqrng type:
            ## dqrng_type <- "pcg64"
            dqrng_type <- "Xoshiro256+"
            ##dqrng_type <- "Xoshiro256++"
            dqrng::dqRNGkind(dqrng_type)
            ## Set dqrng seed:
            dqrng::dqset.seed(seed)
            
            RcppParallel::setThreadOptions(numThreads = n_chains_sampling);
            
            # state <- dqrng_get_state() ; state
            # dqrng_set_state(state)
            # dqrng::dqrng_set_state(c("pcg64", "1", "1", "1"))  # Sets all state components to 1
            # dqrng::dqrng_get_state()
            
            model_samples <-  model_obj$sample(   partitioned_HMC = partitioned_HMC,
                                                  diffusion_HMC = diffusion_HMC,
                                                  ##
                                                  manual_tau = manual_tau,
                                                  tau_if_manual =  tau_if_manual,
                                                  ##
                                                  seed = seed,
                                                  n_burnin = n_burnin,
                                                  n_iter = n_iter,
                                                  n_chains_sampling = n_chains_sampling,
                                                  n_superchains = n_superchains,
                                                  ## Some other arguments:
                                                  Stan_data_list = list(),
                                                  y = y,
                                                  N = N,
                                                  n_params_main = n_params_main,
                                                  n_nuisance = n_nuisance,
                                                  ##
                                                  init_lists_per_chain = init_lists_per_chain,
                                                  n_chains_burnin = n_chains_burnin,
                                                  model_args_list = model_args_list,
                                                  ##
                                                  sample_nuisance = TRUE,
                                                  ##
                                                  force_autodiff = force_autodiff,
                                                  force_PartialLog = force_PartialLog,
                                                  multi_attempts = multi_attempts,
                                                  ##
                                                  adapt_delta = 0.80,
                                                  learning_rate = learning_rate,
                                                  ##
                                                  metric_shape_main = metric_shape_main,
                                                  metric_type_main = metric_type_main,
                                                  ##
                                                  tau_mult = tau_mult,
                                                  clip_iter = clip_iter,
                                                  interval_width_main = interval_width_main,
                                                  ratio_M_us = ratio_M_us,
                                                  ratio_M_main = ratio_M_main,
                                                  ##
                                                  ## parallel_method = "RcppParallel",
                                                     parallel_method = "OpenMP",
                                                  ## vect_type = BayesMVP:::detect_vectorization_support(),
                                                  vect_type = vect_type,
                                                  n_nuisance_to_track = n_nuisance_to_track
                                                  )   
            
            
            #### --- MODEL RESULTS SUMMARY + DIAGNOSTICS -------------------------------------------------------------
            # after fitting, call the "summary()" method to compute + extract e.g. model summaries + traces + plotting methods 
            # model_fit <- model_samples$summary() # to call "summary()" w/ default options 
            require(bridgestan)
            model_fit <- model_samples$summary(save_log_lik_trace = FALSE, 
                                               compute_nested_rhat = FALSE,
                                               compute_transformed_parameters = FALSE) 
            
 
            
            if (i > 5) {
                try({
                  beepr::beep("wilhelm")
                })
            } else {
                try({
                  beepr::beep("ping")
                })
            }
            
            # beepr::beep("ping")
            # 
            # ?beepr
            
              
        }
     

      
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
      
      Min_ESS_per_grad_sampling <- model_efficiency_metrics$Min_ESS_per_grad_sampling ; Min_ESS_per_grad_sampling * 1000
      grad_evals_per_sec <- model_efficiency_metrics$grad_evals_per_sec ; grad_evals_per_sec
      
      {
        
          if (Min_ESS_per_sec_sampling < 0.05) {
            break
          }
        
          Min_ESS_per_sec_sampling_vec[i] <- Min_ESS_per_sec_sampling
          Min_ESS_per_grad_sampling_vec[i] <- Min_ESS_per_grad_sampling * 1000
          ##
          tau_main_vec[i] <- print(model_fit$get_HMC_info()$tau_main)
          eps_main_vec[i] <- print(model_fit$get_HMC_info()$eps_main)
          L_main_vec[i] <- print(ceil(model_fit$get_HMC_info()$tau_main /  model_fit$get_HMC_info()$eps_main))
          ##
          tau_us_vec[i] <- print(model_fit$get_HMC_info()$tau_us)
          eps_us_vec[i] <- print(model_fit$get_HMC_info()$eps_us)
          L_us_vec[i] <- print(ceil(model_fit$get_HMC_info()$tau_us /  model_fit$get_HMC_info()$eps_us))
          ##
          any_divs_indicator_vec[i] <- ifelse(model_fit$get_divergences()$n_divs == 0, FALSE, TRUE)
          pct_divs_vec[i] <- model_fit$get_divergences()$pct_divs
          ##
          total_time_vec[i] <- model_efficiency_metrics$time_total_inc_summaries
      }
      
      
      summary_gq <-   model_fit$get_summary_generated_quantities () %>% print(n = 150) 

      
      
      All_DTA_params_vec <- summary_gq$mean
      names(All_DTA_params_vec) <- summary_gq$parameter
      
      est_Se_vec <- All_DTA_params_vec[1:n_tests]
      est_Sp_vec <- All_DTA_params_vec[(n_tests + 1):(2*n_tests)]
      est_prev <- All_DTA_params_vec[3*n_tests + 1]
      
      # dput(All_DTA_params_vec)
      # 
      # 
      # All_DTA_params_vec_alg_2 <- c(  Se_bin.1 = 0.534576874506674, Se_bin.2 = 0.530367169148889, 
      #                                 Se_bin.3 = 0.568442635400891, Se_bin.4 = 0.64978817958863, Se_bin.5 = 0.682054833747197, 
      #                                 Sp_bin.1 = 0.981303135254505, Sp_bin.2 = 0.954433136480082, Sp_bin.3 = 0.904014016294787, 
      #                                 Sp_bin.4 = 0.911750581250248, Sp_bin.5 = 0.863201731323257,
      #                                 p.1 = 0.219298617679486)*100
      
      
      # {
      #   print(model_fit$get_HMC_info()$tau_us)
      #   print(model_fit$get_HMC_info()$eps_us)
      #   print(ceil(model_fit$get_HMC_info()$tau_us /  model_fit$get_HMC_info()$eps_us))
      # }
  
    }
      
      try({  
        beepr::beep("ready")
      })
      
          message(paste("tau_main_vec = ")) ;   print(round(tau_main_vec, 5))
          message(paste("eps_main_vec = ")) ;   print(round(eps_main_vec, 5))
          message(paste("L_main_vec = ")) ;     print(ceil(tau_main_vec / eps_main_vec)) 
          ##
          message(paste("tau_us_vec = ")) ;   print(round(tau_us_vec, 5))
          message(paste("eps_us_vec = ")) ;   print(round(eps_us_vec, 5))
          message(paste("L_us_vec = ")) ;     print(ceil(tau_us_vec / eps_us_vec)) 
          ##
          message(cat("any_divs_indicator_vec = ", any_divs_indicator_vec))
          ##
          message(paste("SD of tau_main_vec = ")) ; print(sd(tau_main_vec))
          message(paste("SD of eps_main_vec = ")) ; print(sd(eps_main_vec))
          ##
          message(paste("SD of tau_us_vec = ")) ; print(sd(tau_us_vec))
          message(paste("SD of eps_us_vec = ")) ; print(sd(eps_us_vec))
    }
    


      
      

      
      
    
    
    {
          message(paste("-------------------------------------------------------------------------------------------------------------------"))
          message("Summary for N = ", N, ":")
          message("n_runs = ", n_runs)
          ##
          message(paste("--------------------------------------------"))
          message(("Min_ESS_per_sec_sampling_vec = "))
          Min_ESS_per_sec_sampling_vec <- round(Min_ESS_per_sec_sampling_vec, 2)
          print(Min_ESS_per_sec_sampling_vec)
          print(mean(Min_ESS_per_sec_sampling_vec)) ; print(sd(Min_ESS_per_sec_sampling_vec))
          ##
          message(paste("--------------------------------------------"))
          message(("Min_ESS_per_grad_sampling_vec = "))
          Min_ESS_per_grad_sampling_vec <- round(Min_ESS_per_grad_sampling_vec, 2)
          print(Min_ESS_per_grad_sampling_vec)
          print(mean(Min_ESS_per_grad_sampling_vec)) ; print(sd(Min_ESS_per_grad_sampling_vec))
          ##
          message(paste("--------------------------------------------"))
          message(("L_main_vec = "))
          L_main_vec <- round(L_main_vec, 0)
          print(L_main_vec)
          print(mean(L_main_vec)) ; print(sd(L_main_vec))
          ##
          message(paste("--------------------------------------------"))
          message(("pct_divs_vec = "))
          print(pct_divs_vec)
          print(mean(pct_divs_vec)) ; print(sd(pct_divs_vec))
          ##
          message(paste("--------------------------------------------"))
          message(("total_time_vec = "))
          print(total_time_vec)
          print(mean(total_time_vec)) ; print(sd(total_time_vec))
          ##
          message(paste("--------------------------------------------"))
          message("partitioned_HMC = ") ;  print(partitioned_HMC)
          message("diffusion_HMC = ") ;  print(diffusion_HMC)
          message(paste("--------------------------------------------"))
          message("learning_rate = ") ;  print(learning_rate)
          message(paste("--------------------------------------------"))
          message("metric_shape_main = ") ;  print(metric_shape_main)
          message("metric_type_main = ") ;  print(metric_type_main)
          message(paste("--------------------------------------------"))
          message("vect_type = ") ;  print(vect_type)
          ##
          message(paste("--------------------------------------------"))
          message("force_autodiff = ") ;  print(force_autodiff)
          message("force_PartialLog = ") ;  print(force_PartialLog)
          message("multi_attempts = ") ;  print(multi_attempts)
          ##
          message(paste("--------------------------------------------"))
          message("n_burnin = ") ;  print(n_burnin)
          message("tau_mult = ") ;  print(tau_mult)
          message("clip_iter = ") ;  print(clip_iter)
          message("interval_width_main = ") ;  print(interval_width_main)
          message("ratio_M_us = ") ;  print(ratio_M_us)
          message("ratio_M_main = ") ;  print(ratio_M_main)
          ##
          message(paste("--------------------------------------------"))
          message("Difference between true estimates and model estimates for Se:")
          print((true_Se_vec - est_Se_vec)*100)
          message("Difference between true estimates and model estimates for Sp:")
          print((true_Sp_vec - est_Sp_vec)*100)
          message("Difference between true estimates and model estimates for prev:")
          print((true_prev - est_prev)*100)
    }
      
      sum(total_time_vec) / 60
    

      
    }
    #
    
    
    
   # 
   #  model_samples$init_object
   #  
   # outs <- create_summary_and_traces(model_results =  model_samples$result, 
   #                            init_object =     model_samples$init_object,
   #                            n_nuisance =  n_nuisance)
   #  
   #  unique(tau_main_vec)
   #  unique(eps_main_vec)
   #  unique(tau_main_vec)
   #  unique(L_main_vec)
   #  (any_divs_indicator_vec)
    

    
    ## N= 12,500 results using "standrd HMC" (FALSE-FALSE) alg:
    Se_vec <- c(0.535, 0.518, 0.558, 0.642, 0.690)*100
    names(Se_vec) <- c("Se1", "Se2", "Se3", "Se4","Se5")
    Sp_vec <- c(0.984, 0.957, 0.907, 0.917, 0.868)*100
    names(Sp_vec) <- c("Sp1", "Sp2", "Sp3", "Sp4","Sp5")
    Prev <- c(0.224)*100
    names(Prev) <- c("Prev")
    All_DTA_params_vec_alg_1 <- c(Se_vec, Sp_vec, Prev) ; All_DTA_params_vec_alg_1
    
    
    # ## N = 12,500 results when using "Diffusion-pathspace HMC" (TRUE-TRUE) alg:
    # Se_vec <- c(0.527, 0.521, 0.559, 0.648, 0.680)*100
    # names(Se_vec) <- c("Se1", "Se2", "Se3", "Se4","Se5")
    # Sp_vec <- c(0.982, 0.955, 0.905, 0.915, 0.867)*100
    # names(Sp_vec) <- c("Sp1", "Sp2", "Sp3", "Sp4","Sp5")
    # Prev <- c(0.224)*100
    # names(Prev) <- c("Prev")
    # All_DTA_params_vec_alg_2 <- c(Se_vec, Sp_vec, Prev) ; All_DTA_params_vec_alg_2
 
    Abs_diff_between_algs_1_and_2 <- abs(All_DTA_params_vec_alg_1 - All_DTA_params_vec_alg_2)
    Abs_diff_between_algs_1_and_2
    
    max(Abs_diff_between_algs_1_and_2)
    mean(Abs_diff_between_algs_1_and_2)
    
    round(Abs_diff_between_algs_1_and_2, 2)
 
## Set important variables:
n_thread_total_combos <- length(pilot_study_opt_N_chunks_list$n_threads_vec)
##
n_max_chunk_combos <-  pilot_study_opt_N_chunks_list$n_max_chunk_combos 
##
N_vec <- pilot_study_opt_N_chunks_list$N_vec
n_runs <- pilot_study_opt_N_chunks_list$n_runs
start_index <- pilot_study_opt_N_chunks_list$start_index

## Make array to store results:
times_array <- array(dim = c(length(N_vec), n_max_chunk_combos, n_runs, n_thread_total_combos))
str(times_array)
dimnames(times_array) <- list(N = c(500, 1000, 2500, 5000, 12500, 25000),
                              n_chunks_index = seq(from = 1, to = n_max_chunk_combos, by = 1),
                              run_number = seq(from = 1, to = n_runs, by = 1),
                              n_threads_index =   pilot_study_opt_N_chunks_list$n_threads_vec) 


## Set key variables:
n_class <- global_list$Model_settings_list$n_class
n_tests <- global_list$Model_settings_list$n_tests
n_params_main <- global_list$Model_settings_list$n_params_main
n_corrs <-  global_list$Model_settings_list$n_corrs
n_covariates_total <-  global_list$Model_settings_list$n_covariates_total
n_nuisance_to_track <- 10 ## set to small number 
##
pilot_study_opt_N_chunks_list$manual_gradients <- TRUE ## Using manual-gradient function !! (AD doesn't have "chunking")
## Fixed # of * burnin * chains:
n_chains_burnin <- min(parallel::detectCores(), 8)
n_burnin <- 500
##
sample_nuisance <- FALSE
partitioned_HMC <- FALSE
diffusion_HMC <- FALSE
Model_type <- "LC_MVP"
force_autodiff <- force_PartialLog <- FALSE
multi_attempts <- FALSE
metric_shape_main <- "dense"
##




df_index <- 1
N <- 500
sample_nuisance <- TRUE
partitioned_HMC <- FALSE
diffusion_HMC <- FALSE
Model_type <- "LC_MVP"
force_autodiff <- force_PartialLog <- FALSE
multi_attempts <- FALSE
metric_shape_main <- "dense"
##

  
  
  n_nuisance <- N * n_tests
  ## Print:
  print(paste("N = ", N))
  print(paste("n_nuisance = ", n_nuisance))
  ##
  ## Get Rcpp / C++ lists:
  Model_args_as_Rcpp_List <- BayesMVP_model_obj$init_object$Model_args_as_Rcpp_List
  Model_args_as_Rcpp_List$N <- N
  Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
  ##
  EHMC_args_as_Rcpp_List <- BayesMVP:::init_EHMC_args_as_Rcpp_List(diffusion_HMC = diffusion_HMC)
  ## Edit entries to ensure don't get divergences - but also ensure suitable L chosen:
  EHMC_args_as_Rcpp_List$eps_main <- 0.01
  ## Use path length of 16:
  L_main <- 8
  EHMC_args_as_Rcpp_List$tau_main <-     L_main * EHMC_args_as_Rcpp_List$eps_main 
  ## Metric Rcpp / C++ list::
  EHMC_Metric_as_Rcpp_List <- BayesMVP:::init_EHMC_Metric_as_Rcpp_List(   n_params_main = n_params_main, 
                                                                          n_nuisance = n_nuisance, 
                                                                          metric_shape_main = metric_shape_main)  
  ## Assign SIMD_vect_type:
  Model_args_as_Rcpp_List$Model_args_strings[c("vect_type",
                                               "vect_type_exp", "vect_type_log", "vect_type_lse", "vect_type_tanh", 
                                               "vect_type_Phi", "vect_type_log_Phi", "vect_type_inv_Phi", 
                                               "vect_type_inv_Phi_approx_from_logit_prob"), ] <-  "AVX512"
  
  for (c in 1:n_class) {
    for (t in 1:n_tests) {
      Model_args_as_Rcpp_List$Model_args_2_later_vecs_of_mats_double[[1]][[c]][[t]] <- matrix(1, nrow = N, ncol = 1)
    }
  }
  
 
 
    ## Set the number of chunks to use in model_args_list:
    num_chunks <- 1
    
    Model_args_as_Rcpp_List$Model_args_ints[4] <- num_chunks
    ## Using manual-gradient model obj:
    BayesMVP_model_obj <- BayesMVP_LC_MVP_model_using_manual_grad_obj
    
 
      iii <- 1
      seed <- 1
      
 
        
        ## Set number of ** sampling ** chains:
        n_threads <- 8
        n_chains_sampling <- 8
        n_superchains <- 8
        ## Print info:
        print(paste("n_threads = ", n_threads))
        print(paste("n_chains_sampling = ", n_chains_sampling))
        print(paste("n_params_main = ", n_params_main))
        theta_main_vectors_all_chains_input_from_R <- matrix(0.01, ncol = n_chains_sampling, nrow = n_params_main)
        ## Inits for main:
        theta_main_vectors_all_chains_input_from_R[ (n_corrs + 1):(n_corrs + n_covariates_total/2) , ] <- rep(-1, n_covariates_total/2)
        theta_main_vectors_all_chains_input_from_R[ (n_corrs + 1 + n_covariates_total/2):(n_corrs + n_covariates_total), ] <- rep(1, n_covariates_total/2)
        theta_main_vectors_all_chains_input_from_R[ n_params_main ] =  -0.6931472  # this is equiv to starting val of p = 0.20 -  since: 0.5 * (tanh( -0.6931472) + 1)  = -0.6931472
        ##
        index_nuisance = 1:n_nuisance
        index_main = (n_nuisance + 1):(n_nuisance + n_params_main)
        ## Inits for nuisance:
        theta_us_vectors_all_chains_input_from_R <- matrix(0.01, ncol = n_chains_sampling, nrow = n_nuisance)
        for (kk in 1:n_chains_sampling) {
          theta_us_vectors_all_chains_input_from_R[, kk] <-  c(global_list$initial_values_list$inits_u_list[[df_index]])
        }
        ## Get y:
        y_binary_list <- global_list$data_sim_outs$y_binary_list
        y <- y_binary_list[[df_index]]
        ##
        str(Model_args_as_Rcpp_List$Model_args_2_later_vecs_of_mats_double)
        ##
        ### Call C++ parallel sampling function * directly * (skip costly burn-in phase + out of scope for this paper)
        RcppParallel::setThreadOptions(numThreads = n_chains_sampling);
        ##
        
        Model_args_as_Rcpp_List$Model_args_bools <- ifelse(  Model_args_as_Rcpp_List$Model_args_bools == 1, TRUE, FALSE)
        
        
        Model_args_as_Rcpp_List$Model_args_strings <- ifelse(   Model_args_as_Rcpp_List$Model_args_strings  == "AVX512", "Stan", Model_args_as_Rcpp_List$Model_args_strings )
        
        
        Model_args_as_Rcpp_List$Model_args_col_vecs_double 
        Model_args_as_Rcpp_List$Model_args_mats_double 
        Model_args_as_Rcpp_List$Model_args_mats_int 
        Model_args_as_Rcpp_List$Model_args_vecs_of_col_vecs_int 
        Model_args_as_Rcpp_List$Model_args_vecs_of_mats_double 
        Model_args_as_Rcpp_List$Model_args_vecs_of_mats_int
        str(Model_args_as_Rcpp_List$Model_args_2_later_vecs_of_mats_double)
      
        
        
        n_chains_sampling <- 8
        {
          
          
                seed <- 123
               
                # set.seed(seed)
                # require(dqrng)
                # ## Set dqrng type:
                # ## dqrng_type <- "pcg64"
                # dqrng_type <- "Xoshiro256+"
                # ##dqrng_type <- "Xoshiro256++"
                # dqrng::dqRNGkind(dqrng_type)
                # ## Set dqrng seed:
                # dqrng::dqset.seed(seed)
                      
                RcppParallel::setThreadOptions(numThreads = n_chains_sampling);
                
                Model_args_as_Rcpp_List$n_chains_sampling <-   n_chains_sampling
                
                result <- BayesMVP:::Rcpp_fn_RcppParallel_EHMC_sampling(    n_threads_R = n_chains_sampling,
                                                                            sample_nuisance_R = TRUE,
                                                                            n_nuisance_to_track = n_nuisance,
                                                                            seed_R = seed,
                                                                            iter_one_by_one = FALSE,
                                                                            n_iter_R = 100,
                                                                            partitioned_HMC_R = FALSE,
                                                                            Model_type_R = Model_type,
                                                                            force_autodiff_R = FALSE,
                                                                            force_PartialLog = TRUE,
                                                                            multi_attempts_R = FALSE,
                                                                            theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R, # inits stored here
                                                                            theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R,  # inits stored here
                                                                            y =  y,  ## only used in C++ for manual models! (all data passed via Stan_data_list / JSON strings for .stan models!) 
                                                                            Model_args_as_Rcpp_List =  Model_args_as_Rcpp_List,
                                                                            EHMC_args_as_Rcpp_List =   EHMC_args_as_Rcpp_List,
                                                                            EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List)
              
              
              
         
              {
                  trace <- result[[1]]
                  
                  print(sum(unlist(trace)))
                 ##  print(sum(trace[[2]][1, ]))
              }
              
        }
        
        
        
        >         sum(trace[[1]][1, ])
        [1] 91.43257
        >         sum(trace[[2]][1, ])
        [1] -14.06654
        
        
        >         sum(trace[[1]][1, ])
        [1] 91.43257
        >         sum(trace[[2]][1, ])
        [1] -14.06654
    
    
    
    
    
    
    

  
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
  # then compute tail-ESS using posterior::ess_tail:
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
  
  ### You can also extract the log_lik trace (note: for Stan models this will only work )
  log_lik_trace <- model_fit$get_log_lik_trace()
  str(log_lik_trace) # will be NULL unless you specify  "save_log_lik_trace = TRUE" in the "$summary()" method
  ## can then use log_lik_trace e.g. to compute LOO-IC using the loo package 
  
  
  
  
  
  
  