
 



{
  
  # Set working direcory ---------------
  try({  setwd("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/examples")   }, silent = TRUE)
  try({  setwd("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/examples")    }, silent = TRUE)
  #  options(repos = c(CRAN = "http://cran.rstudio.com"))
  
  # options ------------------------------------------------------------------- 
  #  totalCores = 8
  rstan::rstan_options(auto_write = TRUE)
  options(scipen = 99999)
  options(max.print = 1000000000)
  #  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
  
}




## Run LC-MVP model (manual-gradients using SNAPER-diffusion-space HMC)   --------------------------------------------------------------------------------------------------------------------------------------------------



pkg_dir <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP"

## first specify model type (for "Stan" models see other Stan example file)
Model_type <- "Stan_w_manual_grad"

# Source file that generates data 
source(file.path(pkg_dir, "examples/BayesMVP_LC_MVP_prep.R"))
 
## - | ----------  prepare Stan data and inits --------------------------------------------------------------------


## ### source(file.path(pkg_dir, "examples/Stan_LC_MVP_prepare_data.R"))


Stan_Model_Type <- "latent_trait"


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
  
  n_tests <- ncol(y)
  
  n_chunks_target <-  BayesMVP:::find_num_chunks_MVP(N, n_tests)
  
  ## Set important variables
  n_tests <- ncol(y)
  n_class <- 2
  n_covariates <- 1
  n_nuisance <- N * n_tests
  
  # if (Stan_Model_Type == "latent_trait") {
    n_params_main <- n_tests * 2 + 1 + n_tests * n_covariates * 2
  # } else if (Stan_Model_Type == "LC_MVP") {
  #   n_params_main <-  choose(n_tests, 2) * 2 + 1 + n_tests * n_covariates * 2
  # }
  
  ###  model_type <- "LC_MVP"
  fully_vectorised <- 1  ;   GHK_comp_method <- 2 ;       handle_numerical_issues <- 1 ;     overflow_threshold <- +5 ;    underflow_threshold <- -5  #  MAIN MODEL SETTINGS 
  
   # Phi_type <- 2 # using Phi_approx and inv_Phi_approx - in  Stan these are slower than Phi() and inv_Phi() !!!!!!!! (as log/exp are slow)
  Phi_type <- 1 # using Phi and inv_Phi
  
  corr_param <- "Sean" ; prior_lkj <- c(12, 3) ;  corr_prior_beta =  0 ;  corr_force_positive <-  0  ;  prior_param_3 <- 0 ;  
  #  corr_param <- "Sean" ; prior_lkj <- c(10, 2) ;  corr_prior_beta =  0 ;  corr_force_positive <-  0  ;  prior_param_3 <- 0 ;  
  
  CI <- 0
  prior_only <-  0
  corr_prior_norm = 0

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
  
  n_covariates_max <- 1
  n_covariates_max_nd <- 1
  n_covariates_max_d <- 1
  
  X_nd <- list()
  for(t in 1:n_tests) {
    X_nd[[t]] <- array(1, dim = c(N, n_covariates_max_nd))
  }
  
  X_d <- list()
  for(t in 1:n_tests) {
    X_d[[t]] <- array(1, dim = c(N, n_covariates_max_d))
  }
  
  X <- list(X_nd, X_d)
  str(X)
  
  n_params_main
  lb_corr <- list()
  ub_corr <- list()
  known_values <- list()
  known_values_indicator <- list()
  
  for (c in 1:n_class) { 
    lb_corr[[c]] <- matrix(-1, nrow = n_tests, ncol = n_tests)
    ub_corr[[c]] <- matrix(+1, nrow = n_tests, ncol = n_tests)
    known_values[[c]] <- matrix(0, nrow = n_tests, ncol = n_tests)
    known_values_indicator[[c]] <- matrix(0, nrow = n_tests, ncol = n_tests)
  }
  
  
  #### For latent_trait:
  LT_b_priors_shape <- array(1, dim = c(n_class, n_tests))
  LT_b_priors_scale <- array(1, dim = c(n_class, n_tests))
  LT_known_bs_indicator <-  LT_known_bs_values <-  array(0, dim = c(n_class, n_tests))
  
  
  {
    
    
    ## convert some of the 3D arrays to lists of matrices 
    prior_a_mean_Stan <- prior_a_sd_Stan <-  list()
    prior_a_mean_Stan[[1]] <- t(matrix(prior_a_mean[1,,]))
    prior_a_mean_Stan[[2]] <- t(matrix(prior_a_mean[2,,]))
    prior_a_sd_Stan[[1]] <- t(matrix(prior_a_sd[1,,]))
    prior_a_sd_Stan[[2]] <- t(matrix(prior_a_sd[2,,]))
    
    # if (prior_only == 0 ) { 
    stan_data = list(  N =   (N), # length(corr_prior_dist_LKJ_4), #  N, 
                       n_tests = n_tests,
                       y = (y_master_list_seed_123_datasets[[dataset_index]]),
                       n_class = 2,
                       n_pops = 1, #  n_pops,
                       pop = group,
                       # n_covariates_max_nd = n_covariates_max_nd,
                       # n_covariates_max_d = n_covariates_max_d,
                       n_covariates_max = n_covariates_max,
                      # X = X,
                       n_covs_per_outcome = n_covs_per_outcome, 
                       corr_force_positive = corr_force_positive,
                       known_num = 0,
                       overflow_threshold = overflow_threshold,
                       underflow_threshold = underflow_threshold,
                       prior_only = prior_only,  
                       prior_beta_mean =   prior_a_mean_Stan ,
                       prior_beta_sd  =   prior_a_sd_Stan ,
                       lkj_cholesky_eta = prior_lkj, 
                       prior_p_alpha = array(5),
                       prior_p_beta = array(10),
                       n_params_main = n_params_main,
                       n_chunks_target = n_chunks_target,
                       lb_corr = lb_corr,
                       ub_corr = ub_corr,
                       known_values = known_values,
                       known_values_indicator = known_values_indicator,
                       priors_via_Stan = 0,
                       multi_attempts_int = 0,
                       force_autodiff_int = 1,
                       force_PartialLog_int = 1,
                       Model_type_int = 3, # 1 for MVP, 2 for LC_MVP, and 3 for latent_trait 
                       LT_b_priors_shape = LT_b_priors_shape,
                       LT_b_priors_scale = LT_b_priors_scale,
                       LT_known_bs_indicator = LT_known_bs_indicator,
                       LT_known_bs_values = LT_known_bs_values
                    )
    
  }
  
  ## model_obj$init_object$init_model_object$model_args_list$model_args_list$lkj_cholesky_eta
  
  
}
  
  
  


  ##  | ------  Initial values  -------------------------------------------------------------------------
  {
        
        u_raw <- array(0.01, dim = c(N, n_tests))
        
        k_choose_2 = 0.5 * (n_tests - 1) * (n_tests - 0)
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
        
        theta_main_vec <- rep(NA, n_params_main)
 
        # if (Stan_Model_Type == "latent_trait") {
          
              theta_main_vec[1:(2*n_tests)] <- 0 # corrs inits
              theta_main_vec[(2*n_tests + 1):(2*n_tests + n_tests)] <- rep(-1, n_tests) # intercepts in D- class inits
              theta_main_vec[(2*n_tests + n_tests + 1):(4*n_tests)] <- rep(+1, n_tests) # intercepts in D+ class inits
              theta_main_vec[n_params_main] <-  2*atanh(0.20) - 1 #  -0.6931472 # raw prev init
          
        # } else if (Stan_Model_Type == "LC_MVP") {
        #   
        #       theta_main_vec[1:(2*k_choose_2)] <- rep(0.01, 2*k_choose_2) # corrs inits
        #       theta_main_vec[(2*k_choose_2 + 1):(2*k_choose_2 + n_tests)] <- rep(-1, n_tests) # intercepts in D- class inits
        #       theta_main_vec[(2*k_choose_2 + n_tests + 1):(2*k_choose_2 + 2*n_tests)] <- rep(+1, n_tests) # intercepts in D+ class inits
        #       theta_main_vec[n_params_main] <-  2*atanh(0.20) - 1 #  -0.6931472 # raw prev init
        #   
        # }
        
        

        init = list(
          theta_nuisance_vec = c(u_raw),
          theta_main_vec = theta_main_vec
        )
        
        Stan_init_list <- init
        
        n_chains <- 8
        init_lists_per_chain <- rep(list(Stan_init_list), n_chains) 
  
  }
  
  
  
  
  ## - | ----------  Compile Stan model --------------------------------------------------------------------
  {
    
    ##  /inst/BayesMVP/inst/stan_models/LC_MVP_bin_w_mnl_cpp_grad_v1.stan
    
    # OR using the version w/ fast math C++ functions:
    file <- (file.path(pkg_dir, "inst/BayesMVP/inst/stan_models/latent_trait_w_mnl_cpp_grad_v1.stan"))
    
    ## and then input the path to the corresponding C++ files:
    path_to_cpp_user_header <- file.path(pkg_dir, "inst/BayesMVP/src/latent_trait_lp_grad_fn_for_Stan.hpp")
    
    ## Set C++ flags (optional)
    ## CXX_COMPILER <- "/opt/AMD/aocc-compiler-5.0.0/bin/clang++"
    ## CPP_COMPILER <- "/opt/AMD/aocc-compiler-5.0.0/bin/clang"
    CXX_COMPILER <- "g++"
    CPP_COMPILER <- "gcc"
    # CXX_COMPILER <- "clang++"
    # CPP_COMPILER <- "clang"
    ##
    CCACHE <- "/usr/bin/ccache"
    CXX_STD <- "CXX17"
    CPU_BASE_FLAGS <- "-O3  -march=native  -mtune=native"
    FMA_FLAGS <- "-mfma"
    AVX_FLAGS <- "-mavx512f -mavx512vl -mavx512dq" ## for AVX-512
    ## AVX_FLAGS <- "-mavx -mavx2"
    CPU_FLAGS <- paste(CPU_BASE_FLAGS, FMA_FLAGS, AVX_FLAGS)
    MATH_FLAGS <- "-fno-math-errno  -fno-signed-zeros  -fno-trapping-math"
    ##
    ## THREAD_FLAGS <- "-pthread -D_REENTRANT -DSTAN_THREADS -DSTAN_MATH_REV_CORE_INIT_CHAINABLESTACK_HPP"
    THREAD_FLAGS <- "-pthread -D_REENTRANT"
    ##THREAD_FLAGS <- " "
    ##
    OTHER_FLAGS <- " "
    ## OTHER_FLAGS <- "-std=c++17"
    OTHER_FLAGS <- paste(OTHER_FLAGS, "-fPIC")  
    OTHER_FLAGS <- paste(OTHER_FLAGS, "-DNDEBUG -fpermissive ")  
    OTHER_FLAGS <- paste(OTHER_FLAGS, "-DBOOST_DISABLE_ASSERTS")
    OTHER_FLAGS <- paste(OTHER_FLAGS, "-Wno-sign-compare -Wno-ignored-attributes -Wno-class-memaccess -Wno-class-varargs") 
    # OTHER_FLAGS <- paste0(OTHER_FLAGHS, " -ftls-model=global-dynamic") # doesnt fix issue
    # OTHER_FLAGS <- paste(OTHER_FLAGS, "-fno-gnu-unique") # doesnt fix issue
    ##OTHER_FLAGS <- paste(OTHER_FLAGS, "-ftls-model=initial-exec")  # doesnt fix issue
     # OTHER_FLAGS <-  " "
    ##
    # LINKER_FLAGS <- " "
    ## LINKER_FLAGS <- "-lstdc++"
    # ##
    # OMP_LIB_PATH = "/opt/AMD/aocc-compiler-5.0.0/lib"
    # OMP_FLAGS = "-fopenmp"
    # OMP_LIB_FLAGS = "-L'OMP_LIB_PATH' -lomp 'OMP_LIB_PATH/libomp.so'"
    # SHLIB_OPENMP_CFLAGS <- OMP_FLAGS
    # SHLIB_OPENMP_CXXFLAGS <- SHLIB_OPENMP_CFLAGS
    # ##
    CMDSTAN_INCLUDE_PATHS <- "-I stan/lib/stan_math/lib/tbb_2020.3/include"
    CMDSTAN_INCLUDE_PATHS <- paste(CMDSTAN_INCLUDE_PATHS, "-I src")
    CMDSTAN_INCLUDE_PATHS <- paste(CMDSTAN_INCLUDE_PATHS, "-I stan/lib/rapidjson_1.1.0/")
    CMDSTAN_INCLUDE_PATHS <- paste(CMDSTAN_INCLUDE_PATHS, "-I lib/CLI11-1.9.1/")
    CMDSTAN_INCLUDE_PATHS <- paste(CMDSTAN_INCLUDE_PATHS, "-I stan/lib/stan_math/")
    CMDSTAN_INCLUDE_PATHS <- paste(CMDSTAN_INCLUDE_PATHS, "-I stan/lib/stan_math/lib/eigen_3.4.0")
    CMDSTAN_INCLUDE_PATHS <- paste(CMDSTAN_INCLUDE_PATHS, "-I stan/lib/stan_math/lib/boost_1.84.0")
    CMDSTAN_INCLUDE_PATHS <- paste(CMDSTAN_INCLUDE_PATHS, "-I stan/lib/stan_math/lib/sundials_6.1.1/include")
    CMDSTAN_INCLUDE_PATHS <- paste(CMDSTAN_INCLUDE_PATHS, "-I stan/lib/stan_math/lib/sundials_6.1.1/src/sundials")
    CMDSTAN_INCLUDE_PATHS <- paste(CMDSTAN_INCLUDE_PATHS, "-I stan/src")
    CMDSTAN_INCLUDE_PATHS <- paste(CMDSTAN_INCLUDE_PATHS, "-I stan/src")
    ##
    ##
    ## ----- Now set "BASE_FLAGS" --------------------
    BASE_FLAGS <- paste(CPU_FLAGS, 
                        ## `SHLIB_OPENMP_CFLAGS`,
                        MATH_FLAGS, 
                        OTHER_FLAGS, 
                        THREAD_FLAGS, 
                        CMDSTAN_INCLUDE_PATHS)
    
    ## Now set standard C++/C flags. 
    CC <-  paste(CCACHE, CPP_COMPILER)
    CXX <- paste(CCACHE, CXX_COMPILER)
    PKG_CPPFLAGS <- BASE_FLAGS
    PKG_CXXFLAGS <- BASE_FLAGS
    CPPFLAGS <- BASE_FLAGS
    CXXFLAGS <- BASE_FLAGS
    CFLAGS <- CPPFLAGS
    
    ## Linking flags to match Stan's
    LINKER_FLAGS <- paste(
      "-Wl,-L,\"$(CMDSTAN_PATH)/stan/lib/stan_math/lib/tbb\"",
      "-Wl,-rpath,\"$(CMDSTAN_PATH)/stan/lib/stan_math/lib/tbb\"",
      ##  "-lpthread",
      "-ltbb"
    )
    

    
    
    cmdstan_cpp_flags <- list(  paste0("CC = ", CC),
                                paste0("CXX = ", CXX),
                                paste0("PKG_CPPFLAGS = ", PKG_CPPFLAGS),
                                paste0("PKG_CXXFLAGS = ", PKG_CXXFLAGS),
                                paste0("CPPFLAGS = ", CPPFLAGS),
                                paste0("CXXFLAGS = ", CXXFLAGS),
                                paste0("CFLAGS = ", CFLAGS), 
                                paste0("CXX_STD = ", CXX_STD), 
                                paste0("LDFLAGS = ", LINKER_FLAGS))
    
      print(BASE_FLAGS)
    print(cmdstan_cpp_flags)
    
    mod <- cmdstan_model(file, 
                         force_recompile = TRUE,
                         quiet = FALSE,
                         user_header = path_to_cpp_user_header,
                         cpp_options = cmdstan_cpp_flags
                         )
    
    }


  


  ## | ------  Run model - using Stan  -------------------------------------------------------------------------
  {
    
    stan_data$multi_attempts_int   <- 0 
    stan_data$force_autodiff_int   <- 1
    stan_data$force_PartialLog_int <- 1
    
        n_iter <- 500
    
        tictoc::tic()
          
        Stan_mod_sample <- mod$sample( data = stan_data,
                                       init =   init_lists_per_chain, 
                                       chains = n_chains,
                                       parallel_chains = n_chains, 
                                       refresh = 50,
                                       iter_sampling = 500,
                                       iter_warmup = n_iter,
                                       max_treedepth = 10, 
                                       metric = "diag_e")
        
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
  
  
  
  

   
   Stan_model_summary <- Stan_mod_sample$summary(variables = c("Se_bin", "Sp_bin", "prev_var_matrix"))
   Stan_model_summary
   
  #  Stan_model_summary <- Stan_mod_sample$summary(variables = c("beta"))
  
  ## Stan_model_summary$mean
 
  ### results summary
  #Stan_model_results <- Stan_mod_sample$summary()
  #print(Stan_model_results, n = 100)
  
  
  ### some quick efficiency stats 
  {
    
    Stan_model_summary_main <- Stan_mod_sample$summary("theta_main_vec")
    
    Stan_sampler_diagnostics <- as.array(Stan_mod_sample$sampler_diagnostics())
    Stan_treedepth_array <- Stan_sampler_diagnostics[,,1]
    Stan_avg_treedepth_during_sampling <- sum(Stan_treedepth_array) / length(Stan_treedepth_array)
    Stan_avg_L_during_sampling <- 2^Stan_avg_treedepth_during_sampling
    Stan_n_grad_evals_sampling <- n_chains * n_iter * Stan_avg_L_during_sampling
    
    Stan_times_per_chain <- Stan_mod_sample$time()
    time_sampling <- max(Stan_times_per_chain$chains$sampling)
    
    Stan_min_ESS <- min(round(Stan_model_summary_main$ess_bulk))
    
    Stan_min_ESS_per_sec <-  Stan_min_ESS / total_time ; Stan_min_ESS_per_sec
    Stan_min_ESS_per_sec_sampling <-  Stan_min_ESS / time_sampling ; Stan_min_ESS_per_sec_sampling
    Stan_min_ESS_per_grad_sampling <-   ( Stan_min_ESS / Stan_n_grad_evals_sampling )
    Stan_min_ESS_per_grad_sampling_x_1000 <- 1000 * Stan_min_ESS_per_grad_sampling
    Stan_grad_evals_per_sec <- Stan_min_ESS_per_sec_sampling /  Stan_min_ESS_per_grad_sampling 
    Stan_grad_evals_per_sec_div_1000 <- Stan_grad_evals_per_sec / 1000
    
    print(paste("Average L (sampling) = ", round(Stan_avg_L_during_sampling)))
    print(paste("Min ESS/sec (overall) = ", signif(Stan_min_ESS_per_sec, 3)))
    print(paste("Min ESS/sec (sampling) = ", signif(Stan_min_ESS_per_sec_sampling, 3)))
    print(paste("Min ESS/grad (sampling) = ", signif(Stan_min_ESS_per_grad_sampling_x_1000, 3)))
    print(paste("Grad evals / sec = ", signif(Stan_grad_evals_per_sec_div_1000, 3)))
    
  }
  
  
  
  # # expose function to R to test if it works at all
  # mod$expose_functions()
  
  
  
 






 













    
    
    
    
    
  
  
  