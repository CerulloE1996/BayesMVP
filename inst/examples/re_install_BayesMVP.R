
 


{
  
  # Set working direcory ---------------
  try({  setwd("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/examples")   }, silent = TRUE)
  try({  setwd("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/examples")    }, silent = TRUE)
  #  options(repos = c(CRAN = "http://cran.rstudio.com"))
  
  # options -------------------------------------------------------------------------
  #  totalCores = 8
  rstan::rstan_options(auto_write = TRUE)
  options(scipen = 999)
  options(max.print = 1000000000)
  #  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores() / 2)
  
}

 


rm(Rcpp_compute_MCMC_diagnostics,
   generate_summary_tibble,
   generate_summary_tibble_foreach,
   MVP_model,
   MVP_class_extract_and_plot,
   create_summary_and_traces,
   create_stan_summary, 
   create_superchain_ids, 
   MVP_class_summary_plot, 
   MVP_model)

detach("package:BayesMVP", unload = TRUE)
remove.packages("BayesMVP")

 
 ####  rm(list = ls())

## source("load_R_packages.R")

#  
# devtools::document()
# devtools::clean_dll( "~/Documents/Work/PhD_work/R_packages/BayesMVP")
# devtools::load_all( "~/Documents/Work/PhD_work/R_packages/BayesMVP")



devtools::clean_dll("~/Documents/Work/PhD_work/R_packages/BayesMVP")  # Clean compiled code
devtools::document("~/Documents/Work/PhD_work/R_packages/BayesMVP")   # Rebuild documentation
devtools::install("~/Documents/Work/PhD_work/R_packages/BayesMVP", clean = TRUE)  # Clean install



# Set the compiler first using  Sys.setenv(). 
# NOTE: If ccache not available/installed, please delete "ccache" below, or install ccache. 
# However, we strongly recommend installing ccache as it can greatly speed-up C++ compilation times.  

# For AMD CPU's we recommend downloading and using the AMD AOCC clang++ compiler.
# For other CPU's, use standard clang++ compiler (may have to download). 
# If clang++ not available, then use g++ (the default C++ compiler for R). 


# ## For standard clang++ use:
# CXX_COMPILER_TYPE <- "ccache   clang++"
# CPP_COMPILER_TYPE <- "ccache   clang" 
# BASE_FLAGS <- "-O3  -march=native  -mtune=native   -D_REENTRANT    -DSTAN_THREADS -pthread  -fPIC"
# 
# ## for g++ use:
# CXX_COMPILER_TYPE <- "ccache   g++"
# CPP_COMPILER_TYPE <- "ccache   g"
# BASE_FLAGS <- "-O3  -march=native  -mtune=native   -D_REENTRANT    -DSTAN_THREADS -pthread  -fPIC"


## for AMD AOCC clang++ use (must be downloaded separately and only for AMD CPU's):
CXX_COMPILER_TYPE <- "ccache   /opt/AMD/aocc-compiler-5.0.0/bin/clang++"
CPP_COMPILER_TYPE <- "ccache   /opt/AMD/aocc-compiler-5.0.0/bin/clang"
#BASE_FLAGS <- "-O3  -march=znver5  -mtune=znver5   -D_REENTRANT    -DSTAN_THREADS -pthread  -fPIC" # if your AMD CPU architecture is zen5
BASE_FLAGS <- "-O3  -march=znver4  -mtune=znver4   -D_REENTRANT    -DSTAN_THREADS -pthread  -fPIC" # if your AMD CPU architecture is zen4
#BASE_FLAGS <- "-O3  -march=znver3  -mtune=znver3   -D_REENTRANT    -DSTAN_THREADS -pthread  -fPIC" # if your AMD CPU architecture is zen3


 


Sys.setenv(CXX=CXX_COMPILER_TYPE,
           CC=CPP_COMPILER_TYPE,
           CXXFLAGS=BASE_FLAGS,
           CFLAGS=BASE_FLAGS)





rm(create_summary_and_traces,
   fn_Rcpp_wrapper_compute_main_Hessian_num_diff,
   fn_Rcpp_wrapper_update_M_dense_main_Hessian,
   fn_Rcpp_wrapper_update_tau_w_dense_M_ADAM,
   fn_compute_param_constrain_from_trace,
   fn_compute_param_constrain_from_trace_parallel,
   fn_find_initial_eps_us,
   fn_R_RcppParallel_EHMC_single_iter_burnin,
   fn_Rcpp_wrapper_adapt_eps_ADAM,
   fn_find_initial_eps_main,
   detect_vectorization_support,
   fn_2D_to_3D_array_Eigen,
   Rcpp_compute_MCMC_diagnostics,
   generate_summary_tibble,
   generate_summary_tibble_foreach,
   MVP_model,
   MVP_class_extract_and_plot,
   create_summary_and_traces,
   create_stan_summary, 
   create_superchain_ids, 
   MVP_class_summary_plot, 
   MVP_model)

detach("package:BayesMVP", unload = TRUE)
remove.packages("BayesMVP")




Rcpp::compileAttributes( "~/Documents/Work/PhD_work/R_packages/BayesMVP")
options(buildtools.check = function(action) TRUE )   #
devtools::install("~/Documents/Work/PhD_work/R_packages/BayesMVP") 

devtools::document("~/Documents/Work/PhD_work/R_packages/BayesMVP")   # Rebuild documentation

?MVP_model
?MVP_class_extract_and_plot



# 
# # Close all connections
# showConnections(all = TRUE)
# 
# open_conns <- showConnections(all = TRUE)
# n_connections <- nrow(open_conns)
# 
# n_connections_m_std <- n_connections - 3
# 
# for (conn in 4:(n_connections_m_std + 100)) {
#   try({  
#     close(getConnection(as.integer(conn)))
#   })
# }
# 
# # check connections 
# showConnections(all = TRUE)
# 
# 
# 








#### R CMD check ------------------------------------------------------------------------
rm(create_summary_and_traces,
   fn_Rcpp_wrapper_update_tau_w_diag_M_ADAM,
   fn_update_eigen_max_and_eigen_vec,
   fn_update_snaper_m_and_s,
   fn_Rcpp_wrapper_compute_main_Hessian_num_diff,
   fn_Rcpp_wrapper_update_M_dense_main_Hessian,
   fn_Rcpp_wrapper_update_tau_w_dense_M_ADAM,
   fn_compute_param_constrain_from_trace,
   fn_compute_param_constrain_from_trace_parallel,
   fn_find_initial_eps_us,
   fn_R_RcppParallel_EHMC_single_iter_burnin,
   fn_Rcpp_wrapper_adapt_eps_ADAM,
   fn_find_initial_eps_main,
   detect_vectorization_support,
   fn_2D_to_3D_array_Eigen,
   Rcpp_compute_MCMC_diagnostics,
   generate_summary_tibble,
   generate_summary_tibble_foreach,
   MVP_model,
   MVP_class_extract_and_plot,
   create_summary_and_traces,
   create_stan_summary, 
   create_superchain_ids, 
   MVP_class_summary_plot, 
   MVP_model)

rm(list = c("fn_update_snaper_w_dense_M", "fn_update_snaper_w_diag_M",
            "init_hard_coded_model", "init_inits", "init_model", "is_valid",
            "R_fn_EHMC_SNAPER_ADAM_burnin", "Rcpp_Chol", "Rcpp_det", "Rcpp_fn_openMP_EHMC_sampling",
            "Rcpp_fn_RcppParallel_EHMC_sampling", "Rcpp_log_det", "Rcpp_solve",
            "Rcpp_wrapper_EIGEN_double", "Rcpp_wrapper_fn_lp_grad", "sqrtm"))
rm(list = c("initialise_model", "init_hard_coded_model"))




# readLines("NAMESPACE")
# 
# file.copy("NAMESPACE", "NAMESPACE.backup")
# unlink("NAMESPACE") # Delete current NAMESPACE

getwd()  # Should show your package directory


try({  setwd("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/")   }, silent = TRUE)
try({  setwd("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/")    }, silent = TRUE)

# 1. Make sure all dependencies are listed in DESCRIPTION
# 2. Document everything that needs to be documented
devtools::document()

# 3. Build the package
devtools::build()

# 4. Run R CMD check
devtools::check()

# 5. Fix any warnings/errors
# Common fixes:
# - Add @import or @importFrom for functions you use
# - Complete all roxygen tags
# - Make sure examples work
# - Add proper version numbers in DESCRIPTION










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



















