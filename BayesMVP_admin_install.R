





# 
# # remotes::install_github("stan-dev/cmdstanr", force = TRUE)
# # #### Load cmdstanr R outer package:
# # require(cmdstanr) 
# # #### Then, install the latest version of CmdStan:
# # install_cmdstan(cores = parallel::detectCores(),
# #                 overwrite = TRUE,
# #                 cpp_options = list("STAN_MODEL_LDFLAGS" = "-shared",   "CXXFLAGS" = "-fPIC"))
# 
# 
# 
# # Install if needed
# # install.packages("pkgdown")
# # install.packages("Rdpack")
# require(pkgdown)
# require(Rdpack)
# 
# # # Build site with PDF capabilities
# # pkgdown::build_site(local_INNER_pkg_dir)




# ##
rm(list = ls())
# ##
.rs.restartR()  # In RStudio
# # ##
# assign(".Last.warning", NULL, envir = baseenv())
# ##
# # Force garbage collection
# gc(verbose = TRUE)
# ##
# rm(list = ls())
# 
# # This is a safer way to reset warnings
# options(warn = -1)  # Turn off warnings temporarily
# options(warn = 0)   # Turn them back on
# 
# 
# # Step 1: Detach your package if it's loaded
# pkg_name <- "yourPackageName"  # Replace with your package name
# if (pkg_name %in% (.packages())) {
#   detach(paste0("package:", pkg_name), unload = TRUE, force = TRUE)
# }
# 
# # Step 2: Unload the namespace
# if (pkg_name %in% loadedNamespaces()) {
#   unloadNamespace(pkg_name)
# }
# 
# # Step 3: Clear global environment objects
# rm(list = ls())
# 
# # Step 4: Force garbage collection
# gc(full = TRUE)
# 
# # Step 5: Reset warning behavior
# options(warn = -1)
# options(warn = 0)
# 
# # Step 6: Try cleaning DLLs again with your local package directory
# tools::clean_dll("path/to/your/package")
# 
# 
# warnings()
# 
# 
# 
#
##
{
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
}





if (.Platform$OS.type == "windows") {
  local_pkg_dir <-  "C:\\Users\\enzoc\\Documents\\Work\\PhD_work\\R_packages\\BayesMVP"
  local_INNER_pkg_dir <-  "C:\\Users\\enzoc\\Documents\\Work\\PhD_work\\R_packages\\BayesMVP\\inst\\BayesMVP"
} else {
  if (parallel::detectCores() > 16) {   ## if on local HPC
    local_pkg_dir <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP"  
    local_INNER_pkg_dir <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP"
  } else {  ## if on laptop
    local_pkg_dir <- "/home/enzo/Documents/Work/PhD_work/R_packages/BayesMVP"  
    local_INNER_pkg_dir <- "/home/enzo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP"
  }
}

# setwd(local_INNER_pkg_dir)
# 
# devtools::build_manual(path = local_INNER_pkg_dir)
# # 
# # 
# # 
# devtools::document(roclets = c("rd", "collate", "namespace"))
# unlink("*.pdf")
# system("R CMD Rd2pdf .")
# 
# devtools::build_manual()



# #### -------- INNER pkg stuff:
# ## Only works if do this first (on Linux)!! :
# source("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP/src/R_script_load_OMP_Linux.R")
# ## Document INNER package:
# devtools::clean_dll(local_INNER_pkg_dir)
# Rcpp::compileAttributes(local_INNER_pkg_dir)
# ### devtools::document(local_INNER_pkg_dir)
# roxygen2::roxygenize(local_INNER_pkg_dir)


#### -------- ACTUAL (LOCAL) INSTALL: 
## Document:
devtools::clean_dll(local_pkg_dir)
#### devtools::document(local_pkg_dir)
roxygen2::roxygenize(local_pkg_dir)
##
## Install (outer pkg):
devtools::clean_dll(local_pkg_dir)
devtools::install(local_pkg_dir, upgrade = "never")
##
## * MIGHT * have to restart session:
#### rstudioapi::restartSession()
##
##
{
  install_success <- FALSE
  
  try({  
    # ## Sometimes only works if do this first (on Linux) :
    # source("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP/src/R_script_load_OMP_Linux.R")
    # ##
    ## Install (inner pkg):
    require(BayesMVP)
    BayesMVP::install_BayesMVP() ## CUSTOM_FLAGS = CUSTOM_FLAGS)
    require(BayesMVP)
    install_success <- TRUE
  })
  
  try({  
    beepr::beep("ping")
  })
  
  rstudioapi::restartSession()
}







# ##### Using GitHub:
# ## Install (outer pkg):
# BayesMVP_repo_link <- "https://github.com/CerulloE1996/BayesMVP"
# remotes::install_github(repo = BayesMVP_repo_link, upgrade = "never")
# ###
# ## Install (inner pkg):
# require(BayesMVP)
# BayesMVP::install_BayesMVP()
# require(BayesMVP)






# CUSTOM_FLAGS <- list(  "CCACHE_PATH = ccache",
#                        "CXX_COMPILER = g++",
#                        "CPP_COMPILER = gcc", 
#                        "CXX_STD = CXX17",
#                        "CPU_BASE_FLAGS = -O3 -march=native -mtune=native",
#                        "FMA_FLAGS = -mfma",
#                        "AVX_FLAGS = -mavx -mavx2 -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl")
#                        # "OMP_FLAGS",
#                        # "OMP_LIB_PATH",
#                        # "OMP_LIB_FLAGS",
#                        # "PKG_CPPFLAGS",
#                        # "PKG_CXXFLAGS",
#                        # "CPPFLAGS",
#                        # "CXXFLAGS",
#                        # "PKG_LIBS")
# 
# 




 
 



# 
# 
# ### USER INSTALLATION PROCESS: --------------------------------------------------------------------------------------------
# 
# 
# #### ----------------- install cmdstanr first:
# 
# #### Install the cmdstanr "outer" R package:
# remotes::install_github("stan-dev/cmdstanr", force = TRUE)
# ### Load cmdstanr R outer package:
# require(cmdstanr) 
# #### Then, install the latest version of CmdStan:
# install_cmdstan(cores = parallel::detectCores(),
#                 overwrite = TRUE,
#                 cpp_options = list("STAN_MODEL_LDFLAGS" = "-shared",   "CXXFLAGS" = "-fPIC"))
# 
# 
# 
# 
# 
# 
# 
# #### ----------------- Then install bridgestan:
# remotes::install_github("https://github.com/roualdes/bridgestan", subdir="R")
# #### Load bridgestan:
# require(bridgestan)
# 
# 
# 
# 
# 
# #### ----------------- Then install BayesMVP:
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
# 
# 
# ## Install OUTER R package from GitHub:
# remotes::install_github("https://github.com/CerulloE1996/BayesMVP", force = TRUE, upgrade = "never")
# ## Then restart R session:
# rstudioapi::restartSession()
# ## Then install INNTER (i.e. the "real") package:
# require(BayesMVP)
# BayesMVP::install_BayesMVP()
# require(BayesMVP)
# 
# 
# ## Restart session:
# rstudioapi::restartSession()
# ## Install INNER R package:
# BayesMVP:::install_BayesMVP()
# ## Restart session:
# rstudioapi::restartSession()
# 
# 
# ###### other / testing
# 
# setwd(local_INNER_pkg_dir)
# Rcpp::sourceCpp(file.path(local_INNER_pkg_dir, "src", "cpu_check.cpp"))
# checkCPUFeatures()
# 
# 
# 








