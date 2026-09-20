# 
# 
# rm(list = ls())
# 
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
# 
# if (.Platform$OS.type == "windows") {
#   local_pkg_dir <- "C:/users/enzoc/Documents/Bay esMVP"   
# } else {
#   local_pkg_dir <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP"  
#   # local_pkg_dir <- "/home/enzo/Documents/Work/PhD_work/R_packages/BayesMVP"  
# }
# 
# # 
# # #### -------- INNER pkg stuff:
# # local_INNER_pkg_dir <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP"
# # ## Document INNER package:
# # devtools::clean_dll(local_INNER_pkg_dir)
# # Rcpp::compileAttributes(local_INNER_pkg_dir)
# # devtools::document(local_INNER_pkg_dir)
# 
# 
# #### -------- ACTUAL (LOCAL) INSTALL:
# ## Document:
# devtools::clean_dll(local_pkg_dir)
# devtools::document(local_pkg_dir)
# ## Install (outer pkg):
# devtools::clean_dll(local_pkg_dir)
# devtools::install(local_pkg_dir, upgrade = "never")
# ## Delte user dir just created (since need to test w/ "real" non-document install!):
# 
# 
# 
# if (.Platform$OS.type == "windows") {
#   user_home_dir <-  Sys.getenv("USERPROFILE") 
# } else { 
#   user_home_dir <-  Sys.getenv("HOME") 
# }
# 
# user_BayesMVP_dir <- file.path(user_home_dir, "BayesMVP")
# unlink(user_BayesMVP_dir, recursive = TRUE, force = TRUE)
# ## * MIGHT * have to restart session:
# rstudioapi::restartSession()
# ##
# ## Install (inner pkg):
# ##
# # require(BayesMVP)
# # BayesMVP::install_BayesMVP()
# # require(BayesMVP)
# # 
# # 
# # 
# # 
# # 
# # 

rm(list = ls())
.rs.restartR()


{
    ## First remove any possible package fragments:
    ## Find user_pkg_install_dir:
    user_pkg_install_dir <- Sys.getenv("R_LIBS_USER")
    print(paste("user_pkg_install_dir = ", user_pkg_install_dir))
    ##
    ## Find pkg_install_path + pkg_temp_install_path:
    ##
    pkg_install_path <- file.path(user_pkg_install_dir, "BayesMVP")
    pkg_temp_install_path <- file.path(user_pkg_install_dir, "00LOCK-BayesMVP") 
    ##
    ## Remove any (possible) BayesMVP package fragments:
    ##
    remove.packages("BayesMVP")
    unlink(pkg_install_path, recursive = TRUE, force = TRUE)
    unlink(pkg_temp_install_path, recursive = TRUE, force = TRUE)
    ##
    if (.Platform$OS.type == "windows") {
      local_pkg_dir <- "C:/users/enzoc/Documents/BayesMVP"   
    } else {
      local_pkg_dir <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP"
      # local_pkg_dir <- "/home/enzo/Documents/Work/PhD_work/R_packages/BayesMVP"  
    }
}
##
# # In your package directory
# setwd(local_pkg_dir)
# list.files("R/")
# list.files("src/")
# file.exists("NAMESPACE")
# file.exists("DESCRIPTION")
##
##
# pkgbuild::check_build_tools(debug = TRUE)
##
## ---- INNER pkg stuff:
##
try({
    local_INNER_pkg_dir <- "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP"
    ## Document INNER package:
    devtools::clean_dll(local_INNER_pkg_dir)
    Rcpp::compileAttributes(local_INNER_pkg_dir)
    devtools::document(local_INNER_pkg_dir)
})
##
# # In your package directory
# setwd(local_INNER_pkg_dir)
# list.files("R/")
# list.files("src/")
# file.exists("NAMESPACE")
# file.exists("DESCRIPTION")
# readLines("NAMESPACE")
##
# # This will show any syntax errors
# tools::package_native_routine_registration_skeleton(".", character_only = FALSE)
# # Check for lazy load database issues
# tools::checkRdaFiles("R/")
# # Try to manually create the lazy load database
# tools:::makeLazyLoading("BayesMVP", lib.loc = ".")
# # This bypasses .rdb creation
# require(RcppParallel)
# install.packages(".", repos = NUllllllLL, type = "source", 
#                  INSTALL_opts = "--no-lazy")
##
## ---- ACTUAL (LOCAL) INSTALL:
##
## ---- Document:
##
devtools::clean_dll(local_pkg_dir)
devtools::document(local_pkg_dir)
##
## ---- Install (outer pkg):
##
devtools::clean_dll(local_pkg_dir)
# devtools::install(local_pkg_dir, upgrade = "never")
# devtools::install(local_pkg_dir, upgrade = "never", repos = NULL)
devtools::install(local_pkg_dir, upgrade = "never", dependencies = FALSE)
##
## ---- * MIGHT * have to restart session:
##
.rs.restartR()
##
## ---- Install (inner pkg):
##
require(BayesMVP)
BayesMVP::install_BayesMVP()
require(BayesMVP)
  











