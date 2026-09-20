

 

{
  
  
  ## Load in dependencies needed for BayesMVP (and other)
  require(Rcpp)
  require(RcppEigen)
  require(RcppParallel)
  require(RcppZiggurat)
  require(RcppClock)
  require(BH) 
  require(devtools)
  require(githubinstall)
  # require(sitmo)  # maybe need this
  # require(dqrng) # maybe need this
  require(beepr)
  require(plyr)
  require(dplyr)
  require(ggplot2)
  # require(patchwork)
  require(tictoc) 
  require(remotes)  
  require(cmdstanr) 
  require(jsonlite)
  require(expm)
  require(pracma)
  require(R6)
  require(roxygen2)
  ##
  ## Now load in BayesMVP 
  require(BayesMVP)
  ##
  require(Rfast)   # maybe need this
  require(LaplacesDemon)  # maybe need this
  require(TruncatedNormal)  # maybe need this
  require(rockchalk)   # maybe need this

  # remotes::install_github("https://github.com/roualdes/bridgestan", subdir="R")
  #### Load bridgestan:
  require(bridgestan)
  # # # # # # Install the cmdstanr package from CRAN
  # # we recommend running this in a fresh R session or restarting your current session
  # install.packages("cmdstanr", repos = c('https://stan-dev.r-universe.dev', getOption("repos")))  ## latest BETA release
  # remotes::install_github("stan-dev/cmdstanr") ## Latest DEVELOPMENT version (i.e. may be less stable than beta release)
  #  remotes::install_github("stan-dev/cmdstanr",force = TRUE)
  # # Load cmdstanr package
   library(cmdstanr)
  # # 
  # # # # Install the latest version of CmdStan
  # install_cmdstan(cores = 16, overwrite = TRUE, cpp_options = list("STAN_MODEL_LDFLAGS" = "-shared",
  #                                              #  "LDFLAGS" = "-shared" ,
  #                                              "PKG_CPPFLAGS" =  BASE_FLAGS,
  #                                              "PKG_CXXFLAGS" =  BASE_FLAGS,
  #                                              "CPPFLAGS" =  BASE_FLAGS,
  #                                              "CXXFLAGS" =  BASE_FLAGS))
  # "CXX" = "/opt/AMD/aocc-compiler-4.2.0/bin/clang++") )
  # #               
  # # 

  # install.packages("StanHeaders")
  
}
 

 
##
BASE_FLAGS <- "-O3  -march=native  -mtune=native -fPIC -mfma -mavx -mavx2   -mavx512vl -mavx512dq  -mavx512f -fno-math-errno  -fno-signed-zeros -fno-trapping-math -fPIC -DNDEBUG -fpermissive  -DBOOST_DISABLE_ASSERTS -Wno-deprecated-declarations -Wno-sign-compare -Wno-ignored-attributes -Wno-class-memaccess -Wno-class-varargs"

Sys.setenv(PKG_CPPFLAGS = BASE_FLAGS)
Sys.setenv(PKG_CXXFLAGS = BASE_FLAGS)
Sys.setenv(CPPFLAGS = BASE_FLAGS)
# Sys.setenv(CXXFLAGS = BASE_FLAGS)
Sys.unsetenv("PKG_CPPFLAGS")
Sys.unsetenv("PKG_CXXFLAGS")
Sys.unsetenv("CPPFLAGS")
Sys.unsetenv("CXXFLAGS")


 
 
BayesMVP



require(usethis)



install.packages("devtools")
install.packages("githubinstall")



# Create a list of your installed packages
pkgs <- installed.packages()[,"Package"]
# Save this list to a file
write(pkgs, "my_r_packages.txt")
# Then reinstall essential packages
install.packages(c("rlang", "usethis", "devtools"))


library(rlang)  # Load the new version first
library(usethis)
library(devtools)
