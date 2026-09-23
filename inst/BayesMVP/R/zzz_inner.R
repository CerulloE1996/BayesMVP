





#' setup_env_post_install
#' @keywords internal
#' @export
setup_env_post_install <- function() {
  
  
          # Set brigestan and cmdstanr environment variables / directories
          ## bs_dir <- bridgestan_path()
          ## cmdstan_dir <- cmdstanr_path()
          
          library(Rcpp)
          library(RcppParallel)
          
          if (.Platform$OS.type == "windows") {
            mvp_user_dir <- file.path(Sys.getenv("USERPROFILE"), "BayesMVP_development_2026_09_21")
          } else { 
            mvp_user_dir <- file.path(Sys.getenv("HOME"), "BayesMVP_development_2026_09_21")
          }
          
          
          if (.Platform$OS.type == "windows") {
            
                    TBB_STAN_DLL <- TBB_CMDSTAN_DLL <- DUMMY_MODEL_SO <- DUMMY_MODEL_DLL <- NULL
                    
                    cat("Setting up BayesMVP Environment for Windows:\n")
                    
                    cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                    try({   TBB_STAN_DLL <- file.path(mvp_user_dir, "tbb.dll") })
                    ## try({   TBB_CMDSTAN_DLL <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "tbb.dll") }) # prioritise user's installed tbb dll/so
                    try({   DUMMY_MODEL_SO <- file.path(mvp_user_dir, "dummy_stan_modeL_win_model.so") })
                    try({   DUMMY_MODEL_DLL <- file.path(mvp_user_dir, "dummy_stan_modeL_win_model.dll") })
                    
                    dll_paths <- c(TBB_STAN_DLL,
                                   ## TBB_CMDSTAN_DLL,
                                   DUMMY_MODEL_SO,
                                   DUMMY_MODEL_DLL)
                    
                    
                    # Attempt to load each DLL
                    for (dll in dll_paths) {
                      
                      tryCatch(
                        {
                          dyn.load(dll)
                          cat("  Loaded:", dll, "\n")
                        },
                        error = function(e) {
                          cat("  Failed to load:", dll, "\n  Error:", e$message, "\n")
                        }
                      )
                      
                    }
                    
                    
            
          }  else {  ### if Linux or Mac
            
                    # TBB_STAN_SO <- TBB_CMDSTAN_SO <- DUMMY_MODEL_SO <- NULL
                    # 
                    # cat("Setting up BayesMVP Environment for Linux / Mac OS:\n")
                    # 
                    # cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                    # try({  TBB_STAN_SO <- file.path(mvp_user_dir, "libtbb.so.2") })
                    # ## try({  TBB_CMDSTAN_SO <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "libtbb.so.2") })  # prioritise user's installed tbb dll/so
                    # try({  DUMMY_MODEL_SO <- file.path(mvp_user_dir, "dummy_stan_model_model.so") })
                    # 
                    # dll_paths <- c(TBB_STAN_SO,
                    #                ## TBB_CMDSTAN_SO,
                    #                DUMMY_MODEL_SO)
            
          }
  


          
          
  
}







#' R_fn_stop_if_several_TBB_copies_loaded
#' @keywords internal
R_fn_stop_if_several_TBB_copies_loaded <- function(package_name) {

      ##
      ## ---- One TBB runtime per R process (2026-09-22):
      ## If two different libtbb.so.2 files are mapped, the chain workers (RcppParallel::parallelFor) and a BridgeStan
      ## model's reduce_sum run on different TBB copies. RcppParallel::setThreadOptions() then limits only the first,
      ## and reduce_sum uses every CPU the process may use (measured on the 4-class LC-MVOP model: 35-47 busy threads
      ## for 4-16 requested). This happened whenever NicoStan / BayesMVP was loaded before RcppParallel: the package .so
      ## bound to CmdStan's libtbb.so.2, then RcppParallel's .onLoad loaded its own copy globally and the model bound to that.
      ## NAMESPACE now imports RcppParallel, so its TBB is loaded before this package's .so and only one copy exists.
      ## This check makes any other load path fail loudly. Linux only (reads /proc/self/maps).
      ##
      if (!file.exists("/proc/self/maps")) return(invisible(NULL))
      ##
      mapped_file_paths <- sub("^.* ", "", readLines("/proc/self/maps", warn = FALSE))
      TBB_library_paths <- unique(grep("/libtbb\\.so(\\.[0-9]+)*$", mapped_file_paths, value = TRUE))
      ##
      if (length(TBB_library_paths) > 1) {
        stop(paste0(package_name, ": ", length(TBB_library_paths), " different copies of the TBB library are loaded (",
                    paste(TBB_library_paths, collapse = ", "), "), so Stan models using reduce_sum would ignore the thread count. ",
                    "Restart R and load RcppParallel, NicoStan or BayesMVP before anything else that loads a TBB library."))
      }
      ##
      return(invisible(TBB_library_paths))

}




#' .onLoad
#' @keywords internal
#' @export
.onLoad <- function(libname, 
                    pkgname) {
  
      library(Rcpp)
      library(RcppParallel)
      ##
      R_fn_stop_if_several_TBB_copies_loaded(package_name = pkgname)
  
      is_windows <- .Platform$OS.type == "windows"
      
      dll_path <- file.path(libname, 
                            pkgname)
      
      try({  comment(print(paste(dll_path))) })
      try({  comment(print(dll_path)) })
      
      if (is_windows == TRUE) {
              
             ## try({ dyn.load(file.path(dll_path, "tbb.dll")) })
              try({ dyn.load(file.path(dll_path, "dummy_stan_model_win_model.so")) })
              try({ dyn.load(file.path(dll_path, "dummy_stan_model_win_model.dll")) })
              ## try({ dyn.load(file.path(dll_path, "R.dll")) })
              try({ dyn.load(file.path(dll_path, "BayesMVP.dll")) })
        
      } else { 
        
              # dyn.load(file.path(dll_path, "dummy_stan_model_model.so"))
              ##    setup_env_post_install() 
              ##   try({  .make_libs(libname, pkgname) }, silent = TRUE)
        
      }
      

  
}



#' .onAttach
#' @keywords internal
#' @export
.onAttach <- function(libname, 
                      pkgname) {
  
      library(Rcpp)
      library(RcppParallel)
  
      setup_env_post_install()  
  
}


#' .First.lib
#' @keywords internal
#' @export
.First.lib <- function(libname, 
                       pkgname) {
  
      library(Rcpp)
      library(RcppParallel)
   
      setup_env_post_install()  
  
}





