





#' setup_env_post_install
#' @export
setup_env_post_install <- function() {
  
  
          # Set brigestan and cmdstanr environment variables / directories
          ## bs_dir <- bridgestan_path()
          ## cmdstan_dir <- cmdstanr_path()
          
       
          
          if (.Platform$OS.type == "windows") {
            mvp_user_dir <- file.path(Sys.getenv("USERPROFILE"), "BayesMVP")
          } else { 
            mvp_user_dir <- file.path(Sys.getenv("HOME"), "BayesMVP")
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







#' .onLoad
#' @export
.onLoad <- function(libname, 
                    pkgname) {
  
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
#' @export
.onAttach <- function(libname, 
                      pkgname) {

   setup_env_post_install()  
  
}


#' .First.lib
#' @export
.First.lib <- function(libname, 
                       pkgname) {
 
   setup_env_post_install()  
  
}





