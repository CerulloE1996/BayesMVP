


#' setup_env_pre_install
#' @export
setup_env_pre_install <- function() {


        ## Set brigestan and cmdstanr environment variables / directories
        #### bs_dir <- bridgestan_path()
        cmdstan_dir <- cmdstanr_path()

        if (.Platform$OS.type == "windows") {

                  TBB_STAN_DLL <- TBB_CMDSTAN_DLL <- DUMMY_MODEL_SO <- DUMMY_MODEL_DLL <- NULL

                  cat("Setting up BayesMVP Environment for Windows:\n")

                  cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                  try({   TBB_STAN_DLL <- file.path(system.file(package = "BayesMVP"), "BayesMVP", "inst", "tbb_stan", "tbb.dll") })
                  try({   TBB_CMDSTAN_DLL <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "tbb.dll") }) # prioritise user's installed tbb dll/so
                  try({   DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"),  "BayesMVP", "inst", "dummy_stan_modeL_win_model.so") })
                  try({   DUMMY_MODEL_DLL <- file.path(system.file(package = "BayesMVP"),  "BayesMVP", "inst", "dummy_stan_modeL_win_model.dll") })

                  dll_paths <- c(TBB_STAN_DLL,
                                 TBB_CMDSTAN_DLL,
                                 DUMMY_MODEL_SO,
                                 DUMMY_MODEL_DLL)

        }  else {  ### if Linux or Mac

                  TBB_STAN_SO <- TBB_CMDSTAN_SO <- DUMMY_MODEL_SO <- NULL

                  cat("Setting up BayesMVP Environment for Linux / Mac OS:\n")

                  cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                  try({  TBB_STAN_SO <- file.path(system.file(package = "BayesMVP"),  "BayesMVP", "inst", "tbb_stan", "libtbb.so.2") })
                  try({  TBB_CMDSTAN_SO <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "libtbb.so.2") })  # prioritise user's installed tbb dll/so
                  try({  DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"),  "BayesMVP", "inst", "dummy_stan_model_model.so") })

                  dll_paths <- c(TBB_STAN_SO,
                                 TBB_CMDSTAN_SO,
                                 DUMMY_MODEL_SO)
      

        }

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


}





#' setup_env_post_install
#' @export
setup_env_post_install <- function() {


          # Set brigestan and cmdstanr environment variables / directories
          #### bs_dir <- bridgestan_path()
          cmdstan_dir <- cmdstanr_path()

          ## temp_dir <- Sys.getenv("TEMP")

          if (.Platform$OS.type == "windows") {

                    TBB_STAN_DLL <- TBB_CMDSTAN_DLL <- DUMMY_MODEL_SO <- DUMMY_MODEL_DLL <- NULL

                    cat("Setting up BayesMVP Environment for Windows:\n")

                    cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                    try({   TBB_STAN_DLL <- file.path(system.file(package = "BayesMVP"), "tbb_stan", "tbb.dll") })
                    try({   TBB_CMDSTAN_DLL <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "tbb.dll") }) # prioritise user's installed tbb dll/so
                    try({   DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"), "dummy_stan_modeL_win_model.so") })
                    try({   DUMMY_MODEL_DLL <- file.path(system.file(package = "BayesMVP"), "dummy_stan_modeL_win_model.dll") })

                    dll_paths <- c(TBB_STAN_DLL,
                                   TBB_CMDSTAN_DLL,
                                   DUMMY_MODEL_SO,
                                   DUMMY_MODEL_DLL)

          }  else {  ### if Linux or Mac

                    TBB_STAN_SO <- TBB_CMDSTAN_SO <- DUMMY_MODEL_SO <- NULL

                    cat("Setting up BayesMVP Environment for Linux / Mac OS:\n")

                    cat("Preloading critical .DLLs / .SOs for BayesMVP package\n")
                    try({  TBB_STAN_SO <- file.path(system.file(package = "BayesMVP"), "tbb_stan", "libtbb.so.2") })
                    try({  TBB_CMDSTAN_SO <- file.path(cmdstan_dir, "stan", "lib", "stan_math", "lib", "tbb", "libtbb.so.2") })  # prioritise user's installed tbb dll/so
                    try({  DUMMY_MODEL_SO <- file.path(system.file(package = "BayesMVP"), "dummy_stan_model_model.so") })

                    dll_paths <- c(TBB_STAN_SO,
                                   TBB_CMDSTAN_SO,
                                   DUMMY_MODEL_SO)

          }


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

}





#' install_BayesMVP
#' @export
#' @param CUSTOM_FLAGS An (optional) R list containing user-supplied C++ flags. The possible allowed flags one can supply to this function are the following:
#' * \code{CCACHE_PATH}: Note that this variable is passed onto the standard "CC" and "CXX" flags It is the path to your ccache file, if installed on your system. E.g., this may be ""/usr/bin/ccache" on 
#' Linux and "C:/rtools44/mingw64/bin/ccache.exe" on Windows. 
#' * \code{CXX_COMPILER}: Note that this variable is passed onto the standard "CXX" flag The C++ compiler to use. E.g. "g++" or "clang++" or "C:/rtools44/clang64/bin/clang++.exe", etc.
#' * \code{CPP_COMPILER}: Note that this variable is passed onto the standard "CC" flag The C compiler to use. E.g. "gcc" or "clang" or "C:/rtools44/clang64/bin/clang.exe", etc.
#' * \code{CXX_STD}: This is the standard "CXX_STD" macro, and it tells the compiler which C++ standard to use. Default is C++17 i.e. "CXX_STD = CXX17" or ""-std=gnu++17", depending on compiler used. 
#' * \code{CPU_BASE_FLAGS}: Note that this variable is passed onto the standard "PKG_CPPFLAGS", "PKG_CXXFLAGS", "CPPFLAGS", and "CXXFLAGS" flags It is the "base" C/C++ CPU flags to use. 
#'   The default is "CPU_BASE_FLAGS = -O3  -march=native  -mtune=native".
#' * \code{FMA_FLAGS}: Note that this variable is passed onto the standard "PKG_CPPFLAGS", "PKG_CXXFLAGS", "CPPFLAGS", and "CXXFLAGS" flags It specifies the FMA flags to use. 
#'   The default is "FMA_FLAGS = -mfma", if the CPU supports FMA instructions.
#' * \code{AVX_FLAGS}: Note that this variable is passed onto the standard "PKG_CPPFLAGS", "PKG_CXXFLAGS", "CPPFLAGS", and "CXXFLAGS" flags It specifies the AVX flags to use. 
#'   The default is: "AVX_FLAGS = "-mavx", if the CPU supports AVX," AVX_FLAGS = "-mavx -mavx2", if the CPU supports AVX-2, and "AVX_FLAGS = "-mavx -mavx2 -mavx512f -mavx512cd -mavx512bw 
#'   -mavx512dq -mavx512vl", if the CPU (fully) supports AVX-512. 
#' * \code{OMP_FLAGS}: Note that this variable is passed onto the standard "SHLIB_OPENMP_CFLAGS" and "SHLIB_OPENMP_CXXFLAGS" macros, and is then passed onto the standard "PKG_CPPFLAGS", "PKG_CXXFLAGS", "CPPFLAGS", and 
#'   "CXXFLAGS" flags It specifies the OpenMP flags to use (for parallel computations using the OpenMP backend). E.g.: "OMP_FLAGS = -fopenmp". Defaults depend on compiler and operating system.
#'   Also note that BayesMVP will search for OpenMP paths during package installation, and should be able to correctly determine the appropriate flags / libraries to use.
#' * \code{OMP_LIB_PATH}: Note that this variable is passed onto the standard "CC" flag 
#'   Also note that BayesMVP will search for OpenMP paths during package installation, and should be able to correctly determine the appropriate flags / libraries to use.
#' * \code{OMP_LIB_FLAGS}: Note that this variable is passed onto the standard "PKG_LIBS" flag 
#'   This variable specifies the OpenMP flags to use for the OpenMP library path (i.e. the \code{OMP_LIB_PATH} variable - see above). E.g., "OMP_LIB_FLAGS = -L"$(OMP_LIB_PATH)" -lomp".
#'   Also note that BayesMVP will search for OpenMP paths during package installation, and should be able to correctly determine the appropriate flags / libraries to use. 
#' * \code{PKG_CPPFLAGS}: A standard C++/C flag 
#' * \code{PKG_CXXFLAGS}: A standard C++/C flag 
#' * \code{CPPFLAGS}: A standard C++/C flag 
#' * \code{CXXFLAGS}: A standard C++/C flag 
#' * \code{PKG_LIBS}: A standard C++/C flag 
install_BayesMVP <- function(CUSTOM_FLAGS = NULL) {
        
     try({ (.make_user_dir(libname, pkgname)) })
  
     if (!is.null(CUSTOM_FLAGS)) {  ##  -----  Validate CUSTOM_FLAGS if provided
          
          if (!is.list(CUSTOM_FLAGS)) {   stop("CUSTOM_FLAGS must be a list")  }
          
          # Define allowed flag names
          allowed_flags <- c(
            "CCACHE_PATH",
            "CXX_COMPILER",
            "CPP_COMPILER", 
            "CXX_STD",
            "CPU_BASE_FLAGS",
            "FMA_FLAGS",
            "AVX_FLAGS",
            "OMP_FLAGS",
            "OMP_LIB_PATH",
            "OMP_LIB_FLAGS",
            "PKG_CPPFLAGS",
            "PKG_CXXFLAGS",
            "CPPFLAGS",
            "CXXFLAGS",
            "PKG_LIBS"
          )
          
          ## Append "USER_SUPPLIED_" at the start of the flags to be passed onto makevars/makrvars.win. 
          CUSTOM_FLAGS <- paste0("USER_SUPPLIED_", CUSTOM_FLAGS)
          
          # Validate flag names
          invalid_flags <- setdiff(names(CUSTOM_FLAGS), allowed_flags)
          if (length(invalid_flags) > 0) {
            stop("Invalid custom flags provided: ", paste(invalid_flags, collapse = ", "),
                 "\nAllowed flags are: ", paste(allowed_flags, collapse = ", "))
          }
          
          # Create temporary directory for modified package
          temp_dir <- tempfile("BayesMVP_temp_")
          dir.create(temp_dir)
          
          # Copy package files to temporary directory
          pkg_files <- list.files(system.file(package = "BayesMVP"), 
                                  recursive = TRUE, 
                                  full.names = TRUE)
          file.copy(pkg_files, temp_dir, recursive = TRUE)
          
          # Modify Makevars files to include custom flags
          makevars_files <- c(
            file.path(temp_dir, "src", "Makevars"),
            file.path(temp_dir, "src", "Makevars.win")
          )
          
          for (makevars_file in makevars_files) {
            if (file.exists(makevars_file)) {
              # Read existing content
              lines <- readLines(makevars_file)
              
              # Add custom flags at the beginning of the file
              custom_lines <- sapply(names(CUSTOM_FLAGS), function(flag) {
                sprintf('%s = "%s"', flag, CUSTOM_FLAGS[[flag]])
              })
              
              # Write modified content
              writeLines(c(custom_lines, lines), makevars_file)
            }
          }
          
        } else { ##   -------------- standard installation without custom user-supplied flags 
           
          ## do nothing
          
        }
  
  
          # pkg_dir <- system.file(package = "BayesMVP", "BayesMVP", "inst")
          # ## pkg_dir <- file.path(libname, pkgname, "inst", "BayesMVP", "inst")
          
          # mvp_user_dir <- file.path(Sys.getenv("USERPROFILE"), ".BayesMVP")
          # dir.create(mvp_user_dir, showWarnings = FALSE, recursive = TRUE)
          # 
          # ## Copy TBB:
          # tbb_file <- if (.Platform$OS.type == "windows") "tbb.dll" else "libtbb.so"
          # file.copy(
          #   from =  file.path(pkg_dir, "tbb_stan", tbb_file),
          #   to = file.path(mvp_user_dir, tbb_file), 
          #   overwrite = TRUE
          # )
          # 
          # ## Copy dummy model files (for bridgestan):
          # if (.Platform$OS.type == "windows") {
          #   
          #         # Copy dummy model SO
          #         file.copy(
          #           from = file.path(pkg_dir, "dummy_stan_model_win_model.so"),
          #           to = file.path(mvp_user_dir, "dummy_stan_model_win_model.so"),
          #           overwrite = TRUE
          #         )
          #         
          #         # Copy dummy model DLL
          #         file.copy(
          #           from = file.path(pkg_dir, "dummy_stan_model_win_model.dll"),
          #           to = file.path(mvp_user_dir, "dummy_stan_model_win_model.dll"),
          #           overwrite = TRUE
          #         )
          #   
          # } else { 
          #   
          #         # Copy dummy model SO
          #         file.copy(
          #           from = file.path(pkg_dir, "dummy_stan_model_model.so"),
          #           to = file.path(mvp_user_dir, "dummy_stan_model_model.so"),
          #           overwrite = TRUE
          #         )
          #         
          # }
  
  
        try({  setup_env_pre_install() })  # setup env (BEFORE removing wrapper pkg)
  
        # Get inner package from inst/ and copy to temp dir
        try({  
            inner_pkg <- system.file("BayesMVP", package = "BayesMVP")
            temp_dir <- file.path(tempdir(), paste0("BayesMVP_temp_", format(Sys.time(), "%H%M%S")))
            dir.create(temp_dir, recursive = TRUE)
            file.copy(inner_pkg, temp_dir, recursive = TRUE)
        })
        
        # Unload installer namespace
        try(unloadNamespace("BayesMVP"), silent = TRUE)
  
        # Uninstall current package
        try(outer_pkg <- system.file(package = "BayesMVP"), silent = TRUE)
        try(devtools::uninstall(outer_pkg), silent = TRUE)
        try(detach("package:BayesMVP", unload = TRUE), silent = TRUE)
        try(remove.packages("BayesMVP"), silent = TRUE)

  
        Sys.sleep(5) 
        
        
        # Remove lock directory
        lock_dir <- file.path(.libPaths()[1], "00LOCK-BayesMVP")
        if (dir.exists(lock_dir)) {
          unlink(lock_dir, recursive = TRUE, force = TRUE)
        }
        
        # Detach and unload the namespace
        if ("BayesMVP" %in% loadedNamespaces()) {
          try(unloadNamespace("BayesMVP"), silent = TRUE)
        }
        
        # Detach the package if loaded
        if ("package:BayesMVP" %in% search()) {
          try(detach("package:BayesMVP", unload = TRUE), silent = TRUE)
        }
        
        # Remove the installed package directory
        pkg_dir <- file.path(.libPaths()[1], "BayesMVP")
        if (dir.exists(pkg_dir)) {
          unlink(pkg_dir, recursive = TRUE, force = TRUE)
        }
        
        # Install fresh from temp copy
        options(devtools.install.args = c("--no-test-load", "--no-staged-install"))
        path <- file.path(temp_dir, "BayesMVP")
        ## Rcpp::compileAttributes(path)  
        devtools::install(path, 
                          args = c("--no-test-load", "--no-staged-install"),
                          dependencies = FALSE,  # Prevent namespace cycling
                          quiet = FALSE, 
                          force = TRUE
                         #   quick = TRUE
                          )
        
        
        try({   setup_env_post_install()  }) # setup env again before loading 
        
  
}























