

# cmdstanr_path <- function() {
#   
#   suppressMessages({
# 
#            ##  require(cmdstanr)
#   
#   
#             # First attempt just using the built-in cmdstanr fn
#             try({
#               cmdstan_dir_using_cmdstanr <- cmdstanr:::cmdstan_path()
#                 if (!(cmdstan_dir_using_cmdstanr %in% c("", " ", "  "))) {
#                   message("Using CMDSTAN path: ", cmdstan_env)
#                   return(cmdstan_dir_using_cmdstanr) # Use the value from the environment variable
#                 }
#             }, silent = TRUE)
#  
#             # Get the user's home directory
#             home_dir <- Sys.getenv(if (.Platform$OS.type == "windows") "USERPROFILE" else "HOME")
# 
#             # Check for .cmdstan directory
#             cmdstan_dirs <- Sys.glob(file.path(home_dir, ".cmdstan", "cmdstan-*"))
# 
#             if (length(cmdstan_dirs) > 0) {
#               # Sort directories by version (assumes lexicographical sorting works for version strings)
#               recent_dir <- cmdstan_dirs[order(cmdstan_dirs, decreasing = TRUE)][1]
#               message("Found latest cmdstan in .cmdstan: ", recent_dir)
#               return(recent_dir)
#             }
# 
#             # Check for cmdstan directory directly under HOME
#             cmdstan_dir <- file.path(home_dir, "cmdstan")
#             if (dir.exists(cmdstan_dir)) {
#               message("Found cmdstan in home directory: ", cmdstan_dir)
#               return(cmdstan_dir)
#             }
# 
#             # If no valid path is found
#             stop("CmdStan directory not found. Please install CmdStan or set the CMDSTAN environment variable.")
#             
#   })
# 
# }




#' cmdstanr_path
#' @export
cmdstanr_path <- function(preferred_version = "2.36.0") {
  
        suppressMessages({
          
            # Get the user's home directory
            home_dir <- Sys.getenv(if (.Platform$OS.type == "windows") "USERPROFILE" else "HOME")
            
            # Check for preferred version FIRST
            preferred_path <- file.path(home_dir, ".cmdstan", paste0("cmdstan-", preferred_version))
            if (dir.exists(preferred_path)) {
              message("Using preferred cmdstan version: ", preferred_path)
              return(preferred_path)
            }
            
            # If preferred not found, try cmdstanr built-in
            try({
              cmdstan_dir_using_cmdstanr <- cmdstanr:::cmdstan_path()
              if (!(cmdstan_dir_using_cmdstanr %in% c("", " ", "  "))) {
                message("Using CMDSTAN path from cmdstanr: ", cmdstan_dir_using_cmdstanr)
                return(cmdstan_dir_using_cmdstanr)
              }
            }, silent = TRUE)
            
            # Check for .cmdstan directory - get most recent as fallback
            cmdstan_dirs <- Sys.glob(file.path(home_dir, ".cmdstan", "cmdstan-*"))
            if (length(cmdstan_dirs) > 0) {
              recent_dir <- cmdstan_dirs[order(cmdstan_dirs, decreasing = TRUE)][1]
              message("Found latest cmdstan in .cmdstan: ", recent_dir)
              return(recent_dir)
            }
            
            # Check for cmdstan directory directly under HOME
            cmdstan_dir <- file.path(home_dir, "cmdstan")
            if (dir.exists(cmdstan_dir)) {
              message("Found cmdstan in home directory: ", cmdstan_dir)
              return(cmdstan_dir)
            }
            
            # If no valid path is found
            stop("CmdStan directory not found. Please install CmdStan or set the CMDSTAN environment variable.")
          
        })
  
}




