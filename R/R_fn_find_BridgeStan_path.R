


#' bridgestan_path
#' @export
bridgestan_path <- function() {

          # Check if the BRIDGESTAN environment variable is already set
          if (!(Sys.getenv("BRIDGESTAN") %in% c("", " ", "  "))) {
            message(paste(cat("Bridgestan path found at:"), Sys.getenv("BRIDGESTAN")))
            return(Sys.getenv("BRIDGESTAN")) # Use the value from the environment variable
          }

          # Get the user's home directory
          home_dir <- Sys.getenv(if (.Platform$OS.type == "windows") "USERPROFILE" else "HOME")

          # Define the default paths for BridgeStan v2.5.0
          default_path <- file.path(home_dir, ".bridgestan", "bridgestan-2.5.0")

          # Check if the default path exists
          if (dir.exists(default_path) == TRUE) {
            Sys.setenv(BRIDGESTAN=default_path)
            message(paste(cat("Bridgestan path found at:"), default_path))
            return(default_path)
          }


          # If v2.5.0 not available, then fallback to finding the most recent bridgestan directory
          search_pattern <- file.path(home_dir, ".bridgestan" , "bridgestan-*")
          available_dirs <- Sys.glob(search_pattern)
          
          # Filter for valid version directories and sort by version
          if (length(available_dirs) > 0) {
            recent_dir <- available_dirs[order(available_dirs, decreasing = TRUE)][1]
            Sys.setenv(BRIDGESTAN=recent_dir)
            message(paste(cat("Bridgestan path found at:"), recent_dir))
            return(recent_dir)
          }
          
          ## If nothing, then look for plain "bridgestan" dir (i.e. w/o a ".")
          search_pattern_wo_dot <- file.path(home_dir, "bridgestan")
          available_dir_wo_dot <- Sys.glob(search_pattern_wo_dot) ; available_dir_wo_dot
          return(available_dir_wo_dot)
 

          # If no valid path is found
          stop("BridgeStan directory not found.")

}


 

 
 
















