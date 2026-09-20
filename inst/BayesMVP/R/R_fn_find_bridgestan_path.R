




#' bridgestan_path
#' @export
bridgestan_path <- function() {
  
  suppressMessages({
    # Get the user's home directory
    home_dir <- Sys.getenv(if (.Platform$OS.type == "windows") "USERPROFILE" else "HOME")
    
    # Find installed bridgestan version directories and prefer the NEWEST
    # (the old hardcoded 2.5.0 preference bundles stanc 2.35, which rejects
    # the "jacobian +=" idiom in *_jacobian functions that stanc 2.36+ supports).
    # This also overwrites a STALE BRIDGESTAN env value left over from earlier
    # runs of the old code, which otherwise pinned every new session to 2.5.0:
    search_pattern <- file.path(home_dir, ".bridgestan", "bridgestan-*")
    available_dirs <- Sys.glob(search_pattern)
    ##
    if (length(available_dirs) > 0) {
      version_strings <- sub( pattern = ".*bridgestan-",
                              replacement = "",
                              x = basename(available_dirs))
      ##
      recent_dir <- available_dirs[order( numeric_version(x = version_strings),
                                          decreasing = TRUE)][1]
      ##
      Sys.setenv(BRIDGESTAN = recent_dir)
      message(paste(("Bridgestan path found at:"), recent_dir))
      return(recent_dir)
    }
    ##
    ## ---- Fallback: the original default v2.5.0 location
    default_path <- file.path(home_dir, ".bridgestan", "bridgestan-2.5.0")
    ##
    if (dir.exists(default_path)) {
      Sys.setenv(BRIDGESTAN = default_path)
      message(paste(("Bridgestan path found at:"), default_path))
      return(default_path)
    }
    
    ## Check for plain "bridgestan" dir (i.e. w/o a ".")
    bridgestan_dir <- file.path(home_dir, "bridgestan")
    if (dir.exists(bridgestan_dir)) {
      Sys.setenv(BRIDGESTAN=bridgestan_dir)
      message(paste(("Bridgestan path found at:"), bridgestan_dir))
      return(bridgestan_dir)
    }
    
    # If no directory found, just return the BRIDGESTAN environment variable anyway
    # This will allow the Makevars file to show the correct error
    bridgestan_env <- Sys.getenv("BRIDGESTAN", "")
    message("BridgeStan directory not found. Using BRIDGESTAN environment variable: ", bridgestan_env)
    return(bridgestan_env)
  })
  
}
