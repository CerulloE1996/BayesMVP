


#' plot_multiple_params_batched
#' @keywords internal
#' @export
plot_param_group_batched <- function(draws_array, 
                                     param_prefix, 
                                     plot_type = c("density", "trace"),
                                     batch_size = 9) {

    
              plot_type <- match.arg(plot_type)
              
              # # # Get parameter names that start with the prefix
              # # param_names <- dimnames(draws_array)$parameters[grep(paste0("^", param_prefix, "\\["), dimnames(draws_array)$parameters)]
              # 
              # # Get parameter names that start with the prefix
              # ####  param_names <- dimnames(draws_array)[[3]][grep(paste0("^", param_prefix, "\\."), dimnames(draws_array)[[3]])]
              
               # # Get parameter names that start with the prefix
               # param_names <- dimnames(draws_array)[[3]][grep(paste0("^", param_prefix, "\\["), dimnames(draws_array)[[3]])]
              
               # Modified pattern to match both cases:
               # 1. Exact match of prefix (for parameters like "alpha")
               # 2. Prefix followed by [ (for parameters like "beta[1]")
              param_names <- dimnames(draws_array)[[3]][grep(paste0("^", param_prefix, "($|\\[)"), dimnames(draws_array)[[3]])]
               
              if (length(param_names) == 0) {
                    stop(paste("No parameters found starting with", param_prefix))
              }
              
              # Filter out parameters that are all NA
              valid_params <- character(0)
              for (param in param_names) {
                if (!all(is.na(draws_array[,,param]))) {
                  valid_params <- c(valid_params, param)
                }
              }
              
              if (length(valid_params) == 0) {
                warning(paste("All parameters starting with", param_prefix, "contain only NAs"))
                return(list())  # Return empty list if no valid parameters
              }
              
              
              
              # Choose plot function
              plot_func <- switch(plot_type,
                                  density = bayesplot::mcmc_dens,
                                  trace =   bayesplot::mcmc_trace)
              
              # Plot in batches
              n_batches <- ceiling(length(valid_params) / batch_size)
              
              plots <- list()
              for (i in 1:n_batches) {
                
                        start_idx <- (i-1) * batch_size + 1
                        end_idx <- min(i * batch_size, length(valid_params))
                        batch_params <- valid_params[start_idx:end_idx]
                        
                        ### plot 
                        plots[[i]] <- plot_func(draws_array[,,batch_params, drop=FALSE]) +
                                      ggplot2::ggtitle(paste0(param_prefix, " (batch ", i, " of ", n_batches, ")"))
                      
              }
              
              return(plots)
      
}








#' plot_multiple_params_batched
#' @keywords internal
#' @export
plot_multiple_params_batched <- function(draws_array, 
                                         param_prefixes, 
                                         plot_type = c("density", "trace"),
                                         batch_size = 9) {
  
  
              # # Debug prints
              # cat("\nDebugging plot_multiple_params_batched:\n")
              # cat("Array dimensions:", paste(dim(draws_array), collapse=" x "), "\n")
              # cat("Parameter names in array:", paste(dimnames(draws_array)[[3]], collapse=", "), "\n")
              # cat("Looking for prefix:", param_prefixes, "\n")
              
              # Get parameter names from array
              param_names <- dimnames(draws_array)[[3]]
  
              # Convert single string to list if needed
              if (is.character(param_prefixes) && length(param_prefixes) == 1) {
                param_prefixes <- list(param_prefixes)
              }
              
              # Store plots for each parameter group
              all_plots <- list()
              
              # Generate plots for each parameter prefix
              for (prefix in param_prefixes) {
                
                    cat("\nTrying prefix:", prefix, "\n")
                    #### pattern <- paste0("^", prefix, "\\[")  
                    pattern <- paste0("^", prefix, "($|\\[)")
                    cat("Using pattern:", pattern, "\n")
                    
                    matching_params <- grep(pattern, param_names, value = TRUE)
                    cat("Found matching params:", paste(matching_params, collapse=", "), "\n")
                    
                    if(length(matching_params) == 0) {
                      stop(sprintf("No parameters found starting with %s\nAvailable parameters: %s", 
                                   prefix, paste(param_names, collapse=", ")))
                    }
                    
                    
                        plots <- plot_param_group_batched( draws_array = draws_array, 
                                                                      param_prefix = prefix, 
                                                                      plot_type = plot_type, 
                                                                      batch_size = batch_size)
                        
                        all_plots[[prefix]] <- plots
                    
              }
              
              return(all_plots)
  
}






















