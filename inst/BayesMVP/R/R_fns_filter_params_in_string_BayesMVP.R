





## ---------------------------------------------------------------------------------------------------

#' filtered_param_names_string
#' @export
filtered_param_names_string <-   function( all_param_names_vec, 
                                           param_string, 
                                           condition = "exact_match",
                                           debugging = FALSE) {
        if (condition == "exact_match") { 
          
          filtered_param_names_vec <- all_param_names_vec[grep(paste0("^", param_string, "($|\\[)"), all_param_names_vec)]
          
        } else if (condition == "containing") {
          
          filtered_param_names_vec <- all_param_names_vec[grep(param_string, all_param_names_vec, fixed = TRUE)]
          
        } else { 
          
          stop("ERROR: 'condition' argument must be set to either 'exact_match' or 'containing'")
          
        }
        
        if (length(filtered_param_names_vec) == 0) {
          warning(paste("No parameters found for condition:", condition, ", for the string:", param_string))
        }
        
        if (debugging) { 
          print(filtered_param_names_vec)
        }
        
        return(filtered_param_names_vec)
  
}







#' filter_param_names_string_batched
#' @export
filter_param_names_string_batched <- function(  all_param_names_vec, 
                                                param_string_vec,
                                                condition,
                                                debugging = FALSE) {
  
  
        # Convert single string to list if needed
        if (is.character(param_string_vec) && length(param_string_vec) == 1) {
          param_string_vec <- list(param_string_vec)
        }
        
        # Store in list for each parameter in "all_param_names_vec"
        filtered_param_names_list <- list()
        
        for (param_string_i in param_string_vec) {
          
          # if (debugging) { 
          #    cat("\nTrying param_string:", param_string, "\n")
          # }
          
          # #### pattern <- paste0("^", param_string, "\\[")  
          # pattern <- paste0("^", param_string, "($|\\[)")
          # if (debugging) cat("Using pattern:", pattern, "\n")
          
          params <- filtered_param_names_string( all_param_names_vec = all_param_names_vec,
                                                 param_string = param_string_i,
                                                 condition = condition, 
                                                 debugging = debugging)
          
          
          filtered_param_names_list[[param_string_i]] <- params
          
        }
        
        return(filtered_param_names_list)
  
}








#' filter_param_draws_string
#' @export
filter_param_draws_string <- function(  named_draws_array,
                                        param_string, 
                                        condition,
                                        exclude_NA, ## = FALSE,
                                        debugging = FALSE) {
  
        named_draws_array_2 <- format_named_array_for_bayesplot(named_draws_array)
        
        ## Get parameter names that contain the string ANYWHERE:
        all_param_names_vec <- dimnames(named_draws_array_2)[[3]] 
        ## [grep(param_string, dimnames(named_draws_array_2)[[3]], fixed = TRUE)]
        
        {
          draws_array_filtered <- named_draws_array_2
          try({  
            
            ## Filter parameter names (note this WILL include params which have "ALL NA's":
            param_names_filtered <- filtered_param_names_string(  all_param_names_vec = all_param_names_vec,
                                                                  param_string = param_string,
                                                                  condition = condition,
                                                                  debugging = debugging)
            
            ## Then filter/subset "named_draws_array_2":
            draws_array_filtered <- named_draws_array_2[,,param_names_filtered]
          })
        }
        
        ## Then exclude any draws which are ALL NA's (if chosen):
        if (!(exclude_NA)) { 
          
          if (debugging) { 
            print(draws_array_filtered)
          }
          
          return(draws_array_filtered)
          
        } else if (exclude_NA == TRUE) { 
          
          ## Filter out parameters that are all NA:
          param_names_filtered_wo_NAs <- character(0)
          for (param in param_names_filtered) {
            if (!all(is.na(draws_array_filtered[,,param]))) {
              param_names_filtered_wo_NAs <- c(param_names_filtered_wo_NAs, param)
            }
          }
          if (length(param_names_filtered_wo_NAs) == 0) {
            warning(paste("All parameters containing", param_string, "contain only NAs"))
            return(list())  # Return empty list if no valid parameters
          }
          draws_array_filtered_wo_NAs <- draws_array_filtered[,,param_names_filtered_wo_NAs]
          
          if (debugging) { 
            print(draws_array_filtered_wo_NAs)
          }
          
          return(draws_array_filtered_wo_NAs)
          
        }
  
}


# named_draws_array <- draws_array
# param_strings_vec <- c("beta", "raw_scale")
# 
# 
# 
# 
# 
# named_draws_array = named_draws_array
# param_strings_vec = grouped_params$bs_names_main
# condition = "exact_match"
# exclude_NA = exclude_NA
# debugging = TRUE


#' filter_param_draws_string_batched
#' @export
filter_param_draws_string_batched <- function(    named_draws_array,
                                                  param_strings_vec,
                                                  condition,
                                                  exclude_NA,## = FALSE,
                                                  debugging = FALSE) {
        
        param_draws_filtered_list <- list()
        
        for (param_string_i in param_strings_vec) {
          
          try({ 
            
            param_draws_filtered_list[[param_string_i]] <- filter_param_draws_string( named_draws_array = named_draws_array,
                                                                                      param_string = param_string_i, 
                                                                                      condition = condition,
                                                                                      exclude_NA = exclude_NA,
                                                                                      debugging = debugging)
          })
          
        }
        
        dimnames(named_draws_array)[[3]]
        # str(param_draws_filtered_list)
        
        if (is.list(param_draws_filtered_list)) {
          
          ## Turn list into one big 3D array:
          param_draws_filtered_array <- NULL
          try({ 
            param_draws_filtered_array <- convert_list_of_arrays_to_one_big_array( 
              draws_array_list = param_draws_filtered_list )
            param_draws_filtered_array    
          })
          
          # param_draws_filtered_list <- list()
          # for (i in 1:length(param_strings_vec)) {
          #     param_draws_filtered_list[[i]] <- named_draws_array[,,param_names_filtered_list[[i]]]
          # }
          # 
          # 
          # 
          # ## Turn list into one big 3D array:
          # param_draws_filtered_array <- convert_list_of_arrays_to_one_big_array(param_draws_filtered_list)
          # 
          return(list(param_draws_filtered_list = param_draws_filtered_list,
                      param_draws_filtered_array = param_draws_filtered_array))
          
        } else { 
          
          param_draws_filtered_array <- param_draws_filtered_list
          return(list(param_draws_filtered_array = param_draws_filtered_array))
          
        }
        
  
}








#' extract_params_from_tibble_batch
#' @export
extract_params_from_tibble_batch <- function(  debugging = FALSE,
                                               tibble, 
                                               param_strings_vec, 
                                               condition = "exact_match",
                                               combine = TRUE) {
        
        # Extract all parameter names from the tibble
        all_param_names_vec <- unique(tibble$parameter)
        
        # Use your existing batch function to filter parameter names
        filtered_params_list <- filter_param_names_string_batched(
          all_param_names_vec = all_param_names_vec,
          param_string_vec = param_strings_vec,
          condition = condition,
          debugging = debugging
        )
        
        # Filter tibble for each parameter string
        results_list <- list()
        for (param_string in names(filtered_params_list)) {
          if (length(filtered_params_list[[param_string]]) > 0) {
            results_list[[param_string]] <- dplyr::filter(
              tibble, 
              parameter %in% filtered_params_list[[param_string]]
            )
          }
        }
        
        # Return combined or as list
        if (combine && length(results_list) > 0) {
          combined_tibble <- dplyr::bind_rows(results_list, .id = "param_group")
          return(combined_tibble)
        } else {
          return(results_list)
        }
  
}








#' extract_params_from_array_batch
#' @export
extract_params_from_array_batch <- function(debugging,
                                            array_3d, 
                                            param_strings_vec, 
                                            condition,## = "exact_match",
                                            combine = TRUE) {
  
        # Check if input is a 3D array
        if (!is.array(array_3d) || length(dim(array_3d)) != 3) {
          stop("Input must be a 3D array")
        }
        
        # Extract all parameter names from the array
        all_param_names_vec <- dimnames(array_3d)[[3]]  # Third dimension contains parameter names
        
        if (is.null(all_param_names_vec)) {
          stop("Array must have named parameters in the third dimension")
        }
        
        # Use your existing batch function to filter parameter names
        filtered_params_list <- filter_param_names_string_batched(
          all_param_names_vec = all_param_names_vec,
          param_string_vec = param_strings_vec,
          condition = condition,
          debugging = debugging
        )
        
        # Extract array slices for each parameter string
        results_list <- list()
        
        for (param_string in names(filtered_params_list)) {
          param_indices <- which(all_param_names_vec %in% filtered_params_list[[param_string]])
          
          if (length(param_indices) > 0) {
            # Extract the subset of the array for these parameters
            # Keep all iterations and chains, subset parameters
            results_list[[param_string]] <- array_3d[, , param_indices, drop = FALSE]
            
            # Preserve dimension names
            dimnames(results_list[[param_string]]) <- list(
              iterations = dimnames(array_3d)[[1]],
              chains = dimnames(array_3d)[[2]],
              parameters = all_param_names_vec[param_indices]
            )
          }
        }
        
        # Return combined or as list
        if (combine && length(results_list) > 0) {
          # For combining arrays, we concatenate along the parameter dimension
          all_params <- unlist(lapply(results_list, function(x) dimnames(x)[[3]]))
          
          # Create combined array
          n_iter <- dim(array_3d)[1]
          n_chains <- dim(array_3d)[2]
          n_params_total <- length(all_params)
          
          combined_array <- array(
            data = NA,
            dim = c(n_iter, n_chains, n_params_total),
            dimnames = list(
              iterations = dimnames(array_3d)[[1]],
              chains = dimnames(array_3d)[[2]],
              parameters = all_params
            )
          )
          
          # Fill the combined array
          param_start <- 1
          for (param_string in names(results_list)) {
            n_params_group <- dim(results_list[[param_string]])[3]
            param_end <- param_start + n_params_group - 1
            combined_array[, , param_start:param_end] <- results_list[[param_string]]
            param_start <- param_end + 1
          }
          
          # Add attribute to track which parameters came from which group
          attr(combined_array, "param_groups") <- rep(names(results_list), 
                                                      sapply(results_list, function(x) dim(x)[3]))
          
          return(combined_array)
        } else {
          return(results_list)
        }
  
}






 
