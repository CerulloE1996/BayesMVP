






#' convert_stan_data_list_to_JSON
#' @export
convert_stan_data_list_to_JSON <- function(stan_data_list, 
                                           pkg_data_dir = NULL) {
  
        if (is.null(pkg_data_dir)) { 
          pkg_root_dir <- system.file(package = "BayesMVP")
          pkg_data_dir <- file.path(pkg_root_dir, "stan_data")  # directory to store data inc. JSON data files
        }
        ##
        ## Create data directory if it doesn't exist:
        ##
        if (!dir.exists(pkg_data_dir)) {
          dir.create(pkg_data_dir, recursive = TRUE)
        }
        ##
        ## make persistent (non-temp) JSON data file path with unique identifier:
        ##
        data_hash <- digest::digest(stan_data_list)  # Hash the data to create unique identifier
        json_filename <- paste0("data_", data_hash, ".json")
        json_file_path <- file.path(pkg_data_dir, json_filename)
        ##
        ## write JSON data using cmdstanr:
        ##
        if (length(stan_data_list) == 0) {
          
              ## ---- a model with NO data: cmdstanr::write_stan_json() writes an empty
              ## ARRAY ([]), but BridgeStan requires an empty OBJECT ({}):
              writeLines( text = "{}",
                          con = json_file_path)
              
        } else {
          
              cmdstanr::write_stan_json(data = stan_data_list, file = json_file_path)
              
        }
        ##
        ## Output the JSON file path:
        ##
        return(json_file_path)
  
}









#' convert_JSON_string_to_R_vector
#' @export
convert_JSON_string_to_R_vector <- function(json_string) {
  
        require(jsonlite)
        
        ## Helper function to flatten a matrix/array row by row
        flatten_matrix <- function(m) {
          if(is.null(dim(m))) {
            return(as.numeric(m))
          }
          return(as.numeric(t(matrix(unlist(m), ncol=ncol(m), byrow=TRUE))))
        }
        
        ## Helper function to recursively process nested structures
        process_element <- function(x) {
          if(is.numeric(x) && length(x) == 1) {
            return(x)
          }
          if(is.list(x) || is.matrix(x) || is.array(x)) {
            if(is.matrix(x) || (is.list(x) && all(sapply(x, length) == length(x[[1]])))) {
              return(flatten_matrix(x))
            }
            # Recursively process nested elements
            return(unlist(lapply(x, process_element)))
          }
          return(as.numeric(x))
        }
        
        ## Parse JSON
        data <- jsonlite::fromJSON(json_string)
        
        ## Process each top-level element in order of appearance
        result <- c()
        for(name in names(data)) {
          result <- c(result, process_element(data[[name]]))
        }
        
        return(result)
  
}









#' convert_JSON_string_to_ordered_R_vector
#' @export
convert_JSON_string_to_ordered_R_vector <- function(json_string, 
                                                    param_names_main_correct_order) {
  
        require(jsonlite)
        
        ## Helper function to flatten a matrix/array row by row
        flatten_matrix <- function(m) {
          if(is.null(dim(m))) {
            return(as.numeric(m))
          }
          return(as.numeric(t(matrix(unlist(m), ncol=ncol(m), byrow=TRUE))))
        }
        
        ## Helper function to recursively process nested structures:
        process_element <- function(x) {
          if(is.numeric(x) && length(x) == 1) {
            return(x)
          }
          if(is.list(x) || is.matrix(x) || is.array(x)) {
            if(is.matrix(x) || (is.list(x) && all(sapply(x, length) == length(x[[1]])))) {
              return(flatten_matrix(x))
            }
            # Recursively process nested elements
            return(unlist(lapply(x, process_element)))
          }
          return(as.numeric(x))
        }
        
        ## Parse JSON
        data <- jsonlite::fromJSON(json_string)
        
        ## Extract parameter root names (without indices) from the reference list
        root_names <- unique(sapply(strsplit(param_names_main_correct_order, "\\[|_"), function(x) x[1]))
        
        ## Process each parameter in the order specified by param_names_main_correct_order
        result <- numeric()
        
        for (root_name in root_names) {
          ## Find matching parameters in the JSON data
          matching_params <- grep(paste0("^", root_name), names(data), value = TRUE)
          
          ## Skip if no matching parameters found
          if(length(matching_params) == 0) {
            warning(paste("Parameter", root_name, "not found in JSON data"))
            next
          }
          
          ## Process each matching parameter
          for(param in matching_params) {
            values <- process_element(data[[param]])
            result <- c(result, values)
          }
        }
        
        ## Check if the length of the result matches the expected length
        ## This is a basic validation assuming the JSON data contains all expected parameters
        if(length(result) != length(unlist(lapply(param_names_main_correct_order, function(name) {
          param_matches <- grep(name, names(data), fixed = TRUE)
          if(length(param_matches) > 0) {
            return(data[param_matches])
          } else {
            return(NULL)
          }
        })))) {
          warning("The length of the ordered vector may not match the expected parameters")
        }
        
        return(result)
  
}














