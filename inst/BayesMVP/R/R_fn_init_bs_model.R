
  
  
#' get_BayesMVP_Stan_paths
#' @export
get_BayesMVP_Stan_paths <- function() {
  
      Sys.setenv(STAN_THREADS = "true")
      
      # Get package directory paths
      pkg_dir <- system.file(package = "BayesMVP")
      data_dir <- file.path(pkg_dir, "stan_data")  # directory to store data inc. JSON data files
      stan_dir <- file.path(pkg_dir, "stan_models")
      ##
      print(paste("pkg_dir = ", pkg_dir))
      print(paste("data_dir = ", data_dir))
      print(paste("stan_dir = ", stan_dir))
      
      return(list(pkg_dir = pkg_dir,
                  data_dir = data_dir,
                  stan_dir = stan_dir))
  
}






#' init_bs_model_internal
#' @export
init_bs_model_internal <- function(   stream = NULL,
                                      Stan_data_list, 
                                      Stan_model_name
) {
  
        # Get package directory paths
        outs <- get_BayesMVP_Stan_paths()
        pkg_dir <- outs$pkg_dir
        data_dir <- outs$data_dir
        stan_dir <- outs$stan_dir
        
        ## Stan model path
        Stan_model_file_path <- file.path(stan_dir, 
                                          Stan_model_name)
        
        # if (!(is.null(stream))) {
        #   Stan_model_file_path <- copy_.stan_with_worker_id(original_path  = Stan_model_file_path, 
        #                             ii = stream)
        # } 
        ##
        outs_bs_model <- init_bs_model_external(stream = stream,
                                                Stan_data_list = Stan_data_list, 
                                                Stan_model_file_path = Stan_model_file_path)
        ##
        bs_model <- outs_bs_model$bs_model
        json_file_path <- outs_bs_model$json_file_path
        model_so_file <- outs_bs_model$model_so_file
        Stan_model_file_path <- outs_bs_model$Stan_model_file_path
        
        return(list(bs_model = bs_model, 
                    json_file_path = json_file_path, 
                    model_so_file = model_so_file,
                    Stan_model_file_path = Stan_model_file_path))
  
}





 

#' init_bs_model_external
#' @export
init_bs_model_external <- function( stream = NULL,
                                    Stan_data_list,
                                    Stan_model_file_path,
                                    stanc_args = NULL,
                                    make_args = NULL
) {
  
        outs <- get_BayesMVP_Stan_paths()
        pkg_dir <- outs$pkg_dir
        data_dir <- outs$data_dir
        stan_dir <- outs$stan_dir
        ##
        if (length(Stan_data_list) == 0) {
          
              ## ---- a model with NO data: cmdstanr::write_stan_json() would write an
              ## empty ARRAY ([]), but BridgeStan requires an empty OBJECT ({}):
              json_file_path <- file.path( data_dir,
                                           paste0( "data_empty_",
                                                   if (is.null(stream)) 0 else stream,
                                                   ".json"))
              ##
              writeLines( text = "{}",
                          con = json_file_path)
              ##
              validated_json_string <- "{}"
              
        } else {
          
              json_file_path <- convert_stan_data_list_to_JSON( stan_data_list = Stan_data_list,
                                                                pkg_data_dir = data_dir)
              if (!(is.null(stream))) {
                json_file_path <- copy_json_with_worker_id( original_path  = json_file_path, 
                                                            ii = stream)
              } 
              
              # Read the raw content
              raw_content <- paste(readLines(json_file_path), collapse = "")
              
              # Try to parse it
              parsed <- tryCatch({
                jsonlite::fromJSON(raw_content)
              }, error = function(e) {
                stop("Invalid JSON in file: ", e$message)
              })
              
              # Check if it's wrapped in an array containing a string
              if (is.character(parsed) && length(parsed) == 1) {
                # It's double-encoded, extract the inner JSON
                json_data <- jsonlite::fromJSON(parsed[1])
              } else {
                # It's already proper JSON data
                json_data <- parsed
              }
              
              # # Validate it has expected structure
              # if (!all(c("N", "y") %in% names(json_data))) {
              #   warning("JSON data missing required fields N and y")
              # }
              
              # Write it back in the correct format
              jsonlite::write_json( x = json_data, path = json_file_path, 
                                    auto_unbox = TRUE, digits = NA, pretty = FALSE)
              
              # Read the validated string for bridgestan
              validated_json_string <- paste(readLines(json_file_path), collapse = "")
              
        }
        ##
        # if (!(is.null(stream))) {
        #   Stan_model_file_path <- copy_.stan_with_worker_id(original_path  = Stan_model_file_path, 
        #                                                     ii = stream)
        # } 
        ##
        ## stanc_args  -> arguments for the STAN compiler (e.g. list("--O1", "--include-paths=..."))
        ## make_args   -> C++/make options for compiling the generated C++ (e.g.
        ##                list("STAN_THREADS=true") for reduce_sum models). These are
        ##                distinct: stanc_args never affect the C++ compilation flags.
        ##
        ## ---- STAN_THREADS: get_BayesMVP_Stan_paths() exports STAN_THREADS=true to make, so
        ## the model is built with a thread-local autodiff stack (safe for parallel chains).
        ## A model built WITHOUT it has ONE global autodiff stack: concurrent
        ## log_density_gradient calls from parallel chains race on
        ## start_nested()/recover_memory_nested() and abort the session
        ## ("empty_nested() must be false..."). make skips .so files it considers
        ## up-to-date, so a stale non-threads .so would never be rebuilt -- detect
        ## it by its embedded flag and delete it to force a fresh, threads build:
        model_so_file <- normalizePath(transform_stan_path(Stan_model_file_path))
        ##
        if (file.exists(model_so_file)) {
              raw_so <- readBin( con = model_so_file,
                                 what = "raw",
                                 n = file.info(model_so_file)$size)
              ##
              if (length(grepRaw( pattern = "STAN_THREADS=false",
                                  x = raw_so,
                                  fixed = TRUE)) > 0) {
                    file.remove(model_so_file)
              }
        }
        ##
        bs_model <- bridgestan::StanModel$new(lib = Stan_model_file_path, 
                                              data = validated_json_string, 
                                              seed = 123,
                                              stanc_args = stanc_args,
                                              make_args = make_args)
        
        json_file_path <- normalizePath(json_file_path)
        ##
        Stan_model_file_path <- normalizePath(Stan_model_file_path)
        model_so_file <- normalizePath(transform_stan_path(Stan_model_file_path))
        
        return(list(bs_model = bs_model, 
                    json_file_path = json_file_path, 
                    model_so_file = model_so_file,
                    Stan_model_file_path = Stan_model_file_path))
  
}













