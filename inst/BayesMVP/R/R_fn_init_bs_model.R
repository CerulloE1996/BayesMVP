





#' init_bs_model_internal
#' @export
init_bs_model_internal <- function(   stream = NULL,
                                      Stan_data_list, 
                                      Stan_model_name
) {
  
        # Get package directory paths
        pkg_dir <- system.file(package = "BayesMVP")
        outs <- list(pkg_dir = pkg_dir, data_dir = file.path(pkg_dir, "stan_data"), stan_dir = file.path(pkg_dir, "stan_models"))
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
