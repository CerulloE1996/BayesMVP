

# 
# json <- jsonlite::fromJSON("/home/enzocerullo/R/R-4.3.3/lib/R/library/BayesMVP/stan_data/data_6e479c5a0b94fe3f633a31e26aa5dbcc_1000.json")
# json$n_thr_per_ord_test
# json$n_cat_per_ord_test
# json$n_ordinal_tests
# 
# 
# file.remove("/home/enzocerullo/R/R-4.3.3/lib/R/library/BayesMVP/stan_data/data_6e479c5a0b94fe3f633a31e26aa5dbcc_1000.json")


## R_fn_initialise_model.R               
                       
                       
#' initialise_model
#' @export
initialise_model <- function( Model_type,
                              ##
                              stream = NULL,
                              ##
                              sample_nuisance = NULL,
                              n_nuisance_override = NULL,
                              ##
                              model_args_list = NULL, # For internal/hard-coded models 
                              ##
                              Stan_data_list = NULL, ## for user-supplied Stan models
                              ##
                              compile = TRUE,
                              force_recompile = FALSE,
                              ##
                              cmdstanr_model_fit_obj = NULL,
                              ##
                              Stan_model_file_path = NULL,
                              Stan_cpp_user_header = NULL,
                              Stan_cpp_flags = NULL,
                              stanc_args = NULL,
                              make_args = NULL
) {
  
        ##
        ## Generate stream ID if not provided
        ##
        if (is.null(stream)) {
          stream <- sample.int(1e6, 1)  # Random ID to avoid collisions
        }
        ##
        hard_coded_models_vec <- c("LC_MVP", 
                                   "MVP",
                                   "latent_trait",
                                   "LC_MVOP", 
                                   "MVOP")
        ##
        models_vec <- c("Stan", hard_coded_models_vec)
        ##
        if (!(Model_type %in% models_vec)) { 
          stop("Model_type must be set and be one of the following: 'Stan' 'LC_MVP', 'MVP', 'latent_trait', 'LC_MVOP', 'MVOP'")
        }
        ##
        y <- model_args_list$y
        if (Model_type  %in% hard_coded_models_vec) {
          if (is.null(y)) { 
            stop("input data y (inside 'model_args_list') is needed if using a built-in/hard-coded model
                 (i.e. for Model_type = 'LC_MVP', 'MVP', 'latent_trait', 'LC_MVOP', 'MVOP')")
          }
          N <- model_args_list$N
          if (is.null(N)) { 
            warning("N not inputted into 'model_args_list' - assuming N is the number of rows of the data (y)")
            N <- nrow(y)
            print(paste("N = ", N))
          }
        }
        
        if (Model_type != "Stan") {
          if ((is.null(model_args_list$X))) { 
            warning("Assuming intercept-only model - if not please supply X")
          }
        }
        
        if (Model_type == "Stan") {
          
                if (is.null(Stan_model_file_path)) { 
                  stop("Stan_model_file_path must be supplied if Model_type is 'Stan'")
                }
                if (is.null(Stan_data_list)) { 
                  stop("Stan_data_list must be supplied if Model_type is 'Stan'")
                }
                Stan_model_name <- NULL ## not needed for external Stan models
              
        } else {
          
               ##
               ## ---- Get "Stan_model_name" for internal/hard-coded models:
               ##
               if (Model_type == "LC_MVP") { 
                   Stan_model_name <- "LC_MVP_bin_cpp_skeleton.stan"
               } else if (Model_type == "MVP") { 
                   Stan_model_name <- "MVP_bin_cpp_skeleton.stan"
               } else if (Model_type == "latent_trait") { 
                   Stan_model_name <- "latent_trait_bin_cpp_skeleton_10.stan"
               } else if (Model_type == "LC_MVOP") {
                   Stan_model_name <- "LC_MVOP_cpp_skeleton.stan"
               } else if (Model_type == "MVOP") {
                   Stan_model_name <- "MVOP_cpp_skeleton.stan"
               }
          
        }
        ##
        ## ---- Process internal models:
        ##
        if (Model_type != "Stan") {
                ##
                ## ---- Get final "model_args_list" and "Model_args_as_Rcpp_List" for internal/hard-coded models:
                ##
                outs <- init_hard_coded_model( Model_type = Model_type,
                                               model_args_list = model_args_list)
                ##
                model_args_list <- outs$model_args_list
                Model_args_as_Rcpp_List = outs$Model_args_as_Rcpp_List
                ##
                ## ---- Get "Stan_data_list" for internal/hard-coded models:
                ##
                Stan_data_list <- make_Stan_data_list_for_internal_models( Model_type = Model_type,
                                                                           model_args_list = model_args_list)
        }
        ##
        ## ---- A user C++ header (external Stan functions, e.g. the BayesMVP AVX kernels exposed
        ## to Stan) is honoured by BridgeStan through TWO things: stanc needs --allow-undefined
        ## (the functions are declared without bodies) and make needs USER_HEADER=<path>
        ## (BridgeStan's Makefile then adds -include <path>). Stan_cpp_user_header used to be
        ## stored on the object but never reached the compile call, so it was silently ignored.
        ##
        if (!is.null(Stan_cpp_user_header)) {
              ##
              if (!file.exists(Stan_cpp_user_header))
                    stop("Stan_cpp_user_header does not exist: ", Stan_cpp_user_header)
              ##
              stanc_args <- unique(c( as.list(stanc_args),
                                      "--allow-undefined"))
              ##
              make_args <- c( as.list(make_args),
                              paste0( "USER_HEADER=",
                                      normalizePath(Stan_cpp_user_header)))
        }
        ##
        ## ---- Compile using BridgeStan (NEED to have "Stan_data_list" to do this - even for internal models!):
        ##
        outs_model_info <- get_model_info(  Model_type = Model_type,
                                            ##
                                            stream = stream,
                                            ##
                                            Stan_model_name = Stan_model_name,
                                            ##
                                            Stan_data_list = Stan_data_list,
                                            Stan_model_file_path = Stan_model_file_path,
                                            stanc_args = stanc_args,
                                            make_args = make_args,
                                            ##
                                            sample_nuisance = sample_nuisance,
                                            n_nuisance_override = n_nuisance_override,
                                            ##
                                            model_args_list = model_args_list)
        ##
        bs_model <- outs_model_info$outs_bs_model$bs_model
        print(paste("bs_model = "))
        print(bs_model)
        ##
        json_file_path <- outs_model_info$outs_bs_model$json_file_path
        json_file_path <- normalizePath(json_file_path)
        # json_file_path <- copy_json_with_worker_id(json_file_path, stream)
        print(paste("json_file_path = "))  
        print(json_file_path)
        ##
        model_so_file <- outs_model_info$outs_bs_model$model_so_file
        model_so_file <- normalizePath(model_so_file)
        # model_so_file <- copy_.so_with_worker_id(model_so_file, stream)
        print(paste("model_so_file = "))
        print(model_so_file)
        ##
        Stan_model_file_path <- outs_model_info$outs_bs_model$Stan_model_file_path
        Stan_model_file_path <- normalizePath(Stan_model_file_path)
        print(paste("Stan_model_file_path = "))
        print(Stan_model_file_path)
        ##
        ##
        if (Model_type == "Stan") { 
            dummy_json_file_path <- NULL
            dummy_model_so_file <- NULL
        } else { 
            dummy_json_file_path <- json_file_path
            ##
            dummy_model_so_file <- model_so_file
        }
        ##
        bs_main_param_names <- outs_model_info$outs_bs_model$bs_model$param_names()
        bs_main_and_tp_param_names <- outs_model_info$outs_bs_model$bs_model$param_names(include_tp = TRUE)
        bs_main_and_tp_and_gq_param_names <- outs_model_info$outs_bs_model$bs_model$param_names(include_tp = TRUE, include_gq = TRUE)
        ##
        stan_main_param_names <- convert_bridgestan_par_names_to_stan(bs_main_param_names)
        stan_main_and_tp_param_names <- convert_bridgestan_par_names_to_stan(bs_main_and_tp_param_names)
        stan_main_and_tp_and_gq_param_names <- convert_bridgestan_par_names_to_stan(bs_main_and_tp_and_gq_param_names)
        ##
        n_nuisance <- outs_model_info$n_nuisance
        n_params_main <- outs_model_info$n_params_main
        n_params <- outs_model_info$n_params
        n_nuisance_names_constrained <- outs_model_info$n_nuisance_names_constrained
        nuisance_base_name <- outs_model_info$nuisance_base_name
        ##
        print(paste("n_params_main = ", n_params_main))
        print(paste("n_nuisance = ", n_nuisance))
        print(paste("n_params = ", n_params))
        ##
        ##
        ## ---- Make "Model_args_as_Rcpp_List":
        ##
        if (Model_type %in% hard_coded_models_vec) {
          
              # Already created above
              # Model_args_as_Rcpp_List already set from init_hard_coded_model
          
        } else if (Model_type == "Stan") { 
          
              Model_args_as_Rcpp_List <- list()
              ##
              Model_args_as_Rcpp_List$n_params_main <- n_params_main
              Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
              ##
              Model_args_as_Rcpp_List$N <- 100 ## dummy
              Model_args_as_Rcpp_List$n_tests <- 5 ## dummy
              Model_args_as_Rcpp_List$n_class <- 2 ## dummy
              Model_args_as_Rcpp_List$y <- array(0, dim = c(100, 5)) ## dummy
              ##
              ## the C++ list->struct conversion reads these unconditionally; for
              ## external Stan models they are unused dummies:
              Model_args_as_Rcpp_List$n_binary_tests <- 0L
              Model_args_as_Rcpp_List$n_ordinal_tests <- 0L
              
        }
        ##
        Model_args_as_Rcpp_List$json_file_path <- json_file_path
        Model_args_as_Rcpp_List$model_so_file <- model_so_file
        # ##
        # ## ---- Process initial values:
        # ##
        # outs <- R_fn_init_initial_values( Model_type = Model_type,
        #                                   bs_model = bs_model,
        #                                   ##
        #                                   n_chains_burnin = n_chains_burnin,
        #                                   init_lists_per_chain = init_lists_per_chain,
        #                                   ##
        #                                   sample_nuisance = sample_nuisance,
        #                                   n_nuisance = n_nuisance,
        #                                   n_params_main = n_params_main)
        # ##
        # inits_unconstrained_vec_per_chain <- outs$inits_unconstrained_vec_per_chain
        # theta_us_vectors_all_chains_input_from_R <- outs$theta_us_vectors_all_chains_input_from_R
        # theta_main_vectors_all_chains_input_from_R <- outs$theta_main_vectors_all_chains_input_from_R
        # n_chains_burnin <- outs$n_chains_burnin
        # init_lists_per_chain <- outs$init_lists_per_chain
        
 
        
        return(list(   Model_type = Model_type,
                       ##
                       stream = stream,
                       ##
                       cmdstanr_model_fit_obj = NULL,
                       ##
                       sample_nuisance = sample_nuisance,
                       n_nuisance_override = n_nuisance_override,
                       ##
                       n_nuisance = n_nuisance,
                       n_params_main = n_params_main,
                       n_params = n_params,
                       ##
                       n_nuisance_names_constrained = n_nuisance_names_constrained,
                       nuisance_base_name = nuisance_base_name,
                       ##
                       bs_model = bs_model,
                       ##
                       bs_main_param_names = bs_main_param_names,
                       bs_main_and_tp_param_names = bs_main_and_tp_param_names,
                       bs_main_and_tp_and_gq_param_names = bs_main_and_tp_and_gq_param_names,
                       ##
                       stan_main_param_names = stan_main_param_names,
                       stan_main_and_tp_param_names = stan_main_and_tp_param_names,
                       stan_main_and_tp_and_gq_param_names = stan_main_and_tp_and_gq_param_names,
                       ##
                       model_args_list = model_args_list,
                       ##
                       Stan_data_list = Stan_data_list,
                       ##
                       Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                       ##
                       Stan_model_file_path = Stan_model_file_path,
                       ##
                       # init_lists_per_chain = init_lists_per_chain,
                       # n_chains_burnin = n_chains_burnin,
                       ##
                       # inits_unconstrained_vec_per_chain = inits_unconstrained_vec_per_chain,
                       # theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R,
                       # theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R, 
                       ##
                       json_file_path = json_file_path,
                       model_so_file = model_so_file,
                       dummy_json_file_path = dummy_json_file_path,
                       dummy_model_so_file = dummy_model_so_file,
                       ##
                       Stan_cpp_user_header = Stan_cpp_user_header,
                       Stan_cpp_flags = Stan_cpp_flags,
                       stanc_args = stanc_args,
                       make_args = make_args))
  
}





