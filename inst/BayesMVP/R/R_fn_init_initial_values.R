
## R_fn_init_initial_values.R

#' R_fn_init_initial_values
#' @keywords internal
#' @export
R_fn_init_initial_values <- function( Model_type,
                                      bs_model,
                                      ##
                                      n_chains_burnin,
                                      init_lists_per_chain,
                                      ##
                                      sample_nuisance,
                                      n_nuisance,
                                      n_params_main
) {
  
        if (is.null(init_lists_per_chain)) { 
          stop("initial values per chain (init_lists_per_chain) not supplied - please supply")
        }
        if (is.null(n_chains_burnin)) { 
          stop("n_chains_burnin not supplied - please supply")
        }
        if (length(init_lists_per_chain) != n_chains_burnin) { 
          stop("n_chains_burnin must be equal to the length of the initial values list ('init_lists_per_chain')")
        }
        theta_nuisance_vectors_all_chains_input_from_R <- NULL
        theta_main_vectors_all_chains_input_from_R <- NULL
        inits_unconstrained_vec_per_chain <- list()
        ##
        for (kk in 1:n_chains_burnin) {
            
            if (Model_type == "Stan") {
              
              ##
              ## ---- Stan models: preserve the COMPLETE external-model initialisation and
              ## unconstrain it, and ONLY THEN split nuisance and main coordinates. No
              ## parameter is removed by name (previously "u_raw" was deleted), so the
              ## nuisance block may be called anything and a nuisance block that exists
              ## but has zero length contributes zero coordinates naturally.
              ##
              ## BridgeStan's param_unconstrain_json() rejects PARTIAL inits, so missing
              ## entries are completed first with the model's own defaults (the
              ## constrained transform of the zero unconstrained vector) and merged with
              ## the user's values.
              ##
              init_list_kk <- init_lists_per_chain[[kk]]
              ##
              json_string_for_inits_chain_kk <- BayesMVP:::convert_stan_data_list_to_JSON(init_list_kk)
              validated_json_string <- paste(readLines(json_string_for_inits_chain_kk), collapse="")
              ##
              inits_unconstrained_vec_kk <- tryCatch( expr = {
                bs_model$param_unconstrain_json(validated_json_string)
              }, error = function(error_condition) {
                NULL
              })
              ##
              if (is.null(inits_unconstrained_vec_kk)) {
                
                    ## ---- partial initialisation: complete missing parameters with the
                    ## model defaults and retry.
                    bs_names <- bs_model$param_names()
                    ##
                    default_theta_constrained <- tryCatch( expr = {
                      bs_model$param_constrain( rep( x = 0.0,
                                                     times = bs_model$param_unc_num()),
                                                include_tp = FALSE,
                                                include_gq = FALSE,
                                                rng = bs_model$new_rng(seed = 123L))
                    }, error = function(error_condition) {
                      NULL
                    })
                    ##
                    if (is.null(default_theta_constrained)) {
                      stop( "R_fn_init_initial_values: the initial values for chain ", kk,
                            " could not be unconstrained, and model-default initial values could not be ",
                            "constructed either. BridgeStan error: ",
                            conditionMessage(error_condition))
                    }
                    ##
                    names(default_theta_constrained) <- bs_names
                    ##
                    for (param_name in names(init_list_kk)) {
                      matching_indices <- which(bs_names == param_name)
                      ##
                      if (length(matching_indices) == 0) {
                        matching_indices <- grep( pattern = paste0( "^",
                                                                    gsub( pattern = "\\[",
                                                                          replacement = "\\\\[",
                                                                          x = gsub( pattern = "\\]",
                                                                                    replacement = "\\\\]",
                                                                                    x = param_name)),
                                                                  "(\\[|$)"),
                                                  x = bs_names)
                      }
                      ##
                      if (length(matching_indices) > 0) {
                        default_theta_constrained[matching_indices] <- as.numeric(init_list_kk[[param_name]])
                      } else {
                        warning( "R_fn_init_initial_values: '", param_name,
                                 "' in the init list for chain ", kk,
                                 " does not match any parameter of the model and was ignored.")
                      }
                    }
                    ##
                    inits_unconstrained_vec_kk <- bs_model$param_unconstrain( default_theta_constrained )
                    ##
                    message("R_fn_init_initial_values: chain ", kk,
                            " inits were PARTIAL - missing parameters were completed with model defaults.")
                
              }
              ##
              inits_unconstrained_vec_per_chain[[kk]] <- inits_unconstrained_vec_kk
              
            } else {
              
              ## ---- built-in models: unchanged (u_raw dropped by name, as before):
              init_list_kk <- init_lists_per_chain[[kk]]
              init_list_kk$u_raw <- NULL  # Remove nuisance
              ##
              json_string_for_inits_chain_kk <- BayesMVP:::convert_stan_data_list_to_JSON(init_list_kk)
              validated_json_string <- paste(readLines(json_string_for_inits_chain_kk), collapse="")
              inits_unconstrained_vec_per_chain[[kk]] <- bs_model$param_unconstrain_json(validated_json_string)
              
            }
            ##
            cat("n_params_main =", n_params_main, "\n")
            cat("length of unconstrained vec =", length(inits_unconstrained_vec_per_chain[[kk]]), "\n")
            
        }
        ##
        if (Model_type == "Stan") {
          
              ## ---- Stan models: split the COMPLETE unconstrained vector by position -
              ## the first n_nuisance coordinates are nuisance, the rest main. This holds
              ## for n_nuisance = 0 (empty nuisance; everything is main) as well.
              if (isTRUE(sample_nuisance) && n_nuisance > 0) {
                    index_nuisance <- seq_len(n_nuisance)
                    index_main <- (1 + n_nuisance):(n_nuisance + n_params_main)
                    ##
                    theta_nuisance_vectors_all_chains_input_from_R <- array( 0,
                                                                              dim = c(n_nuisance, n_chains_burnin))
                    theta_main_vectors_all_chains_input_from_R      <- array( 0,
                                                                              dim = c(n_params_main, n_chains_burnin))
                    ##
                    for (kk in 1:n_chains_burnin) {
                      theta_nuisance_vectors_all_chains_input_from_R[, kk] <- inits_unconstrained_vec_per_chain[[kk]][index_nuisance]
                      theta_main_vectors_all_chains_input_from_R[, kk]      <- inits_unconstrained_vec_per_chain[[kk]][index_main]
                    }
              } else {
                    ## sample_nuisance = FALSE (no nuisance block) OR n_nuisance = 0
                    ## (zero-length nuisance declaration): the complete unconstrained
                    ## vector is the MAIN block.
                    theta_nuisance_vectors_all_chains_input_from_R <- array( 0,
                                                                              dim = c(0, n_chains_burnin))
                    theta_main_vectors_all_chains_input_from_R      <- array( 0,
                                                                              dim = c(n_params_main, n_chains_burnin))
                    ##
                    for (kk in 1:n_chains_burnin) {
                      theta_main_vectors_all_chains_input_from_R[, kk] <- inits_unconstrained_vec_per_chain[[kk]][1:n_params_main]
                    }
              }
          
        } else {
          
              ## ---- built-in models: unchanged split logic:
              if (sample_nuisance == FALSE) {
                
                    n_nuisance_dummy <- 1 ## bookmark
                    n_params <- n_params_main
                    ##
                    theta_nuisance_vectors_all_chains_input_from_R    <- array(0, dim = c(n_nuisance_dummy, n_chains_burnin))
                    theta_main_vectors_all_chains_input_from_R  <- array(0, dim = c(n_params_main, n_chains_burnin))
                    ##
                    for (kk in 1:n_chains_burnin) {
                        theta_main_vectors_all_chains_input_from_R[, kk] <-     inits_unconstrained_vec_per_chain[[kk]][1:n_params_main]
                    }
                
              } else if (sample_nuisance == TRUE) {
                    
                    # n_params <- n_params_main ## + n_nuisance
                    index_nuisance <- seq_len(n_nuisance)
                    # index_main <- (1 + n_nuisance):n_params
                    ##
                    theta_nuisance_vectors_all_chains_input_from_R    <- array(0, dim = c(n_nuisance, n_chains_burnin))
                    theta_main_vectors_all_chains_input_from_R  <- array(0, dim = c(n_params_main, n_chains_burnin))
                    ##
                    for (kk in 1:n_chains_burnin) {
                      u_raw_kk <- init_lists_per_chain[[kk]]$u_raw
                      theta_main_vectors_all_chains_input_from_R[, kk] <-     inits_unconstrained_vec_per_chain[[kk]] ## [index_main]
                      theta_nuisance_vectors_all_chains_input_from_R[, kk] <-   u_raw_kk ##   inits_unconstrained_vec_per_chain[[kk]][index_nuisance]
                    }
                
              }
          
        }
        ##
        return(list(inits_unconstrained_vec_per_chain = inits_unconstrained_vec_per_chain,
                    theta_nuisance_vectors_all_chains_input_from_R = theta_nuisance_vectors_all_chains_input_from_R,
                    theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                    n_chains_burnin = n_chains_burnin,
                    init_lists_per_chain = init_lists_per_chain))
        
}

















                  
                   
                  
                 
  