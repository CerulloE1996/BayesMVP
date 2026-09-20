
 


 
# R_fn_get_unpermute_map <- function( param_names,
#                                     test_perm,
#                                     family_test_index_pos = list( "Se_baseline" = 1,
#                                                                   "Sp_baseline" = 1,
#                                                                   "Se"    = 1,
#                                                                   "Sp"    = 1,
#                                                                   "beta"  = 2,          ## beta[coeff_row, test]
#                                                                   "C"     = 1,          ## MVOP cutpoints C[test, k] -- EDIT to actual name(s)
#                                                                   "Omega" = c(2, 3)),   ## Omega[class, t1, t2]
#                                     symmetric_families = c("Omega"),
#                                     chol_families = c("L_Omega")                        ## CANNOT be un-permuted by reindexing
# ) {
#         
#         test_perm <- as.integer(test_perm)
#         n_p <- length(param_names)
#         ##
#         identity_out <- list( slice_map = seq_len(n_p),
#                               changed = FALSE,
#                               skipped_chol = character(0))
#         ##
#         if (identical(test_perm, seq_along(test_perm))) return(identity_out)
#         ##
#         ## ---- Parse "fam[1,2]" / "fam.1.2" / plain "fam":
#         ##
#         parse_name <- function(nm) {
#           if (grepl("\\[", nm)) {
#             fam    <- sub("^([^\\[]+)\\[.*$", "\\1", nm)
#             inside <- sub("^[^\\[]+\\[(.*)\\]$", "\\1", nm)
#             return(list(fam = fam,
#                         idxs = as.integer(strsplit(inside, "[,[:space:]]+")[[1]]),
#                         fmt = "bracket"))
#           }
#           parts <- strsplit(nm, "\\.")[[1]]
#           if (length(parts) > 1) {
#             is_num <- grepl("^[0-9]+$", parts)
#             k <- length(parts)
#             while (k >= 1 && is_num[k]) k <- k - 1
#             if (k < length(parts)) {
#               return(list(fam  = paste(parts[1:k], collapse = "."),
#                           idxs = as.integer(parts[(k + 1):length(parts)]),
#                           fmt  = "dot"))
#             }
#           }
#           return(list(fam = nm, idxs = integer(0), fmt = "none"))
#         }
#         ##
#         rebuild_name <- function(fam, idxs, fmt) {
#           if (length(idxs) == 0)  return(fam)
#           if (fmt == "bracket")   return(paste0(fam, "[", paste(idxs, collapse = ","), "]"))
#           return(paste(c(fam, idxs), collapse = "."))
#         }
#         ##
#         ## ---- For each slice, compute its TRUE (original-test-order) name.
#         ##      Semantics (matches R_fn_sample_model): model fit on y[, test_perm],
#         ##      so permuted-model slot j == ORIGINAL test test_perm[j].
#         ##
#         parsed       <- lapply(param_names, parse_name)
#         true_names   <- param_names
#         skipped_chol <- character(0)
#         ##
#         for (h in seq_len(n_p)) {
#           p <- parsed[[h]]
#           if (p$fam %in% chol_families) {
#             skipped_chol <- c(skipped_chol, p$fam)
#             next    ## leave in place (true name = own name => identity slot)
#           }
#           pos <- family_test_index_pos[[p$fam]]
#           if (is.null(pos) || length(p$idxs) == 0) next
#           pos_ok <- pos[pos <= length(p$idxs)]
#           if (length(pos_ok) == 0) next
#           idxs <- p$idxs
#           idxs[pos_ok] <- test_perm[idxs[pos_ok]]
#           true_names[h] <- rebuild_name(p$fam, idxs, p$fmt)
#         }
#         ##
#         ## ---- Pass 1: direct match  (canonical name k  <-  slice whose true name == param_names[k]):
#         ##
#         slice_map <- match(param_names, true_names)
#         ##
#         ## ---- Pass 2: symmetric families stored one-triangle-only -- retry with the
#         ##      two test indices swapped:
#         ##
#         if (anyNA(slice_map)) {
#           for (k in which(is.na(slice_map))) {
#             p <- parsed[[k]]
#             pos <- family_test_index_pos[[p$fam]]
#             if (!(p$fam %in% symmetric_families) || is.null(pos) || length(pos) != 2) next
#             ## invert: canonical test pair (i,j) lives in permuted slot pair (inv[i], inv[j]):
#             inv  <- order(test_perm)
#             idxs <- p$idxs
#             idxs[pos] <- inv[p$idxs[pos]]
#             cand <- c( rebuild_name(p$fam, idxs, p$fmt),
#                        { idxs2 <- idxs; idxs2[pos] <- rev(idxs2[pos]); rebuild_name(p$fam, idxs2, p$fmt) })
#             hit <- match(cand, param_names)
#             hit <- hit[!is.na(hit)]
#             if (length(hit) >= 1) slice_map[k] <- hit[1]
#           }
#         }
#         ##
#         if (anyNA(slice_map)) {
#           stop(paste0("R_fn_get_unpermute_map: could not map: ",
#                       paste(param_names[is.na(slice_map)], collapse = ", "),
#                       " -- check family_test_index_pos."))
#         }
#         ##
#         return(list( slice_map = slice_map,
#                      changed = !identical(slice_map, seq_len(n_p)),
#                      skipped_chol = unique(skipped_chol)))
# }




#' thin_mcmc
#' @export
thin_mcmc <- function(trace, 
                      thin = 10) {
  
        # Validate inputs
        if (!is.list(trace)) {
          stop("trace must be a list")
        }
        if (thin < 1 || thin != round(thin)) {
          stop("thin must be a positive integer")
        }
        
        # Apply thinning to each chain
        thinned_trace <- lapply(trace, function(chain) {
          if (!is.matrix(chain)) {
            stop("Each chain must be a matrix")
          }
          
          # Get dimensions
          n_params <- nrow(chain)
          n_iter <- ncol(chain)
          
          # Create indices for thinning
          keep_indices <- seq(1, n_iter, by = thin)
          
          # Return thinned chain
          chain[, keep_indices, drop = FALSE]
        })
        
        return(thinned_trace)
  
}



#' permute_3d_draws_array
#' @export
permute_3d_draws_array <- function(array,
                                   iter_index = 1,
                                   chain_index = 2,
                                   param_index = 3,
                                   new_iter_index = 2,
                                   new_chain_index = 3,
                                   new_param_index = 1) {
  
        # Create the permutation vector
        # We need to map old positions to new positions
        perm_vector <- numeric(3)
        
        # Find which old index goes to position 1
        if (new_iter_index == 1) perm_vector[1] <- iter_index
        else if (new_chain_index == 1) perm_vector[1] <- chain_index
        else if (new_param_index == 1) perm_vector[1] <- param_index
        
        # Find which old index goes to position 2
        if (new_iter_index == 2) perm_vector[2] <- iter_index
        else if (new_chain_index == 2) perm_vector[2] <- chain_index
        else if (new_param_index == 2) perm_vector[2] <- param_index
        
        # Find which old index goes to position 3
        if (new_iter_index == 3) perm_vector[3] <- iter_index
        else if (new_chain_index == 3) perm_vector[3] <- chain_index
        else if (new_param_index == 3) perm_vector[3] <- param_index
        
        # Apply the permutation
        new_array <- aperm(array, perm = perm_vector)
        
        # Update dimension names if they exist
        if (!is.null(dimnames(array))) {
          old_names <- dimnames(array)
          new_names <- list(
            old_names[[perm_vector[1]]],
            old_names[[perm_vector[2]]],
            old_names[[perm_vector[3]]]
          )
          
          # Rename based on new positions
          names(new_names) <- c(
            if (new_iter_index == 1) "iteration" else if (new_chain_index == 1) "chain" else "parameter",
            if (new_iter_index == 2) "iteration" else if (new_chain_index == 2) "chain" else "parameter",
            if (new_iter_index == 3) "iteration" else if (new_chain_index == 3) "chain" else "parameter"
          )
          
          dimnames(new_array) <- new_names
        }
        
        return(new_array)
  
}


#### --------------------------------------------------------------------------------------------------------------------------------------------------
#' create_summary_and_traces
#' @export
create_summary_and_traces <- function(    model_results,
                                          ##
                                          compute_main_params = TRUE, # excludes nuisance params. and log-lik 
                                          compute_transformed_parameters = TRUE,
                                          compute_generated_quantities = TRUE,
                                          ##
                                          save_log_lik_trace = FALSE, 
                                          save_nuisance_trace = FALSE,
                                          ##
                                          compute_nested_rhat = FALSE,
                                          n_superchains = NULL,
                                          ##
                                          save_trace_tibbles = FALSE,
                                          ##
                                          n_iter_to_store = NULL,
                                          ##
                                          use_disk = TRUE,
                                          use_disk_path = "/tmp/hmc_traces",
                                          use_disk_path_post_hoc_dir = "/tmp/constrain_traces"
) {
  
        require(bridgestan)
        require(stringr)
      
        ## Start timer: 
        tictoc::tic()
        ##
        init_object   <- model_results$init_object
        burnin_object <- model_results$burnin_object
        ## sampling_object <- model_results$sampling_object
        ##
        n_nuisance    <- model_results$init_object$n_nuisance
        n_params_main <- model_results$init_object$n_params_main
        ##
        ## Number of CONSTRAINED names belonging to the nuisance block (differs from the
        ## unconstrained n_nuisance whenever the nuisance block has dimension-changing
        ## constraints, e.g. a simplex). Used to locate the main block within the
        ## constrained draws:
        n_nuisance_names <- model_results$init_object$n_nuisance_names_constrained
        ##
        if (is.null(n_nuisance_names)) {
          n_nuisance_names <- n_nuisance
        }
        ##
        ## ---- ... but only if the nuisance block is actually PART of the model that produces
        ## those constrained draws. param_constrain runs against init_object$bs_model: for an
        ## external Stan model that declares the nuisance coordinates, the constrained draw is
        ## [nuisance | main] and n_nuisance_names locates the split. For a BUILT-IN model it is
        ## the cpp skeleton, and LC_MVP / LC_MVOP / MVP / MVOP skeletons declare the MAIN
        ## parameters only - the latent block lives in the C++ likelihood - so the constrained
        ## draw has NO nuisance rows and the offset must be 0. (latent_trait's skeleton does
        ## declare u_raw, so it keeps the full layout.)
        ##
        ## A model that contains the nuisance block must have MORE unconstrained coordinates
        ## than the nuisance block alone; one that does not will have far fewer. Getting this
        ## wrong makes n_params_main_constrained negative (n_params - n_nuisance) and
        ## array(dim = c(...)) fails with "negative length vectors are not allowed".
        ##
        if (n_nuisance_names > 0) {

              constrain_model_unc_num <- tryCatch(model_results$init_object$bs_model$param_unc_num(),
                                                  error = function(e) NA_integer_)
              ##
              if (!is.na(constrain_model_unc_num) && (constrain_model_unc_num <= n_nuisance)) {

                    message("create_summary_and_traces: the model used for param_constrain has ",
                            constrain_model_unc_num, " unconstrained coordinates, i.e. it does NOT contain the ",
                            n_nuisance, "-coordinate nuisance block (built-in model: the latent block is in the ",
                            "C++ likelihood, not the skeleton). Treating the constrained draws as main-only.")
                    ##
                    n_nuisance_names <- 0L

              }

        }
        ##
        ## Extract essential model info from "model_results$init_object" object:
        ##
        Model_type      <- model_results$init_object$Model_type
        sample_nuisance <- model_results$init_object$sample_nuisance
        ##
        ## ---- Extract traces:
        ##
        main_trace <- model_results$sampling_object[[1]]
        div_trace  <- model_results$sampling_object[[2]]
        ##
        ## Nuisance trace (the FULL unconstrained nuisance draws - required to reconstruct
        ## complete draws before constraining). RAM: sampling_object[[3]]. Disk mode: the
        ## binary file written by the C++ sampler.
        nuisance_trace <- NULL
        ##
        n_chains_sampling <- length(main_trace)
        n_chains <- n_chains_sampling
        ##
        n_iter <- dim(main_trace[[1]])[2]
        ##
        ## Check if using disk mode:
        # use_disk <- TRUE
        ##
        if (use_disk) { ## Read nuisance trace from disk:
         
                read_binary_trace <- function(filepath, 
                                              n_params, 
                                              n_iter) {
                  
                      if (!file.exists(filepath)) {
                        warning(paste("File not found:", filepath))
                        return(NULL)
                      }
                      con <- file(filepath, "rb")
                      data <- readBin(con, "double", n = n_params * n_iter)
                      close(con)
                      matrix(data, nrow = n_params, ncol = n_iter)
                  
                }
                
                # trace_dir <- result$trace_dir
                
                ## Read nuisance traces
                if ((sample_nuisance == TRUE) && (n_nuisance > 0)) {
                  nuisance_trace <- lapply(0:(n_chains - 1), 
                                           function(i) {
                    
                          read_binary_trace(
                            filepath = file.path(use_disk_path, paste0("chain_", i, "_nuisance.bin")),
                            n_params = n_nuisance,
                            n_iter = n_iter)
                                                
                    })
                }
                
                ## Read log_lik traces
                if (Model_type != "Stan") {
                    log_lik_trace <- lapply(0:(n_chains - 1), function(i) {
                      read_binary_trace(
                        file.path(use_disk_path, paste0("chain_", i, "_loglik.bin")),
                        n_params = model_results$init_object$model_args_list$N,
                        n_iter = n_iter
                      )
                    })
                }
          
        } else {    ## Data is in RAM as usual
           
                if ((sample_nuisance == TRUE) && (n_nuisance > 0)) {
                  nuisance_trace <- model_results$sampling_object[[3]]
                  ##
                  ## the sampler stores NO nuisance trace when n_nuisance_to_track = 0 (it is only
                  ## needed to constrain a model that declares the nuisance block); what comes back
                  ## is then empty / zero-row and must be treated as absent, not thinned or fed to
                  ## param_constrain:
                  first_chain <- if (is.list(nuisance_trace) && length(nuisance_trace) > 0) nuisance_trace[[1]] else nuisance_trace
                  if (is.null(first_chain) || length(first_chain) == 0 || NROW(first_chain) == 0) {
                    nuisance_trace <- NULL
                  }
                }
                # log_lik_trace <- result$trace_log_lik
        }
        
        # str(nuisance_trace)
        # 
        # use_disk_path
        # list.files("/tmp/hmc_traces/")
        # ##
        # list.files("/tmp/hmc_traces_1/")
        # list.files("/tmp/hmc_traces_1/chain_")
        # ##
        # list.files("/tmp/constrain_traces")
        
        {
          
                # tictoc::tic("Timer 1")
      
                # try({  
                #   model_results$sampling_object[[3]] <- NULL
                #   gc()
                # })
                ##
                # str(div_trace)
                ##
                if ((is.null(n_iter_to_store))) { 
                  n_iter_to_store <- n_iter
                }
                thin <- round(n_iter/n_iter_to_store)
                ##
                if (thin > 1) {
                    main_trace <-  thin_mcmc( trace = main_trace,
                                              thin = thin)
                    ##
                    div_trace <-  thin_mcmc( trace = div_trace,
                                              thin = thin)
                    ##
                    if (!is.null(nuisance_trace)) {
                      nuisance_trace <-  thin_mcmc( trace = nuisance_trace,
                                                    thin = thin)
                    }
                }
                
                if (save_log_lik_trace == TRUE) {
                  if (Model_type != "Stan") {
                    if (length(model_results$sampling_object[[6]]) > 0 && NROW(model_results$sampling_object[[6]][[1]]) == 0) {
                      stop("save_log_lik_trace = TRUE, but the model was sampled with store_log_lik_trace = FALSE, so there is no log-lik trace. ",
                           "Re-run the sampler with store_log_lik_trace = TRUE (the default).")
                    }
                    log_lik_trace_mnl_models <-  model_results$sampling_object[[6]] 
                  } else { 
                    log_lik_trace_mnl_models <- NULL
                  }
                } else { 
                  log_lik_trace_mnl_models <- NULL
                }
                
                # tictoc::toc(log = TRUE)
              
        }
        
        {
          
                # tictoc::tic("Timer 2")
            
                # if (Model_type != "Stan") {
                #   log_lik_trace_mnl_models <-  model_results$result[[6]] 
                # }
                
                #### time_burnin <- model_results$time_burnin
                time_burnin <- model_results$burnin_object$time_burnin
                if(is.na(time_burnin)) { 
                  time_burnin <- model_results$time_burnin
                }
                time_sampling <- model_results$time_sampling
                time_total_wo_summaries <- time_burnin + time_sampling
                
                #### Other MCMC / HMC info
                n_chains_burnin <- burnin_object$n_chains_burnin
                n_burnin <- model_results$n_burnin
                
                LR_main <- model_results$LR_main
                LR_us <- model_results$LR_us
                adapt_delta <- model_results$adapt_delta
                
                metric_type_main <- model_results$metric_type_main
                metric_shape_main <- model_results$metric_shape_main
                metric_type_nuisance <- model_results$metric_type_nuisance
                metric_shape_nuisance <- model_results$metric_shape_nuisance
                
                diffusion_HMC <- model_results$diffusion_HMC
                partitioned_HMC <- model_results$partitioned_HMC
                
                n_superchains <- model_results$n_superchains
                interval_width_main <- model_results$interval_width_main
                interval_width_nuisance <- model_results$interval_width_nuisance
                ##
                force_autodiff <- model_results$force_autodiff
                force_PartialLog <- model_results$force_PartialLog
                multi_attempts <- model_results$multi_attempts
              
                L_main_during_burnin_vec <- burnin_object$L_main_during_burnin_vec
                L_main_during_burnin <- burnin_object$L_main_during_burnin
                L_us_during_burnin_vec <- burnin_object$L_us_during_burnin_vec
                L_us_during_burnin <- burnin_object$L_us_during_burnin
                 
                # tictoc::toc(log = TRUE)
              
        }
        
         # str(nuisance_trace)
      
        {
          
                # tictoc::tic("Timer 3")
                
                if (sample_nuisance == TRUE) {
                    n_nuisance <- n_nuisance
                    ## what was ACTUALLY stored (0 when the sampler was told not to keep it):
                    n_nuisance_tracked <- if (is.null(nuisance_trace)) 0L else NROW(if (is.list(nuisance_trace)) nuisance_trace[[1]] else nuisance_trace)
                } else { 
                    n_nuisance <- 0
                    n_nuisance_tracked <- 0
                }
                
                n_divs <- sum(unlist(div_trace))
                pct_divs <- 100 * n_divs / length(unlist(div_trace))
                
                n_chains_sampling  <- length(main_trace)
                print(paste("n_chains_sampling = ", n_chains_sampling))
                
                if (is.null(compute_nested_rhat)) {
                  if (n_chains_sampling > 15) {
                    compute_nested_rhat <- TRUE
                  }
                }
                
                n_chains_burnin <- burnin_object$n_chains_burnin
                print(paste("n_chains_burnin = ", n_chains_burnin))
                ##
                n_iter <-   dim(main_trace[[1]])[2]
                print(paste("n_iter = ", n_iter))
                
                n_params_main <- dim(main_trace[[1]])[1]
                print(paste("n_params_main = ", n_params_main))
                ##
              
                nested_rhat_grouping <-  model_results$nested_rhat_grouping
                if (is.null(x = nested_rhat_grouping)) {
                    nested_rhat_grouping <-  list(status = "unavailable_initial_state_mapping_not_recorded")
                }
                if (isTRUE(x = compute_nested_rhat)) {
                    if (nested_rhat_grouping$status %in% c("ok", "singleton_groups")) {
                        cat("Nested R-hat: ", nested_rhat_grouping$n_superchains, " actual starting-state groups x ",
                            nested_rhat_grouping$n_chains_per_superchain, " chains = ", nested_rhat_grouping$n_chains_used,
                            "; ", nested_rhat_grouping$n_chains_omitted, " excluded from nRhat ONLY. ESS uses all chains.\n", sep = "")
                    } else {
                        warning("Nested R-hat unavailable: ", nested_rhat_grouping$status, call. = FALSE)
                    }
                }
                
                #### Stan_model_file_path <- (file.path(pkg_dir, "inst/stan_models/PO_LC_MVP_bin.stan"))  ### TEMP
                #### Stan_model_file_path <- model_results$init_object$Stan_model_file_path
                
                if (Model_type == "Stan") {
                  json_file_path <- model_results$init_object$json_file_path
                  model_so_file <-  model_results$init_object$model_so_file
                } else { 
                  json_file_path <- model_results$init_object$dummy_json_file_path
                  model_so_file <-  model_results$init_object$dummy_model_so_file
                }
                
                if (length(model_results$init_object$Stan_data_list) == 0) {
                  ## a model with NO data: write an empty OBJECT ({}), not an empty ARRAY:
                  writeLines(text = "{}", con = json_file_path)
                } else {
                  cmdstanr::write_stan_json(data = model_results$init_object$Stan_data_list, 
                                            file = json_file_path)
                }
                
                ##
                print(paste("model_results$init_object$json_file_path = "))
                print(model_results$init_object$json_file_path)
                ##
                print(paste("model_results$init_object$model_so_file = "))
                print(model_results$init_object$model_so_file)
                ##
                print(paste("model_results$init_object$dummy_json_file_path = "))
                print(model_results$init_object$dummy_json_file_path)
                ##
                print(paste("model_results$init_object$dummy_model_so_file = "))
                print(model_results$init_object$dummy_model_so_file)
                ##
                Sys.setenv(STAN_THREADS = "true")
                
                # tictoc::toc(log = TRUE)
        
        }
        
        
        {
        
            # tictoc::tic("Timer 4")
           
            #### bs_model <- StanModel$new(Stan_model_file_path, data = json_file_path, 1234) # creates .so file
            bs_model <- model_results$init_object$bs_model
            # bs_names  <-  (bs_model$param_names())
            bs_names <- model_results$init_object$stan_main_param_names
            
            n_stan_parameters_block <- length(bs_names) ; n_stan_parameters_block
            
            # message(print(paste("bs_names - head = ", head(bs_names))))
            # message(print(paste("bs_names - tail = ", tail(bs_names))))
            
            bs_names_inc_tp <-  model_results$init_object$stan_main_and_tp_param_names ##  (bs_model$param_names(include_tp = TRUE))
            bs_names_inc_tp_and_gq <-  model_results$init_object$stan_main_and_tp_and_gq_param_names ##  (bs_model$param_names(include_tp = TRUE, include_gq = TRUE))
            
            pars_names <- bs_names_inc_tp_and_gq
            ####  pars_names <- model_results$init_object$param_names
            
            if (model_results$init_object$stan_main_param_names[1] == "lp__") { 
              pars_names <- pars_names[-1]
            }
            
            # message(print(paste("pars_names - head = ", head(pars_names))))
            # message(print(paste("pars_names - tail = ", tail(pars_names))))
            
            index_lp  <- grep("^lp__", pars_names, invert = FALSE)
            
            # if (index_lp == 1) {  # if Stan model generates __lp variable (not all models will)
            #     pars_names <- pars_names[-c(1)]
            # } else { 
            #     pars_names <- pars_names
            # }
          
            #   pars_names <- bs_model$param_names(  include_tp = TRUE, include_gq = TRUE)
            #   pars_names <- model_results$init_object$init_vals_object$param_names
            n_par_inc_tp_and_gq <- length(pars_names) 
            ##
            ## Names of the nuisance block, as CONSTRAINED names (may exceed n_nuisance
            ## for dimension-changing constraints; may be zero-length when there is no
            ## nuisance block):
            names_nuisance_tracked <- pars_names[seq_len(min(n_nuisance_names, n_par_inc_tp_and_gq))]
            
            index_log_lik  <- grep("^log_lik", pars_names, invert = FALSE)
            names_log_lik <- pars_names[index_log_lik]
            if (length(index_log_lik) == 0) {  
              if (Model_type == "Stan") {  ## if log_lik doesn't exist in Stan model
                warning("No log_lik parameter found in Stan model. Log_lik will not be computed even if save_log_lik = TRUE")
              }
            } 
            
            # tictoc::toc(log = TRUE)
        
        }
        
        {
          
            # tictoc::tic("Timer 5")
          
            n_params  <- length(bs_names)
            n_params_inc_tp <- length(bs_names_inc_tp)
            n_params_inc_tp_and_gq <- length(bs_names_inc_tp_and_gq) 
            
            bs_index <-  1:n_params
            bs_index_inc_tp <-  1:n_params_inc_tp
            bs_index_inc_tp_and_gq <-  1:n_params_inc_tp_and_gq
            
            # Mow find names, indexes and N's of tp and gq ONLY
            index_tp <- setdiff(bs_index_inc_tp, bs_index)
            names_tp <- pars_names[index_tp]
            index_tp_wo_log_lik <- setdiff(index_tp, index_log_lik)
            names_tp_wo_log_lik <- pars_names[index_tp_wo_log_lik]
            ## replace names_tp etc. to be w/o log_lik (as stored in seperate array only if user chooses)
            names_tp <- names_tp_wo_log_lik
            index_tp <- index_tp_wo_log_lik
            ##
            index_gq <- setdiff(bs_index_inc_tp_and_gq, bs_index_inc_tp)
            names_gq <- pars_names[index_gq]
            ##
            n_tp <- length(names_tp)
            n_gq <- length(names_gq)
            ##
            # # exclude nuisance params from summary
            # index_wo_nuisance <- (n_nuisance + 1):n_par_inc_tp_and_gq
            # names_wo_nuisance <- pars_names[index_wo_nuisance]
            # n_params_wo_nuisance <- length(names_wo_nuisance) ; n_params_wo_nuisance
            
            index_wo_nuisance <- 1:n_par_inc_tp_and_gq
            names_wo_nuisance <- pars_names[index_wo_nuisance]
            n_params_wo_nuisance <- length(names_wo_nuisance) ; n_params_wo_nuisance
            
            
            ##
            index_wo_log_lik <-  grep("^log_lik", pars_names, invert = TRUE)
            names_wo_log_lik <-  pars_names[index_wo_log_lik]
            ##
            print(head(index_wo_log_lik))
            print(head(index_wo_nuisance))
            ##
            names_wo_nuisance_and_log_lik <-  intersect(names_wo_log_lik, names_wo_nuisance)
            index_wo_nuisance_and_log_lik <-  intersect(index_wo_log_lik, index_wo_nuisance)
            n_params_wo_nuisance_and_log_lik <- length(index_wo_nuisance_and_log_lik)
            ##
            ## ---- MAIN block within the CONSTRAINED draws: the constrained rows AFTER the
            ## nuisance block. n_params_main is an UNCONSTRAINED count; the constrained main
            ## count can differ whenever the main block has dimension-changing constraints,
            ## so the row range comes from constrained metadata:
            print(paste("n_params_main = ", n_params_main))
            n_params_main_constrained <- n_params - n_nuisance_names
            index_params_main <- (n_nuisance_names + 1):(n_params)
            ##
            print(paste("n_params_main_constrained = ", n_params_main_constrained))
            
            # tictoc::toc(log = TRUE)
            
        }
      
        # print(head(index_params_main))
        
        {
            
             if (sample_nuisance == TRUE) {
                
                   try({  
                      if (n_nuisance_tracked == n_nuisance) {
                        include_nuisance <- TRUE
                      } else { 
                        include_nuisance <- FALSE
                        warning("assumed all nuisance params = 0 as not all nuisance params were tracked during sampling. Hence some outputs (e.g. log_lik) won't be correct.")
                      }
                   })
                
             } else { 
                
                   include_nuisance <- FALSE
                 
             }
             ##
             # pars_indicies_to_track <- 1:n_par_inc_tp_and_gq
             pars_indicies_to_track <- 0:(n_par_inc_tp_and_gq - 1) ## start from 0 as C++ uses 0-based indexing
             n_params_full <- n_par_inc_tp_and_gq
             ##
             options(mc.cores = parallel::detectCores())
             Sys.setenv(OMP_NUM_THREADS = parallel::detectCores())
             RcppParallel::setThreadOptions(numThreads = parallel::detectCores())
             ##
             ##
             read_constrained_traces <- function(result, 
                                                 use_disk,
                                                 n_chains,
                                                 n_iter,
                                                 n_params,
                                                 trace_dir) {
                   
                         if (use_disk == FALSE) {
                           return(result)  # Already in RAM
                         }
                         
                         read_binary_trace <- function( filepath, 
                                                        n_params, 
                                                        n_iter) {
                           
                               con <- file(filepath, "rb")
                               data <- readBin(con, "double", n = n_params * n_iter)
                               close(con)
                               matrix(data, nrow = n_params, ncol = n_iter)
                           
                         }
                         
                         traces <- lapply(0:(n_chains - 1), function(i) {
                             read_binary_trace(
                               filepath = file.path(trace_dir, paste0("chain_", i, "_constrained.bin")),
                               n_params = n_params,
                               n_iter = n_iter)
                         })
                         
                         return(traces)
             
             }
             ##
             # if (use_disk) {
             #   
             #         result <- BayesMVP:::fn_compute_param_constrain_from_trace_parallel( unc_params_trace_input_main = main_trace,
             #                                                                   unc_params_trace_input_nuisance = if (is.null(nuisance_trace)) list() else nuisance_trace,
             #                                                                   pars_indicies_to_track = pars_indicies_to_track,
             #                                                                   n_params_full = n_params_full,
             #                                                                   n_nuisance = n_nuisance,
             #                                                                   n_params_main = n_params_main,
             #                                                                   include_nuisance = include_nuisance,
             #                                                                   model_so_file = model_so_file,
             #                                                                   json_file_path = json_file_path,
             #                                                                   use_disk = TRUE,  # Enable disk mode
             #                                                                   trace_dir = "/tmp/constrain_traces")
             # 
             # } else { 
             #   
             #         result <- BayesMVP:::fn_compute_param_constrain_from_trace_parallel( unc_params_trace_input_main = main_trace,
             #                                                                                 unc_params_trace_input_nuisance = if (is.null(nuisance_trace)) list() else nuisance_trace,
             #                                                                                 pars_indicies_to_track = pars_indicies_to_track,
             #                                                                                 n_params_full = n_params_full,
             #                                                                                 n_nuisance = n_nuisance,
             #                                                                                 n_params_main = n_params_main,
             #                                                                                 include_nuisance = include_nuisance,
             #                                                                                 model_so_file = model_so_file,
             #                                                                                 json_file_path = json_file_path,
             #                                                                                 use_disk = FALSE,
             #                                                                                 trace_dir = "/tmp/constrain_traces")
             #   
             # }
             ##
             ## **** RAM spikes again here (so it ends up being double what it should be), when running "BayesMVP:::fn_compute_param_constrain_from_trace_v2"
             ##
             ## ---- Which LAYOUT does the model being constrained actually use?
             ##
             ## param_constrain runs against the compiled model in 'model_so_file'. For an external
             ## Stan model that is the user's model, whose parameters block DECLARES the nuisance
             ## coordinates, so the complete unconstrained draw is [nuisance | main]. For a BUILT-IN
             ## model it is the cpp SKELETON, and most skeletons (LC_MVP, LC_MVOP, MVP, MVOP)
             ## declare the MAIN parameters only - the latent block lives in the C++ likelihood, not
             ## in the skeleton - so the draw must be main-only. latent_trait is the exception: its
             ## skeleton does declare u_raw.
             ##
             ## Read this off the model's own unconstrained dimension rather than from a list of
             ## model names, so it stays correct if a skeleton changes. The C++ side validates the
             ## same quantity and throws "n_nuisance (...) + n_params_main (...) does not match the
             ## model's unconstrained dimension" when the two disagree.
             ##
             constrain_model_unc_num <- tryCatch(model_results$init_object$bs_model$param_unc_num(),
                                                 error = function(e) NA_integer_)
             ##
             if (!is.na(constrain_model_unc_num) && (constrain_model_unc_num == n_params_main)) {

                   n_nuisance_for_constrain <- 0L
                   nuisance_trace_for_constrain <- list()
                   ##
                   if (n_nuisance > 0) {
                     message("create_summary_and_traces: the model used for param_constrain declares ",
                             n_params_main, " unconstrained coordinates (main only), so the ", n_nuisance,
                             " nuisance coordinates are not part of its parameter vector - constraining ",
                             "from the main block alone.")
                   }

             } else {

                   n_nuisance_for_constrain <- n_nuisance
                   nuisance_trace_for_constrain <- if (is.null(nuisance_trace)) list() else nuisance_trace

             }
             ##
             if (use_disk) {
               
                     result <- BayesMVP:::fn_compute_param_constrain_from_trace_parallel( unc_params_trace_input_main = main_trace,
                                                                                            unc_params_trace_input_nuisance = nuisance_trace_for_constrain,
                                                                                            pars_indicies_to_track = pars_indicies_to_track,
                                                                                            n_params_full = n_params_full,
                                                                                            n_nuisance = n_nuisance_for_constrain,
                                                                                            n_params_main = n_params_main,
                                                                                            model_so_file = model_so_file,
                                                                                            json_file_path = json_file_path,
                                                                                            use_disk = TRUE, # Enable disk mode
                                                                                            trace_dir = use_disk_path_post_hoc_dir)
                     
                     ## Read from disk when needed
                     all_param_outs_trace <- read_constrained_traces( result = result,
                                                                      use_disk = use_disk,
                                                                      n_chains = n_chains,
                                                                      n_iter = n_iter,
                                                                      n_params = n_params_full,
                                                                      trace_dir = use_disk_path_post_hoc_dir)
             
             } else {
                     
                     all_param_outs_trace <- BayesMVP:::fn_compute_param_constrain_from_trace_parallel( unc_params_trace_input_main = main_trace,
                                                                                            unc_params_trace_input_nuisance = nuisance_trace_for_constrain,
                                                                                            pars_indicies_to_track = pars_indicies_to_track,
                                                                                            n_params_full = n_params_full,
                                                                                            n_nuisance = n_nuisance_for_constrain,
                                                                                            n_params_main = n_params_main,
                                                                                            model_so_file = model_so_file,
                                                                                            json_file_path = json_file_path,
                                                                                            use_disk = FALSE, # Enable disk mode
                                                                                            trace_dir = use_disk_path_post_hoc_dir)
             
             }
         
        }
        
         
         # if (index_lp == 1) { 
         #   offset <- 1
         # } else { 
         #   offset <- 0
         # }
        
         {
          
             # tictoc::tic("Timer 7")
             
             offset <- 0
             # offset <- 1
             
             # message(print(paste("offset = ", offset)))
             # 
             # message(print(paste("n_chains_sampling = ", n_chains_sampling)))
             # message(print(paste("n_iter = ", n_iter)))
             # message(print(paste("n_params_main = ", n_params_main)))
             # 
             # message(print(paste("length(all_param_outs_trace) = ", length(all_param_outs_trace))))
             # message(print(paste("length(index_params_main) = ", length(index_params_main))))
             # 
             # message(print(paste("index_params_main = ", index_params_main)))
             
             trace_params_main <- array(dim = c(n_params_main_constrained, n_iter, n_chains_sampling))
             
             # message(print(str(all_param_outs_trace)))
            
             if (compute_main_params == TRUE) {
               kk <- 1
               # all_param_outs_trace[[kk]][index_params_main - offset, 1:n_iter] 
              ##   trace_params_main[1:n_params_main, 1:n_iter, kk] <- all_param_outs_trace[[kk]][index_params_main - offset, 1:n_iter] 
                for (kk in 1:n_chains_sampling) {
                  try({ 
                     # trace_params_main[1:n_params_main, 1:n_iter, kk] <-   all_param_outs_trace[[kk]][index_params_main - offset, 1:n_iter]  
                    trace_params_main[1:n_params_main_constrained, 1:n_iter, kk] <-   all_param_outs_trace[[kk]][index_params_main - offset, 1:n_iter]
                  }, silent = TRUE)
                }
             }
               
             try({
               trace_tp <- NULL
               ##
               message(print(paste("index_tp_wo_log_lik = ")))
               message(print(head(index_tp_wo_log_lik)))
               message(print(length(index_tp_wo_log_lik)))
               ##
               message(print(paste("index_tp = ")))
               message(print(head(index_tp)))
               message(print(length(index_tp)))
               ##
               n_tp_wo_log_lik <- length(index_tp_wo_log_lik)
               
               if (compute_transformed_parameters == TRUE) { 
                 trace_tp <- array(dim = c(n_tp_wo_log_lik, n_iter, n_chains_sampling))
                 if (n_tp_wo_log_lik > 0L) {
                   for (kk in 1:n_chains_sampling) {
                     trace_tp[1:n_tp_wo_log_lik,  1:n_iter, kk] <- all_param_outs_trace[[kk]][index_tp_wo_log_lik - offset, 1:n_iter] #  params_subset_trace[[kk]][param, 1:n_iter]
                   }
                 }
               }
             })
           
             try({ 
               trace_gq <- NULL
               if (compute_generated_quantities == TRUE) {
                   trace_gq <- array(dim = c(n_gq, n_iter, n_chains_sampling))
                   if (n_gq > 0L) {
                     for (kk in 1:n_chains_sampling) {
                       trace_gq[1:n_gq, 1:n_iter, kk] <- all_param_outs_trace[[kk]][index_gq - offset, 1:n_iter] #  params_subset_trace[[kk]][param, 1:n_iter]
                     }
                   }
               }
             })
             
             try({ 
               log_lik_trace <- NULL
               if (save_log_lik_trace == TRUE) {
                 
                     n_log_lik_names <- length(index_log_lik)
                 
                     log_lik_trace <- array(dim = c(n_log_lik_names, n_iter, n_chains_sampling))
                 
                     if (Model_type == "Stan") {
                       
                            if (n_log_lik_names > 0L) {
                              for (kk in 1:n_chains_sampling) {
                                  log_lik_trace[1:n_log_lik_names,  1:n_iter, kk] <- all_param_outs_trace[[kk]][index_log_lik - offset, 1:n_iter] #  params_subset_trace[[kk]][param, 1:n_iter]
                              }
                            }
                       
                     } else {  ## if built-in / manual model
                       
                           log_lik_trace <- log_lik_trace_mnl_models
                         
                     }
                 
               }
             })
             
             # try({ 
             #    nuisance_trace <- NULL
             #    if (save_nuisance_trace == TRUE) {
             #      index_nuisance <- 1:n_nuisance_tracked
             #      nuisance_trace <- array(dim = c(n_nuisance_tracked, n_iter, n_chains_sampling))
             #      for (kk in 1:n_chains_sampling) {
             #        nuisance_trace[1:n_nuisance_tracked,  1:n_iter, kk] <- all_param_outs_trace[[kk]][index_nuisance - offset, 1:n_iter] #  params_subset_trace[[kk]][param, 1:n_iter]
             #      }
             #    }
             # })
          
             n_cores <- parallel::detectCores()
             
             # try({ 
             #   rm(all_param_outs_trace)
             # })
             
             # tictoc::toc(log = TRUE)
         
         }
        
        ## ---- Un-permuting is done in the SKELETON gq (test_perm passed as data).
        ##      Raw main params + tp remain in FITTED order by design (per-fit coords).
        test_perm <- model_results$test_perm
        if (is.null(test_perm) && (Model_type %in% c("LC_MVP", "LC_MVOP"))) {
          warning("create_summary_and_traces: no test_perm in model_results for an LC model -- old run or R_fn_sample.R not updated.")
        }
        
        # ## ---- Un-permute test-indexed params back to ORIGINAL test order  ----------------------
        # ##      (undoes the Dissmann column reordering applied in R_fn_sample_model)
        # ##
        # test_perm <- model_results$test_perm   ## NULL for non-LC models AND for runs saved before this patch
        # ##
        # if (is.null(test_perm)) {
        #         if (Model_type %in% c("LC_MVP", "LC_MVOP")) {
        #           warning(paste0("create_summary_and_traces: no test_perm found in model_results for an LC model -- ",
        #                          "either an old saved run (pre-patch) or R_fn_sample.R wasn't updated. ",
        #                          "Test-indexed params may be in PERMUTED order!"))
        #         }
        # } else if (!identical(as.integer(test_perm), seq_along(test_perm))) {
        #     
        #         message(paste0("Un-permuting test-indexed params (test_perm = ",
        #                        paste(test_perm, collapse = ","), ")"))
        #         ##
        #         names_main_tmp <- head(names_wo_nuisance_and_log_lik, n_params_main)
        #         ##
        #         map_main <- R_fn_get_unpermute_map(names_main_tmp, 
        #                                            test_perm)
        #         if (map_main$changed) trace_params_main <- trace_params_main[map_main$slice_map, , , drop = FALSE]
        #         ##
        #         if (!is.null(trace_tp)) {
        #           map_tp <- R_fn_get_unpermute_map(names_tp_wo_log_lik, 
        #                                            test_perm)
        #           if (map_tp$changed) trace_tp <- trace_tp[map_tp$slice_map, , , drop = FALSE]
        #           if (length(map_tp$skipped_chol) > 0) {
        #             warning(paste0("Cholesky-factor families left in PERMUTED order (cannot reindex a chol): ",
        #                            paste(map_tp$skipped_chol, collapse = ", ")))
        #           }
        #         }
        #         ##
        #         if (!is.null(trace_gq)) {
        #           map_gq <- R_fn_get_unpermute_map(names_gq,
        #                                            test_perm)
        #           if (map_gq$changed) trace_gq <- trace_gq[map_gq$slice_map, , , drop = FALSE]
        #         }
        #   
        # }
        
        ### --------- MAIN PARAMETERS / "PARAMETERS" BLOCK IN STAN  ----------------------------------------
        
        
        {
            # tictoc::tic("Timer 8")
            
            Min_ESS_main <- NULL
            summary_tibble_main_params <- NULL
            names_main <- pars_names[index_params_main]
            
            ## Before calling generate_summary_tibble for main params:
            print(paste("dim trace_params_main:", paste(dim(trace_params_main), collapse=" x ")))
            print(paste("length names_main:", length(names_main)))
            print(paste("n_params_main (unconstrained):", n_params_main))
            print(paste("n_params_main_constrained:", n_params_main_constrained))
            
            if (compute_main_params == TRUE) { 
              
                  summary_tibble_main_params <- generate_summary_tibble(   n_threads = n_cores,
                                                                                      trace = trace_params_main,
                                                                                      param_names = names_main,
                                                                                      n_to_compute = n_params_main_constrained,
                                                                                      compute_nested_rhat = compute_nested_rhat,
                                                                                      n_chains = n_chains_sampling, 
                                                                                      n_superchains = n_superchains,
                                                                                      nested_rhat_grouping = nested_rhat_grouping)
                            
                            
                  ## Missing parameter diagnostics must not silently become infinite efficiency or apparent convergence.
                  main_ess_values <- summary_tibble_main_params$n_eff[seq_len(length.out = n_params_main_constrained)]
                  main_rhat_values <- summary_tibble_main_params$Rhat[seq_len(length.out = n_params_main_constrained)]
                  Min_ESS_main <- if (length(x = main_ess_values) > 0 && all(is.finite(x = main_ess_values)))
                      min(main_ess_values) else NA_real_
                  Max_rhat_main <- if (length(x = main_rhat_values) > 0 && !anyNA(x = main_rhat_values))
                      max(main_rhat_values) else NA_real_
                  Max_nested_rhat_main <- NULL
                  if (compute_nested_rhat == TRUE) {
                     main_nested_rhat_values <- summary_tibble_main_params$n_Rhat[seq_len(length.out = n_params_main_constrained)]
                     Max_nested_rhat_main <- if (length(x = main_nested_rhat_values) > 0 && !anyNA(x = main_nested_rhat_values))
                         max(main_nested_rhat_values) else NA_real_
                  } 
                        
            }
           
            # tictoc::toc(log = TRUE)
        }
      
        ### --------- GENERATED QUANTITIES ---------------------------------------------------------------- 
        
        ## Before calling generate_summary_tibble for gq:
        print(paste("dim trace_tp:", paste(dim(trace_gq), collapse=" x ")))
        print(paste("length names_gq:", length(names_gq)))
        print(paste("n_gq:", n_gq))
        
        {
            # tictoc::tic("Timer 9")
            
            summary_tibble_generated_quantities <- NULL
            
            if  ((compute_generated_quantities == TRUE) && (n_gq > 0))  { 
                      
                  summary_tibble_generated_quantities <- generate_summary_tibble(    n_threads = n_cores,
                                                                                                trace = trace_gq,
                                                                                                param_names = names_gq,
                                                                                                n_to_compute = n_gq,
                                                                                                compute_nested_rhat = compute_nested_rhat,
                                                                                                n_chains = n_chains_sampling, 
                                                                                                n_superchains = n_superchains,
                                                                                                nested_rhat_grouping = nested_rhat_grouping)
                            
              
            }
            
            # tictoc::toc(log = TRUE)
        }
        
        ### --------- TRANSFORMED PARAMETERS -------------------------------------------------------------- 
        
        ## Before calling generate_summary_tibble for tp:
        print(paste("dim trace_tp:", paste(dim(trace_tp), collapse=" x ")))
        print(paste("length names_tp_wo_log_lik:", length(names_tp_wo_log_lik)))
        print(paste("n_tp_wo_log_lik:", n_tp_wo_log_lik))
        
        {
            # tictoc::tic("Timer 10")
            
            summary_tibble_transformed_parameters <- NULL
            if ((compute_transformed_parameters == TRUE) && (n_tp > 0)) {
              
                  ## Remove log-lik trace from tp trace array (since we store it in seperate array called "log_lik_trace" if user chooses to store it)
                  ## trace_tp <- R_fn_remove_log_lik_from_array(trace_tp)
                  
                  summary_tibble_transformed_parameters <- generate_summary_tibble(      n_threads = n_cores,
                                                                                                    trace = trace_tp,
                                                                                                    param_names = names_tp_wo_log_lik,
                                                                                                    n_to_compute = n_tp_wo_log_lik,
                                                                                                    compute_nested_rhat = compute_nested_rhat,
                                                                                                    n_chains = n_chains_sampling, 
                                                                                                    n_superchains = n_superchains,
                                                                                                    nested_rhat_grouping = nested_rhat_grouping)
              
            }
            
            # tictoc::toc(log = TRUE)
        }
         
         #### -----------------------------  DF / tibble creation (for "posterior" and "bayesplot" R packages)  --------------------------------------------
         trace_params_main_tibble <- NULL ;   trace_params_main_reshaped <- NULL
         trace_transformed_params_tibble <- NULL ;   trace_tp_reshaped <- NULL
         trace_generated_quantities_tibble <- NULL ; trace_gq_reshaped <- NULL
      
         ## re-shape data (so can easily use with "posterior" & "bayesplot" R packages)
         trace_params_main_reshaped <-  base::aperm(trace_params_main, c(2, 3, 1))
         if (compute_transformed_parameters == TRUE)   trace_tp_reshaped <- base::aperm(trace_tp, c(2, 3, 1))
         if (compute_generated_quantities == TRUE)     trace_gq_reshaped <- base::aperm(trace_gq, c(2, 3, 1))
        
         ## add the parameter names to the array
         dimnames(trace_params_main_reshaped) <- list( iterations = 1:n_iter, 
                                                       chains = 1:n_chains_sampling, 
                                                       parameters = names_main)
         ##
         if (compute_transformed_parameters == TRUE)  dimnames(trace_tp_reshaped) <- list( iterations = 1:n_iter, 
                                                                                           chains = 1:n_chains_sampling,
                                                                                           parameters = names_tp_wo_log_lik)
         ##
         if (compute_generated_quantities == TRUE) dimnames(trace_gq_reshaped) <- list( iterations = 1:n_iter, 
                                                                                        chains = 1:n_chains_sampling, 
                                                                                        parameters = names_gq)
        
         ## then convert from array -> to df/tibble format  
         if (save_trace_tibbles == TRUE) {
             trace_params_main_tibble <- dplyr::tibble(posterior::as_draws_df(trace_params_main_reshaped))
             if (compute_transformed_parameters == TRUE)  trace_transformed_params_tibble <- dplyr::tibble(posterior::as_draws_df(trace_tp_reshaped))
             if (compute_generated_quantities == TRUE) trace_generated_quantities_tibble <-  dplyr::tibble(posterior::as_draws_df(trace_gq_reshaped))
         }
         ##
         ##  ----- Make OVERALL (full) draws array   --------------- 
         ##
         dims_main <- dim(trace_params_main_reshaped)[3]
         dims_tp <- dim(trace_tp_reshaped)[3]
         dims_gq <- dim(trace_gq_reshaped)[3]
         ##
         dim_total <- dims_main
         if (compute_transformed_parameters == TRUE) dim_total <- dim_total + dims_tp
         if (compute_generated_quantities == TRUE)   dim_total <- dim_total + dims_gq
         ##
         names_total <- names_main
         if (compute_transformed_parameters == TRUE) names_total <- c(names_total, names_tp_wo_log_lik)
         if (compute_generated_quantities == TRUE)   names_total <- c(names_total, names_gq)
         ##
         draws_array <- array(NA, dim = c(n_iter, n_chains_sampling, dim_total))
         draws_array[1:n_iter, 1:n_chains_sampling, 1:dims_main] <- trace_params_main_reshaped # main first 
         ##
         if ((compute_transformed_parameters == TRUE) && (dims_tp > 0L)) {
           draws_array[1:n_iter, 1:n_chains_sampling, (dims_main + 1):(dims_main + dims_tp)] <- trace_tp_reshaped
         }
         if ((compute_generated_quantities == TRUE) && (compute_transformed_parameters == TRUE) && (dims_gq > 0L)) {
           draws_array[1:n_iter, 1:n_chains_sampling, (dims_main + dims_tp + 1):(dims_main + dims_tp + dims_gq)] <- trace_gq_reshaped
         }
         ##
         if ((compute_generated_quantities == TRUE) && (compute_transformed_parameters == FALSE) && (dims_gq > 0L)) {
           draws_array[1:n_iter, 1:n_chains_sampling, (dims_main + 1):(dims_main + dims_gq)] <- trace_gq_reshaped
         }
         if ((compute_generated_quantities == FALSE) && (compute_transformed_parameters == TRUE) && (dims_tp > 0L)) {
           draws_array[1:n_iter, 1:n_chains_sampling, (dims_main + 1):(dims_main + dims_tp)] <- trace_tp_reshaped
         }
         ##
         dimnames(draws_array) <- list( iterations = 1:n_iter, 
                                        chains = 1:n_chains_sampling, 
                                        parameters = names_total)
         ##
         try({
           summaries_toc_result <- tictoc::toc(log = TRUE)
           print(summaries_toc_result)
           tictoc::tic.clearlog()
           ## elapsed seconds straight from toc()'s numbers. Parsing the "X sec elapsed" text with "\\d+\\.\\d+"
           ## gave NA whenever the time rounded to a whole second (tictoc then prints e.g. "2 sec elapsed"):
           time_summaries <- as.numeric(summaries_toc_result$toc - summaries_toc_result$tic)
         })
         ##
         time_total <- time_summaries + time_burnin + time_sampling
         ##
         EHMC_args_as_Rcpp_List <- model_results$burnin_object$EHMC_args_as_Rcpp_List
         ##
         try({
            message(((paste("time_burnin = ", round(time_burnin, 1)))))
            message(((paste("time_sampling = ", round(time_sampling, 1)))))
            message(((paste("time_summaries = ", round(time_summaries, 1)))))
            ##
            message(((paste("time_total = ", round(time_total, 1)))))
            ##
            L_main_during_sampling <- (EHMC_args_as_Rcpp_List$tau_main / EHMC_args_as_Rcpp_List$eps_main)
            message(((paste("L (main) = L_main_during_sampling", round(L_main_during_sampling, 1)))))
         })
      
         ESS_per_sec_samp <- Min_ESS_main / time_sampling
         ESS_per_sec_total <- Min_ESS_main / time_total
        
         try({
            message(((paste("Max R-hat (parameters block, main only) = ", round(Max_rhat_main, 4)))))
         }, silent = TRUE)
         try({
            message(((paste("Max nested R-hat (parameters block, main only) = ", round(Max_nested_rhat_main, 4)))))
         }, silent = TRUE)
         try({
            message(((paste("Min ESS (parameters block, main only) = ", round(Min_ESS_main, 0)))))
            ##
            message(((paste("Min ESS / sec [samp.] (parameters block, main only) = ", signif(ESS_per_sec_samp, 3)))))
            message(((paste("Min ESS / sec [overall] (parameters block, main only) = ", signif(ESS_per_sec_total, 3)))))
         }, silent = TRUE)
         try({ 
            L_main_during_sampling <- (EHMC_args_as_Rcpp_List$tau_main / EHMC_args_as_Rcpp_List$eps_main)
            n_grad_evals_sampling_main <- L_main_during_sampling * n_iter * n_chains_sampling
            Min_ess_per_grad_main_samp <-  Min_ESS_main / n_grad_evals_sampling_main
         }, silent = TRUE)
         try({
            L_us_during_sampling <- (EHMC_args_as_Rcpp_List$tau_us / EHMC_args_as_Rcpp_List$eps_us)
            n_grad_evals_sampling_us <-  L_us_during_sampling  * n_iter * n_chains_sampling
            Min_ess_per_grad_us_samp <-  Min_ESS_main / n_grad_evals_sampling_us
         }, silent = TRUE)
         try({ 
                if (partitioned_HMC == TRUE) { ## i.e. if nuisance are sampledseperately 
                  if (Model_type == "Stan") {
                    weight_nuisance_grad <- 1.0
                    weight_main_grad <- 1.0 
                  } else { 
                    weight_nuisance_grad <- 0.3333333
                    weight_main_grad <- 0.6666667 ## main grad takes ~ 2x as long to compute as nuisance grad 
                  }
                    Min_ess_per_grad_samp_weighted <- (weight_nuisance_grad * Min_ess_per_grad_us_samp + weight_main_grad * Min_ess_per_grad_main_samp) /
                                                      (weight_nuisance_grad + weight_main_grad)
                } else if (partitioned_HMC == FALSE) {  # if not partitioned, grad isnt seperate and "main" grad = "all" grad so use weight_main_grad only !!
                    weight_nuisance_grad <- 0.00
                    weight_main_grad <- 1.00
                    Min_ess_per_grad_samp_weighted <- (weight_nuisance_grad * Min_ess_per_grad_us_samp + weight_main_grad * Min_ess_per_grad_main_samp) / 
                                                      (weight_nuisance_grad + weight_main_grad)
                }
                
            message(((paste("Min ESS / grad [samp., weighted] (parameters block, main only) = ", signif(1000 *  Min_ess_per_grad_samp_weighted, 3)))))
         }, silent = TRUE)
         try({ 
            grad_evals_per_sec <- ESS_per_sec_samp / Min_ess_per_grad_samp_weighted
            message(((paste("Grad evals / sec [samp.] (parameters block, main only) = ", signif(grad_evals_per_sec, 3)))))
         }, silent = TRUE)
         ##
         try({ 
                sampling_time_to_Min_ESS <- time_sampling
                sampling_time_to_100_ESS <- (100 / Min_ESS_main) * sampling_time_to_Min_ESS
                sampling_time_to_1000_ESS <- (1000 / Min_ESS_main) * sampling_time_to_Min_ESS
                sampling_time_to_10000_ESS <- (10000 / Min_ESS_main) * sampling_time_to_Min_ESS
                
                ## w/o summary time
                total_time_to_100_ESS_wo_summaries <-   time_burnin + sampling_time_to_100_ESS
                total_time_to_1000_ESS_wo_summaries <-  time_burnin + sampling_time_to_1000_ESS
                total_time_to_10000_ESS_wo_summaries <- time_burnin + sampling_time_to_10000_ESS
                
                ## w/ summary time (note: assuming that time_summaries scales linearly w/ 
                # the min ESS required (i.e. n_iter and/or n_chains_sampling) Might be over-estimate)
                summary_time_to_Min_ESS <- time_summaries
                summary_time_to_100_ESS <- (100 / Min_ESS_main) * summary_time_to_Min_ESS
                summary_time_to_1000_ESS <- (1000 / Min_ESS_main) * summary_time_to_Min_ESS
                summary_time_to_10000_ESS <- (10000 / Min_ESS_main) * summary_time_to_Min_ESS
                
                total_time_to_100_ESS_with_summaries <-   time_burnin + sampling_time_to_100_ESS   + summary_time_to_100_ESS
                total_time_to_1000_ESS_with_summaries <-  time_burnin + sampling_time_to_1000_ESS  + summary_time_to_1000_ESS
                total_time_to_10000_ESS_with_summaries <- time_burnin + sampling_time_to_10000_ESS + summary_time_to_10000_ESS
         })
         ##
         if (n_divs > 0) { 
            warning("divergences detected!")
            try({ 
              message(print(paste("Number of divergences = ", n_divs)))
              message(print(paste("% divergences = ", pct_divs)))
            })
         } else { 
            message("No divergences detected!")
         }
         ##
         ## ---------  Make R lists to output  --------------------------------------
         ##
         ## list to store summary tibbles / DF's
         summary_tibbles <- list( summary_tibble_main_params = summary_tibble_main_params,
                                  summary_tibble_transformed_parameters = summary_tibble_transformed_parameters,
                                  summary_tibble_generated_quantities = summary_tibble_generated_quantities)
         ##
         ## list to store traces (as 3D arrays)
         ##
         traces_as_arrays <- list(draws_array = draws_array,
                                   trace_params_main = trace_params_main_reshaped,
                                   trace_transformed_params = trace_tp_reshaped,
                                   trace_generated_quantities = trace_gq_reshaped)
         ##
         ## list to store traces (as tibbles/DF's)
         ##
         traces_as_tibbles <- list( trace_params_main_tibble = trace_params_main_tibble,
                                    trace_transformed_params_tibble = trace_transformed_params_tibble,
                                    trace_generated_quantities_tibble = trace_generated_quantities_tibble)
         ##
         HMC_info <- list(  tau_main = EHMC_args_as_Rcpp_List$tau_main,
                            eps_main = EHMC_args_as_Rcpp_List$eps_main,
                            tau_us = EHMC_args_as_Rcpp_List$tau_us,
                            eps_us = EHMC_args_as_Rcpp_List$eps_us,
                            n_chains_sampling = n_chains_sampling,
                            n_chains_burnin = n_chains_burnin,
                            n_iter = n_iter,
                            n_burnin = n_burnin,
                            LR_main = LR_main,
                            LR_us = LR_us,
                            adapt_delta = adapt_delta,
                            metric_type_main = metric_type_main,
                            metric_shape_main = metric_shape_main,
                            metric_type_nuisance = metric_type_nuisance,
                            metric_shape_nuisance = metric_shape_nuisance,
                            diffusion_HMC = diffusion_HMC,
                            diffusion_HMC_integrator = if (is.null(EHMC_args_as_Rcpp_List$diffusion_HMC_integrator)) {
                              "kick_flow_kick"
                            } else {
                              EHMC_args_as_Rcpp_List$diffusion_HMC_integrator
                            },
                            partitioned_HMC = partitioned_HMC,
                            n_superchains = n_superchains,
                            nested_rhat_grouping = nested_rhat_grouping,
                            burnin_schedule = model_results$burnin_schedule,
                            interval_width_main = interval_width_main,
                            interval_width_nuisance = interval_width_nuisance,
                            force_autodiff = force_autodiff,
                            force_PartialLog = force_PartialLog,
                            multi_attempts = multi_attempts,
                            ##
                            test_perm = model_results$test_perm)
          ##
          ## list to store efficiency information
          ##
          efficiency_info <- list(              n_iter = n_iter,
                                                ##
                                                Max_rhat_main = Max_rhat_main,
                                                Max_nested_rhat_main = Max_nested_rhat_main,
                                                ##
                                                Min_ESS_main = Min_ESS_main, 
                                                ESS_per_sec_samp = ESS_per_sec_samp, 
                                                ESS_per_sec_total = ESS_per_sec_total,
                                                ##
                                                time_burnin = time_burnin, 
                                                time_sampling = time_sampling, 
                                                time_summaries = time_summaries,
                                                time_total_wo_summaries = time_total_wo_summaries, 
                                                time_total = time_total, 
                                                ##
                                                ## End-of-iteration adaptation settings; NULL for older burn-in objects.
                                                L_main_during_burnin_vec = burnin_object$L_main_during_burnin_vec,
                                                L_us_during_burnin_vec = burnin_object$L_us_during_burnin_vec,
                                                tau_main_during_burnin_vec = burnin_object$tau_main_during_burnin_vec,
                                                eps_main_during_burnin_vec = burnin_object$eps_main_during_burnin_vec,
                                                tau_us_during_burnin_vec = burnin_object$tau_us_during_burnin_vec,
                                                eps_us_during_burnin_vec = burnin_object$eps_us_during_burnin_vec,
                                                ChEES_criterion_ema_vec = burnin_object$ChEES_criterion_ema_vec,
                                                ChEES_per_tau_gradient_vec = burnin_object$ChEES_per_tau_gradient_vec,
                                                tau_adaptation_version = burnin_object$tau_adaptation_version,
                                                tau_adaptation_iteration_vec = burnin_object$tau_adaptation_iteration_vec,
                                                L_main_during_burnin = L_main_during_burnin,
                                                L_main_during_sampling = L_main_during_sampling,
                                                L_us_during_burnin = L_us_during_burnin,
                                                L_us_during_sampling = L_us_during_sampling,
                                                ##
                                                Min_ess_per_grad_samp_weighted = Min_ess_per_grad_samp_weighted,
                                                grad_evals_per_sec = grad_evals_per_sec,
                                                ##
                                                sampling_time_to_Min_ESS = sampling_time_to_Min_ESS,
                                                sampling_time_to_100_ESS = sampling_time_to_100_ESS,
                                                sampling_time_to_1000_ESS = sampling_time_to_1000_ESS,
                                                sampling_time_to_10000_ESS = sampling_time_to_10000_ESS,
                                                ##
                                                total_time_to_100_ESS_wo_summaries = total_time_to_100_ESS_wo_summaries,
                                                total_time_to_1000_ESS_wo_summaries = total_time_to_1000_ESS_wo_summaries,
                                                total_time_to_10000_ESS_wo_summaries = total_time_to_10000_ESS_wo_summaries,
                                                ##
                                                total_time_to_100_ESS_with_summaries = total_time_to_100_ESS_with_summaries,
                                                total_time_to_1000_ESS_with_summaries = total_time_to_1000_ESS_with_summaries,
                                                total_time_to_10000_ESS_with_summaries = total_time_to_10000_ESS_with_summaries)
          ##
          ## Final lists to output
          ##
          if (save_log_lik_trace == FALSE) log_lik_trace <- NULL
          ##
          if (save_nuisance_trace == TRUE) { 
            
                ## ---- the CONSTRAINED nuisance draws (first n_nuisance_names rows of the
                ## reconstructed output), only if requested:
                if (n_nuisance_names > 0 && exists("all_param_outs_trace") && !is.null(all_param_outs_trace)) {
                  nuisance_trace <- array(dim = c(n_nuisance_names, n_iter, n_chains_sampling))
                  ##
                  for (kk in 1:n_chains_sampling) {
                    nuisance_trace[1:n_nuisance_names, 1:n_iter, kk] <- all_param_outs_trace[[kk]][1:n_nuisance_names, 1:n_iter]
                  }
                } else {
                  nuisance_trace <- NULL
                }
                
          } else {
            nuisance_trace <- NULL
          }
          ##
          traces <- list(traces_as_arrays = traces_as_arrays,
                         traces_as_tibbles = traces_as_tibbles,
                         log_lik_trace = log_lik_trace,
                         nuisance_trace = nuisance_trace)
          ##
          divergences <- list( n_divs = n_divs, 
                               pct_divs = pct_divs)
          ##
          summaries <- list(summary_tibbles = summary_tibbles,
                            divergences = divergences,
                            efficiency_info = efficiency_info, 
                            HMC_info = HMC_info)
          ##
          output_list <- list(  summaries = summaries, ### summary info (incl. efficiency info + divergences)
                                traces = traces      ### trace arrays + trace tibbles
                                ## all_param_outs_trace = all_param_outs_trace
          )
          
          # ### store output (optional)
          # if (store_outputs == TRUE) {
          #   if (is.null(store_outputs_dir)) { 
          #     store_outputs_dir <- getwd() # store in users wd if no dir specified
          #   }
          #   saveRDS(object = output_list, file = paste("BayesMVP_seed_", seed, 
          #                                              "Model_type_", Model_type, 
          #                                              "N_", N, 
          #                                              ))
          #   
          # }
                                                
          
        ### output 
        return(output_list)
  
}



# Now call it with your actual data:
# summary_table <- create_stan_summary(your_trace_vector, pars_names_wo_nuisance)
