


#' generate_summary_tibble
#' @keywords internal
#' @export
generate_summary_tibble <- function(n_threads = NULL,
                                    trace, 
                                    param_names, 
                                    n_to_compute, 
                                    compute_nested_rhat,
                                    n_chains, 
                                    n_superchains,
                                    nested_rhat_grouping = NULL) {
  
        
              n_cores <- round(parallel::detectCores() / 2, 0)
              n_threads <- n_cores

              #### Initialize summary dataframe
              summary_df <- data.frame(     parameter = param_names,
                                            mean = NA,
                                            sd = NA,
                                            `2.5%` = NA,
                                            `50%` = NA,
                                            `97.5%` = NA,
                                            n_eff = NA,
                                            Rhat = NA,
                                            n_Rhat = NA,
                                            check.names = FALSE)
              
              # # Effective Sample Size (ESS) and Rhat - using the fast custom RcppParallel fn "BayesMVP::Rcpp_compute_MCMC_diagnostics()"
              posterior_draws_as_std_vec_of_mats <- list()
              ##
              n_iter <- dim(trace)[2]
              mat <- matrix(nrow = n_iter, ncol = n_chains)

              n_params <- n_to_compute
              
             #  message(print(paste("str(trace) = ")))
             #  message(print(paste(str(trace))))
             # # message(print(str(posterior_draws_as_std_vec_of_mats)))

              for (i in 1:n_params) {
                posterior_draws_as_std_vec_of_mats[[i]] <- mat
                for (kk in 1:n_chains) {
                  posterior_draws_as_std_vec_of_mats[[i]][1:n_iter, kk] <- trace[i, 1:n_iter, kk] 
                }
              }
              
              if (n_params < n_threads) { n_threads = n_params }
              
              #### Compute summary stats using custom Rcpp/C++ functions:
              outs <-  (BayesMVP:::Rcpp_compute_chain_stats(   posterior_draws_as_std_vec_of_mats,
                                                    stat_type = "mean",
                                                    n_threads = n_threads))
              means_between_chains <- outs$statistics[, 1]
              

              outs <-  (BayesMVP:::Rcpp_compute_chain_stats(   posterior_draws_as_std_vec_of_mats,
                                                    stat_type = "sd",
                                                    n_threads = n_threads))
              SDs_between_chains <- outs$statistics[, 1]
              

              outs <-  (BayesMVP:::Rcpp_compute_chain_stats(   posterior_draws_as_std_vec_of_mats,
                                                    stat_type = "quantiles",
                                                    n_threads = n_threads))
              quantiles_between_chains <- outs$statistics
              
              
              #### Compute Effective Sample Size (ESS) and Rhat using custom Rcpp/C++ functions 
              outs <-  (BayesMVP:::Rcpp_compute_MCMC_diagnostics(  posterior_draws_as_std_vec_of_mats,
                                                        diagnostic = "split_ESS_rank",
                                                        n_threads = n_threads))
              ess_vec <- outs$diagnostics[, 1]
              # ess_tail_vec <- outs$diagnostics[, 2]
              ##
              outs <-  (BayesMVP:::Rcpp_compute_MCMC_diagnostics(  posterior_draws_as_std_vec_of_mats,
                                                        diagnostic = "split_rhat_rank",
                                                        n_threads = n_threads))
              rhat_bulk_vec <- outs$diagnostics[, 1]
              rhat_tail_vec <- outs$diagnostics[, 2]
              rhat_vec <- pmax(rhat_bulk_vec, rhat_tail_vec)
              Max_rhat_rank <- max(rhat_vec, na.rm = TRUE)
              ##
              for (i in seq_len(n_to_compute)) {
                
                        #### Get all values for this parameter across iterations and chains
                        param_values <- as.vector(trace[i, , ])
                        
                        #### Calculate summary statistics
                        summary_df$mean[i] <- means_between_chains[i]
                        summary_df$sd[i] <- SDs_between_chains[i]
                        try({  
                            summary_df[i, c("2.5%", "50%", "97.5%")] <- quantiles_between_chains[i, ]
                            #### summary_df[i, c("2.5%", "50%", "97.5%")] <- quantiles_between_chains[, i]
                        })
                        summary_df$n_eff[i] <- round(ess_vec[i])
                        summary_df$Rhat[i] <- rhat_vec[i]
                
              }
              
              if (compute_nested_rhat == TRUE) {
                
                        if (is.null(x = nested_rhat_grouping)) {
                            nested_rhat_grouping <-  fn_prepare_nested_rhat_grouping(
                                superchain_ids = create_superchain_ids(n_chains = n_chains, n_superchains = n_superchains),
                                source = "caller_supplied_nominal_groups")
                        }
                        summary_df$n_Rhat[seq_len(length.out = n_to_compute)] <-  fn_nested_rhat_from_draws_array(
                            draws_array = aperm(a = trace[seq_len(length.out = n_to_compute), , , drop = FALSE], perm = c(2, 3, 1)),
                            nested_rhat_grouping = nested_rhat_grouping)
                                    
              }
              
              summary_tibble <- tibble::tibble(summary_df)
              print(summary_tibble, n = 100)
              
              return(summary_tibble)

    
}





 










