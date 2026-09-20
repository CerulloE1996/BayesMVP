


## helper function:
#' adj_diag_M
#' @export
adj_diag_M <- function(M_diag) { 
  
        M_diag_inv <- recip(M_diag)
        max_variance <- max(M_diag_inv)
        adj_M_diag_inv <- M_diag_inv / max_variance
        adj_M_diag <- recip(adj_M_diag_inv)
        
        return(list(M_diag = M_diag,
                    M_diag_inv = M_diag_inv,
                    max_variance = max_variance,
                    adj_M_diag_inv = adj_M_diag_inv,
                    adj_M_diag = adj_M_diag))
}





## helper function:
#' adj_diag_inv_M
#' @export
adj_diag_inv_M <- function(M_diag_inv) { 
          
          M_diag <- recip(M_diag_inv)
          max_variance <- max(M_diag_inv)
          adj_M_diag_inv <- M_diag_inv / max_variance
          adj_M_diag <- recip(adj_M_diag_inv)
          
          return(list(M_diag = M_diag,
                      M_diag_inv = M_diag_inv,
                      max_variance = max_variance,
                      adj_M_diag_inv = adj_M_diag_inv,
                      adj_M_diag = adj_M_diag))
}



## helper function:
#' is_valid
#' @export
is_valid <- function(x) {
       return( !(is.null(x)) && all(!is.na(x)) && all(!is.nan(x)) && all(is.finite(x)) )
}




## helper function:
#' if_not_NA_or_INF_else
#' @export
if_not_NA_or_INF_else <- function(x, alt_x) { 
        
        success_indicator <- TRUE
        if (is_valid(x) == TRUE) {
          return(x)
        } else { 
          return(alt_x)
        }
        
}



## helper function:
#' recip  
#' @export
recip <- function(x) { 
        
        if (is.vector(x)) {
          
          recip_x <- 1.0 / x
          return(recip_x)
          
        } else if (is.matrix(x)) { 
          
          diag_of_x <- diag(x)
          recip_diag_x <- 1.0 / diag_of_x
          recip_mat_x <- diag(recip_diag_x)
          return(recip_mat_x)
          
        } else {
          
          stop("ERROR: x must be either a vector or a matrix")
          
        }
  
}




#' init_EHMC_Metric_as_Rcpp_List
#' @export
init_EHMC_Metric_as_Rcpp_List   <- function(  n_params_main,
                                              n_nuisance,
                                              metric_shape_main) {
  
        n_params <- n_params_main + n_nuisance
        try({  
          index_nuisance <- seq_len(n_nuisance)
        })
        
        M_diag_vec <- rep(1, n_params)
        
        M_dense_main <- diag(n_params_main)
        M_inv_dense_main <- M_dense_main
        M_inv_dense_main_chol <- M_dense_main
        
        M_inv_main_vec <- matrix(c(diag(M_inv_dense_main)))
        M_main_vec <- matrix(c(1 / M_inv_main_vec))
        
        ## ncol = 1 keeps these TRUE column vectors when n_nuisance == 0
        ## (plain matrix() turns numeric(0) into a 1x1 NA matrix, which the
        ## C++ metric conversion reads as a length-1 NaN vector, desynchronised
        ## from the 0-length nuisance buffers in HMCResult).
        M_inv_us_vec <- matrix(1 / M_diag_vec[index_nuisance], ncol = 1)
        M_us_vec <- matrix(M_diag_vec[index_nuisance], ncol = 1)
        
        theta_hat_us_vec = matrix(rep(0, n_nuisance), ncol = 1)
        
        EHMC_Metric_as_Rcpp_List <- list( 
          ### for main params
          M_dense_main = M_dense_main,
          M_inv_dense_main = M_inv_dense_main,
          M_inv_dense_main_chol = M_inv_dense_main_chol,
          ##
          M_inv_main_vec = M_inv_main_vec,
          M_main_vec = M_main_vec,
          ### for nuisance
          M_inv_us_vec = M_inv_us_vec, 
          M_us_vec = M_us_vec,
          theta_hat_us_vec = theta_hat_us_vec,
          ### shape of main metric
          metric_shape_main = metric_shape_main)
        
        return(EHMC_Metric_as_Rcpp_List)
  
}




#' init_EHMC_args_as_Rcpp_List
#' @param diffusion_HMC_integrator Joint diffusion integrator: "kick_flow_kick" (default) or "flow_kick_flow".
#' @export
init_EHMC_args_as_Rcpp_List   <- function( diffusion_HMC,
                                         diffusion_HMC_integrator = "kick_flow_kick") {

        if (!is.character(x = diffusion_HMC_integrator) || length(x = diffusion_HMC_integrator) != 1L ||
            is.na(x = diffusion_HMC_integrator) ||
            !(diffusion_HMC_integrator %in% c("kick_flow_kick", "flow_kick_flow"))) {
            stop("diffusion_HMC_integrator must be 'kick_flow_kick' or 'flow_kick_flow'.")
        }
  
        tau_main <- 1
        tau_main_ii <- 1
        eps_main <- 1
        tau_us <- 1
        tau_us_ii <- 1
        eps_us <- 1
        
        EHMC_args_as_Rcpp_List <- list(
          ### for main params
          tau_main = tau_main,
          tau_main_ii = tau_main_ii,
          eps_main = eps_main,
          ### for nuisance
          tau_us = tau_us,
          tau_us_ii = tau_us_ii,
          eps_us = eps_us,
          diffusion_HMC = diffusion_HMC,
          diffusion_HMC_integrator = diffusion_HMC_integrator)
        
        return(EHMC_args_as_Rcpp_List)
  
}



#' init_EHMC_burnin_as_Rcpp_List
#' @export
init_EHMC_burnin_as_Rcpp_List   <- function(n_params_main,
                                            n_nuisance,
                                            adapt_delta,
                                            LR_main,
                                            LR_us) {
        
        n_params <- n_params_main + n_nuisance
        index_main <- (1 + n_nuisance):n_params
        try({  
          index_nuisance <- seq_len(n_nuisance)
        })
        
        ##### set ADAM-related params (initialise)
        ### for main params
        adapt_delta_main <- adapt_delta
        LR_main <- LR_main
        eps_m_adam_main <-  1.0
        eps_v_adam_main <-  1.0
        tau_m_adam_main <-  1.0
        tau_v_adam_main <-  1.0
        index_main <- index_main
        M_dense_sqrt <- matrix(c(diag(n_params_main)))
        snaper_m_vec_main  <- matrix(c(rep(1, n_params_main)))
        snaper_w_vec_main  <- matrix(c(rep(0.01, n_params_main)))
        
        
        ### for NUISANCE params
        adapt_delta_us <- adapt_delta
        LR_us <- LR_us
        eps_m_adam_us <-  1.0
        eps_v_adam_us <-  1.0
        tau_m_adam_us <-  1.0
        tau_v_adam_us <-  1.0
        index_nuisance <- index_nuisance
        sqrt_M_us_vec <- matrix(rep(1, n_nuisance), ncol = 1)
        snaper_m_vec_us  <- matrix(rep(1, n_nuisance), ncol = 1)
        snaper_w_vec_us  <- matrix(rep(0.01, n_nuisance), ncol = 1)
        
        
        ### for main params
        adapt_delta_main <- adapt_delta_main
        LR_main <- LR_main
        eps_m_adam_main <- eps_m_adam_main
        eps_v_adam_main <- eps_v_adam_main
        tau_m_adam_main <- tau_m_adam_main
        tau_v_adam_main <- tau_v_adam_main
        index_main <- index_main
        M_dense_sqrt <- M_dense_sqrt
        snaper_m_vec_main <- snaper_m_vec_main
        snaper_w_vec_main <- snaper_w_vec_main
        eigen_max_main <- 0
        eigen_vector_main <- matrix(c(rep(0, n_params_main)))
        
        ### for NUISANCE params
        adapt_delta_us <- adapt_delta_us
        LR_us <- LR_us
        eps_m_adam_us <- eps_m_adam_us
        eps_v_adam_us <- eps_v_adam_us
        tau_m_adam_us <- tau_m_adam_us
        tau_v_adam_us <- tau_v_adam_us
        index_nuisance <- index_nuisance
        M_dense_sqrt <- M_dense_sqrt
        snaper_m_vec_us <- snaper_m_vec_us
        snaper_w_vec_us <- snaper_w_vec_us
        eigen_max_us <- 0 
        eigen_vector_us <- matrix(c(rep(0, n_nuisance)), ncol = 1)
        
        # ------------ put in the list to put into C++
        EHMC_burnin_as_Rcpp_List <- list(   ### for main params
          adapt_delta_main = adapt_delta_main,
          LR_main = LR_main,
          eps_m_adam_main = eps_m_adam_main,
          eps_v_adam_main = eps_v_adam_main,
          tau_m_adam_main = tau_m_adam_main,
          tau_v_adam_main = tau_v_adam_main,
          eigen_max_main =  eigen_max_main,
          index_main = index_main,
          M_dense_sqrt = M_dense_sqrt,
          snaper_m_vec_main = snaper_m_vec_main,
          snaper_w_vec_main = snaper_w_vec_main,
          eigen_vector_main = eigen_vector_main,
          ### for nuisance
          adapt_delta_us = adapt_delta_us,
          LR_us = LR_us,
          eps_m_adam_us = eps_m_adam_us,
          eps_v_adam_us = eps_v_adam_us,
          tau_m_adam_us = tau_m_adam_us,
          tau_v_adam_us = tau_v_adam_us,
          eigen_max_us =  eigen_max_us,
          index_us = index_nuisance,
          sqrt_M_us_vec = sqrt_M_us_vec,
          snaper_m_vec_us = snaper_m_vec_us,
          snaper_w_vec_us = snaper_w_vec_us,
          eigen_vector_us = eigen_vector_us)
        
        
        return(EHMC_burnin_as_Rcpp_List)
  
}







## helper function:
#' update_M_diag_Empirical_overall
#' @export
update_M_diag_Empirical_overall <- function(EHMC_Metric_as_Rcpp_List,
                                            EHMC_burnin_as_Rcpp_List,
                                            proposed_variance_vec_main,
                                            proposed_variance_vec_nuisance,
                                            ratio) {
  
            n_nuisance <- length(proposed_variance_vec_nuisance)
            n_params_main <- length(proposed_variance_vec_main)
            n_params <- n_nuisance + n_params_main
            
            index_nuisance <- seq_len(n_nuisance)
            index_main <- (n_nuisance + 1):n_params
            
            # Ensure proposed variances are positive
            proposed_variance_vec_main <- pmax(proposed_variance_vec_main, 1e-6)
            proposed_variance_vec_nuisance <- pmax(proposed_variance_vec_nuisance, 1e-6)
            
            proposed_variance_vec_ALL <- c(proposed_variance_vec_nuisance, proposed_variance_vec_main)
            
            M_inv_main_vec_current <- c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec)
            M_inv_nuisance_vec_current <- c(EHMC_Metric_as_Rcpp_List$M_inv_us_vec)
            
            # Ensure current values are positive
            M_inv_main_vec_current <- pmax(M_inv_main_vec_current, 1e-6)
            M_inv_nuisance_vec_current <- pmax(M_inv_nuisance_vec_current, 1e-6)
            
            M_inv_vec_current_ALL <- c(M_inv_nuisance_vec_current, M_inv_main_vec_current)
            
            # Compute new values and ensure they're positive
            M_inv_ALL_diag <- ratio * proposed_variance_vec_ALL + (1.0 - ratio) * M_inv_vec_current_ALL
            M_inv_ALL_diag <- pmax(M_inv_ALL_diag, 1e-6)  # Ensure positive
            
            M_inv_ALL_diag_sqrt <- sqrt(M_inv_ALL_diag)
            M_ALL_diag <- 1.0 / M_inv_ALL_diag
            M_ALL_diag_sqrt <- sqrt(M_ALL_diag)
            
            # Update for MAIN
            M_inv_main_diag <- M_inv_ALL_diag[index_main]
            M_main_diag <- M_ALL_diag[index_main]
            M_main_diag_sqrt <- M_ALL_diag_sqrt[index_main]
            
            EHMC_Metric_as_Rcpp_List$M_main_vec <- M_main_diag
            EHMC_Metric_as_Rcpp_List$M_inv_main_vec <- M_inv_main_diag
            EHMC_burnin_as_Rcpp_List$sqrt_M_main_vec <- M_main_diag_sqrt
            
            # Update for NUISANCE
            M_inv_us_diag <- M_inv_ALL_diag[index_nuisance]
            M_us_diag <- M_ALL_diag[index_nuisance]
            M_us_diag_sqrt <- M_ALL_diag_sqrt[index_nuisance]
            
            EHMC_Metric_as_Rcpp_List$M_us_vec <- M_us_diag
            EHMC_Metric_as_Rcpp_List$M_inv_us_vec <- M_inv_us_diag
            EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec <- M_us_diag_sqrt
            
            return(list(EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                        EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List))
  
}



## helper function:
#' update_M_diag_Empirical_nuisance
#' @export
update_M_diag_Empirical_nuisance <- function( EHMC_Metric_as_Rcpp_List,
                                              EHMC_burnin_as_Rcpp_List,
                                              proposed_variance_vec,
                                              ratio_M_nuisance,
                                              ii,
                                              n_adapt,
                                              M_decay_type,
                                              M_decay_power,
                                              M_decay_scale
) { 

          ##
          ratio_M_nuisance_effective <- compute_ratio_M( debug = debug,
                                                         ratio_M = ratio_M_nuisance, 
                                                         ii = ii, 
                                                         n_adapt = n_adapt,
                                                         M_decay_type = M_decay_type,
                                                         M_decay_power = M_decay_power,
                                                         M_decay_scale = M_decay_scale)
          ##
          # M_inv_as_snaper_s_vec_us_empirical_scaled <- EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical
          ## now update M_inv_nuisance:
          EHMC_Metric_as_Rcpp_List$M_inv_us_vec <- c(ratio_M_nuisance_effective * c(proposed_variance_vec) + 
                                                     (1.0 - ratio_M_nuisance_effective) * c(EHMC_Metric_as_Rcpp_List$M_inv_us_vec))
          EHMC_Metric_as_Rcpp_List$M_us_vec <- recip(c(EHMC_Metric_as_Rcpp_List$M_inv_us_vec))
          ##
          EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec  <- sqrt(EHMC_Metric_as_Rcpp_List$M_us_vec)
          ##
          return(list(  EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                        EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List))

}



## helper function:
#' compute_ratio_M
#' @export
compute_ratio_M <- function(   debug,
                               ratio_M, 
                               ii, 
                               n_adapt,
                               M_decay_type = NULL,
                               M_decay_power = NULL,
                               M_decay_scale = NULL
) { 
  
          if (!is.null(ii) && !is.null(n_adapt) && M_decay_type != "none") {
          ##
          if (is.null(M_decay_type)) { 
            M_decay_type <- "inverse"
          }
          if (is.null(M_decay_scale)) { 
            M_decay_scale <- n_adapt / 5
            # M_decay_scale <- n_adapt / 100
          }
          if (is.null(M_decay_power)) { 
            M_decay_power <- 0.50
          }
          ##
          ratio_M_effective <- switch(M_decay_type,
                                      ##
                                      ## Inverse decay (Stan-style): ratio / (1 + ii/scale)^power
                                      ## Starts at ratio_M, decays smoothly
                                      ##
                                      "inverse" = {
                                        ratio_M / (1 + ii / M_decay_scale)^M_decay_power
                                      },
                                      ##
                                      ## Linear decay: ratio * (1 - ii/n_adapt)
                                      ## Goes from ratio_M to 0
                                      ##
                                      "linear" = {
                                        ratio_M * max(0, 1 - ii / n_adapt)
                                      },
                                      ##
                                      ## Exponential decay: ratio * exp(-rate * ii)
                                      ##
                                      "exponential" = {
                                        rate <- -log(0.01) / n_adapt  # decays to 1% by end
                                        ratio_M * exp(-rate * ii)
                                      },
                                      ##
                                      ## Cosine annealing: smooth decay
                                      ##
                                      "cosine" = {
                                        ratio_M * 0.5 * (1 + cos(pi * ii / n_adapt))
                                      },
                                      ##
                                      ## Default: no decay
                                      ##
                                      ratio_M
          )
          
          # if (debug) {
          #   print(paste("ii =", ii, "| ratio_M =", round(ratio_M, 4), 
          #               "| ratio_M_effective =", round(ratio_M_effective, 4)))
          # }
          
        } else {
          ratio_M_effective <- ratio_M
        }
  
        return(ratio_M_effective)

}




## helper function:
#' update_M_Empirical_main
#' @export
update_M_Empirical_main <- function( debug,
                                     metric_shape_main,
                                     ##
                                     EHMC_Metric_as_Rcpp_List,
                                     EHMC_burnin_as_Rcpp_List,
                                     ##
                                     empicical_cov_main = NULL,
                                     ratio_M_main,
                                     ## 
                                     ii, 
                                     n_adapt,
                                     M_decay_type,
                                     M_decay_power,
                                     M_decay_scale,
                                     ##
                                     ## Multiplies the proposed (co)variance before it is blended into the metric.
                                     ## 1 = unchanged. metric_estimator = "chain_mean_scaled" passes n_chains_burnin, since
                                     ## the chain-mean estimate is ~ Sigma / n_chains_burnin for independent chains.
                                     variance_scale = 1,
                                    ## the symmetric square root M_dense_sqrt is only read by the SNAPER functions; the ChESSR
                                    ## burn-in never uses it, so it passes FALSE and skips a pracma::sqrtm() per metric update:
                                    compute_M_dense_sqrt = TRUE
) {

        if (debug) { 
          print(paste("calling update_M_Empirical_main"))
        }
        ##
        ratio_M_effective <- compute_ratio_M(  debug = debug,
                                               ratio_M = ratio_M_main, 
                                               ii = ii, 
                                               n_adapt = n_adapt,
                                               M_decay_type = M_decay_type,
                                               M_decay_power = M_decay_power,
                                               M_decay_scale = M_decay_scale)
        ##
        if (metric_shape_main == "diag") { ## "not a matrix" error
 
                ##
                ## Inverse diag-metric:
                ##
                proposed_variance_vec  <- variance_scale * c(EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical)
                M_inv_main_vec_current <- c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec)
                M_inv_main_diag        <- c(ratio_M_effective * c(proposed_variance_vec) + (1.0 - ratio_M_effective) * c(M_inv_main_vec_current))
                M_inv_main_diag_sqrt   <- c(sqrt(M_inv_main_diag))
                ##
                ## Diag-Metric:
                ##
                M_main_diag      <- c(recip(M_inv_main_diag))
                M_main_diag_sqrt <- c(sqrt(M_main_diag))
                ##
                ## Update C++ lists:
                ##
                EHMC_Metric_as_Rcpp_List$M_main_vec <- M_main_diag
                EHMC_Metric_as_Rcpp_List$M_inv_main_vec  <- c(M_inv_main_diag)
                ##
                EHMC_burnin_as_Rcpp_List$sqrt_M_main_vec <- c(M_main_diag_sqrt)
                ##
                ## Dummy dense parameters:
                ##
                EHMC_Metric_as_Rcpp_List$M_dense_main          <- diag(EHMC_Metric_as_Rcpp_List$M_main_vec)
                EHMC_burnin_as_Rcpp_List$M_dense_sqrt          <- diag(EHMC_burnin_as_Rcpp_List$sqrt_M_main_vec)
                EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- diag(EHMC_Metric_as_Rcpp_List$M_inv_main_vec)
                EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- diag(EHMC_burnin_as_Rcpp_List$sqrt_M_main_vec)
 
          
                # outs <- update_M_diag_Empirical_main(     
                #               EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                #               EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                #               proposed_variance_vec = c(EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical),
                #               ratio_M_effective = ratio_M_effective)
              
                # EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                # EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                ##
                # M_main_diag <- EHMC_Metric_as_Rcpp_List$M_main_vec
                # M_inv_main_diag <- EHMC_Metric_as_Rcpp_List$M_inv_main_vec
                # M_main_diag_sqrt <-  sqrt(  EHMC_Metric_as_Rcpp_List$M_main_vec )
                # M_inv_main_diag_sqrt <- sqrt(M_inv_main_diag)
                # ###
                # M_main_diag <- EHMC_Metric_as_Rcpp_List$M_main_vec
                # M_inv_main_diag <- EHMC_Metric_as_Rcpp_List$M_inv_main_vec
                # M_main_diag_sqrt <-  sqrt(  EHMC_Metric_as_Rcpp_List$M_main_vec )
                # ##
                # ## Dummy dense parameters:
                # ##
                # EHMC_Metric_as_Rcpp_List$M_dense_main          <- diag(10)
                # EHMC_burnin_as_Rcpp_List$M_dense_sqrt          <- diag(10)
                # EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- diag(10)
                # EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- diag(10)
          
        } else if (metric_shape_main == "dense") { ## this works (N = 500)
          
                ##
                M_inv_dense_main_current <- EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                ##
                M_inv_dense_main_new <- ( ratio_M_effective * variance_scale * empicical_cov_main + (1.0 - ratio_M_effective) *  M_inv_dense_main_current )
                # ## inside update_M_Empirical_main dense branch, or as a special-case at the last update:
                # p <- nrow(empicical_cov_main)
                # M_inv_dense_main_new <- 0.9 * empicical_cov_main + 0.1 * mean(diag(empicical_cov_main)) * diag(p)
                ##
                M_inv_dense_main_new_PD <- BayesMVP:::Rcpp_near_PD(M_inv_dense_main_new)
                ##
                EHMC_Metric_as_Rcpp_List$M_dense_main          <- BayesMVP:::Rcpp_solve(M_inv_dense_main_new_PD)
                if (compute_M_dense_sqrt) EHMC_burnin_as_Rcpp_List$M_dense_sqrt <- pracma::sqrtm(EHMC_Metric_as_Rcpp_List$M_dense_main)[[1]]
                EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- M_inv_dense_main_new_PD
                EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- BayesMVP:::Rcpp_Chol(M_inv_dense_main_new_PD)
                ##
                ## And update the diagonal:
                ##
                M_inv_main_diag <- diag(M_inv_dense_main_new_PD)
                M_main_diag <- recip(M_inv_main_diag)
                M_main_diag_sqrt     <- sqrt(M_main_diag)
                ##
                EHMC_Metric_as_Rcpp_List$M_main_vec  <- M_main_diag
                EHMC_Metric_as_Rcpp_List$M_inv_main_vec  <- M_inv_main_diag
                EHMC_burnin_as_Rcpp_List$sqrt_M_main_vec <- M_main_diag_sqrt
          
        }
        
        return(list(  EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                      EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List))
  
}






## helper function:
#' update_M_Hessian_main
#' @export
update_M_Hessian_main <- function( metric_shape_main,
                                   ##
                                   EHMC_Metric_as_Rcpp_List,
                                   EHMC_burnin_as_Rcpp_List,
                                   ##
                                   ratio_M_main,
                                   interval_width_main,
                                   ##
                                   force_autodiff_for_metric,
                                   force_PartialLog_for_metric,
                                   force_multi_attempts_for_metric,
                                   ##
                                   shrinkage_factor,
                                   num_diff_e = 0.0001,
                                   ##
                                   Model_type,
                                   y,
                                   ##
                                   Model_args_as_Rcpp_List,
                                   ii,
                                   n_adapt,
                                   main_vec_for_Hessian
) {

        if (metric_shape_main == "dense") { 
          
                ## //////////// update M_dense (for main param) using Hessian (num_diff)
                try({
                  
                  Rcpp_update_M_dense_using_Hessian_num_diff <- BayesMVP:::fn_Rcpp_wrapper_update_M_dense_main_Hessian(
                    M_dense_main = EHMC_Metric_as_Rcpp_List$M_dense_main,
                    M_inv_dense_main = EHMC_Metric_as_Rcpp_List$M_inv_dense_main,
                    M_inv_dense_main_chol = EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol,
                    shrinkage_factor = shrinkage_factor,
                    ratio_Hess_main = ratio_M_main,
                    interval_width = interval_width_main,
                    num_diff_e = num_diff_e,
                    Model_type = Model_type,
                    force_autodiff = force_autodiff_for_metric,
                    force_PartialLog = force_PartialLog_for_metric,
                    multi_attempts = force_multi_attempts_for_metric,
                    theta_main_vec = main_vec_for_Hessian,
                    theta_us_vec = EHMC_burnin_as_Rcpp_List$snaper_m_vec_us,
                    y = y,
                    Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                    ii = ii,
                    n_burnin = n_adapt,
                    metric_type = "Hessian")
                  
                  M_dense_main_non_scaled <-     Rcpp_update_M_dense_using_Hessian_num_diff[[1]]
                  M_inv_dense_main_non_scaled <- Rcpp_update_M_dense_using_Hessian_num_diff[[2]]
                  M_inv_dense_main_chol_non_scaled <- Rcpp_update_M_dense_using_Hessian_num_diff[[3]]
                  
                  ## Diag-Metric:
                  M_main_diag          <- diag(M_dense_main_non_scaled)
                  M_main_diag_sqrt     <- sqrt(M_main_diag)
                  ## inverse Diag-metric:
                  M_inv_main_diag      <- recip(M_main_diag)
                  M_inv_main_diag_sqrt <- sqrt(M_inv_main_diag)
                  
                  EHMC_Metric_as_Rcpp_List$M_dense_main          <- M_dense_main_non_scaled
                  if (compute_M_dense_sqrt) EHMC_burnin_as_Rcpp_List$M_dense_sqrt <- pracma::sqrtm(M_dense_main_non_scaled)[[1]]
                  EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- M_inv_dense_main_non_scaled
                  EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- M_inv_dense_main_chol_non_scaled
                  ##
                  ## And still update the diagonal parameters:
                  ##
                  EHMC_Metric_as_Rcpp_List$M_main_vec  <- M_main_diag
                  EHMC_Metric_as_Rcpp_List$M_inv_main_vec  <- M_inv_main_diag ## c(diag(EHMC_Metric_as_Rcpp_List$M_inv_dense_main))
                  EHMC_burnin_as_Rcpp_List$sqrt_M_main_vec <- M_main_diag_sqrt ##sqrt(EHMC_burnin_as_Rcpp_List$M_dense_sqrt)
                  
                })
          
        } else if (metric_shape_main == "diag") { 
          
                ## //////////// update M_dense (for main param) using Hessian (num_diff)
                try({
                  
                  Rcpp_update_M_diag_using_Hessian_num_diff <- BayesMVP:::fn_Rcpp_wrapper_update_M_diag_Hessian( 
                    M_main_vec = EHMC_Metric_as_Rcpp_List$M_main_vec,
                    M_inv_main_vec = EHMC_Metric_as_Rcpp_List$M_inv_main_vec,
                    shrinkage_factor = shrinkage_factor,
                    ratio_Hess_main = ratio_M_main,
                    interval_width = interval_width_main,
                    num_diff_e = num_diff_e,
                    Model_type = Model_type,
                    force_autodiff = force_autodiff_for_metric,
                    force_PartialLog = force_PartialLog_for_metric,
                    multi_attempts = force_multi_attempts_for_metric,
                    theta_main_vec = main_vec_for_Hessian,
                    theta_us_vec = EHMC_burnin_as_Rcpp_List$snaper_m_vec_us,
                    y = y,
                    Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                    ii = ii,
                    n_burnin = n_adapt,
                    metric_type = "Hessian")
                  
                  M_main_diag <-     Rcpp_update_M_diag_using_Hessian_num_diff[[1]]
                  M_inv_main_diag <- Rcpp_update_M_diag_using_Hessian_num_diff[[2]]
                  ##
                  M_main_diag_sqrt     <- sqrt(M_main_diag)
                  M_inv_main_diag_sqrt <- sqrt(M_inv_main_diag)
                  ##
                  EHMC_Metric_as_Rcpp_List$M_main_vec <- M_main_diag
                  EHMC_Metric_as_Rcpp_List$M_inv_main_vec <-  M_inv_main_diag
                  EHMC_burnin_as_Rcpp_List$sqrt_M_main_vec <- M_main_diag_sqrt
                  ##
                  ## And still update the "dense" parameters:
                  ##
                  EHMC_Metric_as_Rcpp_List$M_dense_main          <- diag(10)
                  EHMC_burnin_as_Rcpp_List$M_dense_sqrt          <- diag(10)
                  EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- diag(10)
                  EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- diag(10)
                  
                })
          
        }
  
        return(list(  EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                      EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List))

}










