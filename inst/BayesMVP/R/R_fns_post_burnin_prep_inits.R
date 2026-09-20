 

#' R_fn_avg_matrix_columns
#' @keywords internal
#' @export
R_fn_avg_matrix_columns <- function(matrix,
                                    n_chains_per_superchain) {
  
          n_params <- nrow(matrix)
          n_chains <- ncol(matrix)
          
          # Calculate the number of full superchains
          n_full_superchains <- floor(n_chains / n_chains_per_superchain)
          
          # Process full superchains
          for (i in 1:n_full_superchains) {
            start_col <- (i - 1) * n_chains_per_superchain + 1
            end_col <- i * n_chains_per_superchain
            subset <- matrix[, start_col:end_col, drop = FALSE]
            avg <- rowMeans(subset)
            matrix[, start_col:end_col] <- matrix[, rep(start_col, n_chains_per_superchain)]
          }
          
          # Handle remaining columns if any
          if (n_full_superchains * n_chains_per_superchain < n_chains) {
            start_col <- n_full_superchains * n_chains_per_superchain + 1
            subset <- matrix[, start_col:n_chains, drop = FALSE]
            avg <- rowMeans(subset)
            matrix[, start_col:n_chains] <- matrix[, rep(start_col, ncol(subset))]
          }
          
          return(matrix)
  
}








#' R_fn_grp_mat_cols_by_first_col_per_superchain_grp
#' @keywords internal
#' @export
R_fn_grp_mat_cols_by_first_col_per_superchain_grp <- function( matrix,
                                                               n_chains_per_superchain) {
  
        n_params <- nrow(matrix)
        n_chains <- ncol(matrix)
        
        # Calculate the number of full superchains
        n_full_superchains <- floor(n_chains / n_chains_per_superchain)
        
        # Process full superchains
        for (i in 1:n_full_superchains) {
          start_col <- (i - 1) * n_chains_per_superchain + 1
          end_col <- i * n_chains_per_superchain
          subset <- matrix[, start_col:end_col, drop = FALSE]
          # Take first column of subset and copy it across
          matrix[, start_col:end_col] <- matrix[, rep(start_col, n_chains_per_superchain)]
        }
        
        # Handle remaining columns if any
        if (n_full_superchains * n_chains_per_superchain < n_chains) {
          start_col <- n_full_superchains * n_chains_per_superchain + 1
          subset <- matrix[, start_col:n_chains, drop = FALSE]
          # Take first column of remaining subset and copy it across
          matrix[, start_col:n_chains] <- matrix[, rep(start_col, ncol(subset))]
        }
        
        return(matrix)
  
}





#' R_fn_grp_mat_cols_by_median_per_superchain_grp
#' @keywords internal
#' @export
R_fn_grp_mat_cols_by_median_per_superchain_grp <- function(matrix,
                                                           n_chains_per_superchain) {
  
        n_params <- nrow(matrix)
        n_chains <- ncol(matrix)
        
        # Calculate the number of full superchains
        n_full_superchains <- floor(n_chains / n_chains_per_superchain)
        
        # Process full superchains
        for (i in 1:n_full_superchains) {
          start_col <- (i - 1) * n_chains_per_superchain + 1
          end_col <- i * n_chains_per_superchain
          subset <- matrix[, start_col:end_col, drop = FALSE]
          
          # Compute row-wise median across the superchain
          median_vec <- apply(subset, 1, mean)
          
          # Replicate median across all columns in superchain
          matrix[, start_col:end_col] <- rep(median_vec, times = n_chains_per_superchain)
        }
        
        # Handle remaining columns if any
        if (n_full_superchains * n_chains_per_superchain < n_chains) {
          start_col <- n_full_superchains * n_chains_per_superchain + 1
          subset <- matrix[, start_col:n_chains, drop = FALSE]
          
          # Compute row-wise median across remaining chains
          median_vec <- apply(subset, 1, mean)
          
          # Replicate median across remaining columns
          matrix[, start_col:n_chains] <- rep(median_vec, times = ncol(subset))
        }
        
        return(matrix)
  
}





#' R_fn_post_burnin_prep_for_sampling
#' @keywords internal
#' @export
R_fn_post_burnin_prep_for_sampling <- function(n_chains_burnin,
                                               n_chains_sampling, 
                                               n_superchains, 
                                               n_params_main, 
                                               n_nuisance, 
                                               theta_main_vectors_all_chains_input_from_R, 
                                               theta_us_vectors_all_chains_input_from_R)  {
  
        if (n_superchains > n_chains_sampling) {
          n_superchains <- n_chains_sampling
        }
        
        n_chains_per_superchain <- n_chains_sampling / n_superchains ; n_chains_per_superchain
        
        if (n_superchains * n_chains_per_superchain > n_chains_sampling) { 
          
          warning("n_superchains * n_chains_per_superchain > n_chains_sampling - lowering n_chains_sampling as necessary")
          
          while (n_superchains * n_chains_per_superchain > n_chains_sampling) {
            n_chains_sampling <- n_chains_sampling - 1
          }
          
        }
        
        ###  source(file.path(pkg_dir, "/R/fn_compute_inits_for_chains.R"))
        
        theta_main_vectors_all_chains_input_from_R_sampling <- array(dim = c(n_params_main, n_chains_sampling))
        theta_us_vectors_all_chains_input_from_R_sampling <-   array(0.0, dim = c(n_nuisance, n_chains_sampling))
        
        # Fixed loop: cycle through burnin columns with modular arithmetic
        for (kk in 1:n_chains_sampling) { 
          # Calculate which burnin column to use (cycles from 1 to n_chains_burnin)
          burnin_col <- ((kk - 1) %% n_chains_burnin) + 1
          
          try({
            theta_main_vectors_all_chains_input_from_R_sampling[,kk] <- theta_main_vectors_all_chains_input_from_R[,burnin_col]
          })
          try({ 
            theta_us_vectors_all_chains_input_from_R_sampling[,kk] <- theta_us_vectors_all_chains_input_from_R[,burnin_col]
          })
        }
        
        # re-assign
        theta_main_vectors_all_chains_input_from_R <- theta_main_vectors_all_chains_input_from_R_sampling
        theta_us_vectors_all_chains_input_from_R <- theta_us_vectors_all_chains_input_from_R_sampling
        
        ## R_fn_grp_mat_cols_by_first_col_per_superchain_grp
        ## R_fn_grp_mat_cols_by_median_per_superchain_grp
        ##
        #n_superchains <- 4  # we will have n_superchains sets of initial values (chains within same superchains share same init's)
        theta_main_vectors_all_chains_input_from_R <- R_fn_grp_mat_cols_by_median_per_superchain_grp( theta_main_vectors_all_chains_input_from_R,
                                                                                                         n_chains_per_superchain)
        
        # n_superchains <- 4   # we will have n_superchains sets of initial values (chains within same superchains share same init's)
        try({  
          if (n_nuisance > 0) {
            theta_us_vectors_all_chains_input_from_R <- R_fn_grp_mat_cols_by_median_per_superchain_grp( theta_us_vectors_all_chains_input_from_R,
                                                                                                           n_chains_per_superchain)
          }
        })
        
        
        return(list(theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                    theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R))
  
}






























