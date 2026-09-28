 


## -| ------------------ R function to simulate a single dataset for (binary) LC_MVP  --------------------------------------------


simulate_binary_LC_MVP_data <- function( N_vec = c(500, 1000, 2500, 5000, 12500, 25000),
                                         seed = 123,
                                         DGP = 5) {
  
  N_datasets <- length(N_vec)
  n_tests <- 5 # fixed to 5
  
  ## set the seed (keep this OUTSIDE the N loop):
  set.seed(seed, kind = "L'Ecuyer-CMRG") 
  
  ## Make storage lists:
  y_binary_list <- list()
  Sigma_nd_true_observed_list <- Sigma_d_true_observed_list <- list()
  prev_true_observed_list <- Se_true_observed_list <- Sp_true_observed_list <- list()
  Phi_Se_observed_list <- Phi_Fp_observed_list <- list()
  true_correlations_observed_vec_list <- observed_table_probs_list <- true_estimates_observed_list <- observed_cell_counts_list <- list()
  
  ii_dataset <- 0 ## counter
  
  for (N in N_vec) {
    
                      ii_dataset <- ii_dataset + 1 ## increment counter
                      
                      Se_set_1 <- c(0.60, 0.55, 0.60, 0.65, 0.70)
                      Sp_set_1 <- c(0.99, 0.95, 0.90, 0.90, 0.85)
                      
                      Se_set_2 <- c(0.925, 0.86, 0.87, 0.91, 0.86)
                      Sp_set_2 <- c(0.95,  0.81, 0.70, 0.67, 0.85)
                      
                      
                      Sigma_CI <- diag(n_tests)
                      
                      
                      b_vec <- c(0.36,  1.10, 1.10, 1.25, 1.50)
                      Sigma_from_bs <- diag(n_tests) + t(t(b_vec)) %*% (t(b_vec))
                      Omega_from_bs <- round(cov2cor(Sigma_from_bs), 2) ; Omega_from_bs
                      Sigma_varied <- Omega_from_bs
                      
                      
                      Sigma_highly_varied <- matrix(c(1,  0,     0,        0,        0,
                                                      0,  1,     0.50,     0.25,     0,
                                                      0,  0.50,  1,        0.40,     0.40,
                                                      0,  0.25,  0.40,     1,        0.70,
                                                      0,  0,     0.40,     0.70,     1),
                                                    n_tests, n_tests)
                      
                      
                      if (DGP == 1) { # Conditional independence (CI)
                        
                            true_Fp_vec <-        1 - Sp_set_1
                            true_Se_vec <-            Se_set_1
                            
                            Sigma_d <- Sigma_CI
                            Sigma_nd <-  Sigma_CI
                        
                      } else if (DGP == 2) { # CD in D+ group, CI in D- group. CD in D+ group is quite uniform and relatively high
                        
                            true_Fp_vec <-        1 - Sp_set_1
                            true_Se_vec <-            Se_set_1
                            
                            Sigma_d <- Sigma_highly_varied
                            Sigma_nd <-  Sigma_CI
                        
                      } else if ( DGP == 3) {  # CD in D+ group, CI in D- group. CD in D+ group is very NON-uniform and relatively high, but test 1 is uncorrelated to all other tests and test 2 is uncorrelated to test 5, only weakly correlated to test 4 but strongly correlated to test 3. Tests 3,4,5 all correlated (w/ correlation between 0.50-0.80).
                            
                            true_Fp_vec <-        1 - Sp_set_1
                            true_Se_vec <-            Se_set_1
                            
                            Sigma_d <- Sigma_varied
                            Sigma_nd <-  Sigma_CI
                        
                      } else if (DGP == 4) {  # CD in both groups - with CD in D+ group 2x that of CD in D- group. CD pattern is the same as that in DGP 3 (i.e. NON-uniform pattern).
                        
                            true_Fp_vec <-        1 - Sp_set_1
                            true_Se_vec <-            Se_set_1
                            
                            Sigma_d <- Sigma_varied
                            Sigma_nd <-  0.5 * Sigma_varied
                            diag(Sigma_nd) <- rep(1, n_tests)
                        
                        
                      } else if (DGP == 5) {
                        
                            true_Fp_vec <-        1 - Sp_set_1
                            true_Se_vec <-            Se_set_1
                            
                            Sigma_d <- Sigma_highly_varied
                            Sigma_nd <-  0.5 * Sigma_highly_varied
                            diag(Sigma_nd) <- rep(1, n_tests)
                        
                      } else if (DGP == 6) {
                        
                            true_Fp_vec <-        1 - Sp_set_2
                            true_Se_vec <-            Se_set_2
                            
                            Sigma_d <- Sigma_varied
                            Sigma_nd <-  0.5 * Sigma_varied
                            diag(Sigma_nd) <- rep(1, n_tests)
                        
                      } else if (DGP == 7) {
                        
                            true_Fp_vec <-        1 - Sp_set_2
                            true_Se_vec <-            Se_set_2
                            
                            Sigma_d <- Sigma_varied
                            Sigma_nd <-  1 * Sigma_varied
                        
                      }
                      
                      L_Sigma_d  = t(chol(Sigma_d)) # PD check (fails if not PD)
                      L_Sigma_nd  = t(chol(Sigma_nd)) # PD check (fails if not PD)
                      
                      true_prev <- 0.20 # low-ish prevalence (quite common in diagnostic/screening test studies to have <25% prevalence)
                      #### true_prev <- 0.40 # high (relatively) prevalence (same prev. used in Wang et al, 2017)
                      
                      d_ind <- sort(rbinom(n= N, size = 1, prob = true_prev))
                      n_pos <- sum(d_ind)
                      
                      n_neg <- N - sum(d_ind)
                      latent_results_neg <- LaplacesDemon::rmvn(n = n_neg, mu = qnorm(true_Fp_vec), Sigma = Sigma_nd)
                      latent_results_pos <- LaplacesDemon::rmvn(n = n_pos, mu = qnorm(true_Se_vec), Sigma = Sigma_d)
                      latent_results <- rbind(latent_results_neg, latent_results_pos)
                      results_neg <- ifelse(latent_results_neg > 0, 1, 0)
                      results_pos <- ifelse(latent_results_pos > 0, 1, 0)
                      results <- rbind(results_neg, results_pos)
                      y <- results
                      
                      df <- dplyr::tibble(results,latent_results,d_ind)
                      df_pos <- dplyr::filter(df, d_ind == 1)
                      df_neg <- dplyr::filter(df, d_ind == 0)
                      
                      Sigma_nd_true_observed <- Sigma_d_true_observed <- array(dim = c(n_tests, n_tests))
                      observed_correlations <- array(dim = c(n_tests, n_tests))
                      
                      for (i in 2:n_tests) {
                        for (j in 1:(i-1)) {
                          Sigma_nd_true_observed[i, j] <- cor(df_neg$latent_results[,i], df_neg$latent_results[,j])
                          Sigma_nd_true_observed[j, i] <-  Sigma_nd_true_observed[i, j]
                          Sigma_d_true_observed[i, j] <- cor(df_pos$latent_results[,i], df_pos$latent_results[,j])
                          Sigma_d_true_observed[j, i] <-  Sigma_d_true_observed[i, j]
                          observed_correlations[i, j] <- cor(y[, i], y[, j])
                          observed_correlations[j, i] <-  observed_correlations[i, j]
                        }
                      }
                      
                      prev_observed <-  print(round(sum(d_ind)/N, 3))
                      
                      # Se
                      Phi_Se_observed_vec <- Phi_Fp_observed_vec <- c()
                      for (i in 1:n_tests) {
                        Phi_Se_observed_vec[i] <- qnorm(sum(df_pos$results[,i])/nrow(df_pos))
                      }
               
                      # Fp / Sp
                      for (i in 1:n_tests) {
                        Phi_Fp_observed_vec[i] <-  qnorm( 1.0 - ((nrow(df_neg) - sum(df_neg$results[,i]))/nrow(df_neg))  )
                      }
                      
                      prev_true_observed  <-  print(round(sum(d_ind)/N, 3))
                      Se_true_observed <-     print(round(pnorm(Phi_Se_observed_vec), 3))
                      Sp_true_observed <-    print(round(pnorm(-Phi_Fp_observed_vec), 3))
                      
                      true_corrs_observed_vec <- observed_correlations[upper.tri(observed_correlations )]
                      
                      observed_table <- table(y[, 1], y[, 2], y[, 3], y[, 4], y[, 5])
                      observed_table_probs_vec <- c(unlist(round(prop.table(observed_table), 4)))
                      
                      observed_cell_counts <- observed_table_probs_vec * N
                      
                      true_estimates_observed <-  c(Sigma_nd_true_observed[upper.tri(Sigma_nd_true_observed )],  Sigma_d_true_observed[upper.tri(Sigma_d_true_observed )], Sp_true_observed,  Se_true_observed, prev_true_observed ,
                                                    true_corrs_observed_vec, observed_table_probs_vec, NA, NA)
                      true_estimates  <-  c(Sigma_nd[upper.tri(Sigma_nd )],  Sigma_d[upper.tri(Sigma_d )], 1 - true_Fp_vec,  true_Se_vec, true_prev  ,
                                            rep(NA, length(true_corrs_observed_vec)), rep(NA, length(observed_table_probs_vec)), NA, NA)
                      
                      ## Print some key quantities:
                      print(paste("N = ", N))
                      print(paste("prev_observed = ", prev_observed))
                      print(paste("Se_true_observed = ", Se_true_observed))
                      print(paste("Sp_true_observed = ", Sp_true_observed))
                      
                      ## Populate lists:
                      y_binary_list[[ii_dataset]] <- y
                      ##
                      Sigma_nd_true_observed_list[[ii_dataset]] <- Sigma_nd_true_observed
                      Sigma_d_true_observed_list[[ii_dataset]] <- Sigma_d_true_observed
                      ##
                      Phi_Se_observed_list[[ii_dataset]] <- Phi_Se_observed_vec
                      Phi_Fp_observed_list[[ii_dataset]] <- Phi_Fp_observed_vec
                      ##
                      prev_true_observed_list[[ii_dataset]] <- prev_true_observed
                      Se_true_observed_list[[ii_dataset]] <- Se_true_observed
                      Sp_true_observed_list[[ii_dataset]] <- Sp_true_observed
                      ##
                      true_correlations_observed_vec_list[[ii_dataset]] <- true_corrs_observed_vec
                      observed_table_probs_list[[ii_dataset]] <- observed_table_probs_vec
                      true_estimates_observed_list[[ii_dataset]] <- true_estimates_observed
                      observed_cell_counts_list[[ii_dataset]] <- observed_cell_counts
              
          
            
            # if (N == 500)   y_master_list_seed_123_datasets[[1]] <- y_list[[123]]
            # if (N == 1000)  y_master_list_seed_123_datasets[[2]] <- y_list[[123]]
            # if (N == 2500)  y_master_list_seed_123_datasets[[3]] <- y_list[[123]]
            # if (N == 5000)  y_master_list_seed_123_datasets[[4]] <- y_list[[123]]
            # if (N == 12500) y_master_list_seed_123_datasets[[5]] <- y_list[[123]]
            # if (N == 25000) y_master_list_seed_123_datasets[[6]] <- y_list[[123]]
    
  }
  
  
  # # assess sparsity in D- class
  # {
  #   print((sum(df_neg[,1]$results[,1]) / length(df_neg[,1]$results[,1]) ) *100)
  #   print((sum(df_neg[,1]$results[,2]) / length(df_neg[,1]$results[,1]) ) *100)
  #   print((sum(df_neg[,1]$results[,3]) / length(df_neg[,1]$results[,1]) ) *100)
  #   print((sum(df_neg[,1]$results[,4]) / length(df_neg[,1]$results[,1]) ) *100)
  #   print((sum(df_neg[,1]$results[,5]) / length(df_neg[,1]$results[,1]) ) *100)
  # }
  
      return(list(
                    y_binary_list = y_binary_list,
                    ## 
                    Sigma_nd_true_observed_list = Sigma_nd_true_observed_list,
                    Sigma_d_true_observed_list = Sigma_d_true_observed_list,
                    ##
                    Phi_Se_observed_list = Phi_Se_observed_list,
                    Phi_Fp_observed_list = Phi_Fp_observed_list,
                    ##
                    prev_true_observed_list = prev_true_observed_list,
                    Se_true_observed_list = Se_true_observed_list,
                    Sp_true_observed_list = Sp_true_observed_list,
                    ##
                    true_correlations_observed_vec_list = true_correlations_observed_vec_list,
                    observed_table_probs_list = observed_table_probs_list,
                    true_estimates_observed_list = true_estimates_observed_list,
                    observed_cell_counts_list = observed_cell_counts_list
      ))
  
  
}






















