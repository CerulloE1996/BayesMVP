
  
 

# - |  |  | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 




{
  
  source("load_R_packages.R")
  source("load_data_binary_LC_MVP_sim.R")
  
}


 

# - |  |  | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 

 
  n_covariates <- 1
  n_class = 2
  

{
    n_covariates_per_outcome_vec <- c(array(1, dim = c(n_tests, 1)))
        
    if (Model_type == "latent_trait") {
       prior_b_shape_d <-  1.33 ; prior_b_scale_d <-    1.25    # ~ equiv. to  truncated-LKJ(1.5)
       prior_b_shape_nd <- 1.52 ; prior_b_scale_nd <-   0.633 # ~ equiv. to  truncated-LKJ(10)
    }
     
         
    CI <- 0
    corr_force_positive <- 0
      
    tailored_corr_priors <- FALSE
    # tailored_corr_priors <- TRUE
     
    bs_to_set_to_0 = array(0, dim = c(n_class, n_tests)) # for LT model
    
    prior_a_mean <-   array(0,  dim = c(n_class, n_tests, n_covariates))
    prior_a_sd  <-    array(1,  dim = c(n_class, n_tests, n_covariates))
     
    n_pops <- 1
    group <- rep(1, N)
 
}


#  |  ------------------------------------------------------------------------------------
 

{

# ## "non-informative"
  prior_mean_vec = t(prior_a_mean[,,1])
  prior_sd_vec = t(prior_a_sd[,,1])
 
corr_normal_prior_sd <- list()
corr_normal_prior_sd[[1]] <-  array(0.25, dim = c(n_tests, n_tests))
corr_normal_prior_sd[[2]] <-  array(0.25, dim = c(n_tests, n_tests))
  
}

 
 
  
  df_i <- 123
  
 
  
  prior_only <- FALSE
 

   #     corr_param <- "latent_trait"
        corr_param <- "Chol_Schur"
 
        
        if (Model_type == "latent_trait")   corr_param <- "latent_trait"
    
    corr_pos_offset <- 0 
 
    
         #   corr_force_positive = FALSE
           corr_force_positive = TRUE
 
    experiment <- "algorithm_binary"
   # experiment <- "simulation_binary"
 
    prior_sd_vec[,] = 1
    prior_mean_vec[,] = 0
       
       
       known_values_indicator_list <- known_values_list <- list()
       
       for (c in 1:n_class) {
         known_values_indicator_list[[c]] <- diag(n_tests)
         known_values_list[[c]] <- diag(n_tests)
       }
       
       
          tailored_corr_priors <- FALSE
     #   tailored_corr_priors <- TRUE
       
       if (tailored_corr_priors == TRUE) {
           if (DGP == 4) {
               known_values_indicator_list[[1]][1,2:5] <-    known_values_indicator_list[[1]][2:5, 1] <- 1
               known_values_indicator_list[[1]][2,5] <-    known_values_indicator_list[[1]][5, 2] <- 1
               known_values_indicator_list[[2]][1,2:5] <-    known_values_indicator_list[[2]][2:5, 1] <- 1
               known_values_indicator_list[[2]][2,5] <-    known_values_indicator_list[[2]][5, 2] <- 1
           }
           if (DGP == 3) {
             known_values_indicator_list[[1]][,]   <- 1
             known_values_indicator_list[[1]][,]   <- 1
             known_values_indicator_list[[2]][1,2:5] <-    known_values_indicator_list[[2]][2:5, 1] <- 1
             known_values_indicator_list[[2]][2,5] <-    known_values_indicator_list[[2]][5, 2] <- 1
           }
           if (DGP == 2) {
             known_values_indicator_list[[1]][,]   <- 1
             known_values_indicator_list[[1]][,]   <- 1
           } 
           if (DGP == 1) {
             known_values_indicator_list[[1]][,]   <- 1
             known_values_indicator_list[[1]][,]   <- 1
             known_values_indicator_list[[2]][,]   <- 1
             known_values_indicator_list[[2]][,]   <- 1
           } 
       } else { 
         for (c in 1:n_class) {
           known_values_indicator_list[[c]] <- diag(n_tests)
           known_values_list[[c]] <- diag(n_tests)
         }
       }
        
          corr_prior_beta <-  0
          
          
         if (Model_type == "latent_trait")  {
           
            LT_b_priors_shape <- array(1, dim = c(n_class, n_tests))
            LT_b_priors_scale <- array(1, dim = c(n_class, n_tests))

              LT_b_priors_shape[1, ] <-    prior_b_shape_nd
              LT_b_priors_scale[1, ] <-    prior_b_scale_nd

              # LT prior set 1
            LT_b_priors_shape[2, ] <-       prior_b_shape_d
            LT_b_priors_scale[2, ] <-       prior_b_scale_d


                LT_known_bs_indicator <-  LT_known_bs_values <-  array(0, dim = c(n_class, n_tests))


          }
          # 
          # 
            
              # corr_param <- "Chol_Stan"
              corr_param <- "Chol_Schur"
              if (Model_type == "latent_trait")   corr_param <- "latent_trait"
              # lkj_cholesky_eta = c(10, 2)
              lkj_cholesky_eta = c(12, 3)
              corr_force_positive = FALSE
              tailored_corr_priors <- FALSE
 
              
              df_i = 123

              prior_mean_vec[1,1] =  -2.10 ;    prior_sd_vec[1,1] = 0.45
              prior_mean_vec[1,2] = +0.40 ;   prior_sd_vec[1,2] = 0.375
   
          
       if (Model_type == "MVP_LC") {
         
              beta_prior_mean_vec <- beta_prior_sd_vec <-  list()
              for (c in 1:n_class) {
                beta_prior_mean_vec[[c]] <- array(0.0, dim = c(1, n_tests))
                beta_prior_sd_vec[[c]] <- array(1.0, dim = c(1, n_tests))
              }
              
              beta_prior_mean_vec[[1]][1, 1] <- -2.10
              beta_prior_mean_vec[[2]][1, 1] <- 0.40
              
              beta_prior_sd_vec[[1]][1, 1] <- 0.450
              beta_prior_sd_vec[[2]][1, 1] <- 0.375
              
              n_covariates_per_outcome_vec_1 <- n_covariates_per_outcome_vec
              n_covariates_per_outcome_vec <- list(n_covariates_per_outcome_vec_1, n_covariates_per_outcome_vec_1)
              
       }
          
          
 
     
 
    
    
    {
      
            
            nuisance_transformation <- "Phi" # seems correct
            
            
            
            #  vect_type = "AVX2"
            vect_type = "AVX512"
            #  vect_type = "Stan" 
            #   vect_type = "Loop"
            
            #  Phi_type = "Phi"
            Phi_type = "Phi_approx"
            #    Phi_type = "Phi_approx_2"im 
            
            #
            # inv_Phi_type = "inv_Phi"
            inv_Phi_type = "inv_Phi_approx"
            
            #  skip_checks = TRUE 
            skip_checks = FALSE 
            
            
            overflow_threshold <-   +5  ;            underflow_threshold <-  -5
            
    }
    
          
      
      
      

















