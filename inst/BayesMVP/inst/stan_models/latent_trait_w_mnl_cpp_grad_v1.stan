functions {
  
  
   matrix corr_to_chol ( real x,
                         int J) {
        matrix[J, J] cor = add_diag(rep_matrix(x, J, J), 1 - x);
        return cholesky_decompose(cor); 
  }
 
 
 
 matrix lb_ub_lp (matrix y,
                  real lb, 
                  real ub,
                  int log_det_J) {
 
      int N = rows(y); 
      int M = cols(y); 
      matrix[N, M] tanh_y; 
   
      tanh_y = tanh(y);
      
      if (log_det_J == 1)  target +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh_y))) ;
      return lb +  (ub - lb) *  0.5 * (1 + tanh_y) ;   
    
  }  
  
   vector lb_ub_lp (vector y, 
                    real lb,
                    real ub,
                    int log_det_J) {
 
      int N = num_elements(y); 
      vector[N] tanh_y; 
   
      tanh_y = tanh(y);
      
      if (log_det_J == 1)  target +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh_y))) ;
      return lb +  (ub - lb) *  0.5 * (1 + tanh_y) ;   
    
  }  
 
  real lb_ub_lp (real y, 
                 real lb, 
                 real ub, 
                 int log_det_J) {
      
        real  tanh_y = tanh(y);
   
        if (log_det_J == 1) target +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh_y))) ;
        return lb +  (ub - lb) *  0.5 * (1 + tanh_y) ; 
    
  } 
  
  
  

                                
                                
////// this is the entire log_posterior fn for LC-MVP / MVP / latent_trait models w/ manual gradients and AVX-512 (or AVX-2 or if not available then using Stan's built-in fns)                                  
real   Stan_wrapper_lp_fn_latent_trait_var(    int Model_type_int, 
                                      int force_autodiff_int, 
                                      int force_PartialLog_int,
                                      int multi_attempts_int,
                                      vector theta_main_vec,  
                                      vector theta_us_vec, 
                                      matrix y, 
                                      int  n_chunks_target, 
                                      real overflow_threshold, 
                                      real underflow_threshold,
                                      real prev_prior_a, 
                                      real prev_prior_b, 
                                      matrix n_covariates_per_outcome_vec,   
                                      array[] matrix prior_coeffs_mean, 
                                      array[] matrix prior_coeffs_sd, 
                                      matrix LT_b_priors_shape, 
                                      matrix LT_b_priors_scale, 
                                      matrix LT_known_bs_indicator,
                                      matrix LT_known_bs_values); 
                                  
                                    
 
 
}

   
    

data {

      int<lower=1> N;
      int<lower=2> n_tests;
      matrix<lower=0>[N, n_tests]   y;  //////// data
      int<lower=2> n_class;
      int<lower=1> n_pops;
      array[N] int pop;
      int n_covariates_max;
      //array[n_class, n_tests] matrix[N, n_covariates_max] X; /////// covariate array (can have  DIFFERENT NUMBERS of covariates for each  outcome - fill rest of array with 999999 if they vary between outcomes)
      array[n_class, n_tests] int n_covs_per_outcome;
      real overflow_threshold;
      real underflow_threshold;
      ///// priors 
      array[n_class, n_covariates_max, n_tests]  real prior_beta_mean;
      array[n_class, n_covariates_max, n_tests]  real<lower=0> prior_beta_sd;
      vector<lower=0>[n_pops] prior_p_alpha;
      vector<lower=0>[n_pops] prior_p_beta;
      //// other
      int n_params_main;
      int n_chunks_target;
      int priors_via_Stan;
      int multi_attempts_int;
      int force_autodiff_int;
      int force_PartialLog_int;
      //// latent_trait only:
      matrix[n_class, n_tests] LT_b_priors_shape;
      matrix[n_class, n_tests] LT_b_priors_scale;
      matrix[n_class, n_tests] LT_known_bs_indicator;
      matrix[n_class, n_tests] LT_known_bs_values;
  
}



transformed data {
  
        int Model_type_int = 3;  ////  1 for MVP, 2 for LC_MVP, and 3 for latent_trait 
  
      //  int log_det_J_main_indicator = 0; //// note: we are EXCLUDING log_det_J for main (i.e. corr's and prev) because this is handled in C++
   
        vector[n_class] lkj_cholesky_eta_flat = rep_vector(1, n_class);
        array[n_class] matrix[n_covariates_max, n_tests] prior_beta_mean_flat;
        array[n_class] matrix[n_covariates_max, n_tests] prior_beta_sd_flat;
        vector<lower=0>[n_pops] prior_p_alpha_flat = rep_vector(1, n_pops);
        vector<lower=0>[n_pops] prior_p_beta_flat  = rep_vector(1, n_pops);
        
        int k_choose_2 = (n_tests * (n_tests - 1)) / 2;
        int km1_choose_2 = ((n_tests - 1) * (n_tests - 2)) / 2;
        
        int n_covariates_total_nd =    (sum( (n_covs_per_outcome[1,])));
        int n_covariates_total_d =     (sum( (n_covs_per_outcome[2,])));
        int n_covariates_total =       n_covariates_total_nd + n_covariates_total_d;
        
       /////  int n_params_main = n_class*k_choose_2 + n_covariates_total + (n_class - 1);
        int n_us = n_tests * N;
  
        array[n_class, n_tests] matrix[N, n_covariates_max] X;
  
        matrix[n_class, n_tests] n_covariates_per_outcome_vec = to_matrix(n_covs_per_outcome);
        
        array[n_class] matrix[n_covariates_max, n_tests] prior_beta_mean_as_array_of_mat;
        array[n_class] matrix[n_covariates_max, n_tests] prior_beta_sd_as_array_of_mat;
        
        array[n_class] matrix[n_tests, n_tests] prior_for_corr_a_dummy;
        array[n_class] matrix[n_tests, n_tests] prior_for_corr_b_dummy;
        
        vector[n_us] theta_nuisance_vec_as_data = rep_vector(0.01, n_us);
        vector[n_params_main] theta_main_vec_as_data = rep_vector(0.01, n_params_main);
        
        for (c in 1:n_class) {
          prior_beta_mean_flat[c,,] = rep_matrix(0.0, n_covariates_max, n_tests);
          prior_beta_sd_flat[c,,] =   rep_matrix(1.0, n_covariates_max, n_tests);
        }
        
             
        for (c in 1:n_class) { 
            
             prior_for_corr_a_dummy[c,,] = rep_matrix(0, n_tests, n_tests);
             prior_for_corr_b_dummy[c,,] = rep_matrix(0, n_tests, n_tests);
             
            for (t in 1:n_tests) {
                prior_beta_mean_as_array_of_mat[c,,t] = to_vector(prior_beta_mean[c,,t]);
                prior_beta_sd_as_array_of_mat[c,,t] =   to_vector(prior_beta_sd[c,,t]);
                
                X[c, t] = rep_matrix(1, N, 1);
            }
          
        }
        
                                                  

}


parameters {
  
       //  put nuisance parameters FIRST (NOTE: doesnt have to be on "raw" scale to work as grad is computed w.r.t unconstrained anyway!)
       vector[n_us] theta_nuisance_vec; 
       vector[n_params_main] theta_main_vec;
       
}


 


transformed parameters {

     ///// arrays / matrices / vectors of UNCONSTRAINED parameters, categorised by parameter type (e.g. coeffs, etc)
     matrix[n_class, n_tests] LT_bs_unc;
     vector[n_covariates_total] beta_vec;
     vector[n_pops*(n_class - 1)] prev_unconstrained_vec;
     ///////
     // array[n_class] matrix[n_tests, n_tests] Omega;
     // array[n_class] matrix[n_tests, n_tests] L_Omega;
     array[n_class] matrix[n_covariates_max, n_tests] beta;
     matrix[n_pops, n_class]  prev_unconstrained_matrix;
     matrix<lower=0, upper=1>[n_pops, n_class]  prev_var_matrix;
     /// vector[N] log_lik  = rep_vector(0.0, N);
     real log_posterior = 0.0;



     {
           //// set counter for main params
           int counter = 1;

               for (c in 1:n_class) {
                     for (t in 1:n_tests) {
                           LT_bs_unc[c, t] =  theta_main_vec[counter];
                           counter += 1;
                       }
                     }
                     
           ///// COEFFICIENTS are the next set of parameters
           int counter_beta = 1;
           for (c in 1:n_class) {
                 for (t in 1:n_tests) {
                    for (k in 1:n_covs_per_outcome[c, t]) {
                        beta_vec[counter_beta] = theta_main_vec[counter];
                        counter_beta += 1;
                        counter += 1;
                    }
                 }
           }

           ///// PREVELANCE is the last parameter (if model is latent class, otherwise done)
           int n_prevs = n_pops*(n_class - 1);
           for (i in 1:n_prevs) {
               prev_unconstrained_vec[i] = theta_main_vec[counter];
               counter += 1;
           }


     }



     if (n_class > 1) {
         prev_unconstrained_matrix[1, 2] = prev_unconstrained_vec[1];
         prev_unconstrained_matrix[1, 1] = 1.0 - prev_unconstrained_vec[1];
         prev_var_matrix = lb_ub_lp(prev_unconstrained_matrix, 0.0, 1.0, priors_via_Stan);
     }
      
      
              if (priors_via_Stan == 1) {

                   log_posterior =     Stan_wrapper_lp_fn_latent_trait_var(       Model_type_int, /// model_type (1 for MVP, 2 for LC_MVP, 3 for latent_trait)
                                                                                  force_autodiff_int,
                                                                                  force_PartialLog_int,
                                                                                  multi_attempts_int, 
                                                                                  theta_main_vec,
                                                                                  theta_nuisance_vec,
                                                                                  y,
                                                                                  n_chunks_target,
                                                                                  overflow_threshold,
                                                                                  underflow_threshold,
                                                                                  prior_p_alpha_flat[1],  //// FLAT prior as managing priors via Stan
                                                                                  prior_p_beta_flat[1],  //// FLAT prior as managing priors via Stan
                                                                                  n_covariates_per_outcome_vec,
                                                                                  prior_beta_mean_flat,  //// FLAT prior as managing priors via Stan
                                                                                  prior_beta_sd_flat, //// FLAT prior as managing priors via Stan
                                                                                  LT_b_priors_shape,
                                                                                  LT_b_priors_scale,
                                                                                  LT_known_bs_indicator,
                                                                                  LT_known_bs_values);
                                                                        
                  // log_lik = tail(lp_and_log_lik, N);
                  // log_posterior = head(lp_and_log_lik, 1)[1];


              } else { ///// if managing priors directly via C++
                
                   log_posterior     =  Stan_wrapper_lp_fn_latent_trait_var(        Model_type_int, /// model_type (1 for MVP, 2 for LC_MVP, 3 for latent_trait)
                                                                                    force_autodiff_int,
                                                                                    force_PartialLog_int,
                                                                                    multi_attempts_int,
                                                                                    theta_main_vec,
                                                                                    theta_nuisance_vec,
                                                                                    y,
                                                                                    n_chunks_target,
                                                                                    overflow_threshold, 
                                                                                    underflow_threshold,
                                                                                    prior_p_alpha[1],
                                                                                    prior_p_beta[1],
                                                                                    n_covariates_per_outcome_vec,  
                                                                                    prior_beta_mean_as_array_of_mat,
                                                                                    prior_beta_sd_as_array_of_mat,
                                                                                    LT_b_priors_shape,
                                                                                    LT_b_priors_scale,
                                                                                    LT_known_bs_indicator,
                                                                                    LT_known_bs_values);
                                                                        
                    // log_lik = tail(lp_and_log_lik, N);
                    // log_posterior = head(lp_and_log_lik, 1)[1];
                                
          
               }
               

               



}
 


 


model {
  

   if (priors_via_Stan == 1) {
    
           for (c in 1 : n_class) {
                for (t in 1 : n_tests) {
                     for (k in 1 : n_covs_per_outcome[c, t]) {
                       target +=  normal_lpdf(beta[c, k, t]  | prior_beta_mean_as_array_of_mat[c, k, t], prior_beta_sd_as_array_of_mat[c, k, t]);
                    }
                }
                 //// target += lkj_corr_cholesky_lpdf(L_Omega[c,,] | lkj_cholesky_eta[c]); // for MVP / LC_MVP
            }
    
    
            for (g in 1 : n_pops) {
              target += beta_lpdf(prev_var_matrix[g] | prior_p_alpha[g], prior_p_beta[g]);
            }

    }

              
              
                 target += log_posterior;


 
                

}



  



generated quantities {

    vector[n_tests] Se_bin;
    vector[n_tests] Sp_bin;
    vector[n_tests] Fp_bin;


     for (c in 1:n_class) {


        for (t in 1:n_tests) { // for binary tests

            if (n_class == 2) { // summary Se and Sp only calculated if n_class = 2 (i.e. the "standard" # of classes for DTA)
                  Se_bin[t]  =        Phi(   beta[2, 1, t]   );
                  Sp_bin[t]  =    1 - Phi(   beta[1, 1, t]   );
                  Fp_bin[t]  =    1 - Sp_bin[t];
            }
      }



  }


}




