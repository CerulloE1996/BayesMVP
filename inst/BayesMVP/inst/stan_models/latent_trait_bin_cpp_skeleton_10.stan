functions {
 
      matrix cov2cor(matrix V) {
          int p = rows(V);
          vector[p] Is = inv_sqrt(diagonal(V));
          return quad_form_diag(V, Is); 
      }
 
      vector lb_ub_lp (vector y, real lb, real ub) {
 
            int N = num_elements(y); 
            vector[N] tanh_y; 
           // tanh_y = tanh_1(y);
            tanh_y = tanh(y);
              target +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh_y))) ;
            return lb +  (ub - lb) *  0.5 * (1 + tanh_y) ;   
    
      }  
 
      real lb_ub_lp (real y, real lb, real ub) {
          
             real  tanh_y = tanh(y);
          //   real tanh_y = tanh_1(y);  
             target +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh_y))) ;
            return lb +  (ub - lb) *  0.5 * (1 + tanh_y) ; 
    
      } 
      
      //// Bounds y between lb and ub (WITHOUT jacobian adjustment!):
      real lb_ub ( real y, 
                   real lb, 
                   real ub) {
          
            real neg_log_2 = -0.6931471805599453; // 64-bit (i.e. "double") precision
            real tanh_y = tanh(y);
            
            return lb +  (ub - lb) *  0.5 * (1.0 + tanh_y);
    
      }
      vector lb_ub (  vector y, 
                      real lb, 
                      real ub) {
 
            real neg_log_2 = -0.6931471805599453; // 64-bit (i.e. "double") precision
            int N = num_elements(y); 
            vector[N] tanh_y = tanh(y);
            
            return lb +  (ub - lb) *  0.5 * (1.0 + tanh_y);
    
      }
      vector lb_ub ( vector y, 
                     vector lb, 
                     vector ub) {
 
            real neg_log_2 = -0.6931471805599453; // 64-bit (i.e. "double") precision
            int N = num_elements(y); 
            vector[N] tanh_y = tanh(y);
            
            return lb + (ub - lb) .* (0.5 * (1.0 + tanh_y));

      }
 
      // need to add citation to this (slight modification from a HP. calculators forum post)
      real inv_Phi_approx_from_prob(real p) { 
            return 5.494 *  sinh(0.33333333333333331483 * asinh( 0.3418 * logit(p)  )) ;
      }
      
      // need to add citation to this (slight modification from a HP. calculators forum post)
      vector inv_Phi_approx_from_prob(vector p) { 
            return 5.494 *  sinh(0.33333333333333331483 * asinh( 0.3418 * logit(p)  )) ;  
      }
      
      // need to add citation to this  (slight modification from a HP. calculators forum post)
      real inv_Phi_approx_from_logit_prob(real logit_p) { 
            return 5.494 *  sinh(0.33333333333333331483 * asinh( 0.3418 * logit_p  )) ; 
      }
      
      // need to add citation to this (slight modification from a HP. calculators forum post)
      vector inv_Phi_approx_from_logit_prob(vector logit_p) { 
            return 5.494 *  sinh(0.33333333333333331483 * asinh( 0.3418 *logit_p  )) ; 
      }
 
      vector rowwise_sum(matrix M) {      // M is (N x T) matrix
            return M * rep_vector(1.0, cols(M));
      }
      
      vector rowwise_max(matrix M) {      // M is (N x T) matrix
            int N =  rows(M);
            vector[N] rowwise_maxes;
            for (n in 1:N) {
              rowwise_maxes[n] = max(M[n, ]);
            }
            return rowwise_maxes;
      }
      
      vector log_sum_exp_2d(matrix array_2d_to_lse) { 
            int N = rows(array_2d_to_lse);
            matrix[N, 2] rowwise_maxes_2d_array;
            rowwise_maxes_2d_array[, 1] =  rowwise_max(array_2d_to_lse);
            rowwise_maxes_2d_array[, 2] =  rowwise_maxes_2d_array[, 1];
            return  rowwise_maxes_2d_array[, 1] + log(rowwise_sum(exp((array_2d_to_lse  -  rowwise_maxes_2d_array))));
      }
 
}


data {
      int<lower=1> N;
      int<lower=2> n_tests;
      matrix<lower=0>[N, n_tests]   y;  //////// data
      int<lower=2> n_class;
      int<lower=1> n_pops;
      array[N] int pop;
      ////
      // int n_covariates_max_nd;
      // int n_covariates_max_d;
      // int n_covariates_max;
      // array[n_tests] matrix[N, n_covariates_max_nd] X_nd; /////// covariate array (can have  DIFFERENT NUMBERS of covariates for each  outcome - fill rest of array with 999999 if they vary between outcomes)
      // array[n_tests] matrix[N, n_covariates_max_d]  X_d; /////// covariate array (can have  DIFFERENT NUMBERS of covariates for each  outcome - fill rest of array with 999999 if they vary between outcomes)
      // array[n_class, n_tests] int n_covs_per_outcome;
      ////
      ////
      real overflow_threshold;
      real underflow_threshold;
      /////
      int prior_only;
      ////
      array[n_class] matrix[1, n_tests] prior_beta_mean;  ////  // array[n_class, n_tests, n_covariates_max]  real prior_beta_mean;
      array[n_class] matrix<lower=0>[1, n_tests] prior_beta_sd;     //// array[n_class, n_tests, n_covariates_max]  real<lower=0> prior_beta_sd;
      ////
      matrix<lower=0>[n_pops, 1] prev_prior_a; // NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns ; ## NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns 
      matrix<lower=0>[n_pops, 1] prev_prior_b; // NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns 
      ///// other
      int Phi_type;
      int handle_numerical_issues;
      int fully_vectorised;
      ////
      matrix<lower=0>[n_class, n_tests]  LT_b_priors_1;
      matrix<lower=0>[n_class, n_tests]  LT_b_priors_2;
      matrix<lower=0>[n_class, n_tests]  LT_known_bs_values;
      matrix<lower=0>[n_class, n_tests]  LT_known_bs_indicator;
}


parameters {
      matrix[N, n_tests] u_raw; ////  put nuisance parameters FIRST (NOTE: doesnt have to be on "raw" scale to work as grad is computed w.r.t unconstrained anyway!)
      vector[n_class * n_tests] LT_b_raw_vec; //// Put the b's before the a's for this model!
      vector[n_class * n_tests] LT_a_vec; //// Put the b's before the a's for this model!
      vector[n_pops] p_raw;
}


transformed parameters { 
     vector<lower=0, upper=1>[n_pops] prev = lb_ub(p_raw, 0.0, 1.0);
     matrix[n_class, n_tests] LT_a; /// use LT_a in the MVP model. 
     matrix[n_class, n_tests] LT_b_raw;
     matrix[n_class, n_tests] LT_b;
     array[n_class] matrix[1, n_tests] beta;
     array[n_class] matrix[n_tests, n_tests] Sigma;
     array[n_class] matrix[n_tests, n_tests] Omega;
     
     {
          int counter = 1;
          for (c in 1:n_class) {
              for (t in 1:n_tests) {
                 LT_a[c, t] = LT_a_vec[counter];
                 counter += 1;
              }
          }
     }
      
     {
          int counter = 1;
          for (c in 1:n_class) {
             for (t in 1:n_tests) {
               if (LT_known_bs_indicator[c, t] == 0) {
                   LT_b_raw[c, t] = LT_b_raw_vec[counter];
                   // LT_b[c, t] = exp(LT_b_raw[c, t]); 
                   LT_b[c, t] = lb_ub(LT_b_raw[c, t], 0.0, 5.0); 
               } else { 
                   LT_b[c, t] = LT_known_bs_values[c, t];
               }
               beta[c][1, t] = LT_a[c, t] / sqrt(1.0 + square(LT_b[c, t])); // we are putting a prior on the TRANSFORMED parameter (beta) - hence need Jacobian adjustment
               counter += 1;
             }
          }
     }
                   
     //// get covariance and correlation matrix. Note LT model is MVP w/ covariance mtx equal to I + b*b'. 
     for (c in 1 : n_class) {  
              Sigma[c] = diag_matrix(rep_vector(1.0, n_tests)) + to_vector(LT_b[c, ]) * to_row_vector(LT_b[c, ]);
              Omega[c] = cov2cor(Sigma[c]);
     }
}


model {
        // //// set prior on TRANSFORMED parameter - beta - equiv to beta in LC-MVP model!!!
        // for (c in 1:n_class) {
        //     for (t in 1:n_tests) {
        //            beta[c][1, t] ~ normal(prior_beta_mean[c][1, t], prior_beta_sd[c][1, t]);      //// prior directly on the mean params
        //            target += - 0.5 * log(1.0 + square(LT_b[c, t]));      //// Jacobian adjustment for beta -> LT_a 
        //            LT_b[c, t] ~ weibull(LT_b_priors_1[c, t], LT_b_priors_2[c, t]);  //// prior for corr
        //            // target += LT_b_raw_vec[c, t]; 
        //     }
        // }
        // 
        // for (c in 1:n_class) {
        //     for (t in 1:n_tests) {
        //         if (LT_known_bs_indicator[c, t] == 0) { 
        //             target += LT_b_raw[c, t]; //// Jacobian adjustment for corr / b's
        //         }
        //     } 
        // }
}


generated quantities {
     vector[n_tests] Se_bin;
     vector[n_tests] Sp_bin;
     vector[n_tests] Fp_bin;
     vector<lower=0, upper=1>[n_pops] p = prev;
     ////
     // vector[N] log_lik = rep_vector(0.0, N);

     for (c in 1:n_class) {
            for (t in 1:n_tests) { // for binary tests
                 if (n_class == 2) { // summary Se and Sp only calculated if n_class = 2 (i.e. the "standard" # of classes for DTA)
                      Se_bin[t]  =        Phi(   beta[2][1, t]   );
                      Sp_bin[t]  =    1 - Phi(   beta[1][1, t]   );
                      Fp_bin[t]  =    1 - Sp_bin[t];
                }
                else {
                  Se_bin[t] = 999;
                  Sp_bin[t] = 999;
                  Fp_bin[t] = 999;
                }
          }
     }
     // ////
     // //// ---- likelihood:
     // ////
     // for (n in 1:N) {
     //   
     //          vector[n_class] log_prev;
     //          vector[n_class] lp;
     //        
     //          log_prev[1] =  log1m(p[pop[n]]);//(bernoulli_lpmf( 0 | prev[pop[n]]) );
     //          log_prev[2] =  log(p[pop[n]]);// (bernoulli_lpmf( 1 | prev[pop[n]]) );
     // 
     //          for (c in 1:2) { // 2 classes
     // 
     //              vector[n_tests] Z_std_norm;
     //              vector[n_tests] y1;
     //              real inc  = 0.0;
     // 
     //              for (t in 1:n_tests) {
     // 
     //                      real Bound_Z =  -(LT_a[c, t] + inc) / L_Sigma[c][t, t];
     //                      real Bound_U_Phi_Bound_Z = Phi(Bound_Z);
     //                      
     //                      if (y[n, t] == 1) {
     //                        real Phi_Z = Bound_U_Phi_Bound_Z + (1.0 - Bound_U_Phi_Bound_Z) * u[n, t];
     //                        Z_std_norm[t] = inv_Phi(Phi_Z);
     //                        y1[t] = log1m(Bound_U_Phi_Bound_Z);
     //                      } else {
     //                        real Phi_Z = Bound_U_Phi_Bound_Z * u[n, t];
     //                        Z_std_norm[t] = inv_Phi(Phi_Z);
     //                        y1[t] = log(Bound_U_Phi_Bound_Z);
     //                      }
     // 
     //                      if (t < n_tests) inc = L_Sigma[c][t + 1, 1:t] * head(Z_std_norm, t);
     // 
     //              } // end of t loop
     // 
     //              lp[c] =  sum(y1) + log_prev[c];
     // 
     //       } // end of c loop
     // 
     //       log_lik[n] =  log_sum_exp(lp);
     // }

}






