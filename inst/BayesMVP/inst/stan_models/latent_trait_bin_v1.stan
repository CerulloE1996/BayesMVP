  

functions {
      matrix cov2cor(matrix V) {
        
            int p = rows(V);
            vector[p] Is = inv_sqrt(diagonal(V));
            return quad_form_diag(V, Is); 
          
      }
      
      matrix corr_to_chol( real x,
                           int J) {
                             
            matrix[J, J] cor = add_diag(rep_matrix(x, J, J), 1.0 - x);
            return cholesky_decompose(cor);
          
      }
      
      vector lb_ub_lp(  vector y, 
                        real lb,
                        real ub) {
         
            int N = num_elements(y); 
            vector[N] tanh_y; 
            tanh_y = tanh(y);
            target +=  - log(2.0)  +  log( (ub - lb) * (1.0 - square(tanh_y)));
            return lb +  (ub - lb) *  0.5 * (1.0 + tanh_y);
      
      }
      
      real lb_ub_lp( real y, 
                     real lb, 
                     real ub) {
        
            real  tanh_y = tanh(y);
            target +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh_y)));
            return lb +  (ub - lb) *  0.5 * (1 + tanh_y);
          
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
      
      vector rowwise_sum_lp(matrix M) {      // M is (N x T) matrix
      
            vector[rows(M)] col_vec = M * rep_vector(1.0, cols(M));
            target += sum(col_vec);
            return col_vec;
           
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
      real overflow_threshold;
      real underflow_threshold;
      ///// priors 
      int prior_only;
      array[n_class]  matrix[1, n_tests] prior_beta_mean;
      array[n_class]  matrix<lower=0>[1, n_tests]   prior_beta_sd;
      array[n_pops] real<lower=0> prior_p_alpha;
      array[n_pops] real<lower=0> prior_p_beta;
      matrix<lower=0>[n_class, n_tests]  LT_b_priors_shape;
      matrix<lower=0>[n_class, n_tests]  LT_b_priors_scale;
      matrix<lower=0>[n_class, n_tests]  LT_known_bs_values;
      matrix<lower=0>[n_class, n_tests]  LT_known_bs_indicator;
}


parameters {
      matrix<lower=0.0, upper=1.0>[N, n_tests] u; //  put nuisance parameters FIRST (NOTE: doesnt have to be on "raw" scale to work as grad is computed w.r.t unconstrained anyway!)
      vector[n_pops]  p_raw;
      matrix[n_class, n_tests] LT_a; /// use LT_a in the MVP model. 
      matrix<lower=0.0>[n_class, n_tests] LT_b;
}

 
 
transformed parameters {
      // matrix<lower=0, upper=1>[N, n_tests]  u = Phi(u_raw);
      array[n_class, n_tests, 1] real beta;/// "equiv."" to beta in MVP model --- we are putting a prior on this (so need Jacobian adjustment!!)
      array[n_class] matrix[n_tests, n_tests] Omega; // "equiv."" to Omega in LC_MVP model
      vector<lower=0, upper=1>[n_pops] p = lb_ub_lp(p_raw, 0.0, 1.0);
      array[n_class] matrix[n_tests, n_tests] Sigma;
      array[n_class] matrix[n_tests, n_tests] L_Sigma; //// use L_Sigma in the MVP model. 
      ////
      matrix[N, n_class]  log_prev;
      vector[N] log_lik  = rep_vector(0.0, N);

            for (n in 1:N) {
                log_prev[n, 1] =  log1m(p[pop[n]]);//(bernoulli_lpmf( 0 | prev[pop[n]]) );
                log_prev[n, 2] =  log(p[pop[n]]);// (bernoulli_lpmf( 1 | prev[pop[n]]) );
            }
      
            {
                for (c in 1:n_class) {
                  for (t in 1:n_tests) {
                    beta[c, t, 1] = LT_a[c, t] / sqrt(1.0 + square(LT_b[c, t])); // we are putting a prior on the TRANSFORMED parameter (beta) - hence need Jacobian adjustment
                  }
                }
            }
                   
            //// get covariance and correlation matrix. Note LT model is MVP w/ covariance mtx equal to I + b*b'. 
            for (c in 1:n_class) {  
                    Sigma[c]   = diag_matrix(rep_vector(1.0, n_tests)) + to_vector(LT_b[c, ]) *  to_row_vector(LT_b[c, ]);
                    L_Sigma[c] = cholesky_decompose(Sigma[c]);
                    Omega[c]   = cov2cor(Sigma[c]);
            }
            
            {
               // likelihood (2 classes)
               for (n in 1:N) {
                        vector[n_class] lp;
    
                        for (c in 1:2) { // 2 classes
    
                            vector[n_tests] Z_std_norm;
                            vector[n_tests] y1;
                            real inc  = 0.0;
  
                            for (t in 1:n_tests) {

                                    real Bound_Z =  -(LT_a[c, t] + inc) / L_Sigma[c][t, t];
                                    real Bound_U_Phi_Bound_Z = Phi(Bound_Z);
                                    
                                    if (y[n, t] == 1) {
                                      real Phi_Z = Bound_U_Phi_Bound_Z + (1.0 - Bound_U_Phi_Bound_Z) * u[n, t];
                                      Z_std_norm[t] = inv_Phi(Phi_Z);
                                      y1[t] = log1m(Bound_U_Phi_Bound_Z);
                                    } else {
                                      real Phi_Z = Bound_U_Phi_Bound_Z * u[n, t];
                                      Z_std_norm[t] = inv_Phi(Phi_Z);
                                      y1[t] = log(Bound_U_Phi_Bound_Z);
                                    }

                                    if (t < n_tests) inc = L_Sigma[c][t + 1, 1:t] * head(Z_std_norm, t);
  
                            } // end of t loop
  
                            lp[c] =  sum(y1) + log_prev[n, c];
    
                     } // end of c loop
    
                     log_lik[n] =  log_sum_exp(lp);
               }

            }
}


model {
         //// set prior on TRANSFORMED parameter - beta - equiv to beta in LC-MVP model!!!
         for (c in 1:n_class) {
            for (t in 1:n_tests) {
                 //// prior directly on the mean params
                 beta[c, t, 1] ~ normal(prior_beta_mean[c, 1, t], prior_beta_sd[c, 1, t]);
                 target += - 0.5 * log(1.0 + square(LT_b[c, t])); //// Jacobian adjustment for beta -> LT_a 
                 //// prior for corr
                 LT_b[c, t] ~ gamma(1.0, 1.0);
                 // LT_b[c, t] ~ weibull(LT_b_priors_shape[c, t], LT_b_priors_scale[c, t]);    // target += LT_b_raw_vec[c, t]; 
            }
         }
         
         for (g in 1:n_pops) {
            p[g] ~ beta(prior_p_alpha[g], prior_p_beta[g]);
         }
         
         target += sum(log_lik);
}


generated quantities {
      vector[n_tests] Se_bin;
      vector[n_tests] Sp_bin;
      vector[n_tests] Fp_bin;

      for (t in 1:n_tests) { // for binary tests
      
              if (n_class == 2) { // summary Se and Sp only calculated if n_class = 2 (i.e. the "standard" # of classes for DTA)
                  Se_bin[t]  =        Phi(   beta[2, t, 1]   );
                  Sp_bin[t]  =    1 - Phi(   beta[1, t, 1]   );
                  Fp_bin[t]  =    1 - Sp_bin[t];
              } else {
                  Se_bin[t] = 999;
                  Sp_bin[t] = 999;
                  Fp_bin[t] = 999;
              }
              
      }
}


