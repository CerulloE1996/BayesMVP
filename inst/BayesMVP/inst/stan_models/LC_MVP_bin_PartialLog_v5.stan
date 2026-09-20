functions {
 
   // matrix corr_to_chol(real x, int J) {
     //    matrix[J, J] cor = add_diag(rep_matrix(x, J, J), 1 - x);
     //    return cholesky_decompose(cor); 
   // }
 
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
  
      matrix cholesky_corr_constrain_outer_lp( // matrix Omega_raw,
                                           vector col_one_raw, 
                                           vector off_raw,
                                           real lb, 
                                           real ub) {
     
            int K = num_elements(col_one_raw) + 1;
            vector[K - 1] z = lb_ub_lp(col_one_raw, lb, ub);
            matrix[K, K] L = diag_matrix(rep_vector(1, K));
            vector[K] D;
            D[1] = 1;
            L[2:K, 1] = z[1:K - 1];
            D[2] = 1 - L[2, 1]^2;
            int cnt = 1;
           
            for (i in 3:K) {
               D[i] = 1 - L[i, 1]^2; 
               L[i, 2:i - 1] = rep_row_vector(1 - L[i, 1]^2, i - 2);
               real l_ij_old = L[i, 2];
              for (j in 2:i - 1) {
              //  real l_ij_old = L[i, j];
                real b1 = dot_product(L[j, 1:(j - 1)], D[1:j - 1]' .* L[i, 1:(j - 1)]);
                  
                  // how to derive the bounds
                  // we know that the correlation value C is bound by
                  // b1 - Ljj * Lij_old <= C <= b1 + Ljj * Lij_old
                  // Now we want our bounds to be enforced too so
                  // max(lb, b1 - Ljj * Lij_old) <= C <= min(ub, b1 + Ljj * Lij_old)
                  // We have the Lij_new = (C - b1) / Ljj
                  // To get the bounds on Lij_new is
                  // (bound - b1) / Ljj 
                  
                  real low = max({-sqrt(l_ij_old * D[j]), lb - b1});
                  real up = min({sqrt(l_ij_old * D[j]), ub - b1}); 
                  if (is_nan(low) || is_nan(up) || is_inf(low) || is_inf(up) || low >= up)
                      reject("empty or nonfinite correlation interval");
                  
                  real x = lb_ub_lp(off_raw[cnt], low, up);
                  L[i, j] = x / D[j]; 
        
                  target += -0.5 * log(D[j]);
                  
                   l_ij_old -= D[j] * square(L[i, j]);
                   if (is_nan(l_ij_old) || is_inf(l_ij_old) || l_ij_old <= 0)
                       reject("nonpositive or nonfinite LDL remainder");
                  
                 // real mul = 1 - (D[j] * L[i, j]^2) / l_ij_old;
                 // L[i, (j + 1):i - 1] *= mul;
                  //D[i] *= mul;
                  cnt += 1;
                }
                D[i] = l_ij_old;
              }
            
              if (is_nan(sum(D)) || is_inf(sum(D)) || min(D) <= 0)
                  reject("nonpositive or nonfinite LDL pivot");
              return diag_post_multiply(L, sqrt(D));
            
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
      matrix<lower=0>[N, n_tests] y;  //////// data
      int<lower=2> n_class;
      int<lower=1> n_pops;
      array[N] int pop;
      ////
      int n_covariates_max_nd;
      int n_covariates_max_d;
      int n_covariates_max;
      array[n_tests] matrix[N, n_covariates_max_nd] X_nd; /////// covariate array (can have  DIFFERENT NUMBERS of covariates for each  outcome - fill rest of array with 999999 if they vary between outcomes)
      array[n_tests] matrix[N, n_covariates_max_d]  X_d; /////// covariate array (can have  DIFFERENT NUMBERS of covariates for each  outcome - fill rest of array with 999999 if they vary between outcomes)
      array[n_class, n_tests] int n_covs_per_outcome;
      ////
      int corr_force_positive;
      int<lower=0, upper=(n_tests * (n_tests - 1)) %/% 2> known_num;
      // array[n_class] matrix[n_tests, n_tests] lb_corr;
      // array[n_class] matrix[n_tests, n_tests] ub_corr;
      real overflow_threshold;
      real underflow_threshold;
      ///// priors 
      int prior_only;
      array[n_class] matrix[n_covariates_max, n_tests] prior_beta_mean;  ////  // array[n_class, n_tests, n_covariates_max]  real prior_beta_mean;
      array[n_class] matrix<lower=0>[n_covariates_max, n_tests] prior_beta_sd;     //// array[n_class, n_tests, n_covariates_max]  real<lower=0> prior_beta_sd;
      matrix<lower=0>[n_class, 1] prior_LKJ; // NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns 
      matrix<lower=0>[n_pops, 1] prior_p_alpha; // NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns ; ## NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns 
      matrix<lower=0>[n_pops, 1] prior_p_beta; // NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns 
      ///// other
      int Phi_type;
      int handle_numerical_issues;
      int fully_vectorised;
}


transformed data {
       int k_choose_2 = (n_tests * (n_tests - 1)) / 2;
       int km1_choose_2 = ((n_tests - 1) * (n_tests - 2)) / 2;
    
       int n_covariates_total_nd =    (sum( (n_covs_per_outcome[1,])));
       int n_covariates_total_d =     (sum( (n_covs_per_outcome[2,])));
       int n_covariates_total =       n_covariates_total_nd + n_covariates_total_d;
    
       real s = 1 / 1.702;
       real a = 0.07056;
       real b = 1.5976;
       real a_times_3 = 3.0 * 0.07056;
       real<lower=-1, upper=1> lb;
       real<lower=lb, upper=1> ub = 1.0;

       if (corr_force_positive == 1)  lb = 0;
       else lb = -1.0;

}


parameters {
       matrix[N, n_tests] u_raw; //  put nuisance parameters FIRST (NOTE: doesnt have to be on "raw" scale to work as grad is computed w.r.t unconstrained anyway!)
       array[n_class] vector[n_tests - 1] col_one_raw;
       array[n_class] vector[km1_choose_2 - known_num] off_raw;
       vector[n_covariates_total] beta_vec;
       vector[n_pops]  p_raw;
       
}


transformed parameters {
       array[n_class, n_tests, n_covariates_max] real beta;
       vector<lower=0, upper=1>[n_pops]   prev = lb_ub_lp(p_raw, 0.0, 1.0);
       array[n_class] matrix[n_tests, n_tests] Omega;
       array[n_class] matrix[n_tests, n_tests] L_Omega;
       matrix[n_class, n_tests] L_Omega_diag_recip;
       vector[N] log_lik  = rep_vector(0.0, N);
     
       {
            int counter = 1;
            for (c in 1 : n_class) {
                      for (t in 1:n_tests) {
                        for (k in 1:n_covs_per_outcome[c, t]) {
                           beta[c, t, k] = beta_vec[counter];
                           counter += 1;
                        }
                      }
                    L_Omega[c, :  ] =   cholesky_corr_constrain_outer_lp( to_vector(col_one_raw[c, :]), to_vector(off_raw[c, :]), lb, ub);
                    Omega[c, :  ] = multiply_lower_tri_self_transpose(L_Omega[c, :]);
                    L_Omega_diag_recip[c, ] = to_row_vector(1.0 ./ diagonal(L_Omega[c, :  ]));
            }
       }

   if (prior_only == 0) {

          matrix[N, n_class]  log_prev;

           for (n in 1:N) {
             log_prev[n, 1] =  log1m(prev[pop[n]]);//(bernoulli_lpmf( 0 | prev[pop[n]]) );
             log_prev[n, 2] =  log(prev[pop[n]]);// (bernoulli_lpmf( 1 | prev[pop[n]]) );
           }

    {  // Goodrich-based method but with my modifications / vectorisations, etc

       if (fully_vectorised == 1) {

                {  // log-lik block
                       matrix[N, n_tests] Z_std_norm;
                       vector[N] Bound_Z;
                       matrix[N, n_tests] u;
                       matrix[N, n_class] lp;
                       // matrix[N, n_class] inc =    rep_matrix(0.0, N, n_class);
                       vector[N] inc =    rep_vector(0.0, N);
                       matrix[N, n_tests] y1;

                    for (t in 1:n_tests) {
                        u[1:N, t] = lb_ub_lp(u_raw[1:N, t], 0.0, 1.0);
                    }

            // Parameters for likelihood function. Based on code upladed by Ben Goodrich which uses the
            // GHK algorithm for generating TruncMVN. See: https://github.com/stan-dev/example-models/blob/master/misc/multivariate-probit/probit-multi-good.stan#L11
            // Note that this version below is (mostly) vectorised thoough, so it should be much more efficient on most datasets/machines.
            if (handle_numerical_issues == 0) {

                    for (c in 1:n_class) {

                            inc = rep_vector(0.0, N);

                            for (t in 1:n_tests) {

                                             if (n_covariates_max > 1) {
                                                  vector[N] Xbeta; // array[n_tests] matrix[N, n_covariates_max_nd] X_nd; 
                                                  if (c == 1)   Xbeta =  X_nd[t][1:N, 1:n_covs_per_outcome[c, t]] *  to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);
                                                  if (c == 2)   Xbeta =  X_d[t][1:N, 1:n_covs_per_outcome[c, t]]  *  to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);
                                                  Bound_Z  = - (Xbeta + inc )  *  L_Omega_diag_recip[c, t] ; // use as marker for potential overflow
                                             } else {
                                                  Bound_Z  = - (beta[c,t,1] + inc )  *  L_Omega_diag_recip[c, t] ; // use as marker for potential overflow
                                             }

                                    if (Phi_type == 2) {
                                      vector[N] Bound_U_Phi_Bound_Z = Phi_approx(Bound_Z);
                                      vector[N] Phi_Z    =     (y[,t] .*  Bound_U_Phi_Bound_Z  +  (y[,t] -   Bound_U_Phi_Bound_Z ) .*   (y[,t] + (y[,t] - 1.0)) .* u[, t]  )  ;
                                      Z_std_norm[1:N, t]  =    inv_Phi_approx_from_prob(Phi_Z);
                                      y1[, t] =        log(y[, t] .* (1.0 -  Bound_U_Phi_Bound_Z) + (y[, t] - 1.0) .* Bound_U_Phi_Bound_Z .* (y[, t] + (y[, t] - 1.0)));
                                    } else {
                                      vector[N] Bound_U_Phi_Bound_Z = Phi(Bound_Z);
                                      vector[N] Phi_Z    =     (y[,t] .*  Bound_U_Phi_Bound_Z  +  (y[,t] -   Bound_U_Phi_Bound_Z ) .*   (y[,t] + (y[,t] - 1.0)) .* u[, t]  )  ;
                                      Z_std_norm[1:N, t]  =    inv_Phi(Phi_Z);
                                      y1[, t] =        log(y[, t] .* (1.0 -  Bound_U_Phi_Bound_Z) + (y[, t] - 1.0) .* Bound_U_Phi_Bound_Z .* (y[, t] + (y[, t] - 1.0)));
                                    }

                                if (t < n_tests) inc = block(Z_std_norm, 1, 1, N, t) * to_vector(head(L_Omega[c, t + 1, ], t))   ;

                            }  // end of t loop

                               lp[1:N, c] = to_vector(rowwise_sum(y1[1:N, 1:n_tests]))  +  to_vector(log_prev[1:N, c])  ;

                    } // end of c loop

                                  log_lik = log_sum_exp_2d(lp);  //   for (n in 1:N)    log_lik[n] = log_sum_exp(lp[, n]);
                                  
            } else {

                 for (c in 1 : n_class) {
                 
                       inc = rep_vector(0.0, N);
    
                       for (t in 1:n_tests) {
    
                                         if (n_covariates_max > 1) {
                                              vector[N] Xbeta;
                                              if (c == 1)   Xbeta =  X_nd[t][1:N, 1:n_covs_per_outcome[c, t]] *   to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);
                                              if (c == 2)   Xbeta =  X_d[t][1:N, 1:n_covs_per_outcome[c, t]]  *   to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);
                                              Bound_Z  = - (Xbeta + inc  )  *  L_Omega_diag_recip[c, t] ; // use as marker for potential overflow
                                         } else { 
                                              Bound_Z  = - (beta[c, t, 1] + inc  )  *  L_Omega_diag_recip[c, t] ; // use as marker for potential overflow
                                         }
                           {
                                           
                                int num_OK_index = 0 ;
                                int num_Bound_Z_overflows_and_y_eq_1 = 0 ;
                                int num_Bound_Z_underflows_and_y_eq_0 = 0 ; 
    
                                for (n in 1:N) {
                                       if       ( (Bound_Z[n]  >  overflow_threshold)    &&  (y[n, t] == 1) )      num_Bound_Z_overflows_and_y_eq_1  += 1;
                                       else if  ( (Bound_Z[n]  <  underflow_threshold)   &&  (y[n, t] == 0) )      num_Bound_Z_underflows_and_y_eq_0 += 1; 
                                       else   num_OK_index += 1;
                                }
    
                                if (num_OK_index == N)  { // carry on as normal as no * problematic * overflows/underflows 
    
                                       if (Phi_type == 2) {
                                             vector[N] Bound_U_Phi_Bound_Z = Phi_approx(Bound_Z);
                                             vector[N] Phi_Z   =                  (y[, t] .*  Bound_U_Phi_Bound_Z  +  (y[,t] -   Bound_U_Phi_Bound_Z) .* (y[, t] + (y[, t] - 1.0) ) .*  u[, t])  ;
                                             Z_std_norm[, t]  =    inv_Phi_approx_from_prob(Phi_Z);
                                             y1[, t] =         log(  y[, t] .* (1.0 -  Bound_U_Phi_Bound_Z) + (y[, t] - 1.0) .*  Bound_U_Phi_Bound_Z .* ((y[, t]) + ((y[, t]) - 1.0))  );
                                       } else {
                                             vector[N] Bound_U_Phi_Bound_Z = Phi(Bound_Z);
                                             vector[N] Phi_Z   =                  (y[, t] .*  Bound_U_Phi_Bound_Z  +  (y[,t] -   Bound_U_Phi_Bound_Z) .* (y[, t] + (y[, t] - 1.0) ) .*  u[, t])  ;
                                             Z_std_norm[, t]  =    inv_Phi(Phi_Z);
                                             y1[, t] =         log(  y[, t] .* (1.0 -  Bound_U_Phi_Bound_Z) + (y[, t] - 1.0) .*  Bound_U_Phi_Bound_Z .* ((y[, t]) + ((y[, t]) - 1.0))  );
                                       } 
    
                                } else if (num_OK_index < N)  { 
    
                                              int indicator_OK_empty = 0;
                                              if (num_OK_index < 1)  {
                                                num_OK_index = 1;
                                                indicator_OK_empty = 1; 
                                              }
                                              ////
                                              int indicator_overflows_and_y_eq_1_empty = 0;
                                              if ( num_Bound_Z_overflows_and_y_eq_1  < 1)  {
                                                num_Bound_Z_overflows_and_y_eq_1  = 1;
                                                indicator_overflows_and_y_eq_1_empty = 1; 
                                              }
                                              ////
                                              int indicator_underflows_and_y_eq_0_empty = 0;
                                              if (num_Bound_Z_underflows_and_y_eq_0 < 1)  { 
                                                num_Bound_Z_underflows_and_y_eq_0  = 1;
                                                indicator_underflows_and_y_eq_0_empty = 1;
                                              }
                                              ////
                                              array[num_OK_index] int OK_index;
                                              array[num_Bound_Z_overflows_and_y_eq_1] int overflows_and_y_eq_1_index;
                                              array[num_Bound_Z_underflows_and_y_eq_0] int underflows_and_y_eq_0_index; 
                                              int counter_1  = 1;
                                              int counter_2  = 1;
                                              int counter_3  = 1;
                                              ////
                                              for (n in 1:N) {
                                                     if  (    (Bound_Z[n]  >  overflow_threshold)    &&  (y[n, t] == 1) ) {
                                                         overflows_and_y_eq_1_index[counter_1] = n;
                                                         counter_1 += 1;
                                                     } else if  ( (Bound_Z[n]  <  underflow_threshold)   &&  (y[n, t] == 0) )  {
                                                         underflows_and_y_eq_0_index[counter_2] = n; 
                                                         counter_2 += 1;
                                                     } else {
                                                         OK_index[counter_3] = n;
                                                         counter_3 += 1;
                                                    }
                                              }
                                              ////
                                              if (indicator_OK_empty == 0) { 
    
                                                       array[num_OK_index] int index = OK_index; 
                                                       int local_size = num_OK_index;
    
                                                      if (Phi_type == 2) {
                                                         vector[local_size] Bound_U_Phi_Bound_Z = Phi_approx(Bound_Z[index]); 
                                                         vector[local_size] Phi_Z   =                  (y[index,t] .*  Bound_U_Phi_Bound_Z  +  (y[index,t] -   Bound_U_Phi_Bound_Z ) .* (y[index,t] + (y[index,t] - 1.0) ) .*  u[index,  t] )  ;
                                                         Z_std_norm[index, t]  =    inv_Phi_approx_from_prob(Phi_Z);
                                                         y1[index, t] =         log((y[index, t]) .* (1.0 -  Bound_U_Phi_Bound_Z) + ((y[index, t]) - 1.0) .* Bound_U_Phi_Bound_Z .*    ((y[index, t]) + ((y[index, t]) - 1.0)));
                                                      } else {
                                                         vector[local_size] Bound_U_Phi_Bound_Z = Phi(Bound_Z[index]);
                                                         vector[local_size] Phi_Z   =                  (y[index,t] .*  Bound_U_Phi_Bound_Z  +  (y[index,t] -   Bound_U_Phi_Bound_Z ) .* (y[index,t] + (y[index,t] - 1.0) ) .*  u[index, t] )  ;
                                                         Z_std_norm[index, t]  =    inv_Phi(Phi_Z);
                                                         y1[index, t] =         log((y[index, t]) .* (1.0 -  Bound_U_Phi_Bound_Z) + ((y[index, t]) - 1.0) .* Bound_U_Phi_Bound_Z .*    ((y[index, t]) + ((y[index, t]) - 1.0)));
                                                      }
    
                                               }
                                               if (indicator_underflows_and_y_eq_0_empty ==  0) { /// underflow + y == 0 
    
                                                          array[num_Bound_Z_underflows_and_y_eq_0] int index = underflows_and_y_eq_0_index; 
                                                          int local_size = num_Bound_Z_underflows_and_y_eq_0;
    
                                                       // if (Phi_type == 2) {
                                                          vector[local_size] log_Bound_U_Phi_Bound_Z =  log_inv_logit( 0.07056 * square(Bound_Z[index]) .* Bound_Z[index]  + 1.5976 * Bound_Z[index] );
                                                          ///  vector[local_size] Bound_U_Phi_Bound_Z = exp(log_Bound_U_Phi_Bound_Z);
                                                          vector[local_size] log_Phi_Z = log(u[index, t]) +  log_Bound_U_Phi_Bound_Z ;
                                                          vector[local_size] log_1m_Phi_Z =   log1m_exp(log(u[index, t])  + log_Bound_U_Phi_Bound_Z);   /// log1m(u[index, t] .* Bound_U_Phi_Bound_Z);
                                                          vector[local_size] logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
                                                          Z_std_norm[index, t] = inv_Phi_approx_from_logit_prob(logit_Phi_Z); //  fn_colvec(logit_Phi_Z, "inv_Phi_approx_from_logit_prob");
                                                          y1[index, t]  =  log_Bound_U_Phi_Bound_Z ;
                                                       // // } else   {
                                                       //    vector[local_size] log_Bound_U_Phi_Bound_Z =    log(Phi(Bound_Z[index])); 
                                                       //    vector[local_size] log_Phi_Z = log(u[index, t]) +  log_Bound_U_Phi_Bound_Z ;
                                                       //    Z_std_norm[index, t] =   std_normal_log_qf(log_Phi_Z);
                                                       //    y1[index, t]  =  log_Bound_U_Phi_Bound_Z ;
                                                       // // }
    
    
                                               }
                                               if (indicator_overflows_and_y_eq_1_empty == 0) {  //// overflow + y == 1 
    
                                                         array[num_Bound_Z_overflows_and_y_eq_1] int index = overflows_and_y_eq_1_index; 
                                                         int local_size = num_Bound_Z_overflows_and_y_eq_1;
    
    
                                                     // if (Phi_type == 2) {
                                                            vector[local_size] log_Bound_U_Phi_Bound_Z_1m =  log_inv_logit( - 0.07056 * square(Bound_Z[index]) .* Bound_Z[index]  - 1.5976 * Bound_Z[index] );
                                                          ///  vector[local_size] Bound_U_Phi_Bound_Z_1m = exp(log_Bound_U_Phi_Bound_Z_1m);
                                                         {
                                                          /// vector[local_size] Bound_U_Phi_Bound_Z =  1.0 - Bound_U_Phi_Bound_Z_1m;
                                                           
                                                           matrix[num_Bound_Z_overflows_and_y_eq_1, 2] tmp_array_2d_to_lse;
                                                           tmp_array_2d_to_lse[, 1] = log_Bound_U_Phi_Bound_Z_1m + log(u[index, t]);
                                                           vector[local_size] log_Bound_U_Phi_Bound_Z = log1m_exp(log_Bound_U_Phi_Bound_Z_1m);
                                                           tmp_array_2d_to_lse[, 2] =  log_Bound_U_Phi_Bound_Z;
                                                           vector[local_size] log_Phi_Z = log_sum_exp_2d(tmp_array_2d_to_lse);
                                                           
                                                           vector[local_size] log_1m_Phi_Z  =   log1m(u[index, t])  + log_Bound_U_Phi_Bound_Z_1m;
                                                           vector[local_size] logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
                                                           Z_std_norm[index, t] = inv_Phi_approx_from_logit_prob(logit_Phi_Z);
                                                         }
                                                          y1[index, t]  =  log_Bound_U_Phi_Bound_Z_1m ;
                                                     // // } else {
                                                     //         vector[local_size] Bound_U_Phi_Bound_Z_1m = Phi(-Bound_Z[index]);
                                                     //         vector[local_size] log_Bound_U_Phi_Bound_Z_1m =  log(Bound_U_Phi_Bound_Z_1m) ; // log_inv_logit( - 0.07056 * square(Bound_Z[index]) .* Bound_Z[index]  - 1.5976 * Bound_Z[index] );
                                                     //       //  vector[local_size] Bound_U_Phi_Bound_Z_1m = exp_approx(log_Bound_U_Phi_Bound_Z_1m);
                                                     //     {
                                                     //       matrix[num_Bound_Z_overflows_and_y_eq_1, 2] tmp_array_2d_to_lse;
                                                     //       tmp_array_2d_to_lse[, 1] = log_Bound_U_Phi_Bound_Z_1m + log(u[index, t]);
                                                     //       vector[local_size] log_Bound_U_Phi_Bound_Z = log1m(Bound_U_Phi_Bound_Z_1m);
                                                     //       tmp_array_2d_to_lse[, 2] =  log_Bound_U_Phi_Bound_Z;
                                                     //       vector[local_size] log_Phi_Z = log_sum_exp_2d(tmp_array_2d_to_lse);
                                                     //       Z_std_norm[index, t] =   std_normal_log_qf(log_Phi_Z);
                                                     //     }
                                                     //      y1[index, t]  =  log_Bound_U_Phi_Bound_Z_1m ;
                                                     // / / }
    
    
                                               }
                                        } 
    
                                     }
     
                                        if (t < n_tests)   inc = block(Z_std_norm, 1, 1, N, t) * to_vector(head(L_Omega[c, t + 1, ], t))   ;
    
                                }  // end of t loop 
    
                                lp[, c] = to_vector(rowwise_sum(y1[, 1:n_tests]))  +  to_vector(log_prev[, c])  ;    
    
                            } // end of c loop 
    
                            log_lik = log_sum_exp_2d(lp);

                     }

                  }

       } else { // Goodrich  method (big for-loop)

                   {
                        // likelihood (2 classes)
                        for (n in 1 : N) {
                          
                            matrix[n_class, n_tests] Xbeta_n;
                            vector[n_tests] u = lb_ub_lp(to_vector(u_raw[n,]), 0.0, 1.0);
                            vector[n_class] lp;
  
                            for (t in 1:n_tests) {
                              Xbeta_n[1, t] = to_row_vector(X_nd[t][n, 1:n_covs_per_outcome[1, t]]) * to_vector(beta[1, t, 1:n_covs_per_outcome[1, t]]);
                              Xbeta_n[2, t] = to_row_vector(X_d[t][n, 1:n_covs_per_outcome[2, t]])  * to_vector(beta[2, t, 1:n_covs_per_outcome[2, t]]);
                            }

                            for (c in 1:2) { // 2 classes

                                  vector[n_tests] Z_std_norm;
                                  vector[n_tests] y1;
                                  real inc  = 0.0;

                                  for (t in 1 : n_tests) {

                                             real Bound_Z =  -(Xbeta_n[c, t] + inc) * L_Omega_diag_recip[c, t];

                                             if (Phi_type == 2) {
                                                  real Bound_U_Phi_Bound_Z = Phi_approx(Bound_Z);
                                                  if (y[n, t] == 1) {
                                                    real Phi_Z = Bound_U_Phi_Bound_Z + (1.0 - Bound_U_Phi_Bound_Z) * u[t];
                                                    Z_std_norm[t] = inv_Phi_approx_from_prob(Phi_Z);
                                                    y1[t] = log1m(Bound_U_Phi_Bound_Z);
                                                  } else {
                                                    real Phi_Z = Bound_U_Phi_Bound_Z * u[t];
                                                    Z_std_norm[t] = inv_Phi_approx_from_prob(Phi_Z);
                                                    y1[t] = log(Bound_U_Phi_Bound_Z);
                                                  }
                                             } else {
                                                  real Bound_U_Phi_Bound_Z = Phi(Bound_Z);
                                                  if (y[n, t] == 1) {
                                                    real Phi_Z = Bound_U_Phi_Bound_Z + (1.0 - Bound_U_Phi_Bound_Z) * u[t];
                                                    Z_std_norm[t] = inv_Phi(Phi_Z);
                                                    y1[t] = log1m(Bound_U_Phi_Bound_Z);
                                                  } else {
                                                    real Phi_Z = Bound_U_Phi_Bound_Z * u[t];
                                                    Z_std_norm[t] = inv_Phi(Phi_Z);
                                                    y1[t] = log(Bound_U_Phi_Bound_Z);
                                                  }
                                             }

                                            if (t < n_tests)    inc = L_Omega[c, t + 1, 1 : t] * head(Z_std_norm, t);

                                  } // end of t loop

                                 lp[c] =  sum(y1) + log_prev[n, c];

                         } // end of c loop

                         log_lik[n] =  log_sum_exp(lp);

                      } // end of n loop

                  }


       }

    }

  }

}


model {

        for (c in 1 : n_class) {
            for (t in 1 : n_tests) {
                 for (k in 1 : n_covs_per_outcome[c, t]) {
                   beta[c, t, k] ~ normal(prior_beta_mean[c, k, t], prior_beta_sd[c, k, t]);
                }
            }
            target += lkj_corr_cholesky_lpdf(L_Omega[c,,] | prior_LKJ[c, 1]) ;
        }
        ////
        if (n_class > 1) {
          if (beta[2, 1, 1] < beta[1, 1, 1]) target += negative_infinity();
        }
        ////
        for (g in 1 : n_pops) {
          prev[g] ~ beta(prior_p_alpha[g, 1], prior_p_beta[g, 1]);
        }
        ////
        if (prior_only == 0) {
          for (n in 1 : N)
               target += log_lik[n];
        }

}


generated quantities {
     vector[n_tests] Se_bin;
     vector[n_tests] Sp_bin;
     vector[n_tests] Fp_bin;
     vector<lower=0, upper=1>[n_pops] p = prev;
    
     for (c in 1:n_class) {
  
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

}
