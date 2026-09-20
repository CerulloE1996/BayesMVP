  
functions {
 
 
 matrix corr_to_chol(real x, int J) {
    matrix[J, J] cor = add_diag(rep_matrix(x, J, J), 1 - x);
    return cholesky_decompose(cor);
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
  
  
  
   matrix cholesky_corr_constrain_outer_lp( // matrix Omega_raw,
                                           vector col_one_raw, 
                                           vector off_raw,
                                           real lb, 
                                           real ub) {
                                             
   // segment(col(Omega_raw, 1), 2, K - 1);
   
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
  
   
 

    // need to add citation to this (slight modification from a forum post)
      real inv_Phi_approx_from_prob(real p) { 
        return 5.494 *  sinh(0.33333333333333331483 * asinh( 0.3418 * logit(p)  )) ;
      }
      
       // need to add citation to this (slight modification from a forum post)
      vector inv_Phi_approx_from_prob(vector p) { 
        return 5.494 *  sinh(0.33333333333333331483 * asinh( 0.3418 * logit(p)  )) ; 
      }
      
      
      
      // need to add citation to this (slight modification from a forum post)
      real inv_Phi_approx_from_logit_prob(real logit_p) { 
           return 5.494 *  sinh(0.33333333333333331483 * asinh( 0.3418 * logit_p  )) ; 
      }
      
      
       // need to add citation to this (slight modification from a forum post)
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
  /// matrix<lower=0>[N, n_tests]   y;  //////// data
  ////  array[N] int pop;
  int corr_force_positive;
  real overflow_threshold;
  real underflow_threshold;
  ///// covariate stuff 
  int n_covariates_max;
//  array[n_tests] matrix[N, n_covariates_max] X; /////// covariate array (can have  DIFFERENT NUMBERS of covariates for each  outcome - fill rest of array with 999999 if they vary between outcomes)
  array[n_tests] int n_covs_per_outcome;
  ///// priors 
  int prior_only;
  matrix[n_covariates_max, n_tests] prior_beta_mean;
  matrix<lower=0>[n_covariates_max, n_tests]   prior_beta_sd;
  real prior_LKJ;
  ////// other 
  int<lower=0, upper=(n_tests * (n_tests - 1)) %/% 2> known_num;
  // matrix[n_tests, n_tests] lb_corr;
  // matrix[n_tests, n_tests] ub_corr;
  
}


transformed data {
   
    int k_choose_2 = (n_tests * (n_tests - 1)) / 2; 
    int km1_choose_2 = ((n_tests - 1) * (n_tests - 2)) / 2;
  
    int n_covariates_total =    sum( (n_covs_per_outcome));

    real<lower=-1, upper=1> lb;
    real<lower=lb, upper=1> ub = 1.0;
  
    if (corr_force_positive == 1)  lb = 0;
    else lb = -1.0;

}


parameters {

      matrix[N, n_tests] u_raw; //  put nuisance parameters FIRST (NOTE: doesnt have to be on "raw" scale to work as grad is computed w.r.t unconstrained anyway!)
      vector[n_tests - 1] col_one_raw;
      vector[km1_choose_2 - known_num] off_raw;
      vector[n_covariates_total] beta_vec;

}

 
 
transformed parameters {
  
     //matrix<lower=0, upper=1>[N, n_tests]  u = Phi(u_raw);
     array[n_tests, n_covariates_max] real beta;
     matrix[n_tests, n_tests] Omega;
     matrix[n_tests, n_tests] L_Omega;
     
     
      {
            int counter = 1;
        
                      for (t in 1:n_tests) {
                        for (k in 1:n_covs_per_outcome[t]) {
                           beta[t, k] = beta_vec[counter];
                           counter += 1;
                        }
                      }
                    L_Omega =   cholesky_corr_constrain_outer_lp( col_one_raw, off_raw, lb, ub);
                    Omega = multiply_lower_tri_self_transpose(L_Omega);
        
      }
      
}



model {


     
            for (t in 1 : n_tests) {
                 for (k in 1 : n_covs_per_outcome[t]) {
                   beta[t, k] ~ normal(prior_beta_mean[k, t], prior_beta_sd[k, t]);
                }
            }
             target += lkj_corr_cholesky_lpdf(L_Omega  | prior_LKJ) ;
              
              
           //// dummy priors
           to_vector(u_raw) ~ normal(0, 1); 
           to_vector(beta_vec) ~ normal(0, 1); 


}

 





