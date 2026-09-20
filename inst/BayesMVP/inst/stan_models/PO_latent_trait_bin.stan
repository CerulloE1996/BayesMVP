  
functions {
 
 
 
       matrix cov2cor(matrix V) {
        int p = rows(V);
        vector[p] Is = inv_sqrt(diagonal(V));
        return quad_form_diag(V, Is); 
      }
      
      
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
  int corr_force_positive;
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

      matrix[N, n_tests] u_raw; //  put nuisance parameters FIRST (NOTE: doesnt have to be on "raw" scale to work as grad is computed w.r.t unconstrained anyway!)
      vector[n_class * n_tests] LT_a_vec; /// use LT_a in the MVP model. 
      vector[n_class * n_tests] LT_b_raw_vec;
      vector[n_pops]  p_raw;
      
}

 
 
transformed parameters {
  
     // matrix<lower=0, upper=1>[N, n_tests]  u = Phi(u_raw);
     array[n_class, n_tests, 1] real beta;/// "equiv."" to beta in MVP model --- we are putting a prior on this (so need Jacobian adjustment!!)
     array[n_class] matrix[n_tests, n_tests] Omega; // "equiv."" to Omega in LC_MVP model
     vector<lower=0, upper=1>[n_pops]  p = lb_ub_lp(p_raw, 0.0, 1.0);
     ///// latent_trait-specific pars: 
     matrix[n_class, n_tests] LT_a; /// use LT_a in the MVP model. 
     matrix[n_class, n_tests] LT_b;
     array[n_class] matrix[n_tests, n_tests] Sigma;
     array[n_class] matrix[n_tests, n_tests] L_Sigma; /// use L_Sigma in the MVP model. 
  
     
     
           {
              int counter = 1;
              for (c in 1 : n_class) {
                        for (t in 1:n_tests) {
                        //  for (k in 1:n_covs_per_outcome[c, t]) {
                             LT_a[c, t] = LT_a_vec[counter];
                             counter += 1;
                         // }
                        }
              }
            }
      
      {
         int counter = 1;
         for (c in 1 : n_class) {
           for (t in 1 : n_tests) {
             LT_b[c, t] = exp(LT_b_raw_vec[counter]); 
             beta[c, t, 1] = LT_a[c, t] / sqrt(1.0 + square(LT_b[c, t])); // we are putting a prior on the TRANSFORMED parameter (beta) - hence need Jacobian adjustment
             counter += 1;
           }
         }
      }
                     
                   
            //// get covariance and correlation matrix. Note LT model is MVP w/ covariance mtx equal to I + b*b'. 
            for (c in 1 : n_class) {  
                    Sigma[c, :  ] =  diag_matrix(rep_vector(1.0, n_tests)) + to_vector(LT_b[c, ]) *  to_row_vector(LT_b[c, ]);
                    L_Sigma[c,  :] = cholesky_decompose(Sigma[c, :  ]);
                    Omega[c, :  ] = cov2cor(Sigma[c, :  ]);
            }
      
      
}



model {

         //// set prior on TRANSFORMED parameter - beta - equiv to beta in LC-MVP model!!!
        for (c in 1 : n_class) {
            for (t in 1 : n_tests) {
                   //// prior directly on the mean params
                   beta[c, t, 1] ~ normal(prior_beta_mean[c, 1, t], prior_beta_sd[c, 1, t]);
                   //// Jacobian adjustment for beta -> LT_a 
                   target += - 0.5 * log(1.0 + square(LT_b[c, t])); 
                   //// prior for corr
                   LT_b[c, t] ~ weibull(LT_b_priors_shape[c, t], LT_b_priors_scale[c, t]);
                   // target += LT_b_raw_vec[c, t]; 
            }
        }
        
         target += sum(LT_b_raw_vec); //// Jacobian adjustment for corr / b's
        
         


              for (g in 1 : n_pops) {
                p[g] ~ beta(prior_p_alpha[g], prior_p_beta[g]);
              }
              
 
         //// dummy priors
         to_vector(u_raw) ~ normal(0, 1);  
         to_vector(LT_b_raw_vec) ~ normal(0, 1);
         to_vector(LT_a) ~ normal(0, 1);
         to_vector(p_raw) ~ normal(0, 1);
         
}

 
 
generated quantities {

    vector[n_tests] Se_bin;
    vector[n_tests] Sp_bin;
    vector[n_tests] Fp_bin;


   for (c in 1:n_class) {


      for (t in 1:n_tests) { // for binary tests

         if (n_class == 2) { // summary Se and Sp only calculated if n_class = 2 (i.e. the "standard" # of classes for DTA)
              Se_bin[t]  =        Phi(   beta[2, t, 1]   );
              Sp_bin[t]  =    1 - Phi(   beta[1, t, 1]   );
              Fp_bin[t]  =    1 - Sp_bin[t];
        }
        else {
          Se_bin[t] = 999;
          Sp_bin[t] = 999;
          Fp_bin[t] = 999;
        }
    }



}



}


