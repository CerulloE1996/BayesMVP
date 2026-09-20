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
      
  
      // matrix cholesky_corr_constrain_outer_lp( // matrix Omega_raw,
      //                                      vector col_one_raw, 
      //                                      vector off_raw,
      //                                      real lb, 
      //                                      real ub) {
      // 
      //       int K = num_elements(col_one_raw) + 1;
      //       vector[K - 1] z = lb_ub_lp(col_one_raw, lb, ub);
      //       matrix[K, K] L = diag_matrix(rep_vector(1, K));
      //       vector[K] D;
      //       D[1] = 1;
      //       L[2:K, 1] = z[1:K - 1];
      //       D[2] = 1 - L[2, 1]^2;
      //       int cnt = 1;
      //      
      //       for (i in 3:K) {
      //          D[i] = 1 - L[i, 1]^2; 
      //          L[i, 2:i - 1] = rep_row_vector(1 - L[i, 1]^2, i - 2);
      //          real l_ij_old = L[i, 2];
      //         for (j in 2:i - 1) {
      //         //  real l_ij_old = L[i, j];
      //           real b1 = dot_product(L[j, 1:(j - 1)], D[1:j - 1]' .* L[i, 1:(j - 1)]);
      //             
      //             // how to derive the bounds
      //             // we know that the correlation value C is bound by
      //             // b1 - Ljj * Lij_old <= C <= b1 + Ljj * Lij_old
      //             // Now we want our bounds to be enforced too so
      //             // max(lb, b1 - Ljj * Lij_old) <= C <= min(ub, b1 + Ljj * Lij_old)
      //             // We have the Lij_new = (C - b1) / Ljj
      //             // To get the bounds on Lij_new is
      //             // (bound - b1) / Ljj 
      //             
      //             real low = max({-sqrt(l_ij_old) * D[j], lb - b1});
      //             real up = min({sqrt(l_ij_old) * D[j], ub - b1}); 
      //             
      //             real x = lb_ub_lp(off_raw[cnt], low, up);
      //             L[i, j] = x / D[j]; 
      //   
      //             target += -0.5 * log(D[j]);
      //             
      //              l_ij_old *= 1 - (D[j] * L[i, j]^2) / l_ij_old;
      //             
      //            // real mul = 1 - (D[j] * L[i, j]^2) / l_ij_old;
      //            // L[i, (j + 1):i - 1] *= mul;
      //             //D[i] *= mul;
      //             cnt += 1;
      //           }
      //           D[i] = l_ij_old;
      //         }
      //       
      //         return diag_post_multiply(L, sqrt(D));
      //       
      // }
      
      
      ////
      //// This Stan function was provided by Sean Pinkney (may be slightly adapted/different formatting):
      ////
      matrix Pinkney_LDL_bounds_opt_without_J(   vector Omega_raw_vec,
                                                 matrix lb_corr,
                                                 matrix ub_corr,
                                                 matrix known_values_indicator,
                                                 matrix known_values) {

            ////
            //// First construct "Omega_raw_mat" matrix from "Omega_raw_vec" using row-by-row access (as MUST match the BayesMVP C++ functions/models!):
            ////
            int dim = rows(lb_corr);
            matrix[dim, dim] Omega_raw_mat = rep_matrix(0.0, dim, dim);
            // for (i in 2:dim) {
            //    Omega_raw_mat[i, i] = 1.0;
            // }
            int counter = 1;
            for (i in 2:dim) {
              for (j in 1:(i - 1)) { //// j < i
                  Omega_raw_mat[i, j] = Omega_raw_vec[counter];
                  counter += 1;
              }
            }
            ////
            //// Then get 1st raw col excluding first element (0.0):
            ////
            vector[dim - 1] col_one_raw = segment(col(Omega_raw_mat, 1), 2, dim - 1); //// Extract first col. excluding the first element (0.0):
            ////
            vector[dim - 1] z = lb_ub(col_one_raw, segment(col(lb_corr, 1), 2, dim - 1), segment(col(ub_corr, 1), 2, dim - 1));
            ////
            matrix[dim, dim] L = diag_matrix(rep_vector(1.0, dim));
            vector[dim] D;
            D[1] = 1;
            L[2:dim, 1] = z; //// [1:dim - 1];
            D[2] = 1 - L[2, 1]^2;
            ////
            //// Fix known values for first col (if any known):
            ////
            for (i in 2:dim) { ////  skip first element since this is fixed to 1.0 (as corr matrix).
                if (known_values_indicator[i, 1] == 1)    L[i, 1] = known_values[i, 1];
            }
            ////
            //// Compute L_Omega:
            ////
            for (i in 3:dim) {

                   D[i] = 1 - L[i, 1]^2;
                   L[i, 2:i - 1] = rep_row_vector(1 - L[i, 1]^2, i - 2);
                   real L_ij_old = L[i, 2];

                   for (j in 2:(i - 1))  {

                              //  real L_ij_old = L[i, j];
                              real b1 = dot_product(L[j, 1:(j - 1)], D[1:j - 1]' .* L[i, 1:(j - 1)]);

                              // how to derive the bounds
                              // we know that the correlation value C is bound by
                              // b1 - Ljj * Lij_old <= C <= b1 + Ljj * Lij_old
                              // Now we want our bounds to be enforced too so
                              // max(lb_corr, b1 - Ljj * Lij_old) <= C <= min(ub_corr, b1 + Ljj * Lij_old)
                              // We have the Lij_new = (C - b1) / Ljj
                              // To get the bounds on Lij_new is
                              // (bound - b1) / Ljj

                              if (known_values_indicator[i, j] == 1)  {

                                    L[i, j] =  known_values[i, j] / D[j];

                              } else {

                                    real sqrt_L_ij_old = sqrt(L_ij_old);
                                    real low = max({-sqrt_L_ij_old * D[j], lb_corr[i, j] - b1});
                                    real up  = min({ sqrt_L_ij_old * D[j], ub_corr[i, j] - b1});

                                    real x = lb_ub(Omega_raw_mat[i, j], low, up);   ////  lb_ub(off_raw[counter], low, up);
                                    L[i, j] = x / D[j];

                                    // jacobian += -0.5 * log(D[j]);

                              }

                              L_ij_old *= 1.0 - (D[j] * L[i, j]^2) / L_ij_old;

                  }

                  D[i] = L_ij_old;

            }

            return diag_post_multiply(L, sqrt(D)); //// = L_Omega

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
      int n_covariates_max_nd;
      int n_covariates_max_d;
      int n_covariates_max;
      array[n_tests] matrix[N, n_covariates_max_nd] X_nd; /////// covariate array (can have  DIFFERENT NUMBERS of covariates for each  outcome - fill rest of array with 999999 if they vary between outcomes)
      array[n_tests] matrix[N, n_covariates_max_d]  X_d; /////// covariate array (can have  DIFFERENT NUMBERS of covariates for each  outcome - fill rest of array with 999999 if they vary between outcomes)
      array[n_class, n_tests] int n_covs_per_outcome;
      ////
      int corr_force_positive;
      int<lower=0> known_num;
      ////
      array[n_class] matrix[n_tests, n_tests] known_values_list;
      array[n_class] matrix[n_tests, n_tests] known_values_indicator_list;
      ////
      array[n_class] matrix[n_tests, n_tests] lb_corr;
      array[n_class] matrix[n_tests, n_tests] ub_corr;
      matrix<lower=0>[n_class, 1] prior_LKJ; // NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns 
      ////
      real overflow_threshold;
      real underflow_threshold;
      /////
      int prior_only;
      ////
      array[n_class] matrix[n_covariates_max, n_tests] prior_beta_mean;  ////  // array[n_class, n_tests, n_covariates_max]  real prior_beta_mean;
      array[n_class] matrix<lower=0>[n_covariates_max, n_tests] prior_beta_sd;     //// array[n_class, n_tests, n_covariates_max]  real<lower=0> prior_beta_sd;
      ////
      matrix<lower=0>[n_pops, 1] prev_prior_a; // NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns ; ## NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns 
      matrix<lower=0>[n_pops, 1] prev_prior_b; // NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns 
      ///// other
      int Phi_type;
      int handle_numerical_issues;
      int fully_vectorised;
}


transformed data {
  
     array[n_class] matrix[n_tests, n_tests] lb_corr_actual = lb_corr;
     ////
     if (corr_force_positive == 1) {  //// override "lb_corr" if corr_force_positive == 1.
        for (c in 1:n_class) {
          for (i in 2:n_tests) {
            for (j in 1:(i - 1)) {
              lb_corr_actual[c][i, j] = 0.0;
            }
          }
        }
     }
      
     int k_choose_2 = (n_tests * (n_tests - 1)) / 2;
     int n_corrs_per_class = k_choose_2;
     int n_corrs = n_class*n_corrs_per_class;
}


parameters {
     //// matrix[N, n_tests] u_raw; //  put nuisance parameters FIRST (NOTE: doesnt have to be on "raw" scale to work as grad is computed w.r.t unconstrained anyway!)
     array[n_class] vector[n_corrs_per_class] Omega_unconstrained_vec;
     array[n_class] matrix[n_covariates_max, n_tests] beta; 
     vector[n_pops] p_raw;
}


transformed parameters { 
     vector<lower=0, upper=1>[n_pops] prev = lb_ub(p_raw, 0.0, 1.0);
     array[n_class] matrix[n_tests, n_tests] L_Omega;
     array[n_class] matrix[n_tests, n_tests] Omega;
        
     for (c in 1:n_class) {
              L_Omega[c] = Pinkney_LDL_bounds_opt_without_J( Omega_unconstrained_vec[c], 
                                                             lb_corr_actual[c], 
                                                             ub_corr[c],
                                                             known_values_indicator_list[c], 
                                                             known_values_list[c]);
              Omega[c] = multiply_lower_tri_self_transpose(L_Omega[c, :]);
     }
}


model {
     for (c in 1:n_class) {
          for (i in 2:n_tests) {
            for (j in 1:(i - 1)) {
              if (known_values_indicator_list[c][i, j] == 1) {
                  Omega[c][i, j] ~ normal(0.0, 10.0); //// to ensure any corr's we aren't estimating dont cause divergences
              }
            }
          }
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
}






