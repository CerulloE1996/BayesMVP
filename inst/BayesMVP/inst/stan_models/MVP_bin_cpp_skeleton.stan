// Internal one-class parameter reconstruction. This is not a standalone posterior model.
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
            //// First-column fixed entries bypass the raw transform and its Jacobian:
            ////
            ////
            ////
            matrix[dim, dim] L = diag_matrix(rep_vector(1.0, dim));
            vector[dim] D;
            D[1] = 1;
            ////
            //// Fix known values for first col (if any known):
            ////
            for (i in 2:dim) {
                if (known_values_indicator[i, 1] == 1) {
                    L[i, 1] = known_values[i, 1];
                } else {
                    if (lb_corr[i, 1] >= ub_corr[i, 1]) reject("empty first-column correlation interval");
                    L[i, 1] = lb_ub(Omega_raw_mat[i, 1], lb_corr[i, 1], ub_corr[i, 1]);
                }
                if (is_nan(L[i, 1]) || is_inf(L[i, 1]) || abs(L[i, 1]) >= 1)
                    reject("invalid first-column correlation");
            }
            D[2] = 1 - square(L[2, 1]); // After applying fixed entries.
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

                                    L[i, j] =  (known_values[i, j] - b1) / D[j];

                              } else {

                                    real sqrt_L_ij_old = sqrt(L_ij_old * D[j]);
                                    real low = max({-sqrt_L_ij_old, lb_corr[i, j] - b1});
                                    real up  = min({ sqrt_L_ij_old, ub_corr[i, j] - b1});
                                    if (is_nan(low) || is_nan(up) || is_inf(low) || is_inf(up) || low >= up)
                                        reject("empty or nonfinite correlation interval");

                                    real x = lb_ub(Omega_raw_mat[i, j], low, up);   ////  lb_ub(off_raw[counter], low, up);
                                    L[i, j] = x / D[j];

                                    // jacobian += -0.5 * log(D[j]);

                              }

                              L_ij_old -= D[j] * square(L[i, j]);
                              if (is_nan(L_ij_old) || is_inf(L_ij_old) || L_ij_old <= 0)
                                  reject("nonpositive or nonfinite LDL remainder");

                  }

                  D[i] = L_ij_old;

            }

            if (is_nan(sum(D)) || is_inf(sum(D)) || min(D) <= 0)
                reject("nonpositive or nonfinite LDL pivot");
            return diag_post_multiply(L, sqrt(D)); //// = L_Omega

     }
     
   
 
      // need to add citation to this (slight modification from a HP. calculators forum post)
      real inv_Phi_approx_from_prob(real p) { 
            return 5.494448514153059 *  sinh(0.33333333333333331483 * asinh( 0.34176618822627863 * logit(p)  )) ;
      }
      
      // need to add citation to this (slight modification from a HP. calculators forum post)
      vector inv_Phi_approx_from_prob(vector p) { 
            return 5.494448514153059 *  sinh(0.33333333333333331483 * asinh( 0.34176618822627863 * logit(p)  )) ;  
      }
      
      // need to add citation to this  (slight modification from a HP. calculators forum post)
      real inv_Phi_approx_from_logit_prob(real logit_p) { 
            return 5.494448514153059 *  sinh(0.33333333333333331483 * asinh( 0.34176618822627863 * logit_p  )) ; 
      }
      
      // need to add citation to this (slight modification from a HP. calculators forum post)
      vector inv_Phi_approx_from_logit_prob(vector logit_p) { 
            return 5.494448514153059 *  sinh(0.33333333333333331483 * asinh( 0.34176618822627863 *logit_p  )) ; 
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
  int<lower=1> n_covariates_max;
  int<lower=0,upper=1> corr_force_positive;
  array[n_tests] int<lower=1,upper=n_tests> test_perm;
  array[1] matrix[n_tests,n_tests] lb_corr;
  array[1] matrix[n_tests,n_tests] ub_corr;
  array[1] matrix[n_tests,n_tests] known_values_indicator_list;
  array[1] matrix[n_tests,n_tests] known_values_list;

}
transformed data {
  int n_corrs_per_class = n_tests * (n_tests - 1) %/% 2;
  matrix[n_tests,n_tests] lb_corr_actual = lb_corr[1];
  array[n_tests] int inv_perm;
  for (j in 1:n_tests) inv_perm[test_perm[j]] = j;
  if (corr_force_positive == 1) {
    for (i in 2:n_tests) for (j in 1:(i - 1)) lb_corr_actual[i,j] = 0;
  }

}
parameters {
  // Internal constrain-only skeleton. Nuisance values are sampled/stored by the native backend.
  array[1] vector[n_corrs_per_class] Omega_unconstrained_vec;
  array[1] matrix[n_covariates_max,n_tests] beta;

}
transformed parameters {
  array[1] matrix[n_tests,n_tests] L_Omega;
  array[1] matrix[n_tests,n_tests] Omega;
  array[1] matrix[n_tests,n_tests] Omega_orig;
  array[1] matrix[n_tests,n_tests] L_Omega_orig;
  L_Omega[1] = Pinkney_LDL_bounds_opt_without_J(Omega_unconstrained_vec[1], lb_corr_actual, ub_corr[1],
                                               known_values_indicator_list[1], known_values_list[1]);
  Omega[1] = multiply_lower_tri_self_transpose(L_Omega[1]);
  for (i in 1:n_tests) for (j in 1:n_tests) Omega_orig[1][i,j] = Omega[1][inv_perm[i],inv_perm[j]];
  L_Omega_orig[1] = cholesky_decompose(Omega_orig[1]);

}
model {
  // The full log density and manual gradients are evaluated by BayesMVP's native backend.
}
