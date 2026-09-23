

functions {

      // matrix corr_to_chol(real x, int J) {
       //    matrix[J, J] cor = add_diag(rep_matrix(x, J, J), 1 - x);
       //    return cholesky_decompose(cor); 
      // }
      ////
      //// Function to manually construct a cutpoint vector from raw unconstrained parameters - using exp() / log-differences. 
      ////
      vector construct_C(  vector C_raw_vec, 
                           int softplus) {
        
        
                int n_total_cutpoints = num_elements(C_raw_vec);
                vector[n_total_cutpoints] C_vec;  
                //// ----  1st cutpoint (for each test) is unconstrained (no Jacobian needed):
                C_vec[1] = C_raw_vec[1];
                //// ---- Rest of cutpoints are made using LOG-differences (OR softplus-differences):
                if (softplus == 1) {
                        vector[n_total_cutpoints - 1] softplus_C_vec = log1p_exp(C_raw_vec[2:n_total_cutpoints]);
                        for (k in 2:n_total_cutpoints) {
                                   C_vec[k] = C_vec[k - 1] + softplus_C_vec[k - 1];
                                   // jacobian += log_inv_logit(C_raw_vec[k]);   //// Jacobian for trans. C_raw -> C:
                        }
                } else { 
                        vector[n_total_cutpoints - 1] exp_C_vec = exp(C_raw_vec[2:n_total_cutpoints]);
                        for (k in 2:n_total_cutpoints) {
                                   C_vec[k] = C_vec[k - 1] + exp_C_vec[k - 1];
                                   // jacobian += C_raw_vec[k];  //// Jacobian for trans. C_raw -> C:
                        }
                }
                //// ---- Output cutpoints + Jacobian (as 1st element in output vector):
                return(C_vec);
        
      }
      row_vector construct_C(  row_vector C_raw_vec, 
                               int softplus) {
        
                int n_total_cutpoints = num_elements(C_raw_vec);
                row_vector[n_total_cutpoints] C_vec;  
                //// ----  1st cutpoint (for each test) is unconstrained (no Jacobian needed):
                C_vec[1] = C_raw_vec[1];
                //// ---- Rest of cutpoints are made using LOG-differences (OR softplus-differences):
                if (softplus == 1) {
                        row_vector[n_total_cutpoints - 1] softplus_C_vec = log1p_exp(C_raw_vec[2:n_total_cutpoints]);
                        for (k in 2:n_total_cutpoints) {
                                   C_vec[k] = C_vec[k - 1] + softplus_C_vec[k - 1];
                                   // jacobian += log_inv_logit(C_raw_vec[k]);   //// Jacobian for trans. C_raw -> C:
                        }
                } else { 
                        row_vector[n_total_cutpoints - 1] exp_C_vec = exp(C_raw_vec[2:n_total_cutpoints]);
                        for (k in 2:n_total_cutpoints) {
                                   C_vec[k] = C_vec[k - 1] + exp_C_vec[k - 1];
                                   // jacobian += C_raw_vec[k];  //// Jacobian for trans. C_raw -> C:
                        }
                }
                //// ---- Output cutpoints + Jacobian (as 1st element in output vector):
                return(C_vec);
        
      }
      ////
      //// Helper function to calculate start and end indices:
      ////
      array[] int calculate_start_indices( array[] int n_thr, 
                                           int n_tests) {
        
            array[n_tests] int start_index;
            
            start_index[1] = 1;
            for (t in 2:n_tests) {
              start_index[t] = start_index[t - 1] + n_thr[t - 1];
            }
            
            return start_index;
        
      }
      
      ////
      //// Helper function to calculate start and end indices:
      ////
      array[] int calculate_end_indices( array[] int n_thr, 
                                         int n_tests,
                                         array[] int start_index) {
                                           
            array[n_tests] int end_index;
            
            for (t in 1:n_tests) {
              end_index[t] = start_index[t] + n_thr[t] - 1;
            }
            
            return end_index;
        
      }
      
      ////
      //// Helper function to calculate start and end indices:
      ////
      array[] int calculate_start_indices_study( int n_studies,
                                                 array[] int n_thr, 
                                                 int n_tests) {
        
            array[n_tests] int start_index;
            
            start_index[1] = 1;
            for (t in 2:n_tests) {
              start_index[t] = start_index[t - 1] + (n_studies * n_thr[t - 1]);
            }
            
            return start_index;
        
      }
      
      ////
      //// Helper function to calculate start and end indices:
      ////
      array[] int calculate_end_indices_study( int n_studies, 
                                               array[] int n_thr, 
                                               int n_tests,
                                               array[] int start_index) {
                                           
            array[n_tests] int end_index;
            
            for (t in 1:n_tests) {
              end_index[t] = start_index[t] + (n_thr[t] * n_studies) - 1;
            }
            
            return end_index;
        
      }
      
      ////
      //// Function to access elements from a flattened vector representing a ragged array:
      ////
      vector get_test_values( vector flat_values, 
                              data array[] int start_index, 
                              data array[] int end_index, 
                              data int test_index) {
        
              int n_elements = end_index[test_index] - start_index[test_index] + 1;
              vector[n_elements] result;
              
              for (i in 1:n_elements) {
                result[i] = flat_values[start_index[test_index] + i - 1];
              }
              
              return result;
        
      }
      
      ////
      //// Function to set values in the flattened vector:
      ////
      vector update_test_values( vector flat_values_to_update, 
                                 vector new_values, 
                                 data array[] int start_index, 
                                 data array[] int end_index, 
                                 data int test_index) {
        
              int n_elements = end_index[test_index] - start_index[test_index] + 1;
              vector[num_elements(flat_values_to_update)] result = flat_values_to_update;
              
              for (i in 1:n_elements) {
                result[start_index[test_index] + i - 1] = new_values[i];
              }
              
              return result;
            
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
      ////
      int<lower=0> n_binary_tests; //// ---- ordinal-only (int)
      int<lower=0> n_ordinal_tests; //// ---- ordinal-only (int)
      ////
      array[n_ordinal_tests] int n_cat_per_ord_test; //// ---- ordinal-only (col vec, int)
      array[n_ordinal_tests] int n_thr_per_ord_test; //// ---- ordinal-only (col vec, int)
      ////
      array[2] matrix<lower=0>[max(n_cat_per_ord_test), n_ordinal_tests] prior_dirichlet_alpha; //// ---- ordinal-only (col vec, double)
      ////
      matrix<lower=0>[N, n_tests] y;  //////// data
      int<lower=2> n_class;
      ////
      array[n_tests] int test_perm;
      array[n_tests] int n_cat_per_test;   //// FITTED slot order; 2 == binary
      ////
      int<lower=1> n_pops; //// ----
      array[N] int pop; //// ----
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
      ////
      ////
      //// ---- Bounds on the RAW ordinal cutpoint params (element 1 = first cutpoint;
      ////      elements 2+ = LOG-gaps, so gaps are bounded to (exp(lo), exp(hi))):
      ////
      real C_raw_lower; //// ordinal-only
      real C_raw_upper; //// ordinal-only
      /////
      int prior_only;
      ////
      array[n_class] matrix[n_covariates_max, n_tests] prior_beta_mean;  ////  // array[n_class, n_tests, n_covariates_max]  real prior_beta_mean;
      array[n_class] matrix<lower=0>[n_covariates_max, n_tests] prior_beta_sd;     //// array[n_class, n_tests, n_covariates_max]  real<lower=0> prior_beta_sd;
      ////
      matrix<lower=0>[n_pops, 1] prior_prev_a; // NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns ; ## NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns 
      matrix<lower=0>[n_pops, 1] prior_prev_b; // NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns 
      ///// other
      int Phi_type;
      int handle_numerical_issues;
      int fully_vectorised;
      ////
      //// ---- Baseline covariate cases for post-hoc Se/Sp:
      ////
      array[n_tests] vector[n_covariates_max] baseline_case_nd;
      array[n_tests] vector[n_covariates_max] baseline_case_d;
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
     ////
     int k_choose_2 = (n_tests * (n_tests - 1)) / 2;
     int n_corrs_per_class = k_choose_2;
     int n_corrs = n_class*n_corrs_per_class;
     ////
     //// ---- Ordinal only:
     ////
     //// ---- Calculate indices:
     ////
     array[n_ordinal_tests] int start_index = calculate_start_indices(n_thr_per_ord_test, n_ordinal_tests);
     array[n_ordinal_tests] int end_index   = calculate_end_indices(n_thr_per_ord_test, n_ordinal_tests, start_index);
     ////
     array[n_tests] int inv_perm;
     for (j in 1:n_tests) {
        inv_perm[test_perm[j]] = j;
     }
     ////
     //// ---- slot-order ordinal index per FITTED slot (-1 for binary) — mirrors C++ ord_idx_of_test:
     ////
     array[n_tests] int ord_idx_of_slot;
     {
       int k = 0;
       for (j in 1:n_tests) {
         if (n_cat_per_test[j] > 2) { k += 1; ord_idx_of_slot[j] = k; }
         else                       { ord_idx_of_slot[j] = -1; }
       }
     }
}


parameters {
     //// matrix[N, n_tests] u_raw; //  put nuisance parameters FIRST (NOTE: doesnt have to be on "raw" scale to work as grad is computed w.r.t unconstrained anyway!)
     array[n_class] vector[n_corrs_per_class] Omega_unconstrained_vec;
     array[n_class] matrix[n_covariates_max, n_tests] beta; 
     vector[n_pops] p_raw;
     // array[2] vector[sum(n_thr_per_ord_test)] C_unc_vec;  //// RAW LOG-DIFFS - Global cutpoints for each test (staggered array/matrix using "n_thr_per_ord_test[t]" to index correctly)
     array[2] vector[sum(n_thr_per_ord_test)] C_unc_vec;  //// UNCONSTRAINED (mapped to bounded raw log-diffs via lb_ub in TP block)
}


transformed parameters { 
     vector<lower=0, upper=1>[n_pops] prev = lb_ub(p_raw, 0.0, 1.0);
     array[n_class] matrix[n_tests, n_tests] L_Omega;
     array[n_class] matrix[n_tests, n_tests] Omega;
     array[2] vector[sum(n_thr_per_ord_test)] C_raw_vec;
        
     for (c in 1:n_class) {
              L_Omega[c] = Pinkney_LDL_bounds_opt_without_J( Omega_unconstrained_vec[c], 
                                                             lb_corr_actual[c], 
                                                             ub_corr[c],
                                                             known_values_indicator_list[c], 
                                                             known_values_list[c]);
              Omega[c] = multiply_lower_tri_self_transpose(L_Omega[c, :]);
     }
     ////
     array[n_class] matrix[n_tests, n_tests] Omega_orig;
     array[n_class] matrix[n_tests, n_tests] L_Omega_orig;
     ////
     for (c in 1:n_class) {
        for (i in 1:n_tests) {
          for (j in 1:n_tests) {
            Omega_orig[c][i, j] = Omega[c][inv_perm[i], inv_perm[j]];
          }
        }
        L_Omega_orig[c] = cholesky_decompose(Omega_orig[c]);
     }
     ////
     //// ---- Construct (global) cutpoints:
     ////
     array[2] vector[sum(n_thr_per_ord_test)] C_vec;  //// Global cutpoints for each test ("staggered" array/matrix using "n_thr_per_ord_test[t]" to index correctly)
     ////
     // for (t in 1:n_ordinal_tests) {
     //     for (c in 1:2) {
     //           int n_thr_t = n_thr_per_ord_test[t];
     //           vector[n_thr_t] C_raw_vec_test_t = get_test_values(C_unc_vec[c], start_index, end_index, t);
     //           int softplus = 0;
     //           vector[n_thr_t] C_vec_test_t =  construct_C(C_raw_vec_test_t, softplus);
     //           C_vec[c] = update_test_values(C_vec[c], C_vec_test_t, start_index, end_index, t);
     //     }
     // }
     for (t in 1:n_ordinal_tests) {
         for (c in 1:2) {
               int n_thr_t = n_thr_per_ord_test[t];
               ////
               //// ---- Unconstrained -> BOUNDED raw scale (matches the C++ lb_ub transform;
               ////      no Jacobian here -- this skeleton is the "_without_J" variant):
               ////
               vector[n_thr_t] C_unc_test_t = get_test_values(C_unc_vec[c], start_index, end_index, t);
               vector[n_thr_t] C_raw_test_t = lb_ub(C_unc_test_t, C_raw_lower, C_raw_upper);
               ////
               C_raw_vec[c] = update_test_values(C_raw_vec[c], C_raw_test_t, start_index, end_index, t);
               ////
               int softplus = 0;
               vector[n_thr_t] C_vec_test_t = construct_C(C_raw_test_t, softplus);
               C_vec[c] = update_test_values(C_vec[c], C_vec_test_t, start_index, end_index, t);
         }
     }
     ////
     array[n_tests] real Xbeta_baseline_nd;
     array[n_tests] real Xbeta_baseline_d;
     ////
     //// ---- Compute Xbeta_baseline for ALL tests:
     ////
     //// t = ORIGINAL test index (canonical data is binaries-first); s = fitted slot:
     for (t in 1:n_tests) {
        int s = inv_perm[t];
        Xbeta_baseline_nd[t] = dot_product( baseline_case_nd[s][1:n_covs_per_outcome[1, s]],
                                            beta[1][1:n_covs_per_outcome[1, s], s] );
        Xbeta_baseline_d[t]  = dot_product( baseline_case_d[s][1:n_covs_per_outcome[2, s]],
                                            beta[2][1:n_covs_per_outcome[2, s], s] );
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
      ////
      vector<lower=0, upper=1>[n_pops] p = prev;
      ////
      //// ---- Baseline Se/Sp (dot product of baseline_case with beta, mirroring MetaOrdDTA):
      ////
      vector[n_binary_tests] Se_baseline_bin;
      vector[n_binary_tests] Sp_baseline_bin;
      vector[n_binary_tests] Fp_baseline_bin;
      ////
      array[n_ordinal_tests] vector[max(n_thr_per_ord_test)] Se_baseline_ord;
      array[n_ordinal_tests] vector[max(n_thr_per_ord_test)] Sp_baseline_ord;
      array[n_ordinal_tests] vector[max(n_thr_per_ord_test)] Fp_baseline_ord;
      ////
      ////
      // vector[N] log_lik = rep_vector(0.0, N);
      ////
      //// ======================================================================
      //// ---- Baseline Se/Sp (dot product with baseline covariate vector):
      //// ======================================================================
      ////
      //// ---- Binary baseline Se/Sp:
      ////
      if (n_class == 2) {
          for (t in 1:n_binary_tests) {   //// original binary tests are 1..n_binary_tests
              Se_baseline_bin[t] =       Phi( Xbeta_baseline_d[t] );
              Fp_baseline_bin[t] =       Phi( Xbeta_baseline_nd[t] );
              Sp_baseline_bin[t] = 1.0 - Fp_baseline_bin[t];
          }
      }
      ////
      //// ---- Ordinal baseline Se/Sp:
      ////
      if (n_class == 2) {
          for (t_ord in 1:n_ordinal_tests) {
              int t     = n_binary_tests + t_ord;      //// original ordinal test
              int s     = inv_perm[t];                 //// its fitted slot (may sit anywhere)
              int s_ord = ord_idx_of_slot[s];          //// fitted-slot ordinal index
              int n_thr_t = n_thr_per_ord_test[s_ord];
              ////
              Se_baseline_ord[t_ord] = rep_vector(-1.0, max(n_thr_per_ord_test));
              Sp_baseline_ord[t_ord] = rep_vector(-1.0, max(n_thr_per_ord_test));
              Fp_baseline_ord[t_ord] = rep_vector(-1.0, max(n_thr_per_ord_test));
              ////
              vector[n_thr_t] C_nd_t = get_test_values(C_vec[1], start_index, end_index, s_ord);
              vector[n_thr_t] C_d_t  = get_test_values(C_vec[2], start_index, end_index, s_ord);
              for (k in 1:n_thr_t) {
                  Fp_baseline_ord[t_ord][k] =       Phi( -(C_nd_t[k] - Xbeta_baseline_nd[t]) );
                  Sp_baseline_ord[t_ord][k] = 1.0 - Fp_baseline_ord[t_ord][k];
                  Se_baseline_ord[t_ord][k] =       Phi( -(C_d_t[k]  - Xbeta_baseline_d[t])  );
              }
          }
      }
      // ////
      // //// ---- Likelihood:
      // ////
      // if (prior_only == 0) {
      //     
      //     // ---- Non-vectorised loop version (works for both binary + ordinal):
      //     {
      //         for (n in 1:N) {
      //             
      //             vector[n_tests] u = lb_ub(to_vector(u_raw[n,]), 0.0, 1.0);
      //             vector[n_class] lp;
      //             vector[n_class] log_prev;
      //             
      //             log_prev[1] = log1m(prev[pop[n]]);
      //             log_prev[2] = log(prev[pop[n]]);
      //             
      //             for (c in 1:n_class) {
      //                 
      //                 vector[n_tests] Z_std_norm;
      //                 vector[n_tests] y1;
      //                 real inc = 0.0;
      //                 
      //                 for (t in 1:n_tests) {
      //                     
      //                     real Xbeta_nt;
      //                     if (n_covariates_max > 1) {
      //                         if (c == 1) Xbeta_nt = to_row_vector(X_nd[t][n, 1:n_covs_per_outcome[c,t]]) * to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);
      //                         else        Xbeta_nt = to_row_vector(X_d[t][n, 1:n_covs_per_outcome[c,t]])  * to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);
      //                     } else {
      //                         Xbeta_nt = beta[c, t, 1];
      //                     }
      //                     
      //                     if (t <= n_binary_tests) {
      //                       
      //                             ////
      //                             //// ---- BINARY test:
      //                             ////
      //                             real Bound_Z = -(Xbeta_nt + inc) * L_Omega_diag_recip[c, t];
      //                             
      //                             real Bound_U_Phi_BZ = Phi(Bound_Z);
      //                             if (y[n, t] == 1) {
      //                                 real Phi_Z = Bound_U_Phi_BZ + (1.0 - Bound_U_Phi_BZ) * u[t];
      //                                 Z_std_norm[t] = inv_Phi(Phi_Z);
      //                                 y1[t] = log1m(Bound_U_Phi_BZ);
      //                             } else {
      //                                 real Phi_Z = Bound_U_Phi_BZ * u[t];
      //                                 Z_std_norm[t] = inv_Phi(Phi_Z);
      //                                 y1[t] = log(Bound_U_Phi_BZ);
      //                             }
      //                         
      //                     } else {
      //                       
      //                             ////
      //                             //// ---- ORDINAL test:
      //                             ////
      //                             int t_ord = t - n_binary_tests;
      //                             int k_obs = to_int(y[n, t]);  // observed category (1-indexed)
      //                             int n_thr_t = n_thr_per_ord_test[t_ord];
      //                             int n_cat_t = n_cat_per_ord_test[t_ord];
      //                             
      //                             vector[n_thr_t] C_t = get_test_values(C_vec[c], ord_start_index, ord_end_index, t_ord);
      //                             
      //                             real mu_plus_inc = Xbeta_nt + inc;
      //                             real L_tt_recip = L_Omega_diag_recip[c, t];
      //                             
      //                             // Compute standardized upper and lower bounds:
      //                             real Phi_lower;
      //                             real Phi_upper;
      //                             
      //                             if (k_obs == 1) {
      //                                 // First category: lower = -inf
      //                                 Phi_lower = 0.0;
      //                                 real upper_z = (C_t[1] - mu_plus_inc) * L_tt_recip;
      //                                 Phi_upper = Phi(upper_z);
      //                             } else if (k_obs == n_cat_t) {
      //                                 // Last category: upper = +inf
      //                                 real lower_z = (C_t[n_thr_t] - mu_plus_inc) * L_tt_recip;
      //                                 Phi_lower = Phi(lower_z);
      //                                 Phi_upper = 1.0;
      //                             } else {
      //                                 // Middle category: both bounds finite
      //                                 real lower_z = (C_t[k_obs - 1] - mu_plus_inc) * L_tt_recip;
      //                                 real upper_z = (C_t[k_obs] - mu_plus_inc) * L_tt_recip;
      //                                 Phi_lower = Phi(lower_z);
      //                                 Phi_upper = Phi(upper_z);
      //                             }
      //                             
      //                             // GHK contribution:
      //                             real prob_t = Phi_upper - Phi_lower;
      //                             real Phi_Z = Phi_lower + prob_t * u[t];
      //                             Z_std_norm[t] = inv_Phi(Phi_Z);
      //                             y1[t] = log(prob_t);
      //                         
      //                     }
      //                     
      //                     if (t < n_tests) inc = L_Omega[c, t + 1, 1:t] * head(Z_std_norm, t);
      //                     
      //                 } // end t loop
      //                 
      //                 lp[c] = sum(y1) + log_prev[c];
      //                 
      //             } // end c loop
      //             
      //             log_lik[n] = log_sum_exp(lp);
      //             
      //         } // end n loop
      //     }
      // }
}




