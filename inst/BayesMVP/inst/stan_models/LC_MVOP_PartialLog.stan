// LC_MVOP_PartialLog.stan
// 
// Latent Class Multivariate Ordinal Probit (LC-MVOP)
// Extends binary LC-MVP to handle mixed binary + ordinal outcomes.
// First n_binary_tests columns of y are binary (0/1).
// Next n_ordinal_tests columns are ordinal (1, 2, ..., K_t).
//
// Binary tests use the existing GHK algorithm (one-sided truncation).
// Ordinal tests use two-sided truncation in the GHK.
//

functions {

      //////////////////////////////////////////////////////////////////////////
      // ---- Existing binary model functions:
      //////////////////////////////////////////////////////////////////////////
      
      vector lb_ub_lp (vector y, real lb, real ub) {
            int N = num_elements(y); 
            vector[N] tanh_y = tanh(y);
            target +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh_y))) ;
            return lb +  (ub - lb) *  0.5 * (1 + tanh_y) ;   
      }  
 
      real lb_ub_lp (real y, real lb, real ub) {
            real tanh_y = tanh(y);
            target +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh_y))) ;
            return lb +  (ub - lb) *  0.5 * (1 + tanh_y) ; 
      } 
      
      // Non-Jacobian version (for Pinkney bounds):
      real lb_ub(real y, real lb, real ub) {
            return lb + (ub - lb) * 0.5 * (1.0 + tanh(y));
      }
      vector lb_ub(vector y, real lb, real ub) {
            return lb + (ub - lb) * 0.5 * (1.0 + tanh(y));
      }
  
      matrix cholesky_corr_constrain_outer_lp(
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
                real b1 = dot_product(L[j, 1:(j - 1)], D[1:j - 1]' .* L[i, 1:(j - 1)]);
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
                  cnt += 1;
                }
                D[i] = l_ij_old;
              }
              if (is_nan(sum(D)) || is_inf(sum(D)) || min(D) <= 0)
                  reject("nonpositive or nonfinite LDL pivot");
              return diag_post_multiply(L, sqrt(D));
      }

      real inv_Phi_approx_from_prob(real p) { 
            return 5.494448514153059 *  sinh(0.33333333333333331483 * asinh( 0.34176618822627863 * logit(p)  )) ;
      }
      vector inv_Phi_approx_from_prob(vector p) { 
            return 5.494448514153059 *  sinh(0.33333333333333331483 * asinh( 0.34176618822627863 * logit(p)  )) ;  
      }
      real inv_Phi_approx_from_logit_prob(real logit_p) { 
            return 5.494448514153059 *  sinh(0.33333333333333331483 * asinh( 0.34176618822627863 * logit_p  )) ; 
      }
      vector inv_Phi_approx_from_logit_prob(vector logit_p) { 
            return 5.494448514153059 *  sinh(0.33333333333333331483 * asinh( 0.34176618822627863 *logit_p  )) ; 
      }
 
      vector rowwise_sum(matrix M) {
            return M * rep_vector(1.0, cols(M));
      }
      vector rowwise_max(matrix M) {
            int N = rows(M);
            vector[N] rowwise_maxes;
            for (n in 1:N) rowwise_maxes[n] = max(M[n, ]);
            return rowwise_maxes;
      }
      vector log_sum_exp_2d(matrix array_2d_to_lse) { 
            int N = rows(array_2d_to_lse);
            matrix[N, 2] rowwise_maxes_2d_array;
            rowwise_maxes_2d_array[, 1] = rowwise_max(array_2d_to_lse);
            rowwise_maxes_2d_array[, 2] = rowwise_maxes_2d_array[, 1];
            return rowwise_maxes_2d_array[, 1] + log(rowwise_sum(exp((array_2d_to_lse - rowwise_maxes_2d_array))));
      }

      //////////////////////////////////////////////////////////////////////////
      // ---- Ordinal helper functions (from skeleton/MetaOrdDTA):
      //////////////////////////////////////////////////////////////////////////
      
      vector construct_C(vector C_raw_vec, int softplus) {
            int n_total_cutpoints = num_elements(C_raw_vec);
            vector[n_total_cutpoints] C_vec;  
            C_vec[1] = C_raw_vec[1];
            if (softplus == 1) {
                vector[n_total_cutpoints - 1] sp = log1p_exp(C_raw_vec[2:n_total_cutpoints]);
                for (k in 2:n_total_cutpoints) C_vec[k] = C_vec[k - 1] + sp[k - 1];
            } else { 
                vector[n_total_cutpoints - 1] exp_C = exp(C_raw_vec[2:n_total_cutpoints]);
                for (k in 2:n_total_cutpoints) C_vec[k] = C_vec[k - 1] + exp_C[k - 1];
            }
            return C_vec;
      }
      
      array[] int calculate_start_indices(array[] int n_thr, int n_tests) {
            array[n_tests] int start_index;
            start_index[1] = 1;
            for (t in 2:n_tests) start_index[t] = start_index[t - 1] + n_thr[t - 1];
            return start_index;
      }
      
      array[] int calculate_end_indices(array[] int n_thr, int n_tests, array[] int start_index) {
            array[n_tests] int end_index;
            for (t in 1:n_tests) end_index[t] = start_index[t] + n_thr[t] - 1;
            return end_index;
      }
      
      vector get_test_values(vector flat_values, data array[] int start_index, data array[] int end_index, data int test_index) {
            int n_elements = end_index[test_index] - start_index[test_index] + 1;
            vector[n_elements] result;
            for (i in 1:n_elements) result[i] = flat_values[start_index[test_index] + i - 1];
            return result;
      }
      
      vector update_test_values(vector flat_values_to_update, vector new_values, data array[] int start_index, data array[] int end_index, data int test_index) {
            int n_elements = end_index[test_index] - start_index[test_index] + 1;
            vector[num_elements(flat_values_to_update)] result = flat_values_to_update;
            for (i in 1:n_elements) result[start_index[test_index] + i - 1] = new_values[i];
            return result;
      }
      
      // Convert cumulative probs to ordinal (category) probs:
      vector cumul_probs_to_ord_probs(vector cumul_probs) {
            int K_minus_1 = num_elements(cumul_probs);
            int K = K_minus_1 + 1;
            vector[K] ord_probs;
            ord_probs[1] = cumul_probs[1];
            for (k in 2:K_minus_1) ord_probs[k] = cumul_probs[k] - cumul_probs[k - 1];
            ord_probs[K] = 1.0 - cumul_probs[K_minus_1];
            return ord_probs;
      }
      
      // Induced Dirichlet log-density given cutpoints C:
      real induced_dirichlet_lpdf(vector ord_probs, vector C_vec, vector alpha, int use_probit) {
            int K = num_elements(ord_probs);
            int n_thr = K - 1;
            real lp = 0.0;
            // Dirichlet on category probs:
            lp += dirichlet_lpdf(ord_probs | alpha);
            // Jacobian: d(cumul_prob)/d(C) = pdf(C)
            for (k in 1:n_thr) {
                if (use_probit == 1) {
                    lp += std_normal_lpdf(C_vec[k]);
                } else {
                    // logistic density
                    lp += log_inv_logit(C_vec[k]) + log1m_inv_logit(C_vec[k]);
                }
            }
            return lp;
      }
}


data {
      int<lower=1> N;
      int<lower=2> n_tests;
      ////
      //// ---- Ordinal data fields:
      ////
      int<lower=0> n_binary_tests;
      int<lower=0> n_ordinal_tests;
      array[n_ordinal_tests] int<lower=2> n_cat_per_ord_test;
      array[n_ordinal_tests] int<lower=1> n_thr_per_ord_test;
      matrix<lower=0>[n_ordinal_tests > 0 ? max(n_cat_per_ord_test) : 1, max(n_ordinal_tests, 1)] prior_dirichlet_alpha;
      ////
      matrix[N, n_tests] y;
      int<lower=2> n_class;
      int<lower=1> n_pops;
      array[N] int pop;
      ////
      int n_covariates_max_nd;
      int n_covariates_max_d;
      int n_covariates_max;
      array[n_tests] matrix[N, n_covariates_max_nd] X_nd;
      array[n_tests] matrix[N, n_covariates_max_d]  X_d;
      array[n_class, n_tests] int n_covs_per_outcome;
      ////
      int corr_force_positive;
      int<lower=0, upper=(n_tests * (n_tests - 1)) %/% 2> known_num;
      real overflow_threshold;
      real underflow_threshold;
      ////
      int prior_only;
      array[n_class] matrix[n_covariates_max, n_tests] prior_beta_mean;
      array[n_class] matrix<lower=0>[n_covariates_max, n_tests] prior_beta_sd;
      matrix<lower=0>[n_class, 1] prior_LKJ;
      matrix<lower=0>[n_pops, 1] prior_p_alpha;
      matrix<lower=0>[n_pops, 1] prior_p_beta;
      ////
      int Phi_type;
      int handle_numerical_issues;
      int fully_vectorised;
      ////
      //// ---- Baseline cases for GQ:
      ////
      array[n_tests] vector[n_covariates_max] baseline_case_nd;
      array[n_tests] vector[n_covariates_max] baseline_case_d;
}


transformed data {
      int k_choose_2 = (n_tests * (n_tests - 1)) / 2;
      int km1_choose_2 = ((n_tests - 1) * (n_tests - 2)) / 2;
      ////
      int n_covariates_total_nd = sum(n_covs_per_outcome[1,]);
      int n_covariates_total_d  = sum(n_covs_per_outcome[2,]);
      int n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
      ////
      real s = 1 / 1.702;
      real a = 0.07056;
      real b_const = 1.5976;
      real a_times_3 = 3.0 * 0.07056;
      real<lower=-1, upper=1> lb;
      real<lower=lb, upper=1> ub = 1.0;
      if (corr_force_positive == 1) lb = 0;
      else lb = -1.0;
      ////
      //// ---- Ordinal indices:
      ////
      int n_total_C_per_class = (n_ordinal_tests > 0) ? sum(n_thr_per_ord_test) : 0;
      array[max(n_ordinal_tests, 1)] int ord_start_index;
      array[max(n_ordinal_tests, 1)] int ord_end_index;
      if (n_ordinal_tests > 0) {
          ord_start_index = calculate_start_indices(n_thr_per_ord_test, n_ordinal_tests);
          ord_end_index   = calculate_end_indices(n_thr_per_ord_test, n_ordinal_tests, ord_start_index);
      }
}


parameters {
      matrix[N, n_tests] u_raw;
      array[n_class] vector[n_tests - 1] col_one_raw;
      array[n_class] vector[km1_choose_2 - known_num] off_raw;
      vector[n_covariates_total] beta_vec;
      vector[n_pops] p_raw;
      ////
      //// ---- Ordinal cutpoints (raw, unconstrained):
      ////
      array[n_ordinal_tests > 0 ? 2 : 0] vector[n_total_C_per_class] C_raw_vec;
}


transformed parameters {
      array[n_class, n_tests, n_covariates_max] real beta;
      vector<lower=0, upper=1>[n_pops] prev = lb_ub_lp(p_raw, 0.0, 1.0);
      array[n_class] matrix[n_tests, n_tests] Omega;
      array[n_class] matrix[n_tests, n_tests] L_Omega;
      matrix[n_class, n_tests] L_Omega_diag_recip;
      vector[N] log_lik = rep_vector(0.0, N);
      
      {
            int counter = 1;
            for (c in 1:n_class) {
                for (t in 1:n_tests) {
                    for (k in 1:n_covs_per_outcome[c, t]) {
                        beta[c, t, k] = beta_vec[counter];
                        counter += 1;
                    }
                }
                L_Omega[c] = cholesky_corr_constrain_outer_lp(to_vector(col_one_raw[c]), to_vector(off_raw[c]), lb, ub);
                Omega[c] = multiply_lower_tri_self_transpose(L_Omega[c]);
                L_Omega_diag_recip[c, ] = to_row_vector(1.0 ./ diagonal(L_Omega[c]));
            }
      }
      
      ////
      //// ---- Construct ordinal cutpoints:
      ////
      array[n_ordinal_tests > 0 ? 2 : 0] vector[n_total_C_per_class] C_vec;
      if (n_ordinal_tests > 0) {
          for (t_ord in 1:n_ordinal_tests) {
              for (c in 1:2) {
                    int n_thr_t = n_thr_per_ord_test[t_ord];
                    vector[n_thr_t] C_raw_t = get_test_values(C_raw_vec[c], ord_start_index, ord_end_index, t_ord);
                    int softplus = 0;
                    vector[n_thr_t] C_t = construct_C(C_raw_t, softplus);
                    C_vec[c] = update_test_values(C_vec[c], C_t, ord_start_index, ord_end_index, t_ord);
              }
          }
      }
      
      ////
      //// ---- Likelihood:
      ////
      if (prior_only == 0) {
          
          matrix[N, n_class] log_prev;
          for (n in 1:N) {
              log_prev[n, 1] = log1m(prev[pop[n]]);
              log_prev[n, 2] = log(prev[pop[n]]);
          }
          
          // ---- Non-vectorised loop version (works for both binary + ordinal):
          {
              for (n in 1:N) {
                  
                  vector[n_tests] u = lb_ub_lp(to_vector(u_raw[n,]), 0.0, 1.0);
                  vector[n_class] lp;
                  
                  for (c in 1:n_class) {
                      
                      vector[n_tests] Z_std_norm;
                      vector[n_tests] y1;
                      real inc = 0.0;
                      
                      for (t in 1:n_tests) {
                          
                          real Xbeta_nt;
                          if (n_covariates_max > 1) {
                              if (c == 1) Xbeta_nt = to_row_vector(X_nd[t][n, 1:n_covs_per_outcome[c,t]]) * to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);
                              else        Xbeta_nt = to_row_vector(X_d[t][n, 1:n_covs_per_outcome[c,t]])  * to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);
                          } else {
                              Xbeta_nt = beta[c, t, 1];
                          }
                          
                          if (t <= n_binary_tests) {
                            
                                  ////
                                  //// ---- BINARY test:
                                  ////
                                  real Bound_Z = -(Xbeta_nt + inc) * L_Omega_diag_recip[c, t];
                                  
                                  real Bound_U_Phi_BZ = Phi(Bound_Z);
                                  if (y[n, t] == 1) {
                                      real Phi_Z = Bound_U_Phi_BZ + (1.0 - Bound_U_Phi_BZ) * u[t];
                                      Z_std_norm[t] = inv_Phi(Phi_Z);
                                      y1[t] = log1m(Bound_U_Phi_BZ);
                                  } else {
                                      real Phi_Z = Bound_U_Phi_BZ * u[t];
                                      Z_std_norm[t] = inv_Phi(Phi_Z);
                                      y1[t] = log(Bound_U_Phi_BZ);
                                  }
                              
                          } else {
                            
                                  ////
                                  //// ---- ORDINAL test:
                                  ////
                                  int t_ord = t - n_binary_tests;
                                  int k_obs = to_int(y[n, t]);  // observed category (1-indexed)
                                  int n_thr_t = n_thr_per_ord_test[t_ord];
                                  int n_cat_t = n_cat_per_ord_test[t_ord];
                                  
                                  vector[n_thr_t] C_t = get_test_values(C_vec[c], ord_start_index, ord_end_index, t_ord);
                                  
                                  real mu_plus_inc = Xbeta_nt + inc;
                                  real L_tt_recip = L_Omega_diag_recip[c, t];
                                  
                                  // Compute standardized upper and lower bounds:
                                  real Phi_lower;
                                  real Phi_upper;
                                  
                                  if (k_obs == 1) {
                                      // First category: lower = -inf
                                      Phi_lower = 0.0;
                                      real upper_z = (C_t[1] - mu_plus_inc) * L_tt_recip;
                                      Phi_upper = Phi(upper_z);
                                  } else if (k_obs == n_cat_t) {
                                      // Last category: upper = +inf
                                      real lower_z = (C_t[n_thr_t] - mu_plus_inc) * L_tt_recip;
                                      Phi_lower = Phi(lower_z);
                                      Phi_upper = 1.0;
                                  } else {
                                      // Middle category: both bounds finite
                                      real lower_z = (C_t[k_obs - 1] - mu_plus_inc) * L_tt_recip;
                                      real upper_z = (C_t[k_obs] - mu_plus_inc) * L_tt_recip;
                                      Phi_lower = Phi(lower_z);
                                      Phi_upper = Phi(upper_z);
                                  }
                                  
                                  // GHK contribution:
                                  real prob_t = Phi_upper - Phi_lower;
                                  real Phi_Z = Phi_lower + prob_t * u[t];
                                  Z_std_norm[t] = inv_Phi(Phi_Z);
                                  y1[t] = log(prob_t);
                              
                          }
                          
                          if (t < n_tests) inc = L_Omega[c, t + 1, 1:t] * head(Z_std_norm, t);
                          
                      } // end t loop
                      
                      lp[c] = sum(y1) + log_prev[n, c];
                      
                  } // end c loop
                  
                  log_lik[n] = log_sum_exp(lp);
                  
              } // end n loop
          }
      }
}


model {
      ////
      //// ---- Priors on beta:
      ////
      for (c in 1:n_class) {
          for (t in 1:n_tests) {
              for (k in 1:n_covs_per_outcome[c, t]) {
                  beta[c, t, k] ~ normal(prior_beta_mean[c][k, t], prior_beta_sd[c][k, t]);
              }
          }
          target += lkj_corr_cholesky_lpdf(L_Omega[c] | prior_LKJ[c, 1]);
      }
      ////
      //// ---- Prevalence prior:
      ////
      for (g in 1:n_pops) {
          prev[g] ~ beta(prior_p_alpha[g, 1], prior_p_beta[g, 1]);
      }
      ////
      //// ---- Induced-Dirichlet prior on cutpoints + Jacobian for C_raw -> C:
      ////
      if (n_ordinal_tests > 0) {
          for (c in 1:2) {
              for (t_ord in 1:n_ordinal_tests) {
                    int n_thr_t = n_thr_per_ord_test[t_ord];
                    int n_cat_t = n_cat_per_ord_test[t_ord];
                    vector[n_thr_t] C_t = get_test_values(C_vec[c], ord_start_index, ord_end_index, t_ord);
                    vector[n_thr_t] C_raw_t = get_test_values(C_raw_vec[c], ord_start_index, ord_end_index, t_ord);
                    ////
                    //// Induced-Dirichlet prior:
                    vector[n_thr_t] cumul_prob = Phi(C_t);
                    vector[n_cat_t] ord_prob = cumul_probs_to_ord_probs(cumul_prob);
                    target += induced_dirichlet_lpdf(ord_prob | C_t, to_vector(prior_dirichlet_alpha[1:n_cat_t, t_ord]), 1);
                    ////
                    //// Jacobian for C_raw -> C (log-difference parameterisation):
                    //// C[k] = C[k-1] + exp(C_raw[k]) for k >= 2
                    //// log|det J| = sum(C_raw[2:K])
                    if (n_thr_t > 1) {
                        target += sum(C_raw_t[2:n_thr_t]);
                    }
              }
          }
      }
      ////
      //// ---- Likelihood:
      ////
      if (prior_only == 0) {
          for (n in 1:N) target += log_lik[n];
      }
}


generated quantities {
      
      vector<lower=0, upper=1>[n_pops] p = prev;
      ////
      //// ---- Baseline Xbeta for all tests:
      ////
      array[n_tests] real Xbeta_baseline_nd;
      array[n_tests] real Xbeta_baseline_d;
      ////
      for (t in 1:n_tests) {
          Xbeta_baseline_nd[t] = dot_product(baseline_case_nd[t][1:n_covs_per_outcome[1, t]],
                                              to_vector(beta[1, t, 1:n_covs_per_outcome[1, t]]));
          Xbeta_baseline_d[t]  = dot_product(baseline_case_d[t][1:n_covs_per_outcome[2, t]],
                                              to_vector(beta[2, t, 1:n_covs_per_outcome[2, t]]));
      }
      ////
      //// ---- Binary baseline Se/Sp:
      ////
      vector[n_binary_tests] Se_baseline_bin;
      vector[n_binary_tests] Sp_baseline_bin;
      vector[n_binary_tests] Fp_baseline_bin;
      ////
      if (n_class == 2) {
          for (t in 1:n_binary_tests) {
              Se_baseline_bin[t] =       Phi(Xbeta_baseline_d[t]);
              Fp_baseline_bin[t] =       Phi(Xbeta_baseline_nd[t]);
              Sp_baseline_bin[t] = 1.0 - Fp_baseline_bin[t];
          }
      }
      ////
      //// ---- Ordinal baseline Se/Sp at each threshold:
      ////
      int n_thr_max_gq = (n_ordinal_tests > 0) ? max(n_thr_per_ord_test) : 1;
      ////
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Se_baseline_ord;
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Sp_baseline_ord;
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Fp_baseline_ord;
      ////
      if (n_ordinal_tests > 0 && n_class == 2) {
          for (t_ord in 1:n_ordinal_tests) {
              int t = n_binary_tests + t_ord;
              int n_thr_t = n_thr_per_ord_test[t_ord];
              ////
              Se_baseline_ord[t_ord] = rep_vector(-1.0, n_thr_max_gq);
              Sp_baseline_ord[t_ord] = rep_vector(-1.0, n_thr_max_gq);
              Fp_baseline_ord[t_ord] = rep_vector(-1.0, n_thr_max_gq);
              ////
              vector[n_thr_t] C_nd_t = get_test_values(C_vec[1], ord_start_index, ord_end_index, t_ord);
              vector[n_thr_t] C_d_t  = get_test_values(C_vec[2], ord_start_index, ord_end_index, t_ord);
              ////
              for (k in 1:n_thr_t) {
                  Fp_baseline_ord[t_ord][k] =       Phi(-(C_nd_t[k] - Xbeta_baseline_nd[t]));
                  Sp_baseline_ord[t_ord][k] = 1.0 - Fp_baseline_ord[t_ord][k];
                  Se_baseline_ord[t_ord][k] =       Phi(-(C_d_t[k]  - Xbeta_baseline_d[t]));
              }
          }
      }
      ////
      //// ---- Cutpoint arrays (for extraction):
      ////
      array[n_ordinal_tests > 0 ? 2 : 0] vector[n_total_C_per_class] C_vec_out;
      if (n_ordinal_tests > 0) {
          C_vec_out = C_vec;
      }
}
