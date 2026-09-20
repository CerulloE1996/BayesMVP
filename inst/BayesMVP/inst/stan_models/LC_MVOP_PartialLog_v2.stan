// LC_MVOP_PartialLog_v2.stan
//
// Latent Class Multivariate Ordinal Probit (LC-MVOP)
// Extends binary LC-MVP to handle mixed binary + ordinal outcomes.
// First n_binary_tests columns of y are binary (0/1).
// Next n_ordinal_tests columns are ordinal (1, 2, ..., K_t).
//
// This version implements ONLY the fully_vectorised == 1 / handle_numerical_issues == 1
// path (i.e. matches the scheme of LC_MVP_bin_PartialLog_v5.stan). The data flags
// "fully_vectorised" and "handle_numerical_issues" are accepted but ignored.
//
// Scheme (per class c, per test t, vectorised over all N subjects):
//   - BINARY tests: identical to the fully-vectorised partial-log branch of the
//     binary model. Subjects split into 3 index groups per (c, t):
//       (i)   OK                          -> standard scale (Phi_type switch)
//       (ii)  Bound_Z < UF_thresh & y==0  -> log scale (approx-logit forms)
//       (iii) Bound_Z > OF_thresh & y==1  -> log scale (approx-logit forms)
//   - ORDINAL tests: two-sided truncation. Per-subject lower/upper cutpoints are
//     gathered (finite +/- C_sentinel pseudo-infinities for the unbounded
//     bottom/top categories). Since Bound_Z_lo <= Bound_Z_hi always, subjects
//     split into 3 index groups per (c, t):
//       (i)   OK                          -> standard scale (Phi_type switch)
//       (ii)  Bound_Z_hi < UF_thresh      -> BOTH bounds in left tail  -> log scale
//             (covers bottom-category underflow AND middle-category
//              both-cutpoints-in-left-tail cancellation)
//       (iii) Bound_Z_lo > OF_thresh      -> BOTH bounds in right tail -> log scale
//             (covers top-category overflow AND middle-category
//              both-cutpoints-in-right-tail cancellation)
//     For K = 2 the ordinal log-scale formulas reduce EXACTLY to the binary
//     underflow/overflow formulas.
//     Mixed cases (one bound in each tail, or one extreme + one moderate) are
//     numerically fine on the standard scale and stay in group (i).
//

functions {

      //////////////////////////////////////////////////////////////////////////
      // ---- Existing binary model functions:
      //////////////////////////////////////////////////////////////////////////
      vector lb_ub_jacobian( vector y,
                             real lb, 
                             real ub) {
                               
            int N = num_elements(y); 
            vector[N] tanh_y; 
           // tanh_y = tanh_1(y);
            tanh_y = tanh(y);
            // jacobian +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh_y))) ;
            jacobian += log(ub - lb) - log2() + 2.0 * (log2() - y - log1p_exp(-2.0 * y));
            return lb + (ub - lb) * 0.5 * (1.0 + tanh(y));
            
      }
      real lb_ub_jacobian( real y, 
                           real lb, 
                           real ub) {
        
            jacobian += log(ub - lb) - log2() + 2.0 * (log2() - y - log1p_exp(-2.0 * y));
            // jacobian +=  - log(2)  +  log( (ub - lb) * (1 - square(tanh(y)))) ;
            return lb + (ub - lb) * 0.5 * (1.0 + tanh(y));
            
      }
      vector lb_ub_jacobian( vector y,
                             vector lb,
                             vector ub) {
            int N = num_elements(y);
            vector[N] tanh_y;
           // tanh_y = tanh_1(y);
            tanh_y = tanh(y);
            jacobian += sum(log(ub - lb)) - num_elements(y) * log2() + 2.0 * sum(log2() - y - log1p_exp(-2.0 * y));
            // jacobian +=  - log(2)  +  log( (ub - lb) .* (1 - square(tanh_y))) ;
            return lb + (ub - lb) .* (0.5 * (1.0 + tanh(y)));

      }
      ////
      // Non-Jacobian version (for Pinkney bounds):
      real lb_ub(real y, real lb, real ub) {
            return lb + (ub - lb) * 0.5 * (1.0 + tanh(y));
      }
      vector lb_ub(vector y, real lb, real ub) {
            return lb + (ub - lb) * 0.5 * (1.0 + tanh(y));
      }

      ////
      //// This Stan function was provided by Sean Pinkney (may be slightly adapted/different formatting):
      ////
      matrix Pinkney_LDL_bounds_opt_jacobian(  vector Omega_raw_vec,
                                               matrix lb_corr,
                                               matrix ub_corr,
                                               matrix known_values_indicator,
                                               matrix known_values) {

            ////
            //// First construct "Omega_raw_mat" matrix from "Omega_raw_vec" using row-by-row access (as MUST match the BayesMVP C++ functions/models!):
            ////
            int dim = rows(lb_corr);
            matrix[dim, dim] Omega_raw_mat = rep_matrix(0.0, dim, dim);
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
                    L[i, 1] = lb_ub_jacobian(Omega_raw_mat[i, 1], lb_corr[i, 1], ub_corr[i, 1]);
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

                                    real x = lb_ub_jacobian(Omega_raw_mat[i, j], low, up);
                                    L[i, j] = x / D[j];

                                    jacobian += -0.5 * log(D[j]);

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

      real inv_Phi_approx_from_prob(real p) {
            return 5.494 *  sinh(0.33333333333333331483 * asinh( 0.3418 * logit(p)  )) ;
      }
      vector inv_Phi_approx_from_prob(vector p) {
            return 5.494 *  sinh(0.33333333333333331483 * asinh( 0.3418 * logit(p)  )) ;
      }
      real inv_Phi_approx_from_logit_prob(real logit_p) {
            return 5.494 *  sinh(0.33333333333333331483 * asinh( 0.3418 * logit_p  )) ;
      }
      vector inv_Phi_approx_from_logit_prob(vector logit_p) {
            return 5.494 *  sinh(0.33333333333333331483 * asinh( 0.3418 *logit_p  )) ;
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
      vector cumul_probs_to_ord_probs( vector cumul_probs) {
        
                int K_minus_1 = num_elements(cumul_probs);
                int K = K_minus_1 + 1;
                vector[K] ord_probs;
                ord_probs[1] = cumul_probs[1];
                for (k in 2:K_minus_1) ord_probs[k] = cumul_probs[k] - cumul_probs[k - 1];
                ord_probs[K] = 1.0 - cumul_probs[K_minus_1];
                return ord_probs;
            
      }
      
      // Induced Dirichlet log-density given cutpoints C:
      real induced_dirichlet_lpdf( vector ord_probs, 
                                   vector C_vec, 
                                   vector alpha, 
                                   int use_probit,
                                   real anchor) {
                                     
                int K = num_elements(ord_probs);
                int n_thr = K - 1;
                real lp = 0.0;
                // Dirichlet on category probs:
                lp += dirichlet_lpdf(ord_probs | alpha);
                // Jacobian: d(cumul_prob)/d(C) = pdf(C)
                for (k in 1:n_thr) {
                    if (use_probit == 1) {
                        lp += std_normal_lpdf(C_vec[k] - anchor);
                    } else {
                        // logistic density
                        lp += log_inv_logit(C_vec[k] - anchor) + log1m_inv_logit(C_vec[k] - anchor);
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
      array[2] matrix<lower=0>[n_ordinal_tests > 0 ? max(n_cat_per_ord_test) : 1, max(n_ordinal_tests, 1)] prior_dirichlet_alpha;
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
      ////
      array[n_class] matrix[n_tests, n_tests] known_values_list;
      array[n_class] matrix[n_tests, n_tests] known_values_indicator_list;
      ////
      array[n_class] matrix[n_tests, n_tests] lb_corr;
      array[n_class] matrix[n_tests, n_tests] ub_corr;
      int<lower=0, upper=(n_tests * (n_tests - 1)) %/% 2> known_num;
      ////
      real overflow_threshold;
      real underflow_threshold;
      ////
      real C_raw_lower; //// ordinal-only
      real C_raw_upper; //// ordinal-only
      ////
      int prior_only;
      array[n_class] matrix[n_covariates_max, n_tests] prior_beta_mean;
      array[n_class] matrix<lower=0>[n_covariates_max, n_tests] prior_beta_sd;
      matrix<lower=0>[n_class, 1] prior_LKJ;
      matrix<lower=0>[n_pops, 1] prior_prev_a;
      matrix<lower=0>[n_pops, 1] prior_prev_b;
      ////
      int Phi_type;
      int handle_numerical_issues;  //// accepted but IGNORED (this file hard-codes the handle_numerical_issues == 1 path)
      int fully_vectorised;         //// accepted but IGNORED (this file hard-codes the fully_vectorised == 1 path)
      ////
      //// ---- Baseline cases for GQ:
      ////
      array[n_tests] vector[n_covariates_max] baseline_case_nd;
      array[n_tests] vector[n_covariates_max] baseline_case_d;
}


transformed data {
      ////
      int n_covariates_total_nd = sum(n_covs_per_outcome[1,]);
      int n_covariates_total_d  = sum(n_covs_per_outcome[2,]);
      int n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
      ////
      real s = 1 / 1.702;
      real a = 0.07056;
      real b_const = 1.5976;
      real a_times_3 = 3.0 * 0.07056;
      ////
      //// ---- Ordinal indices:
      ////
      int n_thr_max_gq = (n_ordinal_tests > 0) ? max(n_thr_per_ord_test) : 1;
      ////
      int n_total_C_per_class = (n_ordinal_tests > 0) ? sum(n_thr_per_ord_test) : 0;
      array[max(n_ordinal_tests, 1)] int ord_start_index;
      array[max(n_ordinal_tests, 1)] int ord_end_index;
      if (n_ordinal_tests > 0) {
          ord_start_index = calculate_start_indices(n_thr_per_ord_test, n_ordinal_tests);
          ord_end_index   = calculate_end_indices(n_thr_per_ord_test, n_ordinal_tests, ord_start_index);
      }
      ////
      //// ---- Ordinal category data as int (per ordinal test, per person):
      ////
      array[max(n_ordinal_tests, 1), N] int y_ord;
      for (n in 1:N) y_ord[1, n] = 1; //// dummy fill (overwritten below if n_ordinal_tests > 0)
      if (n_ordinal_tests > 0) {
          for (t_ord in 1:n_ordinal_tests) {
              for (n in 1:N) {
                  y_ord[t_ord, n] = to_int(y[n, n_binary_tests + t_ord]);
              }
          }
      }
      ////
      //// ---- Finite "pseudo-infinity" cutpoint sentinel for the unbounded bottom/top categories.
      //// Phi() saturates to EXACTLY 0/1 (with exactly-zero gradient) well before |x| = 38, and
      //// L_tt <= 1 (corr Cholesky diag) so |C_sentinel / L_tt| >= C_sentinel. Must be >> any
      //// admissible cutpoint magnitude (probit cutpoints live in ~(-6, 6)).
      //// NOTE: do NOT use +/-inf here: inf inside var arithmetic produces 0 * inf = NaN
      //// adjoints in reverse-mode autodiff (e.g. adjoint 0 from a saturated Phi() times the
      //// stored -inf partial of the subtraction/multiply nodes).
      ////
      real C_sentinel = 1000.0;
      ////
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

}


parameters {
      matrix[N, n_tests] u_raw;
      array[n_class] vector[n_corrs_per_class] Omega_unconstrained_vec;
      vector[n_covariates_total] beta_vec;
      vector[n_pops] p_raw;
      ////
      //// ---- Ordinal cutpoints (raw, unconstrained):
      ////
      array[n_ordinal_tests > 0 ? 2 : 0] vector[n_total_C_per_class] C_unc_vec;
}


transformed parameters {
      array[n_class, n_tests, n_covariates_max] real beta;
      vector<lower=0, upper=1>[n_pops] prev = lb_ub_jacobian(p_raw, 0.0, 1.0);
      array[n_class] matrix[n_tests, n_tests] Omega;
      array[n_class] matrix[n_tests, n_tests] L_Omega;
      matrix[n_class, n_tests] L_Omega_diag_recip;
      vector[N] log_lik = rep_vector(0.0, N);
      array[n_ordinal_tests > 0 ? 2 : 0] vector<lower=C_raw_lower, upper=C_raw_upper>[n_total_C_per_class] C_raw_vec;
      ////
      // array[n_ordinal_tests > 0 ? 2 : 0] vector[n_total_C_per_class] C_raw_vec;
      for (c in 1:n_class) {
        C_raw_vec[c] = lb_ub_jacobian(C_unc_vec[c], C_raw_lower, C_raw_upper); 
      }
      ////
      for (c in 1:n_class) {
                L_Omega[c] = Pinkney_LDL_bounds_opt_jacobian(  Omega_unconstrained_vec[c],
                                                               lb_corr_actual[c],
                                                               ub_corr[c],
                                                               known_values_indicator_list[c],
                                                               known_values_list[c]);
                Omega[c] = multiply_lower_tri_self_transpose(L_Omega[c, :]);
      }
      ////
      {
            int counter = 1;
            for (c in 1:n_class) {
                for (t in 1:n_tests) {
                    for (k in 1:n_covs_per_outcome[c, t]) {
                        beta[c, t, k] = beta_vec[counter];
                        counter += 1;
                    }
                }
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
                    // vector[n_thr_t] C_raw_t = lb_ub_jacobian(C_unc_t, C_raw_lower, C_raw_upper);
                    int softplus = 0;
                    vector[n_thr_t] C_t = construct_C(C_raw_t, softplus);
                    C_vec[c] = update_test_values(C_vec[c], C_t, ord_start_index, ord_end_index, t_ord);
              }
          }
      }
      ////
      //// ---- Likelihood (fully vectorised over N, partial-log numerical handling):
      ////
      if (prior_only == 0) {

            matrix[N, n_class] log_prev;
            for (n in 1:N) {
                log_prev[n, 1] = log1m(prev[pop[n]]);
                log_prev[n, 2] = log(prev[pop[n]]);
            }

            { //// log-lik block
                  matrix[N, n_tests] Z_std_norm;
                  matrix[N, n_tests] u;
                  matrix[N, n_tests] y1;
                  matrix[N, n_class] lp;
                  vector[N] inc;

                  for (t in 1:n_tests) {
                      u[, t] = lb_ub_jacobian(u_raw[, t], 0.0, 1.0);
                  }

                  for (c in 1:n_class) {

                        inc = rep_vector(0.0, N);

                        for (t in 1:n_tests) {

                              ////
                              //// ---- Linear predictor for test t (vector over subjects).
                              //// NOTE: indexed consistently with the X_nd / X_d declarations
                              //// (i.e. X[t] is [N, n_covariates_max], rows = subjects):
                              ////
                              vector[N] Xbeta;
                              if (n_covariates_max > 1) {
                                    if (c == 1)  Xbeta = X_nd[t][, 1:n_covs_per_outcome[c, t]] * to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);
                                    else         Xbeta = X_d[t][,  1:n_covs_per_outcome[c, t]] * to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);
                              } else {
                                    Xbeta = rep_vector(beta[c, t, 1], N);
                              }

                              if (t <= n_binary_tests) {

                                    ////////////////////////////////////////////////////////////
                                    //// ---- BINARY test t (one-sided truncation) -- identical
                                    //// to the fully-vectorised partial-log branch of the
                                    //// binary LC-MVP model:
                                    ////////////////////////////////////////////////////////////
                                    vector[N] Bound_Z = - (Xbeta + inc) * L_Omega_diag_recip[c, t];

                                    int num_OK_index = 0;
                                    int num_Bound_Z_overflows_and_y_eq_1 = 0;
                                    int num_Bound_Z_underflows_and_y_eq_0 = 0;

                                    for (n in 1:N) {
                                           if       ( (Bound_Z[n]  >  overflow_threshold)    &&  (y[n, t] == 1) )      num_Bound_Z_overflows_and_y_eq_1  += 1;
                                           else if  ( (Bound_Z[n]  <  underflow_threshold)   &&  (y[n, t] == 0) )      num_Bound_Z_underflows_and_y_eq_0 += 1;
                                           else   num_OK_index += 1;
                                    }

                                    if (num_OK_index == N)  { //// carry on as normal as no * problematic * overflows/underflows

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

                                    } else {

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

                                                              vector[local_size] log_Bound_U_Phi_Bound_Z =  log_inv_logit( 0.07056 * square(Bound_Z[index]) .* Bound_Z[index]  + 1.5976 * Bound_Z[index] );
                                                              vector[local_size] log_Phi_Z = log(u[index, t]) +  log_Bound_U_Phi_Bound_Z ;
                                                              vector[local_size] log_1m_Phi_Z =   log1m_exp(log(u[index, t])  + log_Bound_U_Phi_Bound_Z);
                                                              vector[local_size] logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
                                                              Z_std_norm[index, t] = inv_Phi_approx_from_logit_prob(logit_Phi_Z);
                                                              y1[index, t]  =  log_Bound_U_Phi_Bound_Z ;

                                                   }
                                                   if (indicator_overflows_and_y_eq_1_empty == 0) {  //// overflow + y == 1

                                                             array[num_Bound_Z_overflows_and_y_eq_1] int index = overflows_and_y_eq_1_index;
                                                             int local_size = num_Bound_Z_overflows_and_y_eq_1;

                                                             vector[local_size] log_Bound_U_Phi_Bound_Z_1m =  log_inv_logit( - 0.07056 * square(Bound_Z[index]) .* Bound_Z[index]  - 1.5976 * Bound_Z[index] );
                                                             {
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

                                                   }

                                    }

                              } else {

                                    ////////////////////////////////////////////////////////////
                                    //// ---- ORDINAL test t (two-sided truncation):
                                    ////////////////////////////////////////////////////////////
                                    int t_ord = t - n_binary_tests;
                                    int n_thr_t = n_thr_per_ord_test[t_ord];
                                    int n_cat_t = n_cat_per_ord_test[t_ord];
                                    vector[n_thr_t] C_t = get_test_values(C_vec[c], ord_start_index, ord_end_index, t_ord);
                                    ////
                                    //// ---- Gather per-subject lower/upper cutpoints
                                    //// (finite +/- C_sentinel for unbounded bottom/top categories -- see transformed data):
                                    ////
                                    vector[N] C_lo;
                                    vector[N] C_hi;
                                    for (n in 1:N) {
                                        int k_obs = y_ord[t_ord, n];
                                        C_lo[n] = (k_obs == 1)       ? -C_sentinel : C_t[k_obs - 1];
                                        C_hi[n] = (k_obs == n_cat_t) ?  C_sentinel : C_t[k_obs];
                                    }
                                    ////
                                    vector[N] Bound_Z_lo = (C_lo - (Xbeta + inc)) * L_Omega_diag_recip[c, t];  //// use as marker for potential underflow/overflow
                                    vector[N] Bound_Z_hi = (C_hi - (Xbeta + inc)) * L_Omega_diag_recip[c, t];  //// use as marker for potential underflow/overflow
                                    ////
                                    //// ---- Classify (note Bound_Z_lo <= Bound_Z_hi always, since cutpoints are
                                    //// strictly increasing and 1/L_tt > 0):
                                    ////   "left"  <=>  BOTH bounds < underflow_threshold  <=>  Bound_Z_hi < UF
                                    ////               (bottom-category underflow + middle-category left-tail cancellation)
                                    ////   "right" <=>  BOTH bounds > overflow_threshold   <=>  Bound_Z_lo > OF
                                    ////               (top-category overflow + middle-category right-tail cancellation)
                                    ////   else OK (standard scale; incl. one-extreme-one-moderate and straddling cases,
                                    ////            which are numerically fine on the standard scale).
                                    ////
                                    int num_OK_index = 0;
                                    int num_left_tail = 0;
                                    int num_right_tail = 0;

                                    for (n in 1:N) {
                                           if       (Bound_Z_hi[n]  <  underflow_threshold)    num_left_tail  += 1;
                                           else if  (Bound_Z_lo[n]  >  overflow_threshold)     num_right_tail += 1;
                                           else   num_OK_index += 1;
                                    }

                                    if (num_OK_index == N)  { //// carry on as normal as no * problematic * overflows/underflows

                                           if (Phi_type == 2) {
                                                 vector[N] Phi_lo = Phi_approx(Bound_Z_lo);
                                                 vector[N] Phi_hi = Phi_approx(Bound_Z_hi);
                                                 vector[N] prob_t = Phi_hi - Phi_lo;
                                                 vector[N] Phi_Z  = Phi_lo + prob_t .* u[, t];
                                                 Z_std_norm[, t]  = inv_Phi_approx_from_prob(Phi_Z);
                                                 y1[, t] = log(prob_t);
                                           } else {
                                                 vector[N] Phi_lo = Phi(Bound_Z_lo);
                                                 vector[N] Phi_hi = Phi(Bound_Z_hi);
                                                 vector[N] prob_t = Phi_hi - Phi_lo;
                                                 vector[N] Phi_Z  = Phi_lo + prob_t .* u[, t];
                                                 Z_std_norm[, t]  = inv_Phi(Phi_Z);
                                                 y1[, t] = log(prob_t);
                                           }

                                    } else {

                                                  int indicator_OK_empty = 0;
                                                  if (num_OK_index < 1)  {
                                                    num_OK_index = 1;
                                                    indicator_OK_empty = 1;
                                                  }
                                                  ////
                                                  int indicator_left_tail_empty = 0;
                                                  if (num_left_tail < 1)  {
                                                    num_left_tail = 1;
                                                    indicator_left_tail_empty = 1;
                                                  }
                                                  ////
                                                  int indicator_right_tail_empty = 0;
                                                  if (num_right_tail < 1)  {
                                                    num_right_tail = 1;
                                                    indicator_right_tail_empty = 1;
                                                  }
                                                  ////
                                                  array[num_OK_index] int OK_index;
                                                  array[num_left_tail]  int left_tail_index;
                                                  array[num_right_tail] int right_tail_index;
                                                  int counter_1  = 1;
                                                  int counter_2  = 1;
                                                  int counter_3  = 1;
                                                  ////
                                                  for (n in 1:N) {
                                                         if  (Bound_Z_hi[n]  <  underflow_threshold) {
                                                             left_tail_index[counter_1] = n;
                                                             counter_1 += 1;
                                                         } else if  (Bound_Z_lo[n]  >  overflow_threshold)  {
                                                             right_tail_index[counter_2] = n;
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
                                                             vector[local_size] Phi_lo = Phi_approx(Bound_Z_lo[index]);
                                                             vector[local_size] Phi_hi = Phi_approx(Bound_Z_hi[index]);
                                                             vector[local_size] prob_t = Phi_hi - Phi_lo;
                                                             vector[local_size] Phi_Z  = Phi_lo + prob_t .* u[index, t];
                                                             Z_std_norm[index, t] = inv_Phi_approx_from_prob(Phi_Z);
                                                             y1[index, t] = log(prob_t);
                                                          } else {
                                                             vector[local_size] Phi_lo = Phi(Bound_Z_lo[index]);
                                                             vector[local_size] Phi_hi = Phi(Bound_Z_hi[index]);
                                                             vector[local_size] prob_t = Phi_hi - Phi_lo;
                                                             vector[local_size] Phi_Z  = Phi_lo + prob_t .* u[index, t];
                                                             Z_std_norm[index, t] = inv_Phi(Phi_Z);
                                                             y1[index, t] = log(prob_t);
                                                          }

                                                   }
                                                   if (indicator_left_tail_empty ==  0) { /// BOTH bounds < UF (incl. bottom category, where log_Phi_lo -> -huge)

                                                              array[num_left_tail] int index = left_tail_index;
                                                              int local_size = num_left_tail;

                                                              //// log Phi(x) ~=~ log_inv_logit(0.07056*x^3 + 1.5976*x):
                                                              vector[local_size] log_Phi_lo =  log_inv_logit( 0.07056 * square(Bound_Z_lo[index]) .* Bound_Z_lo[index]  + 1.5976 * Bound_Z_lo[index] );
                                                              vector[local_size] log_Phi_hi =  log_inv_logit( 0.07056 * square(Bound_Z_hi[index]) .* Bound_Z_hi[index]  + 1.5976 * Bound_Z_hi[index] );
                                                              ////
                                                              //// y1 = log(Phi_hi - Phi_lo) = log_diff_exp(log_Phi_hi, log_Phi_lo).
                                                              //// (bottom category: log_Phi_lo ~ -1e8 -> exp() underflows to exact 0 -> y1 = log_Phi_hi,
                                                              ////  identical to the binary "underflow + y == 0" formula):
                                                              ////
                                                              y1[index, t] = log_Phi_hi + log1m_exp(log_Phi_lo - log_Phi_hi);
                                                              ////
                                                              //// Phi_Z = Phi_lo * (1 - u) + Phi_hi * u:
                                                              ////
                                                              matrix[num_left_tail, 2] tmp_array_2d_to_lse;
                                                              tmp_array_2d_to_lse[, 1] = log_Phi_lo + log1m(u[index, t]);
                                                              tmp_array_2d_to_lse[, 2] = log_Phi_hi + log(u[index, t]);
                                                              vector[local_size] log_Phi_Z = log_sum_exp_2d(tmp_array_2d_to_lse);
                                                              ////
                                                              vector[local_size] log_1m_Phi_Z = log1m_exp(log_Phi_Z);
                                                              vector[local_size] logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
                                                              Z_std_norm[index, t] = inv_Phi_approx_from_logit_prob(logit_Phi_Z);

                                                   }
                                                   if (indicator_right_tail_empty == 0) {  //// BOTH bounds > OF (incl. top category, where log_1m_Phi_hi -> -huge)

                                                             array[num_right_tail] int index = right_tail_index;
                                                             int local_size = num_right_tail;

                                                             //// log(1 - Phi(x)) ~=~ log_inv_logit( -(0.07056*x^3 + 1.5976*x) ):
                                                             vector[local_size] log_1m_Phi_lo =  log_inv_logit( - 0.07056 * square(Bound_Z_lo[index]) .* Bound_Z_lo[index]  - 1.5976 * Bound_Z_lo[index] );
                                                             vector[local_size] log_1m_Phi_hi =  log_inv_logit( - 0.07056 * square(Bound_Z_hi[index]) .* Bound_Z_hi[index]  - 1.5976 * Bound_Z_hi[index] );
                                                             ////
                                                             //// y1 = log(Phi_hi - Phi_lo) = log((1 - Phi_lo) - (1 - Phi_hi)) = log_diff_exp(log_1m_Phi_lo, log_1m_Phi_hi).
                                                             //// (top category: log_1m_Phi_hi ~ -1e8 -> y1 = log_1m_Phi_lo,
                                                             ////  identical to the binary "overflow + y == 1" formula):
                                                             ////
                                                             y1[index, t] = log_1m_Phi_lo + log1m_exp(log_1m_Phi_hi - log_1m_Phi_lo);
                                                             ////
                                                             //// 1 - Phi_Z = (1 - Phi_lo) * (1 - u) + (1 - Phi_hi) * u:
                                                             ////
                                                             matrix[num_right_tail, 2] tmp_array_2d_to_lse;
                                                             tmp_array_2d_to_lse[, 1] = log_1m_Phi_lo + log1m(u[index, t]);
                                                             tmp_array_2d_to_lse[, 2] = log_1m_Phi_hi + log(u[index, t]);
                                                             vector[local_size] log_1m_Phi_Z = log_sum_exp_2d(tmp_array_2d_to_lse);
                                                             ////
                                                             vector[local_size] log_Phi_Z = log1m_exp(log_1m_Phi_Z);
                                                             vector[local_size] logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
                                                             Z_std_norm[index, t] = inv_Phi_approx_from_logit_prob(logit_Phi_Z);

                                                   }

                                    }

                              } //// end of binary/ordinal branch

                              if (t < n_tests)   inc = block(Z_std_norm, 1, 1, N, t) * to_vector(head(L_Omega[c, t + 1, ], t));

                        }  //// end of t loop

                        lp[, c] = rowwise_sum(y1[, 1:n_tests])  +  to_vector(log_prev[, c]);

                  } //// end of c loop

                  log_lik = log_sum_exp_2d(lp);

            } //// end of log-lik block

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
          prev[g] ~ beta(prior_prev_a[g, 1], prior_prev_b[g, 1]);
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
                    // vector[n_thr_t] C_raw_t = lb_ub_jacobian(C_unc_t, C_raw_lower, C_raw_upper);
                    ////
                    //// Induced-Dirichlet prior:
                    ////
                    int t = t_ord + n_binary_tests;
                    real anchor_ct = prior_beta_mean[c, t, 1];
                    // if (c == 1) {
                    //     anchor_ct = dot_product(baseline_case_nd[t][1:n_covs_per_outcome[c, t]], to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]));
                    // } else {
                    //     anchor_ct = dot_product(baseline_case_d[t][1:n_covs_per_outcome[c, t]],  to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]));
                    // }
                    vector[n_thr_t] cumul_prob = Phi(C_t - anchor_ct);
                    vector[n_cat_t] ord_prob   = cumul_probs_to_ord_probs(cumul_prob);
                    target += induced_dirichlet_lpdf(ord_prob | C_t, to_vector(prior_dirichlet_alpha[c][1:n_cat_t, t_ord]), 1, anchor_ct);
                    // vector[n_thr_t] cumul_prob = Phi(C_t - anchor[c]);
                    // vector[n_cat_t] ord_prob = cumul_probs_to_ord_probs(cumul_prob);
                    // target += induced_dirichlet_lpdf(ord_prob | C_t, to_vector(prior_dirichlet_alpha[c][1:n_cat_t, t_ord]), 1, anchor[c]);
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
      //// ---- Truncation (note: NOT the same as re-parameterisation!) to avoid label-switching:
      ////
      if ((n_binary_tests > 0) && (n_class > 1)) {
         if (beta[2, 1 ,1] < beta[1, 1, 1]) target += negative_infinity();
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
