

////
//// LC_MVP_bin_PartialLog_v5_chunked.stan
////
//// SEQUENTIAL CHUNKING variant of LC_MVP_bin_PartialLog_v5 - cache blocking ONLY,
//// NO parallelism of any kind. Implements ONLY the (fully_vectorised == 1) &&
//// (handle_numerical_issues == 1) path. The flags are still declared in data for
//// drop-in compatibility with the existing Stan_data_list, but they are ignored
//// (add "chunk_size" to the data list).
////
//// PURPOSE: test whether chunk-sized value containers help Stan's autodiff the way
//// they help the manual-gradient BayesMVP implementation. Expected result (the
//// hypothesis being tested): they DON'T, because although all forward-pass value
//// containers below are physically allocated at chunk size (M x n_tests), the
//// reverse-mode autodiff arena/tape still accumulates vari nodes for ALL chunks
//// across the whole model evaluation - the reverse sweep therefore still traverses
//// a working set proportional to N, defeating the cache-blocking.
////
//// KEY STRUCTURAL POINTS vs baseline:
////   (1) Parameterisation is IDENTICAL to the baseline (u_raw is still
////       matrix[N, n_tests]), so lp__ evaluated at the same unconstrained point
////       matches the baseline exactly - a direct validation check.
////   (2) The likelihood moves from transformed parameters into the model block:
////       one sequential loop over ceil(N / chunk_size) chunks, with ALL likelihood
////       containers declared inside the loop at chunk size. The u Jacobian is
////       applied per chunk via lb_ub_lp (same total as baseline).
////   (3) log_lik is NOT stored as a transformed parameter (target incremented
////       directly, per chunk). NOTE: the baseline writes vector[N] log_lik to the
////       output every draw - if benchmarking baseline vs chunked vs reduce_sum,
////       that I/O difference is a confound; consider dropping log_lik from the
////       baseline's output too for the timing runs.
////
//// USAGE: no threading flags needed - compile and run exactly like the baseline,
//// just add chunk_size to the data list (e.g. match your BayesMVP chunk sizes,
//// and/or match grainsize in the reduce_sum variant for a clean 3-way comparison).
////

functions {

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

      matrix cholesky_corr_constrain_outer_lp( vector col_one_raw,
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
            return 5.494 *  sinh(0.33333333333333331483 * asinh( 0.3418 * logit_p  )) ;
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
      int<lower=0, upper=(n_tests * (n_tests - 1)) %/% 2> known_num;
      real overflow_threshold;
      real underflow_threshold;
      ///// priors
      int prior_only;
      array[n_class] matrix[n_covariates_max, n_tests] prior_beta_mean;
      array[n_class] matrix<lower=0>[n_covariates_max, n_tests] prior_beta_sd;
      matrix<lower=0>[n_class, 1] prior_LKJ; // NOTE: Some Stan vector's written as mtx. w/ 1 col to avoid issues w/ custom C++ fns
      matrix<lower=0>[n_pops, 1] prior_p_alpha;
      matrix<lower=0>[n_pops, 1] prior_p_beta;
      ///// other
      int Phi_type;
      int handle_numerical_issues; // ACCEPTED BUT IGNORED (hni == 1 path is hard-wired)
      int fully_vectorised;        // ACCEPTED BUT IGNORED (fully_vectorised == 1 path is hard-wired)
      ////
      int<lower=1> chunk_size;     // NEW: cache-blocking chunk size (obs per chunk)
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

      int n_chunks = (N + chunk_size - 1) %/% chunk_size;  // = ceil(N / chunk_size); last chunk may be ragged

      if (corr_force_positive == 1)  lb = 0;
      else lb = -1.0;

}

parameters {

       matrix[N, n_tests] u_raw; //  UNCHANGED from baseline (identical parameterisation -> lp__ directly comparable)
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

              for (g in 1 : n_pops) {
                prev[g] ~ beta(prior_p_alpha[g, 1], prior_p_beta[g, 1]);
              }

     if (prior_only == 0) {

        //// ---- SEQUENTIAL chunk loop: cache blocking only, zero parallelism. ----
        //// Every likelihood container below is physically allocated at chunk size
        //// (M x n_tests at most), mirroring the BayesMVP workspace-struct design.

        for (ch in 1:n_chunks) {

              int i1 = (ch - 1) * chunk_size + 1;
              int i2 = min(ch * chunk_size, N);
              int M  = i2 - i1 + 1;

              //// chunk-sized containers:
              matrix[M, n_tests] y_s = y[i1:i2, ];
              array[M] int pop_s = pop[i1:i2];
              matrix[M, n_tests] u;
              matrix[M, n_class] log_prev;
              matrix[M, n_tests] Z_std_norm;
              vector[M] Bound_Z;
              matrix[M, n_class] lp;
              vector[M] inc;
              matrix[M, n_tests] y1;

              //// u transform + Jacobian for this chunk (lb_ub_lp is _lp -> Jacobian
              //// goes to target automatically; total over chunks == baseline):
              for (t in 1:n_tests) {
                  u[, t] = lb_ub_lp(u_raw[i1:i2, t], 0.0, 1.0);
              }

              for (m in 1:M) {
                log_prev[m, 1] =  log1m(prev[pop_s[m]]);
                log_prev[m, 2] =  log(prev[pop_s[m]]);
              }

              //// ---- likelihood (hni == 1, fully vectorised path; N -> M): ----

              for (c in 1 : n_class) {

                    inc = rep_vector(0.0, M);

                    for (t in 1:n_tests) {

                               if (n_covariates_max > 1) {
                                    vector[M] Xbeta;
                                    if (c == 1)   Xbeta =  X_nd[t, 1:n_covs_per_outcome[c,t], i1:i2]' *   to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);
                                    if (c == 2)   Xbeta =  X_d[t,  1:n_covs_per_outcome[c,t], i1:i2]' *   to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);
                                    Bound_Z  = - (Xbeta + inc  )  *  L_Omega_diag_recip[c, t] ; // use as marker for potential overflow
                               } else {
                                    Bound_Z  = - (beta[c, t, 1] + inc  )  *  L_Omega_diag_recip[c, t] ; // use as marker for potential overflow
                               }
                 {

                      int num_OK_index = 0 ;
                      int num_Bound_Z_overflows_and_y_eq_1 = 0 ;
                      int num_Bound_Z_underflows_and_y_eq_0 = 0 ;

                      for (m in 1:M) {
                             if       ( (Bound_Z[m]  >  overflow_threshold)    &&  (y_s[m, t] == 1) )      num_Bound_Z_overflows_and_y_eq_1  += 1;
                             else if  ( (Bound_Z[m]  <  underflow_threshold)   &&  (y_s[m, t] == 0) )      num_Bound_Z_underflows_and_y_eq_0 += 1;
                             else   num_OK_index += 1;
                      }

                      if (num_OK_index == M)  { // carry on as normal as no * problematic * overflows/underflows

                             if (Phi_type == 2) {
                                   vector[M] Bound_U_Phi_Bound_Z = Phi_approx(Bound_Z);
                                   vector[M] Phi_Z   =                  (y_s[, t] .*  Bound_U_Phi_Bound_Z  +  (y_s[,t] -   Bound_U_Phi_Bound_Z) .* (y_s[, t] + (y_s[, t] - 1.0) ) .*  u[, t])  ;
                                   Z_std_norm[, t]  =    inv_Phi_approx_from_prob(Phi_Z);
                                   y1[, t] =         log(  y_s[, t] .* (1.0 -  Bound_U_Phi_Bound_Z) + (y_s[, t] - 1.0) .*  Bound_U_Phi_Bound_Z .* ((y_s[, t]) + ((y_s[, t]) - 1.0))  );
                             } else {
                                   vector[M] Bound_U_Phi_Bound_Z = Phi(Bound_Z);
                                   vector[M] Phi_Z   =                  (y_s[, t] .*  Bound_U_Phi_Bound_Z  +  (y_s[,t] -   Bound_U_Phi_Bound_Z) .* (y_s[, t] + (y_s[, t] - 1.0) ) .*  u[, t])  ;
                                   Z_std_norm[, t]  =    inv_Phi(Phi_Z);
                                   y1[, t] =         log(  y_s[, t] .* (1.0 -  Bound_U_Phi_Bound_Z) + (y_s[, t] - 1.0) .*  Bound_U_Phi_Bound_Z .* ((y_s[, t]) + ((y_s[, t]) - 1.0))  );
                             }

                      } else if (num_OK_index < M)  {

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
                                    for (m in 1:M) {
                                           if  (    (Bound_Z[m]  >  overflow_threshold)    &&  (y_s[m, t] == 1) ) {
                                               overflows_and_y_eq_1_index[counter_1] = m;
                                               counter_1 += 1;
                                           } else if  ( (Bound_Z[m]  <  underflow_threshold)   &&  (y_s[m, t] == 0) )  {
                                               underflows_and_y_eq_0_index[counter_2] = m;
                                               counter_2 += 1;
                                           } else {
                                               OK_index[counter_3] = m;
                                               counter_3 += 1;
                                          }
                                    }
                                    ////
                                    if (indicator_OK_empty == 0) {

                                             array[num_OK_index] int index = OK_index;
                                             int local_size = num_OK_index;

                                            if (Phi_type == 2) {
                                               vector[local_size] Bound_U_Phi_Bound_Z = Phi_approx(Bound_Z[index]);
                                               vector[local_size] Phi_Z   =                  (y_s[index,t] .*  Bound_U_Phi_Bound_Z  +  (y_s[index,t] -   Bound_U_Phi_Bound_Z ) .* (y_s[index,t] + (y_s[index,t] - 1.0) ) .*  u[index,  t] )  ;
                                               Z_std_norm[index, t]  =    inv_Phi_approx_from_prob(Phi_Z);
                                               y1[index, t] =         log((y_s[index, t]) .* (1.0 -  Bound_U_Phi_Bound_Z) + ((y_s[index, t]) - 1.0) .* Bound_U_Phi_Bound_Z .*    ((y_s[index, t]) + ((y_s[index, t]) - 1.0)));
                                            } else {
                                               vector[local_size] Bound_U_Phi_Bound_Z = Phi(Bound_Z[index]);
                                               vector[local_size] Phi_Z   =                  (y_s[index,t] .*  Bound_U_Phi_Bound_Z  +  (y_s[index,t] -   Bound_U_Phi_Bound_Z ) .* (y_s[index,t] + (y_s[index,t] - 1.0) ) .*  u[index, t] )  ;
                                               Z_std_norm[index, t]  =    inv_Phi(Phi_Z);
                                               y1[index, t] =         log((y_s[index, t]) .* (1.0 -  Bound_U_Phi_Bound_Z) + ((y_s[index, t]) - 1.0) .* Bound_U_Phi_Bound_Z .*    ((y_s[index, t]) + ((y_s[index, t]) - 1.0)));
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

                           }

                              if (t < n_tests)   inc = block(Z_std_norm, 1, 1, M, t) * to_vector(head(L_Omega[c, t + 1, ], t))   ;

                      }  // end of t loop

                      lp[, c] = to_vector(rowwise_sum(y1[, 1:n_tests]))  +  to_vector(log_prev[, c])  ;

                } // end of c loop

                target += sum(log_sum_exp_2d(lp));

          } // end of chunk loop

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
        }
        else {
          Se_bin[t] = 999;
          Sp_bin[t] = 999;
          Fp_bin[t] = 999;
        }
    }

}

}
