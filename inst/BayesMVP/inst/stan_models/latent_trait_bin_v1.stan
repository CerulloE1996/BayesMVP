  

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

      //////////////////////////////////////////////////////////////////////////
      //// ---- EXACT normal GHK helpers (added 2026-09-22, round 5):
      ////
      //// This file always uses the exact standard normal CDF (no Phi_type branches, no under/overflow tail branches). Before this
      //// change its GHK step formed 1 - Phi(x) by subtraction: the binary y == 1 step used log1m(Phi(Bound_Z)) and
      //// inv_Phi(Phi(Bound_Z) + (1 - Phi(Bound_Z)) * u). Stan's Phi(x) returns exactly 1 for x > 8.25 (and 0 for x < -37.5),
      //// so beyond those points the log-likelihood contribution was log(0) = -Inf (NaN gradient) and Z = +/-Inf, and relative accuracy was
      //// already being lost well before (1 - Phi(x) is below 1e-7 from x ~ 5.2). The routines below use the reflection
      //// 1 - Phi(x) = Phi(-x) on the log scale, so 1 - Phi(x) is never formed by subtraction, and Z is taken from whichever of
      //// log(q) / log(1 - q) is the smaller side. The target is unchanged (at ordinary points the value and gradient agree with the
      //// old code to rounding). log_Phi_stable, inv_Phi_from_log_lower, normal_from_log_uniform, Phi_exact_binary_step and
      //// Phi_exact_interval_step are copied verbatim from LC_MVOP_PartialLog_v2.stan (the same routines as its Phi_type == 0 path).
      ////
      real log_Phi_stable(real x) {
                //// Differentiate the SAME expression used for the log-CDF value. This avoids
                //// the separate approximate derivative in the bundled std_normal_lcdf routine.
                if (x < -10.0) {
                    //// Laplace continued fraction for the Mills ratio Phi(-t) / phi(t).
                    //// Twenty terms agree to double precision in this t > 10 branch.
                    real t = -x;
                    real r = 0.0;
                    for (k in 1:20) r = (21.0 - k) / (t + r);
                    return -0.5 * square(x) - 0.5 * log(2.0 * pi()) - log(t + r);
                }
                if (x <= 0.0) return log(erfc(-x / sqrt(2.0))) - log(2.0);
                return log1m(0.5 * erfc(x / sqrt(2.0)));
      }
      real inv_Phi_from_log_lower(real log_p) {
                //// Called only for a probability <= 0.5. Exponentiating is safe down to -700.
                if (log_p > -700.0) return inv_Phi(exp(log_p));
                real z = -sqrt(-2.0 * log_p);
                for (i in 1:5) {
                    real lp = log_Phi_stable(z);
                    z -= (lp - log_p) * exp(lp - std_normal_lpdf(z));
                }
                return z;
      }
      real normal_from_log_uniform(real log_u, real log1m_u) {
                if (log_u <= -log(2.0)) return inv_Phi_from_log_lower(log_u);
                return -inv_Phi_from_log_lower(log1m_u);
      }
      ////
      //// Binary GHK step (exact normal): y == 1 -> Z truncated to (Bound_Z, Inf), y == 0 -> Z truncated to (-Inf, Bound_Z).
      //// Returns [Z_std_norm, log-likelihood contribution].
      ////
      vector Phi_exact_binary_step(real Bound_Z, real y_obs, real u_obs) {
                vector[2] Z_and_log_lik;
                real log_q;
                real log_1m_q;
                if (y_obs == 1) {
                    //// log(1 - Phi(Bound_Z)) = log Phi(-Bound_Z);  q = Phi(Bound_Z) + (1 - Phi(Bound_Z)) u,  1 - q = (1 - Phi(Bound_Z)) (1 - u).
                    Z_and_log_lik[2] = log_Phi_stable(-Bound_Z);
                    log_1m_q = log1m(u_obs) + Z_and_log_lik[2];
                    log_q = log1m_exp(log_1m_q);
                } else {
                    //// log Phi(Bound_Z);  q = Phi(Bound_Z) u.
                    Z_and_log_lik[2] = log_Phi_stable(Bound_Z);
                    log_q = log(u_obs) + Z_and_log_lik[2];
                    log_1m_q = log1m_exp(log_q);
                }
                Z_and_log_lik[1] = normal_from_log_uniform(log_q, log_1m_q);
                return Z_and_log_lik;
      }
      ////
      //// Interval GHK step (exact normal): Z truncated to (Bound_Z_lo, Bound_Z_hi). Returns [Z_std_norm, log-likelihood contribution].
      //// Bound_Z_lo > 0 uses the reflected (right-side) form, so neither Phi(hi) - Phi(lo) nor Phi(lo) + prob * u is formed near 1.
      ////
      vector Phi_exact_interval_step(real Bound_Z_lo, real Bound_Z_hi, real u_obs) {
                vector[2] Z_and_log_lik;
                real log_q;
                real log_1m_q;
                if (Bound_Z_lo > 0) {
                    //// P = (1 - Phi(lo)) - (1 - Phi(hi)),  1 - q = (1 - u)(1 - Phi(lo)) + u (1 - Phi(hi))  (sum of positive terms).
                    real log_1m_Phi_lo = log_Phi_stable(-Bound_Z_lo);
                    real log_1m_Phi_hi = log_Phi_stable(-Bound_Z_hi);
                    Z_and_log_lik[2] = log_diff_exp(log_1m_Phi_lo, log_1m_Phi_hi);
                    log_1m_q = log_sum_exp(log_1m_Phi_lo + log1m(u_obs), log_1m_Phi_hi + log(u_obs));
                    log_q = log1m_exp(log_1m_q);
                } else {
                    //// P = Phi(hi) - Phi(lo),  q = (1 - u) Phi(lo) + u Phi(hi)  (sum of positive terms; 1 - q >= (1 - u) / 2 here).
                    real log_Phi_lo = log_Phi_stable(Bound_Z_lo);
                    real log_Phi_hi = log_Phi_stable(Bound_Z_hi);
                    Z_and_log_lik[2] = log_diff_exp(log_Phi_hi, log_Phi_lo);
                    log_q = log_sum_exp(log_Phi_lo + log1m(u_obs), log_Phi_hi + log(u_obs));
                    log_1m_q = log1m_exp(log_q);
                }
                Z_and_log_lik[1] = normal_from_log_uniform(log_q, log_1m_q);
                return Z_and_log_lik;
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
                                    //// Exact normal binary step on the log scale with reflection (2026-09-22, round 5). The old code used
                                    //// log1m(Phi(Bound_Z)) and inv_Phi(Phi + (1 - Phi) * u) for y == 1, which give -Inf / +Inf once Phi(Bound_Z) rounds to 1
                                    //// (Bound_Z > 8.25); y == 0 likewise fails once Phi(Bound_Z) rounds to 0 (Bound_Z < -37.5). Same target: see Phi_exact_binary_step.
                                    vector[2] Z_and_log_lik = Phi_exact_binary_step(Bound_Z, y[n, t], u[n, t]);
                                    Z_std_norm[t] = Z_and_log_lik[1];
                                    y1[t] = Z_and_log_lik[2];

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


