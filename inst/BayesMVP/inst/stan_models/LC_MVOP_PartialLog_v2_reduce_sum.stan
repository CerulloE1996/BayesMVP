

// LC_MVOP_PartialLog_v2_reduce_sum.stan
//
// WITHIN-CHAIN PARALLELISM variant (reduce_sum) of LC_MVOP_PartialLog_v2.
//
// Latent Class Multivariate Ordinal Probit (LC-MVOP): mixed binary + ordinal outcomes.
// First n_binary_tests columns of y are binary (0/1).
// Next n_ordinal_tests columns are ordinal (1, 2, ..., K_t).
//
// Implements ONLY the (fully_vectorised == 1) && (handle_numerical_issues == 1) path.
// Those flags are still declared in data for drop-in compatibility with the existing
// Stan_data_list; handle_numerical_issues is ignored and fully_vectorised must be 1 (checked in
// transformed data (add "chunk_size" to the data list).
//
// KEY STRUCTURAL CHANGES vs LC_MVOP_PartialLog_v2 (identical in spirit to the binary
// LC_MVP_bin_PartialLog_v5_reduce_sum):
//   (1) u_raw is now  array[N] row_vector[n_tests]  (was matrix[N, n_tests]) so that
//       reduce_sum can SLICE the nuisance parameters directly -- each worker receives
//       only its own slice of u_raw instead of every worker copying the full
//       N x n_tests block. An R matrix init of dim (N, n_tests) still works unchanged
//       via JSON (maps to array-of-rows).
//   (2) The likelihood lives in a plain (non-_lp) partial-sum function, so the tanh
//       Jacobian for u is added to the RETURNED value instead of via jacobian +=
//       (reduce_sum partial functions cannot be _lp / _jacobian). Total log-density
//       is identical.
//   (3) log_lik is NO LONGER a transformed parameter (it would require evaluating the
//       whole likelihood a second time outside reduce_sum). target is incremented
//       directly. lp__ at a given (mapped) parameter point matches the baseline.
//   (4) Cutpoint construction (C_raw_vec -> C_vec) stays in transformed parameters --
//       it is O(n_thr), independent of N, so there is nothing to parallelise. C_vec is
//       passed into the partial function as a shared argument.
//
// WHAT IS *NOT* PARALLELISED (deliberately, same as the binary variant):
//   - Pinkney LDL correlation constrain (O(n_tests^2), N-independent)
//   - induced-Dirichlet cutpoint prior + its Jacobian (N-independent)
//   - beta / LKJ / prevalence priors
//
// USAGE (cmdstanr):
//   mod <- cmdstan_model("LC_MVOP_PartialLog_v2_reduce_sum.stan",
//                        cpp_options = list(stan_threads = TRUE))
//   fit <- mod$sample(data = ..., threads_per_chain = K, ...)
//
//   - chunk_size (data) controls the slice size; reduce_sum's dynamic TBB scheduler
//     treats it as a lower bound on work-unit size. For a deterministic partition of
//     exactly-chunk_size slices (cleaner benchmark; directly comparable to chunk_size
//     in the chunked variant), swap reduce_sum -> reduce_sum_static below.
//   - threads_per_chain = 1 with reduce_sum still slices the autodiff tape into
//     nested partial evaluations: a free extra experimental condition that isolates
//     "sliced tapes" from "parallelism".
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
      //// ---- EXACT normal GHK helpers for Phi_type == 0:
      ////
      //// Phi_type == 0 now uses the EXACT standard normal CDF in every branch. Before this change the under/overflow tail branches
      //// always used the Phi_approx (cubic-logistic) tails whatever Phi_type was, so at +/-7.5 the Phi_type == 0 target jumped by
      //// 10.67 nats (log Phi(-7.5) = -31.08 vs log Phi_approx(-7.5) = -41.75), its slope jumped, and Z switched to inv_Phi_approx.
      //// The ordinary branch also formed 1 - Phi(x) by subtraction (up to ~1.4e-3 nats lost near +7.5, and Z = +Inf for u > ~0.9965
      //// at lo = 7.5). Every Phi_type == 0 branch (ordinary AND tail) now calls the SAME scalar routines below, so the value, its
      //// gradient and Z are continuous at the thresholds by construction. The upper side is obtained by reflection,
      //// 1 - Phi(x) = Phi(-x), on the log scale, so 1 - Phi(x) is never formed by subtraction, and Z is taken from whichever of
      //// log(q) / log(1 - q) is the smaller side.
      //// log_Phi_stable, inv_Phi_from_log_lower and normal_from_log_uniform are copied verbatim from the NicoStan four_class models.
      //// Phi_type == 1 and Phi_type == 2 never call these functions (their targets are unchanged by this block).
      //// Phi_type == 1 now ALSO calls these scalar routines, in its tail branches and per-subject loop; its vectorised
      //// ordinary branches use Phi_exact_binary_Z_and_log_lik_vectorised / Phi_exact_interval_Z_and_log_lik_vectorised below. Phi_type == 2
      //// never calls them.
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
      ////
      //// Vector wrappers: return an [n, 2] matrix, column 1 = Z_std_norm, column 2 = log-likelihood contribution y1.
      ////
      matrix Phi_exact_binary_Z_and_log_lik(vector Bound_Z, vector y_vec, vector u_vec) {
                int n_obs = num_elements(Bound_Z);
                matrix[n_obs, 2] Z_and_log_lik;
                for (obs_index in 1:n_obs) {
                    Z_and_log_lik[obs_index, ] = Phi_exact_binary_step(Bound_Z[obs_index], y_vec[obs_index], u_vec[obs_index])';
                }
                return Z_and_log_lik;
      }
      matrix Phi_exact_interval_Z_and_log_lik(vector Bound_Z_lo, vector Bound_Z_hi, vector u_vec) {
                int n_obs = num_elements(Bound_Z_lo);
                matrix[n_obs, 2] Z_and_log_lik;
                for (obs_index in 1:n_obs) {
                    Z_and_log_lik[obs_index, ] = Phi_exact_interval_step(Bound_Z_lo[obs_index], Bound_Z_hi[obs_index], u_vec[obs_index])';
                }
                return Z_and_log_lik;
      }

      //////////////////////////////////////////////////////////////////////////
      //// ---- VECTORISED exact normal GHK helpers for Phi_type == 1:
      ////
      //// Phi_type == 1 is BayesMVP's exact "Phi" (R: Phi_type = "Phi" -> 1). Before this change its tail branches used the Phi_approx
      //// (cubic-logistic) tails, so the Phi_type == 1 target jumped at +/-7.5 (up to 1.26 nats measured), and its ordinary
      //// branch formed 1 - Phi(x) by subtraction (1.0 - Phi(Bound_Z), Phi_hi - Phi_lo with both near 1). Now the Phi_type == 1 tail branches
      //// and per-subject loop use the exact scalar routines above (shared with Phi_type == 0), and its ordinary branches use the two
      //// vectorised routines below.
      ////
      //// Both use the reflection 1 - Phi(x) = Phi(-x), so every CDF value that is formed is a LOWER-side value Phi(w) = 0.5 * erfc(-w / sqrt(2))
      //// (erfc keeps full relative accuracy for w < 0, unlike 0.5 * (1 + erf)), and 1 - Phi(x) is never formed by subtraction.
      ////   Binary, sign_y = +1 (y == 1) or -1 (y == 0): likelihood Phi(w) with w = -sign_y * Bound_Z, and
      ////     y == 0:  q     = u Phi(Bound_Z)          ->  Z =  inv_Phi(u Phi(w))
      ////     y == 1:  1 - q = (1 - u) Phi(-Bound_Z)   ->  Z = -inv_Phi((1 - u) Phi(w))       (Phi^{-1}(q) = -Phi^{-1}(1 - q))
      ////   Interval (lo, hi): when lo > 0 the interval is reflected to (a, b) = (-hi, -lo) with u -> 1 - u and Z -> -Z, otherwise (a, b) = (lo, hi).
      ////     After reflection a <= 0, P = Phi(b) - Phi(a) and q = (1 - v) Phi(a) + v Phi(b) is a sum of positive terms.
      //// Only the sign / weight selection is an element loop (on bound VALUES, no autodiff); erfc, log and inv_Phi are vectorised calls.
      ////
      //// Precondition (checked in transformed data): these routines are ONLY called from the ordinary branches, where w (binary) and b
      //// (interval) are >= min(underflow_threshold, -overflow_threshold) >= -35, so Phi(w), Phi(b) >= 1e-268 and never underflow (a may be
      //// very negative: Phi(a) = 0 then, with a zero, not NaN, derivative). At the thresholds log(0.5 * erfc(-w / sqrt(2))) equals
      //// log_Phi_stable(w) = log(erfc(-w / sqrt(2))) - log(2) to rounding, so the Phi_type == 1 target is continuous there.
      //// Phi_type == 0 and Phi_type == 2 never call these two functions.
      ////
      matrix Phi_exact_binary_Z_and_log_lik_vectorised(vector Bound_Z, vector y_vec, vector u_vec) {
                int n_obs = num_elements(Bound_Z);
                vector[n_obs] sign_y = y_vec + (y_vec - 1.0);
                //// Phi(w), w = -sign_y * Bound_Z:  0.5 * erfc(-w / sqrt(2)) = 0.5 * erfc(sign_y * Bound_Z / sqrt(2)).
                vector[n_obs] Phi_w = 0.5 * erfc((sign_y .* Bound_Z) / sqrt(2.0));
                //// v = u (y == 0) or 1 - u (y == 1); 1 - u is exact for u >= 0.5 and has relative error <= 1.1e-16 otherwise.
                vector[n_obs] v_vec = (1.0 - y_vec) .* u_vec + y_vec .* (1.0 - u_vec);
                matrix[n_obs, 2] Z_and_log_lik;
                Z_and_log_lik[, 1] = -sign_y .* inv_Phi(v_vec .* Phi_w);
                Z_and_log_lik[, 2] = log(Phi_w);
                return Z_and_log_lik;
      }
      matrix Phi_exact_interval_Z_and_log_lik_vectorised(vector Bound_Z_lo, vector Bound_Z_hi, vector u_vec) {
                int n_obs = num_elements(Bound_Z_lo);
                //// reflect_weight = 1 when Bound_Z_lo > 0 (reflected right-side form), 0 otherwise; keep_weight = 1 - reflect_weight.
                array[n_obs] int reflect_indicator;
                for (obs_index in 1:n_obs) reflect_indicator[obs_index] = (Bound_Z_lo[obs_index] > 0);
                vector[n_obs] reflect_weight = to_vector(reflect_indicator);
                vector[n_obs] keep_weight = 1.0 - reflect_weight;
                vector[n_obs] reflect_sign = keep_weight - reflect_weight;
                vector[n_obs] Bound_a = keep_weight .* Bound_Z_lo - reflect_weight .* Bound_Z_hi;
                vector[n_obs] Bound_b = keep_weight .* Bound_Z_hi - reflect_weight .* Bound_Z_lo;
                vector[n_obs] v_vec    = keep_weight .* u_vec + reflect_weight .* (1.0 - u_vec);
                vector[n_obs] v_vec_1m = keep_weight .* (1.0 - u_vec) + reflect_weight .* u_vec;
                vector[n_obs] Phi_a = 0.5 * erfc(-Bound_a / sqrt(2.0));
                vector[n_obs] Phi_b = 0.5 * erfc(-Bound_b / sqrt(2.0));
                matrix[n_obs, 2] Z_and_log_lik;
                Z_and_log_lik[, 1] = reflect_sign .* inv_Phi(v_vec_1m .* Phi_a + v_vec .* Phi_b);
                Z_and_log_lik[, 2] = log(Phi_b - Phi_a);
                return Z_and_log_lik;
      }

      //////////////////////////////////////////////////////////////////////////
      //// ---- Stable Phi_approx (Phi_type == 2) GHK helpers:
      ////
      //// poly(x) = 0.07056 x^3 + 1.5976 x and Phi_approx(x) = inv_logit(poly(x)), so log Phi_approx(x) = log_inv_logit(poly(x)) and
      //// log(1 - Phi_approx(x)) = log_inv_logit(-poly(x)) exactly. These helpers compute the SAME target as the old probability-scale
      //// Phi_type == 2 code, without cancellation. The old code formed 1 - Phi_approx(Bound_Z) or Phi_approx(hi) - Phi_approx(lo) on the
      //// probability scale, but 1 - Phi_approx(7.5) = 7.4e-19 is below machine epsilon: it rounded to 0 for Bound_Z > ~7.11 (binary
      //// y == 1, or the ordinal lower bound), giving log(0), a NaN adjoint and Z = +Inf, and lost digits from ~5.5 upwards.
      //// Both helpers return an [n, 2] matrix: column 1 = Z_std_norm, column 2 = log-likelihood contribution y1.
      ////
      matrix Phi_approx_binary_Z_and_log_lik_stable(vector Bound_Z, vector y_vec, vector u_vec) {
            int n_obs = num_elements(Bound_Z);
            //// sign_y = +1 (y == 1) or -1 (y == 0); poly_signed = sign_y * poly, so y1 = log_inv_logit(-poly_signed):
            //// y == 1 gives log(1 - Phi_approx(Bound_Z)), y == 0 gives log(Phi_approx(Bound_Z)).
            vector[n_obs] sign_y = y_vec + (y_vec - 1.0);
            vector[n_obs] poly_signed = sign_y .* (0.07056 * square(Bound_Z) .* Bound_Z + 1.5976 * Bound_Z);
            vector[n_obs] log_Phi_signed = log_inv_logit(poly_signed);
            vector[n_obs] log_1m_Phi_signed = log_inv_logit(-poly_signed);
            //// Z = Phi_approx^{-1}(q) via logit(q). For y == 1: q = Phi + (1 - Phi) u and 1 - q = (1 - Phi)(1 - u). y == 0 (q = Phi u)
            //// is the mirror image (poly -> -poly, u -> 1 - u, logit q -> -logit q), so with v = u (y == 1) or v = 1 - u (y == 0):
            ////     logit(q) = sign_y * [ log_sum_exp(log_Phi_signed, log_1m_Phi_signed + log(v)) - (log(1 - v) + log_1m_Phi_signed) ].
            //// log(v) and log(1 - v) are taken from log(u) and log1m(u) directly (1 - u is never formed).
            vector[n_obs] log_u = log(u_vec);
            vector[n_obs] log_1m_u = log1m(u_vec);
            vector[n_obs] log_v;
            vector[n_obs] log_1m_v;
            for (obs_index in 1:n_obs) {
                  if (y_vec[obs_index] == 1) {
                        log_v[obs_index] = log_u[obs_index];
                        log_1m_v[obs_index] = log_1m_u[obs_index];
                  } else {
                        log_v[obs_index] = log_1m_u[obs_index];
                        log_1m_v[obs_index] = log_u[obs_index];
                  }
            }
            matrix[n_obs, 2] tmp_array_2d_to_lse;
            tmp_array_2d_to_lse[, 1] = log_Phi_signed;
            tmp_array_2d_to_lse[, 2] = log_1m_Phi_signed + log_v;
            matrix[n_obs, 2] Z_and_log_lik;
            Z_and_log_lik[, 1] = inv_Phi_approx_from_logit_prob(sign_y .* (log_sum_exp_2d(tmp_array_2d_to_lse) - (log_1m_v + log_1m_Phi_signed)));
            Z_and_log_lik[, 2] = log_1m_Phi_signed;
            return Z_and_log_lik;
      }

      matrix Phi_approx_interval_Z_and_log_lik_stable(vector Bound_Z_lo, vector Bound_Z_hi, vector u_vec) {
            int n_obs = num_elements(Bound_Z_lo);
            //// With a = poly(lo), b = poly(hi):
            ////     inv_logit(b) - inv_logit(a) = inv_logit(b) * inv_logit(-a) * (1 - exp(-(b - a))),
            ////     b - a = (hi - lo) * (0.07056 * (lo^2 + lo * hi + hi^2) + 1.5976)   (exact factorisation, no cancellation).
            vector[n_obs] poly_lo = 0.07056 * square(Bound_Z_lo) .* Bound_Z_lo + 1.5976 * Bound_Z_lo;
            vector[n_obs] poly_hi = 0.07056 * square(Bound_Z_hi) .* Bound_Z_hi + 1.5976 * Bound_Z_hi;
            vector[n_obs] poly_gap = (Bound_Z_hi - Bound_Z_lo) .* (0.07056 * (square(Bound_Z_lo) + Bound_Z_lo .* Bound_Z_hi + square(Bound_Z_hi)) + 1.5976);
            vector[n_obs] log_Phi_lo = log_inv_logit(poly_lo);
            vector[n_obs] log_Phi_hi = log_inv_logit(poly_hi);
            vector[n_obs] log_1m_Phi_lo = log_inv_logit(-poly_lo);
            vector[n_obs] log_1m_Phi_hi = log_inv_logit(-poly_hi);
            //// Z = Phi_approx^{-1}(q), q = Phi_lo + u * (Phi_hi - Phi_lo), via logit(q) = log(q) - log(1 - q) with
            //// q = (1 - u) Phi_lo + u Phi_hi and 1 - q = (1 - u)(1 - Phi_lo) + u (1 - Phi_hi): both sums of positive terms.
            vector[n_obs] log_u = log(u_vec);
            vector[n_obs] log_1m_u = log1m(u_vec);
            matrix[n_obs, 2] tmp_lower;
            matrix[n_obs, 2] tmp_upper;
            tmp_lower[, 1] = log_Phi_lo + log_1m_u;
            tmp_lower[, 2] = log_Phi_hi + log_u;
            tmp_upper[, 1] = log_1m_Phi_lo + log_1m_u;
            tmp_upper[, 2] = log_1m_Phi_hi + log_u;
            matrix[n_obs, 2] Z_and_log_lik;
            Z_and_log_lik[, 1] = inv_Phi_approx_from_logit_prob(log_sum_exp_2d(tmp_lower) - log_sum_exp_2d(tmp_upper));
            Z_and_log_lik[, 2] = log_Phi_hi + log_1m_Phi_lo + log1m_exp(-poly_gap);
            return Z_and_log_lik;
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

      //////////////////////////////////////////////////////////////////////////
      // ---- reduce_sum partial-sum function.
      ////
      //// Slices u_raw by OBSERVATION. This is valid because the only sequential
      //// dependence in the likelihood (the "inc" recursion via L_Omega) runs across
      //// TESTS within a subject, never across subjects -- so each slice can run its
      //// own complete test recursion independently.
      ////
      //// Body = the (fully_vectorised == 1) && (handle_numerical_issues == 1)
      //// likelihood path from LC_MVOP_PartialLog_v2, with N -> M (slice size) and all
      //// row-indexed objects taken as start:end.
      ////
      //// Returns: (tanh Jacobian for the u slice) + (sum of per-obs log-lik).
      //////////////////////////////////////////////////////////////////////////
      real partial_log_lik( array[] row_vector u_raw_slice,
                            int start,
                            int end,
                            data matrix y,
                            data array[,] int y_ord,
                            data array[] matrix X_nd,
                            data array[] matrix X_d,
                            data array[,] int n_covs_per_outcome,
                            data array[] int pop,
                            array[,,] real beta,
                            array[] matrix L_Omega,
                            matrix L_Omega_diag_recip,
                            vector prev,
                            array[] vector C_vec,
                            data array[] int ord_start_index,
                            data array[] int ord_end_index,
                            data array[] int n_thr_per_ord_test,
                            data array[] int n_cat_per_ord_test,
                            data int n_binary_tests,
                            data real C_sentinel,
                            data int n_tests,
                            data int n_class,
                            data int n_covariates_max,
                            data int Phi_type,
                            data real overflow_threshold,
                            data real underflow_threshold) {

            int M = end - start + 1;
            real out = 0.0;

            ////
            //// ---- u transform + Jacobian (plain-function equivalent of
            //// ---- lb_ub_jacobian(u_raw, 0, 1); Jacobian added to the return value):
            ////
            matrix[M, n_tests] tanh_u;
            for (m in 1:M)  tanh_u[m, ] = tanh(u_raw_slice[m]);
            matrix[M, n_tests] u = 0.5 * (1.0 + tanh_u);
            out += - M * n_tests * log2() + sum(log1m(square(tanh_u)));

            ////
            //// ---- slice-local copies of data (data -> no autodiff cost):
            ////
            matrix[M, n_tests] y_s = y[start:end, ];
            array[M] int pop_s = pop[start:end];

            matrix[M, n_class] log_prev;
            for (m in 1:M) {
                log_prev[m, 1] = log1m(prev[pop_s[m]]);
                log_prev[m, 2] = log(prev[pop_s[m]]);
            }

            ////
            //// ---- likelihood (hni == 1, fully vectorised path; N -> M):
            ////
            matrix[M, n_tests] Z_std_norm;
            matrix[M, n_tests] y1;
            matrix[M, n_class] lp;
            vector[M] inc;

            for (c in 1:n_class) {

                  inc = rep_vector(0.0, M);

                  for (t in 1:n_tests) {

                        ////
                        //// ---- Linear predictor for test t (vector over the SLICE).
                        //// NOTE: indexed consistently with the X_nd / X_d declarations
                        //// (i.e. X[t] is [N, n_covariates_max], rows = subjects), so the
                        //// slice is taken on the ROW index:
                        ////
                        vector[M] Xbeta;
                        if (n_covariates_max > 1) {
                              if (c == 1)  Xbeta = X_nd[t][start:end, 1:n_covs_per_outcome[c, t]] * to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);
                              else         Xbeta = X_d[t][ start:end, 1:n_covs_per_outcome[c, t]] * to_vector(beta[c, t, 1:n_covs_per_outcome[c, t]]);
                        } else {
                              Xbeta = rep_vector(beta[c, t, 1], M);
                        }

                        if (t <= n_binary_tests) {

                              ////////////////////////////////////////////////////////////
                              //// ---- BINARY test t (one-sided truncation):
                              ////////////////////////////////////////////////////////////

                              vector[M] Bound_Z = - (Xbeta + inc) * L_Omega_diag_recip[c, t];

                              int num_OK_index = 0;
                              int num_Bound_Z_overflows_and_y_eq_1 = 0;
                              int num_Bound_Z_underflows_and_y_eq_0 = 0;

                              for (m in 1:M) {
                                     if       ( (Bound_Z[m]  >  overflow_threshold)    &&  (y_s[m, t] == 1) )      num_Bound_Z_overflows_and_y_eq_1  += 1;
                                     else if  ( (Bound_Z[m]  <  underflow_threshold)   &&  (y_s[m, t] == 0) )      num_Bound_Z_underflows_and_y_eq_0 += 1;
                                     else   num_OK_index += 1;
                              }

                              if (num_OK_index == M)  { //// carry on as normal as no * problematic * overflows/underflows

                                     if (Phi_type == 2) {
                                           //// Stable Phi_approx binary step (SAME target, log / logit scale): see Phi_approx_binary_Z_and_log_lik_stable.
                                           matrix[M, 2] Phi_approx_Z_and_log_lik = Phi_approx_binary_Z_and_log_lik_stable(Bound_Z, y_s[, t], u[, t]);
                                           Z_std_norm[, t] = Phi_approx_Z_and_log_lik[, 1];
                                           y1[, t] = Phi_approx_Z_and_log_lik[, 2];
                                     } else if (Phi_type == 0) {
                                           //// Phi_type == 0: EXACT normal binary step on the log scale with reflection (1 - Phi(x) = Phi(-x), never formed by subtraction);
                                           //// the SAME routine is used in the tail branches, so the target is continuous at the thresholds: see Phi_exact_binary_Z_and_log_lik.
                                           matrix[M, 2] Phi_exact_Z_and_log_lik = Phi_exact_binary_Z_and_log_lik(Bound_Z, y_s[, t], u[, t]);
                                           Z_std_norm[, t] = Phi_exact_Z_and_log_lik[, 1];
                                           y1[, t] = Phi_exact_Z_and_log_lik[, 2];
                                     } else {
                                           //// Phi_type == 1 (BayesMVP's exact "Phi"): EXACT normal binary step, vectorised, with reflection (1 - Phi(x) = Phi(-x), never formed
                                           //// by subtraction); continuous with the exact tail branches at the thresholds: see Phi_exact_binary_Z_and_log_lik_vectorised.
                                           matrix[M, 2] Phi_exact_Z_and_log_lik = Phi_exact_binary_Z_and_log_lik_vectorised(Bound_Z, y_s[, t], u[, t]);
                                           Z_std_norm[, t] = Phi_exact_Z_and_log_lik[, 1];
                                           y1[, t] = Phi_exact_Z_and_log_lik[, 2];
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
                                                       //// Stable Phi_approx binary step (SAME target, log / logit scale): see Phi_approx_binary_Z_and_log_lik_stable.
                                                       matrix[local_size, 2] Phi_approx_Z_and_log_lik = Phi_approx_binary_Z_and_log_lik_stable(Bound_Z[index], y_s[index, t], u[index, t]);
                                                       Z_std_norm[index, t] = Phi_approx_Z_and_log_lik[, 1];
                                                       y1[index, t] = Phi_approx_Z_and_log_lik[, 2];
                                                    } else if (Phi_type == 0) {
                                                       //// Phi_type == 0: EXACT normal binary step on the log scale with reflection (1 - Phi(x) = Phi(-x), never formed by subtraction);
                                                       //// the SAME routine is used in the tail branches, so the target is continuous at the thresholds: see Phi_exact_binary_Z_and_log_lik.
                                                       matrix[local_size, 2] Phi_exact_Z_and_log_lik = Phi_exact_binary_Z_and_log_lik(Bound_Z[index], y_s[index,t], u[index, t]);
                                                       Z_std_norm[index, t] = Phi_exact_Z_and_log_lik[, 1];
                                                       y1[index, t] = Phi_exact_Z_and_log_lik[, 2];
                                                    } else {
                                                       //// Phi_type == 1 (BayesMVP's exact "Phi"): EXACT normal binary step, vectorised, with reflection (1 - Phi(x) = Phi(-x), never formed
                                                       //// by subtraction); continuous with the exact tail branches at the thresholds: see Phi_exact_binary_Z_and_log_lik_vectorised.
                                                       matrix[local_size, 2] Phi_exact_Z_and_log_lik = Phi_exact_binary_Z_and_log_lik_vectorised(Bound_Z[index], y_s[index, t], u[index, t]);
                                                       Z_std_norm[index, t] = Phi_exact_Z_and_log_lik[, 1];
                                                       y1[index, t] = Phi_exact_Z_and_log_lik[, 2];
                                                    }

                                             }
                                             if (indicator_underflows_and_y_eq_0_empty ==  0) { /// underflow + y == 0

                                                        array[num_Bound_Z_underflows_and_y_eq_0] int index = underflows_and_y_eq_0_index;
                                                        int local_size = num_Bound_Z_underflows_and_y_eq_0;

                                                        if (Phi_type != 2) {
                                                              //// Phi_type == 1 (BayesMVP's exact "Phi") takes this exact tail branch too (it used the Phi_approx tail below).
                                                              //// Phi_type == 0: EXACT normal lower tail (y == 0, Bound_Z < underflow_threshold), from the SAME routine as the Phi_type == 0
                                                              //// ordinary branch, so the target, its gradient and Z are continuous at underflow_threshold. Phi_type == 2 keeps the Phi_approx tail below (unchanged).
                                                              matrix[local_size, 2] Phi_exact_Z_and_log_lik = Phi_exact_binary_Z_and_log_lik(Bound_Z[index], rep_vector(0.0, local_size), u[index, t]);
                                                              Z_std_norm[index, t] = Phi_exact_Z_and_log_lik[, 1];
                                                              y1[index, t] = Phi_exact_Z_and_log_lik[, 2];
                                                        } else {
                                                        vector[local_size] log_Bound_U_Phi_Bound_Z =  log_inv_logit( 0.07056 * square(Bound_Z[index]) .* Bound_Z[index]  + 1.5976 * Bound_Z[index] );
                                                        vector[local_size] log_Phi_Z = log(u[index, t]) +  log_Bound_U_Phi_Bound_Z ;
                                                        vector[local_size] log_1m_Phi_Z =   log1m_exp(log(u[index, t])  + log_Bound_U_Phi_Bound_Z);
                                                        vector[local_size] logit_Phi_Z = log_Phi_Z - log_1m_Phi_Z;
                                                        Z_std_norm[index, t] = inv_Phi_approx_from_logit_prob(logit_Phi_Z);
                                                        y1[index, t]  =  log_Bound_U_Phi_Bound_Z ;
                                                        }

                                             }
                                             if (indicator_overflows_and_y_eq_1_empty == 0) {  //// overflow + y == 1

                                                       array[num_Bound_Z_overflows_and_y_eq_1] int index = overflows_and_y_eq_1_index;
                                                       int local_size = num_Bound_Z_overflows_and_y_eq_1;

                                                       if (Phi_type != 2) {
                                                             //// Phi_type == 1 (BayesMVP's exact "Phi") takes this exact tail branch too (it used the Phi_approx tail below).
                                                             //// Phi_type == 0: EXACT normal upper tail (y == 1, Bound_Z > overflow_threshold; log(1 - Phi) = log Phi(-Bound_Z)), from the SAME routine as the Phi_type == 0
                                                             //// ordinary branch, so the target, its gradient and Z are continuous at overflow_threshold. Phi_type == 2 keeps the Phi_approx tail below (unchanged).
                                                             matrix[local_size, 2] Phi_exact_Z_and_log_lik = Phi_exact_binary_Z_and_log_lik(Bound_Z[index], rep_vector(1.0, local_size), u[index, t]);
                                                             Z_std_norm[index, t] = Phi_exact_Z_and_log_lik[, 1];
                                                             y1[index, t] = Phi_exact_Z_and_log_lik[, 2];
                                                       } else {
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

                        } else {

                              ////////////////////////////////////////////////////////////
                              //// ---- ORDINAL test t (two-sided truncation):
                              ////////////////////////////////////////////////////////////

                              int t_ord = t - n_binary_tests;
                              int n_thr_t = n_thr_per_ord_test[t_ord];
                              int n_cat_t = n_cat_per_ord_test[t_ord];
                              vector[n_thr_t] C_t = get_test_values(C_vec[c], ord_start_index, ord_end_index, t_ord);
                              ////
                              //// ---- Slice the observed ordinal categories for this test:
                              ////
                              array[M] int y_ord_t = y_ord[t_ord, start:end];
                              ////
                              //// ---- Gather per-subject lower/upper cutpoints
                              //// (finite +/- C_sentinel for unbounded bottom/top categories -- see transformed data):
                              ////
                              vector[M] C_lo;
                              vector[M] C_hi;
                              for (m in 1:M) {
                                  int k_obs = y_ord_t[m];
                                  C_lo[m] = (k_obs == 1)       ? -C_sentinel : C_t[k_obs - 1];
                                  C_hi[m] = (k_obs == n_cat_t) ?  C_sentinel : C_t[k_obs];
                              }
                              ////
                              vector[M] Bound_Z_lo = (C_lo - (Xbeta + inc)) * L_Omega_diag_recip[c, t];  //// use as marker for potential underflow/overflow
                              vector[M] Bound_Z_hi = (C_hi - (Xbeta + inc)) * L_Omega_diag_recip[c, t];  //// use as marker for potential underflow/overflow
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

                              for (m in 1:M) {
                                     if       (Bound_Z_hi[m]  <  underflow_threshold)    num_left_tail  += 1;
                                     else if  (Bound_Z_lo[m]  >  overflow_threshold)     num_right_tail += 1;
                                     else   num_OK_index += 1;
                              }

                              if (num_OK_index == M)  { //// carry on as normal as no * problematic * overflows/underflows

                                     if (Phi_type == 2) {
                                           //// Stable Phi_approx interval step (SAME target, log / logit scale): see Phi_approx_interval_Z_and_log_lik_stable.
                                           matrix[M, 2] Phi_approx_Z_and_log_lik = Phi_approx_interval_Z_and_log_lik_stable(Bound_Z_lo, Bound_Z_hi, u[, t]);
                                           Z_std_norm[, t] = Phi_approx_Z_and_log_lik[, 1];
                                           y1[, t] = Phi_approx_Z_and_log_lik[, 2];
                                     } else if (Phi_type == 0) {
                                           //// Phi_type == 0: EXACT normal interval step on the log scale; Bound_Z_lo > 0 uses the reflected right-side form, so neither
                                           //// Phi_hi - Phi_lo nor Phi_lo + prob_t * u is formed near 1. The SAME routine is used in the tail branches: see Phi_exact_interval_Z_and_log_lik.
                                           matrix[M, 2] Phi_exact_Z_and_log_lik = Phi_exact_interval_Z_and_log_lik(Bound_Z_lo, Bound_Z_hi, u[, t]);
                                           Z_std_norm[, t] = Phi_exact_Z_and_log_lik[, 1];
                                           y1[, t] = Phi_exact_Z_and_log_lik[, 2];
                                     } else {
                                           //// Phi_type == 1 (BayesMVP's exact "Phi"): EXACT normal interval step, vectorised; Bound_Z_lo > 0 uses the reflected form, so neither
                                           //// Phi_hi - Phi_lo nor Phi_lo + prob_t * u is formed near 1; continuous with the exact tail branches: see Phi_exact_interval_Z_and_log_lik_vectorised.
                                           matrix[M, 2] Phi_exact_Z_and_log_lik = Phi_exact_interval_Z_and_log_lik_vectorised(Bound_Z_lo, Bound_Z_hi, u[, t]);
                                           Z_std_norm[, t] = Phi_exact_Z_and_log_lik[, 1];
                                           y1[, t] = Phi_exact_Z_and_log_lik[, 2];
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
                                            for (m in 1:M) {
                                                   if  (Bound_Z_hi[m]  <  underflow_threshold) {
                                                       left_tail_index[counter_1] = m;
                                                       counter_1 += 1;
                                                   } else if  (Bound_Z_lo[m]  >  overflow_threshold)  {
                                                       right_tail_index[counter_2] = m;
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
                                                       //// Stable Phi_approx interval step (SAME target, log / logit scale): see Phi_approx_interval_Z_and_log_lik_stable.
                                                       matrix[local_size, 2] Phi_approx_Z_and_log_lik = Phi_approx_interval_Z_and_log_lik_stable(Bound_Z_lo[index], Bound_Z_hi[index], u[index, t]);
                                                       Z_std_norm[index, t] = Phi_approx_Z_and_log_lik[, 1];
                                                       y1[index, t] = Phi_approx_Z_and_log_lik[, 2];
                                                    } else if (Phi_type == 0) {
                                                       //// Phi_type == 0: EXACT normal interval step on the log scale; Bound_Z_lo > 0 uses the reflected right-side form, so neither
                                                       //// Phi_hi - Phi_lo nor Phi_lo + prob_t * u is formed near 1. The SAME routine is used in the tail branches: see Phi_exact_interval_Z_and_log_lik.
                                                       matrix[local_size, 2] Phi_exact_Z_and_log_lik = Phi_exact_interval_Z_and_log_lik(Bound_Z_lo[index], Bound_Z_hi[index], u[index, t]);
                                                       Z_std_norm[index, t] = Phi_exact_Z_and_log_lik[, 1];
                                                       y1[index, t] = Phi_exact_Z_and_log_lik[, 2];
                                                    } else {
                                                       //// Phi_type == 1 (BayesMVP's exact "Phi"): EXACT normal interval step, vectorised; Bound_Z_lo > 0 uses the reflected form, so neither
                                                       //// Phi_hi - Phi_lo nor Phi_lo + prob_t * u is formed near 1; continuous with the exact tail branches: see Phi_exact_interval_Z_and_log_lik_vectorised.
                                                       matrix[local_size, 2] Phi_exact_Z_and_log_lik = Phi_exact_interval_Z_and_log_lik_vectorised(Bound_Z_lo[index], Bound_Z_hi[index], u[index, t]);
                                                       Z_std_norm[index, t] = Phi_exact_Z_and_log_lik[, 1];
                                                       y1[index, t] = Phi_exact_Z_and_log_lik[, 2];
                                                    }

                                             }
                                             if (indicator_left_tail_empty ==  0) { /// BOTH bounds < UF (incl. bottom category, where log_Phi_lo -> -huge)

                                                        array[num_left_tail] int index = left_tail_index;
                                                        int local_size = num_left_tail;

                                                        if (Phi_type != 2) {
                                                              //// Phi_type == 1 (BayesMVP's exact "Phi") takes this exact tail branch too (it used the Phi_approx tail below).
                                                              //// Phi_type == 0: EXACT normal interval, left tail (BOTH bounds < underflow_threshold), from the SAME routine as the
                                                              //// Phi_type == 0 ordinary branch, so the target, its gradient and Z are continuous at underflow_threshold. Phi_type == 2 keeps the Phi_approx tail below (unchanged).
                                                              matrix[local_size, 2] Phi_exact_Z_and_log_lik = Phi_exact_interval_Z_and_log_lik(Bound_Z_lo[index], Bound_Z_hi[index], u[index, t]);
                                                              Z_std_norm[index, t] = Phi_exact_Z_and_log_lik[, 1];
                                                              y1[index, t] = Phi_exact_Z_and_log_lik[, 2];
                                                        } else {
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

                                             }
                                             if (indicator_right_tail_empty == 0) {  //// BOTH bounds > OF (incl. top category, where log_1m_Phi_hi -> -huge)

                                                       array[num_right_tail] int index = right_tail_index;
                                                       int local_size = num_right_tail;

                                                       if (Phi_type != 2) {
                                                             //// Phi_type == 1 (BayesMVP's exact "Phi") takes this exact tail branch too (it used the Phi_approx tail below).
                                                             //// Phi_type == 0: EXACT normal interval, right tail (BOTH bounds > overflow_threshold; reflected, 1 - Phi(x) = Phi(-x)), from the SAME routine as the
                                                             //// Phi_type == 0 ordinary branch, so the target, its gradient and Z are continuous at overflow_threshold. Phi_type == 2 keeps the Phi_approx tail below (unchanged).
                                                             matrix[local_size, 2] Phi_exact_Z_and_log_lik = Phi_exact_interval_Z_and_log_lik(Bound_Z_lo[index], Bound_Z_hi[index], u[index, t]);
                                                             Z_std_norm[index, t] = Phi_exact_Z_and_log_lik[, 1];
                                                             y1[index, t] = Phi_exact_Z_and_log_lik[, 2];
                                                       } else {
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

                              }

                        } //// end of binary/ordinal branch

                        if (t < n_tests)   inc = block(Z_std_norm, 1, 1, M, t) * to_vector(head(L_Omega[c, t + 1, ], t));

                  }  //// end of t loop

                  lp[, c] = rowwise_sum(y1[, 1:n_tests])  +  to_vector(log_prev[, c]);

            } //// end of c loop

            out += sum(log_sum_exp_2d(lp));

            return out;

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
      int fully_vectorised;         //// must be 1 (this file hard-codes the fully_vectorised == 1 path; any other value is rejected in transformed data)
      ////
      //// ---- Baseline cases for GQ:
      ////
      array[n_tests] vector[n_covariates_max] baseline_case_nd;
      array[n_tests] vector[n_covariates_max] baseline_case_d;
      ////
      int<lower=1> chunk_size;       //// NEW: reduce_sum chunk_size (slice size lower-bound; analogue of chunk_size)
}


transformed data {
      ////
      //// ---- Fail-loud checks on Phi_type and the thresholds:
      //// Phi_type must be 0 (exact, scalar), 1 (BayesMVP's exact "Phi", vectorised ordinary branch) or 2 (Phi_approx); any other value used to
      //// fall silently into the Phi_type == 1 code. The Phi_type == 1 vectorised ordinary branch needs -35 <= underflow_threshold and
      //// overflow_threshold <= 35 (see Phi_exact_binary_Z_and_log_lik_vectorised).
      ////
      if (Phi_type < 0 || Phi_type > 2) reject("Phi_type must be 0, 1 or 2; got Phi_type = ", Phi_type);
      if (Phi_type == 1 && (underflow_threshold < -35.0 || overflow_threshold > 35.0))
          reject("Phi_type = 1 needs -35 <= underflow_threshold and overflow_threshold <= 35; got underflow_threshold = ", underflow_threshold,
                 ", overflow_threshold = ", overflow_threshold);
      ////
      //// ---- Fail-loud check on fully_vectorised:
      //// This file only contains the vectorised (fully_vectorised == 1) likelihood; the flag used to be accepted and ignored, so
      //// fully_vectorised = 0 silently ran the vectorised code. Every live R caller passes 1.
      ////
      if (fully_vectorised != 1) reject("fully_vectorised must be 1: this model only implements the vectorised likelihood; got fully_vectorised = ", fully_vectorised);
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
      //// ---- Padded ordinal metadata.
      //// n_thr_per_ord_test / n_cat_per_ord_test are declared array[n_ordinal_tests], which is
      //// ZERO-LENGTH when there are no ordinal tests. reduce_sum shared arguments must still be
      //// well-formed in that case, so pass padded copies of length max(n_ordinal_tests, 1).
      ////
      array[max(n_ordinal_tests, 1)] int n_thr_per_ord_test_pad = rep_array(1, max(n_ordinal_tests, 1));
      array[max(n_ordinal_tests, 1)] int n_cat_per_ord_test_pad = rep_array(2, max(n_ordinal_tests, 1));
      if (n_ordinal_tests > 0) {
          for (t_ord in 1:n_ordinal_tests) {
              n_thr_per_ord_test_pad[t_ord] = n_thr_per_ord_test[t_ord];
              n_cat_per_ord_test_pad[t_ord] = n_cat_per_ord_test[t_ord];
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
      array[N] row_vector[n_tests] u_raw;   //// CHANGED (was matrix[N, n_tests]): array-of-rows so reduce_sum slices the nuisance parameters directly
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
      ////
      array[n_ordinal_tests > 0 ? 2 : 0] vector<lower=C_raw_lower, upper=C_raw_upper>[n_total_C_per_class] C_raw_vec;
      ////
      for (c in 1:n_class) {
        C_raw_vec[c] = lb_ub_jacobian(C_unc_vec[c], C_raw_lower, C_raw_upper);
      }
      ////
      //// NOTE: log_lik is NOT computed here (see header note (3)). The likelihood is
      //// evaluated once, inside reduce_sum, in the model block.
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
         if (beta[2, 1, 1] < beta[1, 1, 1]) target += negative_infinity();
      }
      ////
      //// ---- Likelihood (within-chain parallel):
      ////
      if (prior_only == 0) {
          //// NOTE: swap "reduce_sum" -> "reduce_sum_static" for a deterministic
          //// partition into slices of exactly chunk_size obs (cleanest benchmark).
          target += reduce_sum(partial_log_lik,
                               u_raw,
                               chunk_size,
                               y,
                               y_ord,
                               X_nd,
                               X_d,
                               n_covs_per_outcome,
                               pop,
                               beta,
                               L_Omega,
                               L_Omega_diag_recip,
                               prev,
                               C_vec,
                               ord_start_index,
                               ord_end_index,
                               n_thr_per_ord_test_pad,
                               n_cat_per_ord_test_pad,
                               n_binary_tests,
                               C_sentinel,
                               n_tests,
                               n_class,
                               n_covariates_max,
                               Phi_type,
                               overflow_threshold,
                               underflow_threshold);
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



