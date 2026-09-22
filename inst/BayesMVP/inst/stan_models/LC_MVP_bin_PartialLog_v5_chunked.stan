

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
//// and/or match chunk_size in the reduce_sum variant for a clean 3-way comparison).
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
            return 5.494448514153059 *  sinh(0.33333333333333331483 * asinh( 0.34176618822627863 * logit_p  )) ;
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

      //////////////////////////////////////////////////////////////////////////
      //// ---- EXACT normal GHK helpers for Phi_type == 0 (added 2026-09-22, round 3):
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
      //// [Round 4, 2026-09-22: Phi_type == 1 now ALSO calls these scalar routines, in its tail branches and per-subject loop; its vectorised
      //// ordinary branches use Phi_exact_binary_Z_and_log_lik_vectorised / Phi_exact_interval_Z_and_log_lik_vectorised below. Phi_type == 2
      //// never calls them.]
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
      //// ---- VECTORISED exact normal GHK helpers for Phi_type == 1 (added 2026-09-22, round 4):
      ////
      //// Phi_type == 1 is BayesMVP's exact "Phi" (R: Phi_type = "Phi" -> 1). Before this change its tail branches used the Phi_approx
      //// (cubic-logistic) tails, so the Phi_type == 1 target jumped at +/-7.5 (up to 1.26 nats measured on 2026-09-22), and its ordinary
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
      //// ---- Stable Phi_approx (Phi_type == 2) GHK helpers (added 2026-09-22):
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
      ////
      //// ---- Fail-loud checks on Phi_type and the thresholds (added 2026-09-22, round 4):
      //// Phi_type must be 0 (exact, scalar), 1 (BayesMVP's exact "Phi", vectorised ordinary branch) or 2 (Phi_approx); any other value used to
      //// fall silently into the Phi_type == 1 code. The Phi_type == 1 vectorised ordinary branch needs -35 <= underflow_threshold and
      //// overflow_threshold <= 35 (see Phi_exact_binary_Z_and_log_lik_vectorised).
      ////
      if (Phi_type < 0 || Phi_type > 2) reject("Phi_type must be 0, 1 or 2; got Phi_type = ", Phi_type);
      if (Phi_type == 1 && (underflow_threshold < -35.0 || overflow_threshold > 35.0))
          reject("Phi_type = 1 needs -35 <= underflow_threshold and overflow_threshold <= 35; got underflow_threshold = ", underflow_threshold,
                 ", overflow_threshold = ", overflow_threshold);

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
                                                      //// Round 4 (2026-09-22): Phi_type == 1 (BayesMVP's exact "Phi") takes this exact tail branch too (it used the Phi_approx tail below).
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
                                                     //// Round 4 (2026-09-22): Phi_type == 1 (BayesMVP's exact "Phi") takes this exact tail branch too (it used the Phi_approx tail below).
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
