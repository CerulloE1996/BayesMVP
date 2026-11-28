//// -------------------------------------------------------------------------------------
//// CLC_MVOP_cpp_skeleton.stan
////
//// Co-morbidity Latent Class Multivariate Ordinal Probit (CLC-MVOP) skeleton model
//// for BayesMVP manual-gradient C++ backend (constrain and generated quantities).
//// Four latent classes over two binary conditions (A, B):
////   class 1 = (0, 0) neither, class 2 = (1, 0) A only,
////   class 3 = (0, 1) B only,  class 4 = (1, 1) both.
//// Each class has its own coefficients, correlation matrix, and ordinal cutpoints.
//// -------------------------------------------------------------------------------------

functions {

      //// -------------------------------------------------------------------------------------
      //// Construct ordered cutpoints from raw unconstrained / bounded log-differences
      //// -------------------------------------------------------------------------------------
      vector construct_C(
            vector C_raw_vec,
            int softplus
      ) {
            int n_total_cutpoints = num_elements(C_raw_vec);
            vector[n_total_cutpoints] C_vec;
            C_vec[1] = C_raw_vec[1];
            if (softplus == 1) {
                  vector[n_total_cutpoints - 1] softplus_C_vec = log1p_exp(C_raw_vec[2:n_total_cutpoints]);
                  for (k in 2:n_total_cutpoints) {
                        C_vec[k] = C_vec[k - 1] + softplus_C_vec[k - 1];
                  }
            } else {
                  vector[n_total_cutpoints - 1] exp_C_vec = exp(C_raw_vec[2:n_total_cutpoints]);
                  for (k in 2:n_total_cutpoints) {
                        C_vec[k] = C_vec[k - 1] + exp_C_vec[k - 1];
                  }
            }
            return C_vec;
      }

      //// -------------------------------------------------------------------------------------
      //// Start and end indices for ragged per-test ordinal cutpoint vectors
      //// -------------------------------------------------------------------------------------
      array[] int calculate_start_indices(
            array[] int n_thr,
            int n_tests
      ) {
            array[n_tests] int start_index;
            start_index[1] = 1;
            for (t in 2:n_tests) {
                  start_index[t] = start_index[t - 1] + n_thr[t - 1];
            }
            return start_index;
      }

      array[] int calculate_end_indices(
            array[] int n_thr,
            int n_tests,
            array[] int start_index
      ) {
            array[n_tests] int end_index;
            for (t in 1:n_tests) {
                  end_index[t] = start_index[t] + n_thr[t] - 1;
            }
            return end_index;
      }

      vector get_test_values(
            vector flat_values,
            data array[] int start_index,
            data array[] int end_index,
            data int test_index
      ) {
            int n_elements = end_index[test_index] - start_index[test_index] + 1;
            vector[n_elements] result;
            for (i in 1:n_elements) {
                  result[i] = flat_values[start_index[test_index] + i - 1];
            }
            return result;
      }

      vector update_test_values(
            vector flat_values_to_update,
            vector new_values,
            data array[] int start_index,
            data array[] int end_index,
            data int test_index
      ) {
            int n_elements = end_index[test_index] - start_index[test_index] + 1;
            vector[num_elements(flat_values_to_update)] result = flat_values_to_update;
            for (i in 1:n_elements) {
                  result[start_index[test_index] + i - 1] = new_values[i];
            }
            return result;
      }

      //// -------------------------------------------------------------------------------------
      //// Bounded tanh transform without Jacobian (skeleton constrain path)
      //// -------------------------------------------------------------------------------------
      real lb_ub(
            real y,
            real lb,
            real ub
      ) {
            real tanh_y = tanh(y);
            return lb + (ub - lb) * 0.5 * (1.0 + tanh_y);
      }

      vector lb_ub(
            vector y,
            real lb,
            real ub
      ) {
            vector[num_elements(y)] tanh_y = tanh(y);
            return lb + (ub - lb) * 0.5 * (1.0 + tanh_y);
      }

      //// -------------------------------------------------------------------------------------
      //// Pinkney bounded LDL correlation transform without Jacobian
      //// -------------------------------------------------------------------------------------
      matrix Pinkney_LDL_bounds_opt_without_J(
            vector Omega_raw_vec,
            matrix lb_corr,
            matrix ub_corr,
            matrix known_values_indicator,
            matrix known_values
      ) {
            int dim = rows(lb_corr);
            matrix[dim, dim] Omega_raw_mat = rep_matrix(
                  0.0,
                  dim,
                  dim
            );
            int counter = 1;
            for (i in 2:dim) {
                  for (j in 1:(i - 1)) {
                        Omega_raw_mat[i, j] = Omega_raw_vec[counter];
                        counter += 1;
                  }
            }
            matrix[dim, dim] L = diag_matrix(
                  rep_vector(
                        1.0,
                        dim
                  )
            );
            vector[dim] D;
            D[1] = 1.0;
            for (i in 2:dim) {
                  if (known_values_indicator[i, 1] == 1) {
                        L[i, 1] = known_values[i, 1];
                  } else {
                        if (lb_corr[i, 1] >= ub_corr[i, 1]) {
                              reject("empty first-column correlation interval");
                        }
                        L[i, 1] = lb_ub(
                              Omega_raw_mat[i, 1],
                              lb_corr[i, 1],
                              ub_corr[i, 1]
                        );
                  }
                  if (is_nan(L[i, 1]) || is_inf(L[i, 1]) || abs(L[i, 1]) >= 1.0) {
                        reject("invalid first-column correlation");
                  }
            }
            D[2] = 1.0 - square(L[2, 1]);
            for (i in 3:dim) {
                  D[i] = 1.0 - square(L[i, 1]);
                  L[i, 2:(i - 1)] = rep_row_vector(
                        1.0 - square(L[i, 1]),
                        i - 2
                  );
                  real L_ij_old = L[i, 2];
                  for (j in 2:(i - 1)) {
                        real b1 = dot_product(
                              L[j, 1:(j - 1)],
                              D[1:(j - 1)]' .* L[i, 1:(j - 1)]
                        );
                        if (known_values_indicator[i, j] == 1) {
                              L[i, j] = (known_values[i, j] - b1) / D[j];
                        } else {
                              real sqrt_L_ij_old = sqrt(L_ij_old * D[j]);
                              real low = max({-sqrt_L_ij_old, lb_corr[i, j] - b1});
                              real up = min({sqrt_L_ij_old, ub_corr[i, j] - b1});
                              if (is_nan(low) || is_nan(up) || is_inf(low) || is_inf(up) || low >= up) {
                                    reject("empty or nonfinite correlation interval");
                              }
                              real x = lb_ub(
                                    Omega_raw_mat[i, j],
                                    low,
                                    up
                              );
                              L[i, j] = x / D[j];
                        }
                        L_ij_old -= D[j] * square(L[i, j]);
                        if (is_nan(L_ij_old) || is_inf(L_ij_old) || L_ij_old <= 0.0) {
                              reject("nonpositive or nonfinite LDL remainder");
                        }
                  }
                  D[i] = L_ij_old;
            }
            if (is_nan(sum(D)) || is_inf(sum(D)) || min(D) <= 0.0) {
                  reject("nonpositive or nonfinite LDL pivot");
            }
            return diag_post_multiply(
                  L,
                  sqrt(D)
            );
      }

      //// -------------------------------------------------------------------------------------
      //// Unpack free correlation vector into full lower-triangular vector
      //// -------------------------------------------------------------------------------------
      vector unpack_free_corr_vec_to_full(
            vector Omega_free_vec,
            matrix known_values_indicator
      ) {
            int dim = rows(known_values_indicator);
            int n_all = (dim * (dim - 1)) %/% 2;
            vector[n_all] Omega_full_vec = rep_vector(
                  0.0,
                  n_all
            );
            int full_counter = 1;
            int free_counter = 1;
            for (i in 2:dim) {
                  for (j in 1:(i - 1)) {
                        if (known_values_indicator[i, j] == 0) {
                              Omega_full_vec[full_counter] = Omega_free_vec[free_counter];
                              free_counter += 1;
                        }
                        full_counter += 1;
                  }
            }
            return Omega_full_vec;
      }

      //// -------------------------------------------------------------------------------------
      //// Log class probabilities per subject (2 x 2 conditional factorial form)
      //// -------------------------------------------------------------------------------------
      matrix log_prevalence_by_subject(
            data matrix X_prevalence,
            data array[] int pop,
            real mu_logit_A,
            real sigma_logit_A,
            vector z_logit_A,
            vector beta_prevalence_A,
            real mu_logit_B,
            real sigma_logit_B,
            vector z_logit_B,
            vector beta_prevalence_B,
            real log_OR
      ) {
            int n_rows = rows(X_prevalence);
            matrix[n_rows, 4] log_prev_subject;
            vector[num_elements(z_logit_A)] intercept_A = mu_logit_A + sigma_logit_A * z_logit_A;
            vector[num_elements(z_logit_B)] intercept_B = mu_logit_B + sigma_logit_B * z_logit_B;
            vector[n_rows] eta_A = intercept_A[pop];
            vector[n_rows] eta_B = intercept_B[pop];
            if (cols(X_prevalence) > 0) {
                  eta_A += X_prevalence * beta_prevalence_A;
                  eta_B += X_prevalence * beta_prevalence_B;
            }
            vector[n_rows] eta_q = eta_B + log_OR;
            vector[n_rows] log_pA = log_inv_logit(eta_A);
            vector[n_rows] log_1mA = log1m_inv_logit(eta_A);
            vector[n_rows] log_pB = log_inv_logit(eta_B);
            vector[n_rows] log_1mB = log1m_inv_logit(eta_B);
            vector[n_rows] log_pQ = log_inv_logit(eta_q);
            vector[n_rows] log_1mQ = log1m_inv_logit(eta_q);
            log_prev_subject[, 1] = log_1mA + log_1mB;
            log_prev_subject[, 2] = log_pA + log_1mQ;
            log_prev_subject[, 3] = log_1mA + log_pB;
            log_prev_subject[, 4] = log_pA + log_pQ;
            return log_prev_subject;
      }

      vector log_Phi_of_each_element(
            vector input_vec
      ) {
            int n_el = num_elements(input_vec);
            vector[n_el] log_Phi_x;
            for (i in 1:n_el) {
                  log_Phi_x[i] = std_normal_lcdf(input_vec[i] |);
            }
            return log_Phi_x;
      }

      vector inv_Phi_from_log_p(
            vector log_p,
            vector log_1m_p
      ) {
            int dim = num_elements(log_p);
            vector[dim] weight_of_lower_side = fmin(
                  1.0,
                  ceil(
                        fmax(
                              0.0,
                              log_1m_p - log_p
                        )
                  )
            );
            return weight_of_lower_side .* std_normal_log_qf(
                  fmin(
                        log_p,
                        log(0.5)
                  )
            ) - (1.0 - weight_of_lower_side) .* std_normal_log_qf(
                  fmin(
                        log_1m_p,
                        log(0.5)
                  )
            );
      }

      matrix biv_norm_log_Pr_Z1_above_a_and_Pr_Z2_given_Z1_above_a(
            vector thresh_a,
            vector thresh_b,
            real rho_val
      ) {
            int n_el = num_elements(thresh_a);
            real s_val = sqrt((1.0 - rho_val) * (1.0 + rho_val));
            vector[n_el] h_vec = -thresh_a;
            vector[n_el] k_vec = -thresh_b;
            for (i in 1:n_el) {
                  if (abs(h_vec[i]) < 1e-10) {
                        h_vec[i] = 1e-10;
                  }
                  if (abs(k_vec[i]) < 1e-10) {
                        k_vec[i] = 1e-10;
                  }
            }
            vector[n_el] beta_term;
            for (i in 1:n_el) {
                  beta_term[i] = (h_vec[i] * k_vec[i] > 0.0) ? 0.0 : 0.5;
            }
            vector[n_el] Pr_both_above = 0.5 * (Phi(h_vec) + Phi(k_vec))
                                         - owens_t(
                                               h_vec,
                                               (k_vec - rho_val * h_vec) ./ (h_vec * s_val)
                                         )
                                         - owens_t(
                                               k_vec,
                                               (h_vec - rho_val * k_vec) ./ (k_vec * s_val)
                                         )
                                         - beta_term;
            vector[n_el] log_Pr_Z1_above_a = log_Phi_of_each_element(-thresh_a);
            vector[n_el] Pr_Z2_above_b_given_Z1_above_a = fmin(
                  fmax(
                        Pr_both_above ./ exp(log_Pr_Z1_above_a),
                        0.0
                  ),
                  1.0
            );
            matrix[n_el, 3] result_mat;
            result_mat[, 1] = log_Pr_Z1_above_a;
            result_mat[, 2] = Pr_Z2_above_b_given_Z1_above_a;
            result_mat[, 3] = 1.0 - Pr_Z2_above_b_given_Z1_above_a;
            return result_mat;
      }

}


data {
      int<lower=1> N;
      int<lower=2> n_tests;
      int<lower=0> n_binary_tests;
      int<lower=0> n_ordinal_tests;
      array[n_ordinal_tests] int n_cat_per_ord_test;
      array[n_ordinal_tests] int n_thr_per_ord_test;
      array[4] matrix<lower=0>[n_ordinal_tests > 0 ? max(n_cat_per_ord_test) : 1, max(n_ordinal_tests, 1)] prior_dirichlet_alpha;
      matrix[N, n_tests] y;
      matrix[N, n_tests] y_upper_when_interval_censored;
      int<lower=4, upper=4> n_class;
      array[n_tests] int test_perm;
      array[n_tests] int n_cat_per_test;
      int<lower=1> n_pops;
      array[N] int pop;
      vector<lower=0>[n_pops] pop_weight;
      array[n_class, 2] int<lower=0, upper=1> class_D;
      array[n_tests] int<lower=1, upper=2> test_condition;
      array[n_tests] int<lower=0, upper=1> is_perfect_ref;
      real<lower=0> perfect_ref_M;
      int<lower=1> n_covariates_max;
      array[n_class, n_tests] matrix[N, n_covariates_max] X;
      array[n_class, n_tests] int<lower=0, upper=n_covariates_max> n_covs_per_outcome;
      int<lower=0> n_prevalence_covariates;
      matrix[N, n_prevalence_covariates] X_prevalence;
      int<lower=0, upper=1> corr_force_positive;
      int<lower=0> known_num;
      int<lower=0, upper=1> pool_correlations_across_classes;
      array[n_class] matrix[n_tests, n_tests] known_values_list;
      array[n_class] matrix[n_tests, n_tests] known_values_indicator_list;
      array[n_class] matrix[n_tests, n_tests] lb_corr;
      array[n_class] matrix[n_tests, n_tests] ub_corr;
      matrix<lower=0>[n_class, 1] prior_LKJ;
      real overflow_threshold;
      real underflow_threshold;
      real C_raw_lower;
      real C_raw_upper;
      int<lower=0, upper=1> prior_only;
      array[n_class] matrix[n_covariates_max, n_tests] prior_beta_mean;
      array[n_class] matrix<lower=0>[n_covariates_max, n_tests] prior_beta_sd;
      int Phi_type;
      int handle_numerical_issues;
      int fully_vectorised;
      int<lower=0, upper=3> use_pop_prev_prior;   //// 3 = the prior_logit_prob_B_given_A arguments sit on Pr(A | B) (manual-gradient C++)
      array[n_class, n_tests] vector[n_covariates_max] baseline_case;
      int<lower=0, upper=1> use_Se_Sp_prior;   //// 0 = the induced Dirichlet only; 1 = the prior on the marginal Se / Sp (Jensen-adjusted, with the gap prior)
      int<lower=1> n_chosen_cutoffs_max;
      array[n_class, max(n_ordinal_tests, 1)] int<lower=0, upper=n_chosen_cutoffs_max> n_chosen_cutoffs;
      array[n_class, max(n_ordinal_tests, 1), n_chosen_cutoffs_max] int<lower=1> chosen_cutoffs;
      array[n_class, max(n_ordinal_tests, 1)] matrix[n_chosen_cutoffs_max, 2] prior_Se_Sp_link_mean_SD;
      int<lower=0, upper=1> use_pop_Se_Sp_prior;
      int<lower=1> n_standardisation;
      array[n_standardisation] int<lower=1, upper=N> standardisation_index;
      vector<lower=0>[n_standardisation] standardisation_weight;
      int<lower=0, upper=1> use_label_constraint;
      array[2] int<lower=1, upper=n_tests> reference_test;
      array[n_tests] int<lower=2> cutoff_category;
      int<lower=0, upper=1> save_subject_probs;
}


transformed data {
      array[n_class] matrix[n_tests, n_tests] lb_corr_actual = lb_corr;
      if (corr_force_positive == 1) {
            for (c in 1:n_class) {
                  for (i in 2:n_tests) {
                        for (j in 1:(i - 1)) {
                              lb_corr_actual[c][i, j] = 0.0;
                        }
                  }
            }
      }
      int n_covariates_total = 0;
      for (c in 1:n_class) {
            for (t in 1:n_tests) {
                  n_covariates_total += n_covs_per_outcome[c, t];
            }
      }
      array[n_tests] int inv_perm;
      for (j in 1:n_tests) {
            inv_perm[test_perm[j]] = j;
      }
      array[n_tests] int test_is_binary;
      array[n_tests] int ord_idx_of_slot;
      {
            int k = 0;
            for (j in 1:n_tests) {
                  if (n_cat_per_test[j] > 2) {
                        k += 1;
                        test_is_binary[j] = 0;
                        ord_idx_of_slot[j] = k;
                  } else {
                        test_is_binary[j] = 1;
                        ord_idx_of_slot[j] = -1;
                  }
            }
      }
      array[max(n_binary_tests, 1)] int original_test_index_of_original_binary_test = rep_array(
            1,
            max(n_binary_tests, 1)
      );
      array[max(n_ordinal_tests, 1)] int original_test_index_of_original_ordinal_test = rep_array(
            1,
            max(n_ordinal_tests, 1)
      );
      array[n_tests] int original_ordinal_slot_of_original_test = rep_array(
            0,
            n_tests
      );
      array[max(n_ordinal_tests, 1)] int n_thr_per_ord_test_orig = rep_array(
            1,
            max(n_ordinal_tests, 1)
      );
      {
            int b = 0;
            int o = 0;
            for (t in 1:n_tests) {
                  int s = inv_perm[t];
                  if (n_cat_per_test[s] > 2) {
                        o += 1;
                        original_test_index_of_original_ordinal_test[o] = t;
                        original_ordinal_slot_of_original_test[t] = o;
                        n_thr_per_ord_test_orig[o] = n_thr_per_ord_test[ord_idx_of_slot[s]];
                  } else {
                        b += 1;
                        original_test_index_of_original_binary_test[b] = t;
                  }
            }
      }
      int n_thr_max_gq = (n_ordinal_tests > 0) ? max(n_thr_per_ord_test) : 1;
      int n_total_C_per_class = (n_ordinal_tests > 0) ? sum(n_thr_per_ord_test) : 0;
      array[max(n_ordinal_tests, 1)] int start_index = rep_array(
            1,
            max(n_ordinal_tests, 1)
      );
      array[max(n_ordinal_tests, 1)] int end_index = rep_array(
            1,
            max(n_ordinal_tests, 1)
      );
      array[max(n_ordinal_tests, 1)] int orig_start_index = rep_array(
            1,
            max(n_ordinal_tests, 1)
      );
      array[max(n_ordinal_tests, 1)] int orig_end_index = rep_array(
            1,
            max(n_ordinal_tests, 1)
      );
      if (n_ordinal_tests > 0) {
            start_index = calculate_start_indices(
                  n_thr_per_ord_test,
                  n_ordinal_tests
            );
            end_index = calculate_end_indices(
                  n_thr_per_ord_test,
                  n_ordinal_tests,
                  start_index
            );
            orig_start_index = calculate_start_indices(
                  n_thr_per_ord_test_orig,
                  n_ordinal_tests
            );
            orig_end_index = calculate_end_indices(
                  n_thr_per_ord_test_orig,
                  n_ordinal_tests,
                  orig_start_index
            );
      }
      array[n_class] int n_corrs_per_class = rep_array(
            0,
            n_class
      );
      array[n_class] int corr_start_index;
      int n_corrs_total = 0;
      for (c in 1:n_class) {
            corr_start_index[c] = n_corrs_total + 1;
            for (i in 2:n_tests) {
                  for (j in 1:(i - 1)) {
                        if (known_values_indicator_list[c][i, j] == 0) {
                              n_corrs_per_class[c] += 1;
                        }
                  }
            }
            n_corrs_total += n_corrs_per_class[c];
      }
      int n_corrs_global = (pool_correlations_across_classes == 1) ? n_corrs_per_class[1] : 0;
      int class_00 = 1;
      int class_10 = 2;
      int class_01 = 3;
      int class_11 = 4;
      int class_with_neither_condition = 0;
      array[2] int class_with_only_condition = { 0, 0 };
      for (c in 1:n_class) {
            if (class_D[c, 1] == 0 && class_D[c, 2] == 0) {
                  class_with_neither_condition = c;
            }
            if (class_D[c, 1] == 1 && class_D[c, 2] == 0) {
                  class_with_only_condition[1] = c;
            }
            if (class_D[c, 1] == 0 && class_D[c, 2] == 1) {
                  class_with_only_condition[2] = c;
            }
      }
      vector[n_standardisation] standardisation_w = standardisation_weight / sum(standardisation_weight);
      vector[n_standardisation] log_standardisation_w = log(standardisation_w);
      matrix[n_standardisation, n_prevalence_covariates] X_prevalence_standardisation = X_prevalence[standardisation_index];
      array[n_standardisation] int pop_standardisation = pop[standardisation_index];
      matrix[n_standardisation, n_pops] profile_is_in_pop = rep_matrix(
            0.0,
            n_standardisation,
            n_pops
      );
      for (i in 1:n_standardisation) {
            profile_is_in_pop[i, pop_standardisation[i]] = 1.0;
      }
      vector[n_pops] standardisation_weight_in_pop = to_vector(standardisation_w' * profile_is_in_pop);
      //// ---- The number of subjects in each population (for the Pinkney partial-centering weights, transformed parameters):
      vector[n_pops] n_of_each_pop = rep_vector(0.0, n_pops);
      for (n in 1:N) {
            n_of_each_pop[pop[n]] += 1.0;
      }
      array[n_class, n_tests] matrix[n_standardisation, n_covariates_max] X_standardisation;
      for (c in 1:n_class) {
            for (t in 1:n_tests) {
                  X_standardisation[c, t] = X[c, t][standardisation_index, ];
            }
      }
}


parameters {
      //// -------------------------------------------------------------------------------------
      //// Parameter layout matches CLC_MVOP_lp_grad_common_T.hpp theta_main (FITTED slot order)
      //// -------------------------------------------------------------------------------------
      vector[n_corrs_total] Omega_unconstrained_vec;
      vector[n_corrs_global] Omega_global_unconstrained_vec;
      array[pool_correlations_across_classes == 1 ? n_class : 0] real<lower=0, upper=1> lambda_class;
      vector[n_covariates_total] beta_vec;
      vector[n_prevalence_covariates] beta_prevalence_A;
      vector[n_prevalence_covariates] beta_prevalence_B;
      vector[n_pops] z_logit_A_pcp;   //// Pinkney partially-centered (w = 0 centered, w = 1 non-centered)
      vector[n_pops] z_logit_B_pcp;
      real mu_logit_A;
      real mu_logit_B;
      real<lower=0> sigma_logit_A;
      real<lower=0> sigma_logit_B;
      //// logit Pr(B | A) intercept; log_OR = mu_logit_B_given_A - mu_logit_B (Jacobian 1); use_pop_prev_prior = 3: log_OR >= 0
      real<lower=(use_pop_prev_prior == 3 ? mu_logit_B : negative_infinity())> mu_logit_B_given_A;
      array[n_ordinal_tests > 0 ? n_class : 0] vector[n_total_C_per_class] C_unc_vec;
}


transformed parameters {
      ////
      //// ---- Pinkney partially-centered population logits (as CLC_MVOP_PartialLog_v2_NicoSIMD_AVX512_fused.stan):
      ////      w = 1 / (1 + n_g p (1 - p) sigma^2), p = inv_logit(mu); tau_wt = sigma - w * (sigma - 1);
      ////      theta = (z * sigma + mu * w) ./ tau_wt; z_logit = (theta - mu) / sigma = (z - mu * (1 - w)) ./ tau_wt
      ////
      vector[n_pops] Pinkney_w_A = 1.0 ./ (1.0 + n_of_each_pop * (inv_logit(mu_logit_A) * (1.0 - inv_logit(mu_logit_A)) * square(sigma_logit_A)));
      vector[n_pops] Pinkney_w_B = 1.0 ./ (1.0 + n_of_each_pop * (inv_logit(mu_logit_B) * (1.0 - inv_logit(mu_logit_B)) * square(sigma_logit_B)));
      vector[n_pops] tau_wt_A = sigma_logit_A - Pinkney_w_A * (sigma_logit_A - 1);
      vector[n_pops] tau_wt_B = sigma_logit_B - Pinkney_w_B * (sigma_logit_B - 1);
      vector[n_pops] z_logit_A = (z_logit_A_pcp - mu_logit_A * (1 - Pinkney_w_A)) ./ tau_wt_A;
      vector[n_pops] z_logit_B = (z_logit_B_pcp - mu_logit_B * (1 - Pinkney_w_B)) ./ tau_wt_B;
      real log_OR = mu_logit_B_given_A - mu_logit_B;
      //// -------------------------------------------------------------------------------------
      //// Per-class regression coefficients in FITTED slot order and ORIGINAL test order
      //// -------------------------------------------------------------------------------------
      array[n_class, n_tests, n_covariates_max] real beta_fitted;
      array[n_class, n_tests, n_covariates_max] real beta;
      {
            int counter = 1;
            for (c in 1:n_class) {
                  for (j in 1:n_tests) {
                        for (k in 1:n_covs_per_outcome[c, j]) {
                              beta_fitted[c, j, k] = beta_vec[counter];
                              counter += 1;
                        }
                        for (k in (n_covs_per_outcome[c, j] + 1):n_covariates_max) {
                              beta_fitted[c, j, k] = 0.0;
                        }
                        if (is_perfect_ref[j] == 1) {
                              beta_fitted[c, j, 1] = (class_D[c, test_condition[j]] == 1) ? perfect_ref_M : -perfect_ref_M;
                              for (k in 2:n_covariates_max) {
                                    beta_fitted[c, j, k] = 0.0;
                              }
                        }
                  }
            }
            for (c in 1:n_class) {
                  for (t in 1:n_tests) {
                        int s = inv_perm[t];
                        for (k in 1:n_covariates_max) {
                              beta[c, t, k] = beta_fitted[c, s, k];
                        }
                  }
            }
      }
      //// -------------------------------------------------------------------------------------
      //// Stratum baseline prevalences and population standardisation targets
      //// -------------------------------------------------------------------------------------
      vector[n_pops] logit_prob_A = mu_logit_A + sigma_logit_A * z_logit_A;
      vector[n_pops] logit_prob_B_given_not_A = mu_logit_B + sigma_logit_B * z_logit_B;
      vector[n_pops] logit_prob_B_given_A = logit_prob_B_given_not_A + log_OR;
      vector<lower=0, upper=1>[n_pops] prob_A = inv_logit(logit_prob_A);
      vector<lower=0, upper=1>[n_pops] prob_B_given_not_A = inv_logit(logit_prob_B_given_not_A);
      vector<lower=0, upper=1>[n_pops] prob_B_given_A = inv_logit(logit_prob_B_given_A);
      //// (the class log prevalences of each standardisation profile, n_standardisation x n_class, are LOCAL in each block
      ////  that uses them, so they are never saved or summarised: they are 29560 numbers per draw)
      real log_Pr_B_given_A_target = 0.0;
      real log1m_Pr_B_given_A_target = 0.0;
      real log_prev_A_target = 0.0;
      real log1m_prev_A_target = 0.0;
      real log_prev_B_target = 0.0;
      real log1m_prev_B_target = 0.0;
      {
            vector[n_standardisation] eta_A = mu_logit_A + sigma_logit_A * z_logit_A[pop_standardisation];
            vector[n_standardisation] eta_B = mu_logit_B + sigma_logit_B * z_logit_B[pop_standardisation];
            if (n_prevalence_covariates > 0) {
                  eta_A += X_prevalence_standardisation * beta_prevalence_A;
                  eta_B += X_prevalence_standardisation * beta_prevalence_B;
            }
            vector[n_standardisation] eta_q = eta_B + log_OR;
            vector[n_standardisation] prob_a_prof = inv_logit(eta_A);
            vector[n_standardisation] one_m_a_prof = inv_logit(-eta_A);
            vector[n_standardisation] prob_b_prof = inv_logit(eta_B);
            vector[n_standardisation] one_m_b_prof = inv_logit(-eta_B);
            vector[n_standardisation] prob_q_prof = inv_logit(eta_q);
            vector[n_standardisation] one_m_q_prof = inv_logit(-eta_q);
            vector[n_standardisation] w_a = standardisation_w .* prob_a_prof;
            vector[n_standardisation] w_1ma = standardisation_w .* one_m_a_prof;
            real log_sum_w_a = log(sum(w_a));
            log_Pr_B_given_A_target = log(dot_product(w_a, prob_q_prof)) - log_sum_w_a;
            log1m_Pr_B_given_A_target = log(dot_product(w_a, one_m_q_prof)) - log_sum_w_a;
            log_prev_A_target = log_sum_w_a;
            log1m_prev_A_target = log(sum(w_1ma));
            log_prev_B_target = log(
                  dot_product(w_1ma, prob_b_prof) + dot_product(w_a, prob_q_prof)
            );
            log1m_prev_B_target = log(
                  dot_product(w_1ma, one_m_b_prof) + dot_product(w_a, one_m_q_prof)
            );
      }
      array[n_pops] vector[n_class] prev;
      {
            matrix[n_standardisation, n_class] log_prev_standardisation = log_prevalence_by_subject(X_prevalence_standardisation, pop_standardisation,
                                                                                                    mu_logit_A, sigma_logit_A, z_logit_A, beta_prevalence_A,
                                                                                                    mu_logit_B, sigma_logit_B, z_logit_B, beta_prevalence_B,
                                                                                                    log_OR);
            matrix[n_standardisation, n_class] prev_standardisation = exp(log_prev_standardisation);
            matrix[n_pops, n_class] weighted_sum_per_pop = profile_is_in_pop' * diag_pre_multiply(
                  standardisation_w,
                  prev_standardisation
            );
            for (g in 1:n_pops) {
                  if (standardisation_weight_in_pop[g] > 0.0) {
                        prev[g] = to_vector(weighted_sum_per_pop[g, ]) / standardisation_weight_in_pop[g];
                  } else {
                        prev[g][1] = (1.0 - prob_A[g]) * (1.0 - prob_B_given_not_A[g]);
                        prev[g][2] = prob_A[g] * (1.0 - prob_B_given_A[g]);
                        prev[g][3] = (1.0 - prob_A[g]) * prob_B_given_not_A[g];
                        prev[g][4] = prob_A[g] * prob_B_given_A[g];
                  }
            }
      }
      //// -------------------------------------------------------------------------------------
      //// Correlations via Pinkney bounded LDL (fitted slot order and original test order)
      //// -------------------------------------------------------------------------------------
      matrix[n_tests, n_tests] L_Omega_global = diag_matrix(
            rep_vector(
                  1.0,
                  n_tests
            )
      );
      matrix[n_tests, n_tests] Omega_global = diag_matrix(
            rep_vector(
                  1.0,
                  n_tests
            )
      );
      array[n_class] matrix[n_tests, n_tests] L_Omega_departure;
      array[n_class] matrix[n_tests, n_tests] L_Omega;
      array[n_class] matrix[n_tests, n_tests] Omega;
      array[n_class] matrix[n_tests, n_tests] Omega_orig;
      array[n_class] matrix[n_tests, n_tests] L_Omega_orig;
      {
            matrix[n_tests, n_tests] Omega_global_fitted = diag_matrix(
                  rep_vector(
                        1.0,
                        n_tests
                  )
            );
            if (pool_correlations_across_classes == 1) {
                  vector[(n_tests * (n_tests - 1)) %/% 2] Omega_global_full = unpack_free_corr_vec_to_full(
                        Omega_global_unconstrained_vec,
                        known_values_indicator_list[1]
                  );
                  L_Omega_global = Pinkney_LDL_bounds_opt_without_J(
                        Omega_global_full,
                        lb_corr_actual[1],
                        ub_corr[1],
                        known_values_indicator_list[1],
                        known_values_list[1]
                  );
                  Omega_global_fitted = multiply_lower_tri_self_transpose(L_Omega_global);
                  for (i in 1:n_tests) {
                        for (j in 1:n_tests) {
                              Omega_global[i, j] = Omega_global_fitted[inv_perm[i], inv_perm[j]];
                        }
                  }
            }
            for (c in 1:n_class) {
                  vector[n_corrs_per_class[c]] Omega_raw_c = rep_vector(
                        0.0,
                        n_corrs_per_class[c]
                  );
                  if (n_corrs_per_class[c] > 0) {
                        Omega_raw_c = segment(
                              Omega_unconstrained_vec,
                              corr_start_index[c],
                              n_corrs_per_class[c]
                        );
                  }
                  vector[(n_tests * (n_tests - 1)) %/% 2] Omega_full_c = unpack_free_corr_vec_to_full(
                        Omega_raw_c,
                        known_values_indicator_list[c]
                  );
                  L_Omega_departure[c] = Pinkney_LDL_bounds_opt_without_J(
                        Omega_full_c,
                        lb_corr_actual[c],
                        ub_corr[c],
                        known_values_indicator_list[c],
                        known_values_list[c]
                  );
                  if (pool_correlations_across_classes == 1) {
                        Omega[c] = (1.0 - lambda_class[c]) * Omega_global_fitted
                                   + lambda_class[c] * multiply_lower_tri_self_transpose(L_Omega_departure[c]);
                        L_Omega[c] = cholesky_decompose(Omega[c]);
                  } else {
                        L_Omega[c] = L_Omega_departure[c];
                        Omega[c] = multiply_lower_tri_self_transpose(L_Omega[c]);
                  }
                  for (i in 1:n_tests) {
                        for (j in 1:n_tests) {
                              Omega_orig[c][i, j] = Omega[c][inv_perm[i], inv_perm[j]];
                        }
                  }
                  L_Omega_orig[c] = cholesky_decompose(Omega_orig[c]);
            }
      }
      //// -------------------------------------------------------------------------------------
      //// Construct per-class ordinal cutpoints (fitted slot order and original test order)
      //// -------------------------------------------------------------------------------------
      array[n_ordinal_tests > 0 ? n_class : 0] vector[n_total_C_per_class] C_raw_vec;
      array[n_ordinal_tests > 0 ? n_class : 0] vector[n_total_C_per_class] C_vec;
      array[n_ordinal_tests > 0 ? n_class : 0] vector[n_total_C_per_class] C_vec_orig;
      if (n_ordinal_tests > 0) {
            for (c in 1:n_class) {
                  C_raw_vec[c] = lb_ub(
                        C_unc_vec[c],
                        C_raw_lower,
                        C_raw_upper
                  );
                  C_vec[c] = rep_vector(
                        0.0,
                        n_total_C_per_class
                  );
                  C_vec_orig[c] = rep_vector(
                        0.0,
                        n_total_C_per_class
                  );
                  for (t_ord_orig in 1:n_ordinal_tests) {
                        int t_orig = original_test_index_of_original_ordinal_test[t_ord_orig];
                        int s_slot = inv_perm[t_orig];
                        int s_ord = ord_idx_of_slot[s_slot];
                        vector[n_thr_per_ord_test[s_ord]] C_raw_t = get_test_values(
                              C_raw_vec[c],
                              start_index,
                              end_index,
                              s_ord
                        );
                        vector[n_thr_per_ord_test[s_ord]] C_t = construct_C(
                              C_raw_t,
                              0
                        );
                        C_vec[c] = update_test_values(
                              C_vec[c],
                              C_t,
                              start_index,
                              end_index,
                              s_ord
                        );
                        C_vec_orig[c] = update_test_values(
                              C_vec_orig[c],
                              C_t,
                              orig_start_index,
                              orig_end_index,
                              t_ord_orig
                        );
                  }
            }
      }
}


model {
      for (c in 1:n_class) {
            for (i in 2:n_tests) {
                  for (j in 1:(i - 1)) {
                        if (known_values_indicator_list[c][i, j] == 1) {
                              Omega[c][i, j] ~ normal(
                                    0.0,
                                    10.0
                              );
                        }
                  }
            }
      }
}


generated quantities {
      array[n_pops] vector[n_class] p = prev;
      vector[n_pops] pi_00;
      vector[n_pops] pi_10;
      vector[n_pops] pi_01;
      vector[n_pops] pi_11;
      vector[n_pops] prev_A;
      vector[n_pops] prev_B;
      vector[n_pops] Pr_B_given_A;
      vector[n_pops] Pr_A_given_B;
      vector[n_pops] log_OR_comorbid;
      vector[n_pops] OR_comorbid;
      for (g in 1:n_pops) {
            pi_00[g] = prev[g][class_00];
            pi_10[g] = prev[g][class_10];
            pi_01[g] = prev[g][class_01];
            pi_11[g] = prev[g][class_11];
            prev_A[g] = pi_10[g] + pi_11[g];
            prev_B[g] = pi_01[g] + pi_11[g];
            Pr_B_given_A[g] = pi_11[g] / prev_A[g];
            Pr_A_given_B[g] = pi_11[g] / prev_B[g];
            log_OR_comorbid[g] = log(pi_11[g]) + log(pi_00[g]) - log(pi_10[g]) - log(pi_01[g]);
            OR_comorbid[g] = exp(log_OR_comorbid[g]);
      }
      vector[n_class] prev_overall;
      {
            matrix[n_standardisation, n_class] log_prev_standardisation = log_prevalence_by_subject(X_prevalence_standardisation, pop_standardisation,
                                                                                                    mu_logit_A, sigma_logit_A, z_logit_A, beta_prevalence_A,
                                                                                                    mu_logit_B, sigma_logit_B, z_logit_B, beta_prevalence_B,
                                                                                                    log_OR);
            prev_overall = (standardisation_w' * exp(log_prev_standardisation))';
      }
      real pi_00_overall = prev_overall[class_00];
      real pi_10_overall = prev_overall[class_10];
      real pi_01_overall = prev_overall[class_01];
      real pi_11_overall = prev_overall[class_11];
      real prev_A_overall = exp(log_prev_A_target);
      real prev_B_overall = exp(log_prev_B_target);
      real Pr_B_given_A_overall = exp(log_Pr_B_given_A_target);
      real Pr_A_given_B_overall = pi_11_overall / prev_B_overall;
      real Pr_B_given_not_A_overall = pi_01_overall / (1.0 - prev_A_overall);
      real log_OR_comorbid_overall = log(pi_11_overall) + log(pi_00_overall) - log(pi_10_overall) - log(pi_01_overall);
      real OR_comorbid_overall = exp(log_OR_comorbid_overall);
      real logit_prev_A_population = log_prev_A_target - log1m_prev_A_target;
      real logit_prev_B_population = log_prev_B_target - log1m_prev_B_target;
      real logit_prob_B_given_A_population = log_Pr_B_given_A_target - log1m_Pr_B_given_A_target;
      //// -------------------------------------------------------------------------------------
      //// Baseline linear predictors and baseline Se/Sp (ORIGINAL test order)
      //// -------------------------------------------------------------------------------------
      array[n_class, n_tests] real Xbeta_baseline;
      for (c in 1:n_class) {
            for (t in 1:n_tests) {
                  int s = inv_perm[t];
                  int nc = n_covs_per_outcome[c, s];
                  if (nc > 0) {
                        Xbeta_baseline[c, t] = dot_product(
                              baseline_case[c, s][1:nc],
                              to_vector(beta_fitted[c, s, 1:nc])
                        );
                  } else {
                        Xbeta_baseline[c, t] = 0.0;
                  }
            }
      }
      matrix[n_class, n_binary_tests] Pr_positive_baseline_bin;
      vector[n_binary_tests] Se_baseline_bin;
      vector[n_binary_tests] Sp_baseline_bin;
      vector[n_binary_tests] Fp_baseline_bin;
      for (t_bin in 1:n_binary_tests) {
            int t = original_test_index_of_original_binary_test[t_bin];
            int s = inv_perm[t];
            for (c in 1:n_class) {
                  Pr_positive_baseline_bin[c, t_bin] = Phi(Xbeta_baseline[c, t]);
            }
            Se_baseline_bin[t_bin] = Pr_positive_baseline_bin[class_with_only_condition[test_condition[s]], t_bin];
            Fp_baseline_bin[t_bin] = Pr_positive_baseline_bin[class_with_neither_condition, t_bin];
            Sp_baseline_bin[t_bin] = 1.0 - Fp_baseline_bin[t_bin];
      }
      array[n_class, max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Pr_positive_baseline_ord;
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Se_baseline_ord;
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Sp_baseline_ord;
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Fp_baseline_ord;
      for (t_ord in 1:max(n_ordinal_tests, 1)) {
            Se_baseline_ord[t_ord] = rep_vector(-1.0, n_thr_max_gq);
            Sp_baseline_ord[t_ord] = rep_vector(-1.0, n_thr_max_gq);
            Fp_baseline_ord[t_ord] = rep_vector(-1.0, n_thr_max_gq);
            for (c in 1:n_class) {
                  Pr_positive_baseline_ord[c, t_ord] = rep_vector(-1.0, n_thr_max_gq);
            }
      }
      if (n_ordinal_tests > 0) {
            for (t_ord_orig in 1:n_ordinal_tests) {
                  int t = original_test_index_of_original_ordinal_test[t_ord_orig];
                  int s = inv_perm[t];
                  int n_thr_t = n_thr_per_ord_test_orig[t_ord_orig];
                  for (c in 1:n_class) {
                        vector[n_thr_t] C_c_t = get_test_values(
                              C_vec_orig[c],
                              orig_start_index,
                              orig_end_index,
                              t_ord_orig
                        );
                        for (k in 1:n_thr_t) {
                              Pr_positive_baseline_ord[c, t_ord_orig][k] = Phi(-(C_c_t[k] - Xbeta_baseline[c, t]));
                        }
                  }
                  for (k in 1:n_thr_t) {
                        Fp_baseline_ord[t_ord_orig][k] = Pr_positive_baseline_ord[class_with_neither_condition, t_ord_orig][k];
                        Sp_baseline_ord[t_ord_orig][k] = 1.0 - Fp_baseline_ord[t_ord_orig][k];
                        Se_baseline_ord[t_ord_orig][k] = Pr_positive_baseline_ord[class_with_only_condition[test_condition[s]], t_ord_orig][k];
                  }
            }
      }
      array[n_ordinal_tests > 0 ? n_class : 0] vector[n_total_C_per_class] C_vec_out;
      if (n_ordinal_tests > 0) {
            C_vec_out = C_vec_orig;
      }
      //// -------------------------------------------------------------------------------------
      //// Survey-averaged class-specific and overall ordinal Se/Sp (ORIGINAL test order)
      //// -------------------------------------------------------------------------------------
      array[n_class, max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Pr_positive_survey_averaged_ord;
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Se_without_the_other_condition_ord;
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Se_with_the_other_condition_ord;
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Sp_without_the_other_condition_ord;
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Sp_with_the_other_condition_ord;
      array[n_pops + 1, max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Se_overall_ord;
      array[n_pops + 1, max(n_ordinal_tests, 1)] vector[n_thr_max_gq] Sp_overall_ord;
      array[n_class, max(n_ordinal_tests, 1)] vector[n_chosen_cutoffs_max] z_pop_at_chosen_cutoffs;
      array[n_class, max(n_ordinal_tests, 1)] vector[n_chosen_cutoffs_max] Se_or_Sp_pop_at_chosen_cutoffs;
      //// the probit gap at each prior cutoff: probit Pr(positive) in the class WITH the other condition minus in the class WITHOUT it
      array[max(n_ordinal_tests, 1)] vector[n_chosen_cutoffs_max] gap_Se_at_chosen_cutoffs;
      array[max(n_ordinal_tests, 1)] vector[n_chosen_cutoffs_max] gap_Sp_at_chosen_cutoffs;
      //// the same gap at every threshold k (score >= k) of every ordinal test
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] gap_Se_ord;
      array[max(n_ordinal_tests, 1)] vector[n_thr_max_gq] gap_Sp_ord;
      for (t_ord in 1:max(n_ordinal_tests, 1)) {
            gap_Se_at_chosen_cutoffs[t_ord] = rep_vector(0.0, n_chosen_cutoffs_max);
            gap_Sp_at_chosen_cutoffs[t_ord] = rep_vector(0.0, n_chosen_cutoffs_max);
            gap_Se_ord[t_ord] = rep_vector(0.0, n_thr_max_gq);
            gap_Sp_ord[t_ord] = rep_vector(0.0, n_thr_max_gq);
            Se_without_the_other_condition_ord[t_ord] = rep_vector(-1.0, n_thr_max_gq);
            Se_with_the_other_condition_ord[t_ord] = rep_vector(-1.0, n_thr_max_gq);
            Sp_without_the_other_condition_ord[t_ord] = rep_vector(-1.0, n_thr_max_gq);
            Sp_with_the_other_condition_ord[t_ord] = rep_vector(-1.0, n_thr_max_gq);
            for (c in 1:n_class) {
                  Pr_positive_survey_averaged_ord[c, t_ord] = rep_vector(-1.0, n_thr_max_gq);
                  z_pop_at_chosen_cutoffs[c, t_ord] = rep_vector(0.0, n_chosen_cutoffs_max);
                  Se_or_Sp_pop_at_chosen_cutoffs[c, t_ord] = rep_vector(-1.0, n_chosen_cutoffs_max);
            }
            for (g in 1:(n_pops + 1)) {
                  Se_overall_ord[g, t_ord] = rep_vector(-1.0, n_thr_max_gq);
                  Sp_overall_ord[g, t_ord] = rep_vector(-1.0, n_thr_max_gq);
            }
      }
      if (n_ordinal_tests > 0) {
            matrix[n_standardisation, n_class] log_prev_standardisation = log_prevalence_by_subject(X_prevalence_standardisation, pop_standardisation,
                                                                                                    mu_logit_A, sigma_logit_A, z_logit_A, beta_prevalence_A,
                                                                                                    mu_logit_B, sigma_logit_B, z_logit_B, beta_prevalence_B,
                                                                                                    log_OR);
            array[n_pops + 1] vector[n_class] prev_of_each_pop_and_pooled;
            prev_of_each_pop_and_pooled[n_pops + 1] = rep_vector(
                  0.0,
                  n_class
            );
            for (g in 1:n_pops) {
                  prev_of_each_pop_and_pooled[g] = prev[g];
                  prev_of_each_pop_and_pooled[n_pops + 1] += (pop_weight[g] / sum(pop_weight)) * prev[g];
            }
            for (t_ord_orig in 1:n_ordinal_tests) {
                  int t = original_test_index_of_original_ordinal_test[t_ord_orig];
                  int s = inv_perm[t];
                  int s_ord = ord_idx_of_slot[s];
                  int n_thr_t = n_thr_per_ord_test_orig[t_ord_orig];
                  int condition_of_the_test = test_condition[s];
                  int other_condition = 3 - condition_of_the_test;
                  int class_with_neither = 0;
                  int class_with_only_the_condition = 0;
                  int class_with_only_the_other_condition = 0;
                  int class_with_both = 0;
                  //// the probit of Pr(positive) of each class at every threshold, from the log scale (no inv_Phi of a rounded 0 or 1)
                  array[n_class] vector[n_thr_max_gq] z_positive_every_k;
                  for (c in 1:n_class) {
                        z_positive_every_k[c] = rep_vector(0.0, n_thr_max_gq);
                  }
                  for (c in 1:n_class) {
                        int nc = n_covs_per_outcome[c, s];
                        vector[n_thr_t] C_c_t = get_test_values(
                              C_vec_orig[c],
                              orig_start_index,
                              orig_end_index,
                              t_ord_orig
                        );
                        vector[n_standardisation] Xbeta_prof = rep_vector(
                              0.0,
                              n_standardisation
                        );
                        if (nc > 0) {
                              Xbeta_prof = X_standardisation[c, s][, 1:nc] * to_vector(beta_fitted[c, s, 1:nc]);
                        }
                        vector[n_standardisation] log_w_prof = log_standardisation_w + log_prev_standardisation[, c];
                        log_w_prof -= log_sum_exp(log_w_prof);
                        vector[n_standardisation] w_prof = exp(log_w_prof);
                        for (k in 1:n_thr_t) {
                              Pr_positive_survey_averaged_ord[c, t_ord_orig][k] = dot_product(
                                    w_prof,
                                    Phi(Xbeta_prof - C_c_t[k])
                              );
                              real log_p_positive_k = log_sum_exp(log_w_prof + log_Phi_of_each_element(Xbeta_prof - C_c_t[k]));
                              real log_p_negative_k = log_sum_exp(log_w_prof + log_Phi_of_each_element(C_c_t[k] - Xbeta_prof));
                              z_positive_every_k[c][k] = inv_Phi_from_log_p(rep_vector(log_p_positive_k, 1),
                                                                            rep_vector(log_p_negative_k, 1))[1];
                        }
                        real sign_of_the_class = (class_D[c, condition_of_the_test] == 1) ? -1.0 : 1.0;
                        int n_chosen_ct = n_chosen_cutoffs[c, s_ord];
                        for (j in 1:n_chosen_ct) {
                              vector[n_standardisation] margin_prof = sign_of_the_class * (C_c_t[chosen_cutoffs[c, s_ord, j]] - Xbeta_prof);
                              real log_p_pop = log_sum_exp(log_w_prof + log_Phi_of_each_element(margin_prof));
                              real log_1m_p_pop = log_sum_exp(log_w_prof + log_Phi_of_each_element(-margin_prof));
                              z_pop_at_chosen_cutoffs[c, t_ord_orig][j] = inv_Phi_from_log_p(
                                    rep_vector(log_p_pop, 1),
                                    rep_vector(log_1m_p_pop, 1)
                              )[1];
                              Se_or_Sp_pop_at_chosen_cutoffs[c, t_ord_orig][j] = exp(log_p_pop);
                        }
                        if (class_D[c, condition_of_the_test] == 0 && class_D[c, other_condition] == 0) {
                              class_with_neither = c;
                        }
                        if (class_D[c, condition_of_the_test] == 1 && class_D[c, other_condition] == 0) {
                              class_with_only_the_condition = c;
                        }
                        if (class_D[c, condition_of_the_test] == 0 && class_D[c, other_condition] == 1) {
                              class_with_only_the_other_condition = c;
                        }
                        if (class_D[c, condition_of_the_test] == 1 && class_D[c, other_condition] == 1) {
                              class_with_both = c;
                        }
                  }
                  for (k in 1:n_thr_t) {
                        Se_without_the_other_condition_ord[t_ord_orig][k] = Pr_positive_survey_averaged_ord[class_with_only_the_condition, t_ord_orig][k];
                        Se_with_the_other_condition_ord[t_ord_orig][k] = Pr_positive_survey_averaged_ord[class_with_both, t_ord_orig][k];
                        Sp_without_the_other_condition_ord[t_ord_orig][k] = 1.0 - Pr_positive_survey_averaged_ord[class_with_neither, t_ord_orig][k];
                        Sp_with_the_other_condition_ord[t_ord_orig][k] = 1.0 - Pr_positive_survey_averaged_ord[class_with_only_the_other_condition, t_ord_orig][k];
                        gap_Se_ord[t_ord_orig][k] = z_positive_every_k[class_with_both][k] - z_positive_every_k[class_with_only_the_condition][k];
                        gap_Sp_ord[t_ord_orig][k] = z_positive_every_k[class_with_only_the_other_condition][k] - z_positive_every_k[class_with_neither][k];
                        for (g in 1:(n_pops + 1)) {
                              vector[n_class] prev_g = prev_of_each_pop_and_pooled[g];
                              Se_overall_ord[g, t_ord_orig][k] = (
                                    prev_g[class_with_only_the_condition] * Se_without_the_other_condition_ord[t_ord_orig][k]
                                    + prev_g[class_with_both] * Se_with_the_other_condition_ord[t_ord_orig][k]
                              ) / (prev_g[class_with_only_the_condition] + prev_g[class_with_both]);
                              Sp_overall_ord[g, t_ord_orig][k] = (
                                    prev_g[class_with_neither] * Sp_without_the_other_condition_ord[t_ord_orig][k]
                                    + prev_g[class_with_only_the_other_condition] * Sp_with_the_other_condition_ord[t_ord_orig][k]
                              ) / (prev_g[class_with_neither] + prev_g[class_with_only_the_other_condition]);
                        }
                  }
                  //// z_pop is the probit of Pr(positive) in the classes with the condition and of Pr(negative) in the classes without it
                  for (j in 1:n_chosen_cutoffs[class_with_neither, s_ord]) {
                        gap_Se_at_chosen_cutoffs[t_ord_orig][j] = z_pop_at_chosen_cutoffs[class_with_both, t_ord_orig][j] - z_pop_at_chosen_cutoffs[class_with_only_the_condition, t_ord_orig][j];
                        gap_Sp_at_chosen_cutoffs[t_ord_orig][j] = z_pop_at_chosen_cutoffs[class_with_neither, t_ord_orig][j] - z_pop_at_chosen_cutoffs[class_with_only_the_other_condition, t_ord_orig][j];
                  }
            }
      }
      //// -------------------------------------------------------------------------------------
      //// Diagnosed co-occurrence at reference clinical cutoffs (ORIGINAL tests 1 and 2)
      //// -------------------------------------------------------------------------------------
      real p_refA_pos_overall = 0.0;
      real p_refB_pos_overall = 0.0;
      real p_ref_both_pos_overall = 0.0;
      real Pr_diagnosed_B_given_diagnosed_A_overall = 0.0;
      real Pr_diagnosed_A_given_diagnosed_B_overall = 0.0;
      {
            matrix[n_standardisation, n_class] log_prev_standardisation = log_prevalence_by_subject(X_prevalence_standardisation, pop_standardisation,
                                                                                                    mu_logit_A, sigma_logit_A, z_logit_A, beta_prevalence_A,
                                                                                                    mu_logit_B, sigma_logit_B, z_logit_B, beta_prevalence_B,
                                                                                                    log_OR);
            int t_ref_A = reference_test[1];
            int t_ref_B = reference_test[2];
            int s_ref_A = inv_perm[t_ref_A];
            int s_ref_B = inv_perm[t_ref_B];
            matrix[n_standardisation, n_class] w_mat = rep_matrix(
                  standardisation_w,
                  n_class
            ) .* exp(log_prev_standardisation);
            real sum_A = 0.0;
            real sum_B = 0.0;
            real sum_AB = 0.0;
            matrix[n_standardisation, n_class] log_weight_A_pos;
            matrix[n_standardisation, n_class] Pr_B_pos_given_A_pos;
            matrix[n_standardisation, n_class] log_weight_B_pos;
            for (c in 1:n_class) {
                  real C_A_c = 0.0;
                  real C_B_c = 0.0;
                  if (test_is_binary[s_ref_A] == 0) {
                        int o_A = original_ordinal_slot_of_original_test[t_ref_A];
                        C_A_c = C_vec_orig[c][orig_start_index[o_A] + cutoff_category[s_ref_A] - 2];
                  }
                  if (test_is_binary[s_ref_B] == 0) {
                        int o_B = original_ordinal_slot_of_original_test[t_ref_B];
                        C_B_c = C_vec_orig[c][orig_start_index[o_B] + cutoff_category[s_ref_B] - 2];
                  }
                  int nc_A = n_covs_per_outcome[c, s_ref_A];
                  int nc_B = n_covs_per_outcome[c, s_ref_B];
                  vector[n_standardisation] Xb_A = (nc_A > 0)
                        ? (X_standardisation[c, s_ref_A][, 1:nc_A] * to_vector(beta_fitted[c, s_ref_A, 1:nc_A]))
                        : rep_vector(0.0, n_standardisation);
                  vector[n_standardisation] Xb_B = (nc_B > 0)
                        ? (X_standardisation[c, s_ref_B][, 1:nc_B] * to_vector(beta_fitted[c, s_ref_B, 1:nc_B]))
                        : rep_vector(0.0, n_standardisation);
                  vector[n_standardisation] a_vec = C_A_c - Xb_A;
                  vector[n_standardisation] b_vec = C_B_c - Xb_B;
                  vector[n_standardisation] w_c = w_mat[, c];
                  matrix[n_standardisation, 3] biv_res = biv_norm_log_Pr_Z1_above_a_and_Pr_Z2_given_Z1_above_a(
                        a_vec,
                        b_vec,
                        Omega_orig[c][t_ref_A, t_ref_B]
                  );
                  sum_A += dot_product(w_c, Phi(-a_vec));
                  sum_B += dot_product(w_c, Phi(-b_vec));
                  sum_AB += dot_product(w_c, exp(biv_res[, 1]) .* biv_res[, 2]);
                  log_weight_A_pos[, c] = log(w_c) + biv_res[, 1];
                  Pr_B_pos_given_A_pos[, c] = biv_res[, 2];
                  log_weight_B_pos[, c] = log(w_c) + log_Phi_of_each_element(-b_vec);
            }
            p_refA_pos_overall = sum_A;
            p_refB_pos_overall = sum_B;
            p_ref_both_pos_overall = sum_AB;
            matrix[n_standardisation, n_class] scaled_w_A = exp(log_weight_A_pos - max(log_weight_A_pos));
            Pr_diagnosed_B_given_diagnosed_A_overall = sum(scaled_w_A .* Pr_B_pos_given_A_pos) / sum(scaled_w_A);
            Pr_diagnosed_A_given_diagnosed_B_overall = sum(
                  exp(log_weight_A_pos + log(Pr_B_pos_given_A_pos) - max(log_weight_B_pos))
            ) / sum(exp(log_weight_B_pos - max(log_weight_B_pos)));
      }
}
