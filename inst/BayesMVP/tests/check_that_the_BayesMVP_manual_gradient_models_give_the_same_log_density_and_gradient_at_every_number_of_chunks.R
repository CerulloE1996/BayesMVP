##
## ==== check_that_the_BayesMVP_manual_gradient_models_give_the_same_log_density_and_gradient_at_every_number_
##      of_chunks.R =============================================================================================
##
## ---- (Paper 1) A check, not a timing: the log density and gradient of BayesMVP's manual-gradient C++ models
##      must be the same at every N_chunks. Compares N_chunks = 2, 3, 4, 8, 16, 32 and 64 with N_chunks = 1, for
##      the binary LC-MVP (N = 500, 6 tests) and the ordinal LC-MVOP (N = 1000, 3 tests), without and with
##      missing results, Phi_type 1 and 2, AVX and Stan::math, and every evaluation (as sampled, PartialLog,
##      autodiff), at the last burn-in state of 4 chains of one short fit and one tail point.
##      Passes: every scaled difference of the log density and gradient at most 1e-12.
##
## The model set-up is copied from check_missing_test_results_coded_y_minus_1_in_the_BayesMVP_manual_gradient_
## models.R.
##
## Reads:   the PS6 data functions (Alg_paper_analysis/1_appendix_pilot_studies/ps_6_basic_MCMC_settings_Stan/
##          functions) and the simulation functions (Alg_paper_analysis/0_utilities/shared_functions)
## Writes:  tests/outputs/same_log_density_and_gradient_at_every_number_of_chunks/
##            comparison_of_every_number_of_chunks_with_one_chunk_<machine>.csv
##            min_ESS_and_settings_of_the_short_fits_that_give_the_points_<machine>.rds (no nuisance values)
##
## Usage (lowest priority, on the CPUs for checks, under the hardware counters):
##   cd ~/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP/tests && \
##   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 190-191 chrt -i 0 nice -n 19 ionice -c 3 \
##       ~/Documents/Work/PhD_work/Alg_paper_analysis/paper_1_chunking_and_parallel_scalability/mechanism_study/\
##   pmc_stat <counters .csv> same_log_density_and_gradient_at_every_number_of_chunks -- \
##       ~/R/R-4.3.3/bin/Rscript --no-echo \
##       check_that_the_BayesMVP_manual_gradient_models_give_the_same_log_density_and_gradient_at_every_number_\
##   of_chunks.R
##
##
## ------- Settings:  --------------------------------------------------------------------------------------------
##
{
      output_folder <-  file.path("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP",
                                  "tests/outputs/same_log_density_and_gradient_at_every_number_of_chunks")
      ##
      numbers_of_chunks_compared_with_one_chunk <-  c(2, 3, 4, 8, 16, 32, 64)
      largest_relative_difference_allowed <-  1e-12
      ##
      proportion_of_results_set_missing <-  0.10
      seed_of_the_missing_results <-  20261016
      subject_with_every_test_missing <-  7
      intercepts_of_the_tail_point <-  8.5
      J_grad_option_of_the_checks <-  "autodiff"     ## the exact gradient of the correlation Jacobian
      ##
      ## (the short fit of each model that gives the points: PS7's settings, Phi_type 1)
      n_chains_of_the_fits <-  4
      n_burnin_of_the_fits <-  100
      n_iter_of_the_fits <-  100
      seed_of_the_fits <-  123
      sd_of_the_initial_nuisance_values <-  0.01
      learning_rate_of_the_fits <-  0.075            ## PS7's learning rate of the shortest burn-in (b125)
      seed_of_the_shuffle_of_the_points_check <-  1
}
##
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1", STAN_NUM_THREADS = "1")
options(scipen = 999, width = 220)
dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)
##
A_dir <-  "/home/enzocerullo/Documents/Work/PhD_work/Alg_paper_analysis"
##
suppressPackageStartupMessages(library(BayesMVP))   ## loads the shared library; every call below is namespaced
##
message(paste0("BayesMVP installed at ", system.file(package = "BayesMVP"), "; BayesMVP.so built ",
               format(file.mtime(file.path(system.file(package = "BayesMVP"), "libs", "BayesMVP.so")),
                      "%d %b %Y %H:%M")))
##
ps6_env <-  new.env()
sys.source(file.path(A_dir, "1_appendix_pilot_studies/ps_6_basic_MCMC_settings_Stan/functions",
                     "ps_6_MCMC_settings_Stan_functions.R"),
           envir = ps6_env)
sim_env <-  new.env()
invisible(utils::capture.output(
      sys.source(file.path(A_dir, "0_utilities/shared_functions/R_fn_sim_bin_COVID_19_LC_MVP_data.R"),
                 envir = sim_env)))
invisible(utils::capture.output(
      sys.source(file.path(A_dir, "0_utilities/shared_functions/R_fn_sim_mixed_bin_ord_LC_MVOP_data.R"),
                 envir = sim_env)))
##
## ---- The vector type of the SIMD functions on this machine, and the scalar Stan::math functions:
##
vectorisation_support <-  BayesMVP:::detect_vectorization_support()
SIMD_vect_type <-  if (grepl("AVX512", vectorisation_support)) "AVX512" else "AVX2"
machine <-  if (SIMD_vect_type == "AVX512") "local_HPC" else "laptop"
vect_types_to_check <-  c(SIMD_vect_type, "Stan")
message(paste0("vectorisation support: ", vectorisation_support, "; vect_types checked: ",
               paste(vect_types_to_check, collapse = ", "), "; machine: ", machine))
##
##
## ---- Stan data (the data and priors of the PS6 runners) of each model family: --------------------------------
##
fn_Stan_data_of_model_family <-  function(model_family) {

      if (model_family == "binary") {
            N <-  500
            sim <-  NULL
            invisible(utils::capture.output(
                  sim <-  sim_env$R_fn_simulate_binary_LC_MVP_data_COVID_19(study_type = "algorithm_benchmarking",
                                                                            seed = 123,
                                                                            corr_force_positive_DGM = FALSE)))
            n_tests <-  ncol(sim$y_list[[1]])
            prior_beta_mean <-  list(matrix(-2.0, 1, n_tests),
                                     matrix(c(1.5, 1.5, rep(1.0, n_tests - 2)), 1, n_tests))
            prior_beta_sd   <-  list(matrix(0.5, 1, n_tests), matrix(0.5, 1, n_tests))
            Stan_data_all_N <-  NULL
            invisible(utils::capture.output(
                  Stan_data_all_N <-  ps6_env$R_fn_prep_Stan_binary_data( binary_sim_outs = sim,
                                                                          prior_beta_mean = prior_beta_mean,
                                                                          prior_beta_sd = prior_beta_sd,
                                                                          prior_LKJ = c(4, 4),
                                                                          prior_prev_a = 2.5,
                                                                          prior_prev_b = 10,
                                                                          n_covariates_max = 1,
                                                                          n_class = 2,
                                                                          pool_populations = FALSE,
                                                                          fully_vectorised = 1,
                                                                          handle_numerical_issues = 1,
                                                                          Phi_type = 1,
                                                                          prior_only = 0,
                                                                          corr_force_positive = 0,
                                                                          overflow_threshold = +7.5,
                                                                          underflow_threshold = -7.5)))
            Stan_data <-  Stan_data_all_N[[which(sim$N_vec == N)]]
            return(list(Stan_data = Stan_data, N = N, n_tests = n_tests, n_cov = 1, cpp_model_type = "LC_MVP"))
      }
      ##
      if (identical(model_family, "CLC_MVOP")) {
            set.seed(123L)
            N <- 500L
            n_tests <- 4L
            n_class <- 4L
            n_pops <- 3L
            n_cov <- 2L
            n_cat_per_test <- c(2L, 2L, 4L, 5L)
            n_cat_per_ord_test <- c(4L, 5L)
            n_thr_per_ord_test <- n_cat_per_ord_test - 1L
            pop <- rep(
                  seq_len(n_pops),
                  length.out = N
            )
            pop_weight <- rep(
                  1.0 / n_pops,
                  n_pops
            )
            class_D <- matrix(
                  c(0L, 0L,
                    1L, 0L,
                    0L, 1L,
                    1L, 1L),
                  nrow = n_class,
                  ncol = 2L,
                  byrow = TRUE
            )
            test_condition <- c(1L, 2L, 1L, 2L)
            shared_covariate_matrix <- cbind(
                  rep(
                        1.0,
                        N
                  ),
                  stats::rnorm(
                        N,
                        0.0,
                        1.0
                  )
            )
            X_array <- array(
                  0.0,
                  dim = c(n_class, n_tests, N, n_cov)
            )
            for (class_index in seq_len(n_class)) {
                  for (test_index in seq_len(n_tests)) {
                        X_array[class_index, test_index, , ] <- shared_covariate_matrix
                  }
            }
            n_covs_per_outcome <- matrix(
                  n_cov,
                  nrow = n_class,
                  ncol = n_tests
            )
            X_prevalence <- matrix(
                  stats::rnorm(
                        N,
                        0.0,
                        1.0
                  ),
                  nrow = N,
                  ncol = 1L
            )
            latent_class_vector <- sample(
                  seq_len(n_class),
                  size = N,
                  replace = TRUE,
                  prob = c(0.75, 0.10, 0.10, 0.05)
            )
            simulated_y_matrix <- matrix(
                  0L,
                  nrow = N,
                  ncol = n_tests
            )
            for (subject_index in seq_len(N)) {
                  latent_class_index <- latent_class_vector[subject_index]
                  for (test_index in seq_len(n_tests)) {
                        own_condition_status <- class_D[latent_class_index, test_condition[test_index]]
                        other_condition_status <- class_D[latent_class_index, 3L - test_condition[test_index]]
                        latent_response_value <- (if (own_condition_status == 1L) 1.4 else -1.2) +
                              0.25 * other_condition_status +
                              0.20 * shared_covariate_matrix[subject_index, 2L] +
                              stats::rnorm(1L)
                        if (n_cat_per_test[test_index] == 2L) {
                              simulated_y_matrix[subject_index, test_index] <- if (latent_response_value > 0.0) 1L else 0L
                        } else {
                              ordinal_cutpoints <- seq(
                                    -1.0,
                                    1.5,
                                    length.out = n_cat_per_test[test_index] - 1L
                              )
                              simulated_y_matrix[subject_index, test_index] <- as.integer(
                                    1L + sum(latent_response_value > ordinal_cutpoints)
                              )
                        }
                  }
            }
            storage.mode(simulated_y_matrix) <- "integer"
            n_chosen_cutoffs <- matrix(
                  1L,
                  nrow = n_class,
                  ncol = 2L
            )
            chosen_cutoffs <- array(
                  2L,
                  dim = c(n_class, 2L, 1L)
            )
            prior_Se_Sp_link_mean_SD <- array(
                  0.0,
                  dim = c(n_class, 2L, 1L, 2L)
            )
            for (class_index in seq_len(n_class)) {
                  for (ord_index in seq_len(2L)) {
                        own_st <- class_D[class_index, test_condition[2L + ord_index]]
                        prior_Se_Sp_link_mean_SD[class_index, ord_index, 1L, 1L] <- if (own_st == 1L) 0.8 else 1.1
                        prior_Se_Sp_link_mean_SD[class_index, ord_index, 1L, 2L] <- 0.5
                  }
            }
            prior_Se_Sp_dirichlet_A <- array(
                  1.0,
                  dim = c(n_class, 2L, 2L)
            )
            prior_Se_Sp_beta <- array(
                  2.0,
                  dim = c(n_class, 2L, 1L, 2L)
            )
            prior_beta_mean <- array(
                  0.0,
                  dim = c(n_class, n_cov, n_tests)
            )
            prior_beta_sd <- array(
                  0.75,
                  dim = c(n_class, n_cov, n_tests)
            )
            for (class_index in seq_len(n_class)) {
                  for (test_index in seq_len(n_tests)) {
                        own_st <- class_D[class_index, test_condition[test_index]]
                        if (n_cat_per_test[test_index] == 2L) {
                              prior_beta_mean[class_index, 1L, test_index] <- if (own_st == 1L) 1.2 else -1.2
                        }
                  }
            }
            prior_dirichlet_alpha <- array(
                  1.0,
                  dim = c(n_class, max(n_cat_per_ord_test), 2L)
            )
            baseline_case <- array(
                  0.0,
                  dim = c(n_class, n_tests, n_cov)
            )
            baseline_case[, , 1L] <- 1.0
            n_standardisation <- 40L
            standardisation_index <- seq_len(n_standardisation)
            standardisation_weight <- rep(
                  1.0 / n_standardisation,
                  n_standardisation
            )
            known_values_list <- array(
                  0.0,
                  dim = c(n_class, n_tests, n_tests)
            )
            known_values_indicator_list <- array(
                  0L,
                  dim = c(n_class, n_tests, n_tests)
            )
            lb_corr <- array(
                  -1.0,
                  dim = c(n_class, n_tests, n_tests)
            )
            ub_corr <- array(
                  1.0,
                  dim = c(n_class, n_tests, n_tests)
            )
            Stan_data <- list(
                  N = N,
                  n_tests = n_tests,
                  n_class = n_class,
                  n_pops = n_pops,
                  pop = pop,
                  pop_weight = pop_weight,
                  y = simulated_y_matrix,
                  n_cat_per_test = n_cat_per_test,
                  n_cat_per_ord_test = n_cat_per_ord_test,
                  n_thr_per_ord_test = n_thr_per_ord_test,
                  class_D = class_D,
                  test_condition = test_condition,
                  is_perfect_ref = rep(
                        0L,
                        n_tests
                  ),
                  perfect_ref_M = 8.0,
                  n_covariates_max = n_cov,
                  X = X_array,
                  n_covs_per_outcome = n_covs_per_outcome,
                  n_prevalence_covariates = 1L,
                  X_prevalence = X_prevalence,
                  prior_beta_prevalence_A_mean = array(
                        0.0,
                        dim = 1L
                  ),
                  prior_beta_prevalence_A_sd = array(
                        0.75,
                        dim = 1L
                  ),
                  prior_beta_prevalence_B_mean = array(
                        0.0,
                        dim = 1L
                  ),
                  prior_beta_prevalence_B_sd = array(
                        0.75,
                        dim = 1L
                  ),
                  pool_correlations_across_classes = 1L,
                  prior_LKJ_global = 2.0,
                  prior_LKJ_departure = 4.0,
                  prior_lambda_shape_a = 2.0,
                  prior_lambda_shape_b = 5.0,
                  known_values_list = known_values_list,
                  known_values_indicator_list = known_values_indicator_list,
                  lb_corr = lb_corr,
                  ub_corr = ub_corr,
                  use_pop_prev_prior = 1L,
                  prior_mu_logit_A_mean = -2.0,
                  prior_mu_logit_A_sd = 0.5,
                  prior_mu_logit_B_mean = -2.2,
                  prior_mu_logit_B_sd = 0.5,
                  prior_sigma_logit_A_sd = 0.75,
                  prior_sigma_logit_B_sd = 0.75,
                  prior_logit_prob_B_given_A_mean = -1.5,
                  prior_logit_prob_B_given_A_sd = 0.8,
                  baseline_case = baseline_case,
                  use_Se_Sp_prior = 1L,
                  Se_Sp_prior_type = 4L,
                  n_chosen_cutoffs_max = 1L,
                  n_chosen_cutoffs = n_chosen_cutoffs,
                  chosen_cutoffs = chosen_cutoffs,
                  prior_Se_Sp_dirichlet_A = prior_Se_Sp_dirichlet_A,
                  prior_Se_Sp_beta = prior_Se_Sp_beta,
                  prior_Se_Sp_link_mean_SD = prior_Se_Sp_link_mean_SD,
                  use_pop_Se_Sp_prior = 1L,
                  n_standardisation = n_standardisation,
                  standardisation_index = standardisation_index,
                  standardisation_weight = standardisation_weight,
                  use_label_constraint = 0L,
                  reference_test = c(1L, 2L),
                  cutoff_category = c(2L, 2L, 2L, 2L),
                  prior_dirichlet_alpha = prior_dirichlet_alpha,
                  prior_beta_mean = prior_beta_mean,
                  prior_beta_sd = prior_beta_sd,
                  prior_LKJ = matrix(
                        rep(
                              4.0,
                              n_class
                        ),
                        ncol = 1L
                  ),
                  C_raw_lower = -10.0,
                  C_raw_upper = 10.0,
                  overflow_threshold = 7.5,
                  underflow_threshold = -7.5
            )
            return(list(
                  Stan_data = Stan_data,
                  N = N,
                  n_tests = n_tests,
                  n_cov = n_cov,
                  cpp_model_type = "CLC_MVOP"
            ))
      }
      N <-  1000
      sim <-  NULL
      invisible(utils::capture.output(
            sim <-  sim_env$R_fn_simulate_mixed_binary_ordinal_LC_MVOP_data(study_type = "algorithm_benchmarking",
                                                                            seed = 123,
                                                                            grouping = FALSE,
                                                                            corr_force_positive_DGM = FALSE,
                                                                            dirichlet_prior_conc = 50)))
      n_tests <-  3
      prior_beta_mean <-  list()
      prior_beta_sd <-  list()
      for (class_index in 1:2) {
            prior_beta_mean[[class_index]] <-  matrix(0.0, 3, n_tests)
            prior_beta_sd[[class_index]] <-  matrix(0.5, 3, n_tests)
      }
      prior_beta_mean[[1]][1, 1] <-  -1.5
      prior_beta_mean[[2]][1, 1] <-  +1.0
      prior_beta_sd[[1]][1, 1] <-  0.25
      prior_beta_sd[[2]][1, 1] <-  0.25
      prior_beta_sd[[1]][1, 2:3] <-  1.0
      prior_beta_sd[[2]][1, 2:3] <-  1.0
      prior_beta_sd[[1]][2:3, 1] <-  0.5
      prior_beta_sd[[2]][2:3, 1] <-  0.5
      Stan_data_all_N <-  NULL
      invisible(utils::capture.output(
            Stan_data_all_N <-  ps6_env$R_fn_prep_Stan_ordinal_data(
                                          ord_sim_outs = sim,
                                          n_covariates_max = 3,
                                          fully_vectorised = 1,
                                          handle_numerical_issues = 1,
                                          prior_beta_mean = prior_beta_mean,
                                          prior_beta_sd = prior_beta_sd,
                                          prior_LKJ = matrix(c(4, 4), ncol = 1),
                                          prior_prev_a = matrix(rep(1.5, 3), ncol = 1),
                                          prior_prev_b = matrix(rep(10.0, 3), ncol = 1),
                                          overflow_threshold = +7.5,
                                          underflow_threshold = -7.5)))
      Stan_data <-  Stan_data_all_N[[which(sim$N_vec == N)]]
      return(list(Stan_data = Stan_data, N = N, n_tests = n_tests, n_cov = 3, cpp_model_type = "LC_MVOP"))

}
##
## ---- y with missing results (fixed seed), and every test of one subject:
##
fn_y_with_missing_results <-  function(y) {

      set.seed(seed_of_the_missing_results)
      y_with_missing_results <-  y
      n_results <-  length(y)
      results_set_missing <-  sample(seq_len(n_results), round(proportion_of_results_set_missing * n_results))
      y_with_missing_results[results_set_missing] <-  -1
      y_with_missing_results[subject_with_every_test_missing, ] <-  -1
      storage.mode(y_with_missing_results) <-  "integer"
      return(y_with_missing_results)

}
##
## ---- The BayesMVP C++ model built from a Stan data list, a y, a vector type, a Phi_type and a number of chunks:
##
fn_matrix_of_class <-  function(x,
                                c) {
      if (is.list(x)) return(as.matrix(x[[c]]))
      return(matrix(x[c, , ], nrow = dim(x)[2]))
}
##
fn_make_cpp_model <-  function(family_data,
                               model_family,
                               y,
                               vect_type,
                               Phi_type,
                               num_chunks) {

      Stan_data <-  family_data$Stan_data
      n_tests <-  family_data$n_tests
      storage.mode(y) <-  "integer"
      if (identical(model_family, "CLC_MVOP")) {
            model_args_list <- list(
                  y = y,
                  N = family_data$N,
                  n_tests = n_tests,
                  n_class = 4L,
                  n_pops = as.integer(Stan_data$n_pops),
                  pop = as.integer(Stan_data$pop),
                  pop_weight = Stan_data$pop_weight,
                  n_cat_per_ord_test = as.integer(Stan_data$n_cat_per_ord_test),
                  class_D = Stan_data$class_D,
                  test_condition = as.integer(Stan_data$test_condition),
                  is_perfect_ref = as.integer(Stan_data$is_perfect_ref),
                  perfect_ref_M = Stan_data$perfect_ref_M,
                  n_covariates_max = as.integer(Stan_data$n_covariates_max),
                  X = Stan_data$X,
                  n_covs_per_outcome = Stan_data$n_covs_per_outcome,
                  n_prevalence_covariates = as.integer(Stan_data$n_prevalence_covariates),
                  X_prevalence = Stan_data$X_prevalence,
                  prior_beta_prevalence_A_mean = Stan_data$prior_beta_prevalence_A_mean,
                  prior_beta_prevalence_A_sd = Stan_data$prior_beta_prevalence_A_sd,
                  prior_beta_prevalence_B_mean = Stan_data$prior_beta_prevalence_B_mean,
                  prior_beta_prevalence_B_sd = Stan_data$prior_beta_prevalence_B_sd,
                  pool_correlations_across_classes = as.integer(Stan_data$pool_correlations_across_classes),
                  prior_LKJ_global = Stan_data$prior_LKJ_global,
                  prior_LKJ_departure = Stan_data$prior_LKJ_departure,
                  prior_lambda_shape_a = Stan_data$prior_lambda_shape_a,
                  prior_lambda_shape_b = Stan_data$prior_lambda_shape_b,
                  known_values_list = Stan_data$known_values_list,
                  known_values_indicator_list = Stan_data$known_values_indicator_list,
                  lb_corr = Stan_data$lb_corr,
                  ub_corr = Stan_data$ub_corr,
                  use_pop_prev_prior = as.integer(Stan_data$use_pop_prev_prior),
                  prior_mu_logit_A_mean = Stan_data$prior_mu_logit_A_mean,
                  prior_mu_logit_A_sd = Stan_data$prior_mu_logit_A_sd,
                  prior_mu_logit_B_mean = Stan_data$prior_mu_logit_B_mean,
                  prior_mu_logit_B_sd = Stan_data$prior_mu_logit_B_sd,
                  prior_sigma_logit_A_sd = Stan_data$prior_sigma_logit_A_sd,
                  prior_sigma_logit_B_sd = Stan_data$prior_sigma_logit_B_sd,
                  prior_logit_prob_B_given_A_mean = Stan_data$prior_logit_prob_B_given_A_mean,
                  prior_logit_prob_B_given_A_sd = Stan_data$prior_logit_prob_B_given_A_sd,
                  baseline_case = Stan_data$baseline_case,
                  use_Se_Sp_prior = as.integer(Stan_data$use_Se_Sp_prior),
                  Se_Sp_prior_type = as.integer(Stan_data$Se_Sp_prior_type),
                  n_chosen_cutoffs_max = as.integer(Stan_data$n_chosen_cutoffs_max),
                  n_chosen_cutoffs = Stan_data$n_chosen_cutoffs,
                  chosen_cutoffs = Stan_data$chosen_cutoffs,
                  prior_Se_Sp_dirichlet_A = Stan_data$prior_Se_Sp_dirichlet_A,
                  prior_Se_Sp_beta = Stan_data$prior_Se_Sp_beta,
                  prior_Se_Sp_link_mean_SD = Stan_data$prior_Se_Sp_link_mean_SD,
                  use_pop_Se_Sp_prior = as.integer(Stan_data$use_pop_Se_Sp_prior),
                  n_standardisation = as.integer(Stan_data$n_standardisation),
                  standardisation_index = as.integer(Stan_data$standardisation_index),
                  standardisation_weight = Stan_data$standardisation_weight,
                  use_label_constraint = as.integer(Stan_data$use_label_constraint),
                  reference_test = as.integer(Stan_data$reference_test),
                  cutoff_category = as.integer(Stan_data$cutoff_category),
                  prior_dirichlet_alpha = Stan_data$prior_dirichlet_alpha,
                  prior_beta_mean = Stan_data$prior_beta_mean,
                  prior_beta_sd = Stan_data$prior_beta_sd,
                  prior_LKJ = Stan_data$prior_LKJ,
                  corr_force_positive = FALSE,
                  C_raw_lower = Stan_data$C_raw_lower,
                  C_raw_upper = Stan_data$C_raw_upper,
                  overflow_threshold = Stan_data$overflow_threshold,
                  underflow_threshold = Stan_data$underflow_threshold,
                  num_chunks = num_chunks,
                  vect_type = vect_type,
                  Phi_type = if (Phi_type == 2) "Phi_approx" else "Phi",
                  inv_Phi_type = if (Phi_type == 2) "inv_Phi_approx" else "inv_Phi"
            )
            final_init <- NULL
            invisible(utils::capture.output(
                  final_init <- BayesMVP::init_hard_coded_model(
                        Model_type = family_data$cpp_model_type,
                        model_args_list = model_args_list
                  )
            ))
            y_final <- final_init$model_args_list$y
            storage.mode(y_final) <- "integer"
            stopifnot(identical(as.integer(y_final), as.integer(y)))
            return(list(
                  Model_args_as_Rcpp_List = final_init$Model_args_as_Rcpp_List,
                  model_args_list = final_init$model_args_list,
                  y = y_final,
                  N = family_data$N,
                  n_tests = n_tests,
                  num_chunks = num_chunks,
                  cpp_model_type = family_data$cpp_model_type,
                  vect_type = vect_type
            ))
      }
      n_class <-  2
      model_args_list <-  list(y = y, n_tests = n_tests, n_class = n_class, N = family_data$N,
                               vect_type = vect_type)
      if (model_family == "ordinal") {
            model_args_list$X <-  list(Stan_data$X_nd, Stan_data$X_d)
            model_args_list$n_cat_per_ord_test <-  as.integer(Stan_data$n_cat_per_ord_test)
      }
      first_init <-  NULL
      invisible(utils::capture.output(
            first_init <-  BayesMVP::init_hard_coded_model(Model_type = family_data$cpp_model_type,
                                                           model_args_list = model_args_list)))
      model_args_list <-  first_init$model_args_list
      model_args_list$n_pops <-  as.integer(Stan_data$n_pops)
      model_args_list$pop <-  as.integer(Stan_data$pop)
      if (model_family == "binary") {
            model_args_list$prior_prev_a <-  as.numeric(Stan_data$prior_p_alpha)
            model_args_list$prior_prev_b <-  as.numeric(Stan_data$prior_p_beta)
      } else {
            model_args_list$prior_prev_a <-  as.numeric(Stan_data$prior_prev_a)
            model_args_list$prior_prev_b <-  as.numeric(Stan_data$prior_prev_b)
            model_args_list$prior_dirichlet_alpha <-  lapply(1:n_class, function(class_index) {
                  as.matrix(Stan_data$prior_dirichlet_alpha[[class_index]])
            })
            model_args_list$C_raw_lower <-  Stan_data$C_raw_lower
            model_args_list$C_raw_upper <-  Stan_data$C_raw_upper
      }
      model_args_list$prior_coeffs_mean_mat <-  lapply(1:n_class, function(class_index) {
            fn_matrix_of_class(Stan_data$prior_beta_mean, class_index)
      })
      model_args_list$prior_coeffs_sd_mat   <-  lapply(1:n_class, function(class_index) {
            fn_matrix_of_class(Stan_data$prior_beta_sd, class_index)
      })
      model_args_list$lkj_cholesky_eta <-  matrix(as.numeric(Stan_data$prior_LKJ), ncol = 1)
      model_args_list$prior_LKJ <-  model_args_list$lkj_cholesky_eta
      model_args_list$corr_force_positive <-  FALSE
      model_args_list$lb_corr <-  list(matrix(-1.0, n_tests, n_tests), matrix(-1.0, n_tests, n_tests))
      model_args_list$ub_corr <-  list(matrix(+1.0, n_tests, n_tests), matrix(+1.0, n_tests, n_tests))
      model_args_list$overflow_threshold  <-  Stan_data$overflow_threshold
      model_args_list$underflow_threshold <-  Stan_data$underflow_threshold
      model_args_list$num_chunks <-  num_chunks
      model_args_list$J_grad_option <-  J_grad_option_of_the_checks
      model_args_list$vect_type <-  vect_type
      model_args_list$Phi_type <-  if (Phi_type == 2) "Phi_approx" else "Phi"
      model_args_list$inv_Phi_type <-  if (Phi_type == 2) "inv_Phi_approx" else "inv_Phi"
      final_init <-  NULL
      invisible(utils::capture.output(
            final_init <-  BayesMVP::init_hard_coded_model(Model_type = family_data$cpp_model_type,
                                                           model_args_list = model_args_list)))
      y_final <-  final_init$model_args_list$y
      storage.mode(y_final) <-  "integer"
      stopifnot(identical(as.integer(y_final), as.integer(y)))
      return(list(Model_args_as_Rcpp_List = final_init$Model_args_as_Rcpp_List,
                  model_args_list = model_args_list,
                  y = y_final,
                  N = family_data$N,
                  n_tests = n_tests,
                  num_chunks = num_chunks,
                  cpp_model_type = family_data$cpp_model_type,
                  vect_type = vect_type))

}
##
## ---- The C++ nuisance layout: the rows of each chunk; NULL when a chunk has fewer subjects than the SIMD width:
##
fn_vec_size_of_vect_type <-  function(vect_type) {
      if (vect_type == "AVX512") return(8)
      if (vect_type == "AVX2") return(4)
      if (vect_type == "AVX") return(2)
      return(1)
}
##
fn_chunk_rows <-  function(vect_type,
                           num_chunks,
                           N) {
      vec_size <-  fn_vec_size_of_vect_type(vect_type)
      normal_chunk_size <-  vec_size * (N %/% (vec_size * max(num_chunks, 1)))
      if (normal_chunk_size < vec_size) return(NULL)
      n_full_chunks <-  N %/% normal_chunk_size
      last_chunk_size <-  N - n_full_chunks * normal_chunk_size
      starts <-  (0:(n_full_chunks - 1)) * normal_chunk_size
      sizes <-  rep(normal_chunk_size, n_full_chunks)
      if (last_chunk_size > 0) {
            starts <-  c(starts, n_full_chunks * normal_chunk_size)
            sizes <-  c(sizes, last_chunk_size)
      }
      return(list(starts = starts, sizes = sizes))
}
##
fn_nuisance_matrix_from_vector_in_chunk_layout <-  function(theta_us,
                                                            chunk_rows,
                                                            N,
                                                            n_tests) {
      u_matrix <-  matrix(NA_real_, N, n_tests)
      offset <-  0
      for (chunk in seq_along(chunk_rows$starts)) {
            rows_of_chunk <-  chunk_rows$starts[chunk] + seq_len(chunk_rows$sizes[chunk])
            n_values <-  length(rows_of_chunk) * n_tests
            u_matrix[rows_of_chunk, ] <-  matrix(theta_us[offset + seq_len(n_values)],
                                                 nrow = length(rows_of_chunk))
            offset <-  offset + n_values
      }
      return(u_matrix)
}
##
fn_nuisance_vector_in_chunk_layout_from_matrix <-  function(u_matrix,
                                                            chunk_rows) {
      return(unlist(lapply(seq_along(chunk_rows$starts), function(chunk) {
            as.vector(u_matrix[chunk_rows$starts[chunk] + seq_len(chunk_rows$sizes[chunk]), , drop = FALSE])
      })))
}
##
## ---- The C++ log density and gradient; evaluation = "as sampled", "PartialLog" or "autodiff":
##
fn_cpp_lp_grad <-  function(cpp_model,
                            theta_main,
                            u_unc_matrix,
                            evaluation) {
      N <-  cpp_model$N
      n_tests <-  cpp_model$n_tests
      chunk_rows <-  fn_chunk_rows(cpp_model$vect_type, cpp_model$num_chunks, N)
      theta_us <-  fn_nuisance_vector_in_chunk_layout_from_matrix(u_unc_matrix, chunk_rows)
      lp_grad_output <-  BayesMVP:::Rcpp_wrapper_fn_lp_grad(
                                   Model_type = cpp_model$cpp_model_type,
                                   force_autodiff = (evaluation == "autodiff"),
                                   force_PartialLog = (evaluation == "PartialLog"),
                                   multi_attempts = TRUE,
                                   theta_main_vec = theta_main,
                                   theta_us_vec = theta_us,
                                   y = cpp_model$y,
                                   grad_option = "all",
                                   Model_args_as_Rcpp_List = cpp_model$Model_args_as_Rcpp_List)
      n_us <-  length(theta_us)
      grad_us <-  fn_nuisance_matrix_from_vector_in_chunk_layout(lp_grad_output[1 + seq_len(n_us)], chunk_rows,
                                                                 N, n_tests)
      return(list(lp = lp_grad_output[1],
                  grad_main = lp_grad_output[1 + n_us + seq_along(theta_main)],
                  grad_us = grad_us))
}
##
##
## ---- The points: the last burn-in state of each chain of one short BayesMVP fit of each model (PS7's settings,
##      Phi_type 1, one chunk; tests in the data's own order). ------------------------------------------------
##
fn_initial_values_of_each_chain <-  function(family_data,
                                             model_family) {

      Stan_data <-  family_data$Stan_data
      N <-  family_data$N
      n_tests <-  family_data$n_tests
      n_pops <-  as.integer(Stan_data$n_pops)
      n_corrs <-  n_tests * (n_tests - 1) / 2
      ## (the nuisance values drawn here in R, one L'Ecuyer-CMRG stream per chain, with a small SD)
      RNGkind("L'Ecuyer-CMRG")
      set.seed(seed_of_the_fits)
      stream_of_the_chain <-  .Random.seed
      initial_values_of_each_chain <-  list()
      for (chain in seq_len(n_chains_of_the_fits)) {
            assign(".Random.seed", stream_of_the_chain, envir = .GlobalEnv)
            if (identical(model_family, "CLC_MVOP")) {
                  n_class <- 4L
                  n_cov_total <- as.integer(sum(Stan_data$n_covs_per_outcome))
                  C_raw <- c()
                  for (ordinal_slot_index in seq_along(Stan_data$n_thr_per_ord_test)) {
                        n_thresholds_for_test <- Stan_data$n_thr_per_ord_test[ordinal_slot_index]
                        C_raw <- c(
                              C_raw,
                              -1.5,
                              rep(
                                    log(3.0 / n_thresholds_for_test),
                                    n_thresholds_for_test - 1L
                              )
                        )
                  }
                  C_unc <- atanh(2.0 * (C_raw - Stan_data$C_raw_lower) /
                                 (Stan_data$C_raw_upper - Stan_data$C_raw_lower) - 1.0)
                  initial_values <- list(
                        u_raw = matrix(
                              stats::rnorm(
                                    n = N * n_tests,
                                    mean = 0.0,
                                    sd = sd_of_the_initial_nuisance_values
                              ),
                              nrow = N,
                              ncol = n_tests
                        ),
                        Omega_unconstrained_vec = rep(
                              0.001,
                              n_class * n_corrs
                        ),
                        Omega_global_unconstrained_vec = rep(
                              0.001,
                              n_corrs
                        ),
                        lambda_class = rep(
                              0.25,
                              n_class
                        ),
                        beta_vec = rep(
                              0.0,
                              n_cov_total
                        ),
                        beta_prevalence_A = array(
                              0.0,
                              dim = ncol(Stan_data$X_prevalence)
                        ),
                        beta_prevalence_B = array(
                              0.0,
                              dim = ncol(Stan_data$X_prevalence)
                        ),
                        z_logit_A = rep(
                              0.0,
                              n_pops
                        ),
                        z_logit_B = rep(
                              0.0,
                              n_pops
                        ),
                        mu_logit_A = -2.0,
                        mu_logit_B = -1.8,
                        sigma_logit_A = 0.35,
                        sigma_logit_B = 0.35,
                        log_OR = 1.0,
                        C_unc_vec = lapply(
                              seq_len(n_class),
                              function(class_index) {
                                    return(C_unc)
                              }
                        )
                  )
            } else {
                  initial_values <-  list(u_raw = matrix(stats::rnorm(n = N * n_tests, mean = 0.0,
                                                                      sd = sd_of_the_initial_nuisance_values),
                                                         nrow = N, ncol = n_tests),
                                          Omega_unconstrained_vec = list(rep(0.001, n_corrs), rep(0.001, n_corrs)),
                                          beta = lapply(1:2, function(class_index) {
                                                fn_matrix_of_class(Stan_data$prior_beta_mean, class_index)
                                          }),
                                          p_raw = array(rep(atanh(2 * 0.10 - 1), n_pops)))
                  if (model_family == "ordinal") {
                        C_raw <-  c()
                        for (k_slot in seq_along(Stan_data$n_thr_per_ord_test)) {
                              n_thr <-  Stan_data$n_thr_per_ord_test[k_slot]
                              C_raw <-  c(C_raw, -2.0, rep(log(4.0 / n_thr), n_thr - 1))
                        }
                        C_unc <-  atanh(2 * (C_raw - Stan_data$C_raw_lower) /
                                            (Stan_data$C_raw_upper - Stan_data$C_raw_lower) - 1)
                        initial_values$C_unc_vec <-  list(C_unc, C_unc)
                  }
            }
            initial_values_of_each_chain[[chain]] <-  initial_values
            stream_of_the_chain <-  parallel::nextRNGStream(stream_of_the_chain)
      }
      RNGkind("default")
      return(initial_values_of_each_chain)

}
##
fn_short_fit_and_its_points <-  function(family_data,
                                         model_family) {

      cpp_model_of_the_fit <-  fn_make_cpp_model(family_data, model_family, y = family_data$Stan_data$y,
                                                 vect_type = "Stan", Phi_type = 1, num_chunks = 1)
      n_adapt <-  n_burnin_of_the_fits - round(n_burnin_of_the_fits / 10)
      settings_of_the_fit <-  list(
            Model_type = family_data$cpp_model_type, N = family_data$N, n_tests = family_data$n_tests,
            vect_type = "Stan", Phi_type = "Phi", inv_Phi_type = "inv_Phi", num_chunks = 1,
            J_grad_option = J_grad_option_of_the_checks,
            n_chains_burnin = n_chains_of_the_fits, n_chains_sampling = n_chains_of_the_fits,
            n_superchains = n_chains_of_the_fits,
            n_threads_WCP_burnin = 1, num_chunks_burnin = 1, n_threads_WCP_sampling = 1, num_chunks_sampling = 1,
            n_burnin = n_burnin_of_the_fits, n_adapt = n_adapt, n_iter = n_iter_of_the_fits,
            seed = seed_of_the_fits, adapt_delta = 0.8, learning_rate = learning_rate_of_the_fits,
            learning_rate_initial = learning_rate_of_the_fits, burnin_algorithm = "CHESSR",
            tau_adaptation_block = "joint", interest_only = "every main parameter", burnin_schedule = "automatic",
            tau_initial = "adaptive", tau_ramp = "staged", tau_weight_by_p_jump = TRUE, manual_tau = FALSE,
            randomize_tau_burnin = TRUE, tau_sampling_scale = "none", share_tau_ii_across_chains_in_burnin = TRUE,
            eps_acceptance_mean = "harmonic", tau_shrink_on_divergence = FALSE,
            eps_reinit_at_ChEES_handover = FALSE, pre_burnin_n_iter = 50, pre_burnin_L = 10,
            reorder_cols_MVP = FALSE, burnin_TBB_pool_equals_n_chains = TRUE,
            diffusion_HMC = TRUE, partitioned_HMC = FALSE, diffusion_HMC_integrator = "kick_flow_kick",
            metric_estimator = "pooled", metric_pooled_window_resets = "stan_style",
            metric_pooled_offdiagonal_shrinkage = "adaptive",
            metric_type_main = "Empirical", metric_shape_main = "diag", ratio_M_main = 0.9,
            interval_width_main = 1, metric_type_nuisance = "Empirical", metric_shape_nuisance = "diag",
            ratio_M_nuisance = 0.9, interval_width_nuisance = 1, M_decay_type = "inverse", M_decay_power = 0.5,
            M_decay_scale = n_adapt / 5, max_tau_main = 50, max_tau_nuisance = 50, max_eps_main = 1.0,
            max_eps_nuisance = 1.0, max_L = 1024, calibrate_eps_on_sampling_chains_before_sampling = FALSE,
            multi_attempts = TRUE, force_autodiff = FALSE, force_PartialLog = FALSE,
            nuisance_to_track = 0, store_log_lik_trace = FALSE, run_in_fresh_R_process = FALSE,
            initial_values = paste0("u_raw ~ normal(0, ", sd_of_the_initial_nuisance_values, ") drawn in R ",
                                    "(L'Ecuyer-CMRG, one stream per chain); correlations 0.001; ",
                                    "coefficients at their prior means; prevalences 0.10",
                                    if (model_family == "ordinal") "; raw cutpoints -2 then log(4 / n_thr)"
                                    else ""))
      message(paste0("\033[36m", "Short fit of the ", model_family, " model for the points: ",
                     paste(names(settings_of_the_fit),
                           unlist(lapply(settings_of_the_fit, paste, collapse = ",")),
                           sep = " = ", collapse = "; "), "\033[0m"))
      ##
      model <-  BayesMVP::MVP_model$new(Model_type = family_data$cpp_model_type,
                                        model_args_list = cpp_model_of_the_fit$model_args_list,
                                        sample_nuisance = TRUE)
      initial_values_of_each_chain <-  fn_initial_values_of_each_chain(family_data, model_family)
      fit <-  NULL
      seconds_of_the_fit <-  system.time({
            fit <-  model$sample( init_lists_per_chain = initial_values_of_each_chain,
                                  seed = seed_of_the_fits,
                                  stream = seed_of_the_fits,
                                  parallel_method = "RcppParallel",
                                  ##
                                  n_chains_burnin = n_chains_of_the_fits,
                                  n_threads_WCP_burnin = 1,
                                  num_chunks_burnin = 1,
                                  n_chains_sampling = n_chains_of_the_fits,
                                  n_superchains = n_chains_of_the_fits,
                                  n_threads_WCP_sampling = 1,
                                  num_chunks_sampling = 1,
                                  n_burnin = n_burnin_of_the_fits,
                                  n_adapt = n_adapt,
                                  n_iter = n_iter_of_the_fits,
                                  ##
                                  adapt_delta = 0.8,
                                  learning_rate = learning_rate_of_the_fits,
                                  learning_rate_initial = learning_rate_of_the_fits,
                                  burnin_algorithm = "CHESSR",
                                  tau_adaptation_block = "joint",
                                  interest_only = NULL,
                                  burnin_schedule = "automatic",
                                  tau_initial = "adaptive",
                                  tau_ramp = "staged",
                                  tau_weight_by_p_jump = TRUE,
                                  manual_tau = FALSE,
                                  randomize_tau_burnin = TRUE,
                                  tau_sampling_scale = "none",
                                  share_tau_ii_across_chains_in_burnin = TRUE,
                                  eps_acceptance_mean = "harmonic",
                                  tau_shrink_on_divergence = FALSE,
                                  eps_reinit_at_ChEES_handover = FALSE,
                                  pre_burnin_n_iter = 50,
                                  pre_burnin_L = 10,
                                  reorder_cols_MVP = FALSE,
                                  burnin_TBB_pool_equals_n_chains = TRUE,
                                  ##
                                  diffusion_HMC = TRUE,
                                  partitioned_HMC = FALSE,
                                  diffusion_HMC_integrator = "kick_flow_kick",
                                  metric_estimator = "pooled",
                                  metric_pooled_window_resets = "stan_style",
                                  metric_pooled_offdiagonal_shrinkage = "adaptive",
                                  metric_type_main = "Empirical",
                                  metric_shape_main = "diag",
                                  ratio_M_main = 0.9,
                                  interval_width_main = 1,
                                  metric_type_nuisance = "Empirical",
                                  metric_shape_nuisance = "diag",
                                  ratio_M_nuisance = 0.9,
                                  interval_width_nuisance = 1,
                                  M_decay_type = "inverse",
                                  M_decay_power = 0.5,
                                  M_decay_scale = n_adapt / 5,
                                  max_tau_main = 50,
                                  max_tau_nuisance = 50,
                                  max_eps_main = 1.0,
                                  max_eps_nuisance = 1.0,
                                  max_L = 1024,
                                  calibrate_eps_on_sampling_chains_before_sampling = FALSE,
                                  ##
                                  multi_attempts = TRUE,
                                  force_autodiff = FALSE,
                                  force_PartialLog = FALSE,
                                  Phi_type = "Phi",
                                  inv_Phi_type = "inv_Phi",
                                  nuisance_to_track = 0,
                                  store_log_lik_trace = FALSE,
                                  n_refresh = n_burnin_of_the_fits,
                                  run_in_fresh_R_process = FALSE)
      })[["elapsed"]]
      message(paste0("Short fit of the ", model_family, " model: ", round(seconds_of_the_fit, 2), " s"))
      ##
      ## ---- its min ESS (bulk, tail and SD ESS over every main parameter) and its divergent transitions:
      ##
      model_fit <-  fit$summary(compute_nested_rhat = FALSE, n_superchains = n_chains_of_the_fits,
                                save_log_lik_trace = FALSE)
      efficiency <-  model_fit$get_efficiency_metrics()
      min_ESS_bulk <-  efficiency$Min_ESS_parameters_of_interest
      min_ESS_tail <-  efficiency$Min_ESS_tail_parameters_of_interest
      min_ESS_sd <-  efficiency$Min_ESS_sd_parameters_of_interest
      if (is.null(min_ESS_bulk)) min_ESS_bulk <-  efficiency$Min_ESS_main
      if (is.null(min_ESS_tail)) min_ESS_tail <-  efficiency$Min_ESS_tail_main
      if (is.null(min_ESS_sd)) min_ESS_sd <-  efficiency$Min_ESS_sd_main
      min_ESS <-  min(min_ESS_bulk, min_ESS_tail, min_ESS_sd)
      sampling_diagnostics <-  fit$result$sampling_object$sampling_diagnostics
      divergence_fields <-  grep("div", names(sampling_diagnostics), value = TRUE, ignore.case = TRUE)
      divergent_transitions <-  vapply(divergence_fields, function(field) {
            sum(as.numeric(sampling_diagnostics[[field]]), na.rm = TRUE)
      }, numeric(1))
      message(paste0("\033[32m", "Short fit of the ", model_family, " model: min ESS ", round(min_ESS, 1),
                     " (bulk ", round(min_ESS_bulk, 1), ", tail ", round(min_ESS_tail, 1), ", SD ",
                     round(min_ESS_sd, 1), "; every main parameter); divergent transitions in sampling: ",
                     paste(names(divergent_transitions), divergent_transitions, sep = " = ", collapse = ", "),
                     "\033[0m"))
      ##
      ## ---- the points: the last burn-in state of each chain (main in the C++ order, nuisance as N x n_tests):
      ##
      theta_main_of_each_chain <-  fit$result$burnin_object$theta_main_vectors_all_chains_input_from_R
      theta_us_of_each_chain <-  fit$result$burnin_object$theta_us_vectors_all_chains_input_from_R
      chunk_rows_of_the_fit <-  fn_chunk_rows("Stan", 1, family_data$N)
      points <-  list()
      for (chain in seq_len(n_chains_of_the_fits)) {
            points[[paste0("last_burnin_state_of_chain_", chain)]] <-  list(
                  theta_main = as.numeric(theta_main_of_each_chain[, chain]),
                  u_unc = fn_nuisance_matrix_from_vector_in_chunk_layout(
                                as.numeric(theta_us_of_each_chain[, chain]), chunk_rows_of_the_fit,
                                family_data$N, family_data$n_tests))
      }
      ##
      ## ---- a check of the points: the log density at each fit point must be finite, and far above the one at
      ##      the first point with its nuisance values shuffled between the subjects:
      ##
      lp_at_the_fit_points <-  vapply(seq_len(n_chains_of_the_fits), function(chain) {
            fn_cpp_lp_grad(cpp_model_of_the_fit, points[[chain]]$theta_main, points[[chain]]$u_unc,
                           "as sampled")$lp
      }, numeric(1))
      set.seed(seed_of_the_shuffle_of_the_points_check)
      rows_shuffled <-  sample(seq_len(family_data$N))
      lp_at_the_first_point_with_its_nuisance_values_shuffled <-
            fn_cpp_lp_grad(cpp_model_of_the_fit, points[[1]]$theta_main,
                           points[[1]]$u_unc[rows_shuffled, , drop = FALSE], "as sampled")$lp
      message(paste0("Log density (Phi_type 1, Stan math) at the 4 fit points: ",
                     paste(round(lp_at_the_fit_points, 2), collapse = ", "),
                     "; at the first with its nuisance values shuffled between the subjects: ",
                     round(lp_at_the_first_point_with_its_nuisance_values_shuffled, 2)))
      stopifnot(all(is.finite(lp_at_the_fit_points)))
      ##
      ## ---- the tail point: the first point with the intercepts of class 1 at -8.5 and of class 2 at +8.5:
      ##
      n_tests <-  family_data$n_tests
      n_cov <-  family_data$n_cov
      tail_point <-  points[[1]]
      if (identical(model_family, "CLC_MVOP")) {
            n_pairs <- as.integer(choose(n_tests, 2L))
            beta_offset <- 4L * n_pairs + n_pairs + 4L
            for (class_index in seq_len(4L)) {
                  for (test_index in seq_len(n_tests)) {
                        pos_intercept <- beta_offset + (class_index - 1L) * n_tests * n_cov +
                              (test_index - 1L) * n_cov + 1L
                        own_st <- family_data$Stan_data$class_D[class_index, family_data$Stan_data$test_condition[test_index]]
                        tail_point$theta_main[pos_intercept] <- if (own_st == 1L) {
                              +intercepts_of_the_tail_point
                        } else {
                              -intercepts_of_the_tail_point
                        }
                  }
            }
      } else {
            n_corrs_of_both_classes <-  2 * n_tests * (n_tests - 1) / 2
            for (class_index in 1:2) for (test_index in 1:n_tests) {
                  position_of_the_intercept <-  n_corrs_of_both_classes + (class_index - 1) * n_tests * n_cov +
                                                (test_index - 1) * n_cov + 1
                  tail_point$theta_main[position_of_the_intercept] <-  if (class_index == 1) {
                        -intercepts_of_the_tail_point
                  } else +intercepts_of_the_tail_point
            }
      }
      points[[paste0("intercepts_pm_", intercepts_of_the_tail_point)]] <-  tail_point
      ##
      return(list(points = points,
                  record_of_the_fit = list(model_family = model_family,
                                           settings = settings_of_the_fit,
                                           seconds = seconds_of_the_fit,
                                           min_ESS = min_ESS,
                                           min_ESS_bulk = min_ESS_bulk,
                                           min_ESS_tail = min_ESS_tail,
                                           min_ESS_sd = min_ESS_sd,
                                           ESS_quantities = "every main parameter",
                                           efficiency = efficiency,
                                           divergent_transitions_in_sampling = divergent_transitions,
                                           log_density_at_the_fit_points = lp_at_the_fit_points,
                                           log_density_at_the_first_point_with_its_nuisance_values_shuffled =
                                                 lp_at_the_first_point_with_its_nuisance_values_shuffled)))

}
##
##
## ==== The comparison of every number of chunks with one chunk ==================================================
##
family_data_list <-  list(binary = fn_Stan_data_of_model_family("binary"),
                          ordinal = fn_Stan_data_of_model_family("ordinal"),
                          CLC_MVOP = fn_Stan_data_of_model_family("CLC_MVOP"))
##
records_of_the_fits <-  list()
points_list <-  list()
for (model_family in names(family_data_list)) {
      short_fit <-  fn_short_fit_and_its_points(family_data_list[[model_family]], model_family)
      points_list[[model_family]] <-  short_fit$points
      records_of_the_fits[[model_family]] <-  short_fit$record_of_the_fit
}
saveRDS(records_of_the_fits,
        file.path(output_folder, paste0("min_ESS_and_settings_of_the_short_fits_that_give_the_points_", machine,
                                        ".rds")))
##
## (one C++ model per model family, data, Phi_type, vector type and number of chunks)
##
models_of_the_comparison <-  expand.grid(model_family = c("binary", "ordinal", "CLC_MVOP"),
                                         data = c("without missing results", "with missing results"),
                                         Phi_type = c(1, 2),
                                         vect_type = vect_types_to_check,
                                         stringsAsFactors = FALSE)
evaluations <-  c("as sampled", "PartialLog", "autodiff")
##
rows_of_the_comparison <-  list()
numbers_of_chunks_left_out <-  c()
for (model_index in seq_len(nrow(models_of_the_comparison))) {

      model_family <-  models_of_the_comparison$model_family[model_index]
      evaluations_of_model <-  evaluations
      data_of_the_model <-  models_of_the_comparison$data[model_index]
      Phi_type <-  models_of_the_comparison$Phi_type[model_index]
      vect_type <-  models_of_the_comparison$vect_type[model_index]
      family_data <-  family_data_list[[model_family]]
      y_of_the_model <-  if (data_of_the_model == "with missing results") {
            fn_y_with_missing_results(family_data$Stan_data$y)
      } else family_data$Stan_data$y
      ##
      cpp_model_of_one_chunk <-  fn_make_cpp_model(family_data, model_family, y = y_of_the_model,
                                                   vect_type = vect_type, Phi_type = Phi_type, num_chunks = 1)
      results_of_one_chunk <-  list()
      for (evaluation in evaluations_of_model) {
            results_of_one_chunk[[evaluation]] <-  lapply(points_list[[model_family]], function(point) {
                  fn_cpp_lp_grad(cpp_model_of_one_chunk, point$theta_main, point$u_unc, evaluation)
            })
      }
      ##
      for (num_chunks in numbers_of_chunks_compared_with_one_chunk) {
            if (is.null(fn_chunk_rows(vect_type, num_chunks, family_data$N))) {
                  numbers_of_chunks_left_out <-  unique(c(numbers_of_chunks_left_out,
                                                          paste0(model_family, " ", vect_type, ": ", num_chunks)))
                  next
            }
            cpp_model_of_these_chunks <-  fn_make_cpp_model(family_data, model_family, y = y_of_the_model,
                                                            vect_type = vect_type, Phi_type = Phi_type,
                                                            num_chunks = num_chunks)
            for (evaluation in evaluations_of_model) for (point_name in names(points_list[[model_family]])) {
                  point <-  points_list[[model_family]][[point_name]]
                  of_one_chunk <-  results_of_one_chunk[[evaluation]][[point_name]]
                  of_these_chunks <-  fn_cpp_lp_grad(cpp_model_of_these_chunks, point$theta_main, point$u_unc,
                                                     evaluation)
                  values_of_one_chunk <-  c(of_one_chunk$lp, of_one_chunk$grad_main, of_one_chunk$grad_us)
                  values_of_these_chunks <-  c(of_these_chunks$lp, of_these_chunks$grad_main,
                                               of_these_chunks$grad_us)
                  all_finite <-  all(is.finite(values_of_one_chunk)) && all(is.finite(values_of_these_chunks))
                  relative_lp_difference <-  abs(of_these_chunks$lp - of_one_chunk$lp) /
                                             max(1, abs(of_one_chunk$lp))
                  relative_grad_main_difference <-  max(abs(of_these_chunks$grad_main - of_one_chunk$grad_main)) /
                                                    max(1, max(abs(of_one_chunk$grad_main)))
                  relative_grad_nuisance_difference <-  max(abs(of_these_chunks$grad_us - of_one_chunk$grad_us)) /
                                                        max(1, max(abs(of_one_chunk$grad_us)))
                  rows_of_the_comparison[[length(rows_of_the_comparison) + 1]] <-  data.frame(
                        model_family = model_family,
                        data = data_of_the_model,
                        Phi_type = Phi_type,
                        vect_type = vect_type,
                        evaluation = evaluation,
                        point = point_name,
                        num_chunks = num_chunks,
                        lp_of_one_chunk = of_one_chunk$lp,
                        lp_of_these_chunks = of_these_chunks$lp,
                        relative_lp_difference = relative_lp_difference,
                        relative_grad_main_difference = relative_grad_main_difference,
                        relative_grad_nuisance_difference = relative_grad_nuisance_difference,
                        n_values_identical = sum(values_of_one_chunk == values_of_these_chunks, na.rm = TRUE),
                        n_values = length(values_of_one_chunk),
                        all_finite = all_finite,
                        passes = all_finite &&
                                 (relative_lp_difference <= largest_relative_difference_allowed) &&
                                 (relative_grad_main_difference <= largest_relative_difference_allowed) &&
                                 (relative_grad_nuisance_difference <= largest_relative_difference_allowed))
            }
      }
      message(paste0(model_family, ", ", data_of_the_model, ", Phi_type ", Phi_type, ", ", vect_type, ": done"))

}
##
table_of_the_comparison <-  do.call(rbind, rows_of_the_comparison)
file_of_the_comparison <-  file.path(output_folder,
                                     paste0("comparison_of_every_number_of_chunks_with_one_chunk_", machine,
                                            ".csv"))
utils::write.csv(table_of_the_comparison, file_of_the_comparison, row.names = FALSE)
##
## ---- the summary: the largest differences of each case, and the cases that do not pass:
##
print(stats::aggregate(cbind(relative_lp_difference, relative_grad_main_difference,
                             relative_grad_nuisance_difference) ~
                             model_family + Phi_type + vect_type + evaluation,
                       data = table_of_the_comparison, FUN = max),
      digits = 3)
cases_that_do_not_pass <-  table_of_the_comparison[!table_of_the_comparison$passes, , drop = FALSE]
if (nrow(cases_that_do_not_pass) > 0) {
      print(utils::head(cases_that_do_not_pass, 40), digits = 3)
}
if (length(numbers_of_chunks_left_out) > 0) {
      message(paste0("Numbers of chunks left out (fewer subjects in a chunk than the SIMD width): ",
                     paste(numbers_of_chunks_left_out, collapse = "; ")))
}
message(paste0("\033[32m", "Same log density and gradient at every number of chunks, within ",
               largest_relative_difference_allowed, ": passed ", sum(table_of_the_comparison$passes), " of ",
               nrow(table_of_the_comparison), " (", sum(!table_of_the_comparison$all_finite),
               " with a non-finite value); written: ", file_of_the_comparison, "\033[0m"))
message("done")






















