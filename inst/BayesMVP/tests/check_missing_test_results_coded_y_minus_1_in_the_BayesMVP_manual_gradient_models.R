##
## ==== check_missing_test_results_coded_y_minus_1_in_the_BayesMVP_manual_gradient_models.R ====================
##
## Checks of the missing test results (coded y = -1) in BayesMVP's manual-gradient C++ models, the binary LC-MVP
## (Model_type = "LC_MVP") and the LC-MVOP (Model_type = "LC_MVOP"), by the GHK method (the untruncated draw
## Z = F^{-1}(u), no likelihood term; MVP_functions/MVP_and_MVOP_missing_results_GHK_fns.hpp). The log density and
## gradient come from BayesMVP:::Rcpp_wrapper_fn_lp_grad (one thread, i.e. the serial evaluation), evaluated
##   (1) as sampled (multi_attempts: the NoLog evaluation, then PartialLog on a non-finite result),
##   (2) with force_PartialLog = TRUE (the log-scale evaluation), and
##   (3) with force_autodiff = TRUE (the autodiff evaluation),
## for Phi_type = 1 (Phi, inv_Phi) and Phi_type = 2 (Phi_approx, inv_Phi_approx), with the AVX-512 (or AVX2)
## functions and with the scalar Stan math functions ("Stan").
##
## Three checks:
##   (a) data WITHOUT missing results: the log density and gradient of the installed BayesMVP must be identical
##       (bit for bit) to those saved from the BayesMVP installed before the missing-result code was added
##       (step 1 below saves them);
##   (b) data WITH missing results (10% of the results set to -1 with a fixed seed, and one subject with every
##       test missing): the gradient against central finite differences of the log density (5-point stencil),
##       for every main parameter and for every nuisance value of 8 subjects (the subject with every test
##       missing, subjects with and without missing results); the largest relative error
##       |analytic - finite difference| / max(1, |finite difference|) must be below 1e-6;
##   (c) the same missing-data case against a Stan model of the same family (BridgeStan, propto = FALSE,
##       jacobian = TRUE) that takes the same y with -1 for a missing result and has the parameters of the
##       staged Stan models (binary: u_raw, col_one_raw, off_raw, beta_vec, p_raw; ordinal: u_raw,
##       Omega_unconstrained_vec, beta_vec, p_raw, C_unc_vec), with the parameter mapping of
##       check_staged_LC_MVOP_models_against_BayesMVP_cpp_lp_grad.R.
##
## The data are those of the PS6 runners (binary N = 500, 6 tests; ordinal N = 1000, 3 tests), as in
## inst/stan_models/staging_Phi_types_1_and_2_vectorised/LC_MVOP_without_Phi_type_0_and_loops_over_N/
## check_staged_LC_MVOP_models_against_BayesMVP_cpp_lp_grad.R, whose C++ model set-up this script reuses.
## Correctness checks only (no timings).
##
## Reads:  the PS6 data functions (Alg_paper_analysis/1_appendix_pilot_studies/ps_6_basic_MCMC_settings_Stan/
##         functions) and the simulation functions (Alg_paper_analysis/0_utilities/shared_functions);
##         for (a), the results saved by step 1; for (c), the Stan model library <name>_model.so.
## Writes: <output folder>/results_of_the_current_install_without_missing_results.rds        (step 1)
##         <output folder>/check_a_no_missing_results_identical_to_the_previous_install.csv (step 2)
##         <output folder>/check_b_gradient_against_finite_differences_with_missing_results.csv (step 2)
##         <output folder>/check_c_against_the_Stan_model_<family>_<Stan model name>.csv      (step 3)
##
## Usage (lowest priority, on the CPUs for checks):
##   step 1 - BEFORE the new C++ and R files are switched in, with the current main install of BayesMVP:
##     chrt -i 0 nice -n 19 ionice -c 3 taskset -c 180-191 ~/R/R-4.3.3/bin/Rscript \
##         check_missing_test_results_coded_y_minus_1_in_the_BayesMVP_manual_gradient_models.R \
##         save_the_results_of_the_current_install_without_missing_results
##   step 2 - AFTER the switch and the reinstall of BayesMVP (checks (a) and (b)):
##     chrt -i 0 nice -n 19 ionice -c 3 taskset -c 180-191 ~/R/R-4.3.3/bin/Rscript \
##         check_missing_test_results_coded_y_minus_1_in_the_BayesMVP_manual_gradient_models.R \
##         check_the_new_install
##   step 3 - check (c), once the Stan model with y = -1 exists (family "binary" or "ordinal"; the C++ nuisance
##            transform must be the Stan model's, set with CPP_NUISANCE_TRANSFORMATION_MATCHING_THE_STAN_MODEL,
##            e.g. Phi):
##     CPP_NUISANCE_TRANSFORMATION_MATCHING_THE_STAN_MODEL=Phi STAN_NUM_THREADS=1 \
##     chrt -i 0 nice -n 19 ionice -c 3 taskset -c 180-191 ~/R/R-4.3.3/bin/Rscript \
##         check_missing_test_results_coded_y_minus_1_in_the_BayesMVP_manual_gradient_models.R \
##         check_against_a_Stan_model <family> <full path of the Stan model's _model.so>
##   (an optional last argument, a library path, is put first in .libPaths(), so that a BayesMVP installed into a
##    separate library can be checked)
##
## ---- Settings: ------------------------------------------------------------------------------------------------
##
{
      command_line_arguments <-  commandArgs(trailingOnly = TRUE)
      what_to_run <-  command_line_arguments[1]
      stopifnot(what_to_run %in% c("save_the_results_of_the_current_install_without_missing_results",
                                   "check_the_new_install",
                                   "check_against_a_Stan_model"))
      if (what_to_run == "check_against_a_Stan_model") {
            model_family_of_the_Stan_model <-  command_line_arguments[2]
            path_of_the_Stan_model_library <-  command_line_arguments[3]
            stopifnot(model_family_of_the_Stan_model %in% c("binary", "ordinal"),
                      file.exists(path_of_the_Stan_model_library))
            library_path_to_put_first <-  if (length(command_line_arguments) >= 4) {
                  command_line_arguments[4]
            } else ""
      } else {
            library_path_to_put_first <-  if (length(command_line_arguments) >= 2) {
                  command_line_arguments[2]
            } else ""
      }
      ##
      output_folder <-  file.path("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP",
                                  "tests/outputs/missing_test_results_coded_y_minus_1")
      ##
      proportion_of_results_set_missing <-  0.10
      seed_of_the_missing_results <-  20261016
      subject_with_every_test_missing <-  7
      ##
      seed_of_the_points <-  20261006
      n_moderate_points <-  3
      intercepts_of_the_tail_point <-  8.5          ## observed results in the tails (the log-scale branches)
      ##
      finite_difference_step <-  1e-4
      largest_relative_error_allowed_against_finite_differences <-  1e-6
      ##
      num_chunks_cpp <-  4
      J_grad_option_of_the_checks <-  "autodiff"     ## the exact gradient of the correlation Jacobian
}
##
if (library_path_to_put_first != "") .libPaths(c(library_path_to_put_first, .libPaths()))
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1", STAN_NUM_THREADS = "1")
options(scipen = 999, width = 220)
dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)
##
A_dir <-  "/home/enzocerullo/Documents/Work/PhD_work/Alg_paper_analysis"
##
suppressPackageStartupMessages(library(BayesMVP))   ## loads the shared library; every call below is namespaced
##
message(paste0("BayesMVP installed at ", system.file(package = "BayesMVP"), "; BayesMVP.so md5 = ",
               unname(tools::md5sum(file.path(system.file(package = "BayesMVP"), "libs", "BayesMVP.so")))))
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
## ---- The vector type of the SIMD functions on this machine (AVX-512 on the local-HPC, AVX2 on the laptop), and
##      the scalar Stan math functions:
##
vectorisation_support <-  BayesMVP:::detect_vectorization_support()
SIMD_vect_type <-  if (grepl("AVX512", vectorisation_support)) "AVX512" else "AVX2"
vect_types_to_check <-  c(SIMD_vect_type, "Stan")
message(paste0("vectorisation support: ", vectorisation_support, "; vect_types checked: ",
               paste(vect_types_to_check, collapse = ", ")))
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
## ---- y with missing results: proportion_of_results_set_missing of the results set to -1 (fixed seed), and every
##      test of one subject:
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
## ---- The BayesMVP C++ model built from a Stan data list and a y (from check_staged_LC_MVOP_models_against_
##      BayesMVP_cpp_lp_grad.R; here with the y given, the Phi_type given, and the package default nuisance
##      transform unless CPP_NUISANCE_TRANSFORMATION_MATCHING_THE_STAN_MODEL is set):
##
fn_matrix_of_class <-  function(x, c) {
      if (is.list(x)) return(as.matrix(x[[c]]))
      return(matrix(x[c, , ], nrow = dim(x)[2]))
}
##
fn_make_cpp_model <-  function(family_data, model_family, y, vect_type, Phi_type) {

      Stan_data <-  family_data$Stan_data
      n_tests <-  family_data$n_tests
      n_class <-  2
      storage.mode(y) <-  "integer"
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
      model_args_list$num_chunks <-  num_chunks_cpp
      model_args_list$J_grad_option <-  J_grad_option_of_the_checks
      model_args_list$vect_type <-  vect_type
      model_args_list$Phi_type <-  if (Phi_type == 2) "Phi_approx" else "Phi"
      model_args_list$inv_Phi_type <-  if (Phi_type == 2) "inv_Phi_approx" else "inv_Phi"
      nuisance_transformation_matching_the_Stan_model <-
            Sys.getenv("CPP_NUISANCE_TRANSFORMATION_MATCHING_THE_STAN_MODEL", unset = "")
      if (nuisance_transformation_matching_the_Stan_model != "") {
            model_args_list$nuisance_transformation <-  nuisance_transformation_matching_the_Stan_model
      }
      final_init <-  NULL
      invisible(utils::capture.output(
            final_init <-  BayesMVP::init_hard_coded_model(Model_type = family_data$cpp_model_type,
                                                           model_args_list = model_args_list)))
      y_final <-  final_init$model_args_list$y
      storage.mode(y_final) <-  "integer"
      stopifnot(identical(as.integer(y_final), as.integer(y)))
      return(list(Model_args_as_Rcpp_List = final_init$Model_args_as_Rcpp_List,
                  y = y_final,
                  N = family_data$N,
                  n_tests = n_tests,
                  cpp_model_type = family_data$cpp_model_type,
                  vect_type = vect_type))

}
##
## ---- The C++ nuisance layout: per chunk (calculate_chunk_sizes), a column-major (chunk_size x n_tests) block:
##
fn_chunk_rows <-  function(vect_type, num_chunks, N) {
      vec_size <-  if (vect_type == "AVX512") 8 else if (vect_type == "AVX2") 4 else
                   if (vect_type == "AVX") 2 else 1
      normal_chunk_size <-  vec_size * (N %/% (vec_size * max(num_chunks, 1)))
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
## ---- The C++ log density and gradient; evaluation = "as sampled", "PartialLog" or "autodiff":
##
fn_cpp_lp_grad <-  function(cpp_model, theta_main, u_unc_matrix, evaluation) {
      N <-  cpp_model$N
      n_tests <-  cpp_model$n_tests
      chunk_rows <-  fn_chunk_rows(cpp_model$vect_type, num_chunks_cpp, N)
      theta_us <-  unlist(lapply(seq_along(chunk_rows$starts), function(chunk) {
            as.vector(u_unc_matrix[chunk_rows$starts[chunk] + seq_len(chunk_rows$sizes[chunk]), , drop = FALSE])
      }))
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
      grad_us_packed <-  lp_grad_output[1 + seq_len(n_us)]
      grad_us <-  matrix(NA_real_, N, n_tests)
      offset <-  0
      for (chunk in seq_along(chunk_rows$starts)) {
            rows_of_chunk <-  chunk_rows$starts[chunk] + seq_len(chunk_rows$sizes[chunk])
            n_values <-  length(rows_of_chunk) * n_tests
            grad_us[rows_of_chunk, ] <-  matrix(grad_us_packed[offset + seq_len(n_values)],
                                                nrow = length(rows_of_chunk))
            offset <-  offset + n_values
      }
      return(list(lp = lp_grad_output[1],
                  grad_main = lp_grad_output[1 + n_us + seq_along(theta_main)],
                  grad_us = grad_us))
}
##
## ---- Points (as in check_staged_LC_MVOP_models_against_BayesMVP_cpp_lp_grad.R): moderate points, and one with
##      the intercepts of class 1 at -intercepts_of_the_tail_point and of class 2 at +intercepts_of_the_tail_point
##      (observed results in the tails):
##
fn_base_point <-  function(family_data, model_family, u_sd) {
      Stan_data <-  family_data$Stan_data
      n_tests <-  family_data$n_tests
      n_cov <-  family_data$n_cov
      n_pops <-  as.integer(Stan_data$n_pops)
      point <-  list()
      point$Omega_unc <-  lapply(1:2, function(class_index) {
            Omega_unc <-  matrix(0, n_tests, n_tests)
            Omega_unc[lower.tri(Omega_unc)] <-  stats::rnorm(sum(lower.tri(Omega_unc)), 0, 0.25)
            Omega_unc
      })
      point$beta <-  lapply(1:2, function(class_index) {
            fn_matrix_of_class(Stan_data$prior_beta_mean, class_index) +
                  matrix(stats::rnorm(n_cov * n_tests, 0, 0.2), n_cov, n_tests)
      })
      point$p_raw <-  atanh(2 * stats::runif(n_pops, 0.05, 0.3) - 1)
      if (model_family == "ordinal") {
            point$C_unc <-  lapply(1:2, function(class_index) {
                  C_raw <-  c()
                  for (k_slot in seq_along(Stan_data$n_thr_per_ord_test)) {
                        n_thr <-  Stan_data$n_thr_per_ord_test[k_slot]
                        spacing <-  4.0 / n_thr * exp(stats::rnorm(n_thr - 1, 0, 0.2))
                        C_raw <-  c(C_raw, -2.0 + stats::rnorm(1, 0, 0.2), log(spacing))
                  }
                  atanh(2 * (C_raw - Stan_data$C_raw_lower) / (Stan_data$C_raw_upper - Stan_data$C_raw_lower) - 1)
            })
      }
      point$u_unc <-  matrix(stats::rnorm(family_data$N * n_tests, 0, u_sd), family_data$N, n_tests)
      return(point)
}
##
fn_points <-  function(family_data, model_family) {
      set.seed(seed_of_the_points)
      points <-  list()
      for (replicate_index in 1:n_moderate_points) {
            points[[paste0("moderate_", replicate_index)]] <-  fn_base_point(family_data, model_family,
                                                                             u_sd = 0.7)
      }
      tail_point <-  fn_base_point(family_data, model_family, u_sd = 0.7)
      for (test_index in 1:family_data$n_tests) {
            tail_point$beta[[1]][1, test_index] <-  -intercepts_of_the_tail_point
            tail_point$beta[[2]][1, test_index] <-  +intercepts_of_the_tail_point
      }
      points[[paste0("intercepts_pm_", intercepts_of_the_tail_point)]] <-  tail_point
      return(points)
}
##
## ---- C++ theta_main = corrs (c, then i > j row by row), beta (c, t, k), p_raw, cutpoints (c, slot, k):
##
fn_theta_main_cpp <-  function(point, family_data, model_family) {
      n_tests <-  family_data$n_tests
      corrs <-  c()
      for (class_index in 1:2) for (i in 2:n_tests) for (j in 1:(i - 1)) {
            corrs <-  c(corrs, point$Omega_unc[[class_index]][i, j])
      }
      betas <-  c()
      for (class_index in 1:2) for (test_index in 1:n_tests) for (covariate_index in 1:family_data$n_cov) {
            betas <-  c(betas, point$beta[[class_index]][covariate_index, test_index])
      }
      theta_main <-  c(corrs, betas, point$p_raw)
      if (model_family == "ordinal") theta_main <-  c(theta_main, point$C_unc[[1]], point$C_unc[[2]])
      return(theta_main)
}
##
## ---- The cases: model family x Phi_type x vect_type x evaluation:
##
cases <-  expand.grid(model_family = c("binary", "ordinal"),
                      Phi_type = c(1, 2),
                      vect_type = vect_types_to_check,
                      evaluation = c("as sampled", "PartialLog", "autodiff"),
                      stringsAsFactors = FALSE)
##
family_data_list <-  list(binary = fn_Stan_data_of_model_family("binary"),
                          ordinal = fn_Stan_data_of_model_family("ordinal"))
points_list <-  list(binary = fn_points(family_data_list$binary, "binary"),
                     ordinal = fn_points(family_data_list$ordinal, "ordinal"))
##
## ==== step 1 and check (a): the results of the data WITHOUT missing results ====================================
##
fn_results_without_missing_results <-  function() {
      results <-  list()
      for (case_index in seq_len(nrow(cases))) {
            model_family <-  cases$model_family[case_index]
            family_data <-  family_data_list[[model_family]]
            cpp_model <-  fn_make_cpp_model(family_data, model_family, y = family_data$Stan_data$y,
                                            vect_type = cases$vect_type[case_index],
                                            Phi_type = cases$Phi_type[case_index])
            for (point_name in names(points_list[[model_family]])) {
                  point <-  points_list[[model_family]][[point_name]]
                  theta_main <-  fn_theta_main_cpp(point, family_data, model_family)
                  case_name <-  paste(model_family, cases$Phi_type[case_index], cases$vect_type[case_index],
                                      cases$evaluation[case_index], point_name, sep = " / ")
                  results[[case_name]] <-  fn_cpp_lp_grad(cpp_model, theta_main, point$u_unc,
                                                          cases$evaluation[case_index])
            }
      }
      return(results)
}
##
file_of_the_results_of_the_current_install <-
      file.path(output_folder, "results_of_the_current_install_without_missing_results.rds")
##
if (what_to_run == "save_the_results_of_the_current_install_without_missing_results") {
      results <-  fn_results_without_missing_results()
      saveRDS(list(results = results,
                   BayesMVP_so_md5 = unname(tools::md5sum(file.path(system.file(package = "BayesMVP"), "libs",
                                                                    "BayesMVP.so")))),
              file_of_the_results_of_the_current_install)
      message(paste0("saved ", length(results), " results to ", file_of_the_results_of_the_current_install))
}
##
if (what_to_run == "check_the_new_install") {
      ##
      ## ---- (a) without missing results: identical to the previous install ----
      ##
      saved <-  readRDS(file_of_the_results_of_the_current_install)
      results <-  fn_results_without_missing_results()
      rows_of_check_a <-  list()
      for (case_name in names(saved$results)) {
            previous <-  saved$results[[case_name]]
            new <-  results[[case_name]]
            rows_of_check_a[[length(rows_of_check_a) + 1]] <-  data.frame(
                  case = case_name,
                  lp_identical = identical(previous$lp, new$lp),
                  grad_main_identical = identical(previous$grad_main, new$grad_main),
                  grad_us_identical = identical(previous$grad_us, new$grad_us),
                  lp_difference = new$lp - previous$lp,
                  largest_abs_grad_difference = max(abs(c(new$grad_main - previous$grad_main,
                                                          new$grad_us - previous$grad_us))))
      }
      table_of_check_a <-  do.call(rbind, rows_of_check_a)
      utils::write.csv(table_of_check_a,
                       file.path(output_folder,
                                 "check_a_no_missing_results_identical_to_the_previous_install.csv"),
                       row.names = FALSE)
      print(table_of_check_a, row.names = FALSE, digits = 3)
      all_identical <-  all(table_of_check_a$lp_identical & table_of_check_a$grad_main_identical &
                            table_of_check_a$grad_us_identical)
      message(paste0("(a) without missing results, identical to the previous install in every case: ",
                     all_identical))
      ##
      ## ---- (b) with missing results: the gradient against central finite differences of the log density ----
      ##
      rows_of_check_b <-  list()
      for (case_index in seq_len(nrow(cases))) {
            model_family <-  cases$model_family[case_index]
            family_data <-  family_data_list[[model_family]]
            evaluation <-  cases$evaluation[case_index]
            y_with_missing_results <-  fn_y_with_missing_results(family_data$Stan_data$y)
            cpp_model <-  fn_make_cpp_model(family_data, model_family, y = y_with_missing_results,
                                            vect_type = cases$vect_type[case_index],
                                            Phi_type = cases$Phi_type[case_index])
            ##
            ## the subjects whose nuisance values are checked: the one with every test missing, 3 with a missing
            ## result, 3 without, and the last subject
            ##
            row_has_a_missing_result <-  apply(y_with_missing_results, 1, function(row_of_y) any(row_of_y == -1))
            rows_checked <-  unique(c(subject_with_every_test_missing,
                                      utils::head(setdiff(which(row_has_a_missing_result),
                                                          subject_with_every_test_missing), 3),
                                      utils::head(which(!row_has_a_missing_result), 3),
                                      family_data$N))
            for (point_name in names(points_list[[model_family]])) {
                  point <-  points_list[[model_family]][[point_name]]
                  theta_main <-  fn_theta_main_cpp(point, family_data, model_family)
                  analytic <-  fn_cpp_lp_grad(cpp_model, theta_main, point$u_unc, evaluation)
                  fn_lp_at <-  function(theta_main_at, u_unc_at) {
                        fn_cpp_lp_grad(cpp_model, theta_main_at, u_unc_at, evaluation)$lp
                  }
                  ## 5-point stencil: (8 (f(x + h) - f(x - h)) - (f(x + 2h) - f(x - 2h))) / (12 h)
                  fn_finite_difference <-  function(shift_function) {
                        (8 * (shift_function(+1) - shift_function(-1))
                         - (shift_function(+2) - shift_function(-2))) / (12 * finite_difference_step)
                  }
                  relative_errors_main <-  sapply(seq_along(theta_main), function(parameter_index) {
                        finite_difference <-  fn_finite_difference(function(n_steps) {
                              theta_main_shifted <-  theta_main
                              theta_main_shifted[parameter_index] <-  theta_main_shifted[parameter_index] +
                                                                      n_steps * finite_difference_step
                              fn_lp_at(theta_main_shifted, point$u_unc)
                        })
                        abs(analytic$grad_main[parameter_index] - finite_difference) /
                              max(1, abs(finite_difference))
                  })
                  relative_errors_us <-  c()
                  relative_errors_us_of_missing_results <-  c()
                  for (row_index in rows_checked) for (test_index in seq_len(family_data$n_tests)) {
                        finite_difference <-  fn_finite_difference(function(n_steps) {
                              u_unc_shifted <-  point$u_unc
                              u_unc_shifted[row_index, test_index] <-  u_unc_shifted[row_index, test_index] +
                                                                       n_steps * finite_difference_step
                              fn_lp_at(theta_main, u_unc_shifted)
                        })
                        relative_error <-  abs(analytic$grad_us[row_index, test_index] - finite_difference) /
                                           max(1, abs(finite_difference))
                        relative_errors_us <-  c(relative_errors_us, relative_error)
                        if (y_with_missing_results[row_index, test_index] == -1) {
                              relative_errors_us_of_missing_results <-  c(relative_errors_us_of_missing_results,
                                                                          relative_error)
                        }
                  }
                  largest_relative_error <-  max(c(relative_errors_main, relative_errors_us))
                  rows_of_check_b[[length(rows_of_check_b) + 1]] <-  data.frame(
                        model_family = model_family,
                        Phi_type = cases$Phi_type[case_index],
                        vect_type = cases$vect_type[case_index],
                        evaluation = evaluation,
                        point = point_name,
                        lp = analytic$lp,
                        lp_is_finite = is.finite(analytic$lp),
                        largest_relative_error_main = max(relative_errors_main),
                        largest_relative_error_nuisance = max(relative_errors_us),
                        largest_relative_error_nuisance_of_missing_results =
                              max(relative_errors_us_of_missing_results),
                        passes = is.finite(largest_relative_error) &&
                                 (largest_relative_error <
                                  largest_relative_error_allowed_against_finite_differences))
                  message(paste0("(b) ", model_family, ", Phi_type ", cases$Phi_type[case_index], ", ",
                                 cases$vect_type[case_index], ", ", evaluation, ", ", point_name,
                                 ": largest relative error ", formatC(largest_relative_error, format = "e",
                                                                      digits = 2)))
            }
      }
      table_of_check_b <-  do.call(rbind, rows_of_check_b)
      utils::write.csv(table_of_check_b,
                       file.path(output_folder,
                                 "check_b_gradient_against_finite_differences_with_missing_results.csv"),
                       row.names = FALSE)
      print(table_of_check_b, row.names = FALSE, digits = 3)
      message(paste0("(b) with missing results, every case below ",
                     largest_relative_error_allowed_against_finite_differences, ": ",
                     all(table_of_check_b$passes)))
      ##
      ## ---- the log density of the missing-data case, without any finite-difference reference: the same value
      ##      in every evaluation (as sampled, PartialLog, autodiff) and vect_type of a model family and Phi_type
      ##      (up to rounding), as an extra consistency check:
      ##
      print(stats::aggregate(lp ~ model_family + Phi_type + point, data = table_of_check_b,
                             FUN = function(lp_values) max(lp_values) - min(lp_values)),
            digits = 3)
}
##
## ==== check (c): the missing-data case against a Stan model of the same family ===============================
##
if (what_to_run == "check_against_a_Stan_model") {
      ##
      model_family <-  model_family_of_the_Stan_model
      family_data <-  family_data_list[[model_family]]
      n_tests <-  family_data$n_tests
      n_cov <-  family_data$n_cov
      y_with_missing_results <-  fn_y_with_missing_results(family_data$Stan_data$y)
      Stan_data <-  family_data$Stan_data
      Stan_data$y <-  y_with_missing_results
      ##
      ## ---- Stan parameters in their declared shapes and their positions in the unconstrained vector (from
      ##      check_staged_LC_MVOP_models_against_BayesMVP_cpp_lp_grad.R):
      ##
      fn_stan_param_list <-  function(point) {
            beta_flat <-  c()
            for (class_index in 1:2) for (test_index in 1:n_tests) for (covariate_index in 1:n_cov) {
                  beta_flat <-  c(beta_flat, point$beta[[class_index]][covariate_index, test_index])
            }
            param_list <-  list(u_raw = point$u_unc)
            if (model_family == "binary") {
                  param_list$col_one_raw <-  t(sapply(1:2, function(class_index) {
                        point$Omega_unc[[class_index]][2:n_tests, 1]
                  }))
                  param_list$off_raw <-  t(sapply(1:2, function(class_index) {
                        lower_triangle <-  c()
                        for (i in 3:n_tests) for (j in 2:(i - 1)) {
                              lower_triangle <-  c(lower_triangle, point$Omega_unc[[class_index]][i, j])
                        }
                        lower_triangle
                  }))
            } else {
                  param_list$Omega_unconstrained_vec <-  t(sapply(1:2, function(class_index) {
                        lower_triangle <-  c()
                        for (i in 2:n_tests) for (j in 1:(i - 1)) {
                              lower_triangle <-  c(lower_triangle, point$Omega_unc[[class_index]][i, j])
                        }
                        lower_triangle
                  }))
            }
            param_list$beta_vec <-  beta_flat
            param_list$p_raw <-  point$p_raw
            if (model_family == "ordinal") param_list$C_unc_vec <-  rbind(point$C_unc[[1]], point$C_unc[[2]])
            return(param_list)
      }
      fn_json_of_param_list <-  function(param_list) {
            return(as.character(jsonlite::toJSON(lapply(param_list, function(parameter_value) {
                                                       if (is.matrix(parameter_value)) parameter_value else
                                                             as.array(parameter_value)
                                                 }),
                                                 digits = NA, matrix = "rowmajor")))
      }
      fn_stan_positions <-  function(stan_model, template_param_list) {
            code_list <-  list()
            offset <-  0
            for (parameter_name in names(template_param_list)) {
                  parameter_value <-  template_param_list[[parameter_name]]
                  code_list[[parameter_name]] <-  parameter_value
                  code_list[[parameter_name]][] <-  offset + seq_along(parameter_value)
                  offset <-  offset + length(parameter_value)
            }
            codes_in_unc_order <-  stan_model$param_unconstrain_json(fn_json_of_param_list(code_list))
            stopifnot(length(codes_in_unc_order) == offset,
                      all(sort(round(codes_in_unc_order)) == seq_len(offset)))
            position_of_code <-  integer(offset)
            position_of_code[round(codes_in_unc_order)] <-  seq_len(offset)
            positions <-  list()
            offset <-  0
            for (parameter_name in names(template_param_list)) {
                  positions_of_parameter <-  offset + seq_along(template_param_list[[parameter_name]])
                  positions[[parameter_name]] <-  position_of_code[positions_of_parameter]
                  offset <-  offset + length(template_param_list[[parameter_name]])
            }
            return(positions)
      }
      fn_stan_grad_in_cpp_layout <-  function(stan_grad, positions, template_param_list) {
            gradient_by_parameter <-  lapply(names(template_param_list), function(parameter_name) {
                  parameter_value <-  template_param_list[[parameter_name]]
                  parameter_value[] <-  stan_grad[positions[[parameter_name]]]
                  parameter_value
            })
            names(gradient_by_parameter) <-  names(template_param_list)
            corrs <-  c()
            for (class_index in 1:2) for (i in 2:n_tests) for (j in 1:(i - 1)) {
                  if (model_family == "ordinal") {
                        position <-  0
                        for (row_index in 2:n_tests) for (column_index in 1:(row_index - 1)) {
                              position <-  position + 1
                              if (row_index == i && column_index == j) position_of_ij <-  position
                        }
                        corrs <-  c(corrs,
                                    gradient_by_parameter$Omega_unconstrained_vec[class_index, position_of_ij])
                  } else if (j == 1) {
                        corrs <-  c(corrs, gradient_by_parameter$col_one_raw[class_index, i - 1])
                  } else {
                        position <-  0
                        for (row_index in 3:n_tests) for (column_index in 2:(row_index - 1)) {
                              position <-  position + 1
                              if (row_index == i && column_index == j) position_of_ij <-  position
                        }
                        corrs <-  c(corrs, gradient_by_parameter$off_raw[class_index, position_of_ij])
                  }
            }
            grad_main <-  c(corrs, gradient_by_parameter$beta_vec, gradient_by_parameter$p_raw)
            if (model_family == "ordinal") {
                  grad_main <-  c(grad_main, gradient_by_parameter$C_unc_vec[1, ],
                                  gradient_by_parameter$C_unc_vec[2, ])
            }
            return(list(grad_main = grad_main, grad_us = gradient_by_parameter$u_raw))
      }
      ##
      rows_of_check_c <-  list()
      for (Phi_type in c(1, 2)) {
            Stan_data_of_Phi_type <-  Stan_data
            Stan_data_of_Phi_type$ps6_col_order <-  NULL
            Stan_data_of_Phi_type$ps6_ordinal_slot_order <-  NULL
            Stan_data_of_Phi_type$chunk_size <-  as.integer(ceiling(family_data$N / num_chunks_cpp))
            Stan_data_of_Phi_type$Phi_type <-  as.integer(Phi_type)
            stan_model <-  bridgestan::StanModel$new(path_of_the_Stan_model_library,
                                                     as.character(jsonlite::toJSON(Stan_data_of_Phi_type,
                                                                                   auto_unbox = TRUE, digits = NA,
                                                                                   matrix = "rowmajor")),
                                                     1234)
            points <-  points_list[[model_family]]
            stan_positions <-  fn_stan_positions(stan_model, fn_stan_param_list(points[[1]]))
            for (vect_type in vect_types_to_check) for (evaluation in c("as sampled", "PartialLog", "autodiff")) {
                  cpp_model <-  fn_make_cpp_model(family_data, model_family, y = y_with_missing_results,
                                                  vect_type = vect_type, Phi_type = Phi_type)
                  for (point_name in names(points)) {
                        point <-  points[[point_name]]
                        param_list <-  fn_stan_param_list(point)
                        theta_stan <-  stan_model$param_unconstrain_json(fn_json_of_param_list(param_list))
                        stan_density <-  stan_model$log_density_gradient(theta_stan, propto = FALSE,
                                                                         jacobian = TRUE)
                        stan_output <-  fn_stan_grad_in_cpp_layout(stan_density$gradient, stan_positions,
                                                                   param_list)
                        theta_main <-  fn_theta_main_cpp(point, family_data, model_family)
                        cpp_output <-  fn_cpp_lp_grad(cpp_model, theta_main, point$u_unc, evaluation)
                        gradient_differences <-  c(cpp_output$grad_main - stan_output$grad_main,
                                                   cpp_output$grad_us - stan_output$grad_us)
                        rows_of_check_c[[length(rows_of_check_c) + 1]] <-  data.frame(
                              Phi_type = Phi_type,
                              vect_type = vect_type,
                              evaluation = evaluation,
                              point = point_name,
                              lp_difference = cpp_output$lp - stan_density$val,
                              largest_abs_grad_difference = max(abs(gradient_differences)),
                              largest_relative_grad_difference = max(abs(gradient_differences) /
                                                                     pmax(1, abs(c(stan_output$grad_main,
                                                                                   stan_output$grad_us)))))
                  }
            }
      }
      table_of_check_c <-  do.call(rbind, rows_of_check_c)
      name_of_the_Stan_model <-  sub("_model\\.so$", "", basename(path_of_the_Stan_model_library))
      utils::write.csv(table_of_check_c,
                       file.path(output_folder, paste0("check_c_against_the_Stan_model_", model_family, "_",
                                                       name_of_the_Stan_model, ".csv")),
                       row.names = FALSE)
      print(table_of_check_c, row.names = FALSE, digits = 3)
}
##
message("done")























