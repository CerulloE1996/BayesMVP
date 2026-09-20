#### =====================================================================================================================================
## test_18_ps7_target_timing.R - target-time arithmetic and summary aggregation; no model fits
## =====================================================================================================================================
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
workspace_directory <-  normalizePath(path = file.path(tests_directory, "..", "..", "..", ".."))
ps7_functions_directory <-  file.path(workspace_directory, "Alg_paper_analysis", "1_appendix_pilot_studies",
                                     "ps_7_basic_MCMC_settings_BayesMVP", "functions")
source(file = file.path(ps7_functions_directory, "fn_ps7_summarise_runs.R"))
source(file = file.path(ps7_functions_directory, "fn_ps7_extract_tau_sweep.R"))
invisible(parse(file = file.path(ps7_functions_directory, "ps_7_MCMC_settings_BayesMVP_functions.R")))
##
## ---- Known arithmetic: keep burn-in fixed, scale BOTH sampling and summaries, then average the per-run totals ----------------------------
##
test_runs <-  data.frame( configuration = c("first", "first", "second"),
                          model_type = "LC_MVP",
                          N = c(10000, 10000, 2500),
                          n_chains_burnin = 4,
                          n_chains_sampling = 180,
                          num_chunks_burnin = 25,
                          n_iter = 50,
                          file = c("first_run1", "first_run2", "second_run1"),
                          readable = TRUE,
                          min_ESS = c(2000, 500, 1000),
                          min_ESS_main = 200,
                          max_Rhat_main = 1.1,
                          max_nRhat_main = 1.01,
                          max_Rhat = 1.04,
                          max_nRhat = 1.01,
                          pct_divs = 0,
                          ESS_per_grad_samp = 0.02,
                          ESS_per_sec_samp = 100,
                          grad_evals_per_sec = 5000,
                          eps_main = 0.2,
                          L_main_samp = 10,
                          time_burnin = c(10, 20, 4),
                          time_sampling = c(8, 6, 2),
                          time_summaries = c(2, 4, 1))
##
test_summary <-  fn_ps7_summarise_runs( runs_table = test_runs,
                                       target_min_ESS = 1000,
                                       print_table = FALSE)
stopifnot(identical(test_summary$runs$time_to_target_min_ESS, c(15, 40, 7)))
first_configuration <-  test_summary$configurations[test_summary$configurations$N == 10000, ]
stopifnot(first_configuration$mean_time_burnin == 15, first_configuration$mean_time_to_target_min_ESS == 27.5)
stopifnot(all(test_summary$configurations$mean_time_to_target_min_ESS >= test_summary$configurations$mean_time_burnin))
##
per_N_summary <-  fn_ps7_summarise_runs( runs_table = test_runs,
                                        target_min_ESS = c("10000" = 1000, "2500" = 2500),
                                        print_table = FALSE)
stopifnot(identical(per_N_summary$runs$time_to_target_min_ESS, c(15, 40, 11.5)))
##
## ---- Incomplete timings must not silently average a faster subset against the full group's burn-in -------------------------------------
##
test_runs$time_summaries[2] <-  NA_real_
missing_timing_warning <-  FALSE
missing_summary <-  withCallingHandlers(
    expr = fn_ps7_summarise_runs( runs_table = test_runs,
                                  target_min_ESS = 1000,
                                  print_table = FALSE),
    warning = function(condition) {
        missing_timing_warning <<-  TRUE
        invokeRestart(r = "muffleWarning")
    })
stopifnot(missing_timing_warning)
stopifnot(is.na(missing_summary$configurations$mean_time_to_target_min_ESS[missing_summary$configurations$N == 10000]))
##
for (invalid_target in c(0, -100)) {
    target_error <-  tryCatch(expr = fn_ps7_summarise_runs( runs_table = test_runs,
                                                          target_min_ESS = invalid_target,
                                                          print_table = FALSE),
                              error = function(condition) condition)
    stopifnot(inherits(x = target_error, what = "error"))
}
##
cat("PASS: scaled sampling AND summaries, unscaled burn-in, scalar/per-N targets, missing-data aggregation and syntax. No model fits.\n")
