#### =====================================================================================================================================
## test_20_ps7_sampling_acceptance_summary.R - saved probability extraction, aggregation, display and sorting; no model fits
## =====================================================================================================================================
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
workspace_directory <-  normalizePath(path = file.path(tests_directory, "..", "..", "..", ".."))
ps7_directory <-  file.path(workspace_directory, "Alg_paper_analysis", "1_appendix_pilot_studies",
                            "ps_7_basic_MCMC_settings_BayesMVP")
source(file = file.path(ps7_directory, "functions", "fn_ps7_extract_tau_sweep.R"))
source(file = file.path(ps7_directory, "functions", "fn_ps7_summarise_runs.R"))
source(file = file.path(workspace_directory, "R_packages", "BayesMVP", "inst", "BayesMVP", "R", "R_fn_create_superchain_ids.R"))
##
## ---- Small saved-run fixtures: probabilities, not accepted/rejected indicators -------------------------------------------------------
##
fixture_directory <-  tempfile(pattern = "ps7_acceptance_summary_")
dir.create(path = fixture_directory)
fixture_run <-  list( min_ESS = 1000,
                      max_Rhat = 1.02,
                      ESS_per_grad_samp = 0.02,
                      ESS_per_sec_samp = 100,
                      divergences = list(pct_divs = 0),
                      HMC_info = list(n_superchains = 4, eps_main = 0.2, tau_main = 2),
                      efficiency_info = list(Min_ESS_main = 100, Max_rhat_main = 1.1, Max_nested_rhat_main = 1.01,
                                             time_burnin = 3, time_sampling = 10, time_summaries = 1, time_total = 14,
                                             L_main_during_sampling = 10, grad_evals_per_sec = 5000))
fixture_probabilities <-  list(matrix(data = c(0.5, 0.7, 0.9, 0.3), nrow = 2),
                               matrix(data = 0.8, nrow = 2, ncol = 2),
                               matrix(data = 0, nrow = 2, ncol = 2),
                               NULL,
                               matrix(data = c(0.7, NA_real_), nrow = 2),
                               matrix(data = c(0.7, 1.2), nrow = 2))
fixture_names <-  paste0("ps7_run_LC_MVP_N", c(2500, 2500, 10000, 500, 50000, 750),
                         "_np2_ta2_cb4_s2_wb1_s1_kb1_s1_b125_it2_LR.075_AD.8_tipi_Ms_trS_run", c(1, 2, 1, 1, 1, 1))
for (fixture_index in seq_along(along.with = fixture_names)) {
    fixture_run$sampler_diagnostics <-  list(sampling = list(p_jump_main = fixture_probabilities[[fixture_index]],
                                                            p_jump_nuisance = matrix(data = 1, nrow = 2, ncol = 2)))
    saveRDS(object = fixture_run, file = file.path(fixture_directory, fixture_names[fixture_index]))
}
extracted_runs <-  fn_ps7_extract_tau_sweep( dir = fixture_directory,
                                            pattern = "^ps7_run_",
                                            verbose = FALSE)
expected_probabilities <-  c(0.6, 0.8, 0, NA_real_, NA_real_, NA_real_)
actual_probabilities <-  extracted_runs$sampling_acceptance_probability[match(x = fixture_names, table = extracted_runs$file)]
stopifnot(isTRUE(all.equal(target = actual_probabilities, current = expected_probabilities)))
##
## ---- Mean of per-run means, default display, full-name/alias sorting, missing values last ---------------------------------------------
##
printed_summary <-  capture.output(summary_result <- fn_ps7_summarise_runs( runs_table = extracted_runs,
                                                                          target_min_ESS = 1000,
                                                                          sort_by = "accept_prob",
                                                                          sort_within_N = FALSE))
stopifnot(abs(summary_result$configurations$mean_sampling_acceptance_probability[1] - 0.7) < 1e-12,
          summary_result$configurations$mean_sampling_acceptance_probability[2] == 0,
          all(is.na(summary_result$configurations$mean_sampling_acceptance_probability[3:5])),
          any(grepl(pattern = "accept_prob", x = printed_summary)))
for (metric_name in c("accept_prob", "sampling_acceptance", "sampling_acceptance_probability", "mean_sampling_acceptance_probability")) {
    metric_specification <-  R_fn_ps7_resolve_metric_names(requested_metric_names = metric_name, argument_name = "sort_by")
    stopifnot(metric_specification$column_name == "mean_sampling_acceptance_probability", metric_specification$bigger_is_better)
}
stopifnot("accept_prob" %in% eval(expr = formals(fun = fn_ps7_summarise_directory)$columns_to_print))
##
## ---- Cached tables predating this field remain readable, with an explicit re-extraction warning --------------------------------------
##
extracted_runs$sampling_acceptance_probability <-  NULL
missing_acceptance_warning <-  FALSE
old_cache_summary <-  withCallingHandlers(
    expr = fn_ps7_summarise_runs(runs_table = extracted_runs, target_min_ESS = 1000, print_table = FALSE),
    warning = function(condition) {
        if (grepl(pattern = "Sampling acceptance is absent", x = conditionMessage(c = condition))) {
            missing_acceptance_warning <<-  TRUE
            invokeRestart(r = "muffleWarning")
        }
    })
stopifnot(missing_acceptance_warning, all(is.na(old_cache_summary$configurations$mean_sampling_acceptance_probability)))
cat("PASS: post-burn-in probability means, missing/invalid diagnostics, configuration averaging, printing, sorting and old caches. No fits.\n")
