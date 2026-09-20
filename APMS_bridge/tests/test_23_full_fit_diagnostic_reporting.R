#### =====================================================================================================================================
## test_23_full_fit_diagnostic_reporting.R - independent diagnostic availability and read-only report refresh; no model fits
## =====================================================================================================================================
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
bridge_directory <-  dirname(path = tests_directory)
workspace_directory <-  normalizePath(path = file.path(bridge_directory, "..", "..", ".."))
source(file = file.path(bridge_directory, "R_fn_APMS_full_fit_benchmark.R"))
##
## ---- Undefined tail ESS must not hide finite bulk ESS or R-hat; constants do not enter any aggregate ----------------------------------
##
## Deliberately keep legacy data.frame fixtures too: saved fits from before the tibble refactor must remain readable.
comparison_summary <-  data.frame( variable = c("Se_ord[1,1]", "prev_A_overall", "constant_padding[1]"),
                                    sd = c(0.01, 0.02, 0),
                                    ess_bulk = c(500, 1000, NA_real_),
                                    ess_tail = c(NA_real_, 250, NA_real_),
                                    rhat = c(1.02, 1.01, NA_real_))
comparison_metrics <-  fn_APMS_benchmark_comparison_metrics(comparison_summary = comparison_summary)
stopifnot(tibble::is_tibble(x = comparison_metrics), comparison_metrics$min_ESS_bulk == 500, comparison_metrics$max_rhat == 1.02,
          is.na(x = comparison_metrics$min_ESS_tail), !comparison_metrics$diagnostics_complete,
          comparison_metrics$n_monitored_active == 2, comparison_metrics$n_monitored_constant == 1,
          comparison_metrics$n_missing_tail_ESS == 1, comparison_metrics$n_missing_bulk_ESS == 0,
          identical(x = comparison_metrics$undefined_tail_ESS_variables, y = "Se_ord[1,1]"))
##
missing_bulk_summary <-  comparison_summary
missing_bulk_summary$ess_bulk[1] <-  NA_real_
missing_bulk_summary$ess_tail[1] <-  300
missing_bulk_metrics <-  fn_APMS_benchmark_comparison_metrics(comparison_summary = missing_bulk_summary)
stopifnot(is.na(x = missing_bulk_metrics$min_ESS_bulk), missing_bulk_metrics$min_ESS_tail == 250,
          missing_bulk_metrics$max_rhat == 1.02, !missing_bulk_metrics$diagnostics_complete)
##
missing_rhat_summary <-  comparison_summary
missing_rhat_summary$rhat[1] <-  NA_real_
missing_rhat_summary$ess_tail[1] <-  300
missing_rhat_metrics <-  fn_APMS_benchmark_comparison_metrics(comparison_summary = missing_rhat_summary)
stopifnot(missing_rhat_metrics$min_ESS_bulk == 500, missing_rhat_metrics$min_ESS_tail == 250,
          is.na(x = missing_rhat_metrics$max_rhat), !missing_rhat_metrics$diagnostics_complete)
##
invalid_sd_summary <-  comparison_summary
invalid_sd_summary$sd[1] <-  NA_real_
invalid_sd_metrics <-  fn_APMS_benchmark_comparison_metrics(comparison_summary = invalid_sd_summary)
stopifnot(invalid_sd_metrics$n_nonfinite_sd == 1, !invalid_sd_metrics$diagnostics_complete,
          is.na(x = invalid_sd_metrics$min_ESS_bulk), is.na(x = invalid_sd_metrics$max_rhat))
constant_metrics <-  fn_APMS_benchmark_comparison_metrics(comparison_summary = comparison_summary[3, , drop = FALSE])
stopifnot(constant_metrics$n_monitored_active == 0, !constant_metrics$diagnostics_complete)
##
## ---- Reproduce the bounded-tail case with posterior itself, without changing or jittering any draws ----------------------------------
##
bounded_probability_draws <-  matrix(data = rep(x = c(seq(from = 0.97, to = 0.999, length.out = 80), rep(x = 1, times = 20)),
                                               times = 4), nrow = 100)
bounded_probability_summary <-  data.frame( variable = "Se_ord[1,1]",
                                            sd = sd(x = bounded_probability_draws),
                                            ess_bulk = posterior::ess_bulk(x = bounded_probability_draws),
                                            ess_tail = posterior::ess_tail(x = bounded_probability_draws),
                                            rhat = posterior::rhat(x = bounded_probability_draws))
bounded_metrics <-  fn_APMS_benchmark_comparison_metrics(comparison_summary = bounded_probability_summary)
stopifnot(is.finite(x = bounded_metrics$min_ESS_bulk), is.finite(x = bounded_metrics$max_rhat),
          is.na(x = bounded_metrics$min_ESS_tail), !bounded_metrics$diagnostics_complete)
##
## ---- Repair saved aggregates and every bulk-ESS ratio, preserving timings, draws and the failed diagnostic gate -----------------------
##
saved_result <-  list(
    row = data.frame(configuration = "cmdstanr_plain", engine = "cmdstanr", model = "M5_imperfect_bin", N = 500, seed = 123,
                     status = "completed", min_ESS_bulk = NA_real_, min_ESS_tail = NA_real_, max_rhat = NA_real_,
                     native_burnin_seconds = 5, native_sampling_seconds = 20, engine_summary_seconds = 2, comparison_seconds = 0,
                     full_fit_seconds = 28, fit_excluding_setup_seconds = 27, setup_seconds = 1, job_wall_seconds = 29,
                     n_grad_sampling = 10000, target_min_ESS = 100, n_divergent = 2, diagnostics_pass = FALSE),
    comparison_summary = comparison_summary,
    draws = bounded_probability_draws)
refreshed_result <-  fn_refresh_APMS_full_fit_benchmark_result(result = saved_result)
stopifnot(tibble::is_tibble(x = refreshed_result$row), refreshed_result$row$min_ESS_bulk == 500, refreshed_result$row$ESS_per_sec_sampling == 25,
          refreshed_result$row$ESS_per_sec_full_fit == 500 / 28, refreshed_result$row$ESS_per_sec_excluding_setup == 500 / 27,
          refreshed_result$row$ESS_per_grad_sampling == 0.05, refreshed_result$row$ESS_per_1000_grad_sampling == 50,
          abs(refreshed_result$row$estimated_time_to_target_ESS - 9.4) < 1e-12, !refreshed_result$row$diagnostics_pass,
          is.na(x = saved_result$row$min_ESS_bulk), identical(x = refreshed_result$draws, y = saved_result$draws))
##
stopifnot(fn_APMS_benchmark_time_to_target_ESS(burnin_sec = 5, sampling_sec = 20, engine_summary_sec = 2,
                                              comparison_summary_sec = 0.5, target_ESS = 100, observed_ESS = 500) == 9.5,
          fn_APMS_benchmark_time_to_target_ESS(burnin_sec = 5, sampling_sec = 20, engine_summary_sec = 2,
                                              comparison_summary_sec = 0.5, target_ESS = 1000, observed_ESS = 500) == 50,
          is.na(fn_APMS_benchmark_time_to_target_ESS(burnin_sec = 5, sampling_sec = 20, engine_summary_sec = NA_real_,
                                                    comparison_summary_sec = 0.5, target_ESS = 100, observed_ESS = 500)))
## Setup is deliberately separate and never enters this estimate.
expensive_setup_result <-  saved_result
expensive_setup_result$row$setup_seconds <-  10000
stopifnot(fn_refresh_APMS_full_fit_benchmark_result(result = expensive_setup_result)$row$estimated_time_to_target_ESS ==
              refreshed_result$row$estimated_time_to_target_ESS)
##
printed_output <-  capture.output(printed_row <- fn_print_APMS_full_fit_benchmark_result(result = saved_result))
stopifnot(printed_row$min_ESS_bulk == 500,
          any(grepl(pattern = "min_ESS (bulk) = 500", x = printed_output, fixed = TRUE)),
          any(grepl(pattern = "Undefined tail_ESS: Se_ord[1,1]", x = printed_output, fixed = TRUE)),
          any(grepl(pattern = "tail ESS 1/2", x = printed_output, fixed = TRUE)))
##
## ---- Each specification and metric gets its own line for both engines; formatting leaves the returned values unchanged ----------------
##
expected_standalone_lines <-  c("Configuration: cmdstanr_plain", "Model: M5_imperfect_bin", "Seed: 123", "Status: COMPLETED",
                                "Setup/compile = 1.0 s", "burn-in = 5.0 s", "sampling = 20.0 s", "engine summaries = 2.0 s",
                                "Full fit = 28.0 s", "excluding setup = 27.0 s", "job wall = 29.0 s",
                                "min_ESS (bulk) = 500", "min_ESS (tail) = NA", "max R-hat = 1.0200",
                                "ESS/sec [sampling] = 25.000", "ESS/grad [estimated] = 0.050000",
                                "ESS/1000 grads [estimated] = 50.000", "Sampling acceptance = NA", "epsilon = NA",
                                "mean L = NA", "divergences = 2", "treedepth hits = NA")
stopifnot(all(expected_standalone_lines %in% printed_output),
          !any(grepl(pattern = " | ", x = printed_output, fixed = TRUE)),
          identical(x = printed_row, y = refreshed_result$row))
##
bayesmvp_print_result <-  saved_result
bayesmvp_print_result$row$configuration <-  "BayesMVP_AVX"
bayesmvp_print_result$row$engine <-  "BayesMVP"
bayesmvp_print_result$row$min_ESS_main <-  134
bayesmvp_print_result$row$max_rhat_main <-  1.1865
bayesmvp_print_result$row$max_nested_rhat_main <-  NA_real_
bayesmvp_print_output <-  capture.output(fn_print_APMS_full_fit_benchmark_result(result = bayesmvp_print_result))
stopifnot(all(c("Configuration: BayesMVP_AVX", "Raw main block:", "  min ESS = 134", "  max R-hat = 1.1865",
                "  max nested R-hat = NA (NA if unavailable/disabled)") %in% bayesmvp_print_output),
          !any(grepl(pattern = " | ", x = bayesmvp_print_output, fixed = TRUE)))
##
complete_result <-  saved_result
complete_result$comparison_summary$ess_tail[1] <-  300
complete_result$row$diagnostics_pass <-  TRUE
complete_result$row$n_divergent <-  0
complete_result <-  fn_refresh_APMS_full_fit_benchmark_result(result = complete_result)
stopifnot(complete_result$row$diagnostics_complete, complete_result$row$diagnostics_pass, complete_result$row$min_ESS_tail == 250)
failed_result <-  list(row = data.frame(status = "error", error = "Existing worker failure."))
refreshed_failed_result <-  fn_refresh_APMS_full_fit_benchmark_result(result = failed_result)
stopifnot(identical(x = refreshed_failed_result$row$status, y = failed_result$row$status),
          identical(x = refreshed_failed_result$row$error, y = failed_result$row$error),
          is.na(x = refreshed_failed_result$row$n_burnin), is.na(x = refreshed_failed_result$row$n_iter))
##
## ---- Iterations come from the saved fit, for both engines and both completed/failed records -------------------------------------------
##
for (engine in c("BayesMVP", "cmdstanr")) {
    saved_iteration_settings <-  if (engine == "BayesMVP") list(n_burnin = 250, n_iter = 125) else
        list(iter_warmup = 500, iter_sampling = 1000)
    expected_iterations <-  if (engine == "BayesMVP") c(250, 125) else c(500, 1000)
    for (status in c("completed", "error")) {
        for (settings_location in c("settings", "configuration", "definition")) {
            iteration_fixture <-  list(row = tibble::tibble(engine = engine, status = status))
            if (settings_location == "settings") iteration_fixture$settings <-  saved_iteration_settings
            if (settings_location == "configuration") iteration_fixture$configuration <-  list(sampler_settings = saved_iteration_settings)
            if (settings_location == "definition") iteration_fixture$definition <-  list(configuration = list(sampler_settings = saved_iteration_settings))
            iteration_result <-  fn_refresh_APMS_full_fit_benchmark_result(result = iteration_fixture)
            stopifnot(identical(x = c(iteration_result$row$n_burnin, iteration_result$row$n_iter), y = expected_iterations))
            ## Already-recorded row counts must not be replaced by other settings.
            iteration_result$row$n_burnin <-  75
            iteration_result$row$n_iter <-  30
            iteration_result <-  fn_refresh_APMS_full_fit_benchmark_result(result = iteration_result)
            stopifnot(iteration_result$row$n_burnin == 75, iteration_result$row$n_iter == 30)
        }
    }
}
##
## ---- The directory reader updates both engines in memory; original RDS files are byte-for-byte unchanged ------------------------------
##
fixture_directory <-  tempfile(pattern = "apms_diagnostic_reporting_")
dir.create(path = fixture_directory)
stan_fixture_file <-  file.path(fixture_directory, "cmdstanr_plain_M5_imperfect_bin_seed123_aaaaaaaaaaaa.rds")
bayesmvp_fixture_file <-  file.path(fixture_directory, "BayesMVP_plain_M6_imperfect_ord_seed123_bbbbbbbbbbbb.rds")
saveRDS(object = saved_result, file = stan_fixture_file)
saved_result$row$configuration <-  "BayesMVP_plain"
saved_result$row$engine <-  "BayesMVP"
saved_result$row$model <-  "M6_imperfect_ord"
saveRDS(object = saved_result, file = bayesmvp_fixture_file)
fixture_files <-  c(stan_fixture_file, bayesmvp_fixture_file)
checksums_before_reading <-  tools::md5sum(files = fixture_files)
refreshed_table <-  fn_read_APMS_full_fit_benchmark(output_dir = fixture_directory)
stopifnot(tibble::is_tibble(x = refreshed_table), nrow(x = refreshed_table) == 2, all(refreshed_table$min_ESS_bulk == 500),
          all(refreshed_table$ESS_per_sec_sampling == 25), all(refreshed_table$n_missing_tail_ESS == 1),
          all(!refreshed_table$diagnostics_pass), all(is.na(x = refreshed_table$min_ESS_tail)),
          identical(x = tools::md5sum(files = fixture_files), y = checksums_before_reading))
##
## Mixed legacy rows, new tibbles and error-only rows bind by name, without filtering or overwriting saved fits.
tibble_fixture_file <-  file.path(fixture_directory, "BayesMVP_AVX_M6_imperfect_ord_seed123_cccccccccccc.rds")
failed_fixture_file <-  file.path(fixture_directory, "cmdstanr_AVX_M6_imperfect_ord_seed123_dddddddddddd.rds")
saveRDS(object = refreshed_result, file = tibble_fixture_file)
saveRDS(object = failed_result, file = failed_fixture_file)
fixture_files <-  c(fixture_files, tibble_fixture_file, failed_fixture_file)
checksums_before_reading <-  tools::md5sum(files = fixture_files)
mixed_results <-  fn_read_APMS_full_fit_benchmark(output_dir = fixture_directory)
stopifnot(tibble::is_tibble(x = mixed_results), nrow(x = mixed_results) == 4,
          sum(mixed_results$status == "error") == 1,
          all(mixed_results$min_ESS_bulk[mixed_results$status == "completed"] == 500),
          is.na(x = mixed_results$min_ESS_bulk[mixed_results$status == "error"]),
          identical(x = tools::md5sum(files = fixture_files), y = checksums_before_reading))
unlink(x = fixture_files)
empty_results <-  fn_read_APMS_full_fit_benchmark(output_dir = fixture_directory)
stopifnot(tibble::is_tibble(x = empty_results), nrow(x = empty_results) == 0)
unlink(x = fixture_directory, recursive = FALSE)
##
invisible(x = parse(file = file.path(workspace_directory, "Autism_BPD_paper", "R_benchmark_APMS_full_fits.R")))
cat("PASS: independent bulk/tail/R-hat availability, bounded-tail NA, unchanged diagnostic gate, both-engine ratios and read-only saved-report refresh. No fits.\n")
