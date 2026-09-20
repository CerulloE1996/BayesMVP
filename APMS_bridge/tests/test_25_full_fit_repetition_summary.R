#### =====================================================================================================================================
## test_25_full_fit_repetition_summary.R - matching seeds, separate settings and narrow reporting; no model fits
## =====================================================================================================================================
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
source(file = file.path(dirname(path = tests_directory), "R_fn_APMS_full_fit_benchmark.R"))
##
fixture_directory <-  tempfile(pattern = "apms_repetition_summary_")
dir.create(path = fixture_directory)
for (run_index in seq_len(length.out = 4)) {
    n_burnin <-  if (run_index <= 2) 125 else 250
    fixture_result <-  list(
        row = tibble::tibble(configuration = "BayesMVP_AVX", engine = "BayesMVP", model = "M6_imperfect_ord", N = 500,
                             seed = run_index, status = if (run_index == 4) "error" else "completed",
                             error = if (run_index == 4) "Recorded failure" else "",
                             min_ESS_bulk = c(100, 300, 400, NA)[run_index],
                             ESS_per_sec_sampling = c(10, 30, 40, NA)[run_index],
                             native_burnin_seconds = c(5, 5, 40, NA)[run_index],
                             native_sampling_seconds = 40, engine_summary_seconds = 8, comparison_seconds = 2,
                             target_min_ESS = 100,
                             ## Stale values must be recomputed per fit, before averaging seeds.
                             estimated_time_to_target_ESS = c(10, 14, 20, NA)[run_index],
                             max_rhat = c(1.02, 1.20, 1.03, NA)[run_index]),
        definition = list(configuration = list(engine = "BayesMVP", sampler_settings = list(n_burnin = n_burnin, n_iter = 50,
                                                                                           seed = run_index)),
                          input = "identical_data", sources = "identical_code", seed = run_index))
    saveRDS(object = fixture_result, file = file.path(fixture_directory, paste0("test_seed", run_index, "_123456789abc.rds")))
}
##
fixture_files <-  list.files(path = fixture_directory, full.names = TRUE)
fingerprints_before <-  tools::md5sum(files = fixture_files)
benchmark_results <-  fn_read_APMS_full_fit_benchmark(output_dir = fixture_directory)
printed_report <-  capture.output(benchmark_summary <- fn_summarise_APMS_full_fit_benchmark(benchmark_results = benchmark_results))
stopifnot(nrow(x = benchmark_summary) == 2,
          all(benchmark_summary$n_runs == c(2, 2)),
          all(benchmark_summary$n_completed == c(2, 1)),
          benchmark_summary$min_ESS_bulk[1] == 200,
          benchmark_summary$ESS_per_sec_sampling[1] == 20,
          abs(benchmark_summary$estimated_time_to_target_ESS[1] - mean(x = c(5 + 50 * 100 / 100, 5 + 50 * 100 / 300))) < 1e-12,
          abs(benchmark_summary$max_rhat[1] - 1.11) < 1e-12,
          benchmark_summary$n_failed[2] == 1,
          max(nchar(x = printed_report)) <= 100,
          any(grepl(pattern = "time_to_target_ESS", x = printed_report, fixed = TRUE)),
          any(grepl(pattern = "Time units: sec", x = printed_report, fixed = TRUE)),
          !any(grepl(pattern = "Target s", x = printed_report, fixed = TRUE)),
          identical(x = fingerprints_before, y = tools::md5sum(files = fixture_files)))
##
sorted_summary <-  fn_summarise_APMS_full_fit_benchmark(benchmark_results = benchmark_results,
                                                       sort_by = "ESS_per_sec_sampling", print_table = FALSE)
stopifnot(sorted_summary$min_ESS_bulk[1] == 400)
##
## Printed metrics and settings remain numeric columns; filtering never changes within-configuration averages.
stopifnot(is.numeric(x = benchmark_summary$time_to_target_ESS),
          identical(x = benchmark_summary$time_to_target_ESS, y = benchmark_summary$estimated_time_to_target_ESS),
          identical(x = benchmark_summary$burnin_sec, y = benchmark_summary$native_burnin_seconds),
          identical(x = benchmark_summary$min_ESS, y = benchmark_summary$min_ESS_bulk))
filtered_summary <-  fn_summarise_APMS_full_fit_benchmark(benchmark_results = benchmark_results, print_table = FALSE,
                                                         filter_values = list(N = 500, n_burnin = 250, n_iter = 50))
stopifnot(nrow(x = filtered_summary) == 1, filtered_summary$id == benchmark_summary$id[2],
          filtered_summary$n_runs == 2, filtered_summary$time_to_target_ESS == benchmark_summary$time_to_target_ESS[2])
stopifnot(nrow(x = dplyr::filter(.data = benchmark_summary, min_ESS > 300, n_burnin == 250)) == 1)
empty_selection <-  fn_summarise_APMS_full_fit_benchmark(benchmark_results = benchmark_results, print_table = FALSE,
                                                       filter_values = list(N = 999))
stopifnot(nrow(x = empty_selection) == 0, "time_to_target_ESS" %in% names(x = empty_selection))
unknown_filter_error <-  tryCatch(expr = {
    fn_summarise_APMS_full_fit_benchmark(benchmark_results = benchmark_results, print_table = FALSE, filter_values = list(typo = 1))
    FALSE
}, error = function(condition) grepl(pattern = "Unknown filter column", x = conditionMessage(c = condition)))
stopifnot(unknown_filter_error)
## A changed implementation must remain separate even when the visible settings match.
benchmark_results$repetition_group[2] <-  "different_code"
stopifnot(nrow(x = fn_summarise_APMS_full_fit_benchmark(benchmark_results = benchmark_results, print_table = FALSE)) == 3)
stopifnot(nrow(x = fn_summarise_APMS_full_fit_benchmark(benchmark_results = tibble::tibble(), print_table = FALSE)) == 0)
cat("PASS: seeds combined, settings/code separated, failed fits retained, available metrics, sorting, narrow printing and read-only refresh. No fits.\n")
