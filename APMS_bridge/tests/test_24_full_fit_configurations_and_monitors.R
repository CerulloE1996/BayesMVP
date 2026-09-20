#### =====================================================================================================================================
## test_24_full_fit_configurations_and_monitors.R - evaluate only configuration construction and diagnostic selection; no model fits
## =====================================================================================================================================
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
bridge_directory <-  dirname(path = tests_directory)
workspace_directory <-  normalizePath(path = file.path(bridge_directory, "..", "..", ".."))
source(file = file.path(bridge_directory, "R_fn_APMS_full_fit_benchmark.R"))
driver_expressions <-  as.list(x = parse(file = file.path(workspace_directory, "Autism_BPD_paper", "R_benchmark_APMS_full_fits.R")))
benchmark_blocks <-  Filter(f = function(expression) {
    is.call(x = expression) && identical(x = expression[[1]], y = as.name(x = "if")) &&
        identical(x = expression[[2]], y = as.name(x = "RUN_BENCHMARK"))
}, x = driver_expressions)
stopifnot(length(x = benchmark_blocks) == 1)
benchmark_expressions <-  as.list(x = benchmark_blocks[[1]][[3]])[-1]
##
## ---- Run only the driver's configuration assignments, with marker settings instead of any application preparation ---------------------
##
configuration_environment <-  new.env(parent = globalenv())
configuration_environment$BASE_DIR <-  file.path(workspace_directory, "Autism_BPD_paper")
configuration_environment$BRIDGE_DIR <-  bridge_directory
configuration_environment$CMDSTANR_NUM_CHUNKS <-  32
configuration_environment$CMDSTANR_SAMPLER_OPTIONS <-  list(marker = "unchanged CmdStan settings")
configuration_environment$bayesmvp_sampler_settings <-  list(marker = "unchanged BayesMVP settings")
for (expression in benchmark_expressions) {
    if (!is.call(x = expression)) next
    if (identical(x = expression[[1]], y = as.name(x = "<-"))) {
        assignment_target <-  expression[[2]]
        if (is.symbol(x = assignment_target) && as.character(x = assignment_target) %in% c("configurations", "AVX_HEADER", "AVX_DEPENDENCIES")) {
            eval(expr = expression, envir = configuration_environment)
        } else if (is.call(x = assignment_target) && identical(x = assignment_target[[1]], y = as.name(x = "$")) &&
                   identical(x = assignment_target[[2]], y = as.name(x = "configurations"))) {
            eval(expr = expression, envir = configuration_environment)
        }
    } else if (identical(x = expression[[1]], y = as.name(x = "for")) &&
               identical(x = expression[[2]], y = as.name(x = "configuration_name"))) {
        eval(expr = expression, envir = configuration_environment)
    }
}
configurations <-  configuration_environment$configurations
stopifnot(setequal(x = names(x = configurations), y = c("cmdstanr_plain", "cmdstanr_AVX", "BayesMVP_AVX")),
          length(x = configurations) == 3,
          identical(x = configurations$BayesMVP_AVX$engine, y = "BayesMVP"),
          identical(x = configurations$cmdstanr_plain$engine, y = "cmdstanr"),
          identical(x = configurations$cmdstanr_AVX$engine, y = "cmdstanr"),
          identical(x = basename(path = configurations$cmdstanr_plain$stan_file), y = "LC_MVOP_4class_joint_v1_reduce_sum.stan"),
          is.null(x = configurations$cmdstanr_plain$user_header),
          identical(x = configurations$BayesMVP_AVX$sampler_settings, y = configuration_environment$bayesmvp_sampler_settings),
          identical(x = configurations$cmdstanr_plain$sampler_settings, y = configuration_environment$CMDSTANR_SAMPLER_OPTIONS),
          identical(x = configurations$cmdstanr_AVX$sampler_settings, y = configuration_environment$CMDSTANR_SAMPLER_OPTIONS))
for (configuration_name in c("BayesMVP_AVX", "cmdstanr_AVX")) {
    configuration <-  configurations[[configuration_name]]
    stopifnot(identical(x = basename(path = configuration$stan_file), y = "LC_MVOP_4class_joint_v1_reduce_sum_AVX.stan"),
              identical(x = configuration$user_header, y = configuration_environment$AVX_HEADER),
              identical(x = configuration$source_dependencies, y = configuration_environment$AVX_DEPENDENCIES),
              all(file.exists(c(configuration$stan_file, configuration$user_header, configuration$source_dependencies))))
}
##
## ---- The actual common worker block selects Se/Sp/prevalence, excluding raw main and nuisance quantities ------------------------------
##
interest_variables <-  c("Se_bin[1,1]", "Sp_bin[1,1]", "Se_ord[1,1]", "Sp_ord[1,1]", "Se_anchor_ord[1,1]", "Sp_anchor_ord[1,1]",
                         "prev_A_overall", "prev_B_overall", "pi_00_overall", "pi_10_overall", "pi_01_overall", "pi_11_overall",
                         "prev_A_par[1]", "prev_B_par[1]")
excluded_variables <-  c("beta[1,1]", "Omega[1,1,1]", "u_raw[1,1]", "lp__", "Se_ord_unconstrained[1]", "Sp_bin_raw[1]", "PPV[1]", "cutpoint[1]")
draw_variable_names <-  c(interest_variables, excluded_variables)
draws <-  posterior::as_draws_array(x = array(data = sin(seq_len(length.out = 20 * 4 * length(x = draw_variable_names))),
                                             dim = c(20, 4, length(x = draw_variable_names)),
                                             dimnames = list(NULL, NULL, draw_variable_names)))
job <-  list(monitor_pattern = eval(expr = formals(fun = fn_run_APMS_full_fit_benchmark)$monitor_pattern))
worker_expressions <-  as.list(x = body(fun = fn_APMS_full_fit_benchmark_worker))[-1]
selection_assignment_names <-  c("available_variables", "monitored_variables", "monitored_draws", "comparison_summary", "comparison_metrics")
for (expression in worker_expressions) {
    if (is.call(x = expression) && identical(x = expression[[1]], y = as.name(x = "<-")) && is.symbol(x = expression[[2]]) &&
        as.character(x = expression[[2]]) %in% selection_assignment_names) eval(expr = expression)
}
stopifnot(identical(x = monitored_variables, y = interest_variables),
          setequal(x = comparison_summary$variable, y = interest_variables),
          !any(comparison_summary$variable %in% excluded_variables))
##
## The application call uses this shared default, not a separate engine-specific monitor override.
benchmark_call_expressions <-  Filter(f = function(expression) {
    is.call(x = expression) && identical(x = expression[[1]], y = as.name(x = "<-")) && is.call(x = expression[[3]]) &&
        identical(x = expression[[3]][[1]], y = as.name(x = "fn_run_APMS_full_fit_benchmark"))
}, x = benchmark_expressions)
stopifnot(length(x = benchmark_call_expressions) == 1,
          !"monitor_pattern" %in% names(x = benchmark_call_expressions[[1]][[3]]))
##
## ---- The reporting block delegates aggregation, retaining failures within the driver's selected N -------------------------------------
##
report_blocks <-  Filter(f = function(expression) {
    is.call(x = expression) && identical(x = expression[[1]], y = as.name(x = "if")) &&
        identical(x = expression[[2]], y = quote(nrow(benchmark_results)))
}, x = driver_expressions)
stopifnot(length(x = report_blocks) == 1)
report_environment <-  new.env(parent = globalenv())
report_environment$`%>%` <-  getExportedValue(ns = "dplyr", name = "%>%")
report_environment$benchmark_results <-  tibble::tibble(
    configuration = c("passing_run", "failed_diagnostics_run", "worker_error_run", "missing_diagnostics_run"),
    engine = c("BayesMVP", "cmdstanr", "BayesMVP", "cmdstanr"),
    model = "M6_imperfect_ord", N = 500, seed = c(123, 1000, 2000, 3000),
    status = c("completed", "completed", "error", "completed"),
    diagnostics_pass = c(TRUE, FALSE, FALSE, NA),
    min_ESS_bulk = c(500, 80, NA, 120), error = c("", "", "Recorded worker error", ""))
printed_report_tables <-  list()
report_environment$fn_summarise_APMS_full_fit_benchmark <-  function(benchmark_results, sort_by, print_table = TRUE, filter_values = NULL) {
    stopifnot(sort_by == "time_to_target_ESS")
    stopifnot(identical(x = filter_values, y = list(N = 500)))
    printed_report_tables[[length(x = printed_report_tables) + 1]] <<-  benchmark_results
    invisible(x = benchmark_results)
}
eval(expr = report_blocks[[1]], envir = report_environment)
stopifnot(length(x = printed_report_tables) == 1)
for (printed_table in printed_report_tables) {
    stopifnot(identical(x = printed_table$configuration, y = report_environment$benchmark_results$configuration),
              identical(x = printed_table$N, y = report_environment$benchmark_results$N),
              identical(x = printed_table$diagnostics_pass, y = report_environment$benchmark_results$diagnostics_pass),
              identical(x = printed_table$error, y = report_environment$benchmark_results$error))
}
## A directory containing only failed fits need not have ESS columns yet.
report_environment$benchmark_results <-  report_environment$benchmark_results[3, c("configuration", "N", "status", "error")]
printed_report_tables <-  list()
eval(expr = report_blocks[[1]], envir = report_environment)
stopifnot(length(x = printed_report_tables) == 1, nrow(x = printed_report_tables[[1]]) == 1,
          identical(x = printed_report_tables[[1]]$status, y = "error"))
report_environment$benchmark_results <-  tibble::tibble()
printed_report_tables <-  list()
eval(expr = report_blocks[[1]], envir = report_environment)
stopifnot(length(x = printed_report_tables) == 0)
cat("PASS: repetition reporting retains failed/missing diagnostics and worker errors within the selected N; empty/error-only inputs work.\n")
cat("PASS: three requested configurations with correct model/header paths, unchanged settings, and shared Se/Sp/prevalence-only metrics. No fits.\n")
