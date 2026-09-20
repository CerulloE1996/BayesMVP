#### =====================================================================================================================================
## test_16_geometry_freeze_schedule.R - actual R scheduling conditions, no MCMC, compilation or installation
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
package_directory <-  normalizePath(path = file.path(tests_directory, "..", "..", "inst", "BayesMVP"))
workspace_directory <-  normalizePath(path = file.path(tests_directory, "..", "..", "..", ".."))
ps7_directory <-  file.path(workspace_directory, "Alg_paper_analysis", "1_appendix_pilot_studies", "ps_7_basic_MCMC_settings_BayesMVP")
source(file = file.path(package_directory, "R", "R_fn_burnin_adaptation_schedule.R"))
source(file = file.path(package_directory, "R", "R_fn_sample.R"))
source(file = file.path(ps7_directory, "functions", "ps_7_MCMC_settings_BayesMVP_functions.R"))
source(file = file.path(ps7_directory, "functions", "fn_ps7_extract_tau_sweep.R"))
source(file = file.path(dirname(path = tests_directory), "R_fn_APMS_BayesMVP.R"))
##
fn_expect_schedule_error <-  function(expression) {
    result <-  tryCatch(expr = force(expr = expression), error = function(error_condition) error_condition)
    stopifnot(inherits(x = result, what = "error"))
}
fn_find_schedule_nodes <-  function(expression, predicate) {
    matches <-  list()
    if (is.call(x = expression) && predicate(expression)) matches <-  list(expression)
    if (is.call(x = expression) || is.expression(x = expression)) {
        for (child in as.list(x = expression)) {
            if (missing(child)) next
            if (is.call(x = child) || is.expression(x = child)) {
                matches <-  c(matches, fn_find_schedule_nodes(expression = child, predicate = predicate))
            }
        }
    }
    matches
}
##
## ---- Exact requested endpoints, interpolation/extrapolation and invalid inputs ------------------------------------------------------
##
short_schedule <-  fn_burnin_adaptation_schedule(n_burnin = 125)
long_schedule <-  fn_burnin_adaptation_schedule(n_burnin = 250)
stopifnot(identical(x = short_schedule$phase_table$iterations, y = c("90", "91-112", "113-125")),
          identical(x = long_schedule$phase_table$iterations, y = c("180", "181-224", "225-250")),
          short_schedule$clip_iter == 20, short_schedule$clip_iter_tau == 50,
          long_schedule$clip_iter == 50, long_schedule$clip_iter_tau == 125)
for (burnin_length in c(5:2000, 10000, 1000000)) {
    schedule <-  fn_burnin_adaptation_schedule(n_burnin = burnin_length)
    stopifnot(schedule$metric_adaptation_end_iter == schedule$theta_hat_us_freeze_iter,
              schedule$metric_adaptation_end_iter < schedule$n_adapt - 1,
              schedule$clip_iter < schedule$clip_iter_tau,
              schedule$clip_iter_tau < schedule$n_adapt,
              schedule$n_adapt <= burnin_length,
              schedule$clip_iter < schedule$theta_hat_us_freeze_iter)
}
stopifnot(fn_burnin_adaptation_schedule(n_burnin = 126)$clip_iter_tau == 51,
          fn_burnin_adaptation_schedule(n_burnin = 2000)$clip_iter_tau == 700,
          fn_burnin_adaptation_schedule(n_burnin = 125, n_adapt = 100)$metric_adaptation_end_iter == 80,
          fn_burnin_adaptation_schedule(n_burnin = 125, metric_adaptation_end_iter = 85)$theta_hat_us_freeze_iter == 85)
for (invalid_length in list(NULL, NA, Inf, -1, 0, 1, 4, 125.5, c(125, 250))) {
    fn_expect_schedule_error(expression = fn_burnin_adaptation_schedule(n_burnin = invalid_length))
}
fn_expect_schedule_error(expression = fn_burnin_adaptation_schedule(n_burnin = 125, metric_adaptation_end_iter = 112))
fn_expect_schedule_error(expression = fn_burnin_adaptation_schedule(n_burnin = 125, theta_hat_us_rule = "running_mean"))
fn_expect_schedule_error(expression = fn_burnin_adaptation_schedule(n_burnin = 125, theta_hat_us_freeze_iter = 20, clip_iter = 20))
fn_expect_schedule_error(expression = fn_burnin_adaptation_schedule(n_burnin = 125, clip_iter_tau = 113))
legacy_schedule <-  fn_burnin_adaptation_schedule(n_burnin = 125, burnin_schedule = "legacy")
stopifnot(legacy_schedule$metric_adaptation_end_iter == 112, legacy_schedule$theta_hat_us_freeze_iter == 68)
cat("PASS: requested endpoints; 5-2000 iterations plus extrapolation; explicit overrides; invalid inputs; legacy option.\n")
##
## ---- Evaluate the sampler's REAL metric gates and centre assignments; do not execute transitions ------------------------------------
##
burnin_expressions <-  parse(file = file.path(package_directory, "R", "R_fn_init_and_run_burnin_CHESS.R"))
metric_gates <-  fn_find_schedule_nodes(expression = burnin_expressions, predicate = function(expression) {
    identical(expression[[1]], as.name(x = "if")) &&
        grepl(pattern = "ii <= metric_adaptation_end_iter", x = paste(deparse(expr = expression[[2]]), collapse = " "), fixed = TRUE)
})
centre_block <-  fn_find_schedule_nodes(expression = burnin_expressions, predicate = function(expression) {
    identical(expression[[1]], as.name(x = "if")) &&
        identical(expression[[2]], quote(identical(theta_hat_us_rule, "zero") || (ii <= clip_iter)))
})
epsilon_gate <-  fn_find_schedule_nodes(expression = burnin_expressions, predicate = function(expression) {
    identical(expression[[1]], as.name(x = "if")) && identical(expression[[2]], quote(ii < n_adapt))
})
tau_gates <-  fn_find_schedule_nodes(expression = burnin_expressions, predicate = function(expression) {
    identical(expression[[1]], as.name(x = "if")) && identical(expression[[2]], quote(!isTRUE(manual_tau) && ii >= gap && ii < n_adapt))
})
stopifnot(length(x = metric_gates) == 2, length(x = centre_block) == 1,
          length(x = epsilon_gate) >= 1, length(x = tau_gates) >= 1)
for (schedule in list(short_schedule, long_schedule)) {
    for (interval_width in c(1, 7, 20)) {
        execution_environment <-  list2env(x = c(schedule, list(ii_min = round(x = schedule$n_burnin / 10),
            ii_max = schedule$n_adapt, metric_ready = TRUE, interval_width_main = interval_width,
            interval_width_nuisance = interval_width, gap = schedule$clip_iter_tau, manual_tau = FALSE)))
        for (gate in metric_gates) {
            permitted_updates <-  vapply(X = seq_len(length.out = schedule$n_burnin), FUN.VALUE = TRUE, FUN = function(iteration) {
                execution_environment$ii <-  iteration
                eval(expr = gate[[2]], envir = execution_environment)
            })
            stopifnot(max(which(x = permitted_updates)) == schedule$metric_adaptation_end_iter,
                      !any(permitted_updates[(schedule$metric_adaptation_end_iter + 1):schedule$n_burnin]))
        }
        for (iteration in (schedule$metric_adaptation_end_iter + 1):(schedule$n_adapt - 1)) {
            execution_environment$ii <-  iteration
            stopifnot(eval(expr = epsilon_gate[[1]][[2]], envir = execution_environment))
        }
        execution_environment$ii <-  schedule$n_adapt - 1
        stopifnot(eval(expr = tau_gates[[1]][[2]], envir = execution_environment))
        for (iteration in schedule$n_adapt:schedule$n_burnin) {
            execution_environment$ii <-  iteration
            stopifnot(!eval(expr = epsilon_gate[[1]][[2]], envir = execution_environment),
                      !eval(expr = tau_gates[[1]][[2]], envir = execution_environment))
        }
    }
    execution_environment$n_nuisance <-  2
    execution_environment$EHMC_Metric_as_Rcpp_List <-  list(theta_hat_us_vec = matrix(data = c(0, 0)))
    for (iteration in seq_len(length.out = schedule$n_burnin)) {
        execution_environment$ii <-  iteration
        execution_environment$EHMC_burnin_as_Rcpp_List <-  list(snaper_m_vec_us = rep(x = iteration, times = 2))
        eval(expr = centre_block[[1]], envir = execution_environment)
        if (iteration >= schedule$theta_hat_us_freeze_iter) {
            stopifnot(all(execution_environment$EHMC_Metric_as_Rcpp_List$theta_hat_us_vec == schedule$theta_hat_us_freeze_iter))
        }
    }
}
cat("PASS: both actual metric gates stop on the cutoff, including width 7; centre stays fixed; epsilon/tau continue and then stop.\n")
##
## ---- PS7 filename/resume identity, round trip, selected controls and length guard ----------------------------------------------------
##
settings <-  fn_sampler_settings_APMS_BayesMVP(n_burnin = 125, n_chains_sampling = 180,
                                               metric_estimator = "chain_mean_scaled", tau_initial = pi / 2)
settings$output_dir <-  tempdir()
settings$int <-  30
settings$int_width <-  1
settings$manual_L <-  NA_real_
settings$manual_tau_value <-  NA_real_
settings$multi_attempts <-  TRUE
settings$reorder_cols_MVP <-  TRUE
fn_schedule_filename <-  function(settings) {
    paste0(basename(path = R_fn_file_name_string(Model_type = "LC_MVP", N = 10000, settings = settings,
        model_args_list = list(n_pops = 2), prior_LKJ_nd = 4, prior_LKJ_d = 4, prior_prev_a = 2.5, prior_prev_b = 10)), "_run1")
}
automatic_name <-  fn_schedule_filename(settings = settings)
resolved_settings <-  fn_ps7_resolve_burnin_schedule(settings = settings)
stopifnot(identical(x = automatic_name, y = fn_schedule_filename(settings = resolved_settings)),
          nchar(x = automatic_name, type = "bytes") <= 255,
          grepl(pattern = "_bs1a113m90c90_", x = automatic_name, fixed = TRUE))
## Long-name cases: the new token must coexist with pi/2, FKF, autodiff and storage options.
for (integrator in c("kick_flow_kick", "flow_kick_flow")) {
    for (jacobian_mode in c("num_diff", "autodiff")) {
        filename_settings <- settings
        filename_settings$diffusion_HMC_integrator <- integrator
        filename_settings$J_grad_option <- jacobian_mode
        filename_settings$store_log_lik_trace <- TRUE
        filename_settings$num_chunks_burnin <- 100
        filename_settings$num_chunks_sampling <- 25
        stopifnot(nchar(x = fn_schedule_filename(settings = filename_settings), type = "bytes") <= 255)
    }
}
parsed_schedule <-  fn_ps7_parse_run_name(x = automatic_name)
stopifnot(parsed_schedule$n_adapt == 113, parsed_schedule$metric_adaptation_end_iter == 90,
          parsed_schedule$centre_adaptation_end_iter == 90, parsed_schedule$theta_hat_us_freeze_iter == 90,
          parsed_schedule$metric_estimator == "chain_mean_scaled", parsed_schedule$tau_ramp == "staged",
          parsed_schedule$burnin_schedule_version == 1)
settings$metric_adaptation_end_iter <-  85
stopifnot(fn_schedule_filename(settings = settings) != automatic_name)
settings$theta_hat_us_freeze_iter <-  80
stopifnot(grepl(pattern = "m85c80_", x = fn_schedule_filename(settings = settings), fixed = TRUE))
settings$burnin_schedule <-  NULL
settings$metric_adaptation_end_iter <-  NULL
legacy_name <-  fn_schedule_filename(settings = settings)
stopifnot(!grepl(pattern = "_bs", x = legacy_name, fixed = TRUE),
          fn_ps7_parse_run_name(x = legacy_name)$burnin_schedule_version == 0)
cat("PASS: filenames encode effective boundaries; pre/post resolution agree; legacy files still parse; current filename fits NAME_MAX.\n")
##
## ---- Public/R6/APMS forwarding and source parse checks -------------------------------------------------------------------------------
##
stopifnot(formals(fun = R_fn_sample_model)$burnin_schedule == "automatic",
          fn_sampler_settings_APMS_BayesMVP()$burnin_schedule == "automatic",
          fn_sampler_settings_APMS_BayesMVP(metric_adaptation_end_iter = 85)$metric_adaptation_end_iter == 85)
r6_expressions <-  parse(file = file.path(package_directory, "R", "R_R6_class_aaa_init_and_sample.R"))
r6_forward_calls <-  fn_find_schedule_nodes(expression = r6_expressions, predicate = function(expression) {
    identical(expression[[1]], as.name(x = "R_fn_sample_model"))
})
stopifnot(length(x = r6_forward_calls) == 1,
          identical(r6_forward_calls[[1]]$burnin_schedule, quote(burnin_schedule)),
          identical(r6_forward_calls[[1]]$metric_adaptation_end_iter, quote(metric_adaptation_end_iter)))
files_to_parse <-  c(file.path(package_directory, "R", c("R_fn_burnin_adaptation_schedule.R", "R_fn_sample.R",
    "R_fn_init_and_run_burnin_CHESS.R", "R_R6_class_aaa_init_and_sample.R", "R_fn_create_summary_and_traces.R")),
    file.path(dirname(path = tests_directory), "R_fn_APMS_BayesMVP.R"),
    file.path(ps7_directory, "ps_7_MCMC_settings_BayesMVP.R"),
    file.path(ps7_directory, "functions", c("ps_7_MCMC_settings_BayesMVP_functions.R", "fn_ps7_extract_tau_sweep.R", "fn_ps7_summarise_runs.R")))
for (source_file in files_to_parse) invisible(x = parse(file = source_file))
for (help_file in c("fn_burnin_adaptation_schedule.Rd", "R_fn_sample_model.Rd", "MVP_model.Rd", "init_and_run_burnin_ChESSR.Rd")) {
    invisible(x = tools::parse_Rd(file = file.path(package_directory, "man", help_file)))
}
cat("PASS: package/R6/APMS controls, all modified R files and help file parse. No fits or C++ builds were run.\n")
