#### =====================================================================================================================================
## test_10_burnin_schedule.R - isolated scheduling checks; no model, MCMC, compilation or package installation
## =====================================================================================================================================
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  if (length(x = test_script_arguments)) {
    dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
} else {
    "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/APMS_bridge/tests"
}
bridge_directory <-  dirname(path = tests_directory)
repository_directory <-  dirname(path = bridge_directory)
workspace_directory <-  dirname(path = dirname(path = repository_directory))
package_R_directory <-  file.path(repository_directory, "inst", "BayesMVP", "R")
ps7_directory <-  file.path(workspace_directory, "Alg_paper_analysis", "1_appendix_pilot_studies",
                            "ps_7_basic_MCMC_settings_BayesMVP")
source(file = file.path(bridge_directory, "R_fn_APMS_BayesMVP.R"))
source(file = file.path(package_R_directory, "R_fn_burnin_adaptation_schedule.R"))
##
## ---- Extract the actual scheduling expressions without executing any sampler or benchmark --------------------------------------------
##
fn_find_schedule_expressions <-  function(expression, predicate) {
    matches <-  list()
    if (is.call(x = expression) && predicate(expression)) matches <-  list(expression)
    if (is.call(x = expression) || is.expression(x = expression)) {
        for (child in as.list(x = expression)) {
            if (missing(child)) next
            if (is.call(x = child) || is.expression(x = child)) {
                matches <-  c(matches, fn_find_schedule_expressions(expression = child, predicate = predicate))
            }
        }
    }
    matches
}
fn_is_assignment_to <-  function(expression, variable_name) {
    identical(expression[[1]], as.name(x = "<-")) && identical(expression[[2]], as.name(x = variable_name))
}
sample_expressions <-  parse(file = file.path(package_R_directory, "R_fn_sample.R"))
default_schedule <-  fn_find_schedule_expressions(expression = sample_expressions, predicate = function(expression) {
    identical(expression[[1]], as.name(x = "if")) &&
        identical(expression[[2]], quote(is.null(clip_iter) || is.null(clip_iter_tau)))
})
stopifnot(length(x = default_schedule) == 1)
fn_resolve_test_schedule <-  function(n_burnin, clip_iter = NULL, clip_iter_tau = NULL) {
    if_null_then_set_to <-  function(x, default) if (is.null(x = x)) default else x
    eval(expr = default_schedule[[1]], envir = environment())
    c(clip_iter = clip_iter, clip_iter_tau = clip_iter_tau)
}
##
## ---- New 125 schedule, unchanged other lengths, and explicit overrides ---------------------------------------------------------------
##
expected_schedules <-  list(`100` = c(24, 64), `125` = c(20, 50), `126` = c(30, 80),
                            `250` = c(50, 125), `500` = c(75, 175), `1000` = c(150, 350))
for (burnin_length in names(x = expected_schedules)) {
    resolved_schedule <-  fn_resolve_test_schedule(n_burnin = as.numeric(x = burnin_length))
    stopifnot(identical(unname(obj = resolved_schedule), expected_schedules[[burnin_length]]))
}
stopifnot(identical(unname(obj = fn_resolve_test_schedule(n_burnin = 125, clip_iter = 7, clip_iter_tau = 40)), c(7, 40)),
          identical(unname(obj = fn_resolve_test_schedule(n_burnin = 125, clip_iter = 7)), c(7, 50)))
for (burnin_length in c(125, 250)) {
    bridge_settings <-  fn_sampler_settings_APMS_BayesMVP(n_burnin = burnin_length)
    bridge_schedule <-  fn_resolve_test_schedule(n_burnin = burnin_length,
                                                  clip_iter = bridge_settings$clip_iter,
                                                  clip_iter_tau = bridge_settings$clip_iter_tau)
    stopifnot(identical(bridge_schedule, fn_resolve_test_schedule(n_burnin = burnin_length)))
}
stopifnot(fn_sampler_settings_APMS_BayesMVP(n_burnin = 125, clip_iter_tau = 43)$clip_iter_tau == 43)
##
## ---- PS7 runner forwards the same schedule for both metric branches -------------------------------------------------------------------
##
ps7_expressions <-  parse(file = file.path(ps7_directory, "ps_7_MCMC_settings_BayesMVP.R"))
ps7_short_schedule <-  fn_find_schedule_expressions(expression = ps7_expressions, predicate = function(expression) {
    identical(expression[[1]], as.name(x = "if")) && identical(expression[[2]], quote(i %in% c(14, 15, 16, 17)))
})
stopifnot(length(x = ps7_short_schedule) == 1)
for (metric_type in c("Empirical", "Hessian")) {
    i <-  14
    ps7_settings <-  list(metric_type_main = metric_type)
    eval(expr = ps7_short_schedule[[1]])
    stopifnot(ps7_settings$clip_iter == 20, ps7_settings$int == 30,
              ps7_settings$clip_iter + ps7_settings$int == 50)
}
##
## ---- Actual burn-in handover occurs once, without subsequent resets -------------------------------------------------------------------
##
burnin_expressions <-  parse(file = file.path(package_R_directory, "R_fn_init_and_run_burnin_CHESS.R"))
handover_assignment <-  fn_find_schedule_expressions(expression = burnin_expressions, predicate = function(expression) {
    fn_is_assignment_to(expression = expression, variable_name = "gap") && identical(expression[[3]], as.name(x = "clip_iter_tau"))
})
handover_block <-  fn_find_schedule_expressions(expression = burnin_expressions, predicate = function(expression) {
    identical(expression[[1]], as.name(x = "if")) && identical(expression[[2]], quote(ii == gap))
})
stopifnot(length(x = handover_assignment) == 1, length(x = handover_block) == 1)
clip_iter_tau <-  50
n_adapt <-  113
eval(expr = handover_assignment[[1]])
stopifnot(gap == 50)
reset_assignment <-  fn_find_schedule_expressions(expression = burnin_expressions, predicate = function(expression) {
    fn_is_assignment_to(expression = expression, variable_name = "vec_points")
})
stopifnot(length(x = reset_assignment) == 0)
##
## ---- Main burn-in always runs to n_burnin, and retired arguments cannot enter any sampler interface -----------------------------------
##
last_iteration_assignment <-  fn_find_schedule_expressions(expression = burnin_expressions, predicate = function(expression) {
    fn_is_assignment_to(expression = expression, variable_name = "last_burnin_iteration")
})
iteration_sequence_assignment <-  fn_find_schedule_expressions(expression = burnin_expressions, predicate = function(expression) {
    fn_is_assignment_to(expression = expression, variable_name = "iter_seq_burnin")
})
stopifnot(length(x = last_iteration_assignment) == 1, length(x = iteration_sequence_assignment) == 1)
for (n_burnin in c(125, 250)) {
    n_adapt <-  n_burnin - round(n_burnin / 10)
    eval(expr = last_iteration_assignment[[1]])
    eval(expr = iteration_sequence_assignment[[1]])
    stopifnot(last_burnin_iteration == n_burnin,
              length(x = iter_seq_burnin) == n_burnin,
              tail(x = iter_seq_burnin, n = 1) == n_burnin)
}
files_to_check <-  c(file.path(package_R_directory, "R_fn_sample.R"),
                    file.path(package_R_directory, "R_fn_init_and_run_burnin_CHESS.R"),
                    file.path(package_R_directory, "R_R6_class_aaa_init_and_sample.R"),
                    file.path(bridge_directory, "R_fn_APMS_BayesMVP.R"),
                    file.path(ps7_directory, "functions", "ps_7_MCMC_settings_BayesMVP_functions.R"))
for (source_file in files_to_check) {
    retired_formals <-  fn_find_schedule_expressions(expression = parse(file = source_file), predicate = function(expression) {
        identical(expression[[1]], as.name(x = "function")) && "burnin_post_adapt_iter" %in% names(x = expression[[2]])
    })
    retired_named_calls <-  fn_find_schedule_expressions(expression = parse(file = source_file), predicate = function(expression) {
        "burnin_post_adapt_iter" %in% names(x = expression)
    })
    stopifnot(length(x = retired_formals) == 0, length(x = retired_named_calls) == 0)
}
cat("\nPASS: 125-iteration handover is 50; legacy ramp boundaries and explicit overrides are unchanged. Automatic phases are tested in test_16.\n")
cat("PASS: tau is initialised once at handover; the repeated-reset schedule has been removed.\n")
cat("PASS: full 125/250 burn-in lengths; retired option absent from all sampler and PS7 summary interfaces.\n")
##
## ---- New filenames have no early-stop token; old filenames still parse without becoming selectable settings ---------------------------
##
source(file = file.path(ps7_directory, "functions", "ps_7_MCMC_settings_BayesMVP_functions.R"))
source(file = file.path(ps7_directory, "functions", "fn_ps7_extract_tau_sweep.R"))
filename_settings <-  fn_sampler_settings_APMS_BayesMVP(n_burnin = 125,
                                                        metric_estimator = "pooled",
                                                        pre_burnin_n_iter = 50)
filename_settings$output_dir <-  tempdir()
filename_settings$int <-  30
filename_settings$int_width <-  1
filename_settings$multi_attempts <-  TRUE
filename_settings$manual_L <-  NA_real_
filename_settings$manual_tau_value <-  NA_real_
filename_arguments <-  list(Model_type = "LC_MVP", N = 10000, settings = filename_settings,
                            model_args_list = list(n_pops = 2), prior_LKJ_nd = 4, prior_LKJ_d = 4,
                            prior_prev_a = 2.5, prior_prev_b = 10)
run_filename <-  paste0(basename(path = do.call(what = R_fn_file_name_string, args = filename_arguments)), "_run1")
stopifnot(grepl(pattern = "_clip20_int30_", x = run_filename),
          grepl(pattern = "_ta2_", x = run_filename),
          !grepl(pattern = "_pa[0-9]+(_|$)", x = run_filename))
legacy_run_filename <-  sub(pattern = "_pb", replacement = "_pa5_pb", x = run_filename)
parsed_legacy_run <-  fn_ps7_parse_run_name(x = legacy_run_filename)
stopifnot(parsed_legacy_run$metric_estimator == "pooled",
          parsed_legacy_run$tau_adaptation_version == 2,
          !("burnin_post_adapt_iter" %in% names(x = parsed_legacy_run)))
filename_arguments$settings$burnin_post_adapt_iter <-  5
retired_filename_error <-  tryCatch(expr = do.call(what = R_fn_file_name_string, args = filename_arguments),
                                    error = function(error_condition) error_condition)
stopifnot(inherits(x = retired_filename_error, what = "error"),
          grepl(pattern = "has been removed", x = conditionMessage(c = retired_filename_error)))
cat("PASS: PS7 filenames encode clip20/int30, never generate _pa, and reject stale early-stop settings.\n")
stopifnot(fn_ps7_parse_run_name(x = sub(pattern = "_ta2", replacement = "", x = run_filename))$tau_adaptation_version == 1)
