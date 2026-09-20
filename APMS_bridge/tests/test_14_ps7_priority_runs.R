#### ======================================================================================================= test_14_ps7_priority_runs.R
##
## ---- Check actual PS7 scheduling expressions without sourcing the runner or running MCMC ------------------------------------------
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
workspace_directory <-  dirname(path = dirname(path = dirname(path = dirname(path = tests_directory))))
ps7_directory <-  file.path(workspace_directory, "Alg_paper_analysis", "1_appendix_pilot_studies",
                            "ps_7_basic_MCMC_settings_BayesMVP")
runner_expressions <-  parse(file = file.path(ps7_directory, "ps_7_MCMC_settings_BayesMVP.R"))
source(file = file.path(ps7_directory, "functions", "ps_7_MCMC_settings_BayesMVP_functions.R"))
source(file = file.path(ps7_directory, "functions", "fn_ps7_extract_tau_sweep.R"))
##
fn_find_priority_expressions <-  function(expression, predicate) {
    matches <-  list()
    if (is.call(x = expression) && predicate(expression)) matches <-  list(expression)
    if (is.call(x = expression) || is.expression(x = expression)) {
        for (child in as.list(x = expression)) {
            if (missing(child)) next
            if (is.call(x = child) || is.expression(x = child)) {
                matches <-  c(matches, fn_find_priority_expressions(expression = child, predicate = predicate))
            }
        }
    }
    matches
}
fn_is_priority_assignment <-  function(expression, variable_name) {
    identical(x = expression[[1]], y = as.name(x = "<-")) && identical(x = expression[[2]], y = as.name(x = variable_name))
}
priority_block <-  Filter(f = function(expression) {
    is.call(x = expression) && identical(x = expression[[1]], y = as.name(x = "{")) &&
        length(x = fn_find_priority_expressions(expression = expression, predicate = function(expression) {
            fn_is_priority_assignment(expression = expression, variable_name = "ps7_priority_configurations")
        })) == 1
}, x = as.list(x = runner_expressions))
grid_assignment <-  fn_find_priority_expressions(expression = runner_expressions, predicate = function(expression) {
    fn_is_priority_assignment(expression = expression, variable_name = "sampler_combinations") &&
        is.call(x = expression[[3]]) && identical(x = expression[[3]][[1]], y = as.name(x = "expand.grid"))
})
priority_grid_block <-  fn_find_priority_expressions(expression = runner_expressions, predicate = function(expression) {
    identical(x = expression[[1]], y = as.name(x = "if")) &&
        identical(x = expression[[2]], y = quote(!identical(x = ps7_run_stage, y = "full_grid"))) &&
        grepl(pattern = "ps7_priority_shared_combinations <-", x = paste(deparse(expr = expression), collapse = " "), fixed = TRUE)
})
stopifnot(length(x = priority_block) == 1, length(x = grid_assignment) == 1, length(x = priority_grid_block) == 1)
##
fn_test_priority_stage <-  function(run_stage, configuration_ids = NULL, n_runs = 2) {
    test_environment <-  list2env(x = list(
        ps7_run_stage = run_stage,
        ps7_confirmation_configuration_ids = configuration_ids,
        settings_combo_vec = c(15, 16, 17, 10, 11, 12, 13),
        n_runs = n_runs, n_combos_2 = 128, N_vec_to_use = 10000, runs_override = 2,
        metric_estimator = c("chain_mean_scaled", "pooled"), tau_initial = c(2 * pi, pi),
        tau_ramp = c("staged", "original"), learning_rate_initial = c(0.10, 0.15),
        diffusion_HMC_integrator = "kick_flow_kick", tau_objective = c("ChEES_per_tau", "KE"),
        tau_weight_by_p_jump = TRUE, tau_weight_by_p_jump_for = c("ChEES", "ChEES_per_tau"),
        manual_L = NA_real_, manual_tau_value = NA_real_, eps_reinit_at_ChEES_handover = c(TRUE, FALSE),
        theta_hat_us_rule = c("running_mean_frozen", "zero"), theta_hat_us_freeze_iter = NULL,
        J_grad_option = c("num_diff", "autodiff"), store_log_lik_trace = FALSE), parent = globalenv())
    invisible(x = capture.output(eval(expr = priority_block[[1]], envir = test_environment)))
    test_environment$ps7_settings <-  as.list.environment(x = test_environment)
    eval(expr = grid_assignment[[1]], envir = test_environment)
    eval(expr = priority_grid_block[[1]], envir = test_environment)
    test_environment$sampler_combinations <-  fn_ps7_collapse_inert_tau_axes(
        sampler_combinations = test_environment$sampler_combinations,
        weight_p_jump_only_for = test_environment$tau_weight_by_p_jump_for)
    test_environment
}
##
screening <-  fn_test_priority_stage(run_stage = "screening")
stopifnot(screening$n_runs == 2, screening$n_fits == 20, is.null(x = screening$runs_override),
          identical(x = screening$settings_combo_vec, y = 16),
          identical(x = screening$sampler_combinations$tau_initial, y = rep(x = c(pi, pi, pi / 2, 2 * pi, pi), each = 2)),
          identical(x = screening$sampler_combinations$tau_ramp,
                    y = rep(x = c("staged", "original", "staged", "staged", "staged"), each = 2)),
          identical(x = screening$sampler_combinations$tau_objective, y = rep(x = c(rep(x = "ChEES_per_tau", times = 4), "KE"), each = 2)),
          all(screening$sampler_combinations$tau_weight_by_p_jump),
          all(screening$sampler_combinations$J_grad_option == "num_diff"),
          all(screening$sampler_combinations$metric_estimator == "chain_mean_scaled"),
          all(screening$sampler_combinations$theta_hat_us_rule == "running_mean_frozen"),
          all(!screening$sampler_combinations$eps_reinit_at_ChEES_handover),
          identical(x = screening$sampler_combinations$learning_rate_initial, y = rep(x = c(0.10, 0.15), times = 5)),
          identical(x = screening$ps7_priority_configuration_ids_for_combinations, y = rep(x = c(1, 2, 3, 4, 5), each = 2)))
cat("PASS: five priority rows x both chosen initial learning rates; no duplicates from the separate tau vector.\n")
##
confirmation <-  fn_test_priority_stage(run_stage = "confirmation", configuration_ids = c(5, 2), n_runs = 3)
stopifnot(confirmation$n_runs == 3, confirmation$n_fits == 12, is.null(x = confirmation$runs_override),
          nrow(x = confirmation$sampler_combinations) == 4,
          identical(x = confirmation$ps7_priority_configuration_ids_to_run, y = c(5, 2)),
          identical(x = confirmation$sampler_combinations$tau_objective, y = rep(x = c("KE", "ChEES_per_tau"), each = 2)),
          all(confirmation$sampler_combinations$tau_weight_by_p_jump))
for (invalid_ids in list(NULL, c(1, 1), c(1, 6), c(1, 2, 3), c(1, NA), c("1", "2"))) {
    invalid_result <-  tryCatch(expr = fn_test_priority_stage(run_stage = "confirmation", configuration_ids = invalid_ids),
                               error = function(error_condition) error_condition)
    stopifnot(inherits(x = invalid_result, what = "error"))
}
full_grid <-  fn_test_priority_stage(run_stage = "full_grid")
stopifnot(identical(x = full_grid$settings_combo_vec, y = c(15, 16, 17, 10, 11, 12, 13)),
          full_grid$n_runs == 2, full_grid$runs_override == 2, full_grid$n_fits == 128 * 7 * 2,
          identical(x = full_grid$metric_estimator, y = c("chain_mean_scaled", "pooled")),
          nrow(x = full_grid$sampler_combinations) > 5)
cat("PASS: confirmation selects only two IDs; invalid selections stop; full-grid controls remain unchanged.\n")
##
## ---- Every stage honours arbitrary user-selected positive repeat counts ----------------------------------------------------------
##
for (selected_n_runs in c(1, 2, 7)) {
    for (run_stage in c("screening", "confirmation", "full_grid")) {
        selected_stage <-  fn_test_priority_stage(run_stage = run_stage, configuration_ids = c(1, 5), n_runs = selected_n_runs)
        expected_configuration_count <-  switch(EXPR = run_stage, screening = 10, confirmation = 4, full_grid = 128 * 7)
        stopifnot(selected_stage$n_runs == selected_n_runs,
                  selected_stage$n_fits == expected_configuration_count * selected_n_runs)
    }
}
for (invalid_n_runs in list(NULL, 0, -1, 1.5, NA_real_, Inf, c(1, 2), "3")) {
    invalid_result <-  tryCatch(expr = fn_test_priority_stage(run_stage = "screening", n_runs = invalid_n_runs),
                               error = function(error_condition) error_condition)
    stopifnot(inherits(x = invalid_result, what = "error"))
}
repeat_count_assignments <-  fn_find_priority_expressions(expression = runner_expressions, predicate = function(expression) {
    fn_is_priority_assignment(expression = expression, variable_name = "n_runs")
})
stopifnot(length(x = repeat_count_assignments) == 1)
cat("PASS: one n_runs control; all stages honour 1, 2 or 7 repeats, and invalid counts stop before fits.\n")
##
## ---- Actual filename builder: distinct files for selected settings, identical names across run stages ------------------------------
##
base_settings <-  list(output_dir = tempdir(), n_chains_burnin = 4, n_chains_sampling = 180,
                       n_threads_WCP_burnin = 16, n_threads_WCP_sampling = 1, num_chunks_burnin = 100,
                       num_chunks_sampling = 25, n_burnin = 125, n_iter = 50, learning_rate = 0.075,
                       adapt_delta = 0.8, clip_iter = 20, int = 30, ratio_M_nuisance = 0.9, ratio_M_main = 0.9,
                       int_width = 1, diffusion_HMC = TRUE, partitioned_HMC = FALSE, metric_type_main = "Empirical",
                       metric_type_nuisance = "uniform_diag", metric_shape_main = "dense", M_decay_type = "inverse",
                       M_decay_power = 0.5, M_decay_scale = 1.13, multi_attempts = TRUE, reorder_cols_MVP = TRUE,
                       pre_burnin_n_iter = 50, pre_burnin_L = 10, share_tau_ii_across_chains_in_burnin = TRUE,
                       burnin_TBB_pool_equals_n_chains = TRUE)
fn_test_priority_file_names <-  function(stage_environment) {
    vapply(X = seq_len(length.out = nrow(x = stage_environment$sampler_combinations)), FUN.VALUE = "",
           FUN = function(configuration_index) {
               settings <-  base_settings
               for (setting_name in names(x = stage_environment$sampler_combinations)) {
                   settings[[setting_name]] <-  stage_environment$sampler_combinations[[setting_name]][configuration_index]
               }
               settings["theta_hat_us_freeze_iter"] <-  list(NULL)
               settings$n_runs <-  stage_environment$n_runs
               R_fn_file_name_string(Model_type = "LC_MVP", N = 10000, settings = settings,
                                     model_args_list = list(n_pops = 2, num_chunks = 100),
                                     prior_LKJ_nd = 4, prior_LKJ_d = 4, prior_prev_a = 2.5, prior_prev_b = 10,
                                     grouping = FALSE)
           })
}
screening_file_names <-  fn_test_priority_file_names(stage_environment = screening)
confirmation_file_names <-  fn_test_priority_file_names(stage_environment = confirmation)
stopifnot(length(x = unique(x = screening_file_names)) == 10,
          all(grepl(pattern = "_ta2_", x = screening_file_names, fixed = TRUE)),
          identical(x = confirmation_file_names, y = screening_file_names[c(9, 10, 3, 4)]))
## Rebuild the same settings through the actual full-grid expression: neither stage nor row order changes names.
full_grid_matching_settings <-  fn_test_priority_stage(run_stage = "full_grid")
full_grid_matching_settings$ps7_settings <-  as.list.environment(x = screening)
full_grid_matching_settings$ps7_settings$tau_initial <-  c(pi / 2, pi, 2 * pi)
full_grid_matching_settings$ps7_settings$tau_ramp <-  c("original", "staged")
full_grid_matching_settings$ps7_settings$tau_objective <-  c("KE", "ChEES_per_tau")
eval(expr = grid_assignment[[1]], envir = full_grid_matching_settings)
eval(expr = priority_grid_block[[1]], envir = full_grid_matching_settings)
full_grid_matching_file_names <-  fn_test_priority_file_names(stage_environment = full_grid_matching_settings)
stopifnot(all(screening_file_names %in% full_grid_matching_file_names))
for (run_file_name in screening_file_names) {
    parsed_run <-  fn_ps7_parse_run_name(x = basename(path = paste0(run_file_name, "_run1")))
    stopifnot(parsed_run$tau_adaptation_version == 2, parsed_run$tau_weight_by_p_jump,
              parsed_run$metric_estimator == "chain_mean_scaled")
}
resume_assignment <-  fn_find_priority_expressions(expression = body(fun = run_ps7_models), predicate = function(expression) {
    fn_is_priority_assignment(expression = expression, variable_name = "runs_to_do") &&
        identical(x = expression[[3]], y = quote(setdiff(1:settings$n_runs, existing_runs)))
})
stopifnot(length(x = resume_assignment) == 1)
resume_environment <-  list2env(x = list(settings = list(n_runs = 3), existing_runs = 1))
eval(expr = resume_assignment[[1]], envir = resume_environment)
stopifnot(identical(x = resume_environment$runs_to_do, y = 2:3))
cat("PASS: identical settings have identical filenames across screening, confirmation and full_grid; saved seeds resume.\n")
##
## The new schedule must preserve stage-independent naming too, while separating all old fits.
source(file = file.path(workspace_directory, "R_packages", "BayesMVP", "inst", "BayesMVP", "R", "R_fn_burnin_adaptation_schedule.R"))
base_settings$burnin_schedule <- "automatic"
automatic_screening_names <- fn_test_priority_file_names(stage_environment = screening)
automatic_confirmation_names <- fn_test_priority_file_names(stage_environment = confirmation)
automatic_full_grid_names <- fn_test_priority_file_names(stage_environment = full_grid_matching_settings)
stopifnot(all(automatic_screening_names %in% automatic_full_grid_names),
          identical(automatic_confirmation_names, automatic_screening_names[c(9, 10, 3, 4)]),
          !any(automatic_screening_names %in% screening_file_names),
          all(grepl(pattern = "_bs1a113m90c90_", x = automatic_screening_names, fixed = TRUE)))
cat("PASS: automatic schedule is distinct from old fits and identical across all run stages.\n")
##
report_pattern_assignment <-  fn_find_priority_expressions(expression = runner_expressions, predicate = function(expression) {
    fn_is_priority_assignment(expression = expression, variable_name = "ps7_report_file_pattern")
})
stopifnot(length(x = report_pattern_assignment) == 1)
eval(expr = report_pattern_assignment[[1]], envir = screening)
stopifnot(all(grepl(pattern = screening$ps7_report_file_pattern, x = basename(path = screening_file_names))),
          !grepl(pattern = screening$ps7_report_file_pattern,
                 x = sub(pattern = "_ta2", replacement = "", x = basename(path = screening_file_names[1]), fixed = TRUE)),
          !grepl(pattern = screening$ps7_report_file_pattern,
                 x = sub(pattern = "_N10000_", replacement = "_N2500_", x = basename(path = screening_file_names[1]), fixed = TRUE)))
cat("PASS: priority report matches selected N and corrected adaptation only. No sampler, build or installation was run.\n")
