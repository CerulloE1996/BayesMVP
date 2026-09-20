#### =====================================================================================================================================
## test_19_ps7_autodiff_fallback_option.R - filenames, parsing, resume and option lifetime; no model fits
## =====================================================================================================================================
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
workspace_directory <-  normalizePath(path = file.path(tests_directory, "..", "..", "..", ".."))
ps7_directory <-  file.path(workspace_directory, "Alg_paper_analysis", "1_appendix_pilot_studies",
                            "ps_7_basic_MCMC_settings_BayesMVP")
source(file = file.path(ps7_directory, "functions", "ps_7_MCMC_settings_BayesMVP_functions.R"))
source(file = file.path(ps7_directory, "functions", "fn_ps7_extract_tau_sweep.R"))
source(file = file.path(ps7_directory, "functions", "fn_ps7_summarise_runs.R"))
source(file = file.path(workspace_directory, "R_packages", "BayesMVP", "inst", "BayesMVP", "R", "R_fn_burnin_adaptation_schedule.R"))
invisible(parse(file = file.path(ps7_directory, "ps_7_MCMC_settings_BayesMVP.R")))
##
filename_settings <-  list( output_dir = tempdir(),
                            n_chains_burnin = 4, n_chains_sampling = 180,
                            n_threads_WCP_burnin = 16, n_threads_WCP_sampling = 1,
                            num_chunks_burnin = 100, num_chunks_sampling = 25,
                            diffusion_HMC = TRUE, partitioned_HMC = FALSE, diffusion_HMC_integrator = "kick_flow_kick",
                            tau_objective = "ChEES_per_tau", tau_weight_by_p_jump = TRUE, tau_initial = pi,
                            n_burnin = 125, n_iter = 50, learning_rate = 0.075, adapt_delta = 0.8,
                            clip_iter = 20, int = 30, int_width = 1, learning_rate_initial = 0.1,
                            ratio_M_main = 0.9, ratio_M_nuisance = 0.9,
                            metric_type_main = "Empirical", metric_type_nuisance = "uniform_diag", metric_shape_main = "dense",
                            M_decay_type = "inverse", M_decay_power = 0.5, M_decay_scale = 1.13,
                            multi_attempts = TRUE, reorder_cols_MVP = TRUE, metric_estimator = "chain_mean_scaled",
                            tau_ramp = "staged", eps_reinit_at_ChEES_handover = FALSE, theta_hat_us_rule = "running_mean_frozen",
                            pre_burnin_n_iter = 50, pre_burnin_L = 10, share_tau_ii_across_chains_in_burnin = TRUE,
                            burnin_TBB_pool_equals_n_chains = TRUE, J_grad_option = "num_diff", store_log_lik_trace = FALSE,
                            n_runs = 2)
fn_test_filename <-  function(settings, model_type = "LC_MVP") {
        return(R_fn_file_name_string( Model_type = model_type,
                                       N = 10000,
                                       settings = settings,
                                       model_args_list = list(n_pops = 2),
                                       prior_LKJ_nd = 4,
                                       prior_LKJ_d = 4,
                                       prior_prev_a = 2.5,
                                       prior_prev_b = 10))
}
##
## ---- Missing/FALSE keeps the historical filename; TRUE adds only _af1 ---------------------------------------------------------------
##
original_name <-  fn_test_filename(settings = filename_settings)
expected_basename <-  paste0("ps7_run_LC_MVP_N10000_np2_ta2_dH1_pH0_toct_tw1_cb4_s180_wb16_s1_kb100_s25_b125_it50",
                             "_LR.075_AD.8_tipi_clip20_int30_Li.1_d_rM.9_.9_mtE_nuUD_msd_w1_mdinv_LKJ4_4_pp2.5_10_ma1",
                             "_mp.5_msc1.13_Ms_trS_nER_cF_pb50_pL10_sT_tbbC")
stopifnot(identical(x = basename(path = original_name), y = expected_basename))
filename_settings$autodiff_fallback <-  FALSE
stopifnot(identical(x = fn_test_filename(settings = filename_settings), y = original_name))
filename_settings$autodiff_fallback <-  TRUE
stopifnot(identical(x = fn_test_filename(settings = filename_settings), y = paste0(original_name, "_af1")))
##
## ---- Parsing the new suffix must not break any existing tags or merge configurations -------------------------------------------------
##
for (jacobian_mode in c("num_diff", "autodiff")) {
    for (store_log_lik in c(FALSE, TRUE)) {
        filename_settings$J_grad_option <-  jacobian_mode
        filename_settings$store_log_lik_trace <-  store_log_lik
        filename_settings$autodiff_fallback <-  FALSE
        disabled_name <-  paste0(basename(path = fn_test_filename(settings = filename_settings)), "_run12")
        filename_settings$autodiff_fallback <-  TRUE
        enabled_name <-  paste0(basename(path = fn_test_filename(settings = filename_settings)), "_run12")
        disabled_settings <-  fn_ps7_parse_run_name(x = disabled_name)
        enabled_settings <-  fn_ps7_parse_run_name(x = enabled_name)
        unchanged_columns <-  setdiff(x = names(x = disabled_settings), y = c("autodiff_fallback", "configuration", "file"))
        stopifnot(identical(x = disabled_settings[unchanged_columns], y = enabled_settings[unchanged_columns]),
                  !disabled_settings$autodiff_fallback, enabled_settings$autodiff_fallback,
                  enabled_settings$configuration != disabled_settings$configuration,
                  identical(x = enabled_settings$file, y = enabled_name), nchar(x = enabled_name, type = "bytes") <= 255,
                  "autodiff_fallback" %in% R_fn_ps7_setting_column_names(runs_table = enabled_settings))
    }
}
##
## ---- Automatic geometry schedule: both integrators and Jacobian modes must still fit the filename limit ------------------------------
##
for (integrator in c("kick_flow_kick", "flow_kick_flow")) {
    for (jacobian_mode in c("num_diff", "autodiff")) {
        scheduled_settings <-  filename_settings
        scheduled_settings$burnin_schedule <-  "automatic"
        scheduled_settings$tau_initial <-  pi / 2
        scheduled_settings$diffusion_HMC_integrator <-  integrator
        scheduled_settings$J_grad_option <-  jacobian_mode
        scheduled_settings$autodiff_fallback <-  TRUE
        scheduled_name <-  paste0(basename(path = fn_test_filename(settings = scheduled_settings)), "_run12")
        scheduled_parser <-  fn_ps7_parse_run_name(x = scheduled_name)
        stopifnot(nchar(x = scheduled_name, type = "bytes") <= 255,
                  scheduled_parser$autodiff_fallback,
                  scheduled_parser$burnin_schedule_version == 1,
                  scheduled_parser$metric_adaptation_end_iter == 90,
                  scheduled_parser$metric_estimator == "chain_mean_scaled",
                  scheduled_parser$J_grad_option == jacobian_mode)
    }
}
##
## ---- Inert and invalid settings ----------------------------------------------------------------------------------------------------------------
##
filename_settings$multi_attempts <-  FALSE
stopifnot(!grepl(pattern = "_af1$", x = fn_test_filename(settings = filename_settings)))
filename_settings$multi_attempts <-  TRUE
stopifnot(!grepl(pattern = "_af1$", x = fn_test_filename(settings = filename_settings, model_type = "Stan")))
for (invalid_option in list(NA, "TRUE", 1, logical(), c(TRUE, FALSE))) {
    invalid_settings <-  filename_settings
    invalid_settings$autodiff_fallback <-  invalid_option
    invalid_result <-  tryCatch(expr = fn_test_filename(settings = invalid_settings), error = function(condition) condition)
    stopifnot(inherits(x = invalid_result, what = "error"))
}
##
## ---- Actual resume path, with fake files and no fitting allowed; saved settings override and restore the session option ---------------
##
resume_environment <-  new.env(parent = environment())
resume_environment$require <-  function(package) TRUE
resume_environment$observed_resume_files <-  character()
resume_environment$file.exists <-  function(file) TRUE
resume_environment$readRDS <-  function(file) {
        resume_environment$observed_resume_files <-  c(resume_environment$observed_resume_files, file)
        return(list(autodiff_fallback = getOption(x = "BayesMVP_autodiff_fallback")))
}
resume_environment$R_fn_sample_model <-  function() stop("No model fitting is allowed in this test.")
resume_runner <-  run_ps7_models
environment(fun = resume_runner) <-  resume_environment
previous_option <-  options(BayesMVP_autodiff_fallback = FALSE)
for (enabled_option in c(TRUE, FALSE)) {
    filename_settings$autodiff_fallback <-  enabled_option
    options(BayesMVP_autodiff_fallback = !enabled_option)
    resume_environment$observed_resume_files <-  character()
    resumed_results <-  resume_runner( Model_type = "LC_MVP", N = 10000, y = matrix(data = 0, nrow = 2, ncol = 6),
                                       settings = filename_settings, vect_type = "AVX2", use_disk = FALSE,
                                       true_prev = 0.1, prior_LKJ_nd = 4, prior_LKJ_d = 4, prior_prev_a = 2.5, prior_prev_b = 10,
                                       pop = c(1, 2), LC_MVOP_grouping = NULL)
    stopifnot(length(x = resumed_results) == 2,
              all(vapply(X = resumed_results, FUN = function(result) identical(x = result$autodiff_fallback, y = enabled_option), FUN.VALUE = TRUE)),
              all(grepl(pattern = "_af1_run", x = resume_environment$observed_resume_files) == enabled_option),
              identical(x = getOption(x = "BayesMVP_autodiff_fallback"), y = !enabled_option))
}
options(previous_option)
cat("PASS: unchanged disabled names, distinct enabled names, full parser round-trip, validation, resume isolation and option restoration. No fits.\n")
