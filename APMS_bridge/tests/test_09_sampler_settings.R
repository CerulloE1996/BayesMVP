#### =====================================================================================================================================
## test_09_sampler_settings.R - pure R checks for the flat APMS settings interface; no MCMC, builds or package loading
## =====================================================================================================================================
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  if (length(x = test_script_arguments)) {
    dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
} else {
    "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/APMS_bridge/tests"
}
bridge_directory <-  dirname(path = tests_directory)
source(file = file.path(bridge_directory, "R_fn_APMS_BayesMVP.R"))
##
fn_expect_settings_error <-  function(expression) {
    caught_error <-  tryCatch(expr = { force(expression); NULL }, error = function(error_condition) error_condition)
    stopifnot(inherits(x = caught_error, what = "error"))
}
##
## ---- Defaults and dependent settings -------------------------------------------------------------------------------------------------
##
## Use an explicit baseline so edits to the user's preferred constructor defaults do not change this test case.
default_settings <-  fn_sampler_settings_APMS_BayesMVP(n_burnin = 250,
                                                      n_iter = 250,
                                                      n_chains_burnin = 4,
                                                      n_chains_sampling = 32,
                                                      metric_estimator = "pooled")
stopifnot(default_settings$n_burnin == 250,
          default_settings$n_iter == 250,
          default_settings$n_chains_burnin == 4,
          default_settings$n_chains_sampling == 32,
          default_settings$n_superchains == 32,
          default_settings$M_decay_scale == 2.25,
          identical(default_settings$metric_estimator, "pooled"),
          identical(default_settings$burnin_TBB_pool_equals_n_chains, FALSE),
          identical(default_settings$summary_options, list(save_log_lik_trace = FALSE, compute_nested_rhat = FALSE)))
stopifnot(isTRUE(x = fn_validate_settings_APMS_BayesMVP(settings = default_settings)))
##
short_burnin_settings <-  fn_sampler_settings_APMS_BayesMVP(n_burnin = 125,
                                                           n_chains_sampling = 64)
stopifnot(short_burnin_settings$M_decay_scale == 1.13,
          is.null(x = short_burnin_settings$clip_iter),
          is.null(x = short_burnin_settings$clip_iter_tau),
          short_burnin_settings$burnin_schedule == "automatic",
          short_burnin_settings$n_superchains == 64,
          fn_sampler_settings_APMS_BayesMVP(n_burnin = 1000)$M_decay_scale == 9,
          fn_sampler_settings_APMS_BayesMVP(n_burnin = 250, n_adapt = 200)$M_decay_scale == 2,
          fn_sampler_settings_APMS_BayesMVP(n_burnin = 125, M_decay_scale = 3)$M_decay_scale == 3,
          fn_sampler_settings_APMS_BayesMVP(n_chains_sampling = 64, n_superchains = 8)$n_superchains == 8)
##
## ---- Direct arguments reach the settings unchanged, including explicit NULL -----------------------------------------------------------
##
direct_arguments <-  list(n_burnin = 125,
                         n_iter = 125,
                         learning_rate = 0.075,
                         metric_estimator = "chain_mean_scaled",
                         tau_ramp = "original",
                         manual_tau = FALSE,
                         tau_if_manual = NULL,
                         tau_if_manual_in_L_units = FALSE,
                         theta_hat_us_rule = "running_mean_frozen",
                         theta_hat_us_freeze_iter = NULL,
                         eps_reinit_at_ChEES_handover = FALSE,
                         share_tau_ii_across_chains_in_burnin = TRUE,
                         burnin_TBB_pool_equals_n_chains = TRUE,
                         debug_burnin_timing = TRUE,
                         store_log_lik_trace = FALSE,
                         n_refresh = 10,
                         nuisance_jitter_scale = 0,
                         summary_options = list(save_log_lik_trace = FALSE, compute_nested_rhat = TRUE))
direct_settings <-  do.call(what = fn_sampler_settings_APMS_BayesMVP, args = direct_arguments)
stopifnot(identical(direct_settings[names(x = direct_arguments)], direct_arguments),
          isTRUE(x = fn_validate_settings_APMS_BayesMVP(settings = direct_settings)))
null_settings <-  fn_sampler_settings_APMS_BayesMVP(M_decay_scale = NULL, tau_if_manual = NULL)
stopifnot("M_decay_scale" %in% names(x = null_settings),
          "tau_if_manual" %in% names(x = null_settings),
          is.null(x = null_settings$M_decay_scale),
          is.null(x = null_settings$tau_if_manual),
          "tau_if_manual" %in% names(x = default_settings),
          is.null(x = default_settings$tau_if_manual))
##
## ---- Every exposed control is forwarded; misspellings, duplicate inputs and old nested inputs fail -----------------------------------
##
constructor_argument_names <-  names(x = formals(fun = fn_sampler_settings_APMS_BayesMVP))
for (argument_name in setdiff(x = constructor_argument_names, y = "summary_options")) {
    argument_value <-  setNames(object = list(NULL), nm = argument_name)
    ## n_burnin cannot be NULL while calculating the dependent default.
    if (identical(argument_name, "n_burnin")) argument_value <-  list(n_burnin = 125)
    argument_settings <-  do.call(what = fn_sampler_settings_APMS_BayesMVP, args = argument_value)
    stopifnot(identical(argument_settings[argument_name], argument_value))
}
fn_expect_settings_error(expression = fn_sampler_settings_APMS_BayesMVP(tau_rmap = "original"))
fn_expect_settings_error(expression = fn_sampler_settings_APMS_BayesMVP(sampler_options = list(n_burnin = 125)))
fn_expect_settings_error(expression = fn_sampler_settings_APMS_BayesMVP(burnin_post_adapt_iter = 5))
retired_option_settings <-  default_settings
retired_option_settings$burnin_post_adapt_iter <-  5
fn_expect_settings_error(expression = fn_validate_settings_APMS_BayesMVP(settings = retired_option_settings))
fn_expect_settings_error(expression = do.call(what = fn_sampler_settings_APMS_BayesMVP,
                                              args = list(n_burnin = 125, n_burnin = 250)))
fn_expect_settings_error(expression = fn_validate_settings_APMS_BayesMVP(
    settings = fn_sampler_settings_APMS_BayesMVP(tau_if_manual_in_L_units = NA)))
stopifnot(!("..." %in% constructor_argument_names), !("sampler_options" %in% constructor_argument_names))
##
## ---- Autodiff fallback: inherited session default, explicit override, worker snapshot and validation ----------------------------------
##
previous_fallback_option <-  options(BayesMVP_autodiff_fallback = NULL)
stopifnot(identical(fn_sampler_settings_APMS_BayesMVP()$autodiff_fallback, FALSE))
options(BayesMVP_autodiff_fallback = TRUE)
fallback_settings <-  fn_sampler_settings_APMS_BayesMVP()
stopifnot(identical(fallback_settings$autodiff_fallback, TRUE),
          identical(fn_sampler_settings_APMS_BayesMVP(autodiff_fallback = FALSE)$autodiff_fallback, FALSE),
          is.null(fn_sampler_settings_APMS_BayesMVP(autodiff_fallback = NULL)$autodiff_fallback))
options(BayesMVP_autodiff_fallback = FALSE)
stopifnot(identical(fallback_settings$autodiff_fallback, TRUE))
for (invalid_fallback in list(NA, "TRUE", 1, logical(), c(TRUE, FALSE))) {
    fn_expect_settings_error(expression = fn_validate_settings_APMS_BayesMVP(
        settings = fn_sampler_settings_APMS_BayesMVP(autodiff_fallback = invalid_fallback)))
}
##
## ---- Evaluate the bridge's actual option block, without initializing or sampling a model ----------------------------------------------
##
bridge_expressions <-  as.list(x = body(fun = fn_fit_APMS_model_BayesMVP))[-1]
fallback_option_blocks <-  Filter(f = function(expression) {
    is.call(x = expression) && identical(expression[[1]], as.name(x = "if")) &&
        identical(expression[[2]], quote(!is.null(x = settings$autodiff_fallback)))
}, x = bridge_expressions)
stopifnot(length(x = fallback_option_blocks) == 1)
fn_test_bridge_fallback_option <-  function(settings, fail_after_option = FALSE) NULL
## Insert the block into a function body so on.exit belongs to that function, not to an eval() call.
body(fun = fn_test_bridge_fallback_option) <-  substitute(expr = {
        OPTION_BLOCK
        if (fail_after_option) stop("Test failure after setting the option, without calling a sampler.")
        return(getOption(x = "BayesMVP_autodiff_fallback"))
}, env = list(OPTION_BLOCK = fallback_option_blocks[[1]]))
stopifnot(isTRUE(fn_test_bridge_fallback_option(settings = fallback_settings)),
          identical(getOption(x = "BayesMVP_autodiff_fallback"), FALSE))
fn_expect_settings_error(expression = fn_test_bridge_fallback_option(settings = fallback_settings, fail_after_option = TRUE))
stopifnot(identical(getOption(x = "BayesMVP_autodiff_fallback"), FALSE))
fallback_settings$autodiff_fallback <-  NULL
stopifnot(identical(fn_test_bridge_fallback_option(settings = fallback_settings), FALSE))
##
## The bridge accepts the option but never passes it as an unsupported R6 sample() argument.
settings <-  fn_sampler_settings_APMS_BayesMVP(autodiff_fallback = TRUE)
sample_argument_names <-  names(x = settings)
forwarding_expressions <-  Filter(f = function(expression) {
    is.call(x = expression) && identical(expression[[1]], as.name(x = "<-")) &&
        is.symbol(x = expression[[2]]) &&
        as.character(x = expression[[2]]) %in% c("constructor_fields", "forwarded_names")
}, x = bridge_expressions)
for (expression in forwarding_expressions) eval(expr = expression)
stopifnot("autodiff_fallback" %in% constructor_fields, !"autodiff_fallback" %in% forwarded_names)
options(previous_fallback_option)
cat("PASS: fallback defaults/overrides, worker-safe settings, invalid values, bridge routing and restoration after errors.\n")
##
cat("\nPASS: flat APMS sampler arguments, derived defaults, NULL preservation and invalid-input checks. No MCMC run.\n")
