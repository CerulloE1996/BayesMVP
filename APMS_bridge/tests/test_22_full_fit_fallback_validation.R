#### =====================================================================================================================================
## test_22_full_fit_fallback_validation.R - bridge-only fallback option in the benchmark validator; no compilation or model fits
## =====================================================================================================================================
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
bridge_directory <-  dirname(path = tests_directory)
workspace_directory <-  normalizePath(path = file.path(bridge_directory, "..", "..", ".."))
##
{
    require(RcppParallel)
    require(BayesMVP)
}
source(file = file.path(bridge_directory, "R_fn_APMS_BayesMVP.R"))
source(file = file.path(bridge_directory, "R_fn_APMS_full_fit_benchmark.R"))
##
## ---- Validation-only fixture: one thread, no model construction, no changes to the application runner ---------------------------------
##
test_configuration <-  list( engine = "BayesMVP",
                             stan_file = file.path(workspace_directory, "Autism_BPD_paper", "LC_MVOP_4class_joint_v1_reduce_sum.stan"),
                             sampler_settings = fn_sampler_settings_APMS_BayesMVP( n_burnin = 125,
                                                                                   n_chains_burnin = 1,
                                                                                   n_chains_sampling = 1,
                                                                                   n_threads_WCP_burnin = 1,
                                                                                   n_threads_WCP_sampling = 1,
                                                                                   num_chunks_burnin = 1,
                                                                                   num_chunks_sampling = 1,
                                                                                   autodiff_fallback = FALSE))
##
fallback_option_before_validation <-  getOption(x = "BayesMVP_autodiff_fallback")
for (fallback_value in list(FALSE, TRUE, NULL)) {
    test_configuration$sampler_settings["autodiff_fallback"] <-  list(fallback_value)
    configuration_before_validation <-  test_configuration
    threads_required <-  fn_validate_APMS_benchmark_configuration( configuration = test_configuration,
                                                                  thread_budget = 1)
    stopifnot(threads_required == 1,
              identical(x = test_configuration, y = configuration_before_validation),
              identical(x = getOption(x = "BayesMVP_autodiff_fallback"), y = fallback_option_before_validation))
}
test_configuration$sampler_settings$autodiff_fallback <-  NULL
stopifnot(fn_validate_APMS_benchmark_configuration(configuration = test_configuration, thread_budget = 1) == 1)
##
## ---- Invalid values and unknown options still fail; accepting this field does not bypass validation ----------------------------------
##
for (invalid_fallback in list(NA, "TRUE", 1, logical(), c(TRUE, FALSE))) {
    test_configuration$sampler_settings["autodiff_fallback"] <-  list(invalid_fallback)
    error_message <-  tryCatch(
        expr = fn_validate_APMS_benchmark_configuration(configuration = test_configuration, thread_budget = 1),
        error = function(condition) conditionMessage(c = condition))
    stopifnot(is.character(x = error_message),
              grepl(pattern = "'autodiff_fallback' must be NULL, TRUE or FALSE", x = error_message, fixed = TRUE))
}
test_configuration$sampler_settings$autodiff_fallback <-  FALSE
test_configuration$sampler_settings$autodiff_fallback_typo <-  TRUE
error_message <-  tryCatch(
    expr = fn_validate_APMS_benchmark_configuration(configuration = test_configuration, thread_budget = 1),
    error = function(condition) conditionMessage(c = condition))
stopifnot(is.character(x = error_message), grepl(pattern = "autodiff_fallback_typo", x = error_message, fixed = TRUE))
##
cat("PASS: benchmark accepts bridge-only fallback FALSE/TRUE/NULL/absent, rejects invalid values and unknown settings, and changes no options. No fits.\n")
