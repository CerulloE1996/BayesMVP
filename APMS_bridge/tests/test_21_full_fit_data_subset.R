#### =====================================================================================================================================
## test_21_full_fit_data_subset.R - common APMS benchmark subsets, using the real subsetting helper; no compilation or model fits
## =====================================================================================================================================
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
workspace_directory <-  normalizePath(path = file.path(tests_directory, "..", "..", "..", ".."))
source(file = file.path(tests_directory, "..", "R_fn_APMS_full_fit_benchmark.R"))
##
## ---- Load only the two existing data helpers; do not evaluate application scripts or sampling code -----------------------------------
##
data_helper_expressions <-  parse(file = file.path(workspace_directory, "Autism_BPD_paper", "R_fn_sim_4class_joint_LC_MVOP_data.R"))
required_helper_names <-  c("fn_subset_stan_data", "fn_build_standardisation_set")
for (helper_expression in data_helper_expressions) {
    if (is.call(x = helper_expression) && identical(x = helper_expression[[1]], y = as.name(x = "<-")) &&
        is.symbol(x = helper_expression[[2]]) && as.character(x = helper_expression[[2]]) %in% required_helper_names) {
        eval(expr = helper_expression)
    }
}
actual_subset_helper <-  fn_subset_stan_data
##
## ---- A small, entirely synthetic ordinal base with aligned covariates, weights and GQ settings ---------------------------------------
##
test_stan_data_base <-  list( N = 240,
                             n_tests = 4,
                             n_pops = 2,
                             y = cbind(rep(x = 1:4, times = 60), rep(x = 1:3, times = 80),
                                       rep(x = c(3, 8, 12), times = 80), rep(x = 1:6, times = 40)),
                             pop = rep(x = 1:2, each = 120),
                             X = array(data = seq_len(length.out = 4 * 240 * 2), dim = c(4, 240, 2)),
                             n_covs_per_outcome = rep(x = 2, times = 4),
                             X_prevalence = cbind(1, seq_len(length.out = 240) / 240),
                             target_row_weight = seq(from = 1, to = 2, length.out = 240),
                             grainsize = 12,
                             n_loo_draws = 17,
                             prior_settings = list(alpha = c(2, 3), prevalence = c(4, 5)))
test_stan_data_base <-  fn_build_standardisation_set( stan_data = test_stan_data_base,
                                                      survey_weight = test_stan_data_base$target_row_weight,
                                                      verbose = FALSE)
subset_call_counter <-  new.env(parent = emptyenv())
subset_call_counter$count <-  0
##
temporary_bridge_file <-  tempfile(pattern = "apms_subset_preparation_", fileext = ".R")
writeLines(text = c(
    "stan_data_base <- test_stan_data_base",
    "MODELS_TO_FIT <- c('M4_perfect_dep', 'M5_imperfect_bin', 'M6_imperfect_ord')",
    "sampler_settings <- list(n_burnin = 125, n_iter = 50)",
    "BASE_DIR <- tempdir()",
    "fn_subset_stan_data <- function(stan_data, n_total, seed, oversample_verified, min_per_cell, verbose) {",
    "    subset_call_counter$count <- subset_call_counter$count + 1",
    "    actual_subset_helper(stan_data = stan_data, n_total = n_total, seed = seed,",
    "                         oversample_verified = oversample_verified, min_per_cell = min_per_cell, verbose = verbose)",
    "}",
    "fn_build_stan_data_real <- function(base, model_config) {",
    "    base$model_configuration <- model_config",
    "    return(base)",
    "}",
    "APMS_PREPARATION_COMPLETE <- TRUE",
    "stop('The preparation helper must never reach the fitting section.')"
), con = temporary_bridge_file)
##
## ---- Full data remain unchanged; a seeded subset is drawn once, before all model-specific builders ------------------------------------
##
full_data_output <-  capture.output(full_data_result <- fn_prepare_APMS_full_fit_benchmark(bridge_file = temporary_bridge_file))
stopifnot(identical(x = full_data_result$stan_data_base, y = test_stan_data_base), subset_call_counter$count == 0,
          is.null(x = full_data_result$N_subset), is.null(x = full_data_result$subset_seed))
##
set.seed(seed = 654)
rng_state_before_preparation <-  .Random.seed
subset_output <-  capture.output(subset_result <- fn_prepare_APMS_full_fit_benchmark( bridge_file = temporary_bridge_file,
                                                                                     N_subset = 120,
                                                                                     subset_seed = 123))
stopifnot(subset_call_counter$count == 1, identical(x = .Random.seed, y = rng_state_before_preparation),
          subset_result$N_full == 240, subset_result$N_subset == 120, subset_result$subset_seed == 123,
          any(grepl(pattern = "shared across all models and engines", x = subset_output)))
selected_rows <-  subset_result$stan_data_base$subset_row_index
stopifnot(length(x = selected_rows) == 120, length(x = unique(x = selected_rows)) == 120)
for (model_data in subset_result$stan_data_by_model) {
    stopifnot(model_data$N == 120,
              identical(x = model_data$subset_row_index, y = selected_rows),
              identical(x = model_data$y, y = test_stan_data_base$y[selected_rows, , drop = FALSE]),
              identical(x = model_data$X, y = test_stan_data_base$X[, selected_rows, , drop = FALSE]),
              identical(x = model_data$X_prevalence, y = test_stan_data_base$X_prevalence[selected_rows, , drop = FALSE]),
              identical(x = model_data$pop, y = test_stan_data_base$pop[selected_rows]),
              identical(x = model_data$target_row_weight, y = test_stan_data_base$target_row_weight[selected_rows]),
              all(model_data$standardisation_index >= 1 & model_data$standardisation_index <= 120),
              abs(sum(model_data$standardisation_weight) - sum(model_data$target_row_weight)) < 1e-10,
              identical(x = model_data$prior_settings, y = test_stan_data_base$prior_settings),
              model_data$n_loo_draws == 17, model_data$grainsize == 12)
}
stopifnot(identical(x = subset_result$sampler_settings, y = full_data_result$sampler_settings))
##
## ---- Reproducible selection and fingerprints; different rows cannot resume a full-data result ----------------------------------------
##
repeat_output <-  capture.output(repeat_result <- fn_prepare_APMS_full_fit_benchmark( bridge_file = temporary_bridge_file,
                                                                                     N_subset = 120,
                                                                                     subset_seed = 123))
different_seed_output <-  capture.output(different_seed_result <- fn_prepare_APMS_full_fit_benchmark( bridge_file = temporary_bridge_file,
                                                                                                     N_subset = 120,
                                                                                                     subset_seed = 456))
stopifnot(identical(x = subset_result$stan_data_by_model, y = repeat_result$stan_data_by_model),
          !identical(x = selected_rows, y = different_seed_result$stan_data_base$subset_row_index),
          identical(x = fn_APMS_benchmark_fingerprint(object = subset_result$stan_data_by_model),
                    y = fn_APMS_benchmark_fingerprint(object = repeat_result$stan_data_by_model)),
          !identical(x = fn_APMS_benchmark_fingerprint(object = subset_result$stan_data_by_model),
                     y = fn_APMS_benchmark_fingerprint(object = full_data_result$stan_data_by_model)),
          !identical(x = fn_APMS_benchmark_fingerprint(object = subset_result$stan_data_by_model),
                     y = fn_APMS_benchmark_fingerprint(object = different_seed_result$stan_data_by_model)))
##
## ---- Invalid controls fail before selection; preparation never evaluates the fitting section -----------------------------------------
##
for (invalid_size in list(0, -1, 1.5, NA_real_, Inf, TRUE, "120", c(10, 20))) {
    error_message <-  tryCatch(expr = fn_prepare_APMS_full_fit_benchmark(bridge_file = temporary_bridge_file, N_subset = invalid_size),
                              error = function(condition) conditionMessage(c = condition))
    stopifnot(is.character(x = error_message), grepl(pattern = "N_subset must be", x = error_message))
}
for (invalid_seed in list(NULL, -1, 1.5, NA_real_, Inf, TRUE, "123", c(1, 2))) {
    error_message <-  tryCatch(expr = fn_prepare_APMS_full_fit_benchmark( bridge_file = temporary_bridge_file,
                                                                         N_subset = 120,
                                                                         subset_seed = invalid_seed),
                              error = function(condition) conditionMessage(c = condition))
    stopifnot(is.character(x = error_message), grepl(pattern = "subset_seed must be", x = error_message))
}
subset_calls_before_invalid_size <-  subset_call_counter$count
error_message <-  tryCatch(expr = fn_prepare_APMS_full_fit_benchmark(bridge_file = temporary_bridge_file, N_subset = 241),
                          error = function(condition) conditionMessage(c = condition))
stopifnot(grepl(pattern = "exceeds the prepared data size", x = error_message), subset_call_counter$count == subset_calls_before_invalid_size)
##
invisible(x = parse(file = file.path(workspace_directory, "Autism_BPD_paper", "R_benchmark_APMS_full_fits.R")))
unlink(x = temporary_bridge_file)
cat("PASS: common stratified subset, reproducibility, aligned inputs/weights, fingerprints, full-data defaults, validation and driver syntax. No fits.\n")
