#### =====================================================================================================================================
## test_11_sampling_chunk_handover.R - isolated handover checks; no MCMC, compilation or package installation
## =====================================================================================================================================
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  if (length(x = test_script_arguments)) {
    dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
} else {
    "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/APMS_bridge/tests"
}
repository_directory <-  dirname(path = dirname(path = tests_directory))
sample_expressions <-  parse(file = file.path(repository_directory, "inst", "BayesMVP", "R", "R_fn_sample.R"))
##
## ---- Test the actual handover expressions, without running the sampler --------------------------------------------------------------
##
fn_find_handover_expressions <-  function(expression, predicate) {
    matches <-  list()
    if (is.call(x = expression) && predicate(expression)) matches <-  list(expression)
    if (is.call(x = expression) || is.expression(x = expression)) {
        for (child in as.list(x = expression)) {
            if (missing(child)) next
            if (is.call(x = child) || is.expression(x = child)) {
                matches <-  c(matches, fn_find_handover_expressions(expression = child, predicate = predicate))
            }
        }
    }
    matches
}
handover_blocks <-  fn_find_handover_expressions(expression = sample_expressions, predicate = function(expression) {
    identical(expression[[1]], as.name(x = "{")) && any(vapply(X = as.list(x = expression)[-1], FUN = function(child) {
        identical(child, quote(EHMC_args_as_Rcpp_List$store_log_lik_trace <- store_log_lik_trace))
    }, FUN.VALUE = FALSE))
})
stopifnot(length(x = handover_blocks) == 1)
handover_expressions <-  as.list(x = handover_blocks[[1]])[-1]
remap_condition <-  quote((Model_type != "Stan") && (n_nuisance > 0) &&
                             (num_chunks_used_in_sampling != num_chunks_used_in_burnin))
remap_position <-  which(x = vapply(X = handover_expressions, FUN = function(expression) {
    is.call(x = expression) && identical(expression[[1]], as.name(x = "if")) && identical(expression[[2]], remap_condition)
}, FUN.VALUE = FALSE))
stopifnot(length(x = remap_position) == 1)
chunk_handover_expressions <-  handover_expressions[seq_len(length.out = remap_position)]
chunk_handover_expressions <-  Filter(f = function(expression) !identical(expression, quote(gc())), x = chunk_handover_expressions)
fn_run_chunk_handover_check <-  function(Model_type, Model_args_as_Rcpp_List, num_chunks_sampling) {
    model_args_list <-  list(num_chunks = 25)
    n_nuisance <-  0
    for (expression in chunk_handover_expressions) eval(expr = expression, envir = environment())
    Model_args_as_Rcpp_List
}
##
## ---- External Stan: never create or overwrite the unused built-in matrix ------------------------------------------------------------
##
stan_model_arguments <-  list(N = 100, n_nuisance = 10, n_params_main = 4, json_file_path = "burnin.json")
for (sampling_chunk_count in list(NULL, 32, 64)) {
    stopifnot(identical(fn_run_chunk_handover_check(Model_type = "Stan",
                                                    Model_args_as_Rcpp_List = stan_model_arguments,
                                                    num_chunks_sampling = sampling_chunk_count), stan_model_arguments))
}
stan_model_arguments_with_matrix <-  stan_model_arguments
stan_model_arguments_with_matrix$Model_args_ints <-  matrix(data = c(1, 2, 3, 64), ncol = 1)
stopifnot(identical(fn_run_chunk_handover_check(Model_type = "Stan",
                                                Model_args_as_Rcpp_List = stan_model_arguments_with_matrix,
                                                num_chunks_sampling = 32), stan_model_arguments_with_matrix))
##
## ---- Built-in models: explicit and fallback chunk counts still update the matrix -----------------------------------------------------
##
for (builtin_model_type in c("MVP", "LC_MVP", "MVOP", "LC_MVOP", "latent_trait")) {
    for (sampling_chunk_count in list(NULL, 32, 64)) {
        builtin_model_arguments <-  list(Model_args_ints = matrix(data = c(1, 2, 3, 64, 5), ncol = 1))
        expected_model_arguments <-  builtin_model_arguments
        expected_model_arguments$Model_args_ints[4] <-  if (is.null(x = sampling_chunk_count)) 25 else sampling_chunk_count
        actual_model_arguments <-  fn_run_chunk_handover_check(Model_type = builtin_model_type,
                                                                Model_args_as_Rcpp_List = builtin_model_arguments,
                                                                num_chunks_sampling = sampling_chunk_count)
        stopifnot(identical(actual_model_arguments, expected_model_arguments), is.matrix(x = actual_model_arguments$Model_args_ints))
    }
}
##
## ---- Stan: independent phase chunk counts still become grainsizes, and sampling data are forwarded to JSON ----------------------------
##
stan_grainsize_setup <-  fn_find_handover_expressions(expression = sample_expressions, predicate = function(expression) {
    identical(expression[[1]], as.name(x = "if")) &&
        identical(expression[[2]], quote(Model_type == "Stan" && (!is.null(num_chunks_burnin) || !is.null(num_chunks_sampling))))
})
stan_sampling_data_update <-  fn_find_handover_expressions(expression = sample_expressions, predicate = function(expression) {
    identical(expression[[1]], as.name(x = "if")) &&
        identical(expression[[2]], quote(Model_type == "Stan" && !is.null(stan_grainsize_sampling)))
})
stopifnot(length(x = stan_grainsize_setup) == 1, length(x = stan_sampling_data_update) == 1)
fn_check_stan_phase_grainsizes <-  function(num_chunks_burnin, num_chunks_sampling) {
    Model_type <-  "Stan"
    Stan_data_list <-  list(N = 7403, grainsize = 500, other_model_data = c(1, 2, 3))
    Model_args_as_Rcpp_List <-  stan_model_arguments
    init_object <-  list(json_file_path = "burnin.json")
    stan_grainsize_sampling <-  NULL
    captured_sampling_data <-  NULL
    convert_stan_data_list_to_JSON <-  function(stan_data_list, pkg_data_dir) {
        captured_sampling_data <<-  stan_data_list
        "sampling.json"
    }
    eval(expr = stan_grainsize_setup[[1]], envir = environment())
    expected_burnin_grainsize <-  if (is.null(x = num_chunks_burnin)) 500 else as.integer(x = ceiling(x = 7403 / num_chunks_burnin))
    expected_sampling_grainsize <-  if (is.null(x = num_chunks_sampling)) 500 else as.integer(x = ceiling(x = 7403 / num_chunks_sampling))
    stopifnot(Stan_data_list$grainsize == expected_burnin_grainsize)
    for (expression in chunk_handover_expressions) eval(expr = expression, envir = environment())
    stopifnot(identical(Model_args_as_Rcpp_List, stan_model_arguments))
    eval(expr = stan_sampling_data_update[[1]], envir = environment())
    if (!is.null(x = num_chunks_burnin) || !is.null(x = num_chunks_sampling)) {
        stopifnot(captured_sampling_data$grainsize == expected_sampling_grainsize,
                  identical(captured_sampling_data$other_model_data, c(1, 2, 3)),
                  identical(Model_args_as_Rcpp_List$json_file_path, "sampling.json"))
    } else {
        stopifnot(is.null(x = captured_sampling_data), identical(Model_args_as_Rcpp_List, stan_model_arguments))
    }
}
for (burnin_chunk_count in list(NULL, 64)) {
    for (sampling_chunk_count in list(NULL, 32)) {
        fn_check_stan_phase_grainsizes(num_chunks_burnin = burnin_chunk_count, num_chunks_sampling = sampling_chunk_count)
    }
}
cat("\nPASS: external Stan handover leaves built-in arguments unchanged; built-in chunk-count updates are preserved.\n")
cat("PASS: independent Stan phase grainsizes and sampling JSON handover are preserved, including NULL defaults.\n")
