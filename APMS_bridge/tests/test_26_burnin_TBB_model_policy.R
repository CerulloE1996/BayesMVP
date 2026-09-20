#### =====================================================================================================================================
## test_26_burnin_TBB_model_policy.R - model-specific burn-in TBB policy; no compilation or MCMC
## =====================================================================================================================================
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  if (length(x = test_script_arguments)) {
    dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
} else {
    "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/APMS_bridge/tests"
}
bridge_directory <-  dirname(path = tests_directory)
package_directory <-  file.path(dirname(path = bridge_directory), "inst", "BayesMVP")
test_environment <-  new.env(parent = globalenv())
sys.source(file = file.path(package_directory, "R", "R_fn_sample.R"), envir = test_environment)
sys.source(file = file.path(package_directory, "R", "R_R6_class_aaa_init_and_sample.R"), envir = test_environment)
sys.source(file = file.path(bridge_directory, "R_fn_APMS_BayesMVP.R"), envir = test_environment)
##
## ---- Exercise the actual sampler entry policy, stopping before all model initialisation and sampling -------------------------------
##
sampler_probe <-  test_environment$R_fn_sample_model
sampler_expressions <-  as.list(x = body(fun = sampler_probe))[-1]
message_index <-  which(vapply(X = sampler_expressions, FUN = function(expression) {
    is.call(x = expression) && identical(x = expression[[1]], y = as.name(x = "message"))
}, FUN.VALUE = FALSE))[1]
stopifnot(message_index > 1L)
body(sampler_probe) <-  as.call(x = c(list(as.name(x = "{")), sampler_expressions[seq_len(message_index - 1L)],
                                    list(quote(return(burnin_TBB_pool_equals_n_chains)))))
## Read the actual burn-in thread-budget expression rather than duplicating its logic in the test.
pool_calls <-  Filter(f = function(expression) {
    is.call(x = expression) && identical(x = expression[[1]], y = quote(RcppParallel::setThreadOptions))
}, x = sampler_expressions)
stopifnot(length(x = pool_calls) == 1L)
##
## ---- R6 method: run the complete sample() wrapper with a lightweight constructor and sampler capture --------------------------------
##
test_class <-  test_environment$MVP_model
test_class$set(which = "public", name = "initialize", overwrite = TRUE,
               value = function(Model_type) {
                   self$Model_type <-  Model_type
                   self$init_object <-  list(Model_type = Model_type, n_nuisance = 0L, n_params_main = 1L)
               })
sampler_capture <-  function() NULL
formals(sampler_capture) <-  formals(test_environment$R_fn_sample_model)
body(sampler_capture) <-  quote(list(init_object = init_object,
                                    burnin_TBB_pool_equals_n_chains = burnin_TBB_pool_equals_n_chains))
test_environment$R_fn_sample_model <-  sampler_capture
##
case_count <-  0L
for (model_type in c("Stan", "MVP", "LC_MVP", "MVOP", "LC_MVOP", "latent_trait")) {
    for (choice in c("omitted", "NULL", "TRUE", "FALSE")) {
        explicit_arguments <-  switch(EXPR = choice, omitted = list(),
                                       "NULL" = list(burnin_TBB_pool_equals_n_chains = NULL),
                                       "TRUE" = list(burnin_TBB_pool_equals_n_chains = TRUE),
                                       "FALSE" = list(burnin_TBB_pool_equals_n_chains = FALSE))
        expected_value <-  model_type != "Stan" && choice != "FALSE"
        messages <-  character()
        direct_result <-  withCallingHandlers(
            expr = do.call(what = sampler_probe, args = c(list(init_object = list(Model_type = model_type)), explicit_arguments)),
            message = function(condition) {
                messages <<- c(messages, conditionMessage(condition))
                invokeRestart(r = "muffleMessage")
            })
        stopifnot(identical(x = direct_result, y = expected_value),
                  identical(x = length(x = messages), y = if (model_type == "Stan" && choice == "TRUE") 1L else 0L))
        ##
        model <-  test_class$new(Model_type = model_type)
        messages <-  character()
        result <-  withCallingHandlers(expr = do.call(what = model$sample, args = explicit_arguments),
                                        message = function(condition) {
                                            messages <<- c(messages, conditionMessage(condition))
                                            invokeRestart(r = "muffleMessage")
                                        })
        stopifnot(identical(x = result$result$burnin_TBB_pool_equals_n_chains, y = expected_value),
                  identical(x = length(x = messages), y = if (model_type == "Stan" && choice == "TRUE") 1L else 0L))
        ##
        thread_budget <-  eval(expr = pool_calls[[1]][["numThreads"]],
                                envir = list(burnin_TBB_pool_equals_n_chains = direct_result,
                                              n_chains_burnin = 4L, n_threads_WCP_burnin = 16L))
        stopifnot(thread_budget == if (expected_value) 4L else 64L)
        case_count <-  case_count + 1L
    }
}
##
stopifnot(identical(x = test_environment$fn_sampler_settings_APMS_BayesMVP()$burnin_TBB_pool_equals_n_chains, y = FALSE))
cat("PASS: ", case_count, " model/input combinations through both sampler entry policy and the R6 wrapper; ",
    "Stan TRUE overrides report once; actual burn-in budgets are 4 or 64 as expected; APMS defaults to FALSE. No MCMC.\n", sep = "")
