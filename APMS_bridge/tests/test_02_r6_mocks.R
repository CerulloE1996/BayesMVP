#### ================================================================================================================================================================
## test_02_r6_mocks.R
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
## Focused R6 interface tests using MOCKS for the heavy underlying functions
## (initialise_model, R_fn_sample_model, create_summary_and_traces,
## MVP_plot_and_diagnose, bridgestan_path). Mocks only prove INTERFACE wiring -
## they are NOT evidence of BridgeStan compilation or sampling.
##
## Covers: deferred initialisation (constructor inputs preserved), argument
## forwarding ($sample -> R_fn_sample_model), state refresh after sampling, and
## $summary() storage-mode consistency.
## ================================================================================================================================================================
##
suppressPackageStartupMessages({
    loadNamespace("RcppParallel")
    require(BayesMVP)
})
##
## ---- locate this script's own directory (works from Rscript AND from a
## ---- source()d RStudio session, whatever getwd() is):
##
{
    command_args <- commandArgs(trailingOnly = FALSE)
    ##
    file_arg <- grep( pattern = "^--file=",
                      x = command_args,
                      value = TRUE)
    ##
    if (length(file_arg) > 0) {
        TESTS_DIR <- normalizePath(path = dirname(path = sub( pattern = "^--file=",
                                                              replacement = "",
                                                              x = file_arg[1])))
    } else {
        fallback_candidates <- c( file.path( getwd(),
                                             "tests"),
                                  "/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/APMS_bridge/tests")
        ##
        existing_candidates <- fallback_candidates[dir.exists(fallback_candidates)]
        ##
        TESTS_DIR <- if (length(existing_candidates) > 0) normalizePath(existing_candidates[1]) else getwd()
    }
}
##
OUT_DIR <- file.path( TESTS_DIR,
                      "test_output")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
##
results <- list()
n_pass <- 0
n_fail <- 0
##
fn_check <- function(name, 
                     expr) {
  
    ok <- tryCatch({ isTRUE(expr) }, error = function(e) { cat("  ERROR:", conditionMessage(e), "\n"); FALSE })
    if (isTRUE(ok)) { n_pass <<- n_pass + 1; cat(sprintf("  PASS: %s\n", name)) }
    else            { n_fail <<- n_fail + 1; cat(sprintf("  FAIL: %s\n", name)) }
    results[[name]] <<- ok
    
}
##
cat("\n======== test_02: R6 interface (mocked underlying fns) ========\n")
##
## ---- mock capture environment -----------------------------------------------------------------------------------------------------------------------------------
##
mock_env <- new.env()
mock_env$initialise_model_calls <- list()
mock_env$sample_model_calls <- list()
mock_env$summary_calls <- list()
##
mock_env$init_object_template <- function() {
    list( Model_type = "Stan",
          stream = 1,
          sample_nuisance = TRUE,
          n_nuisance_override = NULL,
          n_nuisance = 99L,
          n_params_main = 7L,
          n_params = 106L,
          n_nuisance_names_constrained = 99L,
          bs_model = NULL,
          model_args_list = list(),
          Stan_data_list = list(N = 10),
          Stan_model_file_path = "/mock/model.stan",
          json_file_path = "/mock/data.json",
          model_so_file = "/mock/model_model.so",
          dummy_json_file_path = NULL,
          dummy_model_so_file = NULL,
          Stan_cpp_user_header = NULL,
          Stan_cpp_flags = NULL,
          stanc_args = NULL,
          make_args = NULL)
}
##
## ---- stub the heavy functions INSIDE the BayesMVP namespace -----------------------------------------------------------------------------------------------------
##
ns <- asNamespace("BayesMVP")
##
stub_initialise_model <- function( Model_type,
                                   stream = NULL,
                                   sample_nuisance = NULL,
                                   n_nuisance_override = NULL,
                                   model_args_list = NULL,
                                   Stan_data_list = NULL,
                                   compile = TRUE,
                                   force_recompile = FALSE,
                                   cmdstanr_model_fit_obj = NULL,
                                   Stan_model_file_path = NULL,
                                   Stan_cpp_user_header = NULL,
                                   Stan_cpp_flags = NULL,
                                   stanc_args = NULL,
                                   make_args = NULL) {
  
    mock_env$initialise_model_calls[[length(mock_env$initialise_model_calls) + 1L]] <<-
        list( Model_type = Model_type,
              sample_nuisance = sample_nuisance,
              n_nuisance_override = n_nuisance_override,
              model_args_list = model_args_list,
              Stan_data_list = Stan_data_list,
              compile = compile,
              force_recompile = force_recompile,
              Stan_model_file_path = Stan_model_file_path,
              Stan_cpp_user_header = Stan_cpp_user_header,
              Stan_cpp_flags = Stan_cpp_flags,
              stanc_args = stanc_args,
              make_args = make_args)
    ##
    init_object <- mock_env$init_object_template()
    init_object$Stan_data_list <- Stan_data_list
    init_object$Stan_model_file_path <- Stan_model_file_path
    init_object$model_args_list <- model_args_list
    init_object$Stan_cpp_user_header <- Stan_cpp_user_header
    init_object$Stan_cpp_flags <- Stan_cpp_flags
    init_object$stanc_args <- stanc_args
    init_object$make_args <- make_args
    init_object$sample_nuisance <- sample_nuisance
    init_object$n_nuisance_override <- n_nuisance_override
    init_object
    
}
##
stub_sample_model <- function( init_object,
                               n_chains_burnin,
                               init_lists_per_chain,
                               parallel_method,
                               Stan_data_list,
                               model_args_list,
                               sample_nuisance,
                               n_nuisance_override = NULL,
                               seed,
                               n_burnin,
                               n_adapt,
                               gap,
                               n_chains_sampling,
                               n_superchains,
                               n_iter,
                               adapt_delta,
                               learning_rate,
                               tau_mult,
                               tau_initial,
                               manual_tau,
                               tau_if_manual,
                               burnin_algorithm = "ChESSR",
                               diffusion_HMC,
                               partitioned_HMC,
                               clip_iter,
                               clip_iter_tau,
                               n_refresh = 100,
                               use_proposed = TRUE,
                               beta1_adam = 0.00,
                               beta2_adam = 0.95,
                               eps_adam = 1e-8,
                               force_autodiff,
                               force_PartialLog,
                               multi_attempts,
                               force_autodiff_for_metric = TRUE,
                               force_PartialLog_for_metric = FALSE,
                               force_multi_attempts_for_metric = FALSE,
                               vect_type,
                               Phi_type,
                               inv_Phi_type,
                               metric_type_main,
                               metric_shape_main,
                               ratio_M_main,
                               interval_width_main,
                               M_decay_type,
                               M_decay_power,
                               M_decay_scale,
                               metric_type_nuisance,
                               metric_shape_nuisance,
                               ratio_M_nuisance,
                               interval_width_nuisance,
                               max_tau_main = 25.0,
                               max_tau_nuisance = 25.0,
                               max_eps_main,
                               max_eps_nuisance,
                               max_L,
                               n_nuisance_to_track = NULL,
                               use_disk,
                               use_disk_path = "/tmp/hmc_traces",
                               n_threads_WCP_burnin,
                               n_threads_WCP_sampling,
                               reorder_cols_MVP,
                               metric_estimator = "pooled",
                               num_chunks_burnin = NULL,
                               num_chunks_sampling = NULL) {
  
    mock_env$sample_model_calls[[length(mock_env$sample_model_calls) + 1L]] <<-
        list( init_object = init_object,
              M_decay_type = M_decay_type,
              M_decay_power = M_decay_power,
              M_decay_scale = M_decay_scale,
              use_disk = use_disk,
              n_threads_WCP_burnin = n_threads_WCP_burnin,
              n_threads_WCP_sampling = n_threads_WCP_sampling,
              reorder_cols_MVP = reorder_cols_MVP,
              sample_nuisance = sample_nuisance,
              n_chains_burnin = n_chains_burnin,
              init_lists_per_chain = init_lists_per_chain,
              n_iter = n_iter)
    ## the sampler returns an UPDATED init_object:
    updated_init_object <- init_object
    updated_init_object$n_nuisance <- 5L
    updated_init_object$n_params_main <- 3L
    list( init_object = updated_init_object,
           burnin_object = list(n_chains_burnin = 2, time_burnin = 0.1),
           sampling_object = list( list(matrix(1, 3, 5), matrix(2, 3, 5)), list(matrix(0, 1, 5), matrix(0, 1, 5)) ),
           test_perm = NULL, test_inv_perm = NULL, LR_main = 0.1, LR_us = 0.1, adapt_delta = 0.8,
           n_chains_burnin = 2, n_burnin = 10, metric_type_main = "Empirical", metric_shape_main = "diag",
           metric_type_nuisance = "Empirical", metric_shape_nuisance = "diag", diffusion_HMC = TRUE,
           partitioned_HMC = TRUE, n_superchains = 1, interval_width_main = 5, interval_width_nuisance = 5,
           force_autodiff = FALSE, force_PartialLog = FALSE, multi_attempts = FALSE,
           time_burnin = 0.1, time_sampling = 0.2, time_total = 0.3)
    
}
##
stub_create_summary_and_traces <- function( model_results,
                                            compute_main_params = TRUE,
                                            compute_transformed_parameters = TRUE,
                                            compute_generated_quantities = TRUE,
                                            save_log_lik_trace = FALSE,
                                            save_nuisance_trace = FALSE,
                                            compute_nested_rhat = FALSE,
                                            n_superchains = NULL,
                                            save_trace_tibbles = FALSE,
                                            n_iter_to_store = NULL,
                                            use_disk = TRUE,
                                            use_disk_path = "/tmp/hmc_traces",
                                            use_disk_path_post_hoc_dir = "/tmp/constrain_traces") {
  
    mock_env$summary_calls[[length(mock_env$summary_calls) + 1L]] <<-
        list( model_results = model_results,
              use_disk = use_disk,
              use_disk_path = use_disk_path,
              use_disk_path_post_hoc_dir = use_disk_path_post_hoc_dir)
    list( summaries = list(summary_tibbles = list(NULL, NULL, NULL),
                           divergences = list(n_divs = 0, pct_divs = 0),
                           efficiency_info = list(), HMC_info = list()),
           traces = list(traces_as_arrays = NULL, traces_as_tibbles = NULL,
                         log_lik_trace = NULL, nuisance_trace = NULL))
    
}
##
stub_plot_class <- R6::R6Class( "stub_plot_class",
                                public = list(
                                    model_summary = NULL,
                                    init_object = NULL,
                                    n_nuisance = NULL,
                                    initialize = function(model_summary, init_object, n_nuisance) {
                                        self$model_summary <- model_summary
                                        self$init_object <- init_object
                                        self$n_nuisance <- n_nuisance
                                    }))
##
stub_bridgestan_path <- function() "/tmp/nonexistent/bridgestan"
##
## ---- install stubs into the package namespace --------------------------------------------------------------------------------------------------------------------
##
unlockBinding("initialise_model", ns)
assign("initialise_model", stub_initialise_model, envir = ns)
unlockBinding("R_fn_sample_model", ns)
assign("R_fn_sample_model", stub_sample_model, envir = ns)
unlockBinding("create_summary_and_traces", ns)
assign("create_summary_and_traces", stub_create_summary_and_traces, envir = ns)
unlockBinding("MVP_plot_and_diagnose", ns)
assign("MVP_plot_and_diagnose", stub_plot_class, envir = ns)
unlockBinding("bridgestan_path", ns)
assign("bridgestan_path", stub_bridgestan_path, envir = ns)
##
## ---- (a) EAGER initialisation with data --------------------------------------------------------------------------------------------------------------------------
##
cat("\n(a) eager initialisation:\n")
##
model_a <- BayesMVP::MVP_model$new( Model_type = "Stan",
                                    sample_nuisance = TRUE,
                                    Stan_data_list = list(N = 10),
                                    Stan_model_file_path = "/mock/model.stan")
##
fn_check("a.1 initialise_model called once", length(mock_env$initialise_model_calls) == 1L)
fn_check("a.2 is_compiled TRUE", isTRUE(model_a$is_compiled))
fn_check("a.3 n_nuisance refreshed from init_object", model_a$n_nuisance == 99L)
fn_check("a.4 n_params_main refreshed from init_object", model_a$n_params_main == 7L)
fn_check("a.5 Stan_model_file_path stored", model_a$Stan_model_file_path == "/mock/model.stan")
##
## ---- (b) DEFERRED initialisation ---------------------------------------------------------------------------------------------------------------------------------
##
cat("\n(b) deferred initialisation:\n")
##
model_b <- BayesMVP::MVP_model$new( Model_type = "Stan",
                                    sample_nuisance = TRUE,
                                    Stan_model_file_path = "/mock/model.stan")   ## NO data
##
fn_check("b.1 initialise_model NOT called at construction", length(mock_env$initialise_model_calls) == 1L)
fn_check("b.2 is_compiled FALSE (deferred)", isFALSE(model_b$is_compiled))
fn_check("b.3 constructor inputs PRESERVED: sample_nuisance", isTRUE(model_b$sample_nuisance))
fn_check("b.4 constructor inputs PRESERVED: Stan_model_file_path", model_b$Stan_model_file_path == "/mock/model.stan")
fn_check("b.5 constructor inputs PRESERVED: Stan_data_list NULL", is.null(model_b$Stan_data_list))
##
## $sample() with data supplied now -> deferred initialise happens inside sample():
##
model_b$sample( Stan_data_list = list(N = 10),
                n_chains_burnin = 2,
                init_lists_per_chain = list(list(mu = 0), list(mu = 0)),
                parallel_method = "RcppParallel",
                seed = 1,
                n_burnin = 10,
                n_adapt = 8,
                gap = 4,
                n_chains_sampling = 2,
                n_superchains = 1,
                n_iter = 5,
                adapt_delta = 0.8,
                learning_rate = 0.1,
                tau_mult = 1.6,
                tau_initial = 6.28,
                manual_tau = FALSE,
                tau_if_manual = 3.0,
                partitioned_HMC = TRUE,
                diffusion_HMC = TRUE,
                clip_iter = 2,
                clip_iter_tau = 2,
                use_proposed = TRUE,
                force_autodiff = FALSE,
                force_PartialLog = FALSE,
                multi_attempts = FALSE,
                vect_type = "AVX2",
                Phi_type = "Phi",
                inv_Phi_type = "inv_Phi",
                metric_type_main = "Empirical",
                metric_shape_main = "diag",
                ratio_M_main = 0.5,
                interval_width_main = 5,
                metric_type_nuisance = "Empirical",
                metric_shape_nuisance = "diag",
                ratio_M_nuisance = 0.25,
                interval_width_nuisance = 5,
                max_eps_main = 0.75,
                max_eps_nuisance = 0.75,
                max_L = 16,
                use_disk = TRUE,
                n_threads_WCP_burnin = 1,
                n_threads_WCP_sampling = 1,
                reorder_cols_MVP = FALSE)
##
fn_check("b.6 deferred initialise_model called inside $sample()", length(mock_env$initialise_model_calls) == 2L)
fn_check("b.7 is_compiled TRUE after sampling", isTRUE(model_b$is_compiled))
fn_check("b.8 init_object refreshed from result$init_object", model_b$n_nuisance == 5L)
fn_check("b.9 n_params_main refreshed", model_b$n_params_main == 3L)
fn_check("b.10 use_disk recorded on self", isTRUE(model_b$use_disk))
##
## ---- (c) argument FORWARDING to R_fn_sample_model ----------------------------------------------------------------------------------------------------------------
##
cat("\n(c) forwarding of sampler/storage/threading/metric args:\n")
##
last_call <- mock_env$sample_model_calls[[length(mock_env$sample_model_calls)]]
##
fn_check("c.1 M_decay_type forwarded (NULL)", is.null(last_call$M_decay_type))
fn_check("c.2 M_decay_power forwarded (NULL)", is.null(last_call$M_decay_power))
fn_check("c.3 use_disk forwarded (TRUE)", isTRUE(last_call$use_disk))
fn_check("c.4 n_threads_WCP_burnin forwarded (1)", last_call$n_threads_WCP_burnin == 1)
fn_check("c.5 n_threads_WCP_sampling forwarded (1)", last_call$n_threads_WCP_sampling == 1)
fn_check("c.6 reorder_cols_MVP forwarded (FALSE)", isFALSE(last_call$reorder_cols_MVP))
fn_check("c.7 sample_nuisance forwarded", isTRUE(last_call$sample_nuisance))
fn_check("c.8 init_object passed to sampler", !is.null(last_call$init_object))
##
## ---- (d) summary storage-mode consistency ------------------------------------------------------------------------------------------------------------------------
##
cat("\n(d) $summary() uses the sampling storage mode:\n")
##
plot_obj <- model_b$summary()
##
last_summary_call <- mock_env$summary_calls[[length(mock_env$summary_calls)]]
##
fn_check("d.1 summary use_disk matches sampling (TRUE)", isTRUE(last_summary_call$use_disk))
fn_check("d.2 summary model_results is self$result", identical(last_summary_call$model_results, model_b$result))
fn_check("d.3 plot object receives refreshed n_nuisance (5)", plot_obj$n_nuisance == 5L)
##
## ---- (e) default storage mode (no use_disk given) ----------------------------------------------------------------------------------------------------------------
##
cat("\n(e) default storage mode when use_disk is not set:\n")
##
model_e <- BayesMVP::MVP_model$new( Model_type = "Stan",
                                    sample_nuisance = TRUE,
                                    Stan_data_list = list(N = 10),
                                    Stan_model_file_path = "/mock/model.stan")
##
model_e$sample( Stan_data_list = list(N = 10),
                n_chains_burnin = 2,
                init_lists_per_chain = list(list(mu = 0), list(mu = 0)),
                parallel_method = "RcppParallel",
                seed = 1,
                n_burnin = 10,
                n_adapt = 8,
                gap = 4,
                n_chains_sampling = 2,
                n_superchains = 1,
                n_iter = 5,
                adapt_delta = 0.8,
                learning_rate = 0.1,
                tau_mult = 1.6,
                tau_initial = 6.28,
                manual_tau = FALSE,
                tau_if_manual = 3.0,
                partitioned_HMC = TRUE,
                diffusion_HMC = TRUE,
                clip_iter = 2,
                clip_iter_tau = 2,
                use_proposed = TRUE,
                force_autodiff = FALSE,
                force_PartialLog = FALSE,
                multi_attempts = FALSE,
                vect_type = "AVX2",
                Phi_type = "Phi",
                inv_Phi_type = "inv_Phi",
                metric_type_main = "Empirical",
                metric_shape_main = "diag",
                ratio_M_main = 0.5,
                interval_width_main = 5,
                metric_type_nuisance = "Empirical",
                metric_shape_nuisance = "diag",
                ratio_M_nuisance = 0.25,
                interval_width_nuisance = 5,
                max_eps_main = 0.75,
                max_eps_nuisance = 0.75,
                max_L = 16,
                n_threads_WCP_burnin = 1,
                n_threads_WCP_sampling = 1,
                reorder_cols_MVP = FALSE)
##
plot_e <- model_e$summary()
##
last_summary_call_e <- mock_env$summary_calls[[length(mock_env$summary_calls)]]
##
fn_check("e.1 default use_disk resolved to FALSE", isFALSE(last_summary_call_e$use_disk))
##
## ---- save + summary --------------------------------------------------------------------------------------------------------------------------------------------
##
cat(sprintf("\n==== test_02: %d passed, %d failed ====\n", n_pass, n_fail))
saveRDS(results, file = file.path(OUT_DIR, "test_02_results.RDS"))
if (n_fail > 0) quit(status = 1)
