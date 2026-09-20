#### ===================================================================================================================================
## R_fn_APMS_full_fit_benchmark.R - full APMS fits through CmdStanR or BayesMVP, using identical input data
## =====================================================================================================================================
##
## Source only: no compilation or sampling occurs until fn_run_APMS_full_fit_benchmark() is called.
## Each fit runs in a fresh R worker. Results contain plain R objects, never live BridgeStan pointers.
## Settings, input fingerprints and source fingerprints are saved with every completed or failed fit.
##
fn_APMS_benchmark_fingerprint <-  function(object) {

        temporary_file <-  tempfile(pattern = "apms_fingerprint_", fileext = ".rds")
        on.exit(expr = unlink(x = temporary_file))
        saveRDS(object = object, file = temporary_file, compress = FALSE, version = 2)
        unname(obj = tools::md5sum(files = temporary_file))

}
##
## ---- Reuse the bridge's actual data preparation, stopping at its explicit boundary --------------------------------------------------
##
fn_prepare_APMS_full_fit_benchmark <-  function( bridge_file,
                                                 model_configs = NULL,
                                                 N_subset = NULL,
                                                 subset_seed = 123
) {

        if (!is.null(x = N_subset)) {
            if (!is.numeric(x = N_subset) || length(x = N_subset) != 1 || !is.finite(x = N_subset) ||
                N_subset < 1 || N_subset != floor(x = N_subset) || N_subset > .Machine$integer.max) {
                stop("N_subset must be NULL (full data) or one positive integer.")
            }
            ##
            if (!is.numeric(x = subset_seed) || length(x = subset_seed) != 1 || !is.finite(x = subset_seed) ||
                subset_seed < 0 || subset_seed != floor(x = subset_seed) || subset_seed > .Machine$integer.max) {
                stop("subset_seed must be one non-negative integer no larger than .Machine$integer.max.")
            }
        }
        ##

        preparation_environment <-  new.env(parent = globalenv())
        bridge_expressions <-  parse(file = bridge_file)
        boundary_indices <-  which(vapply(X = as.list(bridge_expressions), FUN = function(expression) {
            is.call(expression) && identical(expression[[1]], as.name("<-")) &&
                identical(expression[[2]], as.name("APMS_PREPARATION_COMPLETE"))
        }, FUN.VALUE = FALSE))
        if (length(boundary_indices) != 1) stop("Bridge must contain exactly one APMS_PREPARATION_COMPLETE boundary; nothing evaluated.")
        bridge_expressions <-  bridge_expressions[seq_len(boundary_indices)]
        preparation_finished <-  FALSE
        preparation_start <-  proc.time()[["elapsed"]]
        for (bridge_expression in bridge_expressions) {
            eval(expr = bridge_expression, envir = preparation_environment)
            if (is.call(x = bridge_expression) && identical(x = bridge_expression[[1]], y = as.name(x = "<-")) &&
                identical(x = bridge_expression[[2]], y = as.name(x = "APMS_PREPARATION_COMPLETE"))) {
                preparation_finished <-  TRUE
                break
            }
        }
        if (!preparation_finished) stop("Bridge lacks APMS_PREPARATION_COMPLETE; use the updated BayesMVP bridge.")
        ##
        N_full <-  preparation_environment$stan_data_base$N
        ##
        ## Select ONCE from the ordinal base, before building any model-specific reference views.
        ## The existing helper carries covariates/weights with the rows and rebuilds standardisation.
        ## Keep GQ and sampler settings unchanged; this is not the reduced-output smoke-test path.
        ##
        if (!is.null(x = N_subset)) {
            if (N_subset > N_full) stop("N_subset = ", N_subset, " exceeds the prepared data size N = ", N_full, ".")
            ##
            rng_state_before_subsetting <-  if (exists(x = ".Random.seed", envir = globalenv(), inherits = FALSE)) {
                get(x = ".Random.seed", envir = globalenv(), inherits = FALSE)
            } else NULL
            ##
            on.exit(expr = {
                if (is.null(x = rng_state_before_subsetting)) {
                    if (exists(x = ".Random.seed", envir = globalenv(), inherits = FALSE)) rm(list = ".Random.seed", envir = globalenv())
                } else {
                    assign(x = ".Random.seed", value = rng_state_before_subsetting, envir = globalenv())
                }
            }, add = TRUE)
            ##
            preparation_environment$stan_data_base <-  preparation_environment$fn_subset_stan_data(
                stan_data = preparation_environment$stan_data_base,
                n_total = N_subset,
                seed = subset_seed,
                oversample_verified = 1,
                min_per_cell = 0,
                verbose = TRUE)
        }
        ##
        cat(sprintf(fmt = "\nAPMS benchmark data:\n  N = %d of %d\n  %s\n  shared across all models and engines\n",
                    preparation_environment$stan_data_base$N, N_full,
                    if (is.null(x = N_subset)) "full data" else paste0("stratified subset, seed = ", subset_seed)))
        ##
        if (is.null(model_configs)) model_configs <-  preparation_environment$MODELS_TO_FIT
        stan_data_by_model <-  setNames(object = lapply(X = model_configs, FUN = function(model_i) {
            preparation_environment$fn_build_stan_data_real(base = preparation_environment$stan_data_base, model_config = model_i)
        }), nm = model_configs)
        ## Keep the legacy builder bound to its preparation environment (including its to3d helper).
        list(stan_data_base = preparation_environment$stan_data_base,
             stan_data_by_model = stan_data_by_model,
             sampler_settings = preparation_environment$sampler_settings,
             model_configs = preparation_environment$MODELS_TO_FIT,
             base_dir = preparation_environment$BASE_DIR,
             N_full = N_full,
             N_subset = N_subset,
             subset_seed = if (is.null(x = N_subset)) NULL else subset_seed,
             preparation_seconds = proc.time()[["elapsed"]] - preparation_start)

}
##
## ---- Check layouts before launching any fits; never change requested chain/thread counts -------------------------------------------
##
fn_validate_APMS_benchmark_configuration <-  function(configuration, 
                                                      thread_budget) {

        fn_validate_named_options_APMS(options = configuration, label = "configuration")
        fn_validate_named_options_APMS(options = configuration$sampler_settings, label = "configuration$sampler_settings")
        configuration_fields <-  c("engine", "stan_file", "user_header", "sampler_settings", "num_chunks", "cpp_options",
                                   "stanc_options", "cpp_flags", "stanc_args", "make_args", "source_dependencies")
        unknown_fields <-  setdiff(x = names(configuration), y = configuration_fields)
        if (length(unknown_fields)) stop("Unknown configuration fields: ", paste(unknown_fields, collapse = ", "))
        if (length(configuration$engine) != 1 || !configuration$engine %in% c("BayesMVP", "cmdstanr")) stop("engine must be BayesMVP or cmdstanr.")
        if (!file.exists(configuration$stan_file)) stop("Stan file missing: ", configuration$stan_file)
        if (!is.null(configuration$user_header) && !file.exists(configuration$user_header)) stop("User header missing.")
        settings <-  configuration$sampler_settings
        if (configuration$engine == "BayesMVP") {
            fn_validate_settings_APMS_BayesMVP(settings = settings)
            supported_settings <-  c(names(formals(BayesMVP::MVP_model$public_methods$sample)),
                                      "Stan_model_file_path", "Stan_cpp_user_header", "Stan_cpp_flags", "stanc_args", "make_args",
                                      "summary_options", "compile", "force_recompile", "autodiff_fallback")
            unsupported_settings <-  setdiff(x = names(settings), y = supported_settings)
            if (length(unsupported_settings)) stop("Installed BayesMVP R6 interface does not accept: ",
                                                   paste(unsupported_settings, collapse = ", "),
                                                   ". Install the updated inner package before running these options.")
            counts <-  unlist(x = settings[c("n_chains_burnin", "n_chains_sampling", "n_threads_WCP_burnin", "n_threads_WCP_sampling")])
            if (length(counts) != 4) stop("Supply both BayesMVP chain counts and both WCP thread counts.")
            total_threads <-  max(settings$n_chains_burnin * settings$n_threads_WCP_burnin,
                                  settings$n_chains_sampling * settings$n_threads_WCP_sampling)
        } else {
            required <-  c("chains", "parallel_chains", "threads_per_chain", "iter_warmup", "iter_sampling", "adapt_delta", "max_treedepth")
            if (!all(required %in% names(settings))) stop("CmdStanR settings must include: ", paste(required, collapse = ", "))
            counts <-  unlist(x = settings[c("chains", "parallel_chains", "threads_per_chain")])
            if (settings$parallel_chains > settings$chains) stop("parallel_chains cannot exceed chains.")
            total_threads <-  settings$parallel_chains * settings$threads_per_chain
            if (!is.null(configuration$num_chunks) &&
                (length(configuration$num_chunks) != 1 || !is.finite(configuration$num_chunks) ||
                 configuration$num_chunks < 1 || configuration$num_chunks != floor(configuration$num_chunks))) stop("num_chunks must be a positive integer or NULL.")
        }
        if (any(!is.finite(counts) | counts < 1 | counts != floor(counts))) stop("Chain/thread counts must be positive integers.")
        detected_threads <-  parallel::detectCores(logical = TRUE)
        available_threads <-  if (is.na(detected_threads)) thread_budget else min(thread_budget, detected_threads)
        if (total_threads > available_threads) stop("Requested layout needs ", total_threads, " threads; budget/detected limit is ",
                                                     available_threads, ". Edit the explicit settings; no counts were capped.")
        invisible(x = total_threads)

}
##
## ---- Scalar diagnostics and readable per-fit reporting -----------------------------------------------------------------------------
##
fn_APMS_benchmark_numeric_scalar <-  function(value) {
        if (!is.numeric(x = value) || length(x = value) != 1 || !is.finite(x = value)) return(NA_real_)
        as.numeric(x = value)
}
##
fn_APMS_benchmark_safe_ratio <-  function(numerator, denominator) {
        numerator <-  fn_APMS_benchmark_numeric_scalar(value = numerator)
        denominator <-  fn_APMS_benchmark_numeric_scalar(value = denominator)
        if (!is.finite(numerator) || !is.finite(denominator) || denominator <= 0) return(NA_real_)
        numerator / denominator
}
##
## ---- Target-time estimate: fixed burn-in plus sampling AND summary costs scaled to the target ESS ------------------------------------
##
fn_APMS_benchmark_time_to_target_ESS <-  function( burnin_sec,
                                                   sampling_sec,
                                                   engine_summary_sec,
                                                   comparison_summary_sec,
                                                   target_ESS,
                                                   observed_ESS
) {

        timing_values <-  vapply(X = list(burnin_sec, sampling_sec, engine_summary_sec, comparison_summary_sec),
                                  FUN = fn_APMS_benchmark_numeric_scalar, FUN.VALUE = 0)
        scaling_factor <-  fn_APMS_benchmark_safe_ratio(numerator = target_ESS, denominator = observed_ESS)
        if (any(!is.finite(x = timing_values)) || any(timing_values < 0) || !is.finite(x = scaling_factor) || scaling_factor <= 0) {
            return(NA_real_)
        }
        return(timing_values[1] + sum(timing_values[2:4]) * scaling_factor)

}
##
## ---- Keep availability separate for bulk ESS, tail ESS and R-hat; never hide valid metrics behind another metric's NA -----------------
##
fn_APMS_benchmark_comparison_metrics <-  function(comparison_summary) {

        active_summary <-  dplyr::filter(.data = tibble::as_tibble(x = comparison_summary),
                                         is.finite(x = .data$sd), .data$sd > 0)
        n_monitored_active <-  nrow(x = active_summary)
        n_nonfinite_sd <-  sum(!is.finite(x = comparison_summary$sd))
        ##
        missing_bulk_ESS <-  !is.finite(x = active_summary$ess_bulk)
        missing_tail_ESS <-  !is.finite(x = active_summary$ess_tail)
        missing_Rhat <-  !is.finite(x = active_summary$rhat)
        monitored_set_complete <-  n_monitored_active > 0 && n_nonfinite_sd == 0
        ##
        ## A missing diagnostic stays NA for that aggregate, rather than silently dropping the affected quantity.
        ## Constant quantities remain excluded, exactly as before; a non-finite SD is not treated as a constant.
        ##
        return(tibble::tibble( min_ESS_bulk = if (monitored_set_complete && !any(missing_bulk_ESS)) min(active_summary$ess_bulk) else NA_real_,
                     min_ESS_tail = if (monitored_set_complete && !any(missing_tail_ESS)) min(active_summary$ess_tail) else NA_real_,
                     max_rhat = if (monitored_set_complete && !any(missing_Rhat)) max(active_summary$rhat) else NA_real_,
                     diagnostics_complete = monitored_set_complete && !any(missing_bulk_ESS | missing_tail_ESS | missing_Rhat),
                     n_monitored_active = n_monitored_active,
                     n_monitored_constant = sum(is.finite(x = comparison_summary$sd) & comparison_summary$sd == 0),
                     n_nonfinite_sd = n_nonfinite_sd,
                     n_missing_bulk_ESS = sum(missing_bulk_ESS),
                     n_missing_tail_ESS = sum(missing_tail_ESS),
                     n_missing_Rhat = sum(missing_Rhat),
                     undefined_bulk_ESS_variables = paste(active_summary$variable[missing_bulk_ESS], collapse = ", "),
                     undefined_tail_ESS_variables = paste(active_summary$variable[missing_tail_ESS], collapse = ", "),
                     undefined_Rhat_variables = paste(active_summary$variable[missing_Rhat], collapse = ", "),
                     nonfinite_sd_variables = paste(comparison_summary$variable[!is.finite(x = comparison_summary$sd)], collapse = ", ")))

}
##
## ---- Refresh reporting from saved per-variable summaries only; no model fits and no writes to original result files ------------------
##
fn_refresh_APMS_full_fit_benchmark_result <-  function(result) {

        ## Iteration counts are per chain, as configured, including for failed fits.
        ## Recover older rows from THEIR saved settings, never the current driver's settings.
        saved_settings_sources <-  list(result$settings, result$configuration$sampler_settings,
                                         result$definition$configuration$sampler_settings)
        iteration_setting_names <-  if (identical(x = result$row$engine, y = "BayesMVP")) {
            c(n_burnin = "n_burnin", n_iter = "n_iter")
        } else if (identical(x = result$row$engine, y = "cmdstanr")) {
            c(n_burnin = "iter_warmup", n_iter = "iter_sampling")
        } else c(n_burnin = NA_character_, n_iter = NA_character_)
        ##
        result$row <-  tibble::as_tibble(x = result$row)
        for (iteration_column in names(x = iteration_setting_names)) {
            iteration_count <-  fn_APMS_benchmark_numeric_scalar(value = result$row[[iteration_column]])
            if (!is.finite(x = iteration_count)) {
                for (saved_settings in saved_settings_sources) {
                    if (is.na(x = iteration_setting_names[[iteration_column]])) next
                    iteration_count <-  fn_APMS_benchmark_numeric_scalar(value = saved_settings[[iteration_setting_names[[iteration_column]]]])
                    if (is.finite(x = iteration_count)) break
                }
            }
            result$row <-  dplyr::mutate(.data = result$row, !!iteration_column := iteration_count)
        }
        ##
        if (!identical(x = result$row$status, y = "completed") || is.null(x = result$comparison_summary)) return(result)
        ##
        comparison_metrics <-  fn_APMS_benchmark_comparison_metrics(comparison_summary = result$comparison_summary)
        row <-  dplyr::mutate(.data = tibble::as_tibble(x = result$row), !!!comparison_metrics)
        ##
        row <-  dplyr::mutate(.data = row,
            ESS_per_sec_sampling = fn_APMS_benchmark_safe_ratio(numerator = .data$min_ESS_bulk,
                                                                denominator = row[["native_sampling_seconds"]]),
            ESS_per_sec_full_fit = fn_APMS_benchmark_safe_ratio(numerator = .data$min_ESS_bulk,
                                                                denominator = row[["full_fit_seconds"]]),
            ESS_per_sec_excluding_setup = fn_APMS_benchmark_safe_ratio(numerator = .data$min_ESS_bulk,
                                                                       denominator = row[["fit_excluding_setup_seconds"]]),
            ESS_per_grad_sampling = fn_APMS_benchmark_safe_ratio(numerator = .data$min_ESS_bulk,
                                                                 denominator = row[["n_grad_sampling"]]),
            ESS_per_1000_grad_sampling = 1000 * .data$ESS_per_grad_sampling,
        ## Burn-in is paid once; scale sampling and BOTH non-overlapping summary stages by target / observed ESS.
            estimated_time_to_target_ESS = fn_APMS_benchmark_time_to_target_ESS(
                burnin_sec = row[["native_burnin_seconds"]],
                sampling_sec = row[["native_sampling_seconds"]],
                engine_summary_sec = row[["engine_summary_seconds"]],
                comparison_summary_sec = row[["comparison_seconds"]],
                target_ESS = row[["target_min_ESS"]], observed_ESS = .data$min_ESS_bulk),
        ## Reporting valid bulk ESS does not relax the original diagnostic gate or excuse divergences.
            diagnostics_pass = isTRUE(x = row[["diagnostics_pass"]]) && .data$diagnostics_complete)
        result$row <-  row
        return(result)

}
##
fn_print_APMS_full_fit_benchmark_result <-  function(result) {
        result <-  fn_refresh_APMS_full_fit_benchmark_result(result = result)
        row <-  result$row
        cat(sprintf(fmt = "\n==== FULL FIT ====\nConfiguration: %s\nModel: %s\nSeed: %s\nStatus: %s\n",
                    row$configuration, row$model, row$seed, toupper(x = row$status)))
        cat(sprintf(fmt = "Configured iterations per chain:\n  burn-in (n_burnin) = %.0f\n  sampling (n_iter) = %.0f\n",
                    row$n_burnin, row$n_iter))
        if (!identical(row$status, "completed")) {
            cat("Error: ", row$error, "\n", sep = "")
            cat("Elapsed job time: ", round(x = row$job_wall_seconds, digits = 1), " s\n", sep = "")
            return(invisible(x = row))
        }
        ## Handle older saved rows too, so resume can print their available diagnostics.
        number <-  function(name) fn_APMS_benchmark_numeric_scalar(value = row[[name]])
        cat(sprintf(fmt = "\nSetup/compile = %.1f s\nburn-in = %.1f s\nsampling = %.1f s\nengine summaries = %.1f s\n",
                    number("setup_seconds"), number("native_burnin_seconds"), number("native_sampling_seconds"), number("engine_summary_seconds")))
        cat(sprintf(fmt = "Full fit = %.1f s\nexcluding setup = %.1f s\njob wall = %.1f s\n",
                    number("full_fit_seconds"), number("fit_excluding_setup_seconds"), number("job_wall_seconds")))
        cat("\nParameters of interest: Se / Sp / prevalence (or your monitor_pattern)\n")
        cat(sprintf(fmt = "min_ESS (bulk) = %.0f\nmin_ESS (tail) = %.0f\nmax R-hat = %.4f\n",
                    number("min_ESS_bulk"), number("min_ESS_tail"), number("max_rhat")))
        if ("n_monitored_active" %in% names(x = row)) {
            cat(sprintf(fmt = "\nDiagnostics available (nonconstant quantities):\n  bulk ESS %.0f/%.0f\n  tail ESS %.0f/%.0f\n  R-hat %.0f/%.0f\n",
                        number("n_monitored_active") - number("n_missing_bulk_ESS"), number("n_monitored_active"),
                        number("n_monitored_active") - number("n_missing_tail_ESS"), number("n_monitored_active"),
                        number("n_monitored_active") - number("n_missing_Rhat"), number("n_monitored_active")))
            for (missing_diagnostic in c("bulk_ESS", "tail_ESS", "Rhat")) {
                affected_variables <-  row[[paste0("undefined_", missing_diagnostic, "_variables")]]
                if (nzchar(x = affected_variables)) cat("Undefined ", missing_diagnostic, ": ", affected_variables, "\n", sep = "")
            }
            if (number("n_nonfinite_sd") > 0) cat("Non-finite SD: ", row$nonfinite_sd_variables, "\n", sep = "")
            if (!isTRUE(x = row$diagnostics_complete)) {
                cat("Incomplete diagnostics: affected aggregates remain NA; available metrics are shown and the diagnostic gate remains failed.\n")
            }
        }
        cat(sprintf(fmt = "\nESS/sec [sampling] = %.3f\nESS/sec [full fit] = %.3f\nESS/sec [excluding setup] = %.3f\n",
                    number("ESS_per_sec_sampling"), number("ESS_per_sec_full_fit"), number("ESS_per_sec_excluding_setup")))
        cat(sprintf(fmt = "ESS/grad [estimated] = %.6f\nESS/1000 grads [estimated] = %.3f\ngrads/sec [estimated] = %.0f\n",
                    number("ESS_per_grad_sampling"), number("ESS_per_1000_grad_sampling"), number("grad_evals_per_sec_sampling")))
        cat(sprintf(fmt = "\nSampling acceptance = %.3f\nepsilon = %.4g\nmean L = %.1f\ndivergences = %.0f\ntreedepth hits = %.0f\n",
                    number("sampling_acceptance"), number("eps_sampling"), number("mean_leapfrog_steps"), number("n_divergent"), number("n_treedepth_hits")))
        if (identical(row$engine, "BayesMVP")) {
            cat(sprintf(fmt = "\nRaw main block:\n  min ESS = %.0f\n  max R-hat = %.4f\n  max nested R-hat = %.4f (NA if unavailable/disabled)\n",
                        number("min_ESS_main"), number("max_rhat_main"), number("max_nested_rhat_main")))
        }
        cat(sprintf(fmt = "\ntime_to_target_ESS (target %.0f) = %.1f sec\n",
                    number("target_min_ESS"), number("estimated_time_to_target_ESS")))
        cat("  burn-in + (sampling + engine summaries + comparison summaries) * target ESS / observed ESS\n",
            "  setup/compile is reported separately, not included in this estimate\n", sep = "")
        cat("Sampling-time basis: ", row[["sampling_seconds_basis"]], "\n", sep = "")
        cat("Gradient-work basis: ", row[["gradient_count_basis"]], "\n", sep = "")
        cat("Diagnostics pass: ", row$diagnostics_pass, " (metrics are shown regardless)\n", sep = "")
        flush.console()
        invisible(x = row)
}
##
## ---- One complete fit, in the worker ------------------------------------------------------------------------------------------------
##
fn_APMS_full_fit_benchmark_worker <-  function(job, helper_files) {

        worker_start <-  proc.time()[["elapsed"]]
        {
            require(RcppParallel)
            require(BayesMVP)
            if (identical(job$configuration$engine, "cmdstanr")) require(cmdstanr)
            require(posterior)
        }
        for (helper_file in helper_files) source(file = helper_file, local = TRUE)
        configuration <-  job$configuration
        settings <-  configuration$sampler_settings
        settings$seed <-  job$seed
        stan_data <-  job$stan_data
        cat(sprintf(fmt = "\n==== BENCHMARK SETTINGS ====\nConfiguration: %s\nModel: %s\nN = %d\nSeed = %d\n",
                    job$configuration_name, job$model_i, stan_data$N, job$seed))
        print(x = settings)
        flush.console()
        set.seed(seed = job$seed)
        dir.create(path = job$fit_dir, recursive = TRUE, showWarnings = FALSE)
        fit_start <-  proc.time()[["elapsed"]]
        ##
        if (configuration$engine == "BayesMVP") {
            settings$Stan_model_file_path <-  job$build_stan_file
            settings$Stan_cpp_user_header <-  configuration$user_header
            settings$Stan_cpp_flags <-  configuration$cpp_flags
            settings$stanc_args <-  configuration$stanc_args
            settings$make_args <-  configuration$make_args
            outs_fit <-  fn_fit_APMS_model_BayesMVP(model_i = job$model_i, stan_data = stan_data, settings = settings, verbose = TRUE)
            draws <-  posterior::as_draws_array(x = outs_fit$draws)
            setup_seconds <-  outs_fit$timing$initialise_seconds
            sample_call_seconds <-  outs_fit$timing$sample_call_seconds
            engine_summary_seconds <-  outs_fit$timing$summary_seconds
            efficiency <-  outs_fit$model_fit$summary_object$summaries$efficiency_info
            time_burnin_sec <-  as.numeric(x = efficiency$time_burnin)
            time_sampling_sec <-  as.numeric(x = efficiency$time_sampling)
            n_divergent <-  as.numeric(x = outs_fit$divergences$n_divs)
            if (length(n_divergent) != 1) stop("BayesMVP did not return divergences$n_divs.")
            n_treedepth_hits <-  NA_real_
            engine_summary <-  fn_summary_table_from_BayesMVP(outs_fit = outs_fit)
            native_timing <-  outs_fit$timing
            csv_files <-  character()
            sampling_diagnostics <-  outs_fit$model_samples$result$sampling_object$sampling_diagnostics
            sampling_acceptance <-  if (is.numeric(x = sampling_diagnostics$p_jump_main) && length(x = sampling_diagnostics$p_jump_main)) {
                fn_APMS_benchmark_numeric_scalar(value = mean(x = sampling_diagnostics$p_jump_main))
            } else NA_real_
            mean_leapfrog_steps <-  fn_APMS_benchmark_numeric_scalar(value = efficiency$L_main_during_sampling)
            n_grad_sampling <-  fn_APMS_benchmark_numeric_scalar(value = efficiency$grad_evals_per_sec) * time_sampling_sec
            if (!is.finite(n_grad_sampling)) n_grad_sampling <-  mean_leapfrog_steps * settings$n_iter * settings$n_chains_sampling
            gradient_count_basis <-  "BayesMVP estimated gradient work (not a measured call count)"
            sampling_seconds_basis <-  "BayesMVP sampling wall time"
            min_ESS_main <-  fn_APMS_benchmark_numeric_scalar(value = efficiency$Min_ESS_main)
            max_rhat_main <-  fn_APMS_benchmark_numeric_scalar(value = efficiency$Max_rhat_main)
            max_nested_rhat_main <-  fn_APMS_benchmark_numeric_scalar(value = efficiency$Max_nested_rhat_main)
            eps_sampling <-  fn_APMS_benchmark_numeric_scalar(value = outs_fit$model_fit$summary_object$summaries$HMC_info$eps_main)
        } else {
            setup_start <-  proc.time()[["elapsed"]]
            cpp_options <-  configuration$cpp_options
            if (is.null(cpp_options)) cpp_options <-  list()
            if (!is.null(cpp_options$stan_threads) && !isTRUE(cpp_options$stan_threads)) stop("stan_threads must be TRUE.")
            cpp_options$stan_threads <-  TRUE
            stanc_options <-  configuration$stanc_options
            if (is.null(stanc_options)) stanc_options <-  list()
            if (!is.null(configuration$user_header)) stanc_options[["allow-undefined"]] <-  TRUE
            model <-  cmdstanr::cmdstan_model(stan_file = job$build_stan_file,
                                              user_header = configuration$user_header,
                                              cpp_options = cpp_options,
                                              stanc_options = stanc_options)
            setup_seconds <-  proc.time()[["elapsed"]] - setup_start
            if (!is.null(configuration$num_chunks)) stan_data$grainsize <-  ceiling(stan_data$N / configuration$num_chunks)
            reserved <-  intersect(x = names(settings), y = c("data", "init", "output_dir"))
            if (length(reserved)) stop("Data, init and output_dir are managed by the benchmark: ", paste(reserved, collapse = ", "))
            unsupported <-  setdiff(x = names(settings), y = names(formals(model$sample)))
            if (length(unsupported)) stop("Unsupported CmdStanR arguments: ", paste(unsupported, collapse = ", "))
            sample_arguments <-  settings
            sample_arguments$data <-  fn_stan_input(stan_data = stan_data)
            sample_arguments$init <-  fn_make_inits_BayesMVP(stan_data = stan_data, n_chains_burnin = settings$chains, seed = job$seed)
            sample_arguments$output_dir <-  job$fit_dir
            sample_start <-  proc.time()[["elapsed"]]
            fit <-  do.call(what = model$sample, args = sample_arguments)
            sample_call_seconds <-  proc.time()[["elapsed"]] - sample_start
            summary_start <-  proc.time()[["elapsed"]]
            ## Match the application bridge: structural and application quantities, with latent u_raw omitted from summaries.
            variable_names <-  fit$metadata()$model_params
            variable_names <-  variable_names[!grepl(pattern = "^(u_raw|lp__)(\\[|$)", x = variable_names)]
            draws <-  fit$draws(variables = variable_names, format = "draws_array")
            engine_summary <-  posterior::summarise_draws(.x = draws)
            engine_summary_seconds <-  proc.time()[["elapsed"]] - summary_start
            native_timing <-  fit$time()
            ## Native maxima are diagnostics, not substitutes for the measured full-fit wall time (queued chains can differ).
            time_burnin_sec <-  max(native_timing$chains$warmup)
            time_sampling_sec <-  max(native_timing$chains$sampling)
            diagnostic_summary <-  fit$diagnostic_summary()
            n_divergent <-  sum(diagnostic_summary$num_divergent)
            n_treedepth_hits <-  sum(diagnostic_summary$num_max_treedepth)
            csv_files <-  fit$output_files()
            sampling_diagnostics <-  fit$sampler_diagnostics(format = "draws_matrix")
            n_grad_sampling <-  sum(sampling_diagnostics[, "n_leapfrog__"])
            mean_leapfrog_steps <-  mean(x = sampling_diagnostics[, "n_leapfrog__"])
            sampling_acceptance <-  mean(x = sampling_diagnostics[, "accept_stat__"])
            eps_sampling <-  mean(x = sampling_diagnostics[, "stepsize__"])
            gradient_count_basis <-  "CmdStan n_leapfrog__ sum (gradient-work proxy)"
            sampling_seconds_basis <-  "maximum CmdStan per-chain sampling time; excludes chain queueing"
            min_ESS_main <-  max_rhat_main <-  max_nested_rhat_main <-  NA_real_
        }
        ##
        ## Recompute identical rank-normalised R-hat, bulk ESS and tail ESS for BOTH engines.
        comparison_start <-  proc.time()[["elapsed"]]
        available_variables <-  posterior::variables(x = draws)
        if (is.null(available_variables)) stop("Draw arrays must have Stan variable names.")
        monitored_variables <-  grep(pattern = job$monitor_pattern, x = available_variables, value = TRUE)
        if (!length(monitored_variables)) stop("No variables matched monitor_pattern.")
        monitored_draws <-  posterior::subset_draws(x = draws, variable = monitored_variables)
        comparison_summary <-  posterior::summarise_draws(.x = monitored_draws)
        comparison_metrics <-  fn_APMS_benchmark_comparison_metrics(comparison_summary = comparison_summary)
        if (!comparison_metrics$n_monitored_active) stop("All monitored quantities are constant or non-finite.")
        diagnostics_complete <-  comparison_metrics$diagnostics_complete
        min_ESS_bulk <-  comparison_metrics$min_ESS_bulk
        min_ESS_tail <-  comparison_metrics$min_ESS_tail
        max_rhat <-  comparison_metrics$max_rhat
        diagnostics_pass <-  diagnostics_complete && min_ESS_bulk >= job$min_ESS && max_rhat <= job$max_rhat && n_divergent == 0
        comparison_seconds <-  proc.time()[["elapsed"]] - comparison_start
        full_fit_seconds <-  proc.time()[["elapsed"]] - fit_start
        ##
        row <-  tibble::tibble(configuration = job$configuration_name, engine = .env$configuration$engine, model = job$model_i,
                           N = stan_data$N, seed = job$seed, input_fingerprint = job$input_fingerprint,
                           n_burnin = if (.env$configuration$engine == "BayesMVP") settings$n_burnin else settings$iter_warmup,
                           n_iter = if (.env$configuration$engine == "BayesMVP") settings$n_iter else settings$iter_sampling,
                           n_chains_burnin = if (.env$configuration$engine == "BayesMVP") settings$n_chains_burnin else settings$chains,
                           n_chains_sampling = if (.env$configuration$engine == "BayesMVP") settings$n_chains_sampling else settings$chains,
                           threads_per_chain_burnin = if (.env$configuration$engine == "BayesMVP") settings$n_threads_WCP_burnin else settings$threads_per_chain,
                           threads_per_chain_sampling = if (.env$configuration$engine == "BayesMVP") settings$n_threads_WCP_sampling else settings$threads_per_chain,
                           signature = job$signature, full_fit_seconds = full_fit_seconds,
                           setup_seconds = setup_seconds, sample_call_seconds = sample_call_seconds,
                           engine_summary_seconds = engine_summary_seconds, comparison_seconds = comparison_seconds,
                           fit_excluding_setup_seconds = full_fit_seconds - setup_seconds,
                           native_burnin_seconds = time_burnin_sec, native_sampling_seconds = time_sampling_sec,
                           min_ESS_bulk = min_ESS_bulk, min_ESS_tail = min_ESS_tail, max_rhat = max_rhat,
                           min_ESS_main = min_ESS_main, max_rhat_main = max_rhat_main, max_nested_rhat_main = max_nested_rhat_main,
                           sampling_acceptance = sampling_acceptance, eps_sampling = eps_sampling, mean_leapfrog_steps = mean_leapfrog_steps,
                           n_grad_sampling = n_grad_sampling, gradient_count_basis = gradient_count_basis,
                           sampling_seconds_basis = sampling_seconds_basis,
                           ESS_per_sec_sampling = fn_APMS_benchmark_safe_ratio(numerator = min_ESS_bulk, denominator = time_sampling_sec),
                           ESS_per_sec_excluding_setup = fn_APMS_benchmark_safe_ratio(numerator = min_ESS_bulk, denominator = full_fit_seconds - setup_seconds),
                           ESS_per_grad_sampling = fn_APMS_benchmark_safe_ratio(numerator = min_ESS_bulk, denominator = n_grad_sampling),
                           ESS_per_1000_grad_sampling = 1000 * fn_APMS_benchmark_safe_ratio(numerator = min_ESS_bulk, denominator = n_grad_sampling),
                           grad_evals_per_sec_sampling = fn_APMS_benchmark_safe_ratio(numerator = n_grad_sampling, denominator = time_sampling_sec),
                           target_min_ESS = job$min_ESS,
                           estimated_time_to_target_ESS = fn_APMS_benchmark_time_to_target_ESS(
                               burnin_sec = time_burnin_sec, sampling_sec = time_sampling_sec,
                               engine_summary_sec = engine_summary_seconds, comparison_summary_sec = comparison_seconds,
                               target_ESS = job$min_ESS, observed_ESS = min_ESS_bulk),
                           n_divergent = n_divergent, n_treedepth_hits = n_treedepth_hits,
                           diagnostics_pass = diagnostics_pass, ESS_per_sec_full_fit = min_ESS_bulk / full_fit_seconds,
                           binary_existed_before = job$binary_existed_before, status = "completed", error = "")
        row <-  dplyr::mutate(.data = row, !!!comparison_metrics)
        ##
        list(row = row, configuration = configuration, stan_data = stan_data, settings = settings,
             comparison_summary = comparison_summary, engine_summary = engine_summary,
             draws = if (job$save_draws) draws else NULL, native_timing = native_timing, csv_files = csv_files,
             engine_efficiency = if (configuration$engine == "BayesMVP") efficiency else NULL,
             monitored_variables = monitored_variables, session = utils::sessionInfo(),
             worker_seconds = proc.time()[["elapsed"]] - worker_start)

}
##
## ---- Driver: serial configurations, fresh worker for each fit, immediate checkpoints ------------------------------------------------
##
fn_run_APMS_full_fit_benchmark <-  function( stan_data_by_model,
                                             configurations,
                                             repetition_seeds,
                                             output_dir,
                                             helper_files,
                                             thread_budget,
                                             monitor_pattern = "^(Se_bin|Sp_bin|Se_ord|Sp_ord|Se_anchor_ord|Sp_anchor_ord)\\[|^(prev_A_overall|prev_B_overall|pi_00_overall|pi_10_overall|pi_01_overall|pi_11_overall|prev_A_par|prev_B_par)(\\[|$)",
                                             min_ESS = 100,
                                             max_rhat = 1.10,
                                             save_draws = TRUE,
                                             resume = TRUE,
                                             stop_on_error = FALSE) {

        fn_validate_named_options_APMS(options = configurations, label = "configurations")
        fn_validate_named_options_APMS(options = stan_data_by_model, label = "stan_data_by_model")
        if (!length(configurations) || !length(stan_data_by_model)) stop("Supply at least one model and one configuration.")
        if (!is.numeric(repetition_seeds) || !length(repetition_seeds) || any(!is.finite(repetition_seeds) |
            repetition_seeds < 1 | repetition_seeds != floor(repetition_seeds)) || anyDuplicated(repetition_seeds)) stop("Supply unique positive integer repetition_seeds.")
        if (length(thread_budget) != 1 || !is.finite(thread_budget) || thread_budget < 1) stop("Supply a positive thread_budget.")
        if (any(!file.exists(helper_files))) stop("A helper file is missing.")
        if (any(!grepl(pattern = "^[A-Za-z0-9_-]+$", x = c(names(configurations), names(stan_data_by_model))))) stop("Use letters, numbers, underscores or hyphens in configuration/model names.")
        for (configuration in configurations) fn_validate_APMS_benchmark_configuration(configuration = configuration, thread_budget = thread_budget)
        dir.create(path = output_dir, recursive = TRUE, showWarnings = FALSE)
        source_fingerprints <-  tools::md5sum(files = helper_files)
        ##
        for (repetition_index in seq_along(repetition_seeds)) {
          
            ## Run BayesMVP before CmdStanR for each model; rotate configurations within each engine across repetitions.
            configuration_indices <-  ((seq_along(configurations) + repetition_index - 2) %% length(configurations)) + 1
            configuration_is_BayesMVP <-  vapply(X = configurations[configuration_indices],
                                                 FUN = function(configuration) identical(configuration$engine, "BayesMVP"),
                                                 FUN.VALUE = FALSE)
            configuration_indices <-  c( configuration_indices[configuration_is_BayesMVP],
                                         configuration_indices[!configuration_is_BayesMVP])
            ##
            for (model_i in names(stan_data_by_model)) {
              
                input_fingerprint <-  fn_APMS_benchmark_fingerprint(object = stan_data_by_model[[model_i]])
                ##
                for (configuration_index in configuration_indices) {
                  
                    configuration_name <-  names(configurations)[configuration_index]
                    configuration <-  configurations[[configuration_index]]
                    source_files <-  c(configuration$stan_file, configuration$user_header, configuration$source_dependencies)
                    if (any(!file.exists(source_files))) stop("A Stan/header dependency is missing for ", configuration_name)
                    toolchain_dir <-  if (configuration$engine == "cmdstanr") cmdstanr::cmdstan_path() else BayesMVP::bridgestan_path()
                    toolchain_files <-  c(file.path(toolchain_dir, "make", "local"), file.path(toolchain_dir, "bin", "stanc"),
                                          file.path(toolchain_dir, "Makefile"), file.path(path.expand("~"), ".R", "Makevars"))
                    toolchain_files <-  toolchain_files[file.exists(toolchain_files)]
                    build_inputs <-  list(toolchain_dir = toolchain_dir,
                                          toolchain_fingerprints = tools::md5sum(files = toolchain_files),
                                          configuration = configuration[setdiff(names(configuration), c("sampler_settings", "num_chunks"))],
                                          sources = tools::md5sum(files = source_files), machine = Sys.info()[c("nodename", "machine")])
                    build_signature <-  fn_APMS_benchmark_fingerprint(object = build_inputs)
                    build_dir <-  file.path(output_dir, "builds", configuration$engine, build_signature)
                    dir.create(path = build_dir, recursive = TRUE, showWarnings = FALSE)
                    build_stan_file <-  file.path(build_dir, basename(configuration$stan_file))
                    if (!file.exists(build_stan_file)) {
                        if (!file.copy(from = configuration$stan_file, to = build_stan_file)) stop("Could not copy Stan source to benchmark build directory.")
                    }
                    ## The APMS programs have no Stan #include statements; paths to the AVX header remain absolute.
                    if (any(grepl(pattern = "^[[:space:]]*#include", x = readLines(con = build_stan_file, warn = FALSE))))
                        stop("Stan #include dependencies need an explicit build layout; benchmark the supplied self-contained APMS programs.")
                    fingerprint_packages <-  if (configuration$engine == "cmdstanr") {
                        c("BayesMVP", "cmdstanr", "bridgestan", "posterior")
                    } else c("BayesMVP", "bridgestan", "posterior")
                    installed_package_files <-  unlist(lapply(X = fingerprint_packages, FUN = function(package) {
                        package_dir <-  find.package(package = package)
                        list.files(path = package_dir, pattern = "\\.(rdb|rdx|so|dll)$", recursive = TRUE, full.names = TRUE)
                    }))
                    run_definition <-  list(configuration = configuration, model = model_i, input = input_fingerprint,
                                            seed = repetition_seeds[repetition_index], sources = source_fingerprints, build = build_inputs,
                                            monitor_pattern = monitor_pattern, min_ESS = min_ESS, max_rhat = max_rhat, save_draws = save_draws,
                                            installed_code = tools::md5sum(files = installed_package_files),
                                            package_versions = vapply(X = fingerprint_packages,
                                                                      FUN = function(package) as.character(utils::packageVersion(pkg = package)), FUN.VALUE = ""))
                    signature <-  fn_APMS_benchmark_fingerprint(object = run_definition)
                    run_label <-  paste(configuration_name, model_i, paste0("seed", repetition_seeds[repetition_index]), substr(signature, 1, 12), sep = "_")
                    result_file <-  file.path(output_dir, paste0(run_label, ".rds"))
                    if (file.exists(result_file)) {
                        previous <-  readRDS(file = result_file)
                        if (!identical(previous$signature, signature)) stop("Result fingerprint mismatch: ", result_file)
                        if (resume && identical(previous$row$status, "completed")) {
                            message("Already completed: ", run_label)
                            fn_print_APMS_full_fit_benchmark_result(result = previous)
                            next
                        }
                        if (!resume) stop("Output already exists; choose a new output_dir or resume = TRUE: ", result_file)
                    }
                    job <-  list(configuration = configuration, configuration_name = configuration_name, model_i = model_i,
                                 seed = repetition_seeds[repetition_index], stan_data = stan_data_by_model[[model_i]],
                                 build_stan_file = normalizePath(path = build_stan_file),
                                 binary_existed_before = file.exists(sub(pattern = "\\.stan$", replacement = if (configuration$engine == "BayesMVP") "_model.so" else "", x = build_stan_file)),
                                 fit_dir = file.path(output_dir, "fits", run_label), input_fingerprint = input_fingerprint,
                                 signature = signature, monitor_pattern = monitor_pattern, min_ESS = min_ESS,
                                 max_rhat = max_rhat, save_draws = save_draws,
                                 console_log_file = file.path(output_dir, paste0(run_label, ".log")))
                    ##
                    message("Starting full fit: ", run_label)
                    message("Live progress follows; R stdout is also saved to: ", job$console_log_file)
                    ##
                    job_start <-  proc.time()[["elapsed"]]
                    worker_cluster <-  NULL
                    ##
                    result <-  tryCatch(expr = {
                      
                        ## Empty outfile forwards worker stdout/stderr to the console. Also save R stdout using split = TRUE.
                        worker_cluster <-  parallel::makePSOCKcluster(names = 1, outfile = "")
                        parallel::clusterCall(cl = worker_cluster,
                                              fun = function(job, 
                                                             helper_files) {
                                                
                                worker_log_connection <-  file(description = job$console_log_file, open = "at")
                                sink(file = worker_log_connection, split = TRUE)
                                on.exit(expr = {
                                    sink()
                                    close(con = worker_log_connection)
                                })
                                options(warn = 1)
                                for (helper_file in helper_files) source(file = helper_file, local = .GlobalEnv)
                                withCallingHandlers(expr = fn_APMS_full_fit_benchmark_worker(job = job, helper_files = helper_files),
                                                     error = function(error_condition) {
                                                         cat("\nWorker error: ", conditionMessage(error_condition), "\n", sep = "")
                                                         print(x = conditionCall(error_condition))
                                                         flush.console()
                                                     })
                            
                        }, job = job, helper_files = helper_files)[[1]]
                        
                    }, error = function(error_condition) {
                      
                          list(row = tibble::tibble( configuration = configuration_name, 
                                                 engine = .env$configuration$engine,
                                                 model = model_i,
                                                 N = job$stan_data$N, 
                                                 seed = job$seed, 
                                                 input_fingerprint = input_fingerprint,
                                                 signature = signature,
                                                 status = "error", 
                                                 error = conditionMessage(error_condition)))
                      
                    }, finally = {
                        if (!is.null(worker_cluster)) tryCatch(expr = parallel::stopCluster(cl = worker_cluster), error = function(error_condition) NULL)
                    })
                    result$signature <-  signature
                    result$definition <-  run_definition
                    result <-  fn_refresh_APMS_full_fit_benchmark_result(result = result)
                    result$row <-  dplyr::mutate(.data = result$row, job_wall_seconds = proc.time()[["elapsed"]] - job_start)
                    result_file_temporary <-  paste0(result_file, ".tmp")
                    saveRDS(object = result, file = result_file_temporary)
                    if (!file.rename(from = result_file_temporary, to = result_file)) stop("Cannot checkpoint: ", result_file)
                    summary_table <-  fn_read_APMS_full_fit_benchmark(output_dir = output_dir)
                    utils::write.csv(x = summary_table, file = file.path(output_dir, "benchmark_summary.csv"), row.names = FALSE)
                    fn_print_APMS_full_fit_benchmark_result(result = result)
                    if (stop_on_error && identical(result$row$status, "error")) stop(result$row$error)
                }
            }
        }
        invisible(x = fn_read_APMS_full_fit_benchmark(output_dir = output_dir))

}
##
## ---- Read completed and failed cells, including interrupted benchmark sweeps -------------------------------------------------------
##
fn_read_APMS_full_fit_benchmark <-  function(output_dir) {

        result_files <-  list.files(path = output_dir, pattern = "_seed[0-9]+_[[:xdigit:]]{12}\\.rds$", full.names = TRUE)
        records <-  purrr::map(.x = result_files, .f = function(result_file) {
            result <-  readRDS(file = result_file)
            result <-  fn_refresh_APMS_full_fit_benchmark_result(result = result)
            ## Match repetitions on the recorded run definition, excluding only the MCMC seed.
            ## Retain source/build fingerprints: different implementations must not be averaged together.
            repetition_definition <-  result$definition
            if (!is.null(x = repetition_definition)) {
                repetition_definition$seed <-  NULL
                repetition_definition$configuration$sampler_settings$seed <-  NULL
                repetition_group <-  fn_APMS_benchmark_fingerprint(object = repetition_definition)
            } else {
                ## Without a saved definition, do not guess that two historical fits used identical settings.
                repetition_group <-  result_file
            }
            dplyr::mutate(.data = tibble::as_tibble(x = result$row),
                          repetition_group = .env$repetition_group, result_file = .env$result_file)
        })
        if (!length(records)) return(tibble::tibble())
        return(dplyr::bind_rows(records))

}

##
## ---- Summarise repetitions; keep per-run data separately and print narrow tables by model and N --------------------------------------
##
#' Return numeric summary columns, including the settings printed under each ID.
#' @param filter_values Optional named list of allowed column values, applied AFTER summarising seeds.
#'   For example list(N = 500, n_burnin = 250, n_chains_sampling = c(32, 64)). NULL retains every configuration.
#'   Use dplyr::filter() on the returned tibble for inequalities or more complex conditions.
fn_summarise_APMS_full_fit_benchmark <-  function( benchmark_results,
                                                   sort_by = "time_to_target_ESS",
                                                   print_table = TRUE,
                                                   filter_values = NULL
) {

        benchmark_results <-  tibble::as_tibble(x = benchmark_results)
        ## Keep the stored metric name and existing sorting calls compatible.
        metric_column_aliases <-  c(time_to_target_ESS = "estimated_time_to_target_ESS",
                                    burnin_sec = "native_burnin_seconds", sampling_sec = "native_sampling_seconds",
                                    min_ESS = "min_ESS_bulk", ESS_per_sec = "ESS_per_sec_sampling",
                                    ESS_per_1000_grad = "ESS_per_1000_grad_sampling",
                                    Rhat = "max_rhat", acceptance_probability = "sampling_acceptance",
                                    divergences_per_fit = "n_divergent", treedepth_hits_per_fit = "n_treedepth_hits")
        if (length(x = sort_by) == 1 && sort_by %in% names(x = metric_column_aliases)) sort_by <-  unname(obj = metric_column_aliases[sort_by])
        ##
        if (!is.null(x = filter_values) && (!is.list(x = filter_values) ||
            (length(x = filter_values) > 0 && (is.null(x = names(x = filter_values)) || anyNA(names(x = filter_values)) ||
                                             any(!nzchar(x = names(x = filter_values))) || anyDuplicated(x = names(x = filter_values)))))) {
            stop("filter_values must be NULL or a named list of allowed column values.")
        }
        if (!nrow(x = benchmark_results)) {
            if (print_table) cat("No saved APMS benchmark results to summarise.\n")
            return(tibble::tibble())
        }
        ##
        if (!"repetition_group" %in% names(x = benchmark_results)) {
            stop("Re-read the saved fits with fn_read_APMS_full_fit_benchmark() to recover their grouping metadata; no fits are needed.")
        }
        setting_columns <-  c("n_burnin", "n_iter", "n_chains_burnin", "n_chains_sampling",
                               "threads_per_chain_burnin", "threads_per_chain_sampling", "target_min_ESS")
        metric_columns <-  c("min_ESS_bulk", "min_ESS_tail", "max_rhat", "sampling_acceptance",
                              "estimated_time_to_target_ESS", "ESS_per_sec_sampling", "ESS_per_1000_grad_sampling",
                              "grad_evals_per_sec_sampling", "native_burnin_seconds", "native_sampling_seconds",
                              "n_divergent", "n_treedepth_hits")
        ##
        for (column_name in setdiff(x = c(setting_columns, metric_columns), y = names(x = benchmark_results))) {
            benchmark_results <-  dplyr::mutate(.data = benchmark_results, !!column_name := NA_real_)
        }
        ## Recompute each fit BEFORE averaging, including when the caller still holds an older runs table.
        for (column_name in setdiff(x = c("engine_summary_seconds", "comparison_seconds"), y = names(x = benchmark_results))) {
            benchmark_results <-  dplyr::mutate(.data = benchmark_results, !!column_name := NA_real_)
        }
        benchmark_results <-  benchmark_results |>
            dplyr::mutate(estimated_time_to_target_ESS = purrr::pmap_dbl(
                .l = list(burnin_sec = native_burnin_seconds, sampling_sec = native_sampling_seconds,
                          engine_summary_sec = engine_summary_seconds, comparison_summary_sec = comparison_seconds,
                          target_ESS = target_min_ESS, observed_ESS = min_ESS_bulk),
                .f = fn_APMS_benchmark_time_to_target_ESS))
        ##
        grouping_columns <-  c("model", "N", "engine", "configuration", "repetition_group")
        fn_mean_available <-  function(values) {
            values <-  values[is.finite(x = values)]
            if (!length(x = values)) return(NA_real_)
            return(mean(x = values))
        }
        ##
        configuration_summary <-  benchmark_results |>
            dplyr::group_by(dplyr::across(dplyr::all_of(grouping_columns))) |>
            dplyr::summarise(
                n_runs = dplyr::n(),
                n_seeds = dplyr::n_distinct(seed, na.rm = TRUE),
                seeds = paste(sort(x = unique(x = seed)), collapse = ", "),
                n_completed = sum(status == "completed", na.rm = TRUE),
                n_failed = sum(is.na(status) | status != "completed"),
                ## The definition fingerprint already fixes settings. Older failed rows may lack flattened counts.
                dplyr::across(dplyr::all_of(setting_columns), ~ dplyr::first(.x[is.finite(.x)], default = NA_real_)),
                dplyr::across(dplyr::all_of(metric_columns), ~ fn_mean_available(values = .x[status == "completed"])),
                dplyr::across(dplyr::all_of(metric_columns), ~ sum(status == "completed" & is.finite(.x), na.rm = TRUE),
                              .names = "n_available_{.col}"),
                errors = paste(unique(x = error[!is.na(error) & nzchar(error)]), collapse = "; "),
                .groups = "drop")
        ##
        if (length(x = sort_by) != 1 || !sort_by %in% metric_columns) { 
          stop("sort_by must name one reported metric: ", paste(metric_columns, collapse = ", "))
        }
        ##
        larger_is_better <-  sort_by %in% c("min_ESS_bulk", "min_ESS_tail", "ESS_per_sec_sampling",
                                            "ESS_per_1000_grad_sampling", "grad_evals_per_sec_sampling", "sampling_acceptance")
        configuration_summary <-  configuration_summary |>
            dplyr::arrange(model, N, if (larger_is_better) dplyr::desc(.data[[sort_by]]) else .data[[sort_by]], configuration) |>
            dplyr::mutate(id = dplyr::row_number(), .before = 1)
        ##
        ## Add readable column names without removing legacy columns or converting numeric values to display strings.
        for (column_alias in names(x = metric_column_aliases)) {
            configuration_summary <-  dplyr::mutate(.data = configuration_summary,
                                                     !!column_alias := .data[[metric_column_aliases[[column_alias]]]])
        }
        ##
        if (length(x = filter_values)) {
            unknown_filter_columns <-  setdiff(x = names(x = filter_values), y = names(x = configuration_summary))
            if (length(x = unknown_filter_columns)) stop("Unknown filter column(s): ", paste(unknown_filter_columns, collapse = ", "))
            for (filter_column in names(x = filter_values)) {
                allowed_values <-  filter_values[[filter_column]]
                configuration_summary <-  dplyr::filter(.data = configuration_summary, .data[[filter_column]] %in% .env$allowed_values)
            }
        }
        if (!nrow(x = configuration_summary)) {
            if (print_table) cat("No configurations match filter_values. The unfiltered per-run data are unchanged.\n")
            return(invisible(x = configuration_summary))
        }
        ##
        if (print_table) {
            cat("\n==== APMS: SEEDS SUMMARISED ====\n")
            cat("Means of available metrics from completed fits, including those failing diagnostics.\n",
                "Runs = completed/total; different data, settings or recorded code versions stay separate.\n",
                "ESS and R-hat refer to Se / Sp / prevalence. Per-run data remain in benchmark_results.\n", sep = "")
            cat("Time units: sec (including time_to_target_ESS).\n")
            cat("Target time includes burn-in + scaled sampling and summaries; setup/compile is separate.\n")
            ##
            model_panels <-  configuration_summary |> dplyr::group_by(model, N) |> dplyr::group_split()
            ##
            for (model_panel in model_panels) {
              
                    cat(sprintf(fmt = "\n---- %s | N = %s ----\n", model_panel$model[1], model_panel$N[1]))
                    cat(sprintf(fmt = "%3s %-18s %7s %8s %9s %10s %18s\n",
                                "ID", "Configuration", "Runs", "min ESS", "ESS/sec", "ESS/1kgrad", "time_to_target_ESS"))
                    cat(sprintf(fmt = "%3d %-18s %7s %8.0f %9.1f %10.2f %18.1f\n",
                                model_panel$id, model_panel$configuration, paste0(model_panel$n_completed, "/", model_panel$n_runs),
                                model_panel$min_ESS_bulk, model_panel$ESS_per_sec_sampling,
                                model_panel$ESS_per_1000_grad_sampling, model_panel$estimated_time_to_target_ESS), sep = "")
                    cat(sprintf(fmt = "\n%3s %10s %12s %8s %8s %9s %10s\n",
                                "ID", "burnin_sec", "sampling_sec", "R-hat", "Accept", "Divs/fit", "Depth/fit"))
                    cat(sprintf(fmt = "%3d %10.1f %12.1f %8.3f %8.3f %9.1f %10.1f\n",
                                model_panel$id, model_panel$native_burnin_seconds, model_panel$native_sampling_seconds,
                                model_panel$max_rhat, model_panel$sampling_acceptance, model_panel$n_divergent,
                                model_panel$n_treedepth_hits), sep = "")
                    cat("\nSettings by ID (burn-in/sampling; iterations per chain):\n")
                    ##
                    for (row_index in seq_len(length.out = nrow(x = model_panel))) {
                      
                            row <-  dplyr::slice(.data = model_panel, row_index)
                            cat(sprintf(fmt = "  %d: iter %.0f/%.0f; chains %.0f/%.0f; WCP %.0f/%.0f; target ESS %.0f\n",
                                        row$id, row$n_burnin, row$n_iter, row$n_chains_burnin, row$n_chains_sampling,
                                        row$threads_per_chain_burnin, row$threads_per_chain_sampling, row$target_min_ESS))
                            ##
                            if (row$n_seeds < row$n_runs) cat(" Repeated seed values are present; these are not all independent repetitions.\n")
                            ##
                            missing_metrics <-  metric_columns[vapply(X = metric_columns, FUN = function(metric_name) {
                                row[[paste0("n_available_", metric_name)]] < row$n_completed
                            }, FUN.VALUE = FALSE)]
                            ##
                            ## Tail ESS and treedepth can be unavailable by design; retain availability counts in the returned tibble.
                            ##
                            missing_metrics <-  setdiff(x = missing_metrics, y = c("min_ESS_tail", "n_treedepth_hits"))
                            ##
                            if (length(x = missing_metrics)) cat("     Some completed fits lack metrics; inspect n_available_* in benchmark_summary.\n")
                            if (nzchar(x = row$errors)) cat(paste(strwrap(x = paste0(" Errors: ", row$errors), width = 90), collapse = "\n"), "\n")
                        
                    }
                
            }
        }
        ##
        return(invisible(x = configuration_summary))

}













