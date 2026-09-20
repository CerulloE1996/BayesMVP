##
## ---- run_one_arm.R ---------------------------------------------------------------------------------------------------------------
##
## One ps7 run of the efficiency decomposition. Everything NOT set below is fixed at the 09-17
## N = 10000 configuration (ps7 settings combo 11: n_burnin = 250, learning_rate = 0.05, clip_iter = 50,
## int = 75, n_iter = 100, learning_rate_initial = 0.15, 180 sampling chains, 4 burn-in chains,
## KE tau objective, DGM 3, data_seed = 123).
##
## ---- Two ways to run it:
##
##   (1) From a terminal (a FRESH R process - needed for the legacy C++ builds):
##
##         cd ~ && Rscript run_one_arm.R arm_name=H seed_index=1 output_root_dir=/path/to/outputs \
##                  BayesMVP_library_path=installed metric_estimator=chain_mean_scaled tau_initial=2pi \
##                  tau_ramp=original num_chunks_burnin=25 num_chunks_sampling=25 test_perm_override=546132 \
##                  diffusion_HMC_integrator=kick_flow_kick
##
##       (run_decomposition.sh does exactly this for every arm and seed.)
##
##   (2) In RStudio: edit the "defaults when there are no command-line arguments" block below, then
##       Source the whole file. Only arms with BayesMVP_library_path = "installed" (E0, F, H) can run
##       like this, because an RStudio session that has already loaded BayesMVP cannot switch to
##       a legacy build.
##
##   BayesMVP_library_path   "installed", or a library folder made by build_legacy_variants.sh,
##                           e.g. ~/BayesMVP_legacy_variant_libs/legacy_both
##   metric_estimator        pooled | chain_mean | chain_mean_scaled
##   tau_initial             pi | 2pi | <number>
##   tau_ramp                original | staged
##   test_perm_override      e.g. 546132, or "estimate" to use the pre-burnin's estimated test order
##   learning_rate_initial   default 0.15 (the 09-17 value); any other value is added to the arm's folder name
##   seed_index              k  ->  MCMC seed = k * 1000
##
{
      command_line_args <- commandArgs(trailingOnly = TRUE)
      ##
      ## ---- Defaults when there are no command-line arguments (i.e. when Sourced in RStudio). EDIT HERE:
      ##
      if (length(command_line_args) == 0) {

            default_args_for_RStudio <- c( arm_name                  = "H",
                                           seed_index                = "123",
                                           output_root_dir           = "/home/enzocerullo/Documents/Work/PhD_work/Alg_paper_analysis/1_appendix_pilot_studies/ps_7_basic_MCMC_settings_BayesMVP/outputs/DGM_3/efficiency_decomposition_2026_09_18",
                                           BayesMVP_library_path     = "installed",
                                           ##
                                           metric_estimator          = "chain_mean_scaled",
                                           tau_initial               = "2pi",
                                           tau_ramp                  = "original",
                                           ##
                                           num_chunks_burnin         = "25",
                                           num_chunks_sampling       = "25",
                                           ##
                                           test_perm_override        = "546132",
                                           diffusion_HMC_integrator  = "kick_flow_kick",
                                           learning_rate_initial     = "0.10")
            command_line_args <- paste0(names(default_args_for_RStudio), "=", default_args_for_RStudio)
            message("run_one_arm.R: no command-line arguments - using the RStudio defaults block at the top of the file.")

      }
      ##
      ## ---- Parse "name=value" arguments:
      ##
      command_line_arg_values <- setNames(object = sub(pattern = "^[^=]*=", replacement = "", x = command_line_args),
                                          nm     = sub(pattern = "=.*$",    replacement = "", x = command_line_args))
      if ("burnin_post_adapt_iter" %in% names(command_line_arg_values)) {
            stop("burnin_post_adapt_iter has been removed. The full n_burnin iterations are now always run.")
      }
      ##
      fn_get_command_line_arg <- function(arg_name,
                                          default_value = NULL) {
            if (is.na(command_line_arg_values[arg_name])) return(default_value)
            unname(command_line_arg_values[arg_name])
      }
      ##
      ## ---- All settings for this arm in ONE long-named list, so nothing sourced below can overwrite them:
      ##
      decomposition_arm_settings <- list( arm_name                 = fn_get_command_line_arg("arm_name"),
                                          seed_index               = as.integer(fn_get_command_line_arg("seed_index", "1")),
                                          output_root_dir          = fn_get_command_line_arg("output_root_dir"),
                                          BayesMVP_library_path    = fn_get_command_line_arg("BayesMVP_library_path", "installed"),
                                          metric_estimator         = fn_get_command_line_arg("metric_estimator", "pooled"),
                                          tau_initial_as_string    = fn_get_command_line_arg("tau_initial", "2pi"),
                                          tau_ramp                 = fn_get_command_line_arg("tau_ramp", "original"),
                                          num_chunks_burnin        = as.integer(fn_get_command_line_arg("num_chunks_burnin", "25")),
                                          num_chunks_sampling      = as.integer(fn_get_command_line_arg("num_chunks_sampling", "25")),
                                          test_perm_as_string      = fn_get_command_line_arg("test_perm_override", "546132"),
                                          diffusion_HMC_integrator = fn_get_command_line_arg("diffusion_HMC_integrator", "kick_flow_kick"),
                                          learning_rate_initial    = as.numeric(fn_get_command_line_arg("learning_rate_initial", "0.15")))
      ##
      if (is.null(decomposition_arm_settings$arm_name) || is.null(decomposition_arm_settings$output_root_dir)) {
        stop("run_one_arm.R: 'arm_name' and 'output_root_dir' must be given (command line, or the RStudio defaults block).")
      }
      ##
      decomposition_arm_settings$tau_initial <- switch( decomposition_arm_settings$tau_initial_as_string,
                                                        "pi"  = pi,
                                                        "2pi" = 2*pi,
                                                        as.numeric(decomposition_arm_settings$tau_initial_as_string))
      ##
      decomposition_arm_settings$test_perm_override <- if (identical(decomposition_arm_settings$test_perm_as_string, "estimate")) {
                                                          NULL
                                                        } else {
                                                          as.integer(strsplit(decomposition_arm_settings$test_perm_as_string, split = "")[[1]])
                                                        }
      ##
      ## ---- One folder per (arm, configuration, seed): a non-default learning_rate_initial gets its own folder,
      ##      so different configurations can never be mixed up by the summariser.
      ##
      decomposition_arm_settings$arm_label <- if (decomposition_arm_settings$learning_rate_initial == 0.15) {
                                                 decomposition_arm_settings$arm_name
                                               } else {
                                                 paste0(decomposition_arm_settings$arm_name, "_Li", decomposition_arm_settings$learning_rate_initial)
                                               }
      ##
      ## ---- A legacy C++ build is used by putting its library folder FIRST on the library path:
      ##
      if (!identical(decomposition_arm_settings$BayesMVP_library_path, "installed")) {
        .libPaths(c(path.expand(decomposition_arm_settings$BayesMVP_library_path), .libPaths()))
      }
}
##
## ---- Same environment as the ps7 script (packages, helper functions, the ps7 runner functions):
##
{
      setwd("/home/enzocerullo/Documents/Work/PhD_work/Alg_paper_analysis")
      source(file.path(getwd(), "R_main_v5.R"))
      ##
      algorithm_study_dir <- "/home/enzocerullo/Documents/Work/PhD_work/Alg_paper_analysis"
      computer  <- "Local_HPC"
      vect_type <- "AVX512"
      n_threads <- 180
      options(scipen = 999, mc.cores = n_threads)
      ##
      DGM_to_use       <- 3
      true_prev        <- 0.05
      LC_MVOP_grouping <- FALSE
      ##
      ps7_functions_dir <- file.path(algorithm_study_dir, "1_appendix_pilot_studies", "ps_7_basic_MCMC_settings_BayesMVP", "functions")
      source(file.path(ps7_functions_dir, "ps_7_MCMC_settings_BayesMVP_functions.R"))
      ##
      suppressPackageStartupMessages(library(BayesMVP))
      cat("#### ARM", decomposition_arm_settings$arm_name, "| seed", decomposition_arm_settings$seed_index * 1000,
          "| BayesMVP loaded from:", find.package("BayesMVP"), "\n")
      ##
      ## A legacy build can only be used if BayesMVP was not already loaded from somewhere else in this session:
      ##
      if (!identical(decomposition_arm_settings$BayesMVP_library_path, "installed")) {
            BayesMVP_loaded_from     <- normalizePath(find.package("BayesMVP"))
            BayesMVP_requested_from  <- normalizePath(path.expand(decomposition_arm_settings$BayesMVP_library_path))
            if (!startsWith(BayesMVP_loaded_from, BayesMVP_requested_from)) {
              stop("arm ", decomposition_arm_settings$arm_name, " needs the build in ", BayesMVP_requested_from,
                   " but this session already has BayesMVP loaded from ", BayesMVP_loaded_from,
                   ". Run legacy arms in a fresh R process (./run_decomposition.sh).")
            }
      }
      ##
      Model_type              <- "LC_MVP"
      reorder_cols_MVP        <- TRUE
      multi_attempts          <- TRUE
      corr_force_positive_DGM <- FALSE
      BayesMVP:::set_debug_cutpoint_grads(FALSE)
      options(BayesMVP_force_L1 = FALSE)
}
##
## ---- ps7 settings (the 09-17 configuration + this arm's settings):
##
{
      ps7_settings <- list( n_nuisance_to_track      = 0,
                            n_refresh                = 10,
                            ##
                            eps_initial              = NULL,
                            eps_initial_iter         = NULL,
                            ##
                            DGM                      = DGM_to_use,
                            model_type               = Model_type,
                            reorder_cols_MVP         = reorder_cols_MVP,
                            multi_attempts           = multi_attempts,
                            prior_name               = "LKJ_10_1.5",
                            ##
                            N_vec                    = 10000,
                            data_seed                = 123,
                            ##
                            n_runs                   = 1,
                            ##
                            n_chains_burnin          = 4,
                            n_threads_WCP_burnin     = 10,
                            ##
                            n_chains_sampling        = 180,
                            n_superchains            = 180,
                            n_threads_WCP_sampling   = 1,
                            ##
                            n_iter                   = 100,
                            ##
                            adapt_delta              = 0.80,
                            ##
                            learning_rate_initial      = decomposition_arm_settings$learning_rate_initial,
                            learning_rate_initial_iter = NULL,
                            ##
                            metric_type_main         = "Empirical",
                            metric_shape_main        = "dense",
                            ##
                            metric_type_nuisance     = "uniform_diag",
                            ##
                            diffusion_HMC_integrator = decomposition_arm_settings$diffusion_HMC_integrator,
                            tau_objective            = "KE",
                            tau_weight_by_p_jump     = FALSE,
                            ##
                            manual_L                 = NA_real_,
                            manual_tau_value         = NA_real_,
                            ##
                            diffusion_HMC            = TRUE,
                            partitioned_HMC          = FALSE,
                            M_decay_type             = "inverse",
                            M_decay_power            = 0.50,
                            ##
                            ## ---- this arm:
                            metric_estimator         = decomposition_arm_settings$metric_estimator,
                            tau_initial              = decomposition_arm_settings$tau_initial,
                            tau_ramp                 = decomposition_arm_settings$tau_ramp,
                            ## optional extras (absent on the command line = NULL = the package default):
                            eps_reinit_at_ChEES_handover = if (is.na(command_line_arg_values["eps_reinit_at_ChEES_handover"])) NULL else as.logical(command_line_arg_values[["eps_reinit_at_ChEES_handover"]]),
                            theta_hat_us_rule        = if (is.na(command_line_arg_values["theta_hat_us_rule"])) NULL else command_line_arg_values[["theta_hat_us_rule"]],
                            pre_burnin_n_iter        = if (is.na(command_line_arg_values["pre_burnin_n_iter"])) NULL else as.numeric(command_line_arg_values[["pre_burnin_n_iter"]]),
                            pre_burnin_L             = if (is.na(command_line_arg_values["pre_burnin_L"])) NULL else as.numeric(command_line_arg_values[["pre_burnin_L"]]),
                            test_perm_override       = decomposition_arm_settings$test_perm_override,
                            num_chunks_burnin        = decomposition_arm_settings$num_chunks_burnin,
                            num_chunks_sampling      = decomposition_arm_settings$num_chunks_sampling,
                            output_dir               = file.path(decomposition_arm_settings$output_root_dir,
                                                                 paste0(decomposition_arm_settings$arm_label, "_seed", decomposition_arm_settings$seed_index)))
      ##
      ## ---- combo 11 of the ps7 script:
      ##
      ps7_settings$clip_iter        <- 50
      ps7_settings$int              <- 75
      ##
      ps7_settings$ratio_M_nuisance <- 0.90
      ps7_settings$ratio_M_main     <- 0.90
      ps7_settings$int_width        <- 1
      ##
      ps7_settings$n_burnin         <- 250
      ps7_settings$learning_rate    <- 0.05
      ##
      n_adapt <- ps7_settings$n_burnin - round(ps7_settings$n_burnin / 10)
      ps7_settings$M_decay_scale <- n_adapt / 100
      ##
      dir.create(ps7_settings$output_dir, recursive = TRUE, showWarnings = FALSE)
}
##
## ---- Data (the historical N = 10,000 dataset) and the run:
##
{
      source(file.path(algorithm_study_dir, "0_utilities", "shared_functions", "R_fn_sim_bin_COVID_19_LC_MVP_data.R"))
      binary_sim_outs <- R_fn_simulate_binary_LC_MVP_data_COVID_19( study_type              = "algorithm_parallel_scaling_tests",
                                                                    seed                    = 123,
                                                                    corr_force_positive_DGM = corr_force_positive_DGM)
      ##
      ## The N = 10000 dataset depends on which N are simulated BEFORE it (they share one RNG stream).
      ## Only 500 and 2500 came before it in every run up to 2026-09-17 20:45; anything else gives a
      ## DIFFERENT dataset (this is what adding N = 5000 did).
      ##
      dataset_index_N_10000 <- which(binary_sim_outs$N_vec == 10000)
      N_values_simulated_before_10000 <- binary_sim_outs$N_vec[seq_len(dataset_index_N_10000 - 1)]
      if (!all(N_values_simulated_before_10000 %in% c(500, 2500))) {
        stop("N_vec in R_fn_sim_bin_COVID_19_LC_MVP_data.R puts ", paste(N_values_simulated_before_10000, collapse = ", "),
             " before 10000, so this is not the historical N = 10000 dataset. Put new N values AFTER 10000.")
      }
      y   <- binary_sim_outs$y_list[[dataset_index_N_10000]]
      pop <- binary_sim_outs$pop_list[[dataset_index_N_10000]]
      ##
      ## time stamp before the run - used below to find the ONE file this run writes (ps7 SKIPS a run whose file already exists):
      run_start_time <- Sys.time()
      ##
      raw_results <- run_ps7_models( Model_type                 = Model_type,
                                     N                          = 10000,
                                     y                          = y,
                                     settings                   = ps7_settings,
                                     vect_type                  = vect_type,
                                     runs_override              = c(decomposition_arm_settings$seed_index),
                                     use_disk                   = FALSE,
                                     use_disk_path              = file.path(decomposition_arm_settings$output_root_dir,
                                                                            paste0("hmc_traces_", decomposition_arm_settings$arm_name)),
                                     use_disk_path_post_hoc_dir = file.path(decomposition_arm_settings$output_root_dir,
                                                                            paste0("constrain_traces_", decomposition_arm_settings$arm_name)),
                                     true_prev                  = true_prev,
                                     prior_LKJ_nd               = NULL,
                                     prior_LKJ_d                = NULL,
                                     prior_prev_a               = 2.5,
                                     prior_prev_b               = 10.0,
                                     pop                        = pop,
                                     LC_MVOP_grouping           = LC_MVOP_grouping)
      ##
      ##
      ## ---- Record exactly which file THIS run wrote, so the summariser never picks up anything else:
      ##
      run_files_after_run <- file.info(Sys.glob(file.path(ps7_settings$output_dir, "ps7_run_*")))
      run_files_written   <- rownames(run_files_after_run)[run_files_after_run$mtime >= run_start_time]
      if (length(run_files_written) != 1) {
        stop("expected this run to write exactly ONE ps7_run_* file in ", ps7_settings$output_dir, ", found ", length(run_files_written),
             ". If it is 0, ps7's resume check skipped the run because a file with the same name already exists",
             " (tau_ramp and some other options are not in the file name) - delete it or use another output_root_dir.")
      }
      saveRDS(list( run_file_path              = run_files_written,
                    decomposition_arm_settings = decomposition_arm_settings,
                    finished_at                = Sys.time()),
              file.path(ps7_settings$output_dir, "run_record.rds"))
      ##
      cat("\n######## DONE ARM", decomposition_arm_settings$arm_label, "seed", decomposition_arm_settings$seed_index * 1000, "########\n")
}












