##
## ==== R_fn_time_sampling_of_BayesMVP_model_for_each_layout.R ===================================================
##
## ---- Seconds per iteration of a BayesMVP model with manual gradients (Model_type "MVP", "MVOP", "LC_MVP" or
##      "LC_MVOP") for each sampling layout of chains, within-chain threads and chunks, in the sampler of the
##      sampling phase -
##
## Each layout is timed in the sampler that $sample() runs for the sampling
## (Rcpp_fn_RcppParallel_EHMC_sampling()): every chain runs its iterations independently in one call, so that the
## call waits for the slowest chain once, as the sampling does - and not at every iteration, as the persistent
## burn-in worker of fn_time_iteration_of_BayesMVP_model_for_each_layout() does, which makes a layout with more
## chains look slower than it is in the sampling. The trajectory length is fixed (L_main leapfrog steps,
## randomize_tau = FALSE), at a tiny step size (eps = 1e-5), so that every layout does the same work. The seconds
## per iteration are the difference of a long call and a short call (n_iterations_of_short_call iterations) over
## the difference of their iterations, so that the setup of each call (the chains' workspaces; several seconds
## with 180 chains at N = 50,000, varying by seconds between calls) is not counted, as it is not in a sampling of
## hundreds of iterations. The seconds of each call are those of its chains alone (from the start of the chains to
## the end of the last chain), when the sampler returns them (sampling_call_timing), and otherwise its wall
## seconds, setup included (the difference then cancels the setup only as far as it is the same in both calls).
## The long call has at least minimum_seconds_per_timing seconds of iterations (as estimated from the short
## call). The user-mode
## and kernel-mode CPU seconds of the two calls are returned too, and, when fn_snapshot_hardware_counters is
## given, it is called just before and just after each of them (e.g. to snapshot Paper 1's E4 counter tool
## pmc_stat, which counts user-mode cycles), so that the clock of the busy threads during the iterations is the
## difference of the two calls' user-mode cycles over the difference of their user-mode CPU seconds.
##
##
##
##
#### ---- the timing --------------------------------------------------------------------------------------------
#' Seconds per iteration of a BayesMVP model (manual gradients) in the sampler of the sampling phase, for each
#' layout of chains, within-chain threads and chunks
#'
#' @param Model_type "MVP", "MVOP", "LC_MVP" or "LC_MVOP".
#' @param init_object The model's initialisation object (from BayesMVP::initialise_model() or the init_object of
#'   BayesMVP::MVP_model$new(); it holds Model_args_as_Rcpp_List, model_args_list, n_params_main and n_nuisance).
#' @param layouts A data frame with columns n_chains (the sampling chains, run in parallel), n_threads_WCP
#'   (within-chain threads) and num_chunks.
#' @param minimum_seconds_per_timing The shortest time of the iterations of each layout's long call (default
#'   2 s; the long call is made again, longer, when its iterations took less).
#' @param n_iterations_of_short_call The iterations of the short call (default 1).
#' @param L_main,eps_main The leapfrog steps (fixed) and step size (default 10 and 1e-5).
#' @param fn_snapshot_hardware_counters A function without arguments, called just before and just after each of
#'   the two calls used, which returns the number of its snapshot (default NULL: none).
#' @return The data frame layouts with seconds_per_iteration, seconds_per_draw_over_all_chains (the seconds per
#'   iteration divided by the layout's chains), timed_by_chains_seconds_of_the_sampler (TRUE: from the seconds of
#'   the calls' chains alone; FALSE: from their wall seconds, setup included),
#'   seconds_per_iteration_from_wall_seconds_of_the_calls, n_iterations_of_long_call, seconds_spent_timing (the
#'   wall time of the layout's calls, setup included), user_CPU_seconds_of_short_call,
#'   kernel_CPU_seconds_of_short_call, user_CPU_seconds_of_long_call and kernel_CPU_seconds_of_long_call (of the R
#'   process and all its threads; the long call used, i.e. the one made again when it was), and
#'   counter_snapshot_before_short_call, counter_snapshot_after_short_call, counter_snapshot_before_long_call and
#'   counter_snapshot_after_long_call (NA without fn_snapshot_hardware_counters).
#' @export
fn_time_sampling_of_BayesMVP_model_for_each_layout <-  function( Model_type,
                                                                init_object,
                                                                layouts,
                                                                minimum_seconds_per_timing     = 2,
                                                                n_iterations_of_short_call     = 1,
                                                                L_main                         = 10,
                                                                eps_main                       = 1e-05,
                                                                fn_snapshot_hardware_counters  = NULL) {
        ##
        if (!(Model_type %in% c("MVP", "MVOP", "LC_MVP", "LC_MVOP"))) {
              stop(paste0("fn_time_sampling_of_BayesMVP_model_for_each_layout: Model_type must be \"MVP\", ",
                          "\"MVOP\", \"LC_MVP\" or \"LC_MVOP\" (BayesMVP's models with manual gradients); got: ",
                          paste(as.character(Model_type), collapse = ", ")))
        }
        columns_of_layouts <-  c("n_chains", "n_threads_WCP", "num_chunks")
        if (!is.data.frame(layouts) || !all(columns_of_layouts %in% names(layouts))) {
              stop(paste0("fn_time_sampling_of_BayesMVP_model_for_each_layout: layouts must be a data frame ",
                          "with columns n_chains, n_threads_WCP and num_chunks."))
        }
        ##
        n_params_main <-  init_object$n_params_main
        n_nuisance <-  init_object$n_nuisance
        y <-  init_object$model_args_list$y
        ##
        ## ---- the sampler's arguments: diffusion HMC (kick-flow-kick), a fixed trajectory of L_main leapfrog
        ##      steps at eps_main, no log-lik trace, and a dense main metric (the identity):
        ##
        EHMC_args_as_Rcpp_List <-  init_EHMC_args_as_Rcpp_List( diffusion_HMC = TRUE,
                                                                diffusion_HMC_integrator = "kick_flow_kick")
        EHMC_args_as_Rcpp_List$eps_main <-  eps_main
        EHMC_args_as_Rcpp_List$tau_main <-  L_main * eps_main
        EHMC_args_as_Rcpp_List$eps_us <-  eps_main
        EHMC_args_as_Rcpp_List$tau_us <-  L_main * eps_main
        EHMC_args_as_Rcpp_List$randomize_tau <-  FALSE
        EHMC_args_as_Rcpp_List$store_log_lik_trace <-  FALSE
        EHMC_Metric_as_Rcpp_List <-  init_EHMC_Metric_as_Rcpp_List( n_params_main = n_params_main,
                                                                    n_nuisance = n_nuisance,
                                                                    metric_shape_main = "dense")
        fn_small_values <-  function(n_values) 0.005 * (((seq_len(n_values) * 7) %% 5) - 2)
        ##
        layouts$seconds_per_iteration <-  NA_real_
        layouts$seconds_per_draw_over_all_chains <-  NA_real_
        layouts$timed_by_chains_seconds_of_the_sampler <-  NA
        layouts$seconds_per_iteration_from_wall_seconds_of_the_calls <-  NA_real_
        layouts$n_iterations_of_long_call <-  NA_real_
        layouts$seconds_spent_timing <-  NA_real_
        layouts$user_CPU_seconds_of_short_call <-  NA_real_
        layouts$kernel_CPU_seconds_of_short_call <-  NA_real_
        layouts$user_CPU_seconds_of_long_call <-  NA_real_
        layouts$kernel_CPU_seconds_of_long_call <-  NA_real_
        layouts$counter_snapshot_before_short_call <-  NA_real_
        layouts$counter_snapshot_after_short_call <-  NA_real_
        layouts$counter_snapshot_before_long_call <-  NA_real_
        layouts$counter_snapshot_after_long_call <-  NA_real_
        ##
        ## ---- one call, with its wall seconds, the seconds of its chains alone (when the sampler returns them),
        ##      its user-mode and kernel-mode CPU seconds and the counter snapshots just before and just after it
        ##      (outside its wall seconds):
        ##
        fn_snapshot_or_NA <-  function() {
              if (is.null(fn_snapshot_hardware_counters)) return(NA_real_)
              return(fn_snapshot_hardware_counters())
        }
        fn_timed_call <-  function(fn_call) {
              snapshot_before <-  fn_snapshot_or_NA()
              process_time_before <-  proc.time()
              result_of_call <-  fn_call()
              process_time_after <-  proc.time()
              snapshot_after <-  fn_snapshot_or_NA()
              chains_seconds <-  NA_real_
              if (!is.null(result_of_call$sampling_call_timing)) {
                    chains_seconds <-  result_of_call$sampling_call_timing$chains_seconds
              }
              ## (the divergent transitions of the call, from the sampler's divergence trace, its second element:
              ##  a timing with any divergent transition is invalid, as a divergent trajectory stops early)
              n_divergent_transitions <-  NA_real_
              if (length(result_of_call) >= 2) {
                    n_divergent_transitions <-  sum(unlist(result_of_call[[2]]))
              }
              rm(result_of_call)
              ## (as Paper 1's burn-in timings, R_fn_run_ps_1_burnin_benchmark.R: eps is tiny and L fixed, so
              ##  nothing should diverge; a divergent transition makes the configuration invalid)
              if (isTRUE(n_divergent_transitions > 0)) {
                    stop("Sampling timing encountered ", n_divergent_transitions,
                         " divergent transition(s) in a timed call. The configuration is invalid for a timing ",
                         "comparison.")
              }
              return(list( seconds              = process_time_after[["elapsed"]] -
                                                  process_time_before[["elapsed"]],
                           chains_seconds       = chains_seconds,
                           user_CPU_seconds     = process_time_after[["user.self"]] -
                                                  process_time_before[["user.self"]],
                           kernel_CPU_seconds   = process_time_after[["sys.self"]] -
                                                  process_time_before[["sys.self"]],
                           snapshot_before      = snapshot_before,
                           snapshot_after       = snapshot_after))
        }
        for (layout_index in seq_len(nrow(layouts))) {
              ##
              time_layout_start <-  proc.time()[["elapsed"]]
              n_chains <-  layouts$n_chains[layout_index]
              n_threads_WCP <-  layouts$n_threads_WCP[layout_index]
              Model_args_as_Rcpp_List <-  init_object$Model_args_as_Rcpp_List
              Model_args_as_Rcpp_List$Model_args_ints[4] <-  layouts$num_chunks[layout_index]
              initial_main <-  matrix(fn_small_values(n_params_main * n_chains), nrow = n_params_main,
                                      ncol = n_chains)
              initial_nuisance <-  matrix(fn_small_values(n_nuisance * n_chains), nrow = n_nuisance,
                                          ncol = n_chains)
              RcppParallel::setThreadOptions(numThreads = n_chains * n_threads_WCP)
              fn_sampling_call <-  function(n_iterations, seed_of_call) {
                    Rcpp_fn_RcppParallel_EHMC_sampling(
                          n_threads_R                                 = n_chains,
                          seed_R                                      = seed_of_call,
                          n_iter_R                                    = n_iterations,
                          iter_one_by_one                             = FALSE,
                          partitioned_HMC_R                           = FALSE,
                          diffusion_HMC_R                             = TRUE,
                          Model_type_R                                = Model_type,
                          sample_nuisance_R                           = TRUE,
                          force_autodiff_R                            = FALSE,
                          force_PartialLog_R                          = FALSE,
                          multi_attempts_R                            = TRUE,
                          n_nuisance_to_track                         = 0,
                          theta_main_vectors_all_chains_input_from_R  = initial_main,
                          theta_us_vectors_all_chains_input_from_R    = initial_nuisance,
                          y_Eigen_R                                   = y,
                          Model_args_as_Rcpp_List                     = Model_args_as_Rcpp_List,
                          EHMC_args_as_Rcpp_List                      = EHMC_args_as_Rcpp_List,
                          EHMC_Metric_as_Rcpp_List                    = EHMC_Metric_as_Rcpp_List,
                          use_disk                                    = FALSE,
                          trace_dir                                   = tempdir(),
                          n_threads_WCP                               = n_threads_WCP)
              }
              ##
              ## ---- the short call, then the long call, each with its setup:
              ##
              ## time_before_short_call <-  proc.time()[["elapsed"]]
              ## invisible(fn_sampling_call(n_iterations_of_short_call, 1))
              ## seconds_of_short_call <-  proc.time()[["elapsed"]] - time_before_short_call
              short_call <-  fn_timed_call(function() fn_sampling_call(n_iterations_of_short_call, 1))
              ## (the seconds of the calls' chains alone when the sampler returns them, otherwise their wall
              ##  seconds)
              timed_by_chains_seconds <-  !is.na(short_call$chains_seconds)
              fn_seconds_of_call <-  function(call) {
                    if (timed_by_chains_seconds) return(call$chains_seconds)
                    return(call$seconds)
              }
              seconds_of_short_call <-  fn_seconds_of_call(short_call)
              seconds_per_iteration_at_most <-  seconds_of_short_call / n_iterations_of_short_call
              n_iterations_of_long_call <-  n_iterations_of_short_call +
                                            max(2, ceiling(minimum_seconds_per_timing /
                                                           seconds_per_iteration_at_most))
              ## time_before_long_call <-  proc.time()[["elapsed"]]
              ## invisible(fn_sampling_call(n_iterations_of_long_call, 2))
              ## seconds_of_long_call <-  proc.time()[["elapsed"]] - time_before_long_call
              long_call <-  fn_timed_call(function() fn_sampling_call(n_iterations_of_long_call, 2))
              seconds_of_long_call <-  fn_seconds_of_call(long_call)
              ##
              ## ---- the seconds per iteration: the difference of the two calls over the difference of their
              ##      iterations (the long call's seconds per iteration if the difference is not positive):
              ##
              seconds_per_iteration <-  (seconds_of_long_call - seconds_of_short_call) /
                                        (n_iterations_of_long_call - n_iterations_of_short_call)
              ##
              ## ---- (the short call's seconds include the setup, so that its estimate of the seconds per
              ##       iteration is too high and the long call can have fewer than minimum_seconds_per_timing
              ##       seconds of iterations: then the long call is made again, sized with the seconds per
              ##       iteration found)
              ##
              seconds_of_iterations_of_long_call <-  seconds_per_iteration *
                                                     (n_iterations_of_long_call - n_iterations_of_short_call)
              long_call_too_short <-  seconds_of_iterations_of_long_call < minimum_seconds_per_timing
              if ((seconds_per_iteration > 0) && long_call_too_short) {
                    n_iterations_of_long_call <-  n_iterations_of_short_call +
                                                  max(2, ceiling(minimum_seconds_per_timing /
                                                                 seconds_per_iteration))
                    ## time_before_long_call <-  proc.time()[["elapsed"]]
                    ## invisible(fn_sampling_call(n_iterations_of_long_call, 3))
                    ## seconds_of_long_call <-  proc.time()[["elapsed"]] - time_before_long_call
                    long_call <-  fn_timed_call(function() fn_sampling_call(n_iterations_of_long_call, 3))
                    seconds_of_long_call <-  fn_seconds_of_call(long_call)
                    seconds_per_iteration <-  (seconds_of_long_call - seconds_of_short_call) /
                                              (n_iterations_of_long_call - n_iterations_of_short_call)
              }
              if (!(seconds_per_iteration > 0)) {
                    seconds_per_iteration <-  seconds_of_long_call / n_iterations_of_long_call
              }
              layouts$seconds_per_iteration[layout_index] <-  seconds_per_iteration
              layouts$seconds_per_draw_over_all_chains[layout_index] <-  seconds_per_iteration / n_chains
              layouts$timed_by_chains_seconds_of_the_sampler[layout_index] <-  timed_by_chains_seconds
              layouts$seconds_per_iteration_from_wall_seconds_of_the_calls[layout_index] <-
                    (long_call$seconds - short_call$seconds) /
                    (n_iterations_of_long_call - n_iterations_of_short_call)
              layouts$n_iterations_of_long_call[layout_index] <-  n_iterations_of_long_call
              layouts$seconds_spent_timing[layout_index] <-  proc.time()[["elapsed"]] - time_layout_start
              layouts$user_CPU_seconds_of_short_call[layout_index] <-  short_call$user_CPU_seconds
              layouts$kernel_CPU_seconds_of_short_call[layout_index] <-  short_call$kernel_CPU_seconds
              layouts$user_CPU_seconds_of_long_call[layout_index] <-  long_call$user_CPU_seconds
              layouts$kernel_CPU_seconds_of_long_call[layout_index] <-  long_call$kernel_CPU_seconds
              layouts$counter_snapshot_before_short_call[layout_index] <-  short_call$snapshot_before
              layouts$counter_snapshot_after_short_call[layout_index] <-  short_call$snapshot_after
              layouts$counter_snapshot_before_long_call[layout_index] <-  long_call$snapshot_before
              layouts$counter_snapshot_after_long_call[layout_index] <-  long_call$snapshot_after
              invisible(gc(verbose = FALSE))
        ##
        }
        return(layouts)
##
}






















