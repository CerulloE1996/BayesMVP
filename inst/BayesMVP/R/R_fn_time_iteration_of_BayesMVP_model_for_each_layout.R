##
## ==== R_fn_time_iteration_of_BayesMVP_model_for_each_layout.R ==================================================
##
## ---- Seconds per iteration of a BayesMVP model with manual gradients (Model_type "MVP", "MVOP", "LC_MVP" or
##      "LC_MVOP") for each layout of within-chain threads and chunks, in NicoStan's persistent burn-in worker -
##
## The same timing as NicoStan's fn_time_burnin_iteration_for_each_num_chunks() for Stan models (all chains
## advanced one diffusion-HMC iteration in parallel, at a tiny step size, eps = 1e-5, with a fixed trajectory
## length of L_main leapfrog steps (default 20), so that every layout does the same work; in rounds that each time
## every layout once), but for BayesMVP's own
## models: the number of chunks is that of the manual gradient (Model_args_ints[4]), which BayesMVP splits into
## chunks of a multiple of the SIMD width (calculate_chunk_sizes()), and the within-chain threads are those of
## the worker's TBB pool (n_chains * n_threads_WCP threads).
##
##
##
##
#### ---- the timing --------------------------------------------------------------------------------------------
#' Seconds per iteration of a BayesMVP model (manual gradients) for each layout of threads and chunks
#'
#' @param Model_type "MVP", "MVOP", "LC_MVP" or "LC_MVOP".
#' @param init_object The model's initialisation object (from BayesMVP::initialise_model() or the init_object of
#'   BayesMVP::MVP_model$new(); it holds Model_args_as_Rcpp_List, model_args_list, n_params_main and n_nuisance).
#' @param n_chains The chains advanced in parallel (e.g. the 4 burn-in chains, or the sampling chains).
#' @param layouts A data frame with columns n_threads_WCP (within-chain threads) and num_chunks.
#' @param n_rounds Rounds of timings (default 2; the median over the rounds is used).
#' @param n_timed_iterations The fewest iterations timed per layout and round (default 2; no untimed iteration).
#' @param minimum_seconds_per_timing The shortest time of each layout's timing in each round (default 0.1 s).
#' @param L_main,eps_main The leapfrog steps and step size (default 20 and 1e-5).
#' @param n_untimed_iterations_before_timing Iterations of each new worker run before its timed iterations, so
#'   that the timing does not include its first iteration (default 0: the first iteration is timed).
#' @return The data frame layouts with median_seconds_per_iteration (the median over the rounds) and
#'   seconds_spent_timing (the wall time of the layout's timings over all rounds, including the worker's set-up).
#' @export
fn_time_iteration_of_BayesMVP_model_for_each_layout <-  function( Model_type,
                                                                 init_object,
                                                                 n_chains,
                                                                 layouts,
                                                                 n_rounds                    = 2,
                                                                 n_timed_iterations          = 2,
                                                                 minimum_seconds_per_timing  = 0.1,
                                                                 L_main                      = 20,
                                                                 eps_main                    = 1e-5,
                                                                 n_untimed_iterations_before_timing  = 0
) {
        ##
        if (!(Model_type %in% c("MVP", "MVOP", "LC_MVP", "LC_MVOP"))) {
              stop(paste0("fn_time_iteration_of_BayesMVP_model_for_each_layout: Model_type must be \"MVP\", ",
                          "\"MVOP\", \"LC_MVP\" or \"LC_MVOP\" (BayesMVP's models with manual gradients); got: ",
                          paste(as.character(Model_type), collapse = ", ")))
        }
        if (!is.data.frame(layouts) || !all(c("n_threads_WCP", "num_chunks") %in% names(layouts))) {
              stop(paste0("fn_time_iteration_of_BayesMVP_model_for_each_layout: layouts must be a data frame ",
                          "with columns n_threads_WCP and num_chunks."))
        }
        n_params_main <-  init_object$n_params_main
        n_nuisance <-  init_object$n_nuisance
        y <-  init_object$model_args_list$y
        ##
        ## ---- tiny step size and L leapfrog steps, the default metric, and initial values close to zero on the
        ##      unconstrained scale (deterministic, so that R's random number stream is not touched):
        ##
        EHMC_args_as_Rcpp_List <-  init_EHMC_args_as_Rcpp_List( diffusion_HMC            = TRUE,
                                                                diffusion_HMC_integrator = "kick_flow_kick")
        EHMC_args_as_Rcpp_List$share_tau_ii_across_chains <-  TRUE
        EHMC_args_as_Rcpp_List$eps_main <-  eps_main
        EHMC_args_as_Rcpp_List$tau_main <-  L_main * eps_main
        EHMC_args_as_Rcpp_List$eps_us <-  eps_main
        EHMC_args_as_Rcpp_List$tau_us <-  L_main * eps_main
        EHMC_Metric_as_Rcpp_List <-  init_EHMC_Metric_as_Rcpp_List( n_params_main     = n_params_main,
                                                                    n_nuisance        = n_nuisance,
                                                                    metric_shape_main = "dense")
        fn_small_values <-  function(n_values) 0.005 * (((seq_len(n_values) * 7) %% 5) - 2)
        initial_main <-  matrix(fn_small_values(n_params_main * n_chains), nrow = n_params_main, ncol = n_chains)
        initial_nuisance <-  matrix(fn_small_values(n_nuisance * n_chains), nrow = n_nuisance, ncol = n_chains)
        ##
        ## (every iteration with the same trajectory length, L_main leapfrog steps, in every chain: the burn-in
        ##  draws each iteration's trajectory length uniformly between 0 and twice tau, so that a timing of one
        ##  iteration and a timing of several would otherwise time different numbers of leapfrog steps)
        tau_of_every_chain <-  rep(L_main * eps_main, n_chains)
        ##
        seconds <-  matrix(NA_real_, nrow = n_rounds, ncol = nrow(layouts))
        seconds_spent_timing <-  rep(0, nrow(layouts))
        for (round_index in seq_len(n_rounds)) {
              for (layout_index in seq_len(nrow(layouts))) {
                    time_layout_start <-  proc.time()[["elapsed"]]
                    n_threads_WCP <-  layouts$n_threads_WCP[layout_index]
                    Model_args_as_Rcpp_List <-  init_object$Model_args_as_Rcpp_List
                    Model_args_as_Rcpp_List$Model_args_ints[4] <-  layouts$num_chunks[layout_index]
                    RcppParallel::setThreadOptions(numThreads = n_chains * n_threads_WCP)
                    worker <-  fn_create_persistent_burnin_worker(
                                     n_threads_R = n_chains, partitioned_HMC_R = FALSE,
                                     diffusion_HMC_R = TRUE, Model_type_R = Model_type, sample_nuisance_R = TRUE,
                                     force_autodiff_R = FALSE, force_PartialLog_R = FALSE,
                                     multi_attempts_R = TRUE,
                                     y_Eigen_R = y,
                                     Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                     EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                                     EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                     n_threads_WCP = n_threads_WCP)
                    fn_persistent_burnin_update_adaptation(worker, EHMC_args_as_Rcpp_List,
                                                           EHMC_Metric_as_Rcpp_List)
                    fn_persistent_burnin_set_theta(worker, initial_main, initial_nuisance)
                    ## ## (one untimed iteration, then at least n_timed_iterations and at least
                    ## ##  minimum_seconds_per_timing of timed iterations)
                    ## fn_persistent_burnin_run_one_iter(worker, 1000 * round_index, 1)
                    ## (no untimed iteration: at least n_timed_iterations and at least minimum_seconds_per_timing
                    ##  of timed iterations, each of L_main leapfrog steps)
                    ## (n_untimed_iterations_before_timing untimed iterations first, so that the timing does not
                    ##  include the first iteration of the new worker)
                    for (untimed_iteration in seq_len(n_untimed_iterations_before_timing)) {
                          seed_of_untimed_iteration <-  1000 * round_index + 900 + untimed_iteration
                          fn_persistent_burnin_run_one_iter_tau_jitter(worker, seed_of_untimed_iteration,
                                                                       untimed_iteration, tau_of_every_chain,
                                                                       tau_of_every_chain, FALSE, FALSE)
                    }
                    start_time <-  proc.time()[["elapsed"]]
                    n_iterations_timed <-  0
                    while (n_iterations_timed < n_timed_iterations ||
                           proc.time()[["elapsed"]] - start_time < minimum_seconds_per_timing) {
                          n_iterations_timed <-  n_iterations_timed + 1
##                        fn_persistent_burnin_run_one_iter(worker, 1000 * round_index + 1 + n_iterations_timed,
##                                                          1 + n_iterations_timed)
                          seed_of_iteration <-  1000 * round_index + n_iterations_timed
                          # fn_persistent_burnin_run_one_iter_tau_jitter(worker, seed_of_iteration,
                          #                                              n_iterations_timed,
                          #                                              tau_of_every_chain, tau_of_every_chain,
                          #                                              FALSE, FALSE)
                          burnin_iter_outputs <-  fn_persistent_burnin_run_one_iter_tau_jitter(
                                                        worker, seed_of_iteration, n_iterations_timed,
                                                        tau_of_every_chain, tau_of_every_chain, FALSE, FALSE)
                          ## (as Paper 1's burn-in timings, R_fn_run_ps_1_burnin_benchmark.R: eps is tiny and L
                          ##  fixed, so nothing should diverge; a divergent transition makes the configuration
                          ##  invalid)
                          n_divs_this_iter <-
                                sum(burnin_iter_outputs$other_main_out_vector_all_chains_output_to_R[2, ])
                          if (n_divs_this_iter > 0) {
                                stop("Timing encountered ", n_divs_this_iter,
                                     " divergent transition(s) in timed iter ", n_iterations_timed,
                                     ". The configuration is invalid for a timing comparison.")
                          }
                    }
                    seconds[round_index, layout_index] <-  (proc.time()[["elapsed"]] - start_time) /
                                                           n_iterations_timed
                    rm(worker)
                    invisible(gc(verbose = FALSE))
                    seconds_spent_timing[layout_index] <-  seconds_spent_timing[layout_index] +
                                                           (proc.time()[["elapsed"]] - time_layout_start)
              }
        }
        layouts$median_seconds_per_iteration <-  apply(X = seconds, MARGIN = 2, FUN = stats::median)
        layouts$seconds_spent_timing <-  seconds_spent_timing
        return(layouts)
##
}






















