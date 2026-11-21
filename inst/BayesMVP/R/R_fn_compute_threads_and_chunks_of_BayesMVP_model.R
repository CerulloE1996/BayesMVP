##
##
#### ---- the sampling chunks of a BayesMVP model (manual gradients): Paper 1's equations, then its check ------
##
#' Sampling chunks of a BayesMVP model (manual gradients): Paper 1's equations, then Paper 1's check
#'
#' For BayesMVP's models with manual gradients: M_bytes/chain of the model and data, Paper 1's equations 7a-7h
#' for a manual gradient with the sampling layout and the CPUs of this process, then, when
#' time_Paper_1_check_of_num_chunks = TRUE, Paper 1's check (fn_check_num_chunks_by_doubling_and_halving()).
#' Every N_chunks is a multiple of the within-chain threads.
#'
#' @param Model_type The model: "MVP", "MVOP", "LC_MVP" or "LC_MVOP".
#' @param init_object The model's initialisation object.
#' @param n_chains_sampling Number of sampling chains run in parallel.
#' @param n_threads_WCP_sampling Within-chain threads of each sampling chain (default 1).
#' @param time_Paper_1_check_of_num_chunks TRUE (default): Paper 1's check of the equations' N_chunks, and the
#'   fastest value timed is used; FALSE: Paper 1's equations alone, nothing timed.
#' @param n_rounds,n_timed_iterations,minimum_seconds_per_timing The timings of Paper 1's check (see
#'   [fn_time_iteration_of_BayesMVP_model_for_each_layout()]; defaults 3, 3 and 0.2 s).
#' @return A list: num_chunks_sampling (the chunk count to use), num_chunks_sampling_from_Paper_1_equations,
#'   n_threads_WCP_sampling, num_chunks_check (the list of
#'   NicoStan::fn_check_num_chunks_by_doubling_and_halving(); NULL when not timed), M_bytes_per_chain, rule (the
#'   list of
#'   NicoStan::fn_compute_sampling_num_chunks_for_CPUs_of_this_process()), most_num_chunks and
#'   seconds_spent_choosing.
#' @export
fn_compute_sampling_num_chunks_for_BayesMVP_model <-  function( Model_type,
                                                               init_object,
                                                               n_chains_sampling,
                                                               n_threads_WCP_sampling            = 1,
                                                               time_Paper_1_check_of_num_chunks  = TRUE,
                                                               n_rounds                          = 3,
                                                               n_timed_iterations                = 3,
                                                               minimum_seconds_per_timing        = 0.2
) {
        ##
        time_choice_start <-  proc.time()[["elapsed"]]
        ##
        ## ---- requested N_chunks has no upper limit:
        ##
        N <-  nrow(init_object$model_args_list$y)
        vect_type <-  as.character(init_object$Model_args_as_Rcpp_List$Model_args_strings[1])
        SIMD_lanes <-  if (identical(vect_type, "AVX512")) 8 else
                       if (identical(vect_type, "AVX2")) 4 else
                       if (identical(vect_type, "AVX")) 2 else 1
        most_num_chunks <-  Inf
        ##
        ## ---- M_bytes/chain, and Paper 1's equations 7a-7h for a manual gradient and the sampling layout:
        ##
        ## (M_bytes/chain for a manual gradient: the workspace of the gradient, counted)
        M_bytes <-  NicoStan::fn_M_bytes_per_chain_counted_from_workspace_of_built_in_model(
                          Model_type  = Model_type,
                          N           = nrow(init_object$model_args_list$y),
                          n_tests     = ncol(init_object$model_args_list$y))
        rule <-  NicoStan::fn_compute_sampling_num_chunks_for_CPUs_of_this_process(
                       M_bytes_per_chain       = M_bytes$M_bytes_per_chain,
                       n_chains_sampling       = n_chains_sampling,
                       n_threads_WCP_sampling  = n_threads_WCP_sampling,
                       gradient_type           = "manual")
        num_chunks_sampling_from_Paper_1_equations <-  rule$N_chunks
        ##
        ## ---- Paper 1's check, timed with the sampling layout:
        ##
        num_chunks_check <-  NULL
        if (isTRUE(time_Paper_1_check_of_num_chunks) && (most_num_chunks > n_threads_WCP_sampling)) {
              num_chunks_check <-  NicoStan::fn_check_num_chunks_by_doubling_and_halving(
                                         num_chunks_start        = num_chunks_sampling_from_Paper_1_equations,
                                         fn_median_seconds_per_iteration_of_each_num_chunks =
                                               function(num_chunks_candidates) {
                                                     fn_time_iteration_of_BayesMVP_model_for_each_layout(
                                                       Model_type                  = Model_type,
                                                       init_object                 = init_object,
                                                       n_chains                    = n_chains_sampling,
                                                       layouts                     = data.frame(
                                                             n_threads_WCP  = n_threads_WCP_sampling,
                                                             num_chunks     = num_chunks_candidates),
                                                       n_rounds                    = n_rounds,
                                                       n_timed_iterations          = n_timed_iterations,
                                                       minimum_seconds_per_timing  = minimum_seconds_per_timing
                                                     )$median_seconds_per_iteration
                                               },
                                         fewest_num_chunks       = n_threads_WCP_sampling,
                                         most_num_chunks         = most_num_chunks,
                                         multiple_of_num_chunks  = n_threads_WCP_sampling,
                                         powers_of_two           = FALSE)
        }
        num_chunks_sampling <-  if (is.null(num_chunks_check)) num_chunks_sampling_from_Paper_1_equations else {
              num_chunks_check$num_chunks
        }
        return(list( num_chunks_sampling                          = num_chunks_sampling,
                     num_chunks_sampling_from_Paper_1_equations   = num_chunks_sampling_from_Paper_1_equations,
                     n_threads_WCP_sampling                       = n_threads_WCP_sampling,
                     num_chunks_check                             = num_chunks_check,
                     M_bytes_per_chain                            = M_bytes$M_bytes_per_chain,
                     rule                                         = rule,
                     most_num_chunks                              = most_num_chunks,
                     seconds_spent_choosing                       = proc.time()[["elapsed"]] - time_choice_start))
##
}
##
##
#### ---- the sampling layout of a BayesMVP model on the CPUs of this process: chains, WCP threads, N_chunks ----
##
#' The sampling layout of a BayesMVP model (manual gradients) on the CPUs of this process: the chains, the
#' within-chain threads (WCP) and N_chunks, from Paper 1's equations and Paper 1's check
#'
#' For BayesMVP's models with manual gradients, as
#' NicoStan::fn_compute_sampling_n_chains_n_threads_WCP_and_num_chunks_for_Stan_model() for Stan models: every
#' CPU of this process is used (floor(CPUs / threads per chain) chains, or the n_chains_sampling given). Paper 1's
#' equations give the layout; when it is one chain per CPU with N_chunks = 1, it is used without a check.
#' Otherwise Paper 1's check (NicoStan::fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving())
#' searches the threads per chain and N_chunks together, the layouts compared by their seconds per draw over all
#' chains. Every N_chunks is a power of two, at least one chunk per thread.
#'
#' @param Model_type The model: "MVP", "MVOP", "LC_MVP" or "LC_MVOP".
#' @param init_object The model's initialisation object.
#' @param n_chains_sampling The sampling chains, when they are given (default NULL: floor(CPUs / threads per
#'   chain), every CPU used).
#' @param time_Paper_1_check TRUE (default): Paper 1's check from the equations' layout; FALSE: the equations'
#'   layout, nothing timed.
#' @param n_timed_iterations Not used.
#' @param minimum_seconds_per_timing The shortest time of the iterations of each layout's long timing call
#'   (default 0.25 s; see [fn_time_sampling_of_BayesMVP_model_for_each_layout()]).
#' @param leapfrog_steps_per_timed_iteration The leapfrog steps of each timed iteration (default 10).
#' @param fraction_faster_needed_to_move A neighbour must be faster by more than this fraction for the check to
#'   move to it (default 0.03; see
#'   NicoStan::fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving()).
#' @return A list: n_chains_sampling, n_threads_WCP_sampling and num_chunks_sampling (the layout to use);
#'   n_chains_sampling_from_equations, n_threads_WCP_sampling_from_equations and
#'   num_chunks_sampling_from_equations; M_bytes_per_chain_fits_in_cache_at_one_chain_per_CPU (TRUE: the
#'   equations' N_chunks is 1, and no check); timings (a data frame: n_chains_sampling, n_threads_WCP_sampling,
#'   num_chunks, mean_seconds_per_iteration, mean_seconds_per_draw_over_all_chains and n_timings; NULL when not
#'   timed); n_timings, M_bytes_per_chain, rule, most_num_chunks, n_CPUs and seconds_spent_choosing.
#' @export
fn_compute_sampling_n_chains_n_threads_WCP_and_num_chunks_for_BayesMVP_model <-  function(
        Model_type,
        init_object,
        n_chains_sampling                   = NULL,
        time_Paper_1_check                  = TRUE,
        n_timed_iterations                  = 1,
        minimum_seconds_per_timing          = 0.25,
        leapfrog_steps_per_timed_iteration  = 10,
        fraction_faster_needed_to_move      = 0.03
) {
        ##
        time_choice_start <-  proc.time()[["elapsed"]]
        ##
        ## ---- the CPUs of this process, and the chains (the chains given, or every CPU used):
        ##
        CPU_ids <-  NicoStan::fn_read_L3_cache_topology_of_available_CPUs()$CPU_ids_available
        n_CPUs <-  length(CPU_ids)
        n_chains_given <-  n_chains_sampling
        fn_n_chains_of_threads_per_chain <-  function(threads_per_chain) {
              if (!is.null(n_chains_given)) return(n_chains_given)
              return(floor(n_CPUs / threads_per_chain))
        }
        ##
        ## ---- requested N_chunks has no upper limit; threads are limited by available CPUs only:
        ##
        N <-  nrow(init_object$model_args_list$y)
        vect_type <-  as.character(init_object$Model_args_as_Rcpp_List$Model_args_strings[1])
        SIMD_lanes <-  if (identical(vect_type, "AVX512")) 8 else
                       if (identical(vect_type, "AVX2")) 4 else
                       if (identical(vect_type, "AVX")) 2 else 1
        most_num_chunks <-  Inf
        most_threads_per_chain <-  if (is.null(n_chains_given)) n_CPUs else max(1, floor(n_CPUs / n_chains_given))
        ##
        ## ---- M_bytes/chain, and Paper 1's equations for a manual gradient:
        ##
        ## (M_bytes/chain for a manual gradient: the workspace of the gradient, counted)
        M_bytes <-  NicoStan::fn_M_bytes_per_chain_counted_from_workspace_of_built_in_model(
                          Model_type  = Model_type,
                          N           = nrow(init_object$model_args_list$y),
                          n_tests     = ncol(init_object$model_args_list$y))
        M_bytes_per_chain <-  M_bytes$M_bytes_per_chain
        ##
        ## (one chain per CPU with 1 thread, or the most threads per chain the CPUs allow for the chains given)
        threads_per_chain_from_equations <-  1
        if (!is.null(n_chains_given)) {
              threads_per_chain_from_equations <-  2^floor(log2(most_threads_per_chain))
        }
        rule <-  NicoStan::fn_compute_sampling_num_chunks_for_CPUs_of_this_process(
                       M_bytes_per_chain       = M_bytes_per_chain,
                       n_chains_sampling       = fn_n_chains_of_threads_per_chain(
                                                       threads_per_chain_from_equations),
                       n_threads_WCP_sampling  = threads_per_chain_from_equations,
                       gradient_type           = "manual")
        n_chains_sampling_from_equations <-  fn_n_chains_of_threads_per_chain(threads_per_chain_from_equations)
        n_threads_WCP_sampling_from_equations <-  threads_per_chain_from_equations
        ## (N_chunks: the rule's N_chunks, rounded up to a power of two, at least one chunk per thread)
        num_chunks_sampling_from_equations <-  2^ceiling(log2(rule$N_chunks))
        num_chunks_sampling_from_equations <-  max(threads_per_chain_from_equations,
                                                    num_chunks_sampling_from_equations)
        ## (one chain per CPU and N_chunks = 1: the layout is not checked)
        M_bytes_per_chain_fits_in_cache_at_one_chain_per_CPU <-  (threads_per_chain_from_equations == 1) &&
                                                                 (num_chunks_sampling_from_equations <= 1)
        ##
        ## ---- Paper 1's check: the threads per chain and N_chunks searched together:
        ##
        n_threads_WCP_sampling <-  n_threads_WCP_sampling_from_equations
        num_chunks_sampling <-  num_chunks_sampling_from_equations
        timings <-  NULL
        n_timings <-  0
        if (isTRUE(time_Paper_1_check) && !M_bytes_per_chain_fits_in_cache_at_one_chain_per_CPU &&
            (most_num_chunks > 1)) {
              ## (seconds per draw over all chains: seconds per iteration / number of chains)
              fn_seconds_per_draw_of_each_layout <-  function(layouts) {
                    seconds_per_draw <-  numeric(nrow(layouts))
                    for (index in seq_len(nrow(layouts))) {
                          n_chains <-  fn_n_chains_of_threads_per_chain(layouts$threads_per_chain[index])
                          ## (timed in the sampler of the sampling phase)
                          layout_timed <-  data.frame( n_chains       = n_chains,
                                                       n_threads_WCP  = layouts$threads_per_chain[index],
                                                       num_chunks     = layouts$num_chunks[index])
                          seconds_per_iteration <-  fn_time_sampling_of_BayesMVP_model_for_each_layout(
                                Model_type                  = Model_type,
                                init_object                 = init_object,
                                layouts                     = layout_timed,
                                minimum_seconds_per_timing  = minimum_seconds_per_timing,
                                L_main                      = leapfrog_steps_per_timed_iteration
                          )$seconds_per_iteration[1]
                          seconds_per_draw[index] <-  seconds_per_iteration / n_chains
                    }
                    return(seconds_per_draw)
              }
              layout_search <-  tryCatch(
                    expr  = NicoStan::fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving(
                                  threads_per_chain_start         = n_threads_WCP_sampling_from_equations,
                                  num_chunks_start                = num_chunks_sampling_from_equations,
                                  fn_seconds_of_each_layout       = fn_seconds_per_draw_of_each_layout,
                                  most_threads_per_chain          = most_threads_per_chain,
                                  most_num_chunks                 = most_num_chunks,
                                  fraction_faster_needed_to_move  = fraction_faster_needed_to_move,
                                  also_time_2_threads_per_chain_at_2_chunks_when_the_equations_say_no_chunking =
                                        FALSE),
                    error = function(condition) {
                          message(paste0("Sampling layout not checked (", conditionMessage(condition),
                                         "); Paper 1's equations are used."))
                          NULL
                    })
              if (!is.null(layout_search)) {
                    n_threads_WCP_sampling <-  layout_search$threads_per_chain
                    num_chunks_sampling <-  layout_search$num_chunks
                    n_timings <-  layout_search$n_timings
                    n_chains_of_each_layout_timed <-  vapply(X = layout_search$timings$threads_per_chain,
                                                             FUN = fn_n_chains_of_threads_per_chain,
                                                             FUN.VALUE = 0)
                    timings <-  data.frame(
                          n_chains_sampling                       = n_chains_of_each_layout_timed,
                          n_threads_WCP_sampling                  = layout_search$timings$threads_per_chain,
                          num_chunks                              = layout_search$timings$num_chunks,
                          mean_seconds_per_iteration              = layout_search$timings$mean_seconds *
                                                                    n_chains_of_each_layout_timed,
                          mean_seconds_per_draw_over_all_chains   = layout_search$timings$mean_seconds,
                          n_timings                               = layout_search$timings$n_timings)
              }
        }
        n_chains_sampling <-  fn_n_chains_of_threads_per_chain(n_threads_WCP_sampling)
        return(list( n_chains_sampling                                      = n_chains_sampling,
                     n_threads_WCP_sampling                                 = n_threads_WCP_sampling,
                     num_chunks_sampling                                    = num_chunks_sampling,
                     n_chains_sampling_from_equations                       = n_chains_sampling_from_equations,
                     n_threads_WCP_sampling_from_equations                  =
                           n_threads_WCP_sampling_from_equations,
                     num_chunks_sampling_from_equations                     = num_chunks_sampling_from_equations,
                     M_bytes_per_chain_fits_in_cache_at_one_chain_per_CPU   =
                           M_bytes_per_chain_fits_in_cache_at_one_chain_per_CPU,
                     timings                                                = timings,
                     n_timings                                              = n_timings,
                     M_bytes_per_chain                                      = M_bytes_per_chain,
                     rule                                                   = rule,
                     most_num_chunks                                        = most_num_chunks,
                     n_CPUs                                                 = n_CPUs,
                     seconds_spent_choosing                                 = proc.time()[["elapsed"]] -
                                                                              time_choice_start))
##
}
##
##
#### ---- the burn-in threads and chunks of a BayesMVP model (manual gradients): Paper 1's equations, timed ----
##
##
##
#' Burn-in within-chain threads and chunks of a BayesMVP model (manual gradients): Paper 1's equations, then
#' Paper 1's check
#'
#' For BayesMVP's models with manual gradients, as
#' NicoStan::fn_compute_burnin_n_threads_WCP_and_num_chunks_for_Stan_model() for Stan models: the hardware
#' candidates for the threads per chain (1, the cores and the hardware threads per L3 cache), each with the
#' equations' N_chunks, are timed, and Paper 1's check then searches the threads per chain and N_chunks together
#' from the fastest. Every N_chunks is a power of two, at least one chunk per thread.
#'
#' @param Model_type The model: "MVP", "MVOP", "LC_MVP" or "LC_MVOP".
#' @param init_object The model's initialisation object.
#' @param n_chains_burnin Number of burn-in chains run in parallel (default 4).
#' @param time_Paper_1_check_of_num_chunks TRUE (default): the hardware candidates are timed and Paper 1's check
#'   (of the threads per chain and N_chunks together) follows; FALSE: nothing timed, and the threads per chain of
#'   Paper 1's earlier burn-in rule (NicoStan::fn_compute_burnin_n_threads_WCP_and_num_chunks()) with the
#'   equations' N_chunks are used.
#' @param n_timed_iterations,minimum_seconds_per_timing Each timing: at least n_timed_iterations iterations and at
#'   least minimum_seconds_per_timing seconds of them (defaults 1 and 0.25 s).
#' @param n_untimed_iterations_before_timing Iterations of each new worker run before its timed iterations
#'   (default 1), so that the timing does not count the first touch of its memory.
#' @param leapfrog_steps_per_timed_iteration The leapfrog steps of each timed iteration (default 10).
#' @param fraction_faster_needed_to_move A neighbour must be faster by more than this fraction for the check to
#'   move to it (default 0.03; see
#'   NicoStan::fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving()).
#' @return A list: n_threads_WCP_burnin and num_chunks_burnin (the layout to use);
#'   n_threads_WCP_burnin_from_Paper_1_equations and num_chunks_burnin_from_Paper_1_equations (the threads per
#'   chain of Paper 1's earlier burn-in rule, and the equations' N_chunks at it); burnin_hardware_candidates_timed
#'   (a data frame: threads_per_chain, num_chunks and seconds; seconds is absent when not timed),
#'   n_threads_WCP_burnin_of_fastest_hardware_candidate and num_chunks_burnin_of_fastest_hardware_candidate (the
#'   burn-in equation's layout; NA when not timed); configuration_timings (every layout timed: n_threads_WCP,
#'   num_chunks, mean_seconds_per_iteration and n_timings; NULL when not timed); num_chunks_check (the list of
#'   NicoStan::fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving(); NULL when not timed);
#'   M_bytes_per_chain, most_num_chunks and seconds_spent_choosing.
#' @export
fn_compute_burnin_n_threads_WCP_and_num_chunks_for_BayesMVP_model <-  function(
        Model_type,
        init_object,
        n_chains_burnin                     = 4,
        time_Paper_1_check_of_num_chunks    = TRUE,
        n_timed_iterations                  = 1,
        minimum_seconds_per_timing          = 0.25,
        n_untimed_iterations_before_timing  = 1,
        leapfrog_steps_per_timed_iteration  = 10,
        fraction_faster_needed_to_move      = 0.03
) {
        ##
        time_choice_start <-  proc.time()[["elapsed"]]
        ##
        ## ---- requested N_chunks has no upper limit:
        ##
        N <-  nrow(init_object$model_args_list$y)
        vect_type <-  as.character(init_object$Model_args_as_Rcpp_List$Model_args_strings[1])
        SIMD_lanes <-  if (identical(vect_type, "AVX512")) 8 else
                       if (identical(vect_type, "AVX2")) 4 else
                       if (identical(vect_type, "AVX")) 2 else 1
        most_num_chunks <-  Inf
        ##
        ## ---- M_bytes/chain, and the threads per chain of Paper 1's burn-in rule:
        ##
        ## (M_bytes/chain for a manual gradient: the workspace of the gradient, counted)
        M_bytes <-  NicoStan::fn_M_bytes_per_chain_counted_from_workspace_of_built_in_model(
                          Model_type  = Model_type,
                          N           = nrow(init_object$model_args_list$y),
                          n_tests     = ncol(init_object$model_args_list$y))
        burnin_rule <-  NicoStan::fn_compute_burnin_n_threads_WCP_and_num_chunks(
                              M_bytes_per_chain  = M_bytes$M_bytes_per_chain,
                              N_units            = N,
                              n_chains_burnin    = n_chains_burnin)
        most_threads_per_chain <-  burnin_rule$n_threads_per_chain_from_hardware
        n_threads_WCP_from_Paper_1_burnin_rule <-  min(most_threads_per_chain, burnin_rule$n_threads_WCP_burnin)
        ##
        ## ---- the N_chunks of Paper 1's equations, at least 2 chunks per thread, as a power of two:
        ##
        fn_num_chunks_of_equations <-  function(n_threads_WCP) {
              num_chunks <-  NicoStan::fn_compute_sampling_num_chunks_for_CPUs_of_this_process(
                                   M_bytes_per_chain       = M_bytes$M_bytes_per_chain,
                                   n_chains_sampling       = n_chains_burnin,
                                   n_threads_WCP_sampling  = n_threads_WCP,
                                   gradient_type           = "manual")$N_chunks
              num_chunks <-  max(num_chunks, 2 * n_threads_WCP)
              return(2^ceiling(log2(num_chunks)))
        }
        num_chunks_burnin_from_Paper_1_equations <-
              fn_num_chunks_of_equations(n_threads_WCP_from_Paper_1_burnin_rule)
        ##
        ## ---- the hardware candidates of Paper 1's burn-in equation, each with the equations' N_chunks:
        ##
        threads_per_core <-  NicoStan::fn_read_hardware_threads_per_core()
        threads_per_chain_of_hardware <-  c(1,
                                            burnin_rule$n_threads_per_chain_from_L3_cache / threads_per_core,
                                            burnin_rule$n_threads_per_chain_from_L3_cache)
        threads_per_chain_of_hardware <-  2^floor(log2(pmin(most_threads_per_chain,
                                                             pmax(1, threads_per_chain_of_hardware))))
        burnin_hardware_candidates <-  data.frame(threads_per_chain = sort(unique(threads_per_chain_of_hardware)))
        burnin_hardware_candidates$num_chunks <-  vapply(
                                                        X          = burnin_hardware_candidates$threads_per_chain,
                                                        FUN        = fn_num_chunks_of_equations,
                                                        FUN.VALUE  = 0)
        ##
        ## ---- the hardware candidates timed, then Paper 1's check from the fastest of them:
        ##
        n_threads_WCP_burnin <-  n_threads_WCP_from_Paper_1_burnin_rule
        num_chunks_burnin <-  num_chunks_burnin_from_Paper_1_equations
        n_threads_WCP_burnin_of_fastest_hardware_candidate <-  NA
        num_chunks_burnin_of_fastest_hardware_candidate <-  NA
        configuration_timings <-  NULL
        num_chunks_check <-  NULL
        if (isTRUE(time_Paper_1_check_of_num_chunks)) {
              ## (the seconds per burn-in iteration of each layout, each timed once)
              fn_seconds_per_burnin_iteration_of_each_layout <-  function(layouts) {
                    seconds_per_iteration <-  numeric(nrow(layouts))
                    for (index in seq_len(nrow(layouts))) {
                          layout_timed <-  data.frame( n_threads_WCP  = layouts$threads_per_chain[index],
                                                       num_chunks     = layouts$num_chunks[index])
                          seconds_per_iteration[index] <-  fn_time_iteration_of_BayesMVP_model_for_each_layout(
                                Model_type                          = Model_type,
                                init_object                         = init_object,
                                n_chains                            = n_chains_burnin,
                                layouts                             = layout_timed,
                                n_rounds                            = 1,
                                n_timed_iterations                  = n_timed_iterations,
                                minimum_seconds_per_timing          = minimum_seconds_per_timing,
                                L_main                              = leapfrog_steps_per_timed_iteration,
                                n_untimed_iterations_before_timing  = n_untimed_iterations_before_timing
                          )$median_seconds_per_iteration[1]
                    }
                    return(seconds_per_iteration)
              }
              layout_search <-  tryCatch(
                    expr  = {
                          burnin_hardware_candidates$seconds <-  fn_seconds_per_burnin_iteration_of_each_layout(
                                                                      burnin_hardware_candidates)
                          fastest_candidate <-  burnin_hardware_candidates[
                                                      which.min(burnin_hardware_candidates$seconds), ]
                          NicoStan::fn_search_threads_per_chain_and_num_chunks_by_doubling_and_halving(
                                threads_per_chain_start         = fastest_candidate$threads_per_chain,
                                num_chunks_start                = fastest_candidate$num_chunks,
                                fn_seconds_of_each_layout       = fn_seconds_per_burnin_iteration_of_each_layout,
                                most_threads_per_chain          = most_threads_per_chain,
                                most_num_chunks                 = most_num_chunks,
                                fraction_faster_needed_to_move  = fraction_faster_needed_to_move,
                                timings_already_made            = burnin_hardware_candidates,
                                also_time_2_threads_per_chain_at_2_chunks_when_the_equations_say_no_chunking =
                                      TRUE)
                    },
                    error = function(condition) {
                          message(paste0("Burn-in layout not checked (", conditionMessage(condition),
                                         "); Paper 1's equations are used."))
                          NULL
                    })
              if (!is.null(burnin_hardware_candidates$seconds)) {
                    fastest_hardware_candidate <-  burnin_hardware_candidates[
                                                         which.min(burnin_hardware_candidates$seconds), ]
                    n_threads_WCP_burnin_of_fastest_hardware_candidate <-
                          fastest_hardware_candidate$threads_per_chain
                    num_chunks_burnin_of_fastest_hardware_candidate <-  fastest_hardware_candidate$num_chunks
              }
              if (!is.null(layout_search)) {
                    n_threads_WCP_burnin <-  layout_search$threads_per_chain
                    num_chunks_burnin <-  layout_search$num_chunks
                    num_chunks_check <-  layout_search
                    configuration_timings <-  data.frame(
                          n_threads_WCP               = layout_search$timings$threads_per_chain,
                          num_chunks                  = layout_search$timings$num_chunks,
                          mean_seconds_per_iteration  = layout_search$timings$mean_seconds,
                          n_timings                   = layout_search$timings$n_timings)
              }
        }
        return(list( n_threads_WCP_burnin                                = n_threads_WCP_burnin,
                     num_chunks_burnin                                   = num_chunks_burnin,
                     n_threads_WCP_burnin_from_Paper_1_equations         = n_threads_WCP_from_Paper_1_burnin_rule,
                     num_chunks_burnin_from_Paper_1_equations            =
                           num_chunks_burnin_from_Paper_1_equations,
                     burnin_hardware_candidates_timed                    = burnin_hardware_candidates,
                     n_threads_WCP_burnin_of_fastest_hardware_candidate  =
                           n_threads_WCP_burnin_of_fastest_hardware_candidate,
                     num_chunks_burnin_of_fastest_hardware_candidate     =
                           num_chunks_burnin_of_fastest_hardware_candidate,
                     configuration_timings                               = configuration_timings,
                     num_chunks_check                                    = num_chunks_check,
                     M_bytes_per_chain                                   = M_bytes$M_bytes_per_chain,
                     most_num_chunks                                     = most_num_chunks,
                     seconds_spent_choosing                              = proc.time()[["elapsed"]] -
                                                                           time_choice_start))
##
}






















