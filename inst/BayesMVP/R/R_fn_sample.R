

## R_fn_sample.R


# debug = TRUE
# ##
# stream = MCMC_seed
# ##
# init_object = init_object
# ##
# n_chains_burnin = n_chains_burnin
# init_lists_per_chain = init_lists_per_chain
# ##
# parallel_method = "RcppParallel"
# ## parallel_method = "OpenMP"
# ##
# Stan_data_list = NULL
# model_args_list = model_args_list
# ##
# sample_nuisance = TRUE
# n_nuisance_override = NULL
# ##
# seed = MCMC_seed ## fixed MCMC seed
# ##
# n_burnin = n_burnin
# n_adapt = NULL
# gap = NULL
# ##
# n_chains_sampling = n_chains_sampling
# n_superchains = n_chains_sampling
# ##
# n_iter = n_iter
# ##
# adapt_delta = 0.80
# ##
# learning_rate = learning_rate
# ##s
# tau_mult = 1.60
# tau_initial = 1.0*6.283185
# ##
# manual_tau = FALSE
# tau_if_manual = 3.0
# ##
# burnin_algorithm = "ChESSR"
# ##
# diffusion_HMC = FALSE
# partitioned_HMC = FALSE
# ##
# clip_iter = 50
# clip_iter_tau = 50
# ##
# n_refresh = 100
# use_proposed = TRUE
# ##
# beta1_adam = 0.00
# beta2_adam = 0.95
# eps_adam = 1e-8
# ##
# force_autodiff = FALSE
# force_PartialLog = FALSE
# multi_attempts = FALSE
# ##
# force_autodiff_for_metric = TRUE
# force_PartialLog_for_metric = FALSE
# force_multi_attempts_for_metric = FALSE
# ##
# vect_type = "AVX512"
# ##
# Phi_type = "Phi"
# inv_Phi_type = "inv_Phi"
# ##
# # metric_type_main = "Hessian",
# # metric_shape_main = "dense",
# # ratio_M_main = 0.50,
# # interval_width_main = round(n_burnin/10),
# ##
# metric_type_main = "Empirical"
# metric_shape_main = "diag"
# ## ratio_M_main = 0.25
# ratio_M_main = 0.50
# interval_width_main = round(n_burnin/10)
# ##
# metric_type_nuisance = "Empirical"
# metric_shape_nuisance = "diag"
# ratio_M_nuisance = 0.25
# interval_width_nuisance = round(n_burnin/10)
# ##
# max_tau_main = 25.0
# max_tau_nuisance = 25.0
# ##
# max_eps_main = 0.75
# max_eps_nuisance = 0.75
# ##
# max_L = 512
# n_nuisance_to_track = NULL
##
# M_decay_type = "inverse"
# M_decay_power = 0.50
# M_decay_scale <- n_adapt / 100


#' R_fn_sample_model
#' @details Set \code{options(BayesMVP_autodiff_fallback = TRUE)} before a fit
#'   to enable an autodiff lp/gradient fallback for built-in models with multi_attempts = TRUE.
#'   If unset, it defaults to FALSE for MVP/LC-MVP/MVOP/LC-MVOP and preserves the existing
#'   TRUE default for latent_trait. Standard-scale and partial-log evaluations are tried first;
#'   autodiff is used only if both return nonfinite output, at the same parameter values.
#'   The latent_trait model skips its known-unfinished partial-log path and uses standard -> autodiff.
#'   This does not retry trajectories or rescue finite energy-error divergences. The option
#'   is frozen for the fit and returned as autodiff_fallback. For separate R worker
#'   processes, set the option in the process which calls this function.
#' @param debug_burnin_timing Collect per-iteration native timings and per-chain elapsed times and integrator step counts during burn-in.
#' @param diffusion_HMC_integrator Joint diffusion integrator: "kick_flow_kick" (default) or "flow_kick_flow".
#' @param reorder_cols_MVP Run a short pre-burnin and then re-fit with the test/outcome
#'   columns permuted into the Dissmann (2013) pair-first order of the estimated
#'   correlation matrix. Supported for Model_type = "LC_MVP" and for USER-SUPPLIED
#'   Stan models (Model_type = "Stan").
#'
#'   For Model_type = "Stan" only \code{y} is permuted - BayesMVP cannot know which of
#'   an external model's other data objects are test-indexed - so the option is REFUSED
#'   (with a large warning banner, after which sampling continues in the original column
#'   order) unless all of the following hold: the outcome matrix in your Stan data is
#'   called \code{y} and is N x n_tests with one column per test; the model exposes a
#'   correlation matrix called \code{Omega} whose dimension matches; and NO other entry
#'   of the Stan data is shaped like n_tests. That last condition is the strict one, and
#'   deliberately so: permuting \code{y} while leaving a per-test prior, covariate or
#'   flag in place would pair each test's responses with a different test's metadata,
#'   silently. Note also that test-indexed OUTPUTS are not un-permuted (slot j corresponds
#'   to original test \code{test_perm[j]}; the permutation and its inverse are returned as
#'   \code{test_perm} / \code{test_inv_perm}), and that user-supplied test-indexed initial
#'   values are left alone. This has only been tested to work and/or be beneficial for
#'   multivariate probit-based models.
#' @param burnin_TBB_pool_equals_n_chains NULL selects TRUE for built-in models, whose within-chain work uses OpenMP.
#'   External Stan models always use FALSE, including when TRUE is supplied, so their nested TBB work can use
#'   n_chains_burnin * n_threads_WCP_burnin threads. An explicit FALSE remains available for built-in models.
#' @export
R_fn_sample_model  <-    function(      debug = FALSE,
                                        ##
                                        stream = NULL,
                                        ##
                                        init_object, ## This may be updated within this function
                                        ##
                                        # Model_type, ## removed this
                                        ##
                                        n_chains_burnin,  ## here as may be updated
                                        init_lists_per_chain, ## This may be updated within this function
                                        ##
                                        parallel_method,
                                        ##
                                        Stan_data_list,  ## here as may be updated
                                        model_args_list, ## here as may be updated
                                        ##
                                        sample_nuisance,  ## here as may be updated
                                        n_nuisance_override = NULL,
                                        ##
                                        seed,
                                        n_burnin,
                                        n_adapt,
                                        gap,
                                        ##
                                        n_chains_sampling,
                                        n_superchains,
                                        nuisance_jitter_scale = 0.25,
                                        ##
                                        n_iter,
                                        ##
                                        adapt_delta,
                                        ##
                                        learning_rate,
                                        ##
                                        tau_mult,
                                        tau_initial,
                                        ##
                                        manual_tau,
                                        tau_if_manual,
                                        ##
                                        ## Read tau_if_manual as a LEAPFROG-STEP count instead of an integration time, so a
                                        ## fixed-L sweep stays at the requested L while eps keeps adapting. See
                                        ## R_fn_init_and_run_burnin_CHESS for the full note.
                                        tau_if_manual_in_L_units = NULL,
                                        ##
                                        ## Trajectory-length objective: "KE" (historical kinetic-energy change) or "ChEES"
                                        ## (the position-based criterion of Hoffman et al. 2021).
                                        tau_objective = NULL,
                                        ##
                                        ## Weight the per-chain tau gradients by the Metropolis acceptance probability and
                                        ## aggregate by the acceptance-weighted mean rather than the median.
                                        tau_weight_by_p_jump = NULL,
                                        ##
                                        ## Burn-in tau ramp before the adaptation takes over: "original" (default; the
                                        ## ramp used up to 2026-09-17 12:17 BST) or "staged" (5/10/20 leapfrog steps then
                                        ## pi/8 .. pi, floored at 20 * eps). See init_and_run_burnin_ChESSR.
                                        tau_ramp = NULL,
                                        ##
                                        ## eps at the ChEES handover (iteration gap) of the MAIN burn-in: TRUE (default) re-initialises
                                        ## it with find_initial_eps; FALSE keeps the eps adapted up to gap. See init_and_run_burnin_ChESSR.
                                        eps_reinit_at_ChEES_handover = NULL,
                                        ##
                                        ## Centre of the nuisance Gaussian rotation in the MAIN burn-in: "running_mean_frozen" (default;
                                        ## running mean, frozen from theta_hat_us_freeze_iter so eps is tuned against the kernel sampling
                                        ## uses), "running_mean" (never frozen - the pre-2026-09-18 behaviour, which tunes eps too big)
                                        ## or "zero" (the Feb 2026 code). See init_and_run_burnin_ChESSR.
                                        theta_hat_us_rule = NULL,
                                        theta_hat_us_freeze_iter = NULL,
                                        burnin_schedule = "automatic",
                                        metric_adaptation_end_iter = NULL,
                                        ##
                                        ## Burn-in speed-ups (NULL = unchanged behaviour):
                                        ## Historical early-stop option below is retired; n_burnin always specifies the full main burn-in.
                                        ##   burnin_post_adapt_iter - non-adapting iterations kept after n_adapt in the MAIN burn-in (NULL = all of
                                        ##                            them, i.e. n_burnin - n_adapt). eps/metric/tau are frozen there. e.g. 5.
                                        ##   pre_burnin_n_iter      - length of the test-order pre-burnin (NULL = 125), with its schedule scaled to match.
                                        pre_burnin_n_iter = NULL,
                                        ##   pre_burnin_L           - fix the pre-burnin trajectory at this many leapfrog steps (NULL = tau fixed at 3.0,
                                        ##                            or 1.5 with "pooled"). A fixed tau costs L = tau / eps, i.e. 60-100 steps while eps is
                                        ##                            still small early on; the pre-burnin only needs rough correlations. e.g. 10.
                                        pre_burnin_L = NULL,
                                        ##   share_tau_ii_across_chains_in_burnin - TRUE = one tau_ii per iteration shared by all MAIN burn-in chains
                                        ##                                          (momenta stay per-chain); NULL/FALSE = per-chain tau_ii (as before).
                                        share_tau_ii_across_chains_in_burnin = NULL,
                                        ##   burnin_TBB_pool_equals_n_chains      - NULL defaults to TRUE for built-in models (separate OpenMP WCP teams).
                                        ##                                          Stan models FORCE FALSE, even if TRUE is supplied: nested TBB work
                                        ##                                          shares the n_threads_WCP_burnin * n_chains_burnin thread budget.
                                        burnin_TBB_pool_equals_n_chains = NULL,
                                        ##   store_log_lik_trace                  - FALSE = sampling does not keep the N x n_iter per-chain log-lik trace
                                        ##                                          (720 MB at N = 10,000, 180 chains, 50 iterations, zero-filled and copied
                                        ##                                          into R). Only create_summary_and_traces(save_log_lik_trace = TRUE) reads
                                        ##                                          it. NULL/TRUE = keep it (as before). RcppParallel sampling only.
                                        store_log_lik_trace = NULL,
                                        ##
                                        burnin_algorithm = "ChESSR",
                                        diffusion_HMC,
                                        partitioned_HMC,
                                        ##
                                        clip_iter,
                                        clip_iter_tau,
                                        ##
                                        n_refresh = NULL,
                                        use_proposed = TRUE,
                                        ##
                                        beta1_adam = 0.00,
                                        beta2_adam = 0.95,
                                        eps_adam = 1e-8,
                                        ##
                                        force_autodiff,
                                        force_PartialLog,
                                        multi_attempts,
                                        ##
                                        force_autodiff_for_metric = TRUE,
                                        force_PartialLog_for_metric = FALSE,
                                        force_multi_attempts_for_metric = FALSE,
                                        ##
                                        vect_type,
                                        Phi_type,
                                        inv_Phi_type,
                                        ##
                                        metric_type_main,
                                        metric_shape_main,
                                        ratio_M_main,
                                        interval_width_main,
                                        ##
                                        M_decay_type,
                                        M_decay_power,
                                        M_decay_scale,
                                        ##
                                        metric_type_nuisance,
                                        metric_shape_nuisance,
                                        ratio_M_nuisance,
                                        interval_width_nuisance,
                                        ##
                                        max_tau_main = 25.0,
                                        max_tau_nuisance = 25.0,
                                        ##
                                        max_eps_main,
                                        max_eps_nuisance,
                                        ##
                                        learning_rate_initial = NULL,
                                        learning_rate_initial_iter = NULL,
                                        ##
                                        ## (eps warm start: OFF unless asked for - see eps_initial below)
                                        eps_initial = NULL,
                                        eps_initial_iter = NULL,
                                        ##
                                        max_L,
                                        ##
                                        n_nuisance_to_track = NULL,
                                        ##
                                        use_disk,
                                        use_disk_path = "/tmp/hmc_traces",
                                        ##
                                        n_threads_WCP_burnin,
                                        n_threads_WCP_sampling,
                                        ##
                                        reorder_cols_MVP,
                                        ##
                                        metric_estimator = "pooled",
                                        ##
                                        ## Fix the test (column) order instead of estimating it from the pre-burnin, e.g.
                                        ## c(5, 4, 6, 1, 3, 2). Only used when reorder_cols_MVP = TRUE; the pre-burnin still
                                        ## runs (so a run is otherwise identical), only its estimated order is replaced.
                                        test_perm_override = NULL,
                                        ##
                                        num_chunks_burnin = NULL,
                                        num_chunks_sampling = NULL,
                                        diffusion_HMC_integrator = "kick_flow_kick",
                                        debug_burnin_timing = FALSE
) {
                ## Built-in likelihoods use OpenMP WCP teams; external Stan likelihoods share the outer TBB arena.
                if (identical(x = init_object$Model_type, y = "Stan")) {
                    if (isTRUE(x = burnin_TBB_pool_equals_n_chains)) {
                        message("burnin_TBB_pool_equals_n_chains = TRUE is overridden to FALSE for Stan models to allow TBB within-chain parallelism.")
                    }
                    burnin_TBB_pool_equals_n_chains <-  FALSE
                } else {
                    burnin_TBB_pool_equals_n_chains <-  if (is.null(x = burnin_TBB_pool_equals_n_chains)) TRUE else
                                                          isTRUE(x = burnin_TBB_pool_equals_n_chains)
                }
                ##
                if (!is.logical(debug_burnin_timing) || length(debug_burnin_timing) != 1 || is.na(debug_burnin_timing)) {
                    stop("debug_burnin_timing must be TRUE or FALSE.")
                }
  
                ##
                message("Printing from R_fn_sample_model:")
                ##
                ## Helper function:
                ## ---- Should the sampler KEEP the nuisance (latent-variable) trace?
                ##
                ## Its only consumer is the post-hoc param_constrain, which needs the complete
                ## unconstrained draw [nuisance | main] ONLY when the model being constrained
                ## declares the nuisance coordinates: an external Stan model with a nuisance block,
                ## or the latent_trait skeleton. The LC_MVP / LC_MVOP / MVP / MVOP skeletons declare
                ## the main parameters only, so for them the trace has no reader anywhere - yet
                ## storing it costs n_nuisance x n_iter x n_chains doubles of memory traffic per run
                ## (86 GB at n_nuisance = 60,000, 180 chains, 1000 iterations), which showed up as
                ## a ~20% drop in gradient evaluations per second. Decided from the model's own
                ## unconstrained dimension, not from Model_type, so it stays right if a skeleton
                ## changes. An explicit n_nuisance_to_track always wins.
                ##
                fn_resolve_n_nuisance_to_track <- function(user_value, sample_nuisance, n_nuisance, bs_model) {

                      if (!is.null(user_value)) {
                            if (!is.numeric(user_value) || length(user_value) != 1L || !is.finite(user_value)) {
                              stop("'n_nuisance_to_track' must be NULL (automatic), 0 (do not keep the nuisance trace) ",
                                   "or n_nuisance = ", n_nuisance, " (keep all of it).")
                            }
                            user_value <- as.integer(round(user_value))
                            ##
                            ## There is no such thing as a PARTIAL nuisance trace: the store grows the buffer to
                            ## the full length on the first write (so that its one reader, param_constrain, can
                            ## rebuild the complete vector), which means e.g. n_nuisance_to_track = 1 silently
                            ## costs exactly as much as tracking all 60,000 coordinates. Refuse it rather than
                            ## let a value that LOOKS cheap be the most expensive setting there is:
                            if (user_value != 0L && user_value != n_nuisance) {
                              stop("'n_nuisance_to_track' = ", user_value, " is not supported: the nuisance trace is ",
                                   "all-or-nothing (a partial buffer is grown to the full ", n_nuisance, " on the first ",
                                   "write, so it would cost the same as keeping everything). Use 0 to keep none, ",
                                   n_nuisance, " (= n_nuisance) to keep all, or NULL to decide automatically.")
                            }
                            return(user_value)
                      }
                      ##
                      if (!isTRUE(sample_nuisance) || n_nuisance == 0) return(0L)
                      ##
                      constrain_unc_num <- tryCatch(bs_model$param_unc_num(), error = function(e) NA_integer_)
                      ##
                      ## unknown dimension -> keep the trace (the safe direction); otherwise keep it
                      ## only when the model contains the nuisance block:
                      needs_trace <- is.na(constrain_unc_num) || (constrain_unc_num > n_nuisance)
                      ##
                      if (needs_trace) {
                            message("n_nuisance_to_track: the model used for param_constrain declares the ",
                                    n_nuisance, "-coordinate nuisance block - keeping the nuisance trace.")
                            n_nuisance
                      } else {
                            message("n_nuisance_to_track: the model used for param_constrain has ", constrain_unc_num,
                                    " unconstrained coordinates (main only), so the ", n_nuisance,
                                    "-coordinate nuisance trace has no consumer - NOT storing it. ",
                                    "Pass n_nuisance_to_track = n_nuisance to keep it anyway.")
                            0L
                      }

                }
                ##
                if_null_then_set_to <- function(x, 
                                                set_to_this_if_null) { 
                  
                  if (is.null(x)) {
                    return(set_to_this_if_null)
                  }
                  return(x)
                  
                }
                ##
                if (is.null(seed)) {
                  stop("Please specify a seed for MCMC sampling")
                }
                if (!is.null(adapt_delta) && (adapt_delta <= 0 || adapt_delta >= 1)) {
                  stop("adapt_delta must be between 0 and 1")
                }
                ##
                ##
                # ## bookmark : For the latent_trait model, the MANUAL-LOG-SCALE lp_grad function isn't yet working fully, so don't use it and print error
                # if (Model_type == "latent_trait") {
                #   if (force_PartialLog == TRUE) {
                #      stop("Error: the * MANUAL * LOG-SCALE lp_grad function isn't yet working fully, please set force_PartialLog = FALSE\n
                #           However, note that the autodiff (AD) version is working. ")
                #   }
                # }
                ##
                # if (partitioned_HMC == FALSE) {
                #   if (diffusion_HMC == TRUE) {
                #     stop("Diffusion-pathspace HMC is only allowed if partitioned_HMC is set to TRUE - \n
                #                      since we can only sample the nuisance parameters using diffusion-pathspace HMC; \n
                #                      also, ensure that your model has latent variables/ nuisasnce parameters wich \n
                #                      are (approximately) normaly distributed on the raw (unconstrained) scale")
                #   }
                # }
                ##
                ## ---- Print warnings:
                ##
                if (is.null(diffusion_HMC)) {
                  warning("'diffusion_HMC' not specificed (either TRUE of FALSE) - using default (standard, non-diffusion HMC)")
                  diffusion_HMC <- FALSE 
                }
                
                if (partitioned_HMC == TRUE && is.null(sample_nuisance)) {
                  sample_nuisance <- TRUE
                } 
                ##
                ## ---- Set some MCMC defaults:
                ##
                ratio_M_us <- ratio_M_nuisance
                max_eps_us <- max_eps_nuisance
                ##
                burnin_algorithm <- if_null_then_set_to(burnin_algorithm, "ChESSR")
                ##
                parallel_method <- if_null_then_set_to(parallel_method, "RcppParallel")
                ##
                n_burnin <- if_null_then_set_to(n_burnin, 500)
                ## Resolve the main schedule once; the ordering pre-burn-in deliberately does not receive this option.
                resolved_burnin_schedule <- fn_burnin_adaptation_schedule(
                    n_burnin = n_burnin,
                    n_adapt = n_adapt,
                    burnin_schedule = if_null_then_set_to(burnin_schedule, "automatic"),
                    metric_adaptation_end_iter = metric_adaptation_end_iter,
                    theta_hat_us_freeze_iter = theta_hat_us_freeze_iter,
                    theta_hat_us_rule = if_null_then_set_to(theta_hat_us_rule, "running_mean_frozen"),
                    clip_iter = clip_iter,
                    clip_iter_tau = clip_iter_tau)
                burnin_schedule <- resolved_burnin_schedule$burnin_schedule
                n_adapt <- resolved_burnin_schedule$n_adapt
                metric_adaptation_end_iter <- resolved_burnin_schedule$metric_adaptation_end_iter
                theta_hat_us_freeze_iter <- resolved_burnin_schedule$theta_hat_us_freeze_iter
                clip_iter <- resolved_burnin_schedule$clip_iter
                clip_iter_tau <- resolved_burnin_schedule$clip_iter_tau
                ##
                ## ---- clip_iter / clip_iter_tau defaults: the configurations selected by pilot
                ## study 7 (ps_7_MCMC_settings_BayesMVP.R), which sets them per burnin length and
                ## derives clip_iter_tau as clip_iter + int (see ps_7_..._functions.R):
                ##
                ##    n_burnin   clip_iter   int   clip_iter_tau
                ##      1000        150      200       350          (0.15 / 0.35 of n_burnin)
                ##       500         75      100       175          (0.15 / 0.35)
                ##       250         50       75       125          (0.20 / 0.50)
                ##       125         20       30        50          (0.16 / 0.40; requested earlier handover)
                ##
                ## Historical 125-iteration schedule, retained for reference:
                ##       125         30       50        80          (0.24 / 0.64)
                ##
                ## Previous banding rationale (the exact 125-iteration schedule is now the exception):
                ## Shorter burnins need proportionally MORE clipping, so this is a banding rather
                ## than one ratio. The four studied lengths are reproduced exactly; outside that
                ## range the end bands' ratios are extrapolated so that very long or very short
                ## burnins still scale (and clip_iter_tau can never exceed n_burnin).
                ##
                if (is.null(clip_iter) || is.null(clip_iter_tau)) {

                      if (n_burnin >= 1000) {
                          clip_iter_default     <- round(0.15 * n_burnin)
                          clip_iter_tau_default <- round(0.35 * n_burnin)
                      } else if (n_burnin >= 500) {
                          clip_iter_default     <- 75
                          clip_iter_tau_default <- 175
                      } else if (n_burnin >= 250) {
                          clip_iter_default     <- 50
                          clip_iter_tau_default <- 125
                      } else if (n_burnin == 125) {
                          ## 40% of the 250-iteration ramp: 20 MALA iterations + 30 ramp iterations.
                          clip_iter_default     <- 20
                          clip_iter_tau_default <- 50
                      } else if (n_burnin >= 125) {
                          clip_iter_default     <- 30
                          clip_iter_tau_default <- 80
                      } else {
                          clip_iter_default     <- round(0.24 * n_burnin)
                          clip_iter_tau_default <- round(0.64 * n_burnin)
                      }
                      ##
                      clip_iter <- if_null_then_set_to(clip_iter, clip_iter_default)
                      clip_iter_tau <- if_null_then_set_to(clip_iter_tau, clip_iter_tau_default)

                }
                ##
                n_adapt <- if_null_then_set_to(n_adapt, n_burnin - round(n_burnin/10))
                # gap <- if_null_then_set_to(gap, clip_iter  + round(n_adapt / 5)) 
                ##
                gap <- if_null_then_set_to(gap, clip_iter + clip_iter_tau)
                ##
                n_chains_sampling <- if_null_then_set_to(n_chains_sampling, parallel::detectCores())
                ##
                # ## per-chain jitter of the inherited nuisance init (see cpp_fn_post_burnin_prep_for_sampling);
                # ## 0 = old behaviour, where n_chains_sampling / n_chains_burnin chains share one exact vector
                # nuisance_jitter_scale <- if_null_then_set_to(nuisance_jitter_scale, 0.25)
                # if (!is.numeric(nuisance_jitter_scale) || length(nuisance_jitter_scale) != 1 ||
                #     !is.finite(nuisance_jitter_scale) || nuisance_jitter_scale < 0) {
                #     stop("nuisance_jitter_scale must be a single non-negative finite number (0 disables the jitter).")
                # }
                nuisance_jitter_scale <- 0.0
                ##
                if (is.null(n_superchains)) {
                    if (n_chains_sampling < 16) { 
                      n_superchains <- n_chains_sampling
                    } else { 
                      if (n_chains_sampling %in% c(17, 31)) { 
                           n_superchains <- round(n_chains_sampling/2)
                      } else if (n_chains_sampling %in% c(32, 63)) { 
                           n_superchains <- round(n_chains_sampling/4)
                      } else if (n_chains_sampling %in% c(64, 127)) { 
                           n_superchains <- round(n_chains_sampling/8)
                      } else if (n_chains_sampling > 127) { 
                           n_superchains <- round(n_chains_sampling/16)
                      }
                    }
                }
                ##
                n_superchains <- if_null_then_set_to(n_superchains, "ChESSR")
                ##
                n_iter <- if_null_then_set_to(n_iter, 1000)
                ##
                adapt_delta <- if_null_then_set_to(adapt_delta, 0.80)
                ##
                ## ---- learning_rate default: the pilot study 7 pairing (the same one encoded by
                ## fn_sampler_settings_APMS_BayesMVP()), INTERPOLATED so that every burnin length
                ## gets a rate, not just the four studied ones:
                ##
                ##    n_burnin    1000     500     250     125
                ##    LR        0.0125   0.025   0.05    0.075
                ##
                ## Interpolation is linear in log(n_burnin) vs log(LR), i.e. piecewise power-law.
                ## That is the natural scale here: over [250, 1000] the studied rates are EXACTLY
                ## inversely proportional to the burnin length (LR * n_burnin = 12.5 at all three
                ## anchors), so that whole stretch is one straight line and is reproduced exactly;
                ## only the short [125, 250] segment is shallower (the rate is deliberately held
                ## below what 1/n_burnin would give, which at 125 would be 0.10).
                ##
                ## Outside [125, 1000] the nearest segment's slope is continued, so a 2000-iteration
                ## burnin gets 0.00625 and short ones keep rising - capped at 0.125, the value the
                ## previous banding used for short burnins, so that a very small n_burnin cannot
                ## produce an unstable rate.
                ##
                ## (This REPLACES a banding that tested n_burnin %in% c(601, 900) and similar, which
                ## matched only those exact values: n_burnin of 250, 500 or 700 fell through every
                ## branch and left learning_rate NULL, which then propagated into LR_main / LR_us.)
                ##
                if (is.null(learning_rate)) {

                      log_n_burnin_anchors <- log(c(125, 250, 500, 1000))
                      log_learning_rate_anchors <- log(c(0.075, 0.05, 0.025, 0.0125))
                      ##
                      ## all.inside = TRUE clamps to a real segment, so values below 125 / above 1000
                      ## continue the first / last segment rather than falling off the end:
                      segment_index <- findInterval( x = log(n_burnin),
                                                     vec = log_n_burnin_anchors,
                                                     all.inside = TRUE)
                      ##
                      log_slope <- (log_learning_rate_anchors[segment_index + 1] - log_learning_rate_anchors[segment_index]) /
                                   (log_n_burnin_anchors[segment_index + 1] - log_n_burnin_anchors[segment_index])
                      ##
                      learning_rate <- exp(log_learning_rate_anchors[segment_index] +
                                           log_slope * (log(n_burnin) - log_n_burnin_anchors[segment_index]))
                      ##
                      learning_rate <- min(signif(x = learning_rate, digits = 6), 0.125)

                }
                ##
                if (is.null(learning_rate) || !is.finite(learning_rate) || learning_rate <= 0) {
                  stop("'learning_rate' must be a single positive finite number (or NULL to use the n_burnin-based default).")
                }
                ##
                LR_main <- LR_us <- learning_rate
                ##
                tau_mult <- if_null_then_set_to(tau_mult, 1.60)
                if (!is.null(x = tau_initial)) {
                  if (!is.numeric(x = tau_initial) || length(x = tau_initial) != 1L ||
                      !is.finite(x = tau_initial) || tau_initial <= 0) {
                    stop("'tau_initial' must be NULL or a single positive finite number.")
                  }
                }
                ##
                manual_tau <- if_null_then_set_to(manual_tau, FALSE)
                tau_if_manual <- if_null_then_set_to(tau_if_manual, 3.0)
                ##
                ## ---- Trajectory-length adaptation. Every default here is the behaviour that
                ##      predates these arguments, so an existing call is unchanged by them.
                tau_if_manual_in_L_units <- if_null_then_set_to(tau_if_manual_in_L_units, FALSE)
                tau_objective <- if_null_then_set_to(tau_objective, "KE")
                tau_weight_by_p_jump <- if_null_then_set_to(tau_weight_by_p_jump, FALSE)
                tau_ramp <- if_null_then_set_to(tau_ramp, "original")
                eps_reinit_at_ChEES_handover <- if_null_then_set_to(eps_reinit_at_ChEES_handover, TRUE)
                share_tau_ii_across_chains_in_burnin <- isTRUE(if_null_then_set_to(share_tau_ii_across_chains_in_burnin, FALSE))
                store_log_lik_trace <- if_null_then_set_to(store_log_lik_trace, TRUE)
                if (!is.logical(store_log_lik_trace) || length(store_log_lik_trace) != 1 || is.na(store_log_lik_trace)) {
                    stop("store_log_lik_trace must be NULL, TRUE or FALSE.")
                }
                theta_hat_us_rule <- if_null_then_set_to(theta_hat_us_rule, "running_mean_frozen")
                if (length(theta_hat_us_rule) != 1 || !theta_hat_us_rule %in% c("running_mean_frozen", "running_mean", "zero")) {
                    stop("theta_hat_us_rule must be 'running_mean_frozen', 'running_mean' or 'zero'; got: ", paste(as.character(theta_hat_us_rule), collapse = ", "))
                }
                if (!is.null(pre_burnin_n_iter) && (length(pre_burnin_n_iter) != 1 || !is.finite(pre_burnin_n_iter) || pre_burnin_n_iter < 30)) {
                    stop("pre_burnin_n_iter must be NULL or a single number >= 30.")
                }
                if (!is.null(pre_burnin_L) && (length(pre_burnin_L) != 1 || !is.finite(pre_burnin_L) || pre_burnin_L < 1)) {
                    stop("pre_burnin_L must be NULL or a single leapfrog-step count >= 1.")
                }
                if (!is.logical(eps_reinit_at_ChEES_handover) || length(eps_reinit_at_ChEES_handover) != 1 || is.na(eps_reinit_at_ChEES_handover)) {
                    stop("eps_reinit_at_ChEES_handover must be TRUE or FALSE; got: ", paste(as.character(eps_reinit_at_ChEES_handover), collapse = ", "))
                }
                metric_estimator <- as.character(metric_estimator)
                if (length(metric_estimator) != 1 || !(metric_estimator %in% c("pooled", "chain_mean", "chain_mean_scaled"))) {
                    stop("metric_estimator must be 'pooled', 'chain_mean' or 'chain_mean_scaled'; got: ",
                         paste(as.character(metric_estimator), collapse = ", "))
                }
                if (length(tau_ramp) != 1 || !tau_ramp %in% c("original", "staged")) {
                    stop("tau_ramp must be 'original' or 'staged'; got: ", paste(as.character(tau_ramp), collapse = ", "))
                }
                ##
                if (length(tau_objective) != 1 || !tau_objective %in% c("KE", "ChEES", "ChEES_per_tau")) {
                    stop("tau_objective must be 'KE', 'ChEES' or 'ChEES_per_tau'; got: ", paste(as.character(tau_objective), collapse = ", "))
                }
                if (!is.logical(tau_weight_by_p_jump) || length(tau_weight_by_p_jump) != 1 || is.na(tau_weight_by_p_jump)) {
                    stop("tau_weight_by_p_jump must be a single TRUE or FALSE.")
                }
                if (!is.logical(tau_if_manual_in_L_units) || length(tau_if_manual_in_L_units) != 1 || is.na(tau_if_manual_in_L_units)) {
                    stop("tau_if_manual_in_L_units must be a single TRUE or FALSE.")
                }
                ##
                diffusion_HMC <- if_null_then_set_to(diffusion_HMC, FALSE)
                partitioned_HMC <- if_null_then_set_to(partitioned_HMC, FALSE)
                ##
                ## ---- n_refresh is the burnin print INTERVAL: progress is reported every
                ## n_refresh iterations. NULL keeps roughly the cadence the burnin used before it
                ## was made configurable (about 20 reports across the burnin), so leaving it unset
                ## does not change how chatty a run is:
                ##
                n_refresh <- if_null_then_set_to(n_refresh, max(1L, round(n_burnin / 20)))
                ##
                if (!is.numeric(n_refresh) || length(n_refresh) != 1L || !is.finite(n_refresh) || n_refresh < 1) {
                  stop("'n_refresh' must be NULL or a single finite number >= 1 (the burnin print interval, in iterations).")
                }
                use_proposed <- if_null_then_set_to(use_proposed, TRUE)
                ##
                beta1_adam <- if_null_then_set_to(beta1_adam, 0.00)
                beta2_adam <- if_null_then_set_to(beta2_adam, 0.95)
                eps_adam <- if_null_then_set_to(eps_adam, 1e-8)
                ##
                force_autodiff <- if_null_then_set_to(force_autodiff, FALSE)
                force_PartialLog <- if_null_then_set_to(force_PartialLog, FALSE)
                multi_attempts <- if_null_then_set_to(multi_attempts, TRUE)
                ##
                autodiff_fallback <-  getOption(x = "BayesMVP_autodiff_fallback", default = identical(init_object$Model_type, "latent_trait"))
                ##
                if (!is.logical(autodiff_fallback) || length(autodiff_fallback) != 1 || is.na(autodiff_fallback)) {
                    stop("BayesMVP_autodiff_fallback must be a single TRUE or FALSE.")
                }
                ##
                autodiff_fallback <-  init_object$Model_type %in% c("LC_MVP", "MVP", "LC_MVOP", "MVOP", "latent_trait") && isTRUE(multi_attempts) && autodiff_fallback
                ##
                if (autodiff_fallback) {
                    cat(init_object$Model_type, " lp_grad fallback: ",
                        if (identical(init_object$Model_type, "latent_trait")) "standard -> autodiff" else "standard -> partial-log -> autodiff",
                        " (same position; no trajectory retry).\n", sep = "")
                }
                ##
                force_autodiff_for_metric <- if_null_then_set_to(force_autodiff_for_metric, TRUE)
                force_PartialLog_for_metric <- if_null_then_set_to(force_PartialLog_for_metric, FALSE)
                force_multi_attempts_for_metric <- if_null_then_set_to(force_multi_attempts_for_metric, FALSE)
                ##
                vect_type_default <- BayesMVP:::detect_vectorization_support()
                vect_type <- if_null_then_set_to(vect_type_default, FALSE)
                ##
                Phi_type <- if_null_then_set_to(Phi_type, "Phi")
                inv_Phi_type <- if_null_then_set_to(inv_Phi_type, "inv_Phi")
                ##
                metric_type_main <- if_null_then_set_to(metric_type_main, "Empirical")
                metric_shape_main <- if_null_then_set_to(metric_shape_main, "diag")
                ratio_M_main <- if_null_then_set_to(ratio_M_main, 0.25)
                interval_width_main <- if_null_then_set_to(interval_width_main, 10)
                ##
                metric_type_nuisance <- if_null_then_set_to(metric_type_nuisance, "Empirical")
                metric_shape_nuisance <- if_null_then_set_to(metric_shape_nuisance, "diag")
                ratio_M_nuisance <- if_null_then_set_to(ratio_M_nuisance, 0.25)
                interval_width_nuisance <- if_null_then_set_to(interval_width_nuisance, 10)
                ##
                max_eps_main <- if_null_then_set_to(max_eps_main, 0.75)
                max_eps_nuisance <- if_null_then_set_to(max_eps_nuisance, 0.75)
                ##
                ## ---- eps warm start (see init_and_run_burnin_ChESSR): hold the HMC step-size at a
                ## deliberately LARGE 'eps_initial' for the first 'eps_initial_iter' burnin iterations,
                ## then hand it back to the usual starting step-size and adapt from there. Default is
                ## ON at eps_initial = 0.10 held for n_burnin / 10 iterations - i.e. the 50 iterations
                ## found to work at n_burnin = 500, scaled proportionally. Pass eps_initial = NULL (or
                ## NA) to switch the warm start off entirely.
                ##
                if (!is.null(eps_initial)) {
                      if (length(eps_initial) != 1L || is.na(eps_initial)) {
                            eps_initial <- NULL
                      } else if (!is.numeric(eps_initial) || !is.finite(eps_initial) || eps_initial <= 0) {
                            stop("'eps_initial' must be a single positive finite number, or NULL / NA to disable the eps warm start.")
                      }
                }
                ##
                if (!is.null(eps_initial_iter)) {
                      if (!is.numeric(eps_initial_iter) || length(eps_initial_iter) != 1L ||
                          !is.finite(eps_initial_iter) || eps_initial_iter < 0) {
                            stop("'eps_initial_iter' must be NULL (n_burnin / 10) or a single non-negative finite number of iterations.")
                      }
                }
                ##
                ## ---- learning-rate hold (see init_and_run_burnin_ChESSR): hold the ADAM learning
                ## rate at a HIGH 'learning_rate_initial' for the first 'learning_rate_initial_iter'
                ## burnin iterations - which accelerates BOTH the step-size and the path-length
                ## adaptation - then drop it to the run's normal learning rate. Default is ON at 0.10
                ## held for n_burnin / 10 iterations, i.e. the 50 iterations found to work at
                ## n_burnin = 500, scaled proportionally. Pass NULL (or NA) to switch the hold off.
                ##
                # if (!is.null(learning_rate_initial)) {
                #       if (length(learning_rate_initial) != 1L || is.na(learning_rate_initial)) {
                #             learning_rate_initial <- NULL
                #       } else if (!is.numeric(learning_rate_initial) || !is.finite(learning_rate_initial) ||
                #                  learning_rate_initial <= 0) {
                #             stop("'learning_rate_initial' must be a single positive finite number, or NULL / NA to disable the learning-rate hold.")
                #       }
                # }
                if (is.null(learning_rate_initial)) { 
                  learning_rate_initial <- learning_rate
                }
                ##
                if (!is.null(learning_rate_initial_iter)) {
                      if (!is.numeric(learning_rate_initial_iter) || length(learning_rate_initial_iter) != 1L ||
                          !is.finite(learning_rate_initial_iter) || learning_rate_initial_iter < 0) {
                            stop("'learning_rate_initial_iter' must be NULL (n_burnin / 10) or a single non-negative finite number of iterations.")
                      }
                }
                ##
                max_L <- if_null_then_set_to(max_L, 1024)
                # ##
                # n_nuisance_to_track <- if_null_then_set_to(n_nuisance_to_track, n_nuisance)
                ##
                Model_type <- init_object$Model_type
                ##
                Stan_model_file_path <- init_object$Stan_model_file_path
                Stan_cpp_user_header <- init_object$Stan_cpp_user_header
                Stan_cpp_flags <- init_object$Stan_cpp_flags
                stanc_args <- init_object$stanc_args
                make_args <- init_object$make_args
                ##
                ## ---- 'reorder_cols_MVP' for USER-SUPPLIED Stan models:
                ##
                ##      For the built-in models BayesMVP owns the data layout, so it knows
                ##      exactly which objects are test-indexed and can permute all of them.
                ##      For an EXTERNAL Stan model it knows nothing about the data block
                ##      beyond what is in 'Stan_data_list', so the ONLY thing it permutes is
                ##      the outcome matrix - which must therefore exist and be called 'y'.
                ##      If it is missing, reordering is REFUSED (loudly) instead of being
                ##      applied to something arbitrary:
                ##
                reorder_cols_MVP <- if (missing(reorder_cols_MVP)) FALSE else if_null_then_set_to(reorder_cols_MVP, FALSE)
                ##
                if (isTRUE(reorder_cols_MVP) && (Model_type == "Stan")) {
                      reorder_cols_MVP <- check_reorder_cols_MVP_for_Stan(Stan_data_list = Stan_data_list)
                }
                ##
                ## A fixed test order is only applied inside the reorder_cols_MVP block, so refuse it anywhere
                ## else rather than silently ignoring it (the ps7 filename would otherwise claim "_tp...").
                if (!is.null(test_perm_override) && !((Model_type %in% c("LC_MVP", "Stan")) && isTRUE(reorder_cols_MVP))) {
                    stop("test_perm_override is only used when reorder_cols_MVP = TRUE (Model_type LC_MVP or Stan); ",
                         "here it would be silently ignored.")
                }
                ##
                # if (is.null(Stan_data_list$test_perm))      Stan_data_list$test_perm      <- 1:model_args_list$n_tests
                # if (is.null(Stan_data_list$n_cat_per_test)) Stan_data_list$n_cat_per_test <- model_args_list$n_cat_per_test
                ##
                ## ---- Explicit phase chunk counts for external Stan models use the model's N / grainsize data contract:
                ##
                stan_grainsize_sampling <- NULL
                if (Model_type == "Stan" && (!is.null(num_chunks_burnin) || !is.null(num_chunks_sampling))) {
                    for (chunk_count in list(num_chunks_burnin, num_chunks_sampling)) {
                        if (is.null(chunk_count)) next
                        if (!is.numeric(chunk_count) || length(chunk_count) != 1L || !is.finite(chunk_count) ||
                            chunk_count < 1 || chunk_count != floor(chunk_count) || chunk_count > .Machine$integer.max) {
                            stop("Stan phase chunk counts must be NULL or positive integers within Stan's integer range.")
                        }
                    }
                    for (data_name in c("N", "grainsize")) {
                        data_value <- Stan_data_list[[data_name]]
                        if (!is.numeric(data_value) || length(data_value) != 1L || !is.finite(data_value) ||
                            data_value < 1 || data_value != floor(data_value) || data_value > .Machine$integer.max) {
                            stop("Explicit Stan phase chunk counts require positive integer data fields N and grainsize.")
                        }
                    }
                    ## NULL retains the supplied data grainsize independently for that phase.
                    stan_grainsize_sampling <- if (is.null(num_chunks_sampling)) Stan_data_list$grainsize else
                        as.integer(ceiling(Stan_data_list$N / num_chunks_sampling))
                    if (!is.null(num_chunks_burnin)) {
                        Stan_data_list$grainsize <- as.integer(ceiling(Stan_data_list$N / num_chunks_burnin))
                    }
                }
                ##
                if (is.null(num_chunks_burnin)) { 
                  num_chunks_burnin <- model_args_list$num_chunks
                } else { 
                  model_args_list$num_chunks <- num_chunks_burnin
                }
                ##
                ## ---- Initialise model:
                ##
                ## model_args_list$num_chunks <- 4
                ##
                init_object <- initialise_model( Model_type =  Model_type,
                                                 ##
                                                 stream = stream,
                                                 ##
                                                 sample_nuisance = sample_nuisance,
                                                 n_nuisance_override = n_nuisance_override,
                                                 ##
                                                 model_args_list = model_args_list,
                                                 ##
                                                 compile = TRUE,
                                                 force_recompile = FALSE,
                                                 ##
                                                 cmdstanr_model_fit_obj = NULL,
                                                 ##
                                                 Stan_data_list = Stan_data_list,
                                                 ##
                                                 Stan_model_file_path = Stan_model_file_path,
                                                 Stan_cpp_user_header = Stan_cpp_user_header,
                                                 Stan_cpp_flags = Stan_cpp_flags,
                                                 stanc_args = stanc_args,
                                                 make_args = make_args)
                ##
                y <- init_object$model_args_list$y ; y
                N <- init_object$model_args_list$N ; N
                n_class <- init_object$model_args_list$n_class ; n_class
                ##
                ##
                Model_args_as_Rcpp_List <- init_object$Model_args_as_Rcpp_List ; Model_args_as_Rcpp_List
                Model_args_as_Rcpp_List$autodiff_fallback <-  autodiff_fallback
                model_args_list         <- init_object$model_args_list
                ##
                # Model_args_as_Rcpp_List$Model_args_ints[4, 1] <- 4
                # str(Model_args_as_Rcpp_List)
                ##
                n_nuisance <- init_object$n_nuisance ; n_nuisance
                n_params_main <- init_object$n_params_main ; n_params_main
                ##
                ## ---- Models with NO nuisance block (n_nuisance == 0): there are no
                ## nuisance coordinates to partition off or diffuse, so the COMPLETE
                ## unconstrained vector is sampled jointly as the main block. Main-block
                ## settings (metric, tau, eps, adapt_delta, n_iter, seeds, ...) are
                ## unchanged. This only triggers when n_nuisance resolves to 0 (e.g. a
                ## Stan model run with sample_nuisance = FALSE, or a nuisance declaration
                ## that evaluates to zero length); built-in models always have n_nuisance > 0.
                ##
                ## A Stan model can have an empty nuisance block for SOME data and not others - e.g. a
                ## declaration like "array[prior_only == 1 ? 0 : N] row_vector[n_tests] u_raw;", which
                ## is exactly what a prior-only run produces. So this has to be resolved from the
                ## realised n_nuisance, not from the settings the caller wrote.
                ##
                if (n_nuisance == 0 && (isTRUE(partitioned_HMC) || isTRUE(diffusion_HMC) || isTRUE(sample_nuisance))) {
                    message( "n_nuisance = 0: no nuisance coordinates to partition or diffuse - ",
                             "disabling partitioned / diffusion HMC and nuisance updates; ",
                             "the complete parameter vector is sampled jointly as the main block.")
                    ##
                    sample_nuisance <- FALSE
                    partitioned_HMC <- FALSE
                    diffusion_HMC <- FALSE
                    ##
                    ## ---- The joint-diffusion integrators are only defined WITH a nuisance block, so
                    ## disabling diffusion_HMC while leaving diffusion_HMC_integrator = "flow_kick_flow"
                    ## leaves the two contradicting each other, and the burnin then refuses to start
                    ## ("flow_kick_flow requires diffusion_HMC = TRUE ... and a nonempty nuisance
                    ## block"). Fall back to the default ordering alongside the disable:
                    ##
                    if (!identical(diffusion_HMC_integrator, "kick_flow_kick")) {
                          message( "n_nuisance = 0: diffusion_HMC_integrator '", diffusion_HMC_integrator,
                                   "' is only defined for joint diffusion over a nuisance block - ",
                                   "falling back to 'kick_flow_kick' for this run.")
                          ##
                          diffusion_HMC_integrator <- "kick_flow_kick"
                    }
                }
                ##
                ## the caller's choice (NULL = automatic), kept so a re-initialisation below
                ## re-resolves from the SAME instruction rather than from an already-resolved value:
                n_nuisance_to_track_user <- n_nuisance_to_track
                ##
                n_nuisance_to_track <- fn_resolve_n_nuisance_to_track( user_value = n_nuisance_to_track_user,
                                                                       sample_nuisance = sample_nuisance,
                                                                       n_nuisance = n_nuisance,
                                                                       bs_model = init_object$bs_model)
                # ##
                # if (is.null(n_nuisance_to_track)) {
                #   if (sample_nuisance == FALSE) {
                #     n_nuisance <- 1 # dummy (or 9? try both)
                #     n_nuisance_to_track <- 1 # dummy (or 9? try both)
                #   } else { 
                #     n_nuisance_to_track <- n_nuisance
                #   }
                # }
                n_params <- n_nuisance + n_params_main
                ##
                print(paste("n_params = ", n_params))
                print(paste("n_params_main = ",  n_params_main))
                print(paste("n_nuisance = ",  n_nuisance))
                ##
                Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
                bs_model <- init_object$bs_model
                ##
                index_nuisance <- seq_len(n_nuisance)
                index_main <- (1 + n_nuisance):n_params
                ##
                ## ---- Process initial values BEFORE burnin:
                ##
                outs <- R_fn_init_initial_values( Model_type = Model_type,
                                                  bs_model = bs_model,
                                                  ##
                                                  n_chains_burnin = n_chains_burnin,
                                                  init_lists_per_chain = init_lists_per_chain,
                                                  ##
                                                  sample_nuisance = sample_nuisance,
                                                  n_nuisance = n_nuisance,
                                                  n_params_main = n_params_main)
                ##
                inits_unconstrained_vec_per_chain <- outs$inits_unconstrained_vec_per_chain
                ##
                theta_nuisance_vectors_all_chains_input_from_R <- outs$theta_nuisance_vectors_all_chains_input_from_R
                theta_main_vectors_all_chains_input_from_R <- outs$theta_main_vectors_all_chains_input_from_R
                ##
                n_chains_burnin <- outs$n_chains_burnin
                init_lists_per_chain <- outs$init_lists_per_chain
                ##
                if (Model_type != "Stan")  {
                    n_params_main <- Model_args_as_Rcpp_List$n_params_main # <- n_params_main
                    n_nuisance <-    Model_args_as_Rcpp_List$n_nuisance #<- n_nuisance
                } else if  (Model_type == "Stan")   {
                    Model_args_as_Rcpp_List$n_params_main <- n_params_main
                    Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
                }
                if (Model_type == "Stan") {
                  Model_args_as_Rcpp_List$model_so_file <-    init_object$model_so_file
                  Model_args_as_Rcpp_List$json_file_path <-   init_object$json_file_path
                } else {
                  Model_args_as_Rcpp_List$model_so_file <-  "none" #   init_object$dummy_model_so_file
                  Model_args_as_Rcpp_List$json_file_path <- "none" #  init_object$dummy_json_file_path
                }
                ##
                if (Model_type == "Stan") {
                   if (length(Stan_data_list) == 0) {
                     ## a model with NO data: cmdstanr::write_stan_json() would write an
                     ## empty ARRAY ([]), but BridgeStan requires an empty OBJECT ({}):
                     writeLines(text = "{}", con = Model_args_as_Rcpp_List$json_file_path)
                   } else {
                     cmdstanr::write_stan_json(data = Stan_data_list, file = Model_args_as_Rcpp_List$json_file_path)
                   }
                } else { 
                   # cmdstanr::write_stan_json(data = Stan_data_list, file = Model_args_as_Rcpp_List$dummy_json_file_path)  
                }
                ##
                print(  Model_args_as_Rcpp_List$model_so_file)
                print(  Model_args_as_Rcpp_List$json_file_path)
                
                ## Built-in default: n_chains_burnin TBB threads plus each chain's OpenMP WCP team.
                ## Stan models always use the full shared TBB budget; built-in models can request it with explicit FALSE.
                RcppParallel::setThreadOptions(numThreads = if (burnin_TBB_pool_equals_n_chains) n_chains_burnin else n_threads_WCP_burnin * n_chains_burnin);
                
                if (burnin_algorithm %in% c("CHESSR", "ChESSR", "chessr")) { 
                     fn_burnin <- init_and_run_burnin_ChESSR
                } else if (burnin_algorithm %in% c("SNAPER", "snaper")) { 
                     fn_burnin <- init_and_run_burnin_SNAPER
                }
                ##
                time_pre_burnin <- 0.0
                ##
                ## ---- Second half of the 'reorder_cols_MVP' gate for external Stan models:
                ##      the ordering is chosen from the model's estimated correlation
                ##      matrix, so the model must expose one named "Omega" whose dimension
                ##      matches the number of columns of 'y'. Checked HERE (before the
                ##      pre-burnin) so that a model which cannot be reordered does not waste
                ##      a pre-burnin first:
                ##
                Omega_dims <- NULL
                ##
                if (isTRUE(reorder_cols_MVP) && (Model_type == "Stan")) {

                      Omega_dims <- infer_Omega_dims_from_bs_model(bs_model = bs_model)
                      n_tests_y  <- ncol(Stan_data_list$y)
                      ##
                      if (Omega_dims$n_found == 0) {
                            big_warning_banner(
                                  title = "reorder_cols_MVP: no 'Omega' parameter found in your Stan model",
                                  body = c("The column ordering is chosen from the estimated correlation matrix, which is",
                                           "located by NAME: BayesMVP looks for a parameter / transformed parameter /",
                                           "generated quantity called 'Omega' (e.g. 'Omega[i,j]' or 'Omega[c,i,j]').",
                                           "Your model declares no such quantity.",
                                           "",
                                           "==> reorder_cols_MVP has been DISABLED for this run; sampling continues with",
                                           "    your data in its original column order."))
                            reorder_cols_MVP <- FALSE
                      } else if (!identical(as.integer(Omega_dims$n_tests), as.integer(n_tests_y))) {
                            big_warning_banner(
                                  title = "reorder_cols_MVP: 'Omega' does not match the number of columns of 'y'",
                                  body = c(paste0("Omega in your Stan model is ", Omega_dims$n_tests, " x ", Omega_dims$n_tests,
                                                  ", but 'y' has ", n_tests_y, " columns."),
                                           "Reordering permutes the columns of 'y' using the ordering of Omega, so the two",
                                           "must refer to the same set of tests/outcomes, in the same order.",
                                           "",
                                           "==> reorder_cols_MVP has been DISABLED for this run; sampling continues with",
                                           "    your data in its original column order."))
                            reorder_cols_MVP <- FALSE
                      }

                }
                ##
                test_perm     <- NULL
                test_inv_perm <- NULL
                ##
                ## Default = identity for any model that didn't reorder:
                if (is.null(test_perm)) {
                  ## for external Stan models the test/outcome count is only knowable from
                  ## the user-supplied outcome matrix 'y':
                  n_tests_tmp <- if (Model_type == "Stan") {
                                    tryCatch(ncol(Stan_data_list$y), error = function(e) NULL)
                                 } else {
                                    tryCatch(init_object$model_args_list$n_tests, error = function(e) NULL)
                                 }
                  if (!is.null(n_tests_tmp)) {
                    test_perm     <- 1:n_tests_tmp
                    test_inv_perm <- 1:n_tests_tmp
                  }
                }
                ##
                # reorder_cols_MVP <- TRUE
                ##
                if ((Model_type %in% c("LC_MVP", "Stan")) &&
                    (reorder_cols_MVP == TRUE)) {

                    ## ---- Number of tests + number of latent classes: known from
                    ## 'model_args_list' for the built-in models; for an external Stan model
                    ## n_tests is the column count of the user-supplied 'y' and n_class is
                    ## read off the model's own "Omega" parameter names:
                    if (Model_type == "Stan") {
                          n_tests <- ncol(Stan_data_list$y)
                          ##
                          if (is.null(Omega_dims)) {
                            Omega_dims <- infer_Omega_dims_from_bs_model(bs_model = bs_model)
                          }
                          n_class_for_reorder <- if (is.null(Omega_dims$n_class)) 1L else Omega_dims$n_class
                    } else {
                          n_tests <- init_object$model_args_list$n_tests
                          n_class_for_reorder <- n_class
                    }
                    ##
                    theta_main_orig <- theta_main_vectors_all_chains_input_from_R
                    theta_nuisance_orig <- theta_nuisance_vectors_all_chains_input_from_R
                    ##
                    pre_burnin_params <- list()
                    ##
                    # pre_burnin_params$n_burnin <- 250
                    # pre_burnin_params$n_adapt  <- 200
                    # ##
                    # pre_burnin_params$gap <- 100
                    # ##
                    # pre_burnin_params$LR_main <- 0.10
                    # pre_burnin_params$LR_us <- 0.10
                    # ##
                    # pre_burnin_params$manual_tau <- TRUE
                    # pre_burnin_params$tau_if_manual <- c(3.0, 3.0)
                    # ##
                    # int <- 75
                    # pre_burnin_params$clip_iter <- 50
                    # pre_burnin_params$clip_iter_tau <- pre_burnin_params$clip_iter + int
                    # ##
                    # pre_burnin_params$M_decay_scale <- pre_burnin_params$n_adapt / 100
                    ##
                    pre_burnin_params$n_burnin <- 125
                    pre_burnin_params$n_adapt  <- 100
                    ##
                    pre_burnin_params$gap <- 50
                    ##
                    pre_burnin_params$LR_main <- 0.10
                    pre_burnin_params$LR_us <- 0.10
                    ##
                    pre_burnin_params$manual_tau <- TRUE
                    # pre_burnin_params$tau_if_manual <- c(3.0, 3.0)
                       # pre_burnin_params$tau_if_manual <- c(1.5, 1.5)
                    pre_burnin_params$tau_if_manual <- if (metric_estimator == "pooled") c(1.5, 1.5) else c(3.0, 3.0)
                    ## pre_burnin_L: a fixed number of leapfrog steps instead of a fixed tau (tau is then L * eps each iteration):
                    pre_burnin_params$tau_if_manual_in_L_units <- !is.null(pre_burnin_L)
                    if (!is.null(pre_burnin_L)) pre_burnin_params$tau_if_manual <- c(pre_burnin_L, pre_burnin_L)
                    ##
                    int <- 40
                    pre_burnin_params$clip_iter <- 25
                    pre_burnin_params$clip_iter_tau <- pre_burnin_params$clip_iter + int
                    ##
                    pre_burnin_params$M_decay_scale <- pre_burnin_params$n_adapt / 100
                    ##
                    ## ---- shorter pre-burnin (pre_burnin_n_iter): the same schedule, scaled to the new length:
                    if (!is.null(pre_burnin_n_iter)) {
                          pre_burnin_length_scale <- pre_burnin_n_iter / pre_burnin_params$n_burnin
                          pre_burnin_params$n_burnin      <- as.integer(round(pre_burnin_n_iter))
                          pre_burnin_params$n_adapt       <- round(pre_burnin_params$n_adapt * pre_burnin_length_scale)
                          pre_burnin_params$gap           <- round(pre_burnin_params$gap * pre_burnin_length_scale)
                          pre_burnin_params$clip_iter     <- max(5, round(pre_burnin_params$clip_iter * pre_burnin_length_scale))
                          pre_burnin_params$clip_iter_tau <- pre_burnin_params$clip_iter + round(int * pre_burnin_length_scale)
                          pre_burnin_params$M_decay_scale <- pre_burnin_params$n_adapt / 100
                          message("pre-burnin shortened to ", pre_burnin_params$n_burnin, " iterations (n_adapt = ", pre_burnin_params$n_adapt,
                                  ", clip_iter = ", pre_burnin_params$clip_iter, ", clip_iter_tau = ", pre_burnin_params$clip_iter_tau, ")")
                    }
                    ##
                    pre_burnin_object <-             fn_burnin(  init_object = init_object,
                                                                 debug_burnin_timing = debug_burnin_timing,
                                                                 ##
                                                                 Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                                 ##
                                                                 Model_type = Model_type,
                                                                 ##
                                                                 n_chains_burnin = n_chains_burnin,
                                                                 ##
                                                                 theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                                                 theta_nuisance_vectors_all_chains_input_from_R = theta_nuisance_vectors_all_chains_input_from_R,
                                                                 ##
                                                                 parallel_method = "RcppParallel", ## no OpenMP for burnin yet (only sampling!)
                                                                 ##
                                                                 Stan_data_list = Stan_data_list,
                                                                 model_args_list = model_args_list,
                                                                 ##
                                                                 sample_nuisance = sample_nuisance,
                                                                 n_nuisance_override = n_nuisance_override,
                                                                 ##
                                                                 seed = seed,
                                                                 n_burnin = pre_burnin_params$n_burnin,
                                                                 n_adapt = pre_burnin_params$n_adapt,
                                                                 gap = pre_burnin_params$gap,
                                                                 ##
                                                                 adapt_delta = adapt_delta,
                                                                 LR_main = pre_burnin_params$LR_main,
                                                                 LR_us = pre_burnin_params$LR_us,
                                                                 tau_mult = tau_mult,
                                                                  tau_initial = if_null_then_set_to(
                                                                      x = tau_initial,
                                                                      set_to_this_if_null = 2*pi),
                                                                 ##
                                                                 manual_tau = pre_burnin_params$manual_tau,
                                                                 tau_if_manual = pre_burnin_params$tau_if_manual,
                                                                 tau_if_manual_in_L_units = pre_burnin_params$tau_if_manual_in_L_units,
                                                                 ##
                                                                 burnin_algorithm = burnin_algorithm,
                                                                 diffusion_HMC = diffusion_HMC,
                                                                 diffusion_HMC_integrator = diffusion_HMC_integrator,
                                                                 partitioned_HMC = partitioned_HMC,
                                                                 ##
                                                                 clip_iter = pre_burnin_params$clip_iter,
                                                                 clip_iter_tau = pre_burnin_params$clip_iter_tau,
                                                                 ##
                                                                 n_refresh = n_refresh,
                                                                 use_proposed = use_proposed,
                                                                 ##
                                                                 beta1_adam = beta1_adam,
                                                                 beta2_adam = beta2_adam,
                                                                 eps_adam = eps_adam,
                                                                 ##
                                                                 force_autodiff = force_autodiff,
                                                                 force_PartialLog = force_PartialLog,
                                                                 multi_attempts = multi_attempts,
                                                                 ##
                                                                 force_autodiff_for_metric = force_autodiff_for_metric,
                                                                 force_PartialLog_for_metric = force_PartialLog_for_metric,
                                                                 force_multi_attempts_for_metric = force_multi_attempts_for_metric,
                                                                 ##
                                                                 vect_type = vect_type,
                                                                 Phi_type = Phi_type,
                                                                 inv_Phi_type = inv_Phi_type,
                                                                 ##
                                                                 metric_type_main = metric_type_main,
                                                                 metric_shape_main = metric_shape_main,
                                                                 ratio_M_main = ratio_M_main,
                                                                 interval_width_main = interval_width_main,
                                                                 ##
                                                                 M_decay_type = M_decay_type,
                                                                 M_decay_power = M_decay_power,
                                                                 M_decay_scale = pre_burnin_params$M_decay_scale,
                                                                 ##
                                                                 metric_type_nuisance = metric_type_nuisance,
                                                                 metric_shape_nuisance = metric_shape_nuisance,
                                                                 ratio_M_nuisance = ratio_M_nuisance,
                                                                 interval_width_nuisance = interval_width_nuisance,
                                                                 ##
                                                                 max_tau_main = max_tau_main,
                                                                 max_tau_nuisance = max_tau_nuisance,
                                                                 ##
                                                                 max_eps_main = max_eps_main,
                                                                 max_eps_nuisance = max_eps_nuisance,
                                                                 ##
                                                                 max_L = max_L,
                                                                 ##
                                                                 n_nuisance_to_track = n_nuisance_to_track,
                                                                 ##
                                                                 n_threads_WCP = n_threads_WCP_burnin,
                                                                 ##
                                                                 time_pre_burnin = time_pre_burnin,
                                                                 ##
                                                                 metric_start_iter = NULL,
                                                                 ##
                                                                 eps_init = NULL,
                                                                 ##
                                                                 ## same holds, scaled to the (short) pre-burnin length:
                                                                 learning_rate_initial = learning_rate_initial,
                                                                 learning_rate_initial_iter = NULL,
                                                                 ##
                                                                 eps_initial = eps_initial,
                                                                 eps_initial_iter = NULL,
                                                                 ##
                                                                 metric_estimator = metric_estimator)
                      # ##
                      # theta_main_pre <- pre_burnin_object$theta_main_vectors_all_chains_input_from_R
                      # # theta_main_median <- apply(theta_main_pre, 1, median)
                      # theta_main_median <- apply(theta_main_pre, 1, mean)
                      ##
                      time_pre_burnin <- pre_burnin_object$time_burnin
                      ##
                      theta_main_median <- pre_burnin_object$EHMC_burnin_as_Rcpp_List$snaper_m_vec_main
                      ##
                      ## ---- For an external Stan model a failure to read Omega back out of
                      ## the pre-burnin must NOT abort the whole (expensive) run: fall back
                      ## to the identity ordering instead:
                      if (Model_type == "Stan") {
                            ##
                            ## ---- param_constrain() needs the COMPLETE unconstrained vector. The
                            ## built-in skeletons declare no nuisance block, so theta_main alone is the
                            ## whole vector there; a user-supplied Stan model may declare one FIRST, in
                            ## which case the nuisance mean has to be prepended (same layout as
                            ## 'snaper_m_vec_all' = c(nuisance, main) used by the burnin):
                            ##
                            theta_full_median <- if (n_nuisance > 0) {
                                  c(as.numeric(pre_burnin_object$EHMC_burnin_as_Rcpp_List$snaper_m_vec_us),
                                    as.numeric(theta_main_median))
                            } else {
                                  as.numeric(theta_main_median)
                            }
                            ##
                            Omega_hat_list <- tryCatch(
                                  extract_Omega_via_bridgestan( bs_model = bs_model,
                                                                theta_main = theta_full_median,
                                                                n_tests = n_tests,
                                                                n_class = n_class_for_reorder),
                                  error = function(e) {
                                        big_warning_banner(
                                              title = "reorder_cols_MVP: could not extract 'Omega' after the pre-burnin",
                                              body = c(paste0("Error was: ", conditionMessage(e)),
                                                       "",
                                                       "==> continuing WITHOUT reordering (original column order)."))
                                        NULL
                                  })
                      } else {
                            Omega_hat_list <- extract_Omega_via_bridgestan( bs_model = bs_model,
                                                                            theta_main = theta_main_median,
                                                                            n_tests = n_tests,
                                                                            n_class = n_class_for_reorder)
                      }
                      # ##
                      # test_perm <- compute_optimal_test_order(Omega_hat_list)
                      # ##
                      # # # test_perm <- c(1,2,3,4,5,6)
                      # # test_perm <- c(4, 5, 3, 1, 6, 2)
                      # test_inv_perm <- order(test_perm)
                      # cat("Optimal test order:", test_perm, "\n")
                      ##
                #     ##
                #     ## Carry forward non-Omega params from pre-burnin
                #     ##
                #     theta_main_vectors_all_chains_input_from_R     <- pre_burnin_object$theta_main_vectors_all_chains_input_from_R
                #     theta_nuisance_vectors_all_chains_input_from_R <- pre_burnin_object$theta_us_vectors_all_chains_input_from_R
                #     ##
                      # theta_main_vectors_all_chains_input_from_R <- theta_main_orig
                      # theta_nuisance_vectors_all_chains_input_from_R <- theta_nuisance_orig
                #     
                #     
                #     test_perm <- c(4, 5, 3, 1, 6, 2) ## ----------------------------------- debug
                      
                      if (!is.null(test_perm_override)) {
                        ## User-fixed order (the pre-burnin estimate above is discarded):
                        test_perm <- as.integer(test_perm_override)
                        if (!identical(sort(test_perm), seq_len(as.integer(n_tests)))) {
                          stop("test_perm_override must be a permutation of 1:", n_tests, "; got: ", paste(test_perm_override, collapse = ", "))
                        }
                      } else if (is.null(Omega_hat_list)) {
                        ## Omega unavailable (external Stan model only - see above): identity order.
                        test_perm <- seq_len(n_tests)
                      } else {
                        test_perm <- compute_optimal_test_order(Omega_hat_list)
                      }
                      test_inv_perm <- order(test_perm)
                      cat(if (!is.null(test_perm_override)) "Test order (test_perm_override):" else "Optimal test order:", test_perm, "\n")
                      ##
                      if (!identical(as.integer(test_perm), seq_len(as.integer(n_tests)))) {
                        ##
                        n_tests_loc <- n_tests
                        ##
                      if (Model_type == "Stan") {
                        ##
                        ## ---- USER-SUPPLIED Stan model: the ONLY object permuted is the outcome
                        ##      matrix 'y' of the user's Stan data - BayesMVP cannot know which of
                        ##      the user's other data objects (priors, covariates, ...) are
                        ##      test-indexed, so it touches none of them, and it does not permute
                        ##      user-supplied initial values either:
                        ##
                        Stan_data_list$y <- Stan_data_list$y[, test_perm, drop = FALSE]
                        ##
                        ## ---- If the user's own Stan model declares a 'test_perm' data variable
                        ##      (the convention used by the built-in skeletons, which un-permute
                        ##      test-indexed quantities in generated quantities), keep it in sync so
                        ##      that the model can un-permute its own outputs:
                        ##
                        if (!is.null(Stan_data_list$test_perm)) {
                          Stan_data_list$test_perm <- as.integer(test_perm)
                        }
                        ##
                        message("reorder_cols_MVP (Stan): permuted the columns of 'y' to order [",
                                paste(test_perm, collapse = ", "), "]; ",
                                "fitted slot j holds your ORIGINAL test test_perm[j]. ",
                                "Only 'y'", if (!is.null(Stan_data_list$test_perm)) " (and 'test_perm')" else "",
                                " in the Stan data was changed.")
                        ##
                      } else {
                        ##
                        ## ---- Permute y:
                        ##
                        y_swapped <- model_args_list$y[, test_perm]
                        y <- y_swapped
                        model_args_list$y <- y_swapped
                        Stan_data_list$y <- y_swapped
                        ##
                        ## ---- Permute correlation bounds / known corrs (existing):
                        ##
                        perm_data <- permute_corr_data( model_args_list$lb_corr,
                                                        model_args_list$ub_corr,
                                                        model_args_list$known_values_indicator,
                                                        model_args_list$known_values,
                                                        test_perm)
                        model_args_list$lb_corr <- perm_data$lb_corr
                        model_args_list$ub_corr <- perm_data$ub_corr
                        model_args_list$known_values_indicator <- perm_data$known_values_indicator
                        model_args_list$known_values <- perm_data$known_values
                        ##
                        ## ---- Permute coefficient priors, X, covariate counts, baseline cases:
                        ##
                        for (c in 1:n_class) {
                          model_args_list$prior_coeffs_mean_mat[[c]] <- model_args_list$prior_coeffs_mean_mat[[c]][, test_perm, drop = FALSE]
                          model_args_list$prior_coeffs_sd_mat[[c]]   <- model_args_list$prior_coeffs_sd_mat[[c]][, test_perm, drop = FALSE]
                          model_args_list$X[[c]] <- model_args_list$X[[c]][test_perm]
                        }
                        # model_args_list$n_covs_per_outcome <- model_args_list$n_covs_per_outcome[, test_perm, drop = FALSE]   ## CHECK-NAME-1
                        # model_args_list$baseline_case_nd   <- model_args_list$baseline_case_nd[test_perm]                     ## CHECK-NAME-1
                        # model_args_list$baseline_case_d    <- model_args_list$baseline_case_d[test_perm]                      ## CHECK-NAME-1
                        ##
                        ## ---- Permute beta inits (existing):
                        ##
                        for (kk in 1:n_chains_burnin) {
                          for (c in 1:n_class) {
                            init_lists_per_chain[[kk]]$beta[[c]][,] <- init_lists_per_chain[[kk]]$beta[[c]][, test_perm, drop = FALSE]
                          }
                        }
                        ##
                        ## ================================================================
                        ## vvvvvvvvvvvv  NEW BLOCK — PASTE ALL OF THIS HERE  vvvvvvvvvvvv
                        ## ================================================================
                        ##
                        model_args_list$n_covariates_per_outcome_mat <- model_args_list$n_covariates_per_outcome_mat[, test_perm, drop = FALSE]
                        ##
                        model_args_list$test_perm <- test_perm   ## picked up by make_Stan_data_list at re-init
                        ##
                        # if (Model_type %in% c("LC_MVOP",
                        #                       "MVOP")) {
                        #   
                        #       n_bin_orig    <- model_args_list$n_binary_tests
                        #       ord_of_fitted <- test_perm[test_perm > n_bin_orig] - n_bin_orig
                        #       ##
                        #       if (!identical(ord_of_fitted, seq_along(ord_of_fitted))) {
                        #         
                        #             n_thr_old <- model_args_list$n_thr_per_ord_test
                        #             ##
                        #             model_args_list$prior_dirichlet_alpha <- model_args_list$prior_dirichlet_alpha[, ord_of_fitted, drop = FALSE]
                        #             ##
                        #             model_args_list$n_cat_per_ord_test <- model_args_list$n_cat_per_ord_test[ord_of_fitted]
                        #             model_args_list$n_thr_per_ord_test <- model_args_list$n_thr_per_ord_test[ord_of_fitted]
                        #             ##
                        #             end_old   <- cumsum(n_thr_old)
                        #             start_old <- c(1, head(end_old, -1) + 1)
                        #             ##
                        #             for (kk in 1:n_chains_burnin) {
                        #                 if (is.null(init_lists_per_chain[[kk]]$C_raw_vec)) {
                        #                   stop("permute block: init_lists_per_chain[[kk]]$C_raw_vec not found -- check the actual C_raw init field name!")
                        #                 }
                        #                 for (c in 1:2) {
                        #                   C_raw  <- init_lists_per_chain[[kk]]$C_raw_vec[[c]]
                        #                   blocks <- lapply(seq_along(n_thr_old), function(tt) C_raw[start_old[tt]:end_old[tt]])
                        #                   init_lists_per_chain[[kk]]$C_raw_vec[[c]] <- unlist(blocks[ord_of_fitted])
                        #                 }
                        #             }
                        #         
                        #       }
                        #       
                        # }
                        ##
                        ## ---- Ordinal metadata permutation (FULLY GENERAL -- makes NO assumption
                        ##      that the original data is binaries-first, nor that the permutation
                        ##      keeps binary and ordinal tests in separate blocks).
                        ##
                        ##      All ordinal-indexed metadata (n_cat_per_ord_test, n_thr_per_ord_test,
                        ##      prior_dirichlet_alpha, C_raw init blocks) is stored in ORDINAL-SLOT
                        ##      order: the k-th entry describes the k-th ordinal test *in slot order*.
                        ##      Under a permutation, slot j holds ORIGINAL test test_perm[j], so the
                        ##      new ordinal ordering is read straight off which fitted slots are
                        ##      ordinal, mapped back to their original ordinal indices.
                        ##
                        if (Model_type %in% c("LC_MVOP", 
                                              "MVOP")) {
                          
                              ##
                              ## ---- Which ORIGINAL tests are ordinal, and their ordinal index (NA for binary).
                              ##      Derived from n_cat_per_test, which is in ORIGINAL order at this point
                              ##      (the permutation has not been applied to it yet):
                              ##
                              n_cat_orig    <- model_args_list$n_cat_per_test
                              is_ord_orig   <- (n_cat_orig > 2L)
                              ##
                              ord_idx_orig            <- rep(NA_integer_, n_tests_loc)
                              ord_idx_orig[is_ord_orig] <- seq_len(sum(is_ord_orig))
                              ##
                              ## ---- ord_of_fitted[k] = ORIGINAL ordinal index of the k-th ordinal FITTED slot.
                              ##      Walk the fitted slots in order; slot j holds original test test_perm[j];
                              ##      keep the ordinal ones:
                              ##
                              ord_of_fitted <- ord_idx_orig[test_perm]
                              ord_of_fitted <- ord_of_fitted[!is.na(ord_of_fitted)]
                              ##
                              stopifnot(length(ord_of_fitted) == sum(is_ord_orig))
                              ##
                              if (!identical(ord_of_fitted, seq_along(ord_of_fitted))) {
                                
                                    n_thr_old <- model_args_list$n_thr_per_ord_test
                                    ##
                                    ## ---- Ordinal-indexed metadata:
                                    ##
                                    model_args_list$n_cat_per_ord_test    <- model_args_list$n_cat_per_ord_test[ord_of_fitted]
                                    model_args_list$n_thr_per_ord_test    <- model_args_list$n_thr_per_ord_test[ord_of_fitted]
                                    model_args_list$prior_dirichlet_alpha <- model_args_list$prior_dirichlet_alpha[, ord_of_fitted, drop = FALSE]
                                    ##
                                    ## ---- Ragged C_raw init blocks: whole-block permutation (blocks are
                                    ##      independent per test -- 1st elem is the first cutpoint, rest are
                                    ##      log-gaps WITHIN that test):
                                    ##
                                    end_old   <- cumsum(n_thr_old)
                                    start_old <- c(1, head(end_old, -1) + 1)
                                    ##
                                    for (kk in 1:n_chains_burnin) {
                                          if (is.null(init_lists_per_chain[[kk]]$C_raw_vec)) {
                                            stop("permute block: init_lists_per_chain[[kk]]$C_raw_vec not found -- check the init field name!")
                                          }
                                          for (c in 1:2) {
                                            C_raw  <- init_lists_per_chain[[kk]]$C_raw_vec[[c]]
                                            blocks <- lapply(seq_along(n_thr_old), function(tt) C_raw[start_old[tt]:end_old[tt]])
                                            init_lists_per_chain[[kk]]$C_raw_vec[[c]] <- unlist(blocks[ord_of_fitted])
                                          }
                                    }
                                
                              }
                              ##
                              ## ---- n_cat_per_test is TEST-indexed, so permute directly (independent of
                              ##      whether the ordinal sub-order changed):
                              ##
                              model_args_list$n_cat_per_test <- n_cat_orig[test_perm]
                          
                        }
                        ##
                        ## ================================================================
                        ## ^^^^^^^^^^^^^^^^^^  END OF NEW BLOCK  ^^^^^^^^^^^^^^^^^^^^^^^^
                        ## ================================================================
                        # ##
                        # ## ---- Slot-type vector (single source of truth) + ordinal metadata (LC_MVOP):
                        # ##
                        # model_args_list$n_cat_per_test <- model_args_list$n_cat_per_test[test_perm]                           ## CHECK-NAME-2
                        # ##
                        # if (Model_type == "LC_MVOP") {
                        #   
                        #       n_bin_orig <- sum(model_args_list$n_cat_per_test == 2)   ## count is perm-invariant
                        #       ##
                        #       ## fitted-slot-order list of ORIGINAL ordinal indices:
                        #       # ord_of_fitted <- test_perm[ model_args_list$n_cat_per_test[order(test_perm)][test_perm] > 2 ]  ## see simpler line below
                        #       ord_of_fitted <- test_perm[test_perm > n_bin_orig] - n_bin_orig   ## USE THIS ONE (original data = binaries-first)
                        #       ##
                        #       if (!identical(ord_of_fitted, seq_along(ord_of_fitted))) {
                        #         n_thr_old <- model_args_list$n_thr_per_ord_test
                        #         ##
                        #         model_args_list$n_thr_per_ord_test    <- model_args_list$n_thr_per_ord_test[ord_of_fitted]
                        #         model_args_list$n_cat_per_ord_test    <- model_args_list$n_cat_per_ord_test[ord_of_fitted]
                        #         model_args_list$prior_dirichlet_alpha <- model_args_list$prior_dirichlet_alpha[, ord_of_fitted, drop = FALSE]
                        #         ##
                        #         ## ---- Ragged C_raw init blocks: whole-block permutation (blocks are
                        #         ##      independent per test: 1st elem unconstrained, rest log-diffs WITHIN test):
                        #         end_old   <- cumsum(n_thr_old)
                        #         start_old <- c(1, head(end_old, -1) + 1)
                        #         for (kk in 1:n_chains_burnin) {
                        #           for (c in 1:2) {
                        #             C_raw <- init_lists_per_chain[[kk]]$C_raw_vec[[c]]                               ## CHECK-NAME-3
                        #             blocks <- lapply(seq_along(n_thr_old), function(tt) C_raw[start_old[tt]:end_old[tt]])
                        #             init_lists_per_chain[[kk]]$C_raw_vec[[c]] <- unlist(blocks[ord_of_fitted])       ## CHECK-NAME-3
                        #           }
                        #         }
                        #       }
                        #   
                        # }
                        # ##
                        # ## ---- Pass permutation + slot types to the skeleton:
                        # Stan_data_list$test_perm      <- test_perm
                        # Stan_data_list$n_cat_per_test <- model_args_list$n_cat_per_test
                        # ##
                        # ## ---- Mirror metadata into Stan_data_list (maintained in parallel with model_args_list):
                        # if (Model_type == "LC_MVOP") {
                        #   Stan_data_list$n_thr_per_ord_test    <- model_args_list$n_thr_per_ord_test
                        #   Stan_data_list$n_cat_per_ord_test    <- model_args_list$n_cat_per_ord_test
                        #   Stan_data_list$prior_dirichlet_alpha <- model_args_list$prior_dirichlet_alpha
                        # }
                # #     ##
                #       if (!identical(test_perm, 1:init_object$model_args_list$n_tests)) {
                # #       
                #             ##
                #             ## ---- Permute y:
                #             ##
                #             y_swapped <- model_args_list$y[, test_perm]
                #             y <- y_swapped
                #             model_args_list$y <- y_swapped
                #             Stan_data_list$y <- y_swapped
                #             ##
                #             ## ---- Permute correlation bounds and any known corr's:
                #             ##
                #             perm_data <- permute_corr_data(  model_args_list$lb_corr,
                #                                              model_args_list$ub_corr,
                #                                              model_args_list$known_values_indicator,
                #                                              model_args_list$known_values,
                #                                              test_perm)
                #             model_args_list$lb_corr <- perm_data$lb_corr
                #             model_args_list$ub_corr <- perm_data$ub_corr
                #             model_args_list$known_values_indicator <- perm_data$known_values_indicator
                #             model_args_list$known_values <- perm_data$known_values
                #             ##
                #             ## ---- Permute coefficient priors and X:
                #             ##
                #             for (c in 1:n_class) {
                #               model_args_list$prior_coeffs_mean_mat[[c]] <- model_args_list$prior_coeffs_mean_mat[[c]][, test_perm, drop = FALSE]
                #               model_args_list$prior_coeffs_sd_mat[[c]]   <- model_args_list$prior_coeffs_sd_mat[[c]][, test_perm, drop = FALSE]
                #               model_args_list$X[[c]] <- model_args_list$X[[c]][test_perm]
                #             }
                #             ##
                #             for (kk in 1:n_chains_burnin) { 
                #               for (c in 1:n_class) {
                #                 init_lists_per_chain[[kk]]$beta[[c]][,] <- init_lists_per_chain[[kk]]$beta[[c]][, test_perm, drop = FALSE]
                #               }
                #             }
                # #           ##
                # #           ## Zero out Omega params; keep beta/prevalence from pre-burnin:
                # #           ##
                # #           theta_main_vectors_all_chains_input_from_R <- zero_omega_in_theta( theta_main_pre,
                # #                                                                              bs_model)
                            ##
                      } ## end of "else" (built-in models) of the "if (Model_type == 'Stan')" permute-branch
                            ##
                            ## ---- Update init_object:
                            ##
                            # Stan_data_list$y <- Stan_data_list$y[, test_perm]
                            ##
                            init_object <- initialise_model( Model_type =  Model_type,
                                                             ##
                                                             stream = stream,
                                                             ##
                                                             sample_nuisance = sample_nuisance,
                                                             n_nuisance_override = n_nuisance_override,
                                                             ##
                                                             model_args_list = model_args_list,
                                                             ##
                                                             compile = TRUE,
                                                             force_recompile = FALSE,
                                                             ##
                                                             cmdstanr_model_fit_obj = NULL,
                                                             ##
                                                             Stan_data_list = Stan_data_list,
                                                             ##
                                                             Stan_model_file_path = Stan_model_file_path,
                                                             Stan_cpp_user_header = Stan_cpp_user_header,
                                                             Stan_cpp_flags = Stan_cpp_flags,
                                                             stanc_args = stanc_args,
                                                             make_args = make_args)
                            ##
                            Model_args_as_Rcpp_List <- init_object$Model_args_as_Rcpp_List
                            Model_args_as_Rcpp_List$autodiff_fallback <-  autodiff_fallback
                            model_args_list         <- init_object$model_args_list 
                            ##
                            n_nuisance <- init_object$n_nuisance ; n_nuisance
                            n_params_main <- init_object$n_params_main ; n_params_main
                            ##
                            n_nuisance_to_track <- fn_resolve_n_nuisance_to_track( user_value = n_nuisance_to_track_user,
                                                                                   sample_nuisance = sample_nuisance,
                                                                                   n_nuisance = n_nuisance,
                                                                                   bs_model = init_object$bs_model)
                            # ##
                            # if (is.null(n_nuisance_to_track)) {
                            #   if (sample_nuisance == FALSE) {
                            #     n_nuisance <- 1 # dummy (or 9? try both)
                            #     n_nuisance_to_track <- 1 # dummy (or 9? try both)
                            #   } else {
                            #     n_nuisance_to_track <- n_nuisance
                            #   }
                            # }
                            n_params <- n_nuisance + n_params_main
                            ##
                            print(paste("n_params = ", n_params))
                            print(paste("n_params_main = ",  n_params_main))
                            print(paste("n_nuisance = ",  n_nuisance))
                            ##
                            Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
                            bs_model <- init_object$bs_model
                            ##
                            ## ---- For external Stan models the DATA reaches the C++ sampler through the
                            ##      JSON file (not through Model_args_as_Rcpp_List), and the JSON path is
                            ##      content-addressed (hashed), so re-initialising on the permuted data
                            ##      produced a NEW json file: point the sampler at it (and re-write it from
                            ##      the permuted Stan_data_list, matching what is done pre-reorder):
                            ##
                            if (Model_type == "Stan") {
                                  ##
                                  Model_args_as_Rcpp_List$n_params_main <- n_params_main
                                  ##
                                  Model_args_as_Rcpp_List$model_so_file  <- init_object$model_so_file
                                  Model_args_as_Rcpp_List$json_file_path <- init_object$json_file_path
                                  ##
                                  if (length(Stan_data_list) == 0) {
                                    writeLines(text = "{}", con = Model_args_as_Rcpp_List$json_file_path)
                                  } else {
                                    cmdstanr::write_stan_json(data = Stan_data_list,
                                                              file = Model_args_as_Rcpp_List$json_file_path)
                                  }
                                  ##
                                  print(paste("post-reorder model_so_file = ",  Model_args_as_Rcpp_List$model_so_file))
                                  print(paste("post-reorder json_file_path = ", Model_args_as_Rcpp_List$json_file_path))
                                  ##
                            }
                            ##
                            index_nuisance <- seq_len(n_nuisance)
                            index_main <- (1 + n_nuisance):n_params
                            ##
                            ## ---- Process initial values BEFORE burnin:
                            ##
                            outs <- R_fn_init_initial_values( Model_type = Model_type,
                                                              bs_model = bs_model,
                                                              ##
                                                              n_chains_burnin = n_chains_burnin,
                                                              init_lists_per_chain = init_lists_per_chain,
                                                              ##
                                                              sample_nuisance = sample_nuisance,
                                                              n_nuisance = n_nuisance,
                                                              n_params_main = n_params_main)
                            ##
                            inits_unconstrained_vec_per_chain <- outs$inits_unconstrained_vec_per_chain
                            ##
                            theta_nuisance_vectors_all_chains_input_from_R <- outs$theta_nuisance_vectors_all_chains_input_from_R
                            theta_main_vectors_all_chains_input_from_R     <- outs$theta_main_vectors_all_chains_input_from_R
                            # theta_main_vectors_all_chains_input_from_R <- theta_main_pre ## -------------------
                            ##
                            n_chains_burnin <- outs$n_chains_burnin
                            init_lists_per_chain <- outs$init_lists_per_chain
                            # ##
                            # if (Model_type != "Stan")  {
                            #   n_params_main <- Model_args_as_Rcpp_List$n_params_main # <- n_params_main
                            #   n_nuisance <-    Model_args_as_Rcpp_List$n_nuisance #<- n_nuisance
                            # } else if  (Model_type == "Stan")   {
                            #   Model_args_as_Rcpp_List$n_params_main <- n_params_main
                            #   Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
                            # }
                            # if (Model_type == "Stan") {
                            #   Model_args_as_Rcpp_List$model_so_file <-    init_object$model_so_file
                            #   Model_args_as_Rcpp_List$json_file_path <-   init_object$json_file_path
                            # } else {
                            #   Model_args_as_Rcpp_List$model_so_file <-  "none" #   init_object$dummy_model_so_file
                            #   Model_args_as_Rcpp_List$json_file_path <- "none" #  init_object$dummy_json_file_path
                            # }
                            # ##
                            # if (Model_type == "Stan") {
                            #   cmdstanr::write_stan_json(data = Stan_data_list, file = Model_args_as_Rcpp_List$json_file_path)
                            # } else {
                            #   # cmdstanr::write_stan_json(data = Stan_data_list, file = Model_args_as_Rcpp_List$dummy_json_file_path)
                            # }
                            # ##
                            # print(  Model_args_as_Rcpp_List$model_so_file)
                            # print(  Model_args_as_Rcpp_List$json_file_path)
                            # ##
                            # RcppParallel::setThreadOptions(numThreads = n_chains_burnin);
                            # ##
                            # if (burnin_algorithm %in% c("CHESSR", "ChESSR", "chessr")) {
                            #   fn_burnin <- init_and_run_burnin_ChESSR
                            # } else if (burnin_algorithm %in% c("SNAPER", "snaper")) {
                            #   fn_burnin <- init_and_run_burnin_SNAPER
                            # }
                            # ##
                            # n_tests <- init_object$model_args_list$n_tests
                            ##
                            ## ---- Use FRESH inits from the new init_object, not pre-burnin values:
                            ##
                            # theta_main_vectors_all_chains_input_from_R <- theta_main_orig
                            # theta_nuisance_vectors_all_chains_input_from_R <- theta_nuisance_orig
                            # ##
                            # theta_main_vectors_all_chains_input_from_R <- pre_burnin_object$theta_main_vectors_all_chains_input_from_R  # inits stored here
                            # theta_nuisance_vectors_all_chains_input_from_R <- pre_burnin_object$theta_us_vectors_all_chains_input_from_R
                      }
                      
                } else {
                    pre_burnin_object <- NULL
                }
                ##
                ## ---- Run burnin:
                ##
                ## Explicit starting path lengths take precedence over the metric-estimator default:
                tau_initial <- if_null_then_set_to(x = tau_initial,
                                                  set_to_this_if_null = if (metric_estimator == "pooled") pi else 2*pi)
                ##
                # if (!(is.null(pre_burnin_object))) { 
                #     clip_iter <- 5
                #     clip_iter_tau <- 15
                # }
                ##
                burnin_object <-                 fn_burnin(  init_object = init_object,
                                                             debug_burnin_timing = debug_burnin_timing,
                                                             ##
                                                             Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                             ##
                                                             Model_type = Model_type,
                                                             ##
                                                             n_chains_burnin = n_chains_burnin,
                                                             ##
                                                             theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                                             theta_nuisance_vectors_all_chains_input_from_R = theta_nuisance_vectors_all_chains_input_from_R,
                                                             ##
                                                             parallel_method = "RcppParallel", ## no OpenMP for burnin yet (only sampling!)
                                                             ##
                                                             Stan_data_list = Stan_data_list,
                                                             model_args_list = model_args_list,
                                                             ##
                                                             sample_nuisance = sample_nuisance,
                                                             n_nuisance_override = n_nuisance_override,
                                                             ##
                                                             seed = seed,
                                                             n_burnin = n_burnin,
                                                             n_adapt = n_adapt,
                                                             gap = gap,
                                                             ##
                                                             adapt_delta = adapt_delta,
                                                             LR_main = LR_main,
                                                             LR_us = LR_us,
                                                             tau_mult = tau_mult,
                                                             ##
                                                             # tau_initial = tau_initial,
                                                             # tau_initial = pre_burnin_object$tau_main,
                                                              tau_initial = tau_initial,
                                                             ##
                                                             manual_tau = manual_tau,
                                                             tau_if_manual = tau_if_manual,
                                                             tau_if_manual_in_L_units = tau_if_manual_in_L_units,
                                                             ##
                                                             tau_objective = tau_objective,
                                                             tau_weight_by_p_jump = tau_weight_by_p_jump,
                                                             tau_ramp = tau_ramp,
                                                             eps_reinit_at_ChEES_handover = eps_reinit_at_ChEES_handover,
                                                             theta_hat_us_rule = theta_hat_us_rule,
                                                             theta_hat_us_freeze_iter = theta_hat_us_freeze_iter,
                                                             burnin_schedule = burnin_schedule,
                                                             metric_adaptation_end_iter = metric_adaptation_end_iter,
                                                             share_tau_ii_across_chains_in_burnin = share_tau_ii_across_chains_in_burnin,
                                                             ##
                                                             burnin_algorithm = burnin_algorithm,
                                                             diffusion_HMC = diffusion_HMC,
                                                             diffusion_HMC_integrator = diffusion_HMC_integrator,
                                                             partitioned_HMC = partitioned_HMC,
                                                             ##
                                                             clip_iter = clip_iter,
                                                             clip_iter_tau = clip_iter_tau,
                                                             ##
                                                             n_refresh = n_refresh,
                                                             use_proposed = use_proposed,
                                                             ##
                                                             beta1_adam = beta1_adam,
                                                             beta2_adam = beta2_adam,
                                                             eps_adam = eps_adam,
                                                             ##
                                                             force_autodiff = force_autodiff,
                                                             force_PartialLog = force_PartialLog,
                                                             multi_attempts = multi_attempts,
                                                             ##
                                                             force_autodiff_for_metric = force_autodiff_for_metric,
                                                             force_PartialLog_for_metric = force_PartialLog_for_metric,
                                                             force_multi_attempts_for_metric = force_multi_attempts_for_metric,
                                                             ##
                                                             vect_type = vect_type,
                                                             Phi_type = Phi_type,
                                                             inv_Phi_type = inv_Phi_type,
                                                             ##
                                                             metric_type_main = metric_type_main,
                                                             metric_shape_main = metric_shape_main,
                                                             ratio_M_main = ratio_M_main,
                                                             interval_width_main = interval_width_main,
                                                             ##
                                                             M_decay_type = M_decay_type,
                                                             M_decay_power = M_decay_power,
                                                             M_decay_scale = M_decay_scale,
                                                             ##
                                                             metric_type_nuisance = metric_type_nuisance,
                                                             metric_shape_nuisance = metric_shape_nuisance,
                                                             ratio_M_nuisance = ratio_M_nuisance,
                                                             interval_width_nuisance = interval_width_nuisance,
                                                             ##
                                                             max_tau_main = max_tau_main,
                                                             max_tau_nuisance = max_tau_nuisance,
                                                             ##
                                                             max_eps_main = max_eps_main,
                                                             max_eps_nuisance = max_eps_nuisance,
                                                             ##
                                                             max_L = max_L,
                                                             ##
                                                             n_nuisance_to_track = n_nuisance_to_track,
                                                             ##
                                                             n_threads_WCP = n_threads_WCP_burnin,
                                                             ##
                                                             time_pre_burnin = time_pre_burnin,
                                                             ##
                                                             metric_start_iter = NULL,
                                                             ##
                                                             eps_init = NULL,
                                                             ##
                                                             learning_rate_initial = learning_rate_initial,
                                                             learning_rate_initial_iter = learning_rate_initial_iter,
                                                             ##
                                                             eps_initial = eps_initial,
                                                             eps_initial_iter = eps_initial_iter,
                                                             ##
                                                             metric_estimator = metric_estimator)
                                                            
                
                {

                          theta_main_vectors_all_chains_input_from_R <- burnin_object$theta_main_vectors_all_chains_input_from_R  # inits stored here
                          theta_nuisance_vectors_all_chains_input_from_R <- burnin_object$theta_us_vectors_all_chains_input_from_R
        
                          Model_args_as_Rcpp_List <-  burnin_object$Model_args_as_Rcpp_List
                          EHMC_args_as_Rcpp_List  <-  burnin_object$EHMC_args_as_Rcpp_List 
                          EHMC_Metric_as_Rcpp_List <- burnin_object$EHMC_Metric_as_Rcpp_List
                          EHMC_burnin_as_Rcpp_List <- burnin_object$EHMC_burnin_as_Rcpp_List
        
                          time_burnin <- burnin_object$time_burnin
                          # time_burnin <- time_burnin + time_pre_burnin
                          
                          n_chains_burnin <-  burnin_object$n_chains_burnin
                          n_burnin <-  burnin_object$n_burnin

                }

                {
                          
                          if (is.null(n_superchains)) { 
                            n_superchains <- n_chains_sampling
                          }
                          if (n_superchains > n_chains_sampling) {
                            n_superchains <- n_chains_sampling
                          }
                          
                          print(paste("Start of post_burnin_prep_inits fn"))
                          # post_burnin_prep_inits <- R_fn_post_burnin_prep_for_sampling( 
                          #                           n_chains_burnin = n_chains_burnin,
                          #                           n_chains_sampling = n_chains_sampling,
                          #                           n_superchains = n_superchains,
                          #                           n_params_main = n_params_main,
                          #                           n_nuisance = n_nuisance,
                          #                           theta_main_vectors_all_chains_input_from_R,
                          #                           theta_nuisance_vectors_all_chains_input_from_R)
                          ## nuisance_jitter_scale: each sampling chain starts from its superchain's burn-in
                          ## nuisance endpoint PLUS this multiple of the per-coordinate spread of the burn-in
                          ## endpoints. Without it, n_chains_sampling / n_chains_burnin chains share one exact
                          ## nuisance vector, so a badly-placed burn-in chain is inherited by all of them.
                          ## 0 reproduces the old (duplicated) behaviour.
                          post_burnin_prep_inits <- cpp_fn_post_burnin_prep_for_sampling(  n_chains_burnin = n_chains_burnin,
                                                                                           n_chains_sampling = n_chains_sampling,
                                                                                           n_superchains = n_superchains,
                                                                                           n_params_main = n_params_main,
                                                                                           n_nuisance = n_nuisance,
                                                                                           theta_main_in = theta_main_vectors_all_chains_input_from_R,
                                                                                           theta_us_in = theta_nuisance_vectors_all_chains_input_from_R,
                                                                                           nuisance_jitter_scale = nuisance_jitter_scale,
                                                                                           seed = seed)
                          print(paste("End of post_burnin_prep_inits fn"))
                          ## Record the actual shared starting states, including repeated endpoints across nominal superchains.
                          ## Nonzero per-chain nuisance jitter breaks shared FULL states; do not claim valid nesting then.
                          nested_rhat_grouping <-  fn_nested_rhat_grouping_from_burnin(
                              n_chains_sampling = n_chains_sampling,
                              n_superchains = n_superchains,
                              n_chains_burnin = n_chains_burnin,
                              nuisance_jitter_scale = if (n_nuisance == 0) 0 else nuisance_jitter_scale)
        
                          theta_main_vectors_all_chains_input_from_R <- post_burnin_prep_inits$theta_main_vectors_all_chains_input_from_R
                          # theta_us_vectors_all_chains_input_from_R   <- post_burnin_prep_inits$theta_us_vectors_all_chains_input_from_R
                          theta_nuisance_vectors_all_chains_input_from_R <- post_burnin_prep_inits$theta_us_vectors_all_chains_input_from_R

                }

                # EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- S_post_pd
                # EHMC_Metric_as_Rcpp_List$M_dense_main          <- BayesMVP:::Rcpp_solve(S_post_pd)
                # EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- BayesMVP:::Rcpp_Chol(S_post_pd)
                ##
                # EHMC_args_as_Rcpp_List$eps_main <- 0.5/4
                
                cat("SAMPLING WILL USE: eps_main =", EHMC_args_as_Rcpp_List$eps_main,
                    "| tau_main =", EHMC_args_as_Rcpp_List$tau_main,
                    "| implied L =", ceiling(EHMC_args_as_Rcpp_List$tau_main / EHMC_args_as_Rcpp_List$eps_main), "\n")
                
                # cat("metric check:",
                #     max(abs(EHMC_Metric_as_Rcpp_List$M_dense_main %*% EHMC_Metric_as_Rcpp_List$M_inv_dense_main - diag(43))),
                #     max(abs(EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol %*% t(EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol) - EHMC_Metric_as_Rcpp_List$M_inv_dense_main)), "\n")
                
                {

                          gc()
                          ##
                          ## External Stan models use the grainsize in their JSON data, not this built-in argument matrix.
                          ## Writing [4] into their absent Model_args_ints creates a vector and the native converter rejects it.
                          if (Model_type != "Stan") {
                            num_chunks_used_in_burnin <- Model_args_as_Rcpp_List$Model_args_ints[4]
                            ##
                            if (is.null(num_chunks_sampling)) {
                              Model_args_as_Rcpp_List$Model_args_ints[4] <- model_args_list$num_chunks
                            } else {
                              # model_args_list$num_chunks <- num_chunks_sampling
                              Model_args_as_Rcpp_List$Model_args_ints[4] <- num_chunks_sampling
                            }
                            num_chunks_used_in_sampling <- Model_args_as_Rcpp_List$Model_args_ints[4]
                          }
                          ##
                          ## ---- Re-lay-out every nuisance-indexed quantity for the sampling chunking.
                          ##
                          ## The built-in C++ log densities store the nuisance vector chunk by chunk (see
                          ## fn_nuisance_chunk_layout_positions), so entry i belongs to a DIFFERENT (observation,
                          ## test) when the number of chunks changes. Without this, a burn-in with 10 chunks handed
                          ## to a sampler with 25 gives every chain latent variables assigned to the wrong
                          ## observations and a scrambled diffusion centre theta_hat_us for the whole sampling
                          ## phase: mass divergences at the start of sampling and chains stuck for good.
                          ## External Stan models keep their own (chunk-independent) layout and are left alone.
                          ##
                          if ((Model_type != "Stan") && (n_nuisance > 0) &&
                              (num_chunks_used_in_sampling != num_chunks_used_in_burnin)) {
                                N_for_remap       <- init_object$model_args_list$N
                                n_tests_for_remap <- init_object$model_args_list$n_tests
                                if (is.null(N_for_remap) || is.null(n_tests_for_remap) || (N_for_remap * n_tests_for_remap != n_nuisance)) {
                                  stop("Cannot re-lay-out the nuisance vector for num_chunks_sampling != num_chunks_burnin: ",
                                       "expected n_nuisance = N * n_tests. Use the same chunk count for burn-in and sampling.")
                                }
                                nuisance_chunk_remap_index <- fn_nuisance_chunk_remap_index( N             = N_for_remap,
                                                                                             n_tests       = n_tests_for_remap,
                                                                                             n_chunks_from = num_chunks_used_in_burnin,
                                                                                             n_chunks_to   = num_chunks_used_in_sampling,
                                                                                             vect_type     = Model_args_as_Rcpp_List$Model_args_strings[1])
                                ##
                                ## Re-orders the rows of anything indexed by nuisance coordinate (vector of length
                                ## n_nuisance, or matrix with n_nuisance rows); returns anything else unchanged:
                                fn_remap_nuisance_indexed_rows <- function(object_to_remap) {
                                  if (is.numeric(object_to_remap) && is.matrix(object_to_remap) && nrow(object_to_remap) == n_nuisance) {
                                    return(object_to_remap[nuisance_chunk_remap_index, , drop = FALSE])
                                  }
                                  if (is.numeric(object_to_remap) && !is.matrix(object_to_remap) && length(object_to_remap) == n_nuisance) {
                                    return(object_to_remap[nuisance_chunk_remap_index])
                                  }
                                  object_to_remap
                                }
                                ##
                                theta_nuisance_vectors_all_chains_input_from_R <- fn_remap_nuisance_indexed_rows(theta_nuisance_vectors_all_chains_input_from_R)
                                ##
                                ## Only per-coordinate VALUES move. Position lists (index_us, index_main, ...) say
                                ## WHERE the nuisance block sits in the parameter vector, not which entry is which,
                                ## so they must not be permuted.
                                names_of_remapped_fields <- character(0)
                                for (metric_field_name in names(EHMC_Metric_as_Rcpp_List)) {
                                  if (grepl("^index", metric_field_name) || is.integer(EHMC_Metric_as_Rcpp_List[[metric_field_name]])) next
                                  metric_field_remapped <- fn_remap_nuisance_indexed_rows(EHMC_Metric_as_Rcpp_List[[metric_field_name]])
                                  if (!identical(metric_field_remapped, EHMC_Metric_as_Rcpp_List[[metric_field_name]])) {
                                    names_of_remapped_fields <- c(names_of_remapped_fields, paste0("Metric$", metric_field_name))
                                  }
                                  EHMC_Metric_as_Rcpp_List[[metric_field_name]] <- metric_field_remapped
                                }
                                for (burnin_field_name in names(EHMC_burnin_as_Rcpp_List)) {
                                  if (grepl("^index", burnin_field_name) || is.integer(EHMC_burnin_as_Rcpp_List[[burnin_field_name]])) next
                                  burnin_field_remapped <- fn_remap_nuisance_indexed_rows(EHMC_burnin_as_Rcpp_List[[burnin_field_name]])
                                  if (!identical(burnin_field_remapped, EHMC_burnin_as_Rcpp_List[[burnin_field_name]])) {
                                    names_of_remapped_fields <- c(names_of_remapped_fields, paste0("burnin$", burnin_field_name))
                                  }
                                  EHMC_burnin_as_Rcpp_List[[burnin_field_name]] <- burnin_field_remapped
                                }
                                cat("nuisance layout remapped for sampling: num_chunks ", num_chunks_used_in_burnin, " -> ",
                                    num_chunks_used_in_sampling, " (inits + ", paste(names_of_remapped_fields, collapse = ", "), ")\n", sep = "")
                          }
                          ##
                          ## ---- keep the per-chain log-lik trace during sampling? (read by the C++ as an optional list element):
                          ##
                          if (!store_log_lik_trace && (parallel_method == "OpenMP")) {
                              stop("store_log_lik_trace = FALSE is only implemented for parallel_method = 'RcppParallel'.")
                          }
                EHMC_args_as_Rcpp_List$store_log_lik_trace <- store_log_lik_trace
                EHMC_args_as_Rcpp_List$record_kinetic_energy_tau_derivatives <- FALSE
                          ##
                          tictoc::tic("post-burnin timer")
                          ##
                          if (Model_type == "Stan" && !is.null(stan_grainsize_sampling)) {
                              Stan_data_list$grainsize <- stan_grainsize_sampling
                              ## Sampling workers load this phase's content-addressed data file when constructed.
                              Model_args_as_Rcpp_List$json_file_path <- convert_stan_data_list_to_JSON(
                                  stan_data_list = Stan_data_list,
                                  pkg_data_dir = dirname(init_object$json_file_path))
                          }
                          ##
                          if (Model_type != "Stan") {
                              Model_args_as_Rcpp_List$model_so_file <- "none"
                              Model_args_as_Rcpp_List$json_file_path <- "none"
                          }
        
                          Model_args_as_Rcpp_List$n_nuisance
                          #  RcppParallel::setThreadOptions(numThreads = n_chains_sampling); #### BOOKMARK
                          RcppParallel::setThreadOptions(numThreads = n_chains_sampling * n_threads_WCP_sampling);
                          
                          # if (parallel_method == "OpenMP") { 
                          #   fn <- BayesMVP:::Rcpp_fn_OpenMP_EHMC_sampling
                          # } else { ###  use RcppParallel
                          #   fn <- BayesMVP:::Rcpp_fn_RcppParallel_EHMC_sampling
                          # }
                           
                          if (Model_type == "Stan") {
                             N <- 1000
                             latent_results <- rlogis(n = N, location =  0.0)
                             y <- ifelse(latent_results > 0, 1, 0)
                             y <- matrix(data = c(y), ncol = 1)
                          } 
                          
                           ### Call C++ parallel sampling function
                          print(paste("Start of sampling"))
                          if (parallel_method == "OpenMP") {
                             sampling_object <- BayesMVP:::Rcpp_fn_OpenMP_EHMC_sampling( n_threads_R = n_chains_sampling,
                                                                                            sample_nuisance_R = sample_nuisance,
                                                                                            n_nuisance_to_track = n_nuisance_to_track,
                                                                                            seed_R = seed,
                                                                                            iter_one_by_one = FALSE,
                                                                                            n_iter_R = n_iter,
                                                                                            partitioned_HMC_R = partitioned_HMC,
                                                                                            diffusion_HMC_R = diffusion_HMC,
                                                                                            Model_type_R = Model_type,
                                                                                            force_autodiff_R = force_autodiff,
                                                                                            force_PartialLog = force_PartialLog,
                                                                                            multi_attempts_R = multi_attempts,
                                                                                            theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                                                                            theta_us_vectors_all_chains_input_from_R = theta_nuisance_vectors_all_chains_input_from_R,
                                                                                            y =  y,  ## only used in C++ for mnl models! (data passed via Stan_data_list / JSON for Stan models!)
                                                                                            Model_args_as_Rcpp_List =  Model_args_as_Rcpp_List,
                                                                                            EHMC_args_as_Rcpp_List =   EHMC_args_as_Rcpp_List,
                                                                                            EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                                                            n_threads_WCP = n_threads_WCP_sampling)


                          } else {
                               sampling_object <- BayesMVP:::Rcpp_fn_RcppParallel_EHMC_sampling(   n_threads_R = n_chains_sampling,
                                                          sample_nuisance_R = sample_nuisance,
                                                          n_nuisance_to_track = n_nuisance_to_track,
                                                          seed_R = seed,
                                                          iter_one_by_one = FALSE,
                                                          n_iter_R = n_iter,
                                                          partitioned_HMC_R = partitioned_HMC,
                                                          diffusion_HMC_R = diffusion_HMC,
                                                          Model_type_R = Model_type,
                                                          force_autodiff_R = force_autodiff,
                                                          force_PartialLog = force_PartialLog,
                                                          multi_attempts_R = multi_attempts,
                                                          theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                                          theta_us_vectors_all_chains_input_from_R = theta_nuisance_vectors_all_chains_input_from_R,
                                                          y =  y,  ## only used in C++ for mnl models! (data passed via Stan_data_list / JSON for Stan models!) 
                                                          Model_args_as_Rcpp_List =  Model_args_as_Rcpp_List,
                                                          EHMC_args_as_Rcpp_List =   EHMC_args_as_Rcpp_List,
                                                          EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                          use_disk = use_disk,
                                                          trace_dir = use_disk_path,
                                                          n_threads_WCP = n_threads_WCP_sampling)
                          }
                          print(paste("End of sampling"))
                           
                           # str(sampling_object)
                          
                          try({
                              sampling_toc_result <- tictoc::toc(log = TRUE)
                              print(sampling_toc_result)
                              tictoc::tic.clearlog()
                              ## elapsed seconds straight from toc()'s numbers. Parsing the "X sec elapsed" text with "\\d+\\.\\d+"
                              ## gave NA whenever the time rounded to a whole second (tictoc then prints e.g. "4 sec elapsed"):
                              time_sampling <- as.numeric(sampling_toc_result$toc - sampling_toc_result$tic)
                          })
        
                          print(paste("time_sampling = ",  time_sampling))
                          
                          gc()
                          
                }
                
                time_total <- NULL
                try({
                   time_total <- time_sampling + time_burnin
                   ##
                   print(paste("time_pre_burnin = ",  time_pre_burnin))
                   print(paste("time_burnin = ",  time_burnin))
                   print(paste("time_total = ",  time_total))
                })

  out_list <- list(init_object = init_object,
                   burnin_object = burnin_object,
                   burnin_schedule = burnin_object$burnin_schedule,
                   burnin_profile = if (isTRUE(debug_burnin_timing)) list(
                       pre_burnin = pre_burnin_object$burnin_profile,
                       main_burnin = burnin_object$burnin_profile) else NULL,
                   sampling_object = sampling_object,
                   ##
                   test_perm = test_perm,
                   test_inv_perm = test_inv_perm,
                   ##
                   LR_main = LR_main, 
                   LR_us = LR_us, 
                   adapt_delta = adapt_delta,
                   n_chains_burnin = n_chains_burnin,
                   burnin_TBB_pool_equals_n_chains = burnin_TBB_pool_equals_n_chains,
                   n_burnin = n_burnin,
                   metric_type_main = metric_type_main,
                   metric_shape_main = metric_shape_main,
                   metric_type_nuisance = metric_type_nuisance,
                   metric_shape_nuisance = metric_shape_nuisance,
                   ##
                   diffusion_HMC = diffusion_HMC,
                   diffusion_HMC_integrator = diffusion_HMC_integrator,
                   partitioned_HMC = partitioned_HMC,
                   n_superchains = n_superchains,
                   nested_rhat_grouping = nested_rhat_grouping,
                   interval_width_main = interval_width_main,
                   interval_width_nuisance = interval_width_nuisance,
                   ##
                   force_autodiff = force_autodiff,
                   force_PartialLog = force_PartialLog,
                   multi_attempts = multi_attempts,
                   autodiff_fallback = autodiff_fallback,
                   ##
                   time_burnin = time_burnin,
                   time_sampling = time_sampling,
                   time_total = time_total)
  
  
  return(out_list)

}
