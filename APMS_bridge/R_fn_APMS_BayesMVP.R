## ==============================================================================================
## R_fn_APMS_BayesMVP.R
## ----------------------------------------------------------------------------------------------
## BayesMVP-specific helpers for the APMS 2007 four-class (joint autism/BPD) LC-MVOP model.
##
## These helpers sit ON TOP of the existing application data builders, prior calculations and
## model definitions in R_fn_sim_4class_joint_LC_MVOP_data.R (which is sourced UNCHANGED). They
## do NOT modify the Stan model and do NOT re-implement the likelihood: the model is compiled
## and evaluated by BridgeStan through the updated BayesMVP external-Stan interface.
##
## BayesMVP treats the FIRST declaration of the Stan parameters block as the NUISANCE block
## (here: "u_raw", the N x n_tests latent matrix). Its UNCONSTRAINED dimension is detected
## automatically (no coordinate counting, no renaming needed). sample_nuisance = TRUE uses
## partitioned + diffusion-pathspace HMC for it; sample_nuisance = FALSE means NO nuisance
## block (every declared parameter is main) - the two are distinct situations.
##
## NOTE on settings: cmdstanr's NUTS settings do NOT have exact BayesMVP equivalents. The
## established APMS settings are mapped as follows (see RUN_BayesMVP.md for the details):
##      iter_warmup = 250            ->  n_burnin = 250      (adaptive warm-up)
##      iter_sampling = 100          ->  n_iter   = 100      (post-warm-up draws per chain)
##      adapt_delta = 0.65           ->  adapt_delta = 0.65  (target Metropolis acceptance)
##      max_treedepth = 8            ->  max_L = 256         (max leapfrog steps = 2^8; NOT an
##                                                            exact equivalent - see the doc)
##      Chain counts, WCP threads and requested chunks are independent for burn-in and sampling;
##      see fn_sampler_settings_APMS_BayesMVP() below for the active defaults.
## ==============================================================================================


## -| --------- 1. Sampler settings, as ONE explicit named list ---------------------------------
##
## Everything that controls the MCMC run lives here, so the prior check and the posterior runs
## can reuse the SAME settings while overriding only their own chain / iteration counts.
## These are the established APMS experiment settings (BayesMVP flavour); nothing is silently
## replaced with detectCores() and parallelism is never disabled behind the user's back.
##
## ---- Explicit named option lists ----------------------------------------------------------------------------------------------------
##
fn_validate_named_options_APMS <-  function(options, 
                                            label) {

        if (!is.list(x = options) || (length(x = options) &&
            (is.null(names(x = options)) || anyNA(names(x = options)) || any(!nzchar(names(x = options))) ||
             anyDuplicated(x = names(x = options))))) {
            stop(label, " must be a list with unique, non-empty argument names.")
        }
        invisible(x = TRUE)

}


  
## rank LR_init tau_initial            metric runs nRhat_ok         ESS_grad ESS_sec min_ESS min_ESS_main burnin_s   to_target_s  Rhat
##    1    0.15          pi chain_mean_scaled    2      2/2 30.6 [27.8-33.3]    1397    4162          484      2.9 4.8 [4.5-5.1] 1.042

#' fn_sampler_settings_APMS_BayesMVP
#' @param tau_initial Initial adaptive path length; NULL selects pi for pooled metrics or 2*pi for chain_mean metrics.
#' @param n_threads_WCP_burnin Within-chain threads per burn-in chain.
#' @param n_threads_WCP_sampling Within-chain threads per sampling chain.
#' @param num_chunks_burnin Requested burn-in chunks; Stan grainsize is derived as ceiling(N / num_chunks_burnin).
#' @param num_chunks_sampling Requested sampling chunks; Stan grainsize is derived as ceiling(N / num_chunks_sampling).
#' @param M_decay_scale Metric decay scale. When omitted, calculated from n_adapt (or n_burnin); explicit NULL is retained.
#' @param autodiff_fallback Optional same-position autodiff fallback for built-in models with multi_attempts enabled.
#' Defaults to options(BayesMVP_autodiff_fallback), or FALSE when unset; NULL leaves the fitting process's option unchanged.
#' The APMS external-Stan bridge does not use the built-in evaluator fallbacks.
#' @param summary_options Post-processing arguments for summary(), separate from the direct sampler arguments.
#' @details All sampler controls are direct named arguments. There is no sampler_options list or ellipsis.
#' @export
fn_sampler_settings_APMS_BayesMVP <-  function(
    ##
    ## ---- based on benchmarks (see "R_benchmark_AVX_direct_call_RStudio.R" file:
    ##
    ## ---- MCMC iterations, adaptation and chain layout:
    ##
    tau_initial                             = pi,
    ##
    n_burnin                                = 125,
    n_iter                                  = 125,
    ##
    adapt_delta                             = 0.80,
    learning_rate                           = 0.075, ## ----
    ##
    n_chains_burnin                         = 4,
    n_threads_WCP_burnin                    = 16,
    num_chunks_burnin                       = 64,
    ##
    n_chains_sampling                       = 64,
    n_threads_WCP_sampling                  = 1,
    num_chunks_sampling                     = 32,
    ##
    seed                                    = 123,
    n_superchains                           = n_chains_sampling,
    ##
    max_L                                   = 1024,
    parallel_method                         = "RcppParallel",
    ##
    ## ---- Joint sampler and metrics:
    ##
    partitioned_HMC                         = FALSE,
    diffusion_HMC                           = TRUE,
    diffusion_HMC_integrator                = "kick_flow_kick",
    ##
    sample_nuisance                         = TRUE,
    ##
    metric_type_main                        = "Empirical",
    metric_shape_main                       = "dense",
    ratio_M_main                            = 0.90,
    interval_width_main                     = 1,
    ##
    metric_type_nuisance                    = "uniform_diag",
    metric_shape_nuisance                   = "diag",
    ratio_M_nuisance                        = 0.90,
    interval_width_nuisance                 = 1,
    ##
    metric_estimator                        = "chain_mean_scaled",
    ##
    M_decay_type                            = "inverse",
    M_decay_power                           = 0.50,
    M_decay_scale                           = NULL,
    ##
    ## ---- Step size, path length and learning-rate controls:
    ##
    max_tau_main                            = 50.0,
    max_tau_nuisance                        = 50.0,
    max_eps_main                            = 1.00,
    max_eps_nuisance                        = 1.00,
    ##
    learning_rate_initial                   = 0.15, ## ----
    learning_rate_initial_iter              = NULL,
    ##
    eps_initial                             = NULL,
    eps_initial_iter                        = NULL,
    ##
    use_proposed                            = FALSE,
    ##
    beta1_adam                              = 0.00,
    beta2_adam                              = 0.95,
    eps_adam                                = 1e-8,
    ##
    ## ---- Progress, storage and ordering:
    ##
    n_refresh                               = 5,
    n_nuisance_to_track                     = 0,
    use_disk                                = FALSE,
    reorder_cols_MVP                        = FALSE,
    ##
    ## ---- Burn-in and trajectory controls (NULL delegates to the sampler where supported):
    ##
    n_adapt                                 = NULL,
    gap                                     = NULL,
    ##
    ## 125 iterations: match the package/PS7 ramp, with tau handover at iteration 50 (40%).
    ## Other lengths retain the package schedule; explicit arguments still take precedence.
    ##
    ## NULL now delegates to fn_burnin_adaptation_schedule through sample_model, including the 125-iteration case.
    clip_iter                               = NULL,
    clip_iter_tau                           = NULL,
    ##
    burnin_algorithm                        = NULL,
    ##
    tau_mult                                = NULL,
    manual_tau                              = NULL,
    tau_if_manual                           = NULL,
    tau_if_manual_in_L_units                = NULL,
    ##
    tau_objective                           = "ChEES_per_tau", ## ----
    tau_weight_by_p_jump                    = TRUE,
    ##
    # tau_ramp                                = "original", ## ----
    tau_ramp                                = "staged", ## ----
    ##
    eps_reinit_at_ChEES_handover            = FALSE,
    ##
    theta_hat_us_rule                       = "running_mean_frozen",
    theta_hat_us_freeze_iter                = NULL,
    burnin_schedule                        = "automatic",
    metric_adaptation_end_iter             = NULL,
    ##
    pre_burnin_n_iter                       = 50,
    pre_burnin_L                            = 10,
    ##
    share_tau_ii_across_chains_in_burnin    = TRUE,
    burnin_TBB_pool_equals_n_chains         = FALSE,  ## APMS uses external Stan models, which require the full TBB budget.
    ##
    ## ---- Other sample() controls (data and initial values still belong to the fitting function):
    ##
    debug_burnin_timing                     = TRUE,
    debug                                   = FALSE,
    ##
    stream                                  = NULL,
    nuisance_jitter_scale                   = 0.0,
    ##
    store_log_lik_trace                     = FALSE,
    use_disk_path                           = "/tmp/hmc_traces",
    ##
    test_perm_override                      = NULL,
    n_nuisance_override                     = NULL,
    ##
    force_recompile                         = FALSE,
    ##
    force_autodiff                          = NULL,
    force_PartialLog                        = NULL,
    multi_attempts                          = NULL,
    ##
    ## Built-in lp_grad fallback only, not BridgeStan's evaluation of an external .stan model.
    ## Stored in settings so separate fitting workers receive the selected option too.
    autodiff_fallback                       = getOption(x = "BayesMVP_autodiff_fallback", default = FALSE),
    ##
    force_autodiff_for_metric               = TRUE,
    force_PartialLog_for_metric             = FALSE,
    force_multi_attempts_for_metric         = FALSE,
    ##
    vect_type                               = NULL,
    Phi_type                                = "Phi",
    inv_Phi_type                            = "inv_Phi",
    ##
    ## ---- Post-processing only: these arguments go to summary(), not sample():
    ##
    summary_options = list(save_log_lik_trace = FALSE, compute_nested_rhat = FALSE)
) {

    ##
    n_adapt_for_metric_decay <-  if (is.null(x = n_adapt)) n_burnin - round(n_burnin / 10) else n_adapt
    ## Omitted M_decay_scale follows the chosen burn-in length; explicit NULL is passed through unchanged.
    if (missing(M_decay_scale)) M_decay_scale <-  n_adapt_for_metric_decay / 100
    ##
    # if (is.null(learning_rate)) {
    #   ## learning_rate depends on n_burnin (established PS7 pairing; LR increases as
    #   ## the warm-up shrinks, with the active study combo for n_burnin = 250 -> 0.075):
    #   learning_rate_by_n_burnin <-  c(`1000` = 0.0125,
    #                                   ##
    #                                   `500`  = 0.025,
    #                                   ##
    #                                   #`250`  = 0.05,
    #                                   `250`  = 0.075,
    #                                   ##
    #                                   `125`  = 0.0875)
    #   ##
    #   learning_rate <-  unname(learning_rate_by_n_burnin[as.character(n_burnin)])
    # }
    # learning_rate <- 0.075
    ##
    settings <-  list( ##
          ## ---- MCMC structure (PS7 baseline: 250 warm-up, 100 draws):
          seed                           = seed,
          ##
          n_chains_burnin                = n_chains_burnin,
          ##
          n_chains_sampling              = n_chains_sampling,
          n_superchains                  = n_superchains,
          ##
          n_burnin                       = n_burnin,
          n_iter                         = n_iter,
          ##
          ## ---- adaptation / acceptance (PS7):
          ##
          adapt_delta                    = adapt_delta,
          learning_rate                  = learning_rate,
          ##
          ## ---- initial adaptive path length (e.g. pi, 2*pi, or another positive value):
          ##
          tau_initial                    = tau_initial,
          ##
          ## ---- max leapfrog steps (PS7):
          ##
          max_L                          = max_L,
          ##
          ## ---- parallelisation / model concurrency:
          ##
          parallel_method                = parallel_method,
          n_threads_WCP_burnin           = n_threads_WCP_burnin,
          n_threads_WCP_sampling         = n_threads_WCP_sampling,
          num_chunks_burnin              = num_chunks_burnin,
          num_chunks_sampling            = num_chunks_sampling,
          ##
          use_disk                       = use_disk,
          ##
          ## ---- JOINT diffusion-pathspace HMC for the nuisance (u_raw) block -
          ## the paper's algorithm samples nuisance + main jointly in one leapfrog:
          ## partitioned_HMC = FALSE + diffusion_HMC = TRUE (the dual sampler).
          partitioned_HMC                = partitioned_HMC,
          diffusion_HMC                  = diffusion_HMC,
          ##
          ## ---- joint-diffusion integrator ordering ("kick_flow_kick" is the default;
          ## "flow_kick_flow" selects the Alenlov-Doucet-Lindsten ordering):
          ##
          diffusion_HMC_integrator       = diffusion_HMC_integrator,
          # diffusion_HMC_integrator = "flow_kick_flow", ## Alenlov-Doucet-Lindsten
          ##
          sample_nuisance                = sample_nuisance,
          ##
          ## ---- metrics and decay (PS7):
          ##
          metric_type_main               = metric_type_main,
          ##
          # metric_shape_main = "diag",
          metric_shape_main              = metric_shape_main,
          ##
          ratio_M_main                   = ratio_M_main,
          interval_width_main            = interval_width_main,
          ##
          metric_type_nuisance           = metric_type_nuisance,
           ## metric_type_nuisance = "unit",
          ##
          metric_shape_nuisance          = metric_shape_nuisance,
          ##
          ratio_M_nuisance               = ratio_M_nuisance,
          interval_width_nuisance        = interval_width_nuisance,
          ##
          metric_estimator               = metric_estimator,  ## "pooled" (correct: variance of the draws) | "chain_mean" (legacy: variance of the cross-chain MEAN, ~Sigma / n_chains)
          # metric_estimator = "chain_mean",
          ##
          M_decay_type                   = M_decay_type,
          M_decay_power                  = M_decay_power,
          M_decay_scale                  = M_decay_scale,  ## PS7: n_adapt / 100 (NULL -> n_adapt / 5)
          ##
          ## ---- step-size / path-length bounds (PS7):
          ##
          max_tau_main                   = max_tau_main,
          max_tau_nuisance               = max_tau_nuisance,
          max_eps_main                   = max_eps_main,
          max_eps_nuisance               = max_eps_nuisance,
          ##
          ## ---- burnin progress reporting: print every n_refresh iterations (Stan's 'refresh'
          ## convention). NULL -> n_burnin / 20, i.e. about twenty reports across the burnin:
          ##
          n_refresh                      = n_refresh,
          ##
          ## ---- hold the ADAM learning rate HIGH for the first learning_rate_initial_iter burnin
          ## iterations (NULL -> n_burnin / 10, i.e. 50 at n_burnin = 500), then drop it to
          ## 'learning_rate' above. Set learning_rate_initial = NA to switch the hold off:
          ##
          # learning_rate_initial      = 0.10,
          learning_rate_initial          = learning_rate_initial,  ## alternative; the existing first value (0.10) was effective
          ##
          learning_rate_initial_iter     = learning_rate_initial_iter,
          ##
          ## ---- the analogous warm start on the STEP-SIZE itself, off by default:
          ##
          eps_initial                    = eps_initial,
          eps_initial_iter               = eps_initial_iter,
          ##
          # use_proposed = TRUE, ## -----------------
          use_proposed                   = use_proposed,  ## -----------------
          ##
          beta1_adam                     = beta1_adam,
          beta2_adam                     = beta2_adam,
          eps_adam                       = eps_adam,
          ##
          n_nuisance_to_track            = n_nuisance_to_track,
          ##
          ## ---- column reordering. Supported for built-in LC_MVP models, and for external
          ## Stan models ONLY when 'y' is the sole test-indexed entry in the Stan data - which
          ## is NOT the case here (test_condition, X, prior_beta_mean / _sd, prior_dirichlet_alpha,
          ## is_perfect_ref, ... are all per-test), so setting this TRUE would be refused with a
          ## warning banner and the run would proceed unreordered anyway:
          ##
          reorder_cols_MVP               = reorder_cols_MVP)
    ##
    ## Forward every remaining named argument, including NULL. Defaults are declared above, not in an override list.
    additional_sampler_argument_names <-  c(
        "n_adapt", "gap", "clip_iter",
        "clip_iter_tau", "burnin_algorithm", "tau_mult",
        "manual_tau", "tau_if_manual", "tau_if_manual_in_L_units",
        "tau_objective", "tau_weight_by_p_jump", "tau_ramp",
        "eps_reinit_at_ChEES_handover", "theta_hat_us_rule", "theta_hat_us_freeze_iter",
        "burnin_schedule", "metric_adaptation_end_iter",
        "pre_burnin_n_iter", "pre_burnin_L",
        "share_tau_ii_across_chains_in_burnin", "burnin_TBB_pool_equals_n_chains", "debug_burnin_timing",
        "debug", "stream", "nuisance_jitter_scale",
        "store_log_lik_trace", "use_disk_path", "test_perm_override",
        "n_nuisance_override", "force_recompile", "force_autodiff",
        "force_PartialLog", "multi_attempts", "autodiff_fallback", "force_autodiff_for_metric",
        "force_PartialLog_for_metric", "force_multi_attempts_for_metric", "vect_type",
        "Phi_type", "inv_Phi_type")
    ## List subassignment deliberately preserves explicitly supplied NULL values.
    settings[additional_sampler_argument_names] <-  mget(x = additional_sampler_argument_names,
                                                         envir = environment(),
                                                         inherits = FALSE)
    ##
    fn_validate_named_options_APMS(options = summary_options, label = "summary_options")
    settings$summary_options <-  summary_options
    settings

}


## -| --------- 2. Settings validation gate -------------------------------------------------------
##
## Claude/GPT-review feedback: the sampler refuses to run unless the algorithm- and
## storage-critical settings are SUPPLIED explicitly (no silent hard-coded defaults).
## This catches a settings list that lost a field during editing, and fails fast on
## unsupported integrator/flag combinations instead of sampling with the wrong algorithm.
##
#' fn_validate_settings_APMS_BayesMVP
#' @export
fn_validate_settings_APMS_BayesMVP <-  function(settings) {

      fn_validate_named_options_APMS(options = settings, label = "settings")
      if (!is.null(x = settings$burnin_post_adapt_iter)) {
          stop("burnin_post_adapt_iter has been removed. Use n_burnin for the total burn-in and n_adapt for adaptation.")
      }
      reserved_fields <-  intersect(x = names(x = settings), y = c("init_lists_per_chain", "Stan_data_list", "model_args_list"))
      if (length(x = reserved_fields)) stop("Pass data and initial values through the bridge arguments, not settings: ",
                                           paste(reserved_fields, collapse = ", "))
      ##
      required_fields <-  c("partitioned_HMC", "diffusion_HMC", "M_decay_type",
                            "M_decay_power", "use_disk", "metric_estimator")
      ##
      missing_fields <-  required_fields[!vapply( X = required_fields,
                                                  FUN = function(field) !is.null(settings[[field]]),
                                                  FUN.VALUE = logical(1))]
      ##
      if (length(missing_fields) > 0) {
          stop( "APMS BayesMVP settings are missing required fields: ",
                paste(missing_fields, collapse = ", "),
                ". Supply them explicitly (see fn_sampler_settings_APMS_BayesMVP()).")
      }
      ##
      ## 'eps_initial' may legitimately be NA / NULL (= warm start off), so the check is on
      ## PRESENCE rather than non-NULL: a settings list that simply lost the field would
      ## otherwise silently disable the warm start.
      for (hold_field in c("learning_rate_initial", "eps_initial")) {

          if (!(hold_field %in% names(settings))) {
              stop( "APMS BayesMVP settings are missing '", hold_field, "'. Set it explicitly (e.g. 0.10 to hold, ",
                    "or NA to disable) - see fn_sampler_settings_APMS_BayesMVP().")
          }
          ##
          if (!is.null(settings[[hold_field]])) {
              if (length(settings[[hold_field]]) != 1L ||
                  !(is.na(settings[[hold_field]]) ||
                    (is.numeric(settings[[hold_field]]) && is.finite(settings[[hold_field]]) && settings[[hold_field]] > 0))) {
                  stop("APMS BayesMVP settings: '", hold_field, "' must be a single positive number, or NA / NULL to disable.")
              }
          }

      }
      ##
      for (flag in c("partitioned_HMC", "diffusion_HMC", "use_disk")) {
          if (!is.logical(settings[[flag]]) || length(settings[[flag]]) != 1L || is.na(settings[[flag]])) {
              stop("APMS BayesMVP settings: '", flag, "' must be a single non-NA logical (TRUE/FALSE).")
          }
      }
      ##
      ## ---- Validate this optional flag before constructing / compiling the Stan model:
      ##
      if (!is.null(x = settings$tau_if_manual_in_L_units)) {
          if (!is.logical(x = settings$tau_if_manual_in_L_units) || length(x = settings$tau_if_manual_in_L_units) != 1 ||
              is.na(x = settings$tau_if_manual_in_L_units)) {
              stop("APMS BayesMVP settings: 'tau_if_manual_in_L_units' must be NULL, TRUE or FALSE, not NA.")
          }
      }
      ##
      if (!is.null(x = settings$autodiff_fallback)) {
          if (!is.logical(x = settings$autodiff_fallback) || length(x = settings$autodiff_fallback) != 1 ||
              is.na(x = settings$autodiff_fallback)) {
              stop("APMS BayesMVP settings: 'autodiff_fallback' must be NULL, TRUE or FALSE, not NA.")
          }
      }
      ##
      if (!is.character(settings$metric_estimator) || length(settings$metric_estimator) != 1L ||
          is.na(settings$metric_estimator) ||
          !(settings$metric_estimator %in% c("chain_mean", "pooled", "chain_mean_scaled"))) {
          stop("APMS BayesMVP settings: 'metric_estimator' must be 'pooled', 'chain_mean' or 'chain_mean_scaled'.")
      }
      ##
      for (setting_name in c("n_threads_WCP_burnin", "n_threads_WCP_sampling", "num_chunks_burnin", "num_chunks_sampling")) {
          setting_value <-  settings[[setting_name]]
          if (!is.numeric(x = setting_value) || length(x = setting_value) != 1L || !is.finite(x = setting_value) ||
              setting_value < 1 || setting_value != floor(x = setting_value) || setting_value > .Machine$integer.max) {
              stop("APMS BayesMVP settings: '", setting_name, "' must be a positive integer.")
          }
      }
      ##
      if (!is.null(x = settings$tau_initial)) {
          if (!is.numeric(x = settings$tau_initial) || length(x = settings$tau_initial) != 1L ||
              !is.finite(x = settings$tau_initial) || settings$tau_initial <= 0) {
              stop("APMS BayesMVP settings: 'tau_initial' must be NULL or a single positive finite number.")
          }
      }
      ##
      integrator <-  settings$diffusion_HMC_integrator
      ##
      if (!is.null(integrator)) {
          if (!is.character(integrator) || length(integrator) != 1L || is.na(integrator) ||
              !(integrator %in% c("kick_flow_kick", "flow_kick_flow"))) {
              stop("APMS BayesMVP settings: 'diffusion_HMC_integrator' must be 'kick_flow_kick' or 'flow_kick_flow'.")
          }
          if (identical(integrator, "flow_kick_flow") &&
              (!isTRUE(settings$diffusion_HMC) || !isFALSE(settings$partitioned_HMC))) {
              stop("flow_kick_flow requires diffusion_HMC = TRUE and partitioned_HMC = FALSE.")
          }
      }
      ##
      invisible(TRUE)

}


## -| --------- 3. Per-chain initial values, via the EXISTING fn_make_inits() --------------------
##
## fn_make_inits() (from R_fn_sim_4class_joint_LC_MVOP_data.R) returns a chain_id -> init-values
## function whose values are on the CONSTRAINED scale. BayesMVP unconstrains the COMPLETE init
## list (nuisance + main) itself and only then splits the vector, so no manual u_raw handling
## is needed here.
##
#' fn_make_inits_BayesMVP
#' @export
fn_make_inits_BayesMVP <-  function( stan_data,
                                     prev0 = c(0.985, 0.010, 0.004, 0.001),
                                     n_chains_burnin,
                                     seed = 123) {

      init_fn <-  fn_make_inits( stan_data = stan_data,
                                 prev0 = prev0)
      ##
      lapply( X = seq_len(length.out = n_chains_burnin),
              FUN = function(chain_id) init_fn(chain_id = chain_id))

}


## -| --------- 4. Convergence gate, adapted to the BayesMVP summary tibbles --------------------
##
## The same principle as fn_fit_gate() in the original helper (constant GQ entries have
## undefined R-hat by construction and must not fail the gate silently). BayesMVP's summary
## tibbles carry columns: parameter, mean, sd, `2.5%`, `50%`, `97.5%`, n_eff, Rhat, n_Rhat.
##
#' fn_fit_gate_BayesMVP
#' @export
fn_fit_gate_BayesMVP <-  function( summary_tibble,
                                   max_r_hat_threshold = 1.25,
                                   min_effective_sample_size_threshold = 50) {

      is_active_entry <-  is.finite(x = summary_tibble$sd) & summary_tibble$sd > 0
      ##
      is_monitored_entry <-  is_active_entry & !is.na(x = summary_tibble$Rhat)
      ##
      max_r_hat <-  if (any(is_monitored_entry)) max(summary_tibble$Rhat[is_monitored_entry]) else NA_real_
      ##
      min_effective_sample_size <-  if (any(is_monitored_entry)) min(summary_tibble$n_eff[is_monitored_entry]) else NA_real_
      ##
      has_missing_diagnostic <-  any(is_active_entry & (!is.finite(x = summary_tibble$Rhat) | !is.finite(x = summary_tibble$n_eff)))
      ##
      gate_passed <-  is.finite(x = max_r_hat) && is.finite(x = min_effective_sample_size) &&
          !has_missing_diagnostic && max_r_hat <= max_r_hat_threshold &&
          min_effective_sample_size >= min_effective_sample_size_threshold
      ##
      c(gate_passed = gate_passed,
        max_r_hat = max_r_hat,
        min_effective_sample_size = min_effective_sample_size)

}


## -| --------- 5. Fit ONE model configuration through BayesMVP ----------------------------------
##
#' fn_fit_APMS_model_BayesMVP
#'
#' @param model_i     model configuration (e.g. "M6_imperfect_ord")
#' @param stan_data   the model-specific data list (already subsetted if requested)
#' @param settings    the named-list sampler settings from fn_sampler_settings_APMS_BayesMVP()
#' @param prev0       starting class probabilities for fn_make_inits()
#' @param verbose     print progress
#' @export
fn_fit_APMS_model_BayesMVP <-  function( model_i,
                                         stan_data,
                                         settings,
                                         prev0 = c(0.985, 0.010, 0.004, 0.001),
                                         verbose = TRUE) {

      stopifnot(model_i %in% c("M3_perfect_CI", "M4_perfect_dep", "M5_imperfect_bin", "M6_imperfect_ord"))
  
      fn_validate_settings_APMS_BayesMVP(settings = settings)
      supported_settings <-  c(names(formals(BayesMVP::MVP_model$public_methods$sample)),
                                "Stan_model_file_path", "Stan_cpp_user_header", "Stan_cpp_flags", "stanc_args", "make_args",
                                "summary_options", "compile", "force_recompile", "autodiff_fallback")
      unknown_settings <-  setdiff(x = names(settings), y = supported_settings)
      if (length(unknown_settings)) stop("Unsupported installed BayesMVP $sample() options: ", paste(unknown_settings, collapse = ", "),
                                         ". The updated R6 interface must be installed in the inner package. No model was compiled or fitted.")
      ##
      ## Derive the initial Stan grainsize from burn-in chunks after posterior or prior-check subsetting.
      ## The sampler derives and switches to the sampling grainsize after burn-in.
      stan_data$grainsize <-  as.integer(x = ceiling(x = stan_data$N / settings$num_chunks_burnin))
      ##
      start_time <-  Sys.time()
      ##
      ## ---- inits for the burn-in chains (CONSTRAINED scale, complete lists):
      init_lists_per_chain <-  fn_make_inits_BayesMVP( stan_data = stan_data,
                                                       prev0 = prev0,
                                                       n_chains_burnin = settings$n_chains_burnin,
                                                       seed = settings$seed)
      ##
      ## ---- compile + initialise via the (updated) R6 interface. Stan_data_list is given NOW,
      ## so initialisation happens in $new() (the deferred path is exercised in the tests):
      initialise_start <-  proc.time()[["elapsed"]]
      model_obj <-  BayesMVP::MVP_model$new( Model_type = "Stan",
                                             ##
                                             sample_nuisance = settings$sample_nuisance,
                                             n_nuisance_override = settings$n_nuisance_override,
                                             ##
                                             Stan_data_list = fn_stan_input(stan_data = stan_data),
                                             Stan_model_file_path = settings$Stan_model_file_path,
                                             ##
                                             ## external C++ (e.g. the BayesMVP AVX kernels exposed to Stan): all
                                             ## NULL unless the settings list carries them (see the bridge switch
                                             ## USE_AVX_STAN_EXTERNALS); a header implies --allow-undefined + USER_HEADER.
                                             Stan_cpp_user_header = settings$Stan_cpp_user_header,
                                             Stan_cpp_flags = settings$Stan_cpp_flags,
                                             stanc_args = settings$stanc_args,
                                             make_args = settings$make_args,
                                             ##
                                             compile = if (is.null(settings$compile)) TRUE else settings$compile,
                                             force_recompile = isTRUE(settings$force_recompile))
      initialise_seconds <-  proc.time()[["elapsed"]] - initialise_start
      ##
      if (verbose) {
          cat( sprintf( fmt = "  [%s] n_nuisance = %d   n_params_main = %d   (detected automatically)\n",
                        model_i,
                        model_obj$n_nuisance,
                        model_obj$n_params_main))
      }
      ##
      ## ---- sample:
      sample_arguments <-  list( n_chains_burnin = settings$n_chains_burnin,
                                          init_lists_per_chain = init_lists_per_chain,
                                          ##
                                          parallel_method = settings$parallel_method,
                                          ##
                                          sample_nuisance = settings$sample_nuisance,
                                          n_nuisance_override = NULL,
                                          ##
                                          seed = settings$seed,
                                          n_burnin = settings$n_burnin,
                                          n_iter = settings$n_iter,
                                          ##
                                          n_chains_sampling = settings$n_chains_sampling,
                                          n_superchains = settings$n_superchains,
                                          ##
                                          adapt_delta = settings$adapt_delta,
                                          learning_rate = settings$learning_rate,
                                          ##
                                          tau_initial = settings$tau_initial,
                                          ##
                                          partitioned_HMC = settings$partitioned_HMC,
                                          diffusion_HMC = settings$diffusion_HMC,
                                          diffusion_HMC_integrator = settings$diffusion_HMC_integrator,
                                          ##
                                          metric_type_main = settings$metric_type_main,
                                          metric_shape_main = settings$metric_shape_main,
                                          ratio_M_main = settings$ratio_M_main,
                                          interval_width_main = settings$interval_width_main,
                                          ##
                                          metric_type_nuisance = settings$metric_type_nuisance,
                                          metric_shape_nuisance = settings$metric_shape_nuisance,
                                          ratio_M_nuisance = settings$ratio_M_nuisance,
                                          interval_width_nuisance = settings$interval_width_nuisance,
                                          ##
                                          metric_estimator = settings$metric_estimator,
                                          ##
                                          M_decay_type = settings$M_decay_type,
                                          M_decay_power = settings$M_decay_power,
                                          M_decay_scale = settings$M_decay_scale,
                                          ##
                                          max_tau_main = settings$max_tau_main,
                                          max_tau_nuisance = settings$max_tau_nuisance,
                                          max_eps_main = settings$max_eps_main,
                                          max_eps_nuisance = settings$max_eps_nuisance,
                                          ##
                                          n_refresh = settings$n_refresh,
                                          ##
                                          learning_rate_initial = settings$learning_rate_initial,
                                          learning_rate_initial_iter = settings$learning_rate_initial_iter,
                                          ##
                                          eps_initial = settings$eps_initial,
                                          eps_initial_iter = settings$eps_initial_iter,
                                          max_L = settings$max_L,
                                          ##
                                          use_disk = settings$use_disk,
                                          n_threads_WCP_burnin = settings$n_threads_WCP_burnin,
                                          n_threads_WCP_sampling = settings$n_threads_WCP_sampling,
                                          num_chunks_burnin = settings$num_chunks_burnin,
                                          num_chunks_sampling = settings$num_chunks_sampling,
                                          reorder_cols_MVP = settings$reorder_cols_MVP,
                                          ##
                                          ## nuisance-trace storage. This was in the settings list but NOT forwarded,
                                          ## so the package's auto rule applied - and for an external Stan model that
                                          ## rule keeps the FULL nuisance trace (29,612 x n_iter x n_chains doubles):
                                          ## ~9 GB per worker and minutes of post-processing. 0 = store none.
                                          n_nuisance_to_track = settings$n_nuisance_to_track)
      ##
      sample_argument_names <-  names(x = formals(fun = model_obj$sample))
      constructor_fields <-  c("Stan_model_file_path", "Stan_cpp_user_header", "Stan_cpp_flags", "stanc_args", "make_args",
                               "summary_options", "compile", "force_recompile", "autodiff_fallback")
      unknown_settings <-  setdiff(x = names(x = settings), y = c(sample_argument_names, constructor_fields))
      if (length(x = unknown_settings)) stop("Unsupported BayesMVP settings: ", paste(unknown_settings, collapse = ", "),
                                             ". Check spelling and the installed inner package's $sample() interface.")
      ## Forward every explicitly supplied sampler setting, including options added after this bridge was written.
      ## autodiff_fallback is applied as an R option below, not passed as an unsupported R6 sample() argument.
      forwarded_names <-  setdiff(x = intersect(x = names(x = settings), y = sample_argument_names), y = "autodiff_fallback")
      sample_arguments[forwarded_names] <-  settings[forwarded_names]
      sample_arguments$init_lists_per_chain <-  init_lists_per_chain
      ##
      if (!is.null(x = settings$autodiff_fallback)) {
          previous_autodiff_fallback_option <-  options(BayesMVP_autodiff_fallback = settings$autodiff_fallback)
          on.exit(expr = options(previous_autodiff_fallback_option), add = TRUE)
      }
      if (verbose && isTRUE(x = getOption(x = "BayesMVP_autodiff_fallback", default = FALSE))) {
          cat("  Autodiff fallback requested, but not applicable to this external-Stan fit; BridgeStan evaluates the model.\n")
      }
      ##
      sample_start <-  proc.time()[["elapsed"]]
      model_samples <-  do.call(what = model_obj$sample, args = sample_arguments)
      sample_seconds <-  proc.time()[["elapsed"]] - sample_start

      ##
      ## ---- summaries (uses the storage mode recorded during sampling automatically):
      summary_start <-  proc.time()[["elapsed"]]
      summary_arguments <-  settings$summary_options
      if (is.null(x = summary_arguments)) summary_arguments <-  list(save_log_lik_trace = FALSE, compute_nested_rhat = FALSE)
      fn_validate_named_options_APMS(options = summary_arguments, label = "summary_options")
      unknown_summary <-  setdiff(x = names(x = summary_arguments), y = names(x = formals(fun = model_samples$summary)))
      if (length(x = unknown_summary)) stop("Unsupported summary_options: ", paste(unknown_summary, collapse = ", "))
      model_fit <-  do.call(what = model_samples$summary, args = summary_arguments)
      summary_seconds <-  proc.time()[["elapsed"]] - summary_start
      ##
      elapsed_minutes <-  as.numeric(x = difftime( time1 = Sys.time(),
                                                   time2 = start_time,
                                                   units = "mins"))
      ##
      list( model_i = model_i,
             model_obj = model_obj,
             model_samples = model_samples,
             model_fit = model_fit,
             summary_main = model_fit$get_summary_main(),
             summary_transformed = model_fit$get_summary_transformed(),
             summary_generated_quantities = model_fit$get_summary_generated_quantities(),
             divergences = model_fit$get_divergences(),
             draws = model_fit$get_posterior_draws(),
             wall_time_min = elapsed_minutes,
             settings = settings,
             timing = list(initialise_seconds = initialise_seconds,
                           sample_call_seconds = sample_seconds,
                           summary_seconds = summary_seconds,
                           total_seconds = elapsed_minutes * 60),
             stan_data = stan_data)

}


## -| --------- 6. PRIOR-ONLY check, on its OWN subset configuration -----------------------------
##
## The prior check uses its own chains / warm-up / sampling / subset sizes (4 chains, 200 + 200,
## 1000 PPS rows) and is kept completely separate from the posterior subsetting. prior_only = 1
## bypasses the likelihood, so the number of rows only affects the generated-quantities cost.
##
#' fn_run_prior_check_BayesMVP
#' @export
#' fn_bind_BayesMVP_summaries
#'
#' One data frame from the three BayesMVP summary tibbles (main / transformed / generated
#' quantities), restricted to the columns every block carries. Column names are BayesMVP's
#' (parameter, mean, sd, 2.5%, 50%, 97.5%, n_eff, Rhat), so this is what the BayesMVP gate
#' helpers take.
#'
#' @export
fn_bind_BayesMVP_summaries <-  function( outs_fit) {

      required_columns <-  c( "parameter", "mean", "sd", "2.5%", "50%", "97.5%", "n_eff", "Rhat")
      ##
      pieces <-  Filter( f = Negate(f = is.null),
                         x = list( outs_fit$summary_main,
                                   outs_fit$summary_transformed,
                                   outs_fit$summary_generated_quantities))
      ##
      if (!length(x = pieces)) stop("fn_bind_BayesMVP_summaries: the fit carries no summary tibbles.")
      ##
      pieces <-  lapply( X = pieces,
                         FUN = function(piece) {

          piece <-  as.data.frame( x = piece,
                                   check.names = FALSE,
                                   stringsAsFactors = FALSE)
          ##
          missing_columns <-  setdiff( x = required_columns,
                                       y = names(x = piece))
          ##
          if (length(x = missing_columns))
              stop( "fn_bind_BayesMVP_summaries: a summary tibble lacks column(s): ",
                    paste( missing_columns,
                           collapse = ", "))
          ##
          piece[, required_columns, drop = FALSE]

      })
      ##
      all_summary <-  do.call( what = rbind,
                               args = pieces)
      ##
      ## BridgeStan names carry no spaces, but normalise anyway: the printers match the exact string.
      all_summary$parameter <-  gsub( pattern = " ",
                                      replacement = "",
                                      x = all_summary$parameter)
      ##
      all_summary <-  all_summary[!duplicated(x = all_summary$parameter), , drop = FALSE]
      ##
      rownames(x = all_summary) <-  NULL
      ##
      all_summary

}
##
#' fn_summary_table_from_BayesMVP
#'
#' The SAME table shape cmdstanr's fit$summary() gives the project printers (variable, median,
#' mean, sd, 2.5%, 97.5%, rhat, ess_bulk), built from the BayesMVP summaries. This is what lets
#' a BayesMVP fit be handed to fn_print_all_posteriors() unchanged.
#'
#' @export
fn_summary_table_from_BayesMVP <-  function( outs_fit) {

      all_summary <-  fn_bind_BayesMVP_summaries( outs_fit = outs_fit)
      ##
      summary_table <-  data.frame( variable = all_summary$parameter,
                                    median = all_summary$`50%`,
                                    mean = all_summary$mean,
                                    sd = all_summary$sd,
                                    check.names = FALSE,
                                    stringsAsFactors = FALSE)
      ##
      summary_table[["2.5%"]] <-  all_summary$`2.5%`
      summary_table[["97.5%"]] <-  all_summary$`97.5%`
      summary_table$rhat <-  all_summary$Rhat
      summary_table$ess_bulk <-  all_summary$n_eff
      ##
      summary_table

}
##
#' fn_prior_accuracy_comparison_table_BayesMVP
#'
#' The one table the posterior report does not have - nominal Beta / probit factor vs the
#' realised prior at the anchor vs the population-standardised prior - mirroring the block in
#' fn_check_prior_accuracy() (the cmdstanr checker). Ordinal anchors are read from the
#' Se_anchor_ord / Sp_anchor_ord summary rows. Binary anchors are Phi(+/- beta_own_out): with no
#' per-draw access here they are taken from the summary's median and 2.5% / 97.5% quantiles,
#' which the monotone transform maps exactly (the 'mean' column is therefore the MEDIAN for
#' binary tests).
#'
#' @export
fn_prior_accuracy_comparison_table_BayesMVP <-  function( summary_table,
                                                          stan_data_local,
                                                          test_display_names,
                                                          cutoff_threshold = c(10L, 5L, 10L, 5L)) {

      fn_row <-  function(variable_name) summary_table[summary_table$variable == variable_name, , drop = FALSE]
      ##
      comparison_rows <-  list()
      ##
      for (test_index_global in seq_len(length.out = stan_data_local$n_tests)) for (quantity_label in c("Se", "Sp")) {

          own_status_index <-  if (quantity_label == "Se") 2L else 1L
          ##
          if (test_index_global <= stan_data_local$n_binary_tests) {

              population_name <-  sprintf( fmt = "%s_bin[%d]",
                                           quantity_label,
                                           test_index_global)
              ##
              probit_location <-  stan_data_local$prior_beta_mean[own_status_index, test_index_global] *
                  if (own_status_index == 1L) -1 else 1
              probit_scale <-  stan_data_local$prior_beta_sd[own_status_index, test_index_global]
              ##
              is_pinned <-  stan_data_local$is_perfect_ref[test_index_global] == 1L
              ##
              nominal_mean <-  if (is_pinned) pnorm(q = stan_data_local$perfect_ref_M) else
                  pnorm(q = probit_location / sqrt(x = 1 + probit_scale^2))
              ##
              nominal_interval <-  if (is_pinned) rep( x = nominal_mean,
                                                       times = 2) else
                  pnorm(q = probit_location + qnorm(p = c(.025, .975)) * probit_scale)
              ##
              beta_row <-  fn_row(variable_name = sprintf( fmt = "beta_own_out[%d,%d]",
                                                           own_status_index,
                                                           test_index_global))
              ##
              if (nrow(x = beta_row) != 1L) next
              ##
              sign <-  if (own_status_index == 2L) 1 else -1
              ##
              anchor_values <-  sort(x = pnorm(q = sign * c( beta_row$`2.5%`,
                                                             beta_row$`97.5%`)))
              ##
              anchor_row <-  data.frame( mean = pnorm(q = sign * beta_row$median),
                                         check.names = FALSE)
              anchor_row[["2.5%"]] <-  anchor_values[1]
              anchor_row[["97.5%"]] <-  anchor_values[2]

          } else {

              ordinal_index <-  test_index_global - stan_data_local$n_binary_tests
              ##
              n_categories <-  stan_data_local$n_cat_per_ord_test[ordinal_index]
              ##
              cutoff <-  cutoff_threshold[test_index_global]
              ##
              alpha_vector <-  stan_data_local$prior_dirichlet_alpha[own_status_index, seq_len(length.out = n_categories), ordinal_index]
              alpha_mass_above_cutoff <-  sum(alpha_vector[(cutoff + 1L):n_categories])
              alpha_mass_below_cutoff <-  sum(alpha_vector[seq_len(length.out = cutoff)])
              ##
              reported_alpha_mass <-  if (own_status_index == 2L) alpha_mass_above_cutoff else alpha_mass_below_cutoff
              complement_alpha_mass <-  if (own_status_index == 2L) alpha_mass_below_cutoff else alpha_mass_above_cutoff
              ##
              nominal_mean <-  reported_alpha_mass / (reported_alpha_mass + complement_alpha_mass)
              nominal_interval <-  qbeta( p = c(.025, .975),
                                          shape1 = reported_alpha_mass,
                                          shape2 = complement_alpha_mass)
              ##
              population_name <-  sprintf( fmt = "%s_ord[%d,%d]",
                                           quantity_label,
                                           ordinal_index,
                                           cutoff)
              ##
              anchor_row <-  fn_row(variable_name = sprintf( fmt = "%s_anchor_ord[%d,%d]",
                                                             quantity_label,
                                                             ordinal_index,
                                                             cutoff))

          }
          ##
          population_row <-  fn_row(variable_name = population_name)
          ##
          if (!nrow(x = anchor_row) || !nrow(x = population_row)) next
          ##
          comparison_rows[[length(x = comparison_rows) + 1L]] <-  data.frame(
              test = test_display_names[test_index_global],
              quantity = quantity_label,
              nominal_mean = nominal_mean,
              nominal_lo = nominal_interval[1],
              nominal_hi = nominal_interval[2],
              anchor_mean = anchor_row$mean,
              anchor_lo = anchor_row$`2.5%`,
              anchor_hi = anchor_row$`97.5%`,
              population_mean = population_row$mean,
              population_lo = population_row$`2.5%`,
              population_hi = population_row$`97.5%`,
              stringsAsFactors = FALSE)

      }
      ##
      if (!length(x = comparison_rows)) return(NULL)
      ##
      do.call( what = rbind,
               args = comparison_rows)

}
##
#' fn_reprint_prior_check_BayesMVP
#'
#' Reprint the prior-only report from an EXISTING prior-check object (in session or loaded
#' from prior_checks.RDS) - no re-run. Works on objects saved before the report existed: the
#' table is rebuilt from the summary tibbles and the subset data the object already carries.
#'
#' @export
fn_reprint_prior_check_BayesMVP <-  function( prior_check,
                                              test_labels = NULL,
                                              condition_labels = NULL,
                                              accuracy_window = NULL,
                                              show_cutpoints = FALSE,
                                              show_accuracy_table = TRUE,
                                              show_per_group = TRUE,
                                              show_grid = TRUE,
                                              comparison_cutoff_threshold = c(10L, 5L, 10L, 5L)) {

      if (is.null(x = prior_check$stan_data))
          stop("fn_reprint_prior_check_BayesMVP: the object carries no 'stan_data' (the subset the run used).")
      ##
      model_i <-  prior_check$model_i
      stan_data_local <-  prior_check$stan_data
      ##
      summary_table <-  if (!is.null(x = prior_check$summary)) prior_check$summary else
                            fn_summary_table_from_BayesMVP( outs_fit = prior_check)
      ##
      test_display_names <-  fn_resolve_test_labels( stan_data_local = stan_data_local,
                                                     test_labels = test_labels,
                                                     condition_labels = condition_labels)
      ##
      cat( sprintf( fmt = "\n################ PRIOR-ONLY RUN (reprint): %s ################\n",
                    model_i))
      ##
      fn_print_all_posteriors( summary_table = summary_table,
                               stan_data_local = stan_data_local,
                               label = sprintf( fmt = "%s [PRIOR ONLY]",
                                                model_i),
                               test_labels = test_labels,
                               condition_labels = condition_labels,
                               show_accuracy = TRUE,
                               accuracy_window = accuracy_window,
                               show_cutpoints = show_cutpoints,
                               show_per_group = show_per_group,
                               show_grid = show_grid)
      ##
      comparison_table <-  if (show_accuracy_table) {
          fn_prior_accuracy_comparison_table_BayesMVP( summary_table = summary_table,
                                                       stan_data_local = stan_data_local,
                                                       test_display_names = test_display_names,
                                                       cutoff_threshold = comparison_cutoff_threshold)
      } else NULL
      ##
      if (!is.null(x = comparison_table)) {
          cat("\n--- ACCURACY: nominal factor vs realised prior vs population-standardised prior\n")
          print( x = comparison_table,
                 row.names = FALSE,
                 digits = 3)
      }
      ##
      invisible( x = list( summary = summary_table,
                           comparison_table = comparison_table))

}
##
#' fn_run_prior_check_BayesMVP
#'
#' Prior-only run through BayesMVP, reported with the SAME printer the posterior and the
#' cmdstanr prior check use (fn_print_all_posteriors), so the output is Se / Sp / prevalence /
#' correlations on the constrained scale with the test labels - not the unconstrained block.
#'
#' @export
fn_run_prior_check_BayesMVP <-  function( model_i,
                                          stan_data,
                                          settings,
                                          n_prior_rows = 1000,
                                          n_chains = NULL,      ## NULL = keep settings$n_chains_burnin / n_chains_sampling; a number sets both
                                          n_warmup = 200,
                                          n_sampling = 200,
                                          ##
                                          ## size of the standardisation / PPV-grid set for the PRIOR run
                                          ## (NULL = every kept row, which is what the posterior fits use):
                                          n_standardisation_prior = 50,
                                          ##
                                          seed = 123,
                                          verbose = TRUE,
                                          ##
                                          ## ---- report options (same meaning as in fn_check_prior_accuracy):
                                          test_labels = NULL,
                                          condition_labels = NULL,
                                          accuracy_window = NULL,      ## NULL = every threshold (the whole prior ROC curve)
                                          show_cutpoints = FALSE,
                                          show_accuracy_table = TRUE,
                                          show_per_group = TRUE,
                                          show_grid = TRUE,
                                          comparison_cutoff_threshold = c(10L, 5L, 10L, 5L),
                                          ##
                                          ## ---- convergence gate for a PRIOR run (looser than the posterior's):
                                          max_rhat_thr = 1.25,
                                          min_ESS_thr = 50) {
  
      stan_data_local <-  stan_data
      ##
      ## ---- PPS subset FIRST (the same Hansen-Hurwitz scheme as the original bridge):
      set.seed(seed = seed)
      ##
      total_weight <-  sum(stan_data_local$target_row_weight)
      ##
      sampled_rows <-  sort( x = sample.int( n = stan_data_local$N,
                                             size = n_prior_rows,
                                             replace = TRUE,
                                             prob = stan_data_local$target_row_weight))
      ##
      kept_row_indices <-  unique(x = sampled_rows)
      ##
      multiplicity <-  as.numeric(x = table(factor( x = sampled_rows,
                                                    levels = kept_row_indices)))
      ##
      stan_data_local$N <-  length(x = kept_row_indices)
      stan_data_local$y <-  stan_data_local$y[kept_row_indices, , drop = FALSE]
      stan_data_local$pop <-  stan_data_local$pop[kept_row_indices]
      stan_data_local$X <-  stan_data_local$X[, kept_row_indices, , drop = FALSE]
      ##
      if (!is.null(x = stan_data_local$X_prevalence) && ncol(x = stan_data_local$X_prevalence) > 0L)
          stan_data_local$X_prevalence <-  stan_data_local$X_prevalence[kept_row_indices, , drop = FALSE]
      ##
      stan_data_local$target_row_weight <-  multiplicity * total_weight / n_prior_rows
      ##
      ## ---- Standardisation set for the prior run. A prior-only draw's generated-quantities
      ## cost is dominated by the PPV standardisation grid (once per profile per class per
      ## quadrature node): on all ~900 kept rows one draw took ~0.33 s, so 4 x 200 draws was
      ## ~4 CPU-minutes. The Se / Sp priors do not depend on this grid at all, so a PRIOR
      ## check only needs a small PPS subsample of the kept rows.
      ##
      ## All SIX fields are set together. n_standardisation sizes standardisation_index and
      ## standardisation_weight, and grid_standardisation_position is bounded above by
      ## n_standardisation; changing the count on its own (as a hand edit once did) leaves
      ## 906-entry arrays declared as array[50] and the model refuses to construct.
      ##
      if (!is.null(x = n_standardisation_prior) &&
          (!is.numeric(x = n_standardisation_prior) || length(x = n_standardisation_prior) != 1L ||
           !is.finite(x = n_standardisation_prior) || n_standardisation_prior < 1)) {
          stop("fn_run_prior_check_BayesMVP: 'n_standardisation_prior' must be NULL (all kept rows) or a single number >= 1.")
      }
      ##
      n_standardisation_used <-  if (is.null(x = n_standardisation_prior)) length(x = kept_row_indices) else
                                     min( as.integer(x = n_standardisation_prior),
                                          length(x = kept_row_indices))
      ##
      standardisation_rows <-  if (n_standardisation_used < length(x = kept_row_indices)) {
          sort(x = sample.int( n = length(x = kept_row_indices),
                               size = n_standardisation_used,
                               replace = FALSE,
                               prob = stan_data_local$target_row_weight))
      } else seq_along(along.with = kept_row_indices)
      ##
      stan_data_local$n_standardisation <-  n_standardisation_used
      stan_data_local$standardisation_index <-  standardisation_rows   ## rows of the LOCAL (subset) data, in 1..N
      stan_data_local$standardisation_weight <-  stan_data_local$target_row_weight[standardisation_rows]
      ##
      stan_data_local$n_grid_standardisation <-  n_standardisation_used
      stan_data_local$grid_standardisation_position <-  seq_len(length.out = n_standardisation_used)   ## positions WITHIN the set above
      stan_data_local$grid_standardisation_weight <-  stan_data_local$target_row_weight[standardisation_rows]
      stan_data_local$pop_weight <-  vapply( X = seq_len(length.out = stan_data_local$n_pops),
                                             FUN = function(group_index)
          max( sum(stan_data_local$target_row_weight[stan_data_local$pop == group_index]),
               1e-8),
                                             FUN.VALUE = numeric(length = 1))
      ##
      ## ---- blank the OUTCOMES (the likelihood is bypassed) and switch prior_only on:
      stan_data_local$y <-  matrix( data = -1,
                                    nrow = stan_data_local$N,
                                    ncol = stan_data_local$n_tests)
      stan_data_local$prior_only <-  1L
      stan_data_local$n_loo_draws <-  0L
      stan_data_local$save_subject_probs <-  0L
      stan_data_local$grainsize <-  max( 1L,
                                         as.integer(x = ceiling(x = stan_data_local$N / 64)))
      ##
      ## ---- the prior check's OWN subset configuration (chains / warm-up / sampling), with the
      ## sampler settings otherwise unchanged:
      settings_prior <-  settings
      ## n_chains = NULL keeps the settings' own burn-in / sampling split (e.g. 4 / 32);
      ## a number sets both to that value (with one superchain).
      if (!is.null(x = n_chains)) {
          settings_prior$n_chains_burnin <-  n_chains
          settings_prior$n_chains_sampling <-  n_chains
          settings_prior$n_superchains <-  1
      }
      settings_prior$n_burnin <-  n_warmup
      settings_prior$n_iter <-  n_sampling
      settings_prior$seed <-  seed
      ##
      outs <-  fn_fit_APMS_model_BayesMVP( model_i = model_i,
                                           stan_data = stan_data_local,
                                           settings = settings_prior,
                                           verbose = verbose)
      ##
      ## ---- the SAME report the posterior gets. BayesMVP's three summary tibbles are bound into
      ## the cmdstanr-shaped table (variable / median / mean / sd / 2.5% / 97.5% / rhat / ess_bulk)
      ## that fn_print_all_posteriors() reads, so this prints Se / Sp / prevalence / correlations
      ## on the constrained scale with the test labels - not the unconstrained parameters block.
      summary_table <-  fn_summary_table_from_BayesMVP( outs_fit = outs)
      ##
      gate_summary <-  fn_filter_gate_summary_BayesMVP( summary_tibble = fn_bind_BayesMVP_summaries( outs_fit = outs))
      ##
      convergence_gate <-  fn_fit_gate_BayesMVP( summary_tibble = gate_summary,
                                                 max_r_hat_threshold = max_rhat_thr,
                                                 min_effective_sample_size_threshold = min_ESS_thr)
      ##
      test_display_names <-  fn_resolve_test_labels( stan_data_local = stan_data_local,
                                                     test_labels = test_labels,
                                                     condition_labels = condition_labels)
      ##
      comparison_table <-  if (show_accuracy_table) {
          fn_prior_accuracy_comparison_table_BayesMVP( summary_table = summary_table,
                                                       stan_data_local = stan_data_local,
                                                       test_display_names = test_display_names,
                                                       cutoff_threshold = comparison_cutoff_threshold)
      } else NULL
      ##
      if (verbose) {

          cat( sprintf( fmt = "\n################ PRIOR-ONLY RUN: %s  (gate %s) ################\n",
                        model_i,
                        if (convergence_gate[["gate_passed"]] == 1) "passed" else "FAILED"))
          cat( sprintf( fmt = "  subset: %d distinct rows from %d PPS draws; standardisation set %d rows (likelihood bypassed)\n",
                        length(x = kept_row_indices),
                        n_prior_rows,
                        n_standardisation_used))
          cat( sprintf( fmt = "  %.1f min   max Rhat %.3f   min n_eff %.0f   divergences %.0f (%.2f%%)   [gate: Se/Sp + prevalence rows only]\n",
                        outs$wall_time_min,
                        convergence_gate[["max_r_hat"]],
                        convergence_gate[["min_effective_sample_size"]],
                        outs$divergences$n_divs,
                        outs$divergences$pct_divs))
          cat("  Outcomes blanked; rows, strata, covariates and the standardisation set are the real ones.\n")
          cat("  Everything below is what the PRIORS imply, on the same scale the posterior reports.\n")
          ##
          fn_print_all_posteriors( summary_table = summary_table,
                                   stan_data_local = stan_data_local,
                                   label = sprintf( fmt = "%s [PRIOR ONLY]",
                                                    model_i),
                                   ##
                                   test_labels = test_labels,
                                   condition_labels = condition_labels,
                                   ##
                                   show_accuracy = TRUE,
                                   accuracy_window = accuracy_window,
                                   show_cutpoints = show_cutpoints,
                                   ##
                                   show_per_group = show_per_group,
                                   show_grid = show_grid)
          ##
          if (!is.null(x = comparison_table)) {

              cat("\n--- ACCURACY: nominal factor vs realised prior vs population-standardised prior\n")
              print( x = comparison_table,
                     row.names = FALSE,
                     digits = 3)
              cat("  nominal  = the isolated Beta / probit factor, before any model constraint\n")
              cat("  anchor   = the realised prior at latent mean 0, other condition absent, X = 0\n")
              cat("             (binary tests: median-based, from the beta_own_out summary quantiles)\n")
              cat("  population = after stratum, covariate and class standardisation - directly\n")
              cat("               comparable with the posterior's Se_ord / Sp_ord\n")
              cat("  A gap between nominal and anchor is the stochastic-ordering constraint and the\n")
              cat("  cutpoint priors acting on the stated belief. A gap between anchor and population\n")
              cat("  is the class mix and the covariate distribution.\n")

          }
          ##
          cat( sprintf( fmt = "\n################ end prior-only: %s ################\n",
                        model_i))

      }
      ##
      if (convergence_gate[["gate_passed"]] != 1)
          warning( "prior-only diagnostics failed for ",
                   model_i,
                   "; these prior summaries need a longer or better-conditioned run.")
      ##
      outs$summary <-  summary_table
      outs$comparison_table <-  comparison_table
      outs$convergence_gate <-  convergence_gate
      outs$stan_data <-  stan_data_local
      ##
      invisible(x = outs)

}
##
## -| --------- 7. Headline report for one fitted model ----------------------------------------------------------------------------------------
##
#' fn_filter_gate_summary_BayesMVP
#'
#' Restrict the convergence gate to the quantities this bridge cares about: the Se/Sp parameters
#' of each test and the prevalence-related parameters. GQ rows such as subject-level or
#' standardisation-block quantities can carry sd > 0 without Rhat / n_eff diagnostics, which
#' would otherwise trip the missing-diagnostic check in fn_fit_gate_BayesMVP().
#'
#' @export
fn_filter_gate_summary_BayesMVP <-  function( summary_tibble) {

        se_sp_pattern <-  "^(Se_bin|Sp_bin|Se_ord|Sp_ord|Se_anchor_ord|Sp_anchor_ord)\\["
        ##
        prevalence_pattern <-  "^(prev_A_overall|prev_B_overall|pi_00_overall|pi_10_overall|pi_01_overall|pi_11_overall|prev_A_par|prev_B_par)(\\[[0-9]+\\])?$"
        ##
        is_monitored_parameter <-  grepl( pattern = se_sp_pattern,
                                          x = summary_tibble$parameter) |
                                   grepl( pattern = prevalence_pattern,
                                          x = summary_tibble$parameter)
        ##
        summary_tibble[is_monitored_parameter, , drop = FALSE]

}
##
#' fn_report_fit_BayesMVP
#' @export
fn_report_fit_BayesMVP <-  function( outs_fit,
                                     stan_data_local,
                                     max_rhat_thr = 1.25,
                                     min_ESS_thr = 50
) {

        model_i <-  outs_fit$model_i
        ##
        summary_gq <-  outs_fit$summary_generated_quantities
        summary_main <-  outs_fit$summary_main
        ##
        all_summary <-  if (is.null(summary_gq)) summary_main else rbind(summary_main, summary_gq)
        ##
        gate_summary <-  fn_filter_gate_summary_BayesMVP( summary_tibble = all_summary)
        ##
        gate <-  fn_fit_gate_BayesMVP( summary_tibble = gate_summary,
                                       max_r_hat_threshold = max_rhat_thr,
                                       min_effective_sample_size_threshold = min_ESS_thr)
        ##
        n_divs <-  outs_fit$divergences$n_divs
        pct_divs <-  outs_fit$divergences$pct_divs
        ##
        fn_get_variable_row <-  function(variable_name)
            all_summary[all_summary$parameter == variable_name,]
        ##
        cat( sprintf( fmt = "\n================================  %s  ================================\n",
                      model_i))
        cat( sprintf( fmt = "  %.1f min   max Rhat %.3f   min n_eff %.0f   divergences %.0f (%.2f%%)\n",
                      outs_fit$wall_time_min,
                      gate[["max_r_hat"]],
                      gate[["min_effective_sample_size"]],
                      n_divs,
                      pct_divs))
        ##
        if (gate[["gate_passed"]] != 1) {
            cat("  *** CONVERGENCE GATE FAILED - retained for diagnosis.\n")
        }
        ##
        cat( sprintf( fmt = "  prev_A %.4f [%.4f, %.4f]   prev_B %.4f [%.4f, %.4f]   pi_11 %.5f [%.5f, %.5f]   Pr(B|A) %.3f   Pr(A|B) %.3f\n",
                      fn_get_variable_row(variable_name = "prev_A_overall")$mean,
                      fn_get_variable_row(variable_name = "prev_A_overall")$`2.5%`,
                      fn_get_variable_row(variable_name = "prev_A_overall")$`97.5%`,
                      fn_get_variable_row(variable_name = "prev_B_overall")$mean,
                      fn_get_variable_row(variable_name = "prev_B_overall")$`2.5%`,
                      fn_get_variable_row(variable_name = "prev_B_overall")$`97.5%`,
                      fn_get_variable_row(variable_name = "pi_11_overall")$mean,
                      fn_get_variable_row(variable_name = "pi_11_overall")$`2.5%`,
                      fn_get_variable_row(variable_name = "pi_11_overall")$`97.5%`,
                      fn_get_variable_row(variable_name = "Pr_B_given_A_overall")$mean,
                      fn_get_variable_row(variable_name = "Pr_A_given_B_overall")$mean))
        ##
        invisible(gate)

}


## -| --------- 8. Application table: every fitted model side by side -----------------------------
##
#' fn_application_table_BayesMVP
#' @export
fn_application_table_BayesMVP <-  function( fits_by_model,
                                            target = "phase_one_weighted") {

    fmt3 <-  function( g,
                       v,
                       d = 4,
                       scale = 100)
        sprintf( fmt = paste0( "%.",
                               d,
                               "f (%.",
                               d,
                               "f, %.",
                               d,
                               "f)"),
                 scale * g(v)$mean,
                 scale * g(v)$`2.5%`,
                 scale * g(v)$`97.5%`)
    ##
    do.call( what = rbind,
             args = lapply( X = names(x = fits_by_model),
                            FUN = function(model_i) {

            outs_fit <-  fits_by_model[[model_i]]
            ##
            summary_gq <-  outs_fit$summary_generated_quantities
            summary_main <-  outs_fit$summary_main
            ##
            all_summary <-  if (is.null(summary_gq)) summary_main else rbind(summary_main, summary_gq)
            ##
            fn_get_summary_row <-  function(variable_name)
                all_summary[all_summary$parameter == variable_name,]
            ##
            gate <-  fn_fit_gate_BayesMVP( summary_tibble = fn_filter_gate_summary_BayesMVP( summary_tibble = all_summary))
            ##
            data.frame( model = model_i,
                        target = target,
                        prev_A = fmt3( g = fn_get_summary_row,
                                       v = "prev_A_overall"),
                        prev_B = fmt3( g = fn_get_summary_row,
                                       v = "prev_B_overall"),
                        pi_11 = fmt3( g = fn_get_summary_row,
                                      v = "pi_11_overall",
                                      d = 5),
                        Pr_B_g_A = fmt3( g = fn_get_summary_row,
                                         v = "Pr_B_given_A_overall",
                                         d = 3),
                        Pr_A_g_B = fmt3( g = fn_get_summary_row,
                                         v = "Pr_A_given_B_overall",
                                         d = 3),
                        log_OR = fmt3( g = fn_get_summary_row,
                                       v = "log_OR_comorbid_overall",
                                       d = 2,
                                       scale = 1),
                        max_Rhat = sprintf( fmt = "%.3f",
                                            gate[["max_r_hat"]]),
                        min_ESS = sprintf( fmt = "%.0f",
                                           gate[["min_effective_sample_size"]]),
                        mins = sprintf( fmt = "%.1f",
                                        outs_fit$wall_time_min),
                        stringsAsFactors = FALSE)

        }))

}


## -| --------- 9. Posterior summaries -> DGM arguments (one set per model) ------------------------
##
## Mirrors fn_extract_DGM_params() in the original helper, reading from the BayesMVP summary
## tibbles instead of the cmdstanr summary. The componentwise-median caveat applies unchanged:
## these are NOT one joint posterior draw.
##
#' fn_extract_DGM_params_BayesMVP
#' @export
fn_extract_DGM_params_BayesMVP <-  function( outs_fit,
                                             seed = 123) {

    model_i <-  outs_fit$model_i
    ##
    summary_gq <-  outs_fit$summary_generated_quantities
    summary_main <-  outs_fit$summary_main
    summary_transformed <-  outs_fit$summary_transformed
    ##
    all_summary <-  do.call( what = rbind,
                             args = Filter( f = Negate(f = is.null),
                                            x = list( summary_main,
                                                      summary_transformed,
                                                      summary_gq)))
    ##
    stan_data_model_i <-  outs_fit$stan_data
    ##
    n_tests <-  stan_data_model_i$n_tests
    ##
    fn_get_posterior_means <-  function(variable_prefix) {

        matching_summary_rows <-  all_summary[grepl( pattern = paste0( "^",
                                                                       variable_prefix),
                                                     x = all_summary$parameter),]
        ##
        setNames( object = matching_summary_rows$mean,
                  nm = matching_summary_rows$parameter)

    }
    ##
    class_probability_means <-  c( pi_00 = all_summary$mean[all_summary$parameter == "pi_00_overall"],
                                   pi_10 = all_summary$mean[all_summary$parameter == "pi_10_overall"],
                                   pi_01 = all_summary$mean[all_summary$parameter == "pi_01_overall"],
                                   pi_11 = all_summary$mean[all_summary$parameter == "pi_11_overall"])
    ##
    class_probability_means <-  class_probability_means / sum(class_probability_means)
    ##
    prev_A <-  class_probability_means[["pi_11"]] + class_probability_means[["pi_10"]]
    prev_B <-  class_probability_means[["pi_11"]] + class_probability_means[["pi_01"]]
    ##
    ## beta_own_out[s, t] and delta_cross_out[s, t]: 2 x 4 each:
    beta_own_means <-  matrix( data = NA_real_,
                               nrow = 2,
                               ncol = n_tests)
    ##
    delta_cross_means <-  matrix( data = NA_real_,
                                  nrow = 2,
                                  ncol = n_tests)
    ##
    for (own_status_index in 1:2) for (test_index in 1:n_tests) {
        beta_own_means[own_status_index, test_index] <-  all_summary$mean[
            all_summary$parameter == sprintf( fmt = "beta_own_out[%d,%d]",
                                              own_status_index,
                                              test_index)
        ]
        ##
        delta_cross_means[own_status_index, test_index] <-  all_summary$mean[
            all_summary$parameter == sprintf( fmt = "delta_cross_out[%d,%d]",
                                              own_status_index,
                                              test_index)
        ]
    }
    ##
    simulable <-  stan_data_model_i$n_ordinal_tests == 4L
    ##
    ## cutpoints C_vec_out[s, k], split by test (ordinal tests only):
    C_true <-  vector( mode = "list",
                       length = 2L)
    ##
    for (own_status_index in 1:2) {
        C_flat <-  sapply( X = seq_len(length.out = sum(stan_data_model_i$n_thr_per_ord_test)),
                           FUN = function(threshold_index) {

            matching_indices <-  which( x = all_summary$parameter == sprintf( fmt = "C_vec_out[%d,%d]",
                                                                              own_status_index,
                                                                              threshold_index))
            ##
            stopifnot(length(x = matching_indices) == 1L)
            ##
            all_summary$mean[matching_indices]

        })
        ##
        C_true[[own_status_index]] <-  split( x = C_flat,
                                              f = rep( x = seq_along(along.with = stan_data_model_i$n_thr_per_ord_test),
                                                       times = stan_data_model_i$n_thr_per_ord_test))
    }
    ##
    names(x = C_true) <-  c("absent", "present")
    ##
    ## class-specific correlation matrices:
    Omega_list_means <-  lapply( X = 1:stan_data_model_i$n_class,
                                 FUN = function(class_index) {

        correlation_matrix <-  matrix( data = NA_real_,
                                       nrow = n_tests,
                                       ncol = n_tests)
        ##
        for (row_index in 1:n_tests) for (column_index in 1:n_tests) {
            matching_indices <-  which( x = all_summary$parameter == sprintf( fmt = "Omega[%d,%d,%d]",
                                                                              class_index,
                                                                              row_index,
                                                                              column_index))
            ##
            stopifnot(length(x = matching_indices) == 1L)
            ##
            correlation_matrix[row_index, column_index] <-  all_summary$mean[matching_indices]
        }
        ##
        correlation_matrix <-  (correlation_matrix + t(x = correlation_matrix)) / 2
        ##
        as.matrix(x = Matrix::nearPD( x = correlation_matrix,
                                      corr = TRUE)$mat)

    })
    ##
    names(x = Omega_list_means) <-  c("00", "10", "01", "11")
    ##
    list( model = model_i,
          simulable = simulable,
          single_population_approximation =
              stan_data_model_i$n_pops > 1L || any(stan_data_model_i$n_covs_per_outcome > 0L) ||
                  stan_data_model_i$n_prevalence_covariates > 0L,
          parameter_summary_approximation = TRUE,  ## componentwise posterior means are not one joint draw
          fitted_stratum_design = list( pop_weight = stan_data_model_i$pop_weight,
                                        standardisation_index = stan_data_model_i$standardisation_index,
                                        standardisation_weight = stan_data_model_i$standardisation_weight,
                                        covariate_posterior_means = fn_get_posterior_means(variable_prefix = "beta_cov_by_own_status"),
                                        prevalence_posterior_means = fn_get_posterior_means(variable_prefix = "beta_prevalence")),
          true_prev_A = prev_A,
          true_prev_B = prev_B,
          true_Pr_B_given_A = class_probability_means[["pi_11"]] / prev_A,
          n_cat_per_ord_test = stan_data_model_i$n_cat_per_ord_test,
          ref_thr_cat = stan_data_model_i$ref_thr_cat,
          scr_thr_cat = stan_data_model_i$scr_thr_cat,
          beta_own_true = beta_own_means,
          delta_cross_true = delta_cross_means,  ## FITTED - this is what makes each arm's truth its model's estimate
          C_true = C_true,
          Omega_list = Omega_list_means,
          verification = "two_phase",
          ver_target_n = sum(stan_data_model_i$y[, 1] >= 0 & stan_data_model_i$y[, 2] >= 0),
          source = sprintf( fmt = "APMS 2007, model %s, posterior means (BayesMVP/BridgeStan)",
                            model_i),
          collapsing = "none - category = raw score + 1",
          seed = seed)

}
