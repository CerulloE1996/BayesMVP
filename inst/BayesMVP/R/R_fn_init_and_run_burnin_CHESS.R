

## R_fn_init_and_run_burnin_CHESS.R

# Burnin - using R fn (which calls Rcpp fn @ each iter)  -------------------------------------------------------------------------------------


#' init_and_run_burnin_ChESSR
#' @param debug_burnin_timing Collect native component timings and per-chain integrator step counts; default FALSE.
#' @param diffusion_HMC_integrator Joint diffusion integrator: "kick_flow_kick" (default) or "flow_kick_flow".
#' @export
init_and_run_burnin_ChESSR   <- function(  debug,
                                           ##
                                           init_object,
                                           ##
                                           Model_args_as_Rcpp_List,
                                           ##
                                           Model_type,
                                           ##
                                           n_chains_burnin,
                                           ##
                                           theta_main_vectors_all_chains_input_from_R,
                                           theta_nuisance_vectors_all_chains_input_from_R,
                                           ##
                                           parallel_method,
                                           ##
                                           Stan_data_list,
                                           model_args_list,
                                           ##
                                           sample_nuisance,
                                           n_nuisance_override,
                                           ##
                                           seed,
                                           n_burnin,
                                           n_adapt,
                                           gap,
                                           ##
                                           adapt_delta,
                                           LR_main,
                                           LR_us,
                                           ##
                                           learning_rate_initial = NULL,
                                           learning_rate_initial_iter = NULL,
                                           tau_mult,
                                           tau_initial,
                                           ##
                                           manual_tau,
                                           tau_if_manual,
                                           ##
                                           ## When TRUE, tau_if_manual is read as a number of LEAPFROG STEPS rather than as an
                                           ## integration time, and tau is re-derived as L * eps at every iteration from the
                                           ## CURRENTLY adapted step size. That pins the trajectory length while leaving the
                                           ## step size free to keep dual-averaging to adapt_delta, which is what a
                                           ## "sweep L, measure ESS per gradient" experiment needs; passing a raw tau instead
                                           ## lets L drift as eps adapts, so the swept quantity is not the one being held fixed.
                                           ## Note tau_ii ~ Uniform(0, 2*tau), so this fixes the MEAN number of leapfrog steps.
                                           tau_if_manual_in_L_units = FALSE,
                                           ##
                                           ## ---- Trajectory-length objective. "KE" is squared kinetic-energy change (v2),
                                           ##      "ChEES" is the position-based criterion of Hoffman et al. (2021). See the
                                           ##      header of R_fn_compute_gradients_for_tau_using_ChEES_position for why the
                                           ##      OLD signed kinetic-energy statistic had zero expectation at stationarity.
                                           tau_objective = c("KE", "ChEES", "ChEES_per_tau"),
                                           ##
                                           ## Weight each chain's gradient by its Metropolis acceptance probability rather than
                                           ## by the realised accept indicator. Both options now aggregate by a mean.
                                           ## Proposal weighting never also applies the accept indicator.
                                           tau_weight_by_p_jump = FALSE,
                                           ##
                                           ## Burn-in tau ramp over [clip_iter, gap), before the tau adaptation takes over:
                                           ##   "original" - 5*eps, then tau_initial/4, tau_initial/2, tau_initial.
                                           ##   "staged"   - 5, 10, 20 steps, then tau_initial times (1/8, 1/4, 1/2, 1).
                                           ## Both use equal slices of the actual window, cap their step-count floor at
                                           ## tau_initial, and hand over once. Neither is a bit-identical historical replay.
                                           tau_ramp = c("original", "staged"),
                                           ##
                                           ## eps at iteration gap (the handover to the tau adaptation):
                                           ##   TRUE  - re-initialise eps with find_initial_eps (halves from 0.5 until ONE leapfrog step
                                           ##           from the chain mean accepts at >= 0.8, so it returns 0.0625, 0.125, ...) and reset
                                           ##           the eps ADAM moments. The behaviour since at least 2026-09-06.
                                           ##   FALSE - keep the eps (and its ADAM moments) adapted up to gap. tau is reset either way.
                                           eps_reinit_at_ChEES_handover = TRUE,
                                           ##
                                           ## Centre (theta_hat_us) of the exact Gaussian rotation for the nuisance block:
                                           ##   "running_mean"        - re-set EVERY iteration to the running mean of the chains' current
                                           ##                           nuisance states. The centre then follows the chains, which inflates
                                           ##                           the burn-in acceptance (same state/eps/metric: ~0.7 vs ~0.45 with the
                                           ##                           centre frozen, as it is in sampling), so eps is tuned too big for
                                           ##                           sampling. The behaviour up to 2026-09-18 (default here, so a call
                                           ##                           that does not pass it - e.g. the pre-burnin - is unchanged).
                                           ##   "running_mean_frozen" - the same running mean, but FROZEN from theta_hat_us_freeze_iter on,
                                           ##                           so eps finishes adapting against exactly the kernel sampling uses.
                                           ##   "zero"                - centre fixed at 0 throughout (the Feb 2026 code).
                                           theta_hat_us_rule = c("running_mean", "running_mean_frozen", "zero"),
                                           theta_hat_us_freeze_iter = NULL,   ## automatic: linked to the final metric update; legacy: round(0.6 * n_adapt)
                                           burnin_schedule = "legacy",       ## preserve ordering pre-burn-in and direct historical callers
                                           metric_adaptation_end_iter = NULL,
                                           ##
                                           ## Non-adapting iterations to run after n_adapt. NULL = all of them, up to n_burnin (the behaviour
                                           ## up to 2026-09-18). eps, the metric and tau are frozen after n_adapt, so these iterations only let
                                           ## the few burn-in chains settle - which the many sampling chains do far more cheaply. e.g. 5.
                                           ## Historical early-stop description above is retired: the loop now always runs to n_burnin.
                                           ##
                                           ## TRUE = one trajectory length tau_ii per iteration, shared by ALL burn-in chains (the C++ burn-in
                                           ## worker draws it from a separate RNG; each chain's momentum stays its own). Iterations then stop
                                           ## waiting for whichever chain drew the longest trajectory. FALSE = per-chain tau_ii (as before).
                                           share_tau_ii_across_chains_in_burnin = FALSE,
                                           ##
                                           burnin_algorithm,
                                           diffusion_HMC,
                                           partitioned_HMC,
                                           ##
                                           clip_iter,
                                           clip_iter_tau,
                                           ##
                                           n_refresh,
                                           use_proposed,
                                           ##
                                           beta1_adam,
                                           beta2_adam,
                                           eps_adam,
                                           ##
                                           force_autodiff,
                                           force_PartialLog,
                                           multi_attempts,
                                           ##
                                           force_autodiff_for_metric,
                                           force_PartialLog_for_metric,
                                           force_multi_attempts_for_metric,
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
                                           max_tau_main,
                                           max_tau_nuisance,
                                           ##
                                           max_eps_main,
                                           max_eps_nuisance,
                                           ##
                                           max_L,
                                           ##
                                           n_nuisance_to_track,
                                           ##
                                           n_threads_WCP,
                                           ##
                                           time_pre_burnin,
                                           ##
                                           metric_start_iter = NULL,  # first iteration whose draws feed the metric; NULL = max(clip_iter, 0.35*n_adapt)
                                           metric_estimator = "pooled",
                                           ##
                                           ## NOTE: 'eps_init' and 'eps_initial' are DIFFERENT things.
                                           ## eps_init  = carry the step-size over from a previous stage
                                           ##             (used by the reorder pre-burnin); it replaces the
                                           ##             heuristic starting eps for the WHOLE run.
                                           ## eps_initial = a deliberately LARGE step-size held for the
                                           ##             first eps_initial_iter iterations only, after
                                           ##             which eps is handed back to the starting value
                                           ##             (heuristic or eps_init) and adapted as normal.
                                           eps_init = NULL,
                                           ##
                                           eps_initial = NULL,
                                           eps_initial_iter = NULL,
                                           ##
                                           diffusion_HMC_integrator = "kick_flow_kick",
                                           debug_burnin_timing = FALSE
) {
  if (!is.logical(debug_burnin_timing) || length(debug_burnin_timing) != 1 || is.na(debug_burnin_timing)) {
      stop("debug_burnin_timing must be TRUE or FALSE.")
  }
  
  # Rprof("burnin_prof.out", interval = 0.005)
  
  ratio_M_us <- ratio_M_nuisance
  ##
  max_tau_us <- max_tau_nuisance
  max_eps_us <- max_eps_nuisance
  ##
  theta_us_vectors_all_chains_input_from_R <- theta_nuisance_vectors_all_chains_input_from_R 
  ##
  # Model_args_as_Rcpp_List <- init_object$Model_args_as_Rcpp_List
  n_nuisance <- init_object$n_nuisance
  n_params_main <- init_object$n_params_main
  n_params <- n_nuisance + n_params_main
  ##
  Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
  ##
  index_nuisance <- seq_len(n_nuisance)
  index_us <- index_nuisance
  index_main <- (1 + n_nuisance):n_params
  
  y <- model_args_list$y ; y
  N <- model_args_list$N ; N
  n_class <- model_args_list$n_class ; n_class
  
  if (Model_type == "Stan") {
      N <- 1000
      latent_results <- rlogis(n = N, location =  0.0)
      y <- ifelse(latent_results > 0, 1, 0)
      y <- matrix(data = c(y), ncol = 1)
  } 
  
                                        
  
  {  # ---------------------------------------------------------------- list for EHMC params / EHMC struct in C++
    
    EHMC_Metric_as_Rcpp_List <- init_EHMC_Metric_as_Rcpp_List(   n_params_main = n_params_main, 
                                                                 n_nuisance = n_nuisance, 
                                                                 metric_shape_main = metric_shape_main)  
    
  }
  
  
  
  
  
  { # ----------------------------------------------------------------- list for EHMC params / EHMC struct in C++  #
    
    EHMC_args_as_Rcpp_List <- init_EHMC_args_as_Rcpp_List( diffusion_HMC = diffusion_HMC,
                                                       diffusion_HMC_integrator = diffusion_HMC_integrator)
    ##
    if (diffusion_HMC_integrator == "flow_kick_flow" &&
        (!isTRUE(x = diffusion_HMC) || !isFALSE(x = partitioned_HMC) || n_nuisance < 1L)) {
        stop("flow_kick_flow requires diffusion_HMC = TRUE, partitioned_HMC = FALSE and a nonempty nuisance block.")
    }

  }


  { # ----------------------------------------------------------------- trajectory-length adaptation options

    tau_objective <- match.arg(arg = tau_objective, choices = c("KE", "ChEES", "ChEES_per_tau"))
    tau_ramp <- match.arg(arg = tau_ramp, choices = c("original", "staged"))
    if (!is.logical(eps_reinit_at_ChEES_handover) || length(eps_reinit_at_ChEES_handover) != 1 || is.na(eps_reinit_at_ChEES_handover)) {
      stop("eps_reinit_at_ChEES_handover must be TRUE or FALSE.")
    }
    theta_hat_us_rule <- match.arg(arg = theta_hat_us_rule, choices = c("running_mean", "running_mean_frozen", "zero"))
    resolved_burnin_schedule <- fn_burnin_adaptation_schedule(
        n_burnin = n_burnin,
        n_adapt = n_adapt,
        burnin_schedule = burnin_schedule,
        metric_adaptation_end_iter = metric_adaptation_end_iter,
        theta_hat_us_freeze_iter = theta_hat_us_freeze_iter,
        theta_hat_us_rule = theta_hat_us_rule,
        clip_iter = clip_iter,
        clip_iter_tau = clip_iter_tau)
    n_adapt <- resolved_burnin_schedule$n_adapt
    metric_adaptation_end_iter <- resolved_burnin_schedule$metric_adaptation_end_iter
    theta_hat_us_freeze_iter <- resolved_burnin_schedule$theta_hat_us_freeze_iter
    if (identical(burnin_schedule, "automatic")) {
        cat("\nMain burn-in schedule (version ", resolved_burnin_schedule$schedule_version, "):\n", sep = "")
        print(x = resolved_burnin_schedule$phase_table, row.names = FALSE)
    }
    ##
    ## Metric estimator. "chain_mean" and "chain_mean_scaled" both estimate the variance of the
    ## CROSS-CHAIN MEAN of the burn-in chains, which is ~ Sigma / n_chains_burnin when the chains are
    ## independent; "chain_mean_scaled" multiplies that estimate by n_chains_burnin before it enters
    ## the metric, putting it back on the posterior (Sigma) scale. "pooled" estimates Sigma directly.
    ## Any other value used to leave the metric silently un-updated (metric_ready never TRUE).
    metric_estimator <- as.character(metric_estimator)   ## a factor (e.g. from expand.grid) would otherwise skip the x n_chains_burnin below
    if (length(metric_estimator) != 1 || !(metric_estimator %in% c("pooled", "chain_mean", "chain_mean_scaled"))) {
        stop("metric_estimator must be 'pooled', 'chain_mean' or 'chain_mean_scaled'; got: ",
             paste(as.character(metric_estimator), collapse = ", "))
    }
    metric_estimator <- as.character(metric_estimator)
    metric_variance_scale <- if (metric_estimator == "chain_mean_scaled") n_chains_burnin else 1
    ##
    if (!is.logical(tau_weight_by_p_jump) || length(tau_weight_by_p_jump) != 1 || is.na(tau_weight_by_p_jump)) {
        stop("tau_weight_by_p_jump must be a single TRUE or FALSE.")
    }
    ##
    if (!is.logical(tau_if_manual_in_L_units) || length(tau_if_manual_in_L_units) != 1 || is.na(tau_if_manual_in_L_units)) {
        stop("tau_if_manual_in_L_units must be a single TRUE or FALSE.")
    }
    ##
    ## The position-based ChEES objective is only wired into the JOINT branch
    ## (partitioned_HMC = FALSE), which is the production path. Failing here is better than
    ## silently running the kinetic-energy objective under a setting that says otherwise.
    if (tau_objective %in% c("ChEES", "ChEES_per_tau") && isTRUE(partitioned_HMC)) {
        stop("Position-based ChEES objectives are implemented for the joint sampler only (partitioned_HMC = FALSE).")
    }
    ##
    if (isTRUE(tau_if_manual_in_L_units)) {
        if (!isTRUE(manual_tau)) {
            stop("tau_if_manual_in_L_units = TRUE requires manual_tau = TRUE.")
        }
        if (!is.numeric(tau_if_manual) || any(!is.finite(tau_if_manual)) || any(tau_if_manual < 1)) {
            stop("With tau_if_manual_in_L_units = TRUE, tau_if_manual is a leapfrog-step count and must be >= 1; got: ",
                 paste(as.character(tau_if_manual), collapse = ", "))
        }
    }
    ##
    message(paste0("metric estimator = ", metric_estimator,
                   if (metric_variance_scale != 1) paste0(" (x ", metric_variance_scale, ")") else "",
                   " | tau adaptation: objective = ", tau_objective,
                   ", ramp = ", tau_ramp,
                   ", eps re-init at handover = ", eps_reinit_at_ChEES_handover,
                   " | nuisance centre: ", theta_hat_us_rule,
                   if (identical(theta_hat_us_rule, "running_mean_frozen")) paste0(" (frozen from iteration ", theta_hat_us_freeze_iter, ")") else "",
                   ", weight by p_jump = ", tau_weight_by_p_jump,
                   ", manual_tau = ", manual_tau,
                   if (isTRUE(manual_tau)) paste0(" (tau_if_manual = ", paste(tau_if_manual, collapse = "/"),
                                                  if (isTRUE(tau_if_manual_in_L_units)) " leapfrog steps)" else ")") else ""))

  }
  
  
  {  # ----------------------------------------------------------------- list for EHMC params / EHMC struct in C++
    
    EHMC_burnin_as_Rcpp_List <- init_EHMC_burnin_as_Rcpp_List( n_params_main = n_params_main,
                                                               n_nuisance = n_nuisance,
                                                               adapt_delta = adapt_delta,
                                                               LR_main = LR_main,
                                                               LR_us = LR_us)
    
  }
  
  # str(EHMC_Metric_as_Rcpp_List)
  # str(EHMC_args_as_Rcpp_List)
  # str(EHMC_burnin_as_Rcpp_List)
  
  
 # theta_main_vectors_all_chains_input_from_R <- init_object$theta_main_vectors_all_chains_input_from_R
#  theta_us_vectors_all_chains_input_from_R <-   init_object$theta_us_vectors_all_chains_input_from_R
  ##
  theta_main_vectors_all_chains_input_from_R
  theta_us_vectors_all_chains_input_from_R
  ##
  velocity_main_vectors_all_chains_input_from_R <- theta_main_vectors_all_chains_input_from_R
  velocity_us_vectors_all_chains_input_from_R   <- theta_us_vectors_all_chains_input_from_R
  
  ###  theta_vec_mean <- rowMeans(rbind(theta_us_vectors_all_chains_input_from_R, theta_main_vectors_all_chains_input_from_R))
  ##
  EHMC_args_as_Rcpp_List$diffusion_HMC <- diffusion_HMC
  ##
  tictoc::tic()

  debug <- FALSE ## BOOKMARK
  
  message("Printing from init_and_run_burnin_ChESSR:")

  # RcppParallel::setThreadOptions(numThreads = n_chains_burnin * n_threads_WCP_burnin);

  Model_type_R <- Model_type
  
  # ## Fixed SNAPER-HMC / ADAM constants: 
  # beta1_adam = 0.00
  # beta2_adam = 0.95
  # eps_adam = 1e-8
  ##
  kappa = 8.0
  eta_w <- 3
  
  if (sample_nuisance == TRUE) { 
    n_params <- n_params_main + n_nuisance
    index_nuisance <- seq_len(n_nuisance)
    index_main <- (1 + n_nuisance):n_params
  } else { 
    n_params <- n_params_main  
    index_main <- 1:n_params
    if (Model_type == "Stan") {
      ## ---- Stan models with sample_nuisance = FALSE have NO nuisance block:
      ## n_nuisance stays 0 (the complete unconstrained vector is main). No dummy
      ## coordinate is introduced: the C++ gradient/sampler paths handle empty
      ## nuisance blocks, and BridgeStan must always see the full main vector.
      index_nuisance <- integer(0)
      n_nuisance <- 0
    } else {
      ## ---- built-in models: keep the established 1-coordinate dummy convention.
      index_nuisance <- 1
      n_nuisance <- 1
    }
  }

  Model_args_as_Rcpp_List$n_nuisance <- n_nuisance
  Model_args_as_Rcpp_List$n_params_main <- n_params_main

  if (Model_type != "Stan") {
   # n_tests <- ncol(y)
    Model_args_as_Rcpp_List$Model_args_bools[15, 1] <- FALSE # debug
  }
  
  
  n_chains <- n_chains_burnin
  
  print(paste("n_params_main = ", n_params_main))
  print(paste("n_nuisance = ", n_nuisance))
 
  if (sample_nuisance == TRUE) { 
     theta_vec_mean <-  rowMeans( rbind(theta_us_vectors_all_chains_input_from_R, theta_main_vectors_all_chains_input_from_R))
  } else { 
     #### theta_main_vectors_all_chains_input_from_R[,] <- 0 
     theta_vec_mean <-  rowMeans( rbind(theta_main_vectors_all_chains_input_from_R))
  }
  length(theta_vec_mean)
  

 #   theta_vec <- c(theta_us_vectors_all_chains_input_from_R[, 1], theta_main_vectors_all_chains_input_from_R[, 1]) # using inits from chain 1

  ## Historical option, now retired:
  ## burnin_post_adapt_iter: stop this many iterations after n_adapt instead of at n_burnin (NULL = run to n_burnin):
  ## n_burnin is the total main burn-in length; its non-adapting tail is determined by n_adapt, not a separate input.
  last_burnin_iteration <- n_burnin
  iter_seq_burnin <- seq(from = 1, to = last_burnin_iteration, by = 1)

  
  if (metric_shape_main == "diag") {
    EHMC_Metric_as_Rcpp_List$M_dense_main          <- diag(10)
    EHMC_burnin_as_Rcpp_List$M_dense_sqrt          <- diag(10)
    EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- diag(10)
    EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- diag(10)
  }
  
  M_dense_main_non_scaled <-           EHMC_Metric_as_Rcpp_List$M_dense_main
  M_inv_dense_main_non_scaled <-       EHMC_Metric_as_Rcpp_List$M_inv_dense_main
  M_inv_dense_main_chol_non_scaled <-  EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol
  ##
  M_main_diag <- EHMC_Metric_as_Rcpp_List$M_main_vec
  M_inv_main_diag <- EHMC_Metric_as_Rcpp_List$M_inv_main_vec
  
 
  {
      
          ## --------  SNAPER stuff for main params (to load into C++ structs)
          EHMC_burnin_as_Rcpp_List$snaper_m_vec_main <-    theta_vec_mean[index_main]
          EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical <- rep(1, n_params_main)
          ##
          EHMC_burnin_as_Rcpp_List$snaper_w_vec_main <- rep(0.01, n_params_main)
          EHMC_burnin_as_Rcpp_List$eigen_max_main <- sqrt(sum(EHMC_burnin_as_Rcpp_List$snaper_w_vec_main^2) )
          EHMC_burnin_as_Rcpp_List$eigen_vector_main =   EHMC_burnin_as_Rcpp_List$snaper_w_vec_main/sqrt(abs(EHMC_burnin_as_Rcpp_List$eigen_max_main))
          ##
          ## --------  SNAPER stuff for nuisance (to load into C++ structs)
          EHMC_burnin_as_Rcpp_List$snaper_m_vec_us <-     theta_vec_mean[index_nuisance]
          EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical  <- rep(1, n_nuisance)
          EHMC_burnin_as_Rcpp_List$snaper_w_vec_us <- rep(0.01, n_nuisance)
          EHMC_burnin_as_Rcpp_List$eigen_max_us  <- sqrt(sum(  EHMC_burnin_as_Rcpp_List$snaper_w_vec_us^2) )
          EHMC_burnin_as_Rcpp_List$eigen_vector_us =     EHMC_burnin_as_Rcpp_List$snaper_w_vec_us/sqrt(abs( EHMC_burnin_as_Rcpp_List$eigen_max_us ))
          ##
          EHMC_Metric_as_Rcpp_List$M_us_vec <-   1.0 / EHMC_Metric_as_Rcpp_List$M_inv_us_vec  
          ##
          EHMC_burnin_as_Rcpp_List$M_dense_sqrt <- pracma::sqrtm(EHMC_Metric_as_Rcpp_List$M_dense_main)[[1]]
          ##
          snaper_m_vec_all <-    theta_vec_mean
          snaper_s_vec_all_empirical <- rep(1, n_params)
          snaper_w_vec_all <- c(EHMC_burnin_as_Rcpp_List$snaper_w_vec_us, EHMC_burnin_as_Rcpp_List$snaper_w_vec_main)
          eigen_max_all <- sqrt(sum(snaper_w_vec_all^2) )
          eigen_vector_all =   snaper_w_vec_all/sqrt(abs(eigen_max_all))

  }
  
   
  EHMC_args_as_Rcpp_List$eps_main <- 0.00001 # just in case fn_find_initial_eps_main_and_us fails
  EHMC_args_as_Rcpp_List$eps_us   <- 0.00001 # just in case fn_find_initial_eps_main_and_us fails

  
  # At initialization
  EHMC_burnin_as_Rcpp_List$tau_m_adam_main <- 0
  EHMC_burnin_as_Rcpp_List$tau_v_adam_main <- 0
  ##
  EHMC_burnin_as_Rcpp_List$tau_m_adam_us <- 0
  EHMC_burnin_as_Rcpp_List$tau_v_adam_us <- 0
  
  
  # print(paste(" -------------------- Model_args_as_Rcpp_List = -------------------- "))
  # str(Model_args_as_Rcpp_List)
  
  
  ## INITIAL VALUE(S) FOR EPSILON (I.E. THE HMC STEP-SIZE(S)) ---- BOOKMARK ------------------------------------------------------------------------------:
  try({

        if (sample_nuisance == TRUE) {
          theta_us_vec <-   c(theta_vec_mean[index_nuisance])
        } else {
          theta_us_vec <-   c(rep(1, n_nuisance))
        }

        par_res <- (BayesMVP:::fn_find_initial_eps_main_and_us(     theta_main_vec_initial_ref = matrix(c(theta_vec_mean[index_main]), ncol = 1),
                                                                       theta_us_vec_initial_ref = matrix(c(theta_us_vec), ncol = 1),
                                                                       partitioned_HMC = partitioned_HMC,
                                                                       seed = seed,
                                                                       Model_type = Model_type,
                                                                       force_autodiff = force_autodiff,
                                                                       force_PartialLog = force_PartialLog,
                                                                       multi_attempts = multi_attempts,
                                                                       y_ref = y,
                                                                       Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                                       EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                                                                       EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List))


        message(paste("Initial step_sizes = "))
        ## main
        EHMC_args_as_Rcpp_List$eps_main <- min(0.50, par_res[[1]])
        cat(paste("eps_main = "))
        print(EHMC_args_as_Rcpp_List$eps_main)

        ## nuisance
        if (partitioned_HMC == TRUE) {

              EHMC_args_as_Rcpp_List$eps_us <-   min(0.50, par_res[[2]])
              cat(paste("eps_us = "))
              print(EHMC_args_as_Rcpp_List$eps_us)

        } else if (partitioned_HMC == FALSE) {

              EHMC_args_as_Rcpp_List$eps_us <-  EHMC_args_as_Rcpp_List$eps_main

        }

  })
  
  if (!is.null(eps_init)) {
    EHMC_args_as_Rcpp_List$eps_main <- eps_init
    EHMC_args_as_Rcpp_List$eps_us   <- eps_init
    message(paste("eps initialised from previous stage:", signif(eps_init, 4)))
  }
  ##
  ## ---- LARGE-STEP WARM START (eps_initial) ----------------------------------------------
  ##
  ## The step-size found by the heuristic (or carried over via eps_init) can be small enough
  ## that the first iterations barely move, which wastes warm-up on a region the chains should
  ## be crossing quickly. Holding eps at a deliberately LARGE value for the first
  ## eps_initial_iter iterations lets them cover ground first; eps is then handed back to the
  ## value it would have started at and adapted from there as normal.
  ##
  ## During this window tau is 1 * eps (the clip_iter MALA branch below), so these are big
  ## single-leapfrog (MALA) steps rather than long trajectories.
  ##
  ## eps_initial_iter defaults to n_burnin / 10 - i.e. the 50 iterations that were found to
  ## work at n_burnin = 500, scaled proportionally to any other burnin length.
  ##
  eps_main_after_warm_start <- EHMC_args_as_Rcpp_List$eps_main
  eps_us_after_warm_start   <- EHMC_args_as_Rcpp_List$eps_us
  ##
  if (is.null(eps_initial_iter)) {
    eps_initial_iter <- round(n_burnin / 10)
  }
  eps_initial_iter <- max(0L, min(as.integer(round(eps_initial_iter)), as.integer(n_burnin)))
  ##
  use_eps_warm_start <- (!is.null(eps_initial)) && (eps_initial_iter >= 1L)
  ##
  if (use_eps_warm_start) {

        if (!is.numeric(eps_initial) || length(eps_initial) != 1L || !is.finite(eps_initial) || eps_initial <= 0) {
          stop("'eps_initial' must be NULL or a single positive finite number.")
        }
        ##
        ## respect the user's step-size ceilings - a warm start must not exceed them:
        eps_initial_main <- min(eps_initial, max_eps_main)
        eps_initial_us   <- min(eps_initial, max_eps_us)
        ##
        EHMC_args_as_Rcpp_List$eps_main <- eps_initial_main
        EHMC_args_as_Rcpp_List$eps_us   <- eps_initial_us
        ##
        message(paste0( "eps warm start: holding eps at ", signif(eps_initial_main, 4),
                        " for the first ", eps_initial_iter, " of ", n_burnin,
                        " burnin iterations, then reverting to ",
                        signif(eps_main_after_warm_start, 4), " (main)."))

  }

  {
      # --------  eps for main (to load into C++ structs)
      EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- EHMC_args_as_Rcpp_List$eps_main
      EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
  
      # --------  eps for nuisance (to load into C++ structs)
      EHMC_burnin_as_Rcpp_List$eps_m_adam_us <- EHMC_args_as_Rcpp_List$eps_us
      EHMC_burnin_as_Rcpp_List$eps_v_adam_us  <- 0
  
      # --------  tau for main (to load into C++ structs)
      EHMC_args_as_Rcpp_List$tau_main  <-  EHMC_args_as_Rcpp_List$eps_main
      EHMC_burnin_as_Rcpp_List$tau_m_adam_main <- 0
      EHMC_burnin_as_Rcpp_List$tau_v_adam_main <- 0
  
      # --------  tau for nuisance (to load into C++ structs)
      EHMC_args_as_Rcpp_List$tau_us  <-  EHMC_args_as_Rcpp_List$eps_us
      EHMC_burnin_as_Rcpp_List$tau_m_adam_us <- 0
      EHMC_burnin_as_Rcpp_List$tau_v_adam_us <- 0
  
      # --------  M for nuisance (to load into C++ structs)
      M_us_vec <-   rep(1, n_nuisance)
      EHMC_Metric_as_Rcpp_List$M_inv_us_vec <- 1 / M_us_vec
      EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec <- rep(1, n_nuisance)
  
      # -------- M for main (to load into C++ structs)
      EHMC_Metric_as_Rcpp_List$M_inv_dense_main <- diag(rep(1, n_params_main))
      EHMC_Metric_as_Rcpp_List$M_dense_main <- diag(rep(1, n_params_main))
      EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- diag(rep(1, n_params_main))
  }
 
  if (metric_shape_main == "diag") {
    EHMC_Metric_as_Rcpp_List$M_dense_main          <- diag(10)
    EHMC_burnin_as_Rcpp_List$M_dense_sqrt          <- diag(10)
    EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- diag(10)
    EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- diag(10)
  }

  div_main <- p_jump_per_chain <- div_us <- p_jump_us_per_chain <- c()
  
  time_burnin <- 0 
  
  ####  print(theta_main_vectors_all_chains_input_from_R)
  
  shrinkage_factor <- 1
  
  # ## ---- Stan-style windowed dense-metric adaptation (main params) ----
  # win_init_end <- max(clip_iter, round(0.10 * n_adapt))  # eps-only warmup buffer
  # win_slow_end <- round(0.80 * n_adapt)                  # metric frozen after this; eps adapts to n_adapt
  # ##
  # compute_window_ends <- function(first_iter, last_iter, base_size = 25) {
  #   ends <- integer(0); pos <- first_iter; size <- base_size
  #   while (pos <= last_iter) {
  #     this_end <- pos + size - 1
  #     if (this_end + 2 * size > last_iter) this_end <- last_iter  # absorb remainder
  #     this_end <- min(this_end, last_iter)
  #     ends <- c(ends, this_end)
  #     pos  <- this_end + 1
  #     size <- 2 * size
  #   }
  #   ends
  # }
  # win_ends <- compute_window_ends(win_init_end + 1, win_slow_end)
  # if (length(win_ends) == 0) warning("burnin too short for windowed metric adaptation!")
  # message("Metric windows end at iterations: ", paste(win_ends, collapse = ", "))
  # ##
  # win_sum_x   <- rep(0.0, n_params_main)
  # win_sum_xxT <- matrix(0.0, n_params_main, n_params_main)
  # win_n       <- 0L
  
  # ## ---- Phase-2: windowed empirical refinement of the Hessian metric ----
  # win_first_start <- round(0.50 * n_adapt)   # Hessian-only before this
  # win_metric_end  <- round(0.80 * n_adapt)   # metric frozen after this; eps keeps adapting
  # ##
  # compute_window_ends <- function(first_iter, last_iter, base_size = 50) {
  #   ends <- integer(0); pos <- first_iter; size <- base_size
  #   while (pos <= last_iter) {
  #     this_end <- pos + size - 1
  #     if (this_end + 2 * size > last_iter) this_end <- last_iter
  #     this_end <- min(this_end, last_iter)
  #     ends <- c(ends, this_end); pos <- this_end + 1; size <- 2 * size
  #   }
  #   ends
  # }
  # win_ends <- compute_window_ends(win_first_start + 1, win_metric_end)
  # message("Phase-2 metric windows end at: ", paste(win_ends, collapse = ", "))
  # ##
  # win_sum_x    <- rep(0.0, n_params_main)
  # win_sum_xxT  <- matrix(0.0, n_params_main, n_params_main)
  # win_n        <- 0L
  # win_prev_X   <- NULL      # previous iteration's chain states (for lag-1 rho)
  # win_rho_sum  <- 0.0
  # win_rho_n    <- 0L
  # H_inv_seed   <- NULL      # snapshot of Hessian-derived M_inv, taken at first window start
  
  # ## ---- Phase-2: windowed empirical refinement of the Hessian metric ----
  # # win_first_start <- round(0.50 * n_adapt)
  # win_first_start <- n_adapt + 999999   ## Phase-2 disabled
  # ##
  # win_metric_end  <- round(0.80 * n_adapt)
  # compute_window_ends <- function(first_iter, last_iter, base_size = 50) {
  #   ends <- integer(0); pos <- first_iter; size <- base_size
  #   while (pos <= last_iter) {
  #     this_end <- pos + size - 1
  #     if (this_end + 2 * size > last_iter) this_end <- last_iter
  #     this_end <- min(this_end, last_iter)
  #     ends <- c(ends, this_end); pos <- this_end + 1; size <- 2 * size
  #   }
  #   ends
  # }
  # win_ends <- compute_window_ends(win_first_start + 1, win_metric_end)
  # message("Phase-2 metric windows end at: ", paste(win_ends, collapse = ", "))
  # win_sum_x    <- rep(0.0, n_params_main)
  # win_sum_xxT  <- matrix(0.0, n_params_main, n_params_main)
  # win_n        <- 0L
  # win_prev_X   <- NULL
  # win_rho_sum  <- 0.0
  # win_rho_n    <- 0L
  # H_inv_seed   <- NULL
  
  # ## Cross-chain covariance accumulator (replaces Welford-of-the-mean):
  # # cov_crosschain_ema <- NULL
  # # ema_alpha <- 0.05   ## smoothing weight per iteration; ~1/alpha iter memory
  # cov_crosschain_ema <- NULL
  # ema_alpha <- 0.10                                  # faster forgetting
  # cov_start_iter <- round(0.5 * n_adapt)             # accumulate late only
  
  L_main_during_burnin_vec <- numeric(length = 0L)
  L_us_during_burnin_vec <-   numeric(length = 0L)
  ## Post-update settings at each iteration, not the realised per-chain jittered integration times.
  tau_main_during_burnin_vec <- eps_main_during_burnin_vec <- rep(x = NA_real_, times = n_burnin)
  tau_us_during_burnin_vec <- eps_us_during_burnin_vec <- rep(x = NA_real_, times = n_burnin)
  ChEES_criterion_ema_vec <- ChEES_per_tau_gradient_vec <- rep(x = NA_real_, times = n_burnin)
  ChEES_criterion_ema <- NA_real_
  
  ## print every n_refresh iterations (Stan's 'refresh' convention). This used to be
  ## hard-wired to n_burnin/25, which silently ignored the n_refresh the caller passed:
  n_refresh_iter_counter <- max(1L, as.integer(round(n_refresh)))
  
  EHMC_args_as_Rcpp_List$tau_main <- EHMC_args_as_Rcpp_List$eps_main
  
  
  gap <- clip_iter_tau
  ## Tau's optimiser starts at the handover, not at the beginning of the epsilon/metric warm-up.
  n_tau_adaptation_iterations <- n_adapt - gap
  if (!isTRUE(manual_tau)) {
      if (!is.numeric(tau_initial) || length(tau_initial) != 1 || !is.finite(tau_initial) || tau_initial <= 0) {
          stop("tau_initial must be a single positive finite number.")
      }
      if (gap < clip_iter || clip_iter < 1 || gap != round(gap) || clip_iter != round(clip_iter) ||
          n_tau_adaptation_iterations < 1) {
          stop("Adaptive tau requires 1 <= clip_iter <= clip_iter_tau < n_adapt.")
      }
      message("tau adaptation v2: ", tau_ramp, " ramp to tau_initial = ", signif(tau_initial, 5),
              "; one handover at iteration ", gap, "; updates ", gap, "-", n_adapt - 1,
              "; then frozen. No later resets.")
  }
  EHMC_args_as_Rcpp_List$record_kinetic_energy_tau_derivatives <- identical(tau_objective, "KE") && !isTRUE(manual_tau)
  tau_adaptation_iteration_vec <- rep(x = NA_real_, times = n_burnin)
  
  print(paste("clip_iter_tau = ", clip_iter_tau))
  
  ii <- 1
  
  # lp_grad_outs_AD <- BayesMVP:::Rcpp_wrapper_fn_lp_grad(Model_type = Model_type,
  #                                                          force_autodiff = TRUE,
  #                                                          force_PartialLog = FALSE,
  #                                                          multi_attempts = FALSE,
  #                                                          theta_main_vec = theta_vec_mean,
  #                                                          theta_us_vec   =  theta_us_vec,
  #                                                          y = y,
  #                                                          grad_option = "all",
  #                                                          Model_args_as_Rcpp_List = Model_args_as_Rcpp_List)
  # ##
  # print(paste("lp_grad_outs_AD = ", head(lp_grad_outs_AD)))
  # ##
  # lp_grad_outs_MD <- BayesMVP:::Rcpp_wrapper_fn_lp_grad(Model_type = Model_type,
  #                                                          force_autodiff = FALSE,
  #                                                          force_PartialLog = FALSE,
  #                                                          multi_attempts = FALSE,
  #                                                          theta_main_vec = theta_vec_mean,
  #                                                          theta_us_vec   = theta_us_vec,
  #                                                          y = y,
  #                                                          grad_option = "all",
  #                                                          Model_args_as_Rcpp_List = Model_args_as_Rcpp_List)
  # ##
  # print(paste("lp_grad_outs_MD = ", head(lp_grad_outs_MD)))
  # ##
  # print(paste("diff_sum = ", sum(lp_grad_outs_AD - lp_grad_outs_MD)))
  # ##
  # print(paste("diffs (head) = ", head(lp_grad_outs_AD - lp_grad_outs_MD, 100)))
  # ##
  # print(paste("diffs (grad of last 100 params) = ", head( tail((lp_grad_outs_AD - lp_grad_outs_MD), N + 100) , 100)  ))
  # ##
  # print(paste("AD (grad of last 100 params) = ", head( tail((lp_grad_outs_AD), N + 200) , 200)  ))
  # ##
  # print(paste("MD (grad of last 100 params) = ", head( tail((lp_grad_outs_MD), N + 200) , 200)  ))
  
  
  t_cpp_total <- 0
  t_push_total <- 0
  t_tau <- 0
  ##
  ## The separate profiling entry point leaves the normal native return format unchanged.
  ## No timing records are allocated when profiling is disabled.
  ##
  burnin_profile <- NULL
  run_burnin_iteration <- BayesMVP:::fn_persistent_burnin_run_one_iter
  if (debug_burnin_timing) {
      run_burnin_iteration <- BayesMVP:::fn_persistent_burnin_run_one_iter_profiled
      burnin_iteration_profiles <- vector(mode = "list", length = last_burnin_iteration)
      burnin_chain_profiles <- vector(mode = "list", length = last_burnin_iteration)
  }
  
  ##
  ## ---- share_tau_ii_across_chains_in_burnin: read by the C++ burn-in worker from this list, which is pushed to it every iteration:
  ##
  EHMC_args_as_Rcpp_List$share_tau_ii_across_chains <- isTRUE(share_tau_ii_across_chains_in_burnin)
  ##
  worker_ptr <- BayesMVP:::fn_create_persistent_burnin_worker(
                n_threads_R = n_chains_burnin,
                partitioned_HMC_R = partitioned_HMC,
                diffusion_HMC_R = diffusion_HMC,
                Model_type_R = Model_type,
                sample_nuisance_R = sample_nuisance,
                force_autodiff_R = force_autodiff,
                force_PartialLog_R = force_PartialLog,
                multi_attempts_R = multi_attempts,
                y_Eigen_R = y,
                Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,   ## final version, incl. n_nuisance / bools edits
                EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                n_threads_WCP = n_threads_WCP)

  ## Load initial theta into the resident state (ONCE):
  BayesMVP:::fn_persistent_burnin_set_theta(
                worker_ptr,
                theta_main_vectors_all_chains_input_from_R,
                theta_us_vectors_all_chains_input_from_R)

  ## IMPORTANT: worker_ptr must be created AFTER all edits to Model_args_as_Rcpp_List
  ## (n_nuisance, n_params_main, Model_args_bools[15,1] etc.) -- the worker snapshots
  ## Model_args at construction and never re-reads it.
  #   
  # if (is.null(metric_start_iter)) metric_start_iter <- max(clip_iter, round(0.35 * n_adapt))
  # wf_n  <- 0
  # wf_m  <- rep(0, n_params)                          # running mean of the DRAWS (all chains pooled)
  # wf_M2 <- rep(0, n_params)                          # running sum of squared deviations, per coordinate
  # wf_C2 <- matrix(0, n_params_main, n_params_main)   # running cross-products, main block
  # empicical_cov_main <- diag(rep(1, n_params_main))
  # var_draws_all      <- rep(1, n_params)
  # metric_ready       <- FALSE
  ##
  if (is.null(metric_start_iter)) metric_start_iter <- round(n_burnin/10)     # same start as old ii_min
  metric_window_resets <- round(c(0.30, 0.60) * n_adapt)                      # discard early, drift-contaminated draws
  wf_min_draws <- n_params_main + 1
  ##
  wf_n  <- 0
  wf_m  <- rep(0, n_params)
  wf_M2 <- rep(0, n_params)
  wf_C2 <- matrix(0, n_params_main, n_params_main)
  empicical_cov_main <- diag(rep(1, n_params_main))
  var_draws_all      <- rep(1, n_params)
  ##
  metric_ready <- (metric_estimator %in% c("chain_mean", "chain_mean_scaled"))   # pooled: wait for draws; chain_mean(_scaled): behave exactly as before
  
  ## ---- HOLD THE ADAM LEARNING RATE HIGH TO START (learning_rate_initial) ----------------
  ##
  ## The ADAM learning rate drives BOTH the step-size (eps) and the path-length (tau)
  ## adaptation. Starting it high lets the adaptation move a long way in the first few
  ## iterations instead of creeping; it is then dropped back to the run's normal learning rate
  ## for the rest of the burnin, so the later, finer adaptation is not made noisy by a big rate.
  ##
  ## learning_rate_initial_iter defaults to n_burnin / 10 - i.e. the 50 iterations that work at
  ## n_burnin = 500, scaled proportionally to any other burnin length.
  ##
  learning_rate_main_after_hold <- LR_main
  learning_rate_us_after_hold   <- LR_us
  ##
  if (is.null(learning_rate_initial_iter)) {
    learning_rate_initial_iter <- round(n_burnin / 10)
  }
  learning_rate_initial_iter <- max(0L, min(as.integer(round(learning_rate_initial_iter)), as.integer(n_burnin)))
  ##
  use_learning_rate_hold <- (!is.null(learning_rate_initial)) && (learning_rate_initial_iter >= 1L)
  ##
  if (use_learning_rate_hold) {

        if (!is.numeric(learning_rate_initial) || length(learning_rate_initial) != 1L ||
            !is.finite(learning_rate_initial) || learning_rate_initial <= 0) {
          stop("'learning_rate_initial' must be NULL or a single positive finite number.")
        }
        ##
        message(paste0( "learning-rate hold: LR held at ", signif(learning_rate_initial, 4),
                        " for the first ", learning_rate_initial_iter, " of ", n_burnin,
                        " burnin iterations, then set to ", signif(learning_rate_main_after_hold, 4),
                        " (the run's learning_rate) for the remaining ",
                        n_burnin - learning_rate_initial_iter, "."))
        ##
        ## ---- the hold only does something if it is ABOVE the rate it hands back to. Equal values
        ## are almost always an accident (a learning_rate that was raised to the hold value, or a
        ## hold left at the default), and silently doing nothing is the wrong answer:
        ##
        if (learning_rate_initial <= learning_rate_main_after_hold) {

              warning( sprintf( fmt = paste0(
                          "learning_rate_initial (%.4g) is NOT above the run's learning_rate (%.4g), so the ",
                          "learning-rate hold %s. Either raise learning_rate_initial, or lower learning_rate ",
                          "so there is something to hand back down to."),
                          learning_rate_initial,
                          learning_rate_main_after_hold,
                          if (isTRUE(all.equal(learning_rate_initial, learning_rate_main_after_hold)))
                            "changes NOTHING - the rate is the same before and after" else
                            "RAISES the rate after the hold rather than lowering it"),
                       call. = FALSE,
                       immediate. = TRUE)

        }

  }
  ##
 ####  Start burnin   ------------------------------------------------------------------------------------------------------------------------------------------------
 for (ii in iter_seq_burnin) {

              ## ---- learning-rate hold. Nothing else writes LR_main / LR_us inside the loop, so
              ## setting them here, at the top, is what every adaptation in THIS iteration sees:
              if (use_learning_rate_hold) {

                    if (ii <= learning_rate_initial_iter) {
                          EHMC_burnin_as_Rcpp_List$LR_main <- learning_rate_initial
                          EHMC_burnin_as_Rcpp_List$LR_us   <- learning_rate_initial
                    } else {
                          EHMC_burnin_as_Rcpp_List$LR_main <- learning_rate_main_after_hold
                          EHMC_burnin_as_Rcpp_List$LR_us   <- learning_rate_us_after_hold
                          ##
                          if (ii == learning_rate_initial_iter + 1L) {
                            message(paste0( "learning-rate hold finished at iteration ", ii, ": LR ",
                                            signif(learning_rate_initial, 4), " -> ",
                                            signif(learning_rate_main_after_hold, 4),
                                            if (learning_rate_initial <= learning_rate_main_after_hold)
                                              "  (NOT a reduction - see the warning at the start of the burnin)"
                                            else ""))
                          }
                    }

              }
   
              if (metric_type_nuisance == "unit") { 
               
                   #### ---- for unit metric --------------------------------------
                   EHMC_Metric_as_Rcpp_List$M_inv_us_vec  <- rep(1, n_nuisance)
                   EHMC_Metric_as_Rcpp_List$M_us_vec      <- rep(1, n_nuisance)
                   EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec <- rep(1, n_nuisance)
               
              }
                        
              if (ii %% n_refresh_iter_counter == 0) {
                    print(ii)
              }
            
              # if (ii %% round(n_burnin/2) == 0) { 
              #       gc() ## ; gc()
              # }
              
              if (manual_tau == FALSE) {
                      
                try({

                      ## Initialise ONCE. Repeated resets here used to discard already-learned tau values.
                      if (ii == gap) {

                            EHMC_burnin_as_Rcpp_List$tau_m_adam_main <- 0
                            EHMC_burnin_as_Rcpp_List$tau_v_adam_main <- 0
                            EHMC_burnin_as_Rcpp_List$tau_m_adam_us <- 0
                            EHMC_burnin_as_Rcpp_List$tau_v_adam_us <- 0
                            ChEES_criterion_ema <- NA_real_

                            if (partitioned_HMC == TRUE) {

                                        ## For main:
                                        ## or: ## tau_mult * sqrt(EHMC_burnin_as_Rcpp_List$eigen_max_main)
                                        tau_main_prop <-    tau_initial
                                        EHMC_args_as_Rcpp_List$tau_main  <-   if_not_NA_or_INF_else(tau_main_prop, 5.0 *   EHMC_args_as_Rcpp_List$eps_main)
                                        message(paste("tau handover: main =", EHMC_args_as_Rcpp_List$tau_main))

                                        ## For nuisance:
                                        tau_us_prop <-   tau_initial
                                        EHMC_args_as_Rcpp_List$tau_us  <-   if_not_NA_or_INF_else(tau_us_prop, 5.0 * EHMC_args_as_Rcpp_List$eps_us)
                                        message(paste("tau handover: nuisance =", EHMC_args_as_Rcpp_List$tau_us))

                            } else if (partitioned_HMC == FALSE) {

                                        ## For ALL:
                                        tau_main_prop <- tau_initial
                                        EHMC_args_as_Rcpp_List$tau_main <-   if_not_NA_or_INF_else(tau_main_prop, 5.0 * EHMC_args_as_Rcpp_List$eps_main)
                                        message(paste("tau handover: joint =", EHMC_args_as_Rcpp_List$tau_main))
                                        ##
                                        # EHMC_args_as_Rcpp_List$tau_main <- EHMC_args_as_Rcpp_List$tau_main
                                        EHMC_args_as_Rcpp_List$tau_us   <- EHMC_args_as_Rcpp_List$tau_main
                            }

                    }
                })
                
              }
                
               if (manual_tau == TRUE) {

                     ## In L units the target is a LEAPFROG-STEP COUNT, so tau is rebuilt from the
                     ## step size as it currently stands. eps keeps adapting to adapt_delta; only the
                     ## trajectory length is pinned.
                     if (isTRUE(tau_if_manual_in_L_units)) {
                           tau_main_manual <- tau_if_manual[1] * EHMC_args_as_Rcpp_List$eps_main
                           tau_us_manual   <- if (length(tau_if_manual) >= 2)
                                                  tau_if_manual[2] * EHMC_args_as_Rcpp_List$eps_us else tau_main_manual
                     } else {
                           tau_main_manual <- tau_if_manual[1]
                           tau_us_manual   <- if (length(tau_if_manual) >= 2) tau_if_manual[2] else tau_if_manual[1]
                     }
                     ##
                     if (length(tau_if_manual) == 1) {

                       EHMC_args_as_Rcpp_List$tau_main <- tau_main_manual
                       EHMC_args_as_Rcpp_List$tau_us   <- tau_main_manual
                       if (partitioned_HMC == FALSE) EHMC_args_as_Rcpp_List$tau_main <- tau_main_manual

                     } else if (length(tau_if_manual) == 2) {

                       EHMC_args_as_Rcpp_List$tau_main <- tau_main_manual
                       EHMC_args_as_Rcpp_List$tau_us   <- tau_us_manual
                       if (partitioned_HMC == FALSE) EHMC_args_as_Rcpp_List$tau_main <- tau_main_manual

                     }
                 
               }
   
                try({
                     if (!isTRUE(manual_tau) && ii >= clip_iter && ii < gap) {

                             ## Position within the actual ramp window; its final iteration reaches the last stage.
                             tau_ramp_position <- (ii - clip_iter) / max(1, gap - clip_iter - 1)
    
                             if (identical(tau_ramp, "original")) {

                             ## ---- Original four-stage shape: 5*eps, then 1/4, 1/2 and all of tau_initial.
                             ## Historically these were fractions of absolute gap and hard-coded pi, skipping
                             ## the first stage and jumping at handover whenever tau_initial was not pi.
                             if (tau_ramp_position < 0.25)   {

                                     tau_main_prop  <-  min(tau_initial, 5.0 * EHMC_args_as_Rcpp_List$eps_main)
                                     tau_us_prop  <-    min(tau_initial, 5.0 * EHMC_args_as_Rcpp_List$eps_us)
                                     tau_overall_prop <- tau_main_prop

                             } else if (tau_ramp_position < 0.5) {

                                     tau_main_prop  <-  min(tau_initial, max(0.25 * tau_initial, 5.0 * EHMC_args_as_Rcpp_List$eps_main))
                                     tau_us_prop  <-    min(tau_initial, max(0.25 * tau_initial, 5.0 * EHMC_args_as_Rcpp_List$eps_us))
                                     tau_overall_prop <- tau_main_prop

                             } else if (tau_ramp_position < 0.75) {

                                     tau_main_prop  <-  min(tau_initial, max(0.5 * tau_initial, 5.0 * EHMC_args_as_Rcpp_List$eps_main))
                                     tau_us_prop  <-    min(tau_initial, max(0.5 * tau_initial, 5.0 * EHMC_args_as_Rcpp_List$eps_us))
                                     tau_overall_prop <- tau_main_prop

                             } else { 
                               
                                     tau_main_prop  <-  tau_initial
                                     tau_us_prop  <-    tau_initial
                                     tau_overall_prop <- tau_main_prop
                               
                             }
         
                             if (partitioned_HMC == TRUE) {

                                     EHMC_args_as_Rcpp_List$tau_main  <-   if_not_NA_or_INF_else(tau_main_prop, 5.0 * EHMC_args_as_Rcpp_List$eps_main)
                                     EHMC_args_as_Rcpp_List$tau_us    <-   if_not_NA_or_INF_else(tau_us_prop,   5.0 * EHMC_args_as_Rcpp_List$eps_us)

                             } else if (partitioned_HMC == FALSE) {

                                     EHMC_args_as_Rcpp_List$tau_main <- if_not_NA_or_INF_else(tau_main_prop, 5.0 * EHMC_args_as_Rcpp_List$eps_main)
                                     ##
                                     EHMC_args_as_Rcpp_List$tau_main <- EHMC_args_as_Rcpp_List$tau_main
                                     EHMC_args_as_Rcpp_List$tau_us   <- EHMC_args_as_Rcpp_List$tau_main

                             }

                             } else { ## tau_ramp == "staged"

                             ## ---- tau ramp across the clip window [clip_iter, gap).
                             ##
                             ## Short trajectories first - a handful of leapfrog steps, i.e.
                             ## tau = L * eps - lengthening, then fixed fractions of a period.
                             ## Each stage gets an equal slice of the window. To change the ramp,
                             ## edit the two vectors below; nothing else needs touching.
                             ##
                             ## gap is the absolute handover iteration (clip_iter_tau), not an added duration.
                             ## Both ramps use the window itself and the same user-selected endpoint.
                             ##
                             tau_ramp_leapfrog_steps <- c(5, 10, 20)                   ## tau = L * eps
                             tau_ramp_fixed_tau <- tau_initial * c(0.125, 0.25, 0.50, 1.00)
                             ##
                             n_tau_ramp_stages <- length(tau_ramp_leapfrog_steps) + length(tau_ramp_fixed_tau)
                             ##
                             ## 0 at clip_iter, approaching 1 at gap:
                             ## tau_ramp_position was computed above for both ramp shapes.
                             ##
                             tau_ramp_stage <- min( n_tau_ramp_stages,
                                                    1 + floor(tau_ramp_position * n_tau_ramp_stages))
                             ##
                             if (tau_ramp_stage <= length(tau_ramp_leapfrog_steps)) {

                                     tau_main_prop <- min(tau_initial, tau_ramp_leapfrog_steps[tau_ramp_stage] * EHMC_args_as_Rcpp_List$eps_main)
                                     tau_us_prop   <- min(tau_initial, tau_ramp_leapfrog_steps[tau_ramp_stage] * EHMC_args_as_Rcpp_List$eps_us)

                             } else {

                                     ## The leapfrog stages scale with eps; the fixed stages do not.
                                     ## If eps has grown enough that the last leapfrog stage already
                                     ## exceeds the first fixed stage, the ramp would SHORTEN at the
                                     ## handover (e.g. 20 * eps = 1.00 dropping to 0.125 * pi = 0.39
                                     ## at eps = 0.05). Floor the fixed stages at where the leapfrog
                                     ## phase ended so the ramp only ever lengthens; the fixed targets
                                     ## take over again as soon as they exceed that floor.
                                     ## (Restored 2026-09-18: this floor was active in every run up to
                                     ## 09-18 10:21; removing it changed the burn-in relative to those runs.)
                                     ##
                                     tau_ramp_floor <- max(tau_ramp_leapfrog_steps) * EHMC_args_as_Rcpp_List$eps_main
                                     ##
                                     tau_main_prop <- min(tau_initial, max(tau_ramp_fixed_tau[tau_ramp_stage - length(tau_ramp_leapfrog_steps)],
                                                                          tau_ramp_floor))
                                     tau_us_prop <- min(tau_initial, max(tau_ramp_fixed_tau[tau_ramp_stage - length(tau_ramp_leapfrog_steps)],
                                                                        max(tau_ramp_leapfrog_steps) * EHMC_args_as_Rcpp_List$eps_us))

                             }
                             ##
                             tau_overall_prop <- tau_main_prop
         
                             if (partitioned_HMC == TRUE) {

                                     EHMC_args_as_Rcpp_List$tau_main  <-   if_not_NA_or_INF_else(tau_main_prop, 5.0 * EHMC_args_as_Rcpp_List$eps_main)
                                     EHMC_args_as_Rcpp_List$tau_us    <-   if_not_NA_or_INF_else(tau_us_prop,   5.0 * EHMC_args_as_Rcpp_List$eps_us)

                             } else if (partitioned_HMC == FALSE) {

                                     ## the fallback here used to be 5 * tau_main, which compounds the
                                     ## CURRENT path length rather than falling back to a short one;
                                     ## matched to the partitioned branch (5 * eps_main):
                                     EHMC_args_as_Rcpp_List$tau_main <- if_not_NA_or_INF_else(tau_main_prop, 5.0 * EHMC_args_as_Rcpp_List$eps_main)
                                     ##
                                     EHMC_args_as_Rcpp_List$tau_us   <- EHMC_args_as_Rcpp_List$tau_main

                             }

                             } ## end tau_ramp switch
    
                     }
                })
              # }
    

          
               ## ---- Re-assert the manual trajectory length.
               ##
               ## The tau ramp above fires on ii in [clip_iter, gap) and is NOT conditioned on
               ## manual_tau, so without this a manual run spends the whole ramp window at the ramp's
               ## tau rather than the requested one - and in L units the requested tau also has to be
               ## rebuilt here because eps has moved since it was set at the top of the iteration.
               ## Placed BEFORE the max_tau clip so that clip still applies, and before the
               ## ii < clip_iter rule so the L = 1 warm start still happens.
               if (manual_tau == TRUE) {
                     if (isTRUE(tau_if_manual_in_L_units)) {
                           EHMC_args_as_Rcpp_List$tau_main <- tau_if_manual[1] * EHMC_args_as_Rcpp_List$eps_main
                           EHMC_args_as_Rcpp_List$tau_us   <- if (length(tau_if_manual) >= 2)
                                                                  tau_if_manual[2] * EHMC_args_as_Rcpp_List$eps_us else
                                                                  EHMC_args_as_Rcpp_List$tau_main
                     } else {
                           EHMC_args_as_Rcpp_List$tau_main <- tau_if_manual[1]
                           EHMC_args_as_Rcpp_List$tau_us   <- if (length(tau_if_manual) >= 2)
                                                                  tau_if_manual[2] else tau_if_manual[1]
                     }
                     if (partitioned_HMC == FALSE) EHMC_args_as_Rcpp_List$tau_us <- EHMC_args_as_Rcpp_List$tau_main
               }
               if ( EHMC_args_as_Rcpp_List$tau_main  > max_tau_main) {
                 EHMC_args_as_Rcpp_List$tau_main  <- max_tau_main
               }
               if ( EHMC_args_as_Rcpp_List$tau_us  > max_tau_us) {
                 EHMC_args_as_Rcpp_List$tau_us  <- max_tau_us
               }
               if (partitioned_HMC == FALSE) { 
                   if ( EHMC_args_as_Rcpp_List$tau_main  > max_tau_main) {
                     EHMC_args_as_Rcpp_List$tau_main  <- max_tau_main
                   }
               }
               ## Make sure this comes last in the tau adaptation above:
               if (ii < clip_iter) {
                     EHMC_args_as_Rcpp_List$tau_us  <-    1 * EHMC_args_as_Rcpp_List$eps_us;
                     EHMC_args_as_Rcpp_List$tau_main  <-  1 * EHMC_args_as_Rcpp_List$eps_main;
                     if (partitioned_HMC == FALSE) EHMC_args_as_Rcpp_List$tau_main <- 1 * EHMC_args_as_Rcpp_List$eps_main;
               }
   
               #### ---------------------------------------------------------------------------------------------------------------------------------------------------------------
               #### current mean theta across all K chains:
               if (sample_nuisance == TRUE) { 
                        ## same values as rowMeans(rbind(us, main)), without copying the (n_nuisance + n_main) x K matrix every iteration:
                        theta_vec_current_mean <- c(rowMeans(theta_us_vectors_all_chains_input_from_R), rowMeans(theta_main_vectors_all_chains_input_from_R))
                        theta_vec_current_us <-   theta_vec_current_mean[index_nuisance]
                        theta_vec_current_main <- theta_vec_current_mean[index_main]
               } else { 
                        theta_vec_current_mean <- rowMeans(rbind(theta_main_vectors_all_chains_input_from_R))
                        theta_vec_current_main <- theta_vec_current_mean
               }
               # ##
               # if (ii >= metric_start_iter) {
               #     
               #          X_all <- if (sample_nuisance) rbind(theta_us_vectors_all_chains_input_from_R,
               #                                              theta_main_vectors_all_chains_input_from_R)
               #                   else theta_main_vectors_all_chains_input_from_R                       # n_params x K
               #          K     <- ncol(X_all)
               #          m_b   <- rowMeans(X_all)                                  # ONE common mean this iteration, same for every chain
               #          D_b   <- X_all - m_b
               #          delta <- m_b - wf_m
               #          n_new <- wf_n + K
               #          wf_M2 <- wf_M2 + rowSums(D_b^2) + delta^2 * (wf_n * K / n_new)
               #          wf_C2 <- wf_C2 + tcrossprod(D_b[index_main, , drop = FALSE]) +
               #                           tcrossprod(delta[index_main]) * (wf_n * K / n_new)
               #          wf_m  <- wf_m + delta * (K / n_new)
               #          wf_n  <- n_new
               #          ##
               #          if (wf_n >= 2 * n_params_main) {                          # need > p draws for a usable 44x44
               #              var_draws_all <- wf_M2 / (wf_n - 1)
               #              cov_draws     <- wf_C2 / (wf_n - 1)
               #              w <- wf_n / (wf_n + 5)                                # Stan-style regularisation, keeps it PD early on
               #              empicical_cov_main <- w * cov_draws + (1 - w) * 1e-3 * diag(n_params_main)
               #              empicical_cov_main <- 0.5 * (empicical_cov_main + t(empicical_cov_main))
               #              metric_ready <- TRUE
               #          }
               #      
               # }
               if ((metric_estimator == "pooled") && (ii >= metric_start_iter)) {
                    if (ii %in% (metric_window_resets + 1)) { wf_n <- 0; wf_m[] <- 0; wf_M2[] <- 0; wf_C2[] <- 0 }
                    ##
                    X_all <- if (sample_nuisance) rbind(theta_us_vectors_all_chains_input_from_R,
                                                        theta_main_vectors_all_chains_input_from_R)
                             else theta_main_vectors_all_chains_input_from_R
                    K     <- ncol(X_all)
                    m_b   <- rowMeans(X_all)
                    D_b   <- X_all - m_b
                    delta <- m_b - wf_m
                    n_new <- wf_n + K
                    wf_M2 <- wf_M2 + rowSums(D_b^2) + delta^2 * (wf_n * K / n_new)
                    wf_C2 <- wf_C2 + tcrossprod(D_b[index_main, , drop = FALSE]) +
                                     tcrossprod(delta[index_main]) * (wf_n * K / n_new)
                    wf_m  <- wf_m + delta * (K / n_new)
                    wf_n  <- n_new
                    ##
                    if (wf_n >= wf_min_draws) {
                        var_draws_all <- wf_M2 / (wf_n - 1)
                        cov_draws     <- wf_C2 / (wf_n - 1)
                        w <- wf_n / (wf_n + 5)
                        empicical_cov_main <- w * cov_draws + (1 - w) * 1e-3 * diag(n_params_main)
                        empicical_cov_main <- 0.5 * (empicical_cov_main + t(empicical_cov_main))
                        metric_ready <- TRUE
                    }
               }
               ##
               if (ii < 0.333333 * n_burnin)  {
                        shrinkage_factor <- 0.75
                        main_vec_for_Hessian <- theta_vec_current_main
               } else {
                        shrinkage_factor <- 0.25
                        main_vec_for_Hessian <- EHMC_burnin_as_Rcpp_List$snaper_m_vec_main
               }
              
            #### ////  updates for MAIN: --------------------------------------------------------------------------------------------------------------------
            if (partitioned_HMC == TRUE) {
                             ## update snaper_m and snaper_s_empirical (for MAIN):
                             try({
                               delta_old_main <- c(c(theta_vec_current_main) - c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_main))
                               outs_update_snaper_m_and_s <- BayesMVP:::fn_update_snaper_m_and_s( EHMC_burnin_as_Rcpp_List$snaper_m_vec_main,
                                                                                                      EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical,
                                                                                                      theta_vec_current_main,
                                                                                                      ii)
                               EHMC_burnin_as_Rcpp_List$snaper_m_vec_main <- outs_update_snaper_m_and_s[,1]
                               EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical <- outs_update_snaper_m_and_s[,2]
                               delta_new_main <- c(c(theta_vec_current_main) - c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_main))
                             })
            }
            #### ////  updates for NUISANCE: ----------------------------------------------------------------------------------------------------------------
            if (sample_nuisance == TRUE) {

                   if (partitioned_HMC == TRUE) { 
                    
                              ## update snaper_m and snaper_s_empirical (for NUISANCE):
                              try({
                                outs_update_snaper_m_and_s <-   BayesMVP:::fn_update_snaper_m_and_s( EHMC_burnin_as_Rcpp_List$snaper_m_vec_us,  
                                                                                                         EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical, 
                                                                                                         theta_vec_current_us, 
                                                                                                         ii)
                                EHMC_burnin_as_Rcpp_List$snaper_m_vec_us <- outs_update_snaper_m_and_s[,1]
                                EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical <- outs_update_snaper_m_and_s[,2]
                    
                              })
                   }
            }
            #### ////  updates for ALL: --------------------------------------------------------------------------------------------------------------------
            if (partitioned_HMC == FALSE) {
              
                   snaper_m_vec_all <- c(EHMC_burnin_as_Rcpp_List$snaper_m_vec_us, EHMC_burnin_as_Rcpp_List$snaper_m_vec_main)
                   # Check for NaN in initial values
                   if (any(!is.finite(snaper_m_vec_all))) {
                     cat("Warning: NaN in initial snaper_m_vec_all, using theta_vec_current_mean\n")
                     snaper_m_vec_all <- theta_vec_current_mean
                   }
                   snaper_s_vec_all_empirical <- c(EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical, 
                                                   EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical)
                  
                   ## update snaper_m and snaper_s_empirical (for MAIN):
                   try({
                      delta_old_main <- c(c(theta_vec_current_main) - snaper_m_vec_all[index_main])
                      outs_update_snaper_m_and_s <- BayesMVP:::fn_update_snaper_m_and_s( snaper_m = snaper_m_vec_all,
                                                                                             snaper_s_empirical = snaper_s_vec_all_empirical,
                                                                                             theta_vec_mean = theta_vec_current_mean,
                                                                                             ii = ii)
                      snaper_m_vec_all           <- outs_update_snaper_m_and_s[,1]
                      snaper_s_vec_all_empirical <- outs_update_snaper_m_and_s[,2]
                      delta_new_main <- c(c(theta_vec_current_main) - c(snaper_m_vec_all[index_main]))
                      ##
                      EHMC_burnin_as_Rcpp_List$snaper_m_vec_main           <- snaper_m_vec_all[index_main]
                      EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical <- snaper_s_vec_all_empirical[index_main]
                      EHMC_burnin_as_Rcpp_List$snaper_m_vec_us             <- snaper_m_vec_all[index_nuisance]
                      EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical   <- snaper_s_vec_all_empirical[index_nuisance]
                   })
                   # try({
                   #    for (kk in 1:n_chains_burnin) {
                   #      n_draws_kk   <- (ii - 1) * n_chains_burnin + kk                         # Welford count = draws seen so far
                   #      theta_kk     <- c(theta_us_vectors_all_chains_input_from_R[, kk], theta_main_vectors_all_chains_input_from_R[, kk])
                   #      delta_old_kk <- theta_kk[index_main] - snaper_m_vec_all[index_main]
                   #      outs <- BayesMVP:::fn_update_snaper_m_and_s(snaper_m = snaper_m_vec_all,
                   #                                                      snaper_s_empirical = snaper_s_vec_all_empirical,
                   #                                                      theta_vec_mean = theta_kk,
                   #                                                      ii = n_draws_kk)
                   #      snaper_m_vec_all           <- outs[, 1]
                   #      snaper_s_vec_all_empirical <- outs[, 2]
                   #      delta_new_kk <- theta_kk[index_main] - snaper_m_vec_all[index_main]
                   #      if ((metric_type_main == "Empirical") && (metric_shape_main == "dense") && (ii >= 20)) {
                   #        empicical_cov_main <- update_cov_Welford(delta_old = delta_old_kk, delta_new = delta_new_kk,
                   #                                                 ii = n_draws_kk, cov_mat = empicical_cov_main)
                   #      }
                   #    }
                   #    ##
                   #    # if ((metric_type_main == "Empirical") && (metric_shape_main == "dense")) {
                   #    #   if (ii < 20) empicical_cov_main <- diag(snaper_s_vec_all_empirical[index_main])
                   #    #   else         empicical_cov_main <- BayesMVP:::Rcpp_shrink_matrix(shrinkage_factor = shrinkage_factor, mat = empicical_cov_main)
                   #    # }
                   #    EHMC_burnin_as_Rcpp_List$snaper_m_vec_main           <- snaper_m_vec_all[index_main]
                   #    EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical <- snaper_s_vec_all_empirical[index_main]
                   #    EHMC_burnin_as_Rcpp_List$snaper_m_vec_us             <- snaper_m_vec_all[index_nuisance]
                   #    EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical   <- snaper_s_vec_all_empirical[index_nuisance]
                   # })
                   ##
                   # Before computing sqrt_M_all_vec
                   if (any(EHMC_Metric_as_Rcpp_List$M_us_vec <= 0, na.rm = TRUE)) {
                     EHMC_Metric_as_Rcpp_List$M_us_vec[EHMC_Metric_as_Rcpp_List$M_us_vec <= 0] <- 1.0
                   }
                   if (any(EHMC_Metric_as_Rcpp_List$M_main_vec <= 0, na.rm = TRUE)) {
                     EHMC_Metric_as_Rcpp_List$M_main_vec[EHMC_Metric_as_Rcpp_List$M_main_vec <= 0] <- 1.0
                   }
                   ##
                   ## (sqrt_M_all_vec was computed here every iteration but never used - removed; the guards above stay)
                   
           }
           ##
           {
                  # # EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <- matrix(EHMC_burnin_as_Rcpp_List$snaper_m_vec_us) ## bookmark
                  # EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <- matrix(rep(0.0, n_nuisance)) # -------------------
                  
                  ## theta_hat_us = centre of the exact Gaussian rotation for the nuisance (see theta_hat_us_rule).
                  ## Sampling uses whatever centre burn-in ends with, FROZEN - so eps must finish adapting against a
                  ## frozen centre too, or the burn-in acceptance is inflated and eps comes out too big.
                  if (identical(theta_hat_us_rule, "zero") || (ii <= clip_iter)) {
                    EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <- matrix(rep(0.0, n_nuisance))
                  } else if (identical(theta_hat_us_rule, "running_mean") || (ii < theta_hat_us_freeze_iter)) {
                    EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <- matrix(EHMC_burnin_as_Rcpp_List$snaper_m_vec_us)
                  } else if (ii == theta_hat_us_freeze_iter) {
                    EHMC_Metric_as_Rcpp_List$theta_hat_us_vec <- matrix(EHMC_burnin_as_Rcpp_List$snaper_m_vec_us)
                    cat("nuisance centre (theta_hat_us) FROZEN from iteration", ii, "(theta_hat_us_rule = running_mean_frozen)\n")
                  } ## else: running_mean_frozen past the freeze iteration - keep the centre exactly as it is
           }
           ##
           # if ( (metric_type_main == "Empirical") && (metric_shape_main == "dense") ) {
           #   
           #        if (ii < 20) {
           #              empicical_cov_main <- diag(EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical)
           #        } else {
           #              ## update covariance using Welford's algorithm
           #              try({  
           #                              cov_outs <- update_cov_Welford(  
           #                                                                delta_old = delta_old_main,
           #                                                                delta_new = delta_new_main,
           #                                                                ii = ii,
           #                                                                cov_mat = empicical_cov_main)
           #                              ## Get cov mat:
           #                              empicical_cov_main <- cov_outs # for unbiased estimate
           #                              empicical_cov_main <- BayesMVP:::Rcpp_shrink_matrix( shrinkage_factor = shrinkage_factor, 
           #                                                                                      mat = empicical_cov_main)
           #              })
           #        }
           # }
           if (metric_estimator %in% c("chain_mean", "chain_mean_scaled")) {
              if ((metric_type_main == "Empirical") && (metric_shape_main == "dense")) {
                  if (ii < 20) {
                    empicical_cov_main <- diag(EHMC_burnin_as_Rcpp_List$snaper_s_vec_main_empirical)
                  } else {
                    try({
                      empicical_cov_main <- update_cov_Welford(delta_old = delta_old_main, delta_new = delta_new_main,
                                                               ii = ii, cov_mat = empicical_cov_main)
                      empicical_cov_main <- BayesMVP:::Rcpp_shrink_matrix(shrinkage_factor = shrinkage_factor, mat = empicical_cov_main)
                    })
                  }
              }
          }
          ##
          { ## //// Update metric(s):
                    
                    # ii_min <- round(n_burnin/10) ## (- 1 + round(n_burnin/5))
                    ##
                    if (n_burnin > 500) {
                       # ii_min <- 50
                       ii_min <- round(n_burnin/10)
                       # ii_max <- 750
                       ii_max <- n_adapt
                    } else { 
                       ii_min <- round(n_burnin/10)
                       ii_max <- n_adapt
                    }
            
                    #### ii_max <- (n_burnin - round(n_burnin/20))
                    num_diff_e <- 0.0001
            
                   if (burnin_schedule == "automatic" && ii == metric_adaptation_end_iter && !metric_ready) {
                       warning("Final metric update skipped: the pooled estimate is not ready. Existing metrics remain fixed; consider longer burn-in.")
                   }
                   ## if  ((ii < n_adapt) && (ii > ii_min) && (ii %% interval_width_main == 0) && (ii < ii_max))  {
                   ## Request the final update even when the interval width does not divide the cutoff.
                   if ((ii <= metric_adaptation_end_iter) && (ii > ii_min) &&
                       ((ii %% interval_width_main == 0) || (burnin_schedule == "automatic" && ii == metric_adaptation_end_iter)) &&
                       (ii < ii_max) && metric_ready)  {
                                   
                         if (metric_type_main == "unit") { 
                           
                                                   EHMC_Metric_as_Rcpp_List$M_inv_main_vec <- c(rep(1, n_params_main))
                                                   EHMC_Metric_as_Rcpp_List$M_inv_dense_main <-  diag(c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec))
                                                   EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <-  EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                                                   EHMC_Metric_as_Rcpp_List$M_dense_sqrt <-           EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                           
                         } else { 
                           
                                 if (metric_type_main == "Hessian") {
                                   
                                     # if (ii < win_first_start) {
                                       
                                             outs <- update_M_Hessian_main(
                                                     metric_shape_main = metric_shape_main,
                                                     ##
                                                     EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                     EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                                                     ##
                                                     ratio_M_main = ratio_M_main,
                                                     interval_width_main = interval_width_main,
                                                     ##
                                                     force_autodiff_for_metric = force_autodiff_for_metric,
                                                     force_PartialLog_for_metric = force_PartialLog_for_metric,
                                                     force_multi_attempts_for_metric = force_multi_attempts_for_metric,
                                                     ##
                                                     shrinkage_factor = shrinkage_factor,
                                                     num_diff_e = num_diff_e,
                                                     ##
                                                     Model_type = Model_type,
                                                     y = y,
                                                     ##
                                                     Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                                                     ii = ii,
                                                     n_adapt = n_adapt,
                                                     main_vec_for_Hessian = main_vec_for_Hessian)
                                             
                                             EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                                             EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                                             
                                     # }
                                   
                                 # } else if (metric_type_main == "Empirical") {
                                 #   
                                 #           outs <-  update_M_Empirical_main(
                                 #                    debug = debug,
                                 #                    ##
                                 #                    metric_shape_main = metric_shape_main,
                                 #                    ##
                                 #                    EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                 #                    EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                                 #                    ##
                                 #                    # empicical_cov_main = empicical_cov_main,
                                 #                    empicical_cov_main = cov_crosschain_ema, 
                                 #                    ##
                                 #                    ratio_M_main = ratio_M_main,
                                 #                    ##
                                 #                    ii = ii,
                                 #                    n_adapt = n_adapt,
                                 #                    ##
                                 #                    M_decay_type = M_decay_type,
                                 #                    # M_decay_type = "inverse",
                                 #                    # M_decay_type = "linear",
                                 #                    # M_decay_type = "exponential",
                                 #                    # M_decay_type = "cosine",
                                 #                    ##
                                 #                    M_decay_power = M_decay_power,
                                 #                    M_decay_scale = M_decay_scale)
                                 #           
                                 #           EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                                 #           EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                                 #           
                                 #           # str(EHMC_Metric_as_Rcpp_List)
                                 #           
                                 # }
                           
                           
                         # } else if (metric_type_main == "Empirical") {
                         #       
                         #       if (metric_shape_main == "dense" && is.null(cov_crosschain_ema)) {
                         #         message("ii = ", ii, ": cross-chain cov not ready yet — skipping metric update this interval")
                         #       } else {
                         #         outs <- update_M_Empirical_main(  debug = debug,
                         #                                           metric_shape_main = metric_shape_main,
                         #                                           EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                         #                                           EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                         #                                           empicical_cov_main = cov_crosschain_ema,     ## the new estimator
                         #                                           ratio_M_main = ratio_M_main,
                         #                                           ii = ii, n_adapt = n_adapt,
                         #                                           M_decay_type = M_decay_type,
                         #                                           M_decay_power = M_decay_power,
                         #                                           M_decay_scale = M_decay_scale)
                         #         
                         #         EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                         #         EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                         #       }
                         #   
                         # }
                                           
                           } else if (metric_type_main == "Empirical") {
                             
                             if (metric_shape_main == "dense") {
                               ##### dense metric handled by the windowed scheme (EDIT B) — do nothing here
                               outs <- update_M_Empirical_main(  debug = debug,
                                                                 metric_shape_main = metric_shape_main,
                                                                 EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                                 EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                                                                 empicical_cov_main = empicical_cov_main,   ## reverted (diag path ignores it anyway)
                                                                 ratio_M_main = ratio_M_main,
                                                                 ii = ii, n_adapt = n_adapt,
                                                                 M_decay_type = M_decay_type,
                                                                 M_decay_power = M_decay_power,
                                                                 M_decay_scale = M_decay_scale,
                                                                 variance_scale = metric_variance_scale,   ## x n_chains_burnin only for chain_mean_scaled
                                                                 compute_M_dense_sqrt = FALSE)             ## M_dense_sqrt is only read by SNAPER
                               
                               EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                               EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                             } else {
                               outs <- update_M_Empirical_main(  debug = debug,
                                                                 metric_shape_main = metric_shape_main,
                                                                 EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                                 EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                                                                 empicical_cov_main = empicical_cov_main,   ## reverted (diag path ignores it anyway)
                                                                 ratio_M_main = ratio_M_main,
                                                                 ii = ii, n_adapt = n_adapt,
                                                                 M_decay_type = M_decay_type,
                                                                 M_decay_power = M_decay_power,
                                                                 M_decay_scale = M_decay_scale,
                                                                 variance_scale = metric_variance_scale,   ## x n_chains_burnin only for chain_mean_scaled
                                                                 compute_M_dense_sqrt = FALSE)             ## M_dense_sqrt is only read by SNAPER
                               
                               EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                               EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                             }
                             
                           }
                      
                           
                         }
                           
                           if (sample_nuisance == TRUE)  {
                             
                             # if  ((ii < n_adapt) && (ii > ii_min) && (ii %% interval_width_nuisance == 0) && (ii < ii_max))  {
                             # if ((ii < n_adapt) && (ii > ii_min) && (ii %% interval_width_main == 0) && (ii < ii_max) && metric_ready)  {
                             if ((ii <= metric_adaptation_end_iter) && (ii > ii_min) &&
                                 ((ii %% interval_width_nuisance == 0) || (burnin_schedule == "automatic" && ii == metric_adaptation_end_iter)) &&
                                 (ii < ii_max) && metric_ready)  {
                               
                                     try({
                                       
                                       if (metric_type_nuisance == "unit") { 
                                         
                                               #### ---- for unit metric --------------------------------------
                                               EHMC_Metric_as_Rcpp_List$M_inv_us_vec  <- rep(1, n_nuisance)
                                               EHMC_Metric_as_Rcpp_List$M_us_vec      <- rep(1, n_nuisance)
                                               EHMC_burnin_as_Rcpp_List$sqrt_M_us_vec <- rep(1, n_nuisance)
                                         
                                       } else if (metric_type_nuisance == "Empirical") {
                                         
                                               if (metric_estimator == "pooled") {
                                                    proposed_variance_vec_us <- var_draws_all[index_nuisance]
                                               } else {
                                                    proposed_variance_vec_us <- metric_variance_scale * EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical
                                               }
                                               if (length(proposed_variance_vec_us) != n_nuisance) stop("BUG: proposed_variance_vec_us wrong length!")
                                               ##
                                               outs <- update_M_diag_Empirical_nuisance(  EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                                                          EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                                                                                          ##
                                                                                          proposed_variance_vec = proposed_variance_vec_us,
                                                                                          ##
                                                                                          ratio = ratio_M_us,
                                                                                          ##
                                                                                          ii = ii,
                                                                                          n_adapt = n_adapt,
                                                                                          ##
                                                                                          M_decay_type = M_decay_type,
                                                                                          # M_decay_type = "inverse",
                                                                                          # M_decay_type = "linear",
                                                                                          # M_decay_type = "exponential",
                                                                                          # M_decay_type = "cosine",
                                                                                          ##
                                                                                          M_decay_power = M_decay_power,
                                                                                          M_decay_scale = M_decay_scale)
                                               ##
                                               EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                                               EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                                               # M_inv_us_vec <- EHMC_Metric_as_Rcpp_List$M_inv_us_vec
                                         
                                       } else if (metric_type_nuisance == "uniform_diag") {
                                         
                                               if (metric_estimator == "pooled") {
                                                    proposed_variance_vec_us <- var_draws_all[index_nuisance]
                                               } else {
                                                    proposed_variance_vec_us <- metric_variance_scale * EHMC_burnin_as_Rcpp_List$snaper_s_vec_us_empirical
                                               }
                                               if (length(proposed_variance_vec_us) != n_nuisance) stop("BUG: proposed_variance_vec_us wrong length!")
                                               ##
                                               outs <- update_M_diag_Empirical_nuisance(   EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                                                                           EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                                                                                           ##
                                                                                           proposed_variance_vec = proposed_variance_vec_us,
                                                                                           ##
                                                                                           ratio = ratio_M_us,
                                                                                           ##
                                                                                           ii = ii,
                                                                                           n_adapt = n_adapt,
                                                                                           ##
                                                                                           M_decay_type = M_decay_type,
                                                                                           # M_decay_type = "inverse",
                                                                                           # M_decay_type = "linear",
                                                                                           # M_decay_type = "exponential",
                                                                                           # M_decay_type = "cosine",
                                                                                           ##
                                                                                           M_decay_power = M_decay_power,
                                                                                           M_decay_scale = M_decay_scale)
                                               ##
                                               EHMC_Metric_as_Rcpp_List <- outs$EHMC_Metric_as_Rcpp_List
                                               EHMC_burnin_as_Rcpp_List <- outs$EHMC_burnin_as_Rcpp_List
                                               ##
                                               EHMC_Metric_as_Rcpp_List$M_inv_us_vec <- rep(median(EHMC_Metric_as_Rcpp_List$M_inv_us_vec), n_nuisance)
                                               EHMC_Metric_as_Rcpp_List$M_us_vec <- rep(median(EHMC_Metric_as_Rcpp_List$M_us_vec), n_nuisance)
                                               EHMC_Metric_as_Rcpp_List$sqrt_M_us_vec <- rep(median(EHMC_Metric_as_Rcpp_List$sqrt_M_us_vec), n_nuisance)
                                         
                                       }
                                       
                                     })
                               
                             }
                             
                           }
                      
                    } ## // end of "if  ( (ii >  (- 1 + round(n_burnin/5))) && (ii %% interval_width_main == 0) && (ii < (n_burnin - round(n_burnin/10))) )  {" loop
            
          }  ## // end of metric block
   
          ## ---- re-initialise eps against the settled metric at ChEES handover (eps_reinit_at_ChEES_handover = TRUE) ----
          if ((ii == gap) && !isTRUE(eps_reinit_at_ChEES_handover)) {
            cat("eps KEPT at ChEES handover (ii =", ii, "):", signif(EHMC_args_as_Rcpp_List$eps_main, 4), "(eps_reinit_at_ChEES_handover = FALSE)\n")
          }
          if ((ii == gap) && isTRUE(eps_reinit_at_ChEES_handover)) {
            try({
              par_res <- BayesMVP:::fn_find_initial_eps_main_and_us(
                  theta_main_vec_initial_ref = matrix(rowMeans(theta_main_vectors_all_chains_input_from_R), ncol = 1),
                  theta_us_vec_initial_ref   = matrix(rowMeans(theta_us_vectors_all_chains_input_from_R),   ncol = 1),
                  partitioned_HMC = partitioned_HMC,
                  seed = seed + ii,
                  Model_type = Model_type,
                  force_autodiff = force_autodiff,
                  force_PartialLog = force_PartialLog,
                  multi_attempts = multi_attempts,
                  y_ref = y,
                  Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                  EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                  EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List)
              eps_new <- min(max_eps_main, par_res[[1]])
              EHMC_args_as_Rcpp_List$eps_main <- eps_new
              EHMC_args_as_Rcpp_List$eps_us   <- eps_new
              EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- eps_new ; EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
              EHMC_burnin_as_Rcpp_List$eps_m_adam_us   <- eps_new ; EHMC_burnin_as_Rcpp_List$eps_v_adam_us   <- 0
              cat("eps re-initialised at ChEES handover (ii =", ii, "):", signif(eps_new, 4), "\n")
            })
          }
          ##
          if (isTRUE(getOption("BayesMVP_force_L1", FALSE))) {
              EHMC_args_as_Rcpp_List$tau_main <- EHMC_args_as_Rcpp_List$eps_main
              EHMC_args_as_Rcpp_List$tau_us   <- EHMC_args_as_Rcpp_List$eps_us
          }
          ##
          ## //////////////////   --------------------------------  Perform iteration  ------------------------------------------------------------------------------
          try({
            
                                        # if (parallel_method == "RcppParallel") {
                                        #   fn <- BayesMVP:::fn_R_RcppParallel_EHMC_single_iter_burnin
                                        # } else {  ### OpenMP
                                        #   fn <- BayesMVP:::fn_R_OpenMP_EHMC_single_iter_burnin
                                        # }
                                        # # 
                                        # # # t0 <- proc.time()[3]
                                        # # ##
                                        # result <-   fn(      n_threads_R = n_chains_burnin,
                                        #                      seed_R = seed + ii, ## seed + ii
                                        #                      sample_nuisance_R = sample_nuisance,
                                        #                      n_nuisance_to_track = 1, ## n_nuisance_to_track,
                                        #                      n_iter_R = 1,
                                        #                      current_iter_R = ii,
                                        #                      n_adapt  = 0,
                                        #                      partitioned_HMC_R = partitioned_HMC,
                                        #                      diffusion_HMC_R = diffusion_HMC,
                                        #                      clip_iter = clip_iter,
                                        #                      gap =  gap,
                                        #                      burnin_indicator = FALSE,
                                        #                      metric_type_nuisance = metric_type_nuisance,
                                        #                      metric_type_main = metric_type_main,
                                        #                      shrinkage_factor = shrinkage_factor,
                                        #                      max_eps_main = max_eps_main,
                                        #                      max_eps_us = max_eps_us,
                                        #                      tau_main_target = 0, # dummy arg
                                        #                      tau_us_target = 0,   # dummy arg
                                        #                      main_L_manual = FALSE,  # dummy arg
                                        #                      L_main_if_manual = 0,   # dummy arg
                                        #                      us_L_manual = FALSE,    # dummy arg
                                        #                      L_us_if_manual = 0,     # dummy arg
                                        #                      max_L = max_L,
                                        #                      tau_mult = tau_mult,
                                        #                      ratio_M_us = ratio_M_us,
                                        #                      ratio_Hess_main = ratio_M_main,
                                        #                      M_interval_width = interval_width_main,
                                        #                      Model_type_R =  Model_type,
                                        #                      force_autodiff_R = force_autodiff,
                                        #                      force_PartialLog_R = force_PartialLog ,
                                        #                      multi_attempts_R = multi_attempts,
                                        #                      ##
                                        #                      theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                                        #                      velocity_main_vectors_all_chains_input_from_R = velocity_main_vectors_all_chains_input_from_R,
                                        #                      ##
                                        #                      theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R,
                                        #                      velocity_us_vectors_all_chains_input_from_R = velocity_us_vectors_all_chains_input_from_R,
                                        #                      ##
                                        #                      y_Eigen_R =  y,
                                        #                      Model_args_as_Rcpp_List =  Model_args_as_Rcpp_List,
                                        #                      EHMC_args_as_Rcpp_List =   EHMC_args_as_Rcpp_List,
                                        #                      EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                                        #                      EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                                        #                      ##
                                        #                      n_threads_WCP = n_threads_WCP)
                                        #
                                        # ## Push this iteration's eps / tau / metric (same wholesale overwrite the old
                                        # ## wrapper did internally every call -- adaptation semantics identical):
                                        # BayesMVP:::fn_persistent_burnin_update_adaptation( worker_ptr,
                                        #                                                       EHMC_args_as_Rcpp_List,
                                        #                                                       EHMC_Metric_as_Rcpp_List)
                                        # 
                                        # t0 <- proc.time()[3]
                                        ##
                                        t0 <- proc.time()[3]
                                        BayesMVP:::fn_persistent_burnin_update_adaptation( worker_ptr,
                                                                                              EHMC_args_as_Rcpp_List,
                                                                                              EHMC_Metric_as_Rcpp_List)
                                        ##
                                        push_elapsed_seconds <- proc.time()[3] - t0
                                        t_push_total <- t_push_total + push_elapsed_seconds
                                        t0 <- proc.time()[3]
                                        ##
                                        ## Run one iteration (all heavy state resident in C++):
                                        result <- run_burnin_iteration(
                                                  worker_ptr,
                                                  seed_R = seed + ii,
                                                  current_iter_R = ii)
                                        ##
                                        ## Everything AFTER this point in the loop stays byte-for-byte identical:
                                        ## the returned list has the same element names, so all the
                                        ## "theta_main_vectors_all_chains_input_from_R <- result$..." unpacking,
                                        ## snaper updates, eps/tau ADAM, metric updates, printing -- all unchanged.
                                        # ##
                                        cpp_elapsed_seconds <- proc.time()[3] - t0
                                        t_cpp_total <- t_cpp_total + cpp_elapsed_seconds
                                        if (debug_burnin_timing) {
                                            profile <- result$burnin_profile
                                            if (is.null(profile)) stop("The native burn-in profiler returned no diagnostics.")
                                            burnin_iteration_profiles[[ii]] <- data.frame(
                                                iter = ii,
                                                n_chains_burnin = n_chains_burnin,
                                                n_threads_WCP = n_threads_WCP,
                                                cpp_call_seconds = unname(cpp_elapsed_seconds),
                                                push_seconds = unname(push_elapsed_seconds),
                                                parallel_seconds = profile$parallel_seconds,
                                                output_seconds = profile$output_seconds,
                                                call_residual_seconds = unname(cpp_elapsed_seconds) - profile$parallel_seconds - profile$output_seconds,
                                                max_chain_seconds = max(profile$chain_seconds),
                                                mean_chain_seconds = mean(profile$chain_seconds),
                                                total_leapfrog_steps = sum(profile$n_leapfrog_steps))
                                            burnin_chain_profiles[[ii]] <- data.frame(
                                                iter = ii,
                                                chain = seq_len(n_chains_burnin),
                                                elapsed_seconds = profile$chain_seconds,
                                                start_offset_seconds = profile$chain_start_offset_seconds,
                                                finish_offset_seconds = profile$chain_start_offset_seconds + profile$chain_seconds,
                                                n_leapfrog_steps = profile$n_leapfrog_steps)
                                        }
                                    
                                    
                                        # if (ii %% 25 == 0) {
                                        # 
                                        #     ## us_only
                                        #     ## corr_only
                                        #     ## coeff_only
                                        #     ## prev_only
                                        #     ## cutpoints_only
                                        #     ##
                                        #     ## L_diag
                                        # 
                                        #     # ##
                                        #     BayesMVP:::set_debug_cutpoint_grads(TRUE)   # debug on
                                        # 
                                        #     Model_args_as_Rcpp_List$Model_args_ints[4, 1] <- 10
                                        #     ##
                                        #     lp_grad_outs_AD <- BayesMVP:::Rcpp_wrapper_fn_lp_grad(Model_type = Model_type,
                                        #                                                              force_autodiff = FALSE,
                                        #                                                              force_PartialLog = FALSE,
                                        #                                                              multi_attempts = FALSE,
                                        #                                                              theta_main_vec = theta_main_vectors_all_chains_input_from_R[, 1],
                                        #                                                              theta_us_vec   =  theta_us_vectors_all_chains_input_from_R[, 1],
                                        #                                                              y = y,
                                        #                                                              grad_option = "all",
                                        #                                                              Model_args_as_Rcpp_List = Model_args_as_Rcpp_List)
                                        #     ##
                                        #     Model_args_as_Rcpp_List$Model_args_ints[4, 1] <- 10
                                        #     ##
                                        #     # print(paste("lp_grad_outs_AD = ", head(lp_grad_outs_AD)))
                                        #     ##
                                        #     lp_grad_outs_MD <- BayesMVP:::Rcpp_wrapper_fn_lp_grad(Model_type = Model_type,
                                        #                                                              force_autodiff = FALSE,
                                        #                                                              force_PartialLog = FALSE,
                                        #                                                              multi_attempts = FALSE,
                                        #                                                              theta_main_vec = theta_main_vectors_all_chains_input_from_R[, 1],
                                        #                                                              theta_us_vec   = theta_us_vectors_all_chains_input_from_R[, 1],
                                        #                                                              y = y,
                                        #                                                              grad_option = "all",
                                        #                                                              Model_args_as_Rcpp_List = Model_args_as_Rcpp_List)
                                        #     ##
                                        #     # print(paste("lp_grad_outs_MD = ", head(lp_grad_outs_MD)))
                                        #     # ##
                                        #     # print(paste("diff_sum = ", sum(lp_grad_outs_AD - lp_grad_outs_MD)))
                                        #     ##
                                        #     print(paste("diffs (head - lp + first 99 u's) = ", head(lp_grad_outs_AD - lp_grad_outs_MD, 100)))
                                        #     ##
                                        #     ## ---- Main params:
                                        #     ##
                                        #     n_pars_main <- n_params_main
                                        #     ##
                                        #     print(paste("AD (grad of last n_pars_main params) = ",
                                        #                 head( tail((lp_grad_outs_AD), N + n_pars_main) , n_pars_main)  ))
                                        #     print(paste("MD (grad of last n_pars_main params) = ",
                                        #                 head( tail((lp_grad_outs_MD), N + n_pars_main) , n_pars_main)  ))
                                        #     ##
                                        #     # print(paste("AD (grad of last n_pars_main params - index 74 to 79) = ",
                                        #     #             head( tail((lp_grad_outs_AD), N + n_pars_main) , n_pars_main)[74:79]   ))
                                        #     # print(paste("MD (grad of last n_pars_main params - index 74 to 79) = ",
                                        #     #             head( tail((lp_grad_outs_MD), N + n_pars_main) , n_pars_main)[74:79]   ))
                                        #     ##
                                        #     print(paste("diffs (grad of last n_pars_main params) = ",
                                        #                 head( tail((lp_grad_outs_AD - lp_grad_outs_MD), N + n_pars_main) , n_pars_main)  ))
                                        #     # ##
                                        #     # print(paste("diffs (grad of last n_pars_main params - index 74:79) = ",
                                        #     #             head( tail((lp_grad_outs_AD - lp_grad_outs_MD), N + n_pars_main) , n_pars_main)[74:79] ))
                                        #     ##
                                        #     ll_1  <- tail(lp_grad_outs_AD, N)
                                        #     ll_10 <- tail(lp_grad_outs_MD, N)
                                        #     d <- ll_1 - ll_10
                                        #     print(c(sum_d = sum(d), max_abs = max(abs(d)), n_big = sum(abs(d) > 0.05), n_zero_slots = sum(ll_10 == 0)))
                                        #     print(which(abs(d) > 0.05))
                                        #     ##
                                        #     BayesMVP:::set_debug_cutpoint_grads(FALSE)  # debug off
                                        #     ##
                                        #     # ## Your main params are ordered: correlations, coefficients, prev
                                        #     # ## With n_corrs=30, what are params 46 and 52?
                                        #     # n_corrs <- 30
                                        #     # n_covariates_total <- sum(model_args_list$n_covariates_per_outcome_mat)
                                        #     # cat(sprintf("n_corrs=%d, n_covs_total=%d, n_pops=%d, total=%d\n",
                                        #     #             n_corrs, n_covariates_total, model_args_list$n_pops,
                                        #     #             n_corrs + n_covariates_total + model_args_list$n_pops))
                                        #     #
                                        #     # ## Map index to parameter:
                                        #     # for (idx in c(35, 41)) {
                                        #     #   if (idx <= n_corrs) {
                                        #     #     cat(sprintf("Index %d = correlation param %d\n", idx, idx))
                                        #     #   } else if (idx <= n_corrs + n_covariates_total) {
                                        #     #     coeff_idx <- idx - n_corrs
                                        #     #     cat(sprintf("Index %d = coefficient %d\n", idx, coeff_idx))
                                        #     #   } else {
                                        #     #     prev_idx <- idx - n_corrs - n_covariates_total
                                        #     #     cat(sprintf("Index %d = prevalence pop %d\n", idx, prev_idx))
                                        #     #   }
                                        #     # }
                                        # 
                                        # 
                                        # }

                           #  result <- parallel::mccollect(result)

                            if (isTRUE(EHMC_args_as_Rcpp_List$record_kinetic_energy_tau_derivatives)) {
                                if (length(result$kinetic_energy_rate_main_proposed) != n_chains_burnin ||
                                    length(result$kinetic_energy_rate_us_proposed) != n_chains_burnin) {
                                    stop("Corrected KE adaptation needs the updated native BayesMVP build. Rebuild and restart R; no legacy KE fallback is used.")
                                }
                            }
                            theta_main_vectors_all_chains_input_from_R <-                  result$theta_main_vectors_all_chains_output_to_R
                            theta_main_0_burnin_tau_adapt_all_chains_input_from_R <-       result$theta_main_0_burnin_tau_adapt_all_chains_input_from_R
                            theta_main_prop_burnin_tau_adapt_all_chains_input_from_R <-    result$theta_main_prop_burnin_tau_adapt_all_chains_input_from_R
                            ##
                            velocity_main_vectors_all_chains_input_from_R <-               result$velocity_main_vectors_all_chains_output_to_R
                            velocity_main_0_burnin_tau_adapt_all_chains_input_from_R <-    result$velocity_main_0_burnin_tau_adapt_all_chains_input_from_R
                            velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R <- result$velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R
                            ##
                            tau_main_ii_vec <- result[[3]][6,]
                            
                            # print( result$theta_us_prop_burnin_tau_adapt_all_chains_input_from_R)
                            
                         # if (sample_nuisance == TRUE) { 
                           
                                theta_us_vectors_all_chains_input_from_R <-                  result$theta_us_vectors_all_chains_output_to_R
                                theta_us_0_burnin_tau_adapt_all_chains_input_from_R <-       result$theta_us_0_burnin_tau_adapt_all_chains_input_from_R
                                theta_us_prop_burnin_tau_adapt_all_chains_input_from_R <-    result$theta_us_prop_burnin_tau_adapt_all_chains_input_from_R
                                ##
                                velocity_us_vectors_all_chains_input_from_R <-               result$velocity_us_vectors_all_chains_output_to_R
                                velocity_us_0_burnin_tau_adapt_all_chains_input_from_R <-    result$velocity_us_0_burnin_tau_adapt_all_chains_input_from_R
                                velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R <- result$velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R
                                ##
                                tau_us_ii_vec <-   result[[6]][6,]
                                if (debug) {
                                  if (ii %% 25 == 0) {
                                      cat("velocity_us_0 range:", range(velocity_us_0_burnin_tau_adapt_all_chains_input_from_R), "\n")
                                      cat("velocity_us_prop range:", range(velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R), "\n")
                                  }
                                }
                         # }
 
          })
   
   
   
   
                        try({
                          
                              div_main <- div_us <- c()
                              p_jump_per_chain <- p_jump_us_per_chain <- c()
                              
                              for (kk in 1:n_chains_burnin) {
                                  div_main[kk] <- result$other_main_out_vector_all_chains_output_to_R[, kk][2]
                                  p_jump_per_chain[kk] <-   result$other_main_out_vector_all_chains_output_to_R[, kk][1]
                                  if (sample_nuisance == TRUE) { 
                                      div_us[kk] <- result$other_us_out_vector_all_chains_output_to_R[, kk][2]
                                      p_jump_us_per_chain[kk] <-   result$other_us_out_vector_all_chains_output_to_R[, kk][1]
                                  }
                              }
                          
                        })
                         
                         # ## ---- Phase-2 accumulation + blended assignment (TOP LEVEL of loop) ----
                         # if (metric_type_main == "Hessian" && metric_shape_main == "dense") {
                         #   tryCatch({
                         #     
                         #     if (ii == win_first_start) {
                         #       H_inv_seed <- EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                         #       message("ii = ", ii, ": Hessian metric snapshotted as Phase-2 prior")
                         #     }
                         #     
                         #     if (ii > win_first_start && ii <= win_metric_end) {
                         #       Xc <- theta_main_vectors_all_chains_input_from_R
                         #       win_sum_x   <- win_sum_x   + rowSums(Xc)
                         #       win_sum_xxT <- win_sum_xxT + tcrossprod(Xc)
                         #       win_n       <- win_n + ncol(Xc)
                         #       if (!is.null(win_prev_X)) {
                         #         for (jj in 1:min(5, n_params_main)) {
                         #           r1 <- suppressWarnings(cor(Xc[jj, ], win_prev_X[jj, ]))
                         #           if (is.finite(r1)) { win_rho_sum <- win_rho_sum + r1; win_rho_n <- win_rho_n + 1L }
                         #         }
                         #       }
                         #       win_prev_X <- Xc
                         #     }
                         #     
                         #     if ((ii %in% win_ends) && (win_n > 5L) && !is.null(H_inv_seed)) {
                         #       p      <- n_params_main
                         #       n_used <- win_n
                         #       mean_w <- win_sum_x / n_used
                         #       cov_w  <- (win_sum_xxT - n_used * tcrossprod(mean_w)) / (n_used - 1)
                         #       cov_w  <- 0.5 * (cov_w + t(cov_w))
                         #       rho_hat <- if (win_rho_n > 0) max(0, min(0.99, win_rho_sum / win_rho_n)) else 0.9
                         #       tau_hat <- (1 + rho_hat) / (1 - rho_hat)
                         #       n_eff   <- n_used / tau_hat
                         #       # w_blend <- n_eff / (n_eff + 5 * p)
                         #       w_blend <- n_eff / (n_eff + 20 * p)
                         #       cov_blend <- w_blend * cov_w + (1 - w_blend) * H_inv_seed
                         #       cov_blend <- cov_blend + 1e-4 * mean(diag(cov_blend)) * diag(p)
                         #       cov_blend <- BayesMVP:::Rcpp_near_PD(0.5 * (cov_blend + t(cov_blend)))
                         #       EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- cov_blend
                         #       EHMC_Metric_as_Rcpp_List$M_dense_main          <- BayesMVP:::Rcpp_solve(cov_blend)
                         #       EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- BayesMVP:::Rcpp_Chol(cov_blend)
                         #       EHMC_burnin_as_Rcpp_List$M_dense_sqrt          <- pracma::sqrtm(EHMC_Metric_as_Rcpp_List$M_dense_main)[[1]]
                         #       EHMC_Metric_as_Rcpp_List$M_inv_main_vec  <- diag(cov_blend)
                         #       EHMC_Metric_as_Rcpp_List$M_main_vec      <- 1 / diag(cov_blend)
                         #       EHMC_burnin_as_Rcpp_List$sqrt_M_main_vec <- sqrt(EHMC_Metric_as_Rcpp_List$M_main_vec)
                         #       message(sprintf("ii = %d: Phase-2 metric assigned (n=%d, n_eff~%.0f, w=%.2f, tau_hat~%.0f)",
                         #                       ii, n_used, n_eff, w_blend, tau_hat))
                         #       
                         #       # ## re-initialize eps against the NEW metric + restore tau:
                         #       # try({
                         #       #   par_res <- BayesMVP:::fn_find_initial_eps_main_and_us(
                         #       #     theta_main_vec_initial_ref = matrix(rowMeans(theta_main_vectors_all_chains_input_from_R), ncol = 1),
                         #       #     theta_us_vec_initial_ref   = matrix(rowMeans(theta_us_vectors_all_chains_input_from_R), ncol = 1),
                         #       #     partitioned_HMC = partitioned_HMC, seed = seed + ii,
                         #       #     Model_type = Model_type,
                         #       #     force_autodiff = force_autodiff, force_PartialLog = force_PartialLog,
                         #       #     multi_attempts = multi_attempts, y_ref = y,
                         #       #     Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                         #       #     EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                         #       #     EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List)
                         #       #   EHMC_args_as_Rcpp_List$eps_main <- min(max_eps_main, par_res[[1]])
                         #       #   EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- EHMC_args_as_Rcpp_List$eps_main
                         #       #   EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
                         #       #   EHMC_args_as_Rcpp_List$tau_main <- max(EHMC_args_as_Rcpp_List$tau_main,
                         #       #                                          12 * EHMC_args_as_Rcpp_List$eps_main)
                         #       #   message(sprintf("ii = %d: eps re-initialized to %.4f, tau restored to %.3f",
                         #       #                   ii, EHMC_args_as_Rcpp_List$eps_main, EHMC_args_as_Rcpp_List$tau_main))
                         #       # })
                         #       
                         #       ## analytic eps rescale against the new metric:
                         #       ev_old <- eigen(H_inv_seed, symmetric = TRUE, only.values = TRUE)$values
                         #       ev_new <- eigen(cov_blend,  symmetric = TRUE, only.values = TRUE)$values
                         #       scale_fac <- sqrt(max(ev_old) / max(ev_new))
                         #       EHMC_args_as_Rcpp_List$eps_main <- max(0.01, min(max_eps_main,
                         #                                                        EHMC_args_as_Rcpp_List$eps_main * scale_fac))
                         #       EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- EHMC_args_as_Rcpp_List$eps_main
                         #       EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
                         #       EHMC_args_as_Rcpp_List$tau_main <- min(EHMC_args_as_Rcpp_List$tau_main,
                         #                                              40 * EHMC_args_as_Rcpp_List$eps_main)
                         #       message(sprintf("ii = %d: eps rescaled to %.4f (factor %.2f), tau capped to %.3f",
                         #                       ii, EHMC_args_as_Rcpp_List$eps_main, scale_fac, EHMC_args_as_Rcpp_List$tau_main))
                         #       
                         #       win_sum_x[] <- 0.0; win_sum_xxT[] <- 0.0; win_n <- 0L
                         #       win_prev_X <- NULL; win_rho_sum <- 0.0; win_rho_n <- 0L
                         #     }
                         #     
                         #   }, error = function(e) message("PHASE-2 METRIC FAILED at ii = ", ii, ": ", conditionMessage(e)))
                         # }
   
                         ## ---- Per-iteration cross-chain covariance (the correct estimator) ----
                         # try({
                         #   # if (ii > clip_iter) {   ## skip the earliest transient
                         #     if (ii > cov_start_iter) {
                         #       cov_ii <- cov(t(theta_main_vectors_all_chains_input_from_R))
                         #       cov_ii <- cov_ii + 1e-3 * mean(diag(cov_ii)) * diag(n_params_main)   # rank floor
                         #     # # cov_ii <- cov(t(theta_main_vectors_all_chains_input_from_R))   ## K draws -> p x p
                         #     # if (is.null(cov_crosschain_ema)) {
                         #     #   cov_crosschain_ema <- cov_ii
                         #     # } else {
                         #     #   cov_crosschain_ema <- ema_alpha * cov_ii + (1 - ema_alpha) * cov_crosschain_ema
                         #     # }
                         #   }
                         # })
   
                     
                     # tryCatch({
                     #   if (ii > cov_start_iter) {
                     #     cov_ii <- cov(t(theta_main_vectors_all_chains_input_from_R))
                     #     cov_ii <- cov_ii + 1e-3 * mean(diag(cov_ii)) * diag(nrow(cov_ii))
                     #     if (is.null(cov_crosschain_ema)) {
                     #       cov_crosschain_ema <- cov_ii
                     #     } else {
                     #       cov_crosschain_ema <- ema_alpha * cov_ii + (1 - ema_alpha) * cov_crosschain_ema
                     #     }
                     #   }
                     # }, error = function(e) message("COV ACCUM FAILED at ii=", ii, ": ", conditionMessage(e)))
                       
                       # ## ---- Windowed metric: accumulate + assign at window ends ----
                       # if (metric_type_main == "Empirical" && metric_shape_main == "dense") {
                       #   tryCatch({
                       #     if (ii > win_init_end && ii <= win_slow_end) {
                       #       Xc <- theta_main_vectors_all_chains_input_from_R          # p x K
                       #       win_sum_x   <- win_sum_x   + rowSums(Xc)
                       #       win_sum_xxT <- win_sum_xxT + tcrossprod(Xc)
                       #       win_n       <- win_n + ncol(Xc)
                       #     }
                       #     if ((ii %in% win_ends) && (win_n > 5L)) {
                       #       p      <- n_params_main
                       #       n_used <- win_n
                       #       mean_w <- win_sum_x / n_used
                       #       cov_w  <- (win_sum_xxT - n_used * tcrossprod(mean_w)) / (n_used - 1)
                       #       cov_w  <- 0.5 * (cov_w + t(cov_w))                                    # symmetrize
                       #       cov_w  <- (n_used / (n_used + 5)) * cov_w +
                       #         (5 / (n_used + 5) + 1e-4) * mean(diag(cov_w)) * diag(p)     # Stan reg + ridge floor
                       #       cov_w  <- BayesMVP:::Rcpp_near_PD(cov_w)
                       #       ## DIRECT assignment of the full consistent triple:
                       #       EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- cov_w
                       #       EHMC_Metric_as_Rcpp_List$M_dense_main          <- BayesMVP:::Rcpp_solve(cov_w)
                       #       EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- BayesMVP:::Rcpp_Chol(cov_w)
                       #       EHMC_burnin_as_Rcpp_List$M_dense_sqrt          <- pracma::sqrtm(EHMC_Metric_as_Rcpp_List$M_dense_main)[[1]]
                       #       ## diagonal mirrors:
                       #       EHMC_Metric_as_Rcpp_List$M_inv_main_vec  <- diag(cov_w)
                       #       EHMC_Metric_as_Rcpp_List$M_main_vec      <- 1 / diag(cov_w)
                       #       EHMC_burnin_as_Rcpp_List$sqrt_M_main_vec <- sqrt(EHMC_Metric_as_Rcpp_List$M_main_vec)
                       #       ## restart eps adaptation against the new metric (Stan-style):
                       #       EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
                       #       EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- EHMC_args_as_Rcpp_List$eps_main
                       #       ## reset window accumulators:
                       #       win_sum_x[]   <- 0.0
                       #       win_sum_xxT[] <- 0.0
                       #       win_n         <- 0L
                       #       message("ii = ", ii, ": metric DIRECTLY assigned from window (", n_used, " draws)")
                       #     }
                       #   }, error = function(e) message("WINDOWED METRIC FAILED at ii = ", ii, ": ", conditionMessage(e)))
                       # }
                       
                       # ## ---- Phase-2 accumulation + blended assignment ----
                       # if (metric_type_main == "Hessian" && metric_shape_main == "dense") {
                       #   tryCatch({
                       #     
                       #     ## snapshot the Hessian metric once, at handover:
                       #     if (ii == win_first_start) {
                       #       H_inv_seed <- EHMC_Metric_as_Rcpp_List$M_inv_dense_main
                       #       message("ii = ", ii, ": Hessian metric snapshotted as Phase-2 prior")
                       #     }
                       #     
                       #     ## accumulate:
                       #     if (ii > win_first_start && ii <= win_metric_end) {
                       #       Xc <- theta_main_vectors_all_chains_input_from_R           # p x K
                       #       win_sum_x   <- win_sum_x   + rowSums(Xc)
                       #       win_sum_xxT <- win_sum_xxT + tcrossprod(Xc)
                       #       win_n       <- win_n + ncol(Xc)
                       #       ## lag-1 autocorrelation, first 5 params, averaged over chains:
                       #       if (!is.null(win_prev_X)) {
                       #         for (jj in 1:min(5, n_params_main)) {
                       #           r1 <- suppressWarnings(cor(Xc[jj, ], win_prev_X[jj, ]))
                       #           if (is.finite(r1)) { win_rho_sum <- win_rho_sum + r1; win_rho_n <- win_rho_n + 1L }
                       #         }
                       #       }
                       #       win_prev_X <- Xc
                       #     }
                       #     
                       #     ## at window end: blend with Hessian prior, assign directly:
                       #     if ((ii %in% win_ends) && (win_n > 5L) && !is.null(H_inv_seed)) {
                       #       p      <- n_params_main
                       #       n_used <- win_n
                       #       mean_w <- win_sum_x / n_used
                       #       cov_w  <- (win_sum_xxT - n_used * tcrossprod(mean_w)) / (n_used - 1)
                       #       cov_w  <- 0.5 * (cov_w + t(cov_w))
                       #       ## effective sample size of the window:
                       #       rho_hat <- if (win_rho_n > 0) max(0, min(0.99, win_rho_sum / win_rho_n)) else 0.9
                       #       tau_hat <- (1 + rho_hat) / (1 - rho_hat)
                       #       n_eff   <- n_used / tau_hat
                       #       w_blend <- n_eff / (n_eff + 5 * p)
                       #       ## shrinkage blend toward the Hessian prior:
                       #       cov_blend <- w_blend * cov_w + (1 - w_blend) * H_inv_seed
                       #       cov_blend <- cov_blend + 1e-4 * mean(diag(cov_blend)) * diag(p)
                       #       cov_blend <- BayesMVP:::Rcpp_near_PD(0.5 * (cov_blend + t(cov_blend)))
                       #       ## direct assignment of the consistent triple:
                       #       EHMC_Metric_as_Rcpp_List$M_inv_dense_main      <- cov_blend
                       #       EHMC_Metric_as_Rcpp_List$M_dense_main          <- BayesMVP:::Rcpp_solve(cov_blend)
                       #       EHMC_Metric_as_Rcpp_List$M_inv_dense_main_chol <- BayesMVP:::Rcpp_Chol(cov_blend)
                       #       EHMC_burnin_as_Rcpp_List$M_dense_sqrt          <- pracma::sqrtm(EHMC_Metric_as_Rcpp_List$M_dense_main)[[1]]
                       #       EHMC_Metric_as_Rcpp_List$M_inv_main_vec  <- diag(cov_blend)
                       #       EHMC_Metric_as_Rcpp_List$M_main_vec      <- 1 / diag(cov_blend)
                       #       EHMC_burnin_as_Rcpp_List$sqrt_M_main_vec <- sqrt(EHMC_Metric_as_Rcpp_List$M_main_vec)
                       #       ## restart eps adaptation against the new metric:
                       #       EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
                       #       EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- EHMC_args_as_Rcpp_List$eps_main
                       #       ## reset window:
                       #       win_sum_x[] <- 0.0; win_sum_xxT[] <- 0.0; win_n <- 0L
                       #       win_prev_X <- NULL; win_rho_sum <- 0.0; win_rho_n <- 0L
                       #       message(sprintf("ii = %d: Phase-2 metric assigned (n=%d, n_eff~%.0f, w=%.2f, tau_hat~%.0f)",
                       #                       ii, n_used, n_eff, w_blend, tau_hat))
                       #     }
                       #     
                       #   }, error = function(e) message("PHASE-2 METRIC FAILED at ii = ", ii, ": ", conditionMessage(e)))
                       # }
   
                        if (!isTRUE(manual_tau) && ii > clip_iter && ii < n_adapt && sum(div_main) > 1) {
                             warning(sprintf("Too many divergences at iter %d, reducing tau", ii))
                             ##
                             EHMC_args_as_Rcpp_List$tau_main <- 0.95 * EHMC_args_as_Rcpp_List$tau_main
                             EHMC_args_as_Rcpp_List$tau_us   <- 0.95 * EHMC_args_as_Rcpp_List$tau_us
                        }
   
                        # if (ii == cov_start_iter) {
                        #      cat("cov accumulation starts; ensemble spread (max sd):",
                        #      max(apply(theta_main_vectors_all_chains_input_from_R, 1, sd)), "\n")
                        # }
  
                        p_jump_main <- mean(p_jump_per_chain, na.rm = TRUE)
                        if (sample_nuisance == TRUE) {
                           p_jump_us <- mean(p_jump_us_per_chain, na.rm = TRUE)
                        }
                         # p_jump_main <- median(p_jump_per_chain, na.rm = TRUE)
                         # if (sample_nuisance == TRUE) { 
                         #   p_jump_us <- median(p_jump_us_per_chain, na.rm = TRUE)
                         # }

                        if (ii < n_adapt) {
    
                            ## //////////////////   --------------------------------  update eps (step-size) for main ----------------------------------------------------------
                                            adapt_eps_outs <-  BayesMVP:::fn_Rcpp_wrapper_adapt_eps_ADAM( EHMC_args_as_Rcpp_List$eps_main,
                                                                                               EHMC_burnin_as_Rcpp_List$eps_m_adam_main,
                                                                                               EHMC_burnin_as_Rcpp_List$eps_v_adam_main,
                                                                                               ii, 
                                                                                               n_adapt,
                                                                                               EHMC_burnin_as_Rcpp_List$LR_main,
                                                                                               p_jump_main, 
                                                                                               EHMC_burnin_as_Rcpp_List$adapt_delta_main,
                                                                                               beta1_adam,
                                                                                               beta2_adam, 
                                                                                               eps_adam)
    
                                            EHMC_args_as_Rcpp_List$eps_main <-    min(max_eps_main, adapt_eps_outs[1])   ;  EHMC_args_as_Rcpp_List$eps_main
                                            EHMC_burnin_as_Rcpp_List$eps_m_adam_main <-  adapt_eps_outs[2]   ; EHMC_burnin_as_Rcpp_List$eps_m_adam_main
                                            EHMC_burnin_as_Rcpp_List$eps_v_adam_main <-  adapt_eps_outs[3]   ;    EHMC_burnin_as_Rcpp_List$eps_v_adam_main
    
                            ## //////////////////   --------------------------------  update eps (step-size) for nuisance ------------------------------------------------------
                                            if ((partitioned_HMC == TRUE) && (sample_nuisance == TRUE)) {
                                                    adapt_eps_outs <-  BayesMVP:::fn_Rcpp_wrapper_adapt_eps_ADAM(  EHMC_args_as_Rcpp_List$eps_us,
                                                                                                        EHMC_burnin_as_Rcpp_List$eps_m_adam_us,
                                                                                                        EHMC_burnin_as_Rcpp_List$eps_v_adam_us ,
                                                                                                        ii, 
                                                                                                        n_adapt,
                                                                                                        EHMC_burnin_as_Rcpp_List$LR_us,
                                                                                                        p_jump_us, 
                                                                                                        EHMC_burnin_as_Rcpp_List$adapt_delta_us,
                                                                                                        beta1_adam, 
                                                                                                        beta2_adam, 
                                                                                                        eps_adam)
            
                                                    EHMC_args_as_Rcpp_List$eps_us <-          min(max_eps_us,   adapt_eps_outs[1] )
                                                    EHMC_burnin_as_Rcpp_List$eps_m_adam_us  <- adapt_eps_outs[2]
                                                    EHMC_burnin_as_Rcpp_List$eps_v_adam_us <- adapt_eps_outs[3]
                                            }

                        }

                        ## //////////////////   --------------------------------  eps warm start (eps_initial) ------------------------------------------------------------
                        ##
                        ## eps set at the END of iteration ii is what iteration ii + 1 runs with, so:
                        ##   ii <  eps_initial_iter -> iteration ii + 1 is still inside the window: re-impose
                        ##                             eps_initial, discarding what ADAM just proposed;
                        ##   ii == eps_initial_iter -> the window ends here, so hand eps back to the value it
                        ##                             would have started at and reset the ADAM moments to it,
                        ##                             so the warm-start value is not carried forward.
                        ## Iterations 1 .. eps_initial_iter therefore run at eps_initial (it is also set before
                        ## the loop, which is what iteration 1 uses).
                        ##
                        if (use_eps_warm_start) {

                              if (ii < eps_initial_iter) {

                                    EHMC_args_as_Rcpp_List$eps_main <- eps_initial_main
                                    EHMC_args_as_Rcpp_List$eps_us   <- eps_initial_us
                                    ##
                                    EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- eps_initial_main
                                    EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
                                    EHMC_burnin_as_Rcpp_List$eps_m_adam_us   <- eps_initial_us
                                    EHMC_burnin_as_Rcpp_List$eps_v_adam_us   <- 0

                              } else if (ii == eps_initial_iter) {

                                    EHMC_args_as_Rcpp_List$eps_main <- eps_main_after_warm_start
                                    EHMC_args_as_Rcpp_List$eps_us   <- eps_us_after_warm_start
                                    ##
                                    EHMC_burnin_as_Rcpp_List$eps_m_adam_main <- eps_main_after_warm_start
                                    EHMC_burnin_as_Rcpp_List$eps_v_adam_main <- 0
                                    EHMC_burnin_as_Rcpp_List$eps_m_adam_us   <- eps_us_after_warm_start
                                    EHMC_burnin_as_Rcpp_List$eps_v_adam_us   <- 0
                                    ##
                                    message(paste0( "eps warm start finished at iteration ", ii,
                                                    ": eps reset to ", signif(eps_main_after_warm_start, 4),
                                                    " (main) and adapting normally from here."))

                              }

                        }

                        ## //////////////////   --------------------------------  update tau(s): ------------------------------------------------------------------------------
                        if (ii >= gap) {
                          tau_adaptation_iteration <- ii - gap + 1
                          if (ii < n_adapt && !isTRUE(manual_tau)) tau_adaptation_iteration_vec[ii] <- tau_adaptation_iteration
                          
                          if (partitioned_HMC == TRUE) {
                                          
                               ## //////////////////   --------------------------------  update tau for main --------------------------------------------------------------
                               if ( (ii < n_adapt) && (manual_tau == FALSE)) {
                                 
                                       noisy_grads_per_chain <- noisy_grads_prop_per_chain <- c()
                                       ## around the whole "update tau" try-block:
                                       t0 <- proc.time()[3]
                                       ##
                                       try({
                                             valid_chains <- 0
                                             for (kk in 1:n_chains_burnin) {
                                               
                                                 if (div_main[kk] == 0) {
                                                           
                                                       valid_chains <- valid_chains + 1
                                                       
                                                       # if (use_proposed) {
                                                       #   theta_to_use_kk <- theta_main_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                       #   v_to_use_kk     <- velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                       # } else { 
                                                       #   theta_to_use_kk <- theta_main_vectors_all_chains_input_from_R[, kk]
                                                       #   v_to_use_kk     <- velocity_main_vectors_all_chains_input_from_R[, kk]
                                                       # }
                                                       
                                                       if (isTRUE(tau_weight_by_p_jump)) {
                                                         main_theta_to_use_kk <- theta_main_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                         main_v_to_use_kk <- velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                         ##
                                                         us_theta_to_use_kk <- theta_us_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                         us_v_to_use_kk <- velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                       } else { 
                                                         main_theta_to_use_kk <- theta_main_vectors_all_chains_input_from_R[, kk]
                                                         main_v_to_use_kk     <- velocity_main_vectors_all_chains_input_from_R[, kk]
                                                         ##
                                                         us_theta_to_use_kk <- theta_us_vectors_all_chains_input_from_R[, kk]
                                                         us_v_to_use_kk <- velocity_us_vectors_all_chains_input_from_R[, kk]
                                                       }
                                                       
                                                       if (sample_nuisance == FALSE) { 
                                                         theta_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- rep(0.01, n_nuisance)
                                                         theta_us_prop_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- rep(0.01, n_nuisance)
                                                         ##
                                                         velocity_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- rep(0.01, n_nuisance)
                                                         velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- rep(0.01, n_nuisance)
                                                       } else { 
                                                         theta_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- theta_us_0_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                         theta_us_prop_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- us_theta_to_use_kk
                                                         ##
                                                         velocity_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- velocity_us_0_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                         velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- us_v_to_use_kk
                                                       }
                                                       
                                                       theta_vec_initial <- c(theta_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp,
                                                                              theta_main_0_burnin_tau_adapt_all_chains_input_from_R[, kk])
                                                       theta_vec_prop    <- c(theta_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp,  # same us: no displacement
                                                                              main_theta_to_use_kk)
                                                       
                                                       # M_inv_diag_all <- c(c(EHMC_Metric_as_Rcpp_List$M_inv_us_vec), c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec))
                                                       # sqrt_M_vec_all <-  sqrt(1/M_inv_diag_all)
                                                       # snaper_m_all <- c( EHMC_burnin_as_Rcpp_List$snaper_m_vec_us,  EHMC_burnin_as_Rcpp_List$snaper_m_vec_main)
                                                             
                                                       noisy_grads_prop_per_chain[kk] <- R_fn_compute_gradients_for_tau_using_KE(
                                                          velocity_initial = velocity_main_0_burnin_tau_adapt_all_chains_input_from_R[, kk],
                                                          velocity_proposed = main_v_to_use_kk,
                                                          metric_shape = metric_shape_main,
                                                          mass_matrix = if (metric_shape_main == "diag") 1 / EHMC_Metric_as_Rcpp_List$M_inv_main_vec else EHMC_Metric_as_Rcpp_List$M_dense_main,
                                                          kinetic_energy_rate_proposed = result$kinetic_energy_rate_main_proposed[kk],
                                                          tau_ii = tau_main_ii_vec[kk])
                                                       
                                                       
                                                 }
                                             }
                                             ##
                                             out_vec <- R_fn_update_tau_using_ADAM(      debug =  FALSE,
                                                                                         n_chains = n_chains,
                                                                                         noisy_grads_prop_per_chain = noisy_grads_prop_per_chain,
                                                                                         valid_chains = valid_chains,
                                                                                         ##
                                                                                         tau = EHMC_args_as_Rcpp_List$tau_main,
                                                                                         LR = EHMC_burnin_as_Rcpp_List$LR_main,
                                                                                         ii = tau_adaptation_iteration,
                                                                                         n_burnin = n_tau_adaptation_iterations,
                                                                                         ##
                                                                                         tau_m_adam = EHMC_burnin_as_Rcpp_List$tau_m_adam_main,
                                                                                         tau_v_adam = EHMC_burnin_as_Rcpp_List$tau_v_adam_main,
                                                                                         ##
                                                                                         beta1_adam = beta1_adam,
                                                                                         beta2_adam = beta2_adam,
                                                                                         eps_adam = eps_adam,
                                                                                         weights_per_chain = if (isTRUE(tau_weight_by_p_jump)) p_jump_per_chain else NULL,
                                                                                         aggregation = "weighted_mean")
                                             EHMC_args_as_Rcpp_List$tau_main <-  out_vec[1]
                                             EHMC_burnin_as_Rcpp_List$tau_m_adam_main <- out_vec[2]
                                             EHMC_burnin_as_Rcpp_List$tau_v_adam_main <- out_vec[3]
                                       })
                                       t_tau <- t_tau + (proc.time()[3] - t0)
                                       
                                       L_main_iter_ii <-  EHMC_args_as_Rcpp_List$tau_main /  EHMC_args_as_Rcpp_List$eps_main 
                                       L_main_during_burnin_vec[ii] <- L_main_iter_ii
                                             
                                 
                               }                                             # }
                
                               #### //////////////////   --------------------------------  update tau for us ------------------------------------------------------------------------------
                               if  ((ii < n_adapt) && (sample_nuisance == TRUE) && (manual_tau == FALSE)) {
                                             
                                        noisy_grads_per_chain <- noisy_grads_prop_per_chain <- c()
        
                                        t0 <- proc.time()[3]
                                        try({
                                           valid_chains <- 0
                                           for (kk in 1:n_chains_burnin) {
                                             
                                                 if (div_us[kk] == 0) {
                                                   
                                                     valid_chains <- valid_chains + 1
                                                     
                                                     if (isTRUE(tau_weight_by_p_jump)) {
                                                           theta_to_use_kk <- theta_us_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                           v_to_use_kk <- velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                     } else { 
                                                           theta_to_use_kk <- theta_us_vectors_all_chains_input_from_R[, kk]
                                                           v_to_use_kk <- velocity_us_vectors_all_chains_input_from_R[, kk]
                                                     }
                                                     
                                                     if (isTRUE(tau_weight_by_p_jump)) {
                                                       main_theta_to_use_kk <- theta_main_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                       main_v_to_use_kk <- velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                       ##
                                                       us_theta_to_use_kk <- theta_us_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                       us_v_to_use_kk <- velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                     } else { 
                                                       main_theta_to_use_kk <- theta_main_vectors_all_chains_input_from_R[, kk]
                                                       main_v_to_use_kk     <- velocity_main_vectors_all_chains_input_from_R[, kk]
                                                       ##
                                                       us_theta_to_use_kk <- theta_us_vectors_all_chains_input_from_R[, kk]
                                                       us_v_to_use_kk <- velocity_us_vectors_all_chains_input_from_R[, kk]
                                                     }
                                                     
                                                     if (sample_nuisance == FALSE) { 
                                                       theta_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- rep(0.01, n_nuisance)
                                                       theta_us_prop_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- rep(0.01, n_nuisance)
                                                       ##
                                                       velocity_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- rep(0.01, n_nuisance)
                                                       velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- rep(0.01, n_nuisance)
                                                     } else { 
                                                       theta_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- theta_us_0_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                       theta_us_prop_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- us_theta_to_use_kk
                                                       ##
                                                       velocity_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- velocity_us_0_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                       velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- us_v_to_use_kk
                                                     }
                                                     
                                                     theta_vec_initial <- c(theta_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp,
                                                                            theta_main_0_burnin_tau_adapt_all_chains_input_from_R[, kk])
                                                     theta_vec_prop    <- c(theta_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp,  # same us: no displacement
                                                                            main_theta_to_use_kk)
              
                                                     
                                                     noisy_grads_prop_per_chain[kk] <- R_fn_compute_gradients_for_tau_using_KE(
                                                          velocity_initial = velocity_us_0_burnin_tau_adapt_all_chains_input_from_R[, kk],
                                                          velocity_proposed = us_v_to_use_kk,
                                                          metric_shape = "diag",
                                                          mass_matrix = EHMC_Metric_as_Rcpp_List$M_us_vec,
                                                          kinetic_energy_rate_proposed = result$kinetic_energy_rate_us_proposed[kk],
                                                          tau_ii = tau_us_ii_vec[kk])
                                               }
          
                                           }
                                           ##
                                           out_vec <- R_fn_update_tau_using_ADAM(    debug =  FALSE,
                                                                                     n_chains = n_chains,
                                                                                     noisy_grads_prop_per_chain = noisy_grads_prop_per_chain,
                                                                                     valid_chains = valid_chains,
                                                                                     ##
                                                                                     tau = EHMC_args_as_Rcpp_List$tau_us,
                                                                                     LR = EHMC_burnin_as_Rcpp_List$LR_us,
                                                                                     ii = tau_adaptation_iteration,
                                                                                     n_burnin = n_tau_adaptation_iterations,
                                                                                     ##
                                                                                     tau_m_adam = EHMC_burnin_as_Rcpp_List$tau_m_adam_us,
                                                                                     tau_v_adam = EHMC_burnin_as_Rcpp_List$tau_v_adam_us,
                                                                                     ##
                                                                                     beta1_adam = beta1_adam,
                                                                                     beta2_adam = beta2_adam,
                                                                                     eps_adam = eps_adam,
                                                                                     weights_per_chain = if (isTRUE(tau_weight_by_p_jump)) p_jump_us_per_chain else NULL,
                                                                                     aggregation = "weighted_mean")
                                           EHMC_args_as_Rcpp_List$tau_us <-  out_vec[1]
                                           EHMC_burnin_as_Rcpp_List$tau_m_adam_us <- out_vec[2]  
                                           EHMC_burnin_as_Rcpp_List$tau_v_adam_us <-  out_vec[3] 
                                          })
                                          t_tau <- t_tau + (proc.time()[3] - t0)
                                        
                                      L_us_iter_ii <-  EHMC_args_as_Rcpp_List$tau_us /  EHMC_args_as_Rcpp_List$eps_us 
                                      L_us_during_burnin_vec[ii] <- L_us_iter_ii
                                            
                               }
                             
                           } else if (partitioned_HMC == FALSE) { 
                             
                                   noisy_grads_per_chain <- noisy_grads_prop_per_chain <- c()
                                   if (identical(tau_objective, "ChEES_per_tau")) {
                                       noisy_grads_prop_per_chain <- rep(x = NA_real_, times = n_chains_burnin)
                                       ChEES_criteria_per_chain <- rep(x = NA_real_, times = n_chains_burnin)
                                   }
                                  
                                  if ( (ii < n_adapt) && (manual_tau == FALSE)) {
                                 
                                    M_inv_diag_all <- c(c(EHMC_Metric_as_Rcpp_List$M_inv_us_vec), c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec))
                                    sqrt_M_vec_all <-  sqrt(1/M_inv_diag_all)
                                    snaper_m_all <- c( EHMC_burnin_as_Rcpp_List$snaper_m_vec_us,  EHMC_burnin_as_Rcpp_List$snaper_m_vec_main)

                                    ## ---- M for the main block, in the form the ChEES objective wants (M, not M_inv).
                                    ## Built ONCE per iteration rather than once per chain, since for a dense metric this
                                    ## is a 185 x 185 matrix that every chain reuses unchanged.
                                    if (tau_objective %in% c("ChEES", "ChEES_per_tau")) {
                                          if (metric_shape_main == "diag") {
                                                M_main_for_ChEES <- c(EHMC_Metric_as_Rcpp_List$M_main_vec)
                                                if (length(M_main_for_ChEES) != n_params_main || any(!is.finite(M_main_for_ChEES))) {
                                                      M_main_for_ChEES <- 1.0 / c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec)
                                                }
                                          } else {
                                                M_main_for_ChEES <- EHMC_Metric_as_Rcpp_List$M_dense_main
                                                if (is.null(M_main_for_ChEES) || any(!is.finite(M_main_for_ChEES))) {
                                                      M_main_for_ChEES <- solve(EHMC_Metric_as_Rcpp_List$M_inv_dense_main)
                                                }
                                          }
                                    }

                                    t0 <- proc.time()[3]
                                    try({
                                      valid_chains <- 0
                                      for (kk in 1:n_chains_burnin) {
                                        
                                          if (div_main[kk] == 0) {
                                            
                                                  valid_chains <- valid_chains + 1
                                                  
                                                  ## Either weight PROPOSALS by alpha, or use the realised accept/reject
                                                  ## endpoints. Combining alpha with accepted states would weight twice.
                                                  if (isTRUE(tau_weight_by_p_jump)) {
                                                      main_theta_to_use_kk <- theta_main_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                      main_v_to_use_kk <- velocity_main_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                      ##
                                                      us_theta_to_use_kk <- theta_us_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                      us_v_to_use_kk <- velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                  } else { 
                                                      main_theta_to_use_kk <- theta_main_vectors_all_chains_input_from_R[, kk]
                                                      main_v_to_use_kk     <- velocity_main_vectors_all_chains_input_from_R[, kk]
                                                      ##
                                                      us_theta_to_use_kk <- theta_us_vectors_all_chains_input_from_R[, kk]
                                                      us_v_to_use_kk <- velocity_us_vectors_all_chains_input_from_R[, kk]
                                                  }
                                                  
                                                  if (sample_nuisance == FALSE) { 
                                                      theta_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- rep(0.01, n_nuisance)
                                                      theta_us_prop_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- rep(0.01, n_nuisance)
                                                      ##
                                                      velocity_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- rep(0.01, n_nuisance)
                                                      velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- rep(0.01, n_nuisance)
                                                  } else { 
                                                      theta_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- theta_us_0_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                      theta_us_prop_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- us_theta_to_use_kk
                                                      ##
                                                      velocity_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- velocity_us_0_burnin_tau_adapt_all_chains_input_from_R[, kk]
                                                      velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R_kk_temp <- us_v_to_use_kk
                                                  }
                                                  
                                                  theta_vec_initial <- c(theta_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp, 
                                                                         theta_main_0_burnin_tau_adapt_all_chains_input_from_R[, kk])
                                                  theta_vec_prop    <- c(theta_us_prop_burnin_tau_adapt_all_chains_input_from_R_kk_temp,
                                                                         main_theta_to_use_kk)
                                                  ##
                                                  velocity_prop     <- c(velocity_us_prop_burnin_tau_adapt_all_chains_input_from_R_kk_temp,
                                                                         main_v_to_use_kk)
                                                  velocity_0        <- c(velocity_us_0_burnin_tau_adapt_all_chains_input_from_R_kk_temp,
                                                                         velocity_main_0_burnin_tau_adapt_all_chains_input_from_R[, kk])

                                                  if (tau_objective %in% c("ChEES", "ChEES_per_tau")) {

                                                      ## Position-based ChEES on the MAIN block. main_theta_to_use_kk and
                                                      ## main_v_to_use_kk already carry the weighting choice: without
                                                      ## p_jump weighting they are the POST-Metropolis state, so a rejected
                                                      ## chain has theta_prop == theta_0 and contributes exactly zero - the
                                                      ## accept indicator, applied for free. tau_weight_by_p_jump replaces that
                                                      ## indicator with its conditional mean further down.
                                                      ChEES_position_result <- R_fn_compute_gradients_for_tau_using_ChEES_position(
                                                                                        debug = FALSE,
                                                                                        ##
                                                                                        theta_main_0       = theta_main_0_burnin_tau_adapt_all_chains_input_from_R[, kk],
                                                                                        theta_main_prop    = main_theta_to_use_kk,
                                                                                        velocity_main_prop = main_v_to_use_kk,
                                                                                        ##
                                                                                        snaper_m_vec_main = EHMC_burnin_as_Rcpp_List$snaper_m_vec_main,
                                                                                        ##
                                                                                        metric_shape_main = metric_shape_main,
                                                                                        M_main_for_ChEES  = M_main_for_ChEES,
                                                                                        ##
                                                                                        ii = ii,
                                                                                        tau_ii = tau_main_ii_vec[kk],
                                                                                        return_criterion = identical(tau_objective, "ChEES_per_tau"))
                                                      if (identical(tau_objective, "ChEES_per_tau")) {
                                                          noisy_grads_prop_per_chain[kk] <- ChEES_position_result$gradient
                                                          ChEES_criteria_per_chain[kk] <- ChEES_position_result$criterion
                                                      } else {
                                                          noisy_grads_prop_per_chain[kk] <- ChEES_position_result
                                                      }

                                                  } else if (diffusion_HMC == FALSE) {
                                                    
                                                      noisy_grads_prop_per_chain[kk] <- R_fn_compute_gradients_for_tau_using_KE(
                                                          velocity_initial = velocity_main_0_burnin_tau_adapt_all_chains_input_from_R[, kk],
                                                          velocity_proposed = main_v_to_use_kk,
                                                          metric_shape = metric_shape_main,
                                                          mass_matrix = if (metric_shape_main == "diag") 1 / EHMC_Metric_as_Rcpp_List$M_inv_main_vec else EHMC_Metric_as_Rcpp_List$M_dense_main,
                                                          kinetic_energy_rate_proposed = result$kinetic_energy_rate_main_proposed[kk],
                                                          tau_ii = tau_main_ii_vec[kk])
                                                      
                                                  } else {
                                                    
                                                      noisy_grads_prop_per_chain[kk] <- R_fn_compute_gradients_for_tau_using_KE(
                                                          velocity_initial = velocity_main_0_burnin_tau_adapt_all_chains_input_from_R[, kk],
                                                          velocity_proposed = main_v_to_use_kk,
                                                          metric_shape = metric_shape_main,
                                                          mass_matrix = if (metric_shape_main == "diag") 1 / EHMC_Metric_as_Rcpp_List$M_inv_main_vec else EHMC_Metric_as_Rcpp_List$M_dense_main,
                                                          kinetic_energy_rate_proposed = result$kinetic_energy_rate_main_proposed[kk],
                                                          tau_ii = tau_main_ii_vec[kk])
                                                      
                                                  }
                                                  
                                                  
                                          }
                                        
                                      }
                                      ##
                                      ## Cost-aware arm normalises the MEAN gradient before the existing ADAM update.
                                      if (identical(tau_objective, "ChEES_per_tau")) {
                                          cost_aware_tau_update <- fn_normalise_ChEES_per_tau(
                                              gradients = noisy_grads_prop_per_chain,
                                              criteria = ChEES_criteria_per_chain,
                                              criterion_ema = ChEES_criterion_ema,
                                              weights = if (isTRUE(tau_weight_by_p_jump)) p_jump_per_chain else NULL)
                                          ChEES_criterion_ema <- cost_aware_tau_update$criterion_ema
                                          ChEES_criterion_ema_vec[ii] <- ChEES_criterion_ema
                                          ChEES_per_tau_gradient_vec[ii] <- cost_aware_tau_update$gradient
                                          noisy_grads_prop_per_chain <- cost_aware_tau_update$gradient
                                      }
                                      if (identical(tau_objective, "ChEES_per_tau") && !cost_aware_tau_update$informative) {
                                          out_vec <- c(EHMC_args_as_Rcpp_List$tau_main,
                                                       EHMC_burnin_as_Rcpp_List$tau_m_adam_main,
                                                       EHMC_burnin_as_Rcpp_List$tau_v_adam_main)
                                      } else {
                                      out_vec <- R_fn_update_tau_using_ADAM(     
                                                                                 debug =  FALSE,
                                                                                 n_chains = n_chains,
                                                                                 noisy_grads_prop_per_chain = noisy_grads_prop_per_chain,
                                                                                 valid_chains = valid_chains,
                                                                                 ##
                                                                                 tau = EHMC_args_as_Rcpp_List$tau_main,
                                                                                 LR  = EHMC_burnin_as_Rcpp_List$LR_main,
                                                                                 ii = tau_adaptation_iteration,
                                                                                 n_burnin = n_tau_adaptation_iterations,
                                                                                 ##
                                                                                 tau_m_adam = EHMC_burnin_as_Rcpp_List$tau_m_adam_main,
                                                                                 tau_v_adam = EHMC_burnin_as_Rcpp_List$tau_v_adam_main,
                                                                                 ##
                                                                                 beta1_adam = beta1_adam,
                                                                                 beta2_adam = beta2_adam,
                                                                                 eps_adam = eps_adam,
                                                                                 ##
                                                                                 ## Every corrected objective estimates an expectation, hence a mean.
                                                                                 weights_per_chain = if (isTRUE(tau_weight_by_p_jump) && !identical(tau_objective, "ChEES_per_tau")) p_jump_per_chain else NULL,
                                                                                 aggregation = "weighted_mean")
                                      }
                                      ##
                                      EHMC_args_as_Rcpp_List$tau_main <-  out_vec[1]
                                      # EHMC_args_as_Rcpp_List$tau_main <- out_vec[1]
                                      EHMC_burnin_as_Rcpp_List$tau_m_adam_main <- out_vec[2] # ifelse( is.nan( tau_ADAM_main_avg[[2]]), EHMC_burnin_as_Rcpp_List$tau_m_adam_main, tau_ADAM_main_avg[[2]])
                                      EHMC_burnin_as_Rcpp_List$tau_v_adam_main <-  out_vec[3] # ifelse( is.nan( tau_ADAM_main_avg[[3]]), EHMC_burnin_as_Rcpp_List$tau_v_adam_main, tau_ADAM_main_avg[[3]])
                                      
                                    })
                                    t_tau <- t_tau + (proc.time()[3] - t0)
                                    
                                    
                                    if (debug) {
                                       cat("Tau updated by ADAM to:", EHMC_args_as_Rcpp_List$tau_main, "\n")
                                    }
                                    
                                    L_main_iter_ii <-  EHMC_args_as_Rcpp_List$tau_main /  EHMC_args_as_Rcpp_List$eps_main 
                                    L_main_during_burnin_vec[ii] <- L_main_iter_ii
                                    
                                  }
                          
                           }
                        }
                        
                            ## Apply user ceilings immediately, including the last update before adaptation freezes.
                            if (!isTRUE(manual_tau) && ii >= gap && ii < n_adapt) {
                                EHMC_args_as_Rcpp_List$tau_main <- min(max_tau_main, EHMC_args_as_Rcpp_List$tau_main)
                                EHMC_args_as_Rcpp_List$tau_us <- if (isTRUE(partitioned_HMC))
                                    min(max_tau_us, EHMC_args_as_Rcpp_List$tau_us) else EHMC_args_as_Rcpp_List$tau_main
                            }
                            ## //////////////////   --------------------------------  update tau for ALL PARAMS ------------------------------------------------------------------------------
                            ## n_refresh is the print INTERVAL, i.e. report every n_refresh
                            ## iterations. (This block used to overwrite the argument with a
                            ## hard-coded 20 and then divide n_burnin by it, so nothing the caller
                            ## passed had any effect.) max(1L, ...) guards ii %% 0 = NA:
                            n_refresh_steps <- max(1L, as.integer(round(n_refresh)))
                            ##
                            # print(n_burnin)
                            # print(n_refresh)
                            # print(ii)
                            # print(p_jump_main)
                            # print(EHMC_args_as_Rcpp_List$eps_main)
                            # print(EHMC_args_as_Rcpp_List$tau_main)
                            # print(div_main)
                            ##
                            try({  
                            if (ii %% n_refresh_steps == 0) {
                              
                                if (n_nuisance > 0) {
                                  cat("max |theta_hat|:", max(abs(EHMC_Metric_as_Rcpp_List$theta_hat_us_vec)), "\n")
                                  cat("max M_i:", max(EHMC_Metric_as_Rcpp_List$M_us_vec), "\n")
                                  cat("min M_i:", min(EHMC_Metric_as_Rcpp_List$M_us_vec), "\n")
                                  cat("max/min M ratio:", max(EHMC_Metric_as_Rcpp_List$M_us_vec)/min(EHMC_Metric_as_Rcpp_List$M_us_vec), "\n")
                                }
                                # 
                                # ## --------------------------------
                                # var_draws_us <- apply(theta_us_vectors_all_chains_input_from_R, 1, var)   # across the 4 chains, this iteration
                                # cat("median var across chains:", median(var_draws_us), " vs 1/M_us:", 1/median(EHMC_Metric_as_Rcpp_List$M_us_vec), "\n")
                                # cat("ratio (should be ~1 if estimator is right):", median(var_draws_us) * median(EHMC_Metric_as_Rcpp_List$M_us_vec), "\n")
                                # ## --------------------------------

                              
                                        # print(paste("iter = ", ii))
                                        cat(colourise(    (paste("iter = ", ii))          , "purple"), "\n")
      
                                        if (partitioned_HMC == TRUE) { # if NOT sampling all parameters at once
                                                  cat(colourise(    (paste("p_jump_main = ", round(p_jump_main, 3)))          , "green"), "\n")
                                                  cat(colourise(    (paste("eps_main = ", round(EHMC_args_as_Rcpp_List$eps_main, 3)))          , "blue"), "\n")
                                                  ##
                                                  message((paste("tau_main = ",  round(EHMC_args_as_Rcpp_List$tau_main, 3))))#
                                                  ##
                                                  message((paste("L_main = ",  round(ceiling(EHMC_args_as_Rcpp_List$tau_main/EHMC_args_as_Rcpp_List$eps_main), 0))))
                                                  cat(colourise(    (paste("div_main = ", sum(div_main)))          , "red"), "\n")
                                        } else {
                                                  cat(colourise(    (paste("p_jump = ", round(p_jump_main, 3)))          , "green"), "\n")
                                                  cat(colourise(    (paste("eps = ", round(EHMC_args_as_Rcpp_List$eps_main, 3)))          , "blue"), "\n")
                                                  ##
                                                  message((paste("tau = ",  signif(EHMC_args_as_Rcpp_List$tau_main, 3))))
                                                  ##
                                                  message((paste("L = ",  round(ceiling(EHMC_args_as_Rcpp_List$tau_main/EHMC_args_as_Rcpp_List$eps_main), 0))))
                                                  cat(colourise(    (paste("div = ", sum(div_main)))          , "red"), "\n")
                                        }


                                        if (partitioned_HMC == TRUE) { # if NOT sampling all parameters at once
                                            if     (sample_nuisance == TRUE)   {
                                                cat(colourise(    (paste("p_jump_us = ", round(p_jump_us, 3)))          , "green"), "\n")
                                                cat(colourise(    (paste("eps_us = ", round(EHMC_args_as_Rcpp_List$eps_us, 3)))          , "blue"), "\n")
                                                ##
                                                message((paste("tau_us = ",  round(EHMC_args_as_Rcpp_List$tau_us, 3))))
                                                ##
                                                message((paste("L_us = ",  round(ceiling(EHMC_args_as_Rcpp_List$tau_us / EHMC_args_as_Rcpp_List$eps_us), 0))))
                                                cat(colourise(    (paste("div_us = ", sum(div_us)))          , "red"), "\n")
                                            }
                                        }
                                          
                                        # print(head(c(EHMC_Metric_as_Rcpp_List$M_inv_main_vec)))
                                      
                            }
                            })
                                    
                            if (ii == last_burnin_iteration) {

                                    try({
                                        burnin_toc_result <- tictoc::toc(log = TRUE)
                                        print(burnin_toc_result)
                                        tictoc::tic.clearlog()
                                        ## elapsed seconds straight from toc()'s numbers. Parsing the "X sec elapsed" text with "\\d+\\.\\d+"
                                        ## gave NA whenever the time rounded to a whole second (tictoc then prints e.g. "4 sec elapsed"):
                                        time_burnin <- as.numeric(burnin_toc_result$toc - burnin_toc_result$tic)
                                    })
                              
                            } 
                        
                        L_main_iter_ii <-  EHMC_args_as_Rcpp_List$tau_main /  EHMC_args_as_Rcpp_List$eps_main 
                        L_main_during_burnin_vec[ii] <- L_main_iter_ii
                        tau_main_during_burnin_vec[ii] <- EHMC_args_as_Rcpp_List$tau_main
                        eps_main_during_burnin_vec[ii] <- EHMC_args_as_Rcpp_List$eps_main
                        if (isTRUE(partitioned_HMC) && isTRUE(sample_nuisance)) {
                            tau_us_during_burnin_vec[ii] <- EHMC_args_as_Rcpp_List$tau_us
                            eps_us_during_burnin_vec[ii] <- EHMC_args_as_Rcpp_List$eps_us
                        }


    }
    ##
    stage_wall <- time_burnin                      # wall time of THIS stage only
    time_burnin <- time_burnin + time_pre_burnin   # cumulative (returned, leave as-is)
    ##
    L_main_during_burnin <- if (length(x = L_main_during_burnin_vec) > 0L) mean(L_main_during_burnin_vec, na.rm = TRUE) else NA_real_
    L_us_during_burnin <-   if (length(x = L_us_during_burnin_vec) > 0L) mean(L_us_during_burnin_vec, na.rm = TRUE) else NA_real_
    ##
    cat("stage wall time:        ", stage_wall, "\n")
    cat("time inside C++ call:   ", t_cpp_total, "\n")
    cat("time pushing state to C++:", t_push_total, "\n")
    cat("R adaptation + misc:    ", stage_wall - t_cpp_total - t_push_total, "\n")
    ##
    # time_burnin <- time_burnin + time_pre_burnin
    # ##
    # L_main_during_burnin <- mean(L_main_during_burnin_vec, na.rm = TRUE)
    # L_us_during_burnin <-   mean(L_us_during_burnin_vec, na.rm = TRUE)
    # ##
    # cat("total burnin wall time:", time_burnin, "\n")
    # cat("time inside C++ call:  ", t_cpp_total, "\n")
    # cat("R overhead + adaptation:", time_burnin - t_cpp_total, "\n")
    ##
    cat("time inside tau C++ call:  ", t_tau, "\n")
    if (debug_burnin_timing) {
        burnin_profile <- list(
            iterations = do.call(what = rbind, args = burnin_iteration_profiles),
            chains = do.call(what = rbind, args = burnin_chain_profiles),
            stage_totals = data.frame(stage_wall_seconds = stage_wall,
                                      cpp_call_seconds = t_cpp_total,
                                      push_seconds = t_push_total,
                                      R_adaptation_and_misc_seconds = stage_wall - t_cpp_total - t_push_total))
        burnin_profile$stage_totals$parallel_seconds <- sum(burnin_profile$iterations$parallel_seconds)
        burnin_profile$stage_totals$output_seconds <- sum(burnin_profile$iterations$output_seconds)
        burnin_profile$stage_totals$call_residual_seconds <- sum(burnin_profile$iterations$call_residual_seconds)
        cat("native parallel chains: ", sum(burnin_profile$iterations$parallel_seconds), "\n")
        cat("native output allocation/copy: ", sum(burnin_profile$iterations$output_seconds), "\n")
        ## The call residual includes diagnostic packaging and R/native boundary costs.
        ## It may be slightly negative because proc.time() has coarser resolution than steady_clock.
        ## Chain times overlap: do not sum them to estimate wall time. Step counts include an entered
        ## divergent step; for partitioned HMC they sum main and nuisance integrator steps.
    }
    ##
    # Rprof(NULL)
    # summaryRprof("burnin_prof.out")$by.self[1:15, ]
    ##
    rm(worker_ptr); gc()
    ##
    ## ---- debugging hook, INERT unless the environment variable is set: save the end-of-burn-in state and everything
    ##      needed to rebuild the persistent burn-in worker, so the C++ kernel can be replayed offline on it.
    ##      One file per burn-in (the pre-burnin and the main burn-in differ in n_burnin).
    dump_burnin_end_state_path <- Sys.getenv("BAYESMVP_DUMP_BURNIN_END_STATE_PATH")
    if (nzchar(dump_burnin_end_state_path)) {
          dump_burnin_end_state_file <- paste0(dump_burnin_end_state_path, "_nburnin", n_burnin, ".rds")
          saveRDS(list( y                                          = y,
                        Model_args_as_Rcpp_List                    = Model_args_as_Rcpp_List,
                        EHMC_args_as_Rcpp_List                     = EHMC_args_as_Rcpp_List,
                        EHMC_Metric_as_Rcpp_List                   = EHMC_Metric_as_Rcpp_List,
                        theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                        theta_us_vectors_all_chains_input_from_R   = theta_us_vectors_all_chains_input_from_R,
                        n_chains_burnin                            = n_chains_burnin,
                        partitioned_HMC                            = partitioned_HMC,
                        diffusion_HMC                              = diffusion_HMC,
                        Model_type                                 = Model_type,
                        sample_nuisance                            = sample_nuisance,
                        force_autodiff                             = force_autodiff,
                        force_PartialLog                           = force_PartialLog,
                        multi_attempts                             = multi_attempts,
                        n_threads_WCP                              = n_threads_WCP,
                        seed                                       = seed,
                        eps_main_during_burnin_vec                 = eps_main_during_burnin_vec),
                  file = dump_burnin_end_state_file)
          message("burn-in end state saved to ", dump_burnin_end_state_file)
          if (identical(Sys.getenv("BAYESMVP_STOP_AFTER_BURNIN_DUMP"), as.character(n_burnin))) {
                stop("BAYESMVP_STOP_AFTER_BURNIN_DUMP = ", n_burnin, ": stopping after the burn-in dump.")
          }
    }
    ##
    return(list(n_chains_burnin = n_chains_burnin,
                burnin_profile = burnin_profile,
                burnin_schedule = resolved_burnin_schedule,
                n_burnin = n_burnin,
                time_burnin = time_burnin,
                ##
                eps_main =  EHMC_args_as_Rcpp_List$eps_main,
                tau_main =  EHMC_args_as_Rcpp_List$tau_main,
                eps_us =  EHMC_args_as_Rcpp_List$eps_us,
                tau_us =  EHMC_args_as_Rcpp_List$tau_us,
                tau_main_during_burnin_vec = tau_main_during_burnin_vec,
                eps_main_during_burnin_vec = eps_main_during_burnin_vec,
                tau_us_during_burnin_vec = tau_us_during_burnin_vec,
                eps_us_during_burnin_vec = eps_us_during_burnin_vec,
                ChEES_criterion_ema_vec = ChEES_criterion_ema_vec,
                ChEES_per_tau_gradient_vec = ChEES_per_tau_gradient_vec,
                tau_adaptation_version = 2,
                tau_adaptation_iteration_vec = tau_adaptation_iteration_vec,
                L_main_during_burnin_vec = L_main_during_burnin_vec,
                L_us_during_burnin_vec = L_us_during_burnin_vec,
                L_main_during_burnin = L_main_during_burnin,
                L_us_during_burnin = L_us_during_burnin,
                ##
                Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
                EHMC_Metric_as_Rcpp_List = EHMC_Metric_as_Rcpp_List,
                EHMC_args_as_Rcpp_List = EHMC_args_as_Rcpp_List,
                EHMC_burnin_as_Rcpp_List = EHMC_burnin_as_Rcpp_List,
                ##
                theta_main_vectors_all_chains_input_from_R = theta_main_vectors_all_chains_input_from_R,
                ##
                theta_nuisance_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R,
                theta_us_vectors_all_chains_input_from_R = theta_us_vectors_all_chains_input_from_R))
    
  
  
}
  


              
   
