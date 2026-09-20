##
## -| ------------------ Bridge: APMS 2007 -> Stan data -> fit M3-M10 via BayesMVP/BridgeStan --------------------------------------------------
##
## This is the BayesMVP version of R_bridge_APMS2007_styled.R. The DATA-PREPARATION sections
## (archive reading, the four ordinal tests, missing-code handling, phase-one grouping, target
## weights, correlation bounds, covariates, priors and the stan_data_base list) are unchanged
## from that bridge. What differs is the FITTING path: the four-class LC-MVOP model is compiled
## and sampled through the UPDATED BayesMVP external-Stan (BridgeStan) interface, i.e. with
## BayesMVP's adaptive diffusion-space HMC rather than cmdstanr's NUTS.
##
## Key points:
##   * BayesMVP_AVX_fused uses the current four-class AVX model with fused normal calculations
##     and partially manual C++ gradients. The existing generator preserves the model and priors.
##   * BayesMVP treats the FIRST declaration of the Stan parameters block ("u_raw") as the
##     NUISANCE block; its unconstrained dimension is detected automatically. Users do not
##     count coordinates and do not rename anything.
##   * Sampler settings are ONE explicit named list (fn_sampler_settings_APMS_BayesMVP()).
##     cmdstanr NUTS settings are mapped to the closest BayesMVP quantities and are documented
##     in RUN_BayesMVP.md - they are NOT exact equivalents.
##   * The prior check has its OWN subset configuration (chains / warm-up / sampling / rows).
##     Posterior subsetting is controlled independently.
##
## COLUMN ORDER (fixed throughout the project):
##     1  R_A   ADOS-4 total (comrsitot2)                         condition A = autism
##     2  R_B   BPD criteria count at phase 2 (St2BDSc)           condition B = BPD
##     3  Y_A   AQ-20 (DVTotal)
##     4  Y_B   phase-1 borderline screen, nine criteria reconstructed from PD73-PD87
##
## NO COLLAPSING ANYWHERE. Category = raw score + 1, for every instrument. See
## R_bridge_APMS2007_styled.R for the full data documentation (this bridge reuses its logic).
##
suppressPackageStartupMessages({
    require(RcppParallel)  ## Load before BayesMVP, including when fitting one model serially.
    require(BayesMVP)
    require(dplyr)
})
options(scipen = 999999)
##
## ---------------------------------------------------------------------------------------------------------------------------------------------
## ---- 0. Paths and switches
## ---------------------------------------------------------------------------------------------------------------------------------------------
##
BASE_DIR <-  "/home/enzocerullo/Documents/Work/PhD_work/Autism_BPD_paper"
##
DTA_FILE <-  file.path( BASE_DIR, "apms07arch.dta")
##
## ---- EDIT HERE: posterior model, data and execution controls -----------------------------------------------------------------------------
## These are local, editable settings, copied from the current benchmark. This script does not run/source the benchmark driver.
##
BAYESMVP_CONFIGURATION <-  "BayesMVP_AVX_fused"
## Options: "BayesMVP_plain", "BayesMVP_AVX", "BayesMVP_AVX_fused", "BayesMVP_AVX_whole_vector".
##
MODEL_CONFIGS <-  c("M3_perfect_CI", "M4_perfect_dep", "M5_imperfect_bin", "M6_imperfect_ord")
MODELS_TO_FIT <-  "M6_imperfect_ord"  ## Select one or more names from MODEL_CONFIGS.
##
N_subset <-  5000         ## NULL = full APMS data; otherwise use the benchmark's stratified subset procedure.
subset_seed <-  123       ## Selects rows independently of the sampler seed.
SEED <-  123              ## One fit per selected model; matches the benchmark's first repetition seed.
##
RUN_PARALLEL <-  FALSE    ## FALSE = fit selected models sequentially; chains within each fit still run in parallel.
N_CORES <-  140           ## Existing capacity limit when RUN_PARALLEL = TRUE.
##
max_rhat_thr <-  1.10     ## Existing bridge reporting thresholds, independent of the sampler settings.
min_ESS_thr <-  100
##
if (!is.character(x = BAYESMVP_CONFIGURATION) || length(x = BAYESMVP_CONFIGURATION) != 1L ||
    is.na(x = BAYESMVP_CONFIGURATION) || !BAYESMVP_CONFIGURATION %in%
    c("BayesMVP_plain", "BayesMVP_AVX", "BayesMVP_AVX_fused", "BayesMVP_AVX_whole_vector")) {
    stop("Select one supported BAYESMVP_CONFIGURATION in the EDIT HERE section.")
}
##
if (!is.character(x = MODELS_TO_FIT) || !length(x = MODELS_TO_FIT) || anyNA(x = MODELS_TO_FIT) ||
    anyDuplicated(x = MODELS_TO_FIT) || !all(MODELS_TO_FIT %in% MODEL_CONFIGS)) {
    stop("MODELS_TO_FIT must select unique names from MODEL_CONFIGS.")
}
##
subset <-  !is.null(x = N_subset)  ## Derived only; edit N_subset above to change the data selection.
if (subset && (!is.numeric(x = N_subset) || length(x = N_subset) != 1L || !is.finite(x = N_subset) ||
               N_subset < 1 || N_subset != floor(x = N_subset) || N_subset > .Machine$integer.max)) {
    stop("N_subset must be NULL (full data) or one positive integer.")
}
if (subset && (!is.numeric(x = subset_seed) || length(x = subset_seed) != 1L || !is.finite(x = subset_seed) ||
               subset_seed < 0 || subset_seed != floor(x = subset_seed) || subset_seed > .Machine$integer.max)) {
    stop("subset_seed must be one non-negative integer no larger than .Machine$integer.max.")
}
##
STAN_FILE <-  file.path( BASE_DIR, "LC_MVOP_4class_joint_v1_reduce_sum.stan")
##
## ---- OPTIONAL: the BayesMVP AVX-512 / AVX2 special functions as Stan EXTERNAL functions.
## The _AVX model is generated from the base model by stan_ext/make_AVX_variant.py (identical
## model; the vectorised special functions inside partial_log_lik are the externals in
## stan_ext/BayesMVP_AVX_stan_fns.hpp). The externals are only reached on the VECTORISED
## likelihood path, i.e. Phi_type != 1; Phi_type = 1 (the scalar exact-interval routine)
## never calls them, so the switch also selects Phi_type (0 = exact Phi, vectorised).
##
USE_AVX_STAN_EXTERNALS <-  BAYESMVP_CONFIGURATION != "BayesMVP_plain"
##
STAN_FILE_AVX <-  file.path( BASE_DIR, "LC_MVOP_4class_joint_v1_reduce_sum_AVX.stan")
STAN_USER_HEADER_AVX <-  file.path( BASE_DIR, "stan_ext", "BayesMVP_AVX_stan_fns.hpp")
##
PHI_TYPE <-  0   ## 0 = exact Phi, VECTORISED likelihood path (the one to use); 2 = Phi_approx, vectorised; 1 = scalar exact-interval (not wanted)
##
if (USE_AVX_STAN_EXTERNALS) {
    if (!file.exists(STAN_FILE_AVX)) stop("USE_AVX_STAN_EXTERNALS = TRUE but the generated model is missing; run stan_ext/make_AVX_variant.py: ", STAN_FILE_AVX)
    if (!file.exists(STAN_USER_HEADER_AVX)) stop("USE_AVX_STAN_EXTERNALS = TRUE but the user header is missing: ", STAN_USER_HEADER_AVX)
    STAN_FILE <-  STAN_FILE_AVX
}
##
## ---- Reuse the EXISTING application data builders, prior calculations and model
## definitions (unchanged), plus the BayesMVP-specific helpers:
##
source(file = file.path( BASE_DIR, "R_fn_sim_4class_joint_LC_MVOP_data.R"), local = TRUE)
##
# source(file = file.path( BASE_DIR,
#                          "~/Documents/Work/PhD_work/R_packages/BayesMVP/APMS_bridge/R_fn_APMS_BayesMVP.R"))
source(file = "~/Documents/Work/PhD_work/R_packages/BayesMVP/APMS_bridge/R_fn_APMS_BayesMVP.R", local = TRUE)
##
## ---- EDIT HERE: sampler settings, matching R_benchmark_APMS_full_fits.R at N_subset = 5000 -------------------------------------------------
## Change the named arguments here; there are no later chunk/thread overrides. Other arguments retain the shared constructor defaults.
## Changing N_subset does not silently retune chunks or WCP threads. The sampler recalculates grainsizes for the chosen data size.
## summary_options controls post-processing only. The package derives the adaptation schedule from n_burnin.
##
sampler_settings <-  fn_sampler_settings_APMS_BayesMVP(
    seed                                    = SEED,
    ##
    n_burnin                                = 125,
    n_iter                                  = 125,
    ##
    n_chains_burnin                         = 4,
    n_threads_WCP_burnin                    = 32,
    num_chunks_burnin                       = 128,
    ##
    n_chains_sampling                       = 64,
    n_threads_WCP_sampling                  = 1,
    num_chunks_sampling                     = 16,
    ##
    learning_rate                           = 0.0625,
    learning_rate_initial                   = 0.10,
    metric_estimator                        = "chain_mean_scaled",
    adapt_delta                             = 0.80,
    ##
    tau_objective                           = "ChEES_per_tau",
    tau_weight_by_p_jump                    = TRUE,
    tau_ramp                                = "original",
    tau_initial                             = pi,
    manual_tau                              = FALSE,
    tau_if_manual                           = NULL,
    tau_if_manual_in_L_units                = FALSE,
    ##
    share_tau_ii_across_chains_in_burnin    = TRUE,
    burnin_TBB_pool_equals_n_chains         = FALSE,  ## Must stay FALSE for external Stan models, including AVX/fused.
    eps_reinit_at_ChEES_handover            = FALSE,
    theta_hat_us_rule                       = "running_mean_frozen",
    theta_hat_us_freeze_iter                = NULL,
    ##
    pre_burnin_n_iter                       = 50,
    pre_burnin_L                            = 10,
    debug_burnin_timing                     = TRUE,
    store_log_lik_trace                     = FALSE,
    summary_options = list(save_log_lik_trace = FALSE, compute_nested_rhat = FALSE))
fn_validate_settings_APMS_BayesMVP(settings = sampler_settings)
if (is.null(x = sampler_settings$num_chunks_burnin) || is.null(x = sampler_settings$num_chunks_sampling)) {
    stop("This bridge requires explicit num_chunks_burnin and num_chunks_sampling in fn_sampler_settings_APMS_BayesMVP().")
}
##
## timing records, the BayesMVP-vs-cmdstanr efficiency comparison, and the sim-study runtime estimator:
source(file = file.path( BASE_DIR, "R_fn_APMS_efficiency_and_runtime.R"), local = TRUE)
##
## NOTE: THREADS_PER_CHAIN / cpp_options below are the CmdStanR conventions; BayesMVP
## compiles via BridgeStan (stanc_args vs make_args - see RUN_BayesMVP.md). Within-chain
## threading in BayesMVP is selected separately by n_threads_WCP_burnin / n_threads_WCP_sampling;
## chain-level parallelism is n_chains_sampling.
##
{
    ESTIMAND_TARGET <-  "phase_one_weighted"
    PHASE_ONE_WEIGHT_VAR <-  "wt_ints1"  ## archive label: Weight to use with phase one data
}
##
## Unique output directory (the BayesMVP sampler writes its traces in RAM here; the C++ only
## uses the disk path if settings$use_disk = TRUE):
{
    dir.create( path = file.path( BASE_DIR, "BayesMVP_out"),
                recursive = TRUE,
                showWarnings = FALSE)
    ##
    RUN_DIR <-  tempfile( pattern = "apms07_BayesMVP_",
                          tmpdir = file.path( BASE_DIR, "BayesMVP_out"))
    dir.create( path = RUN_DIR,
                recursive = TRUE)
}
##
{
    CORR_FORCE_POSITIVE <-  FALSE
    ##
    n_tests <-  4
    n_class <-  4
    ##
    test_condition <-  c(1L, 2L, 1L, 2L)
    ##
    to3d <-  function( m,
                       k) {

        array( data = rep( x = m,
                           times = k),
               dim = c(nrow(x = m), ncol(x = m), k)) |> aperm(perm = c(3, 1, 2))

    }
}
##
## -| --------- 1. Read and build the four tests (UNCHANGED from R_bridge_APMS2007_styled.R) ---------------------------------------------------
##
{
    raw <-  foreign::read.dta( DTA_FILE, convert.factors = FALSE)
    ##
    v <-  function(nm) {

        column_index <-  which(x = tolower(x = names(x = raw)) == tolower(x = nm))
        ##
        if (length(x = column_index) != 1L) stop( "variable not found (or ambiguous) in the .dta: ", nm)
        ##
        raw[[column_index]]

    }
    ##
    N_all <-  nrow(x = raw)
    ##
    stopifnot(N_all == 7403L)
    ##
    pd_items <-  paste0( "PD",
                         73:87)
    ##
    stopifnot(all(pd_items %in% names(x = raw)))
    ##
    bpd_crit_items <-  list( "PD73",
                             "PD74",
                             c("PD75", "PD76", "PD77", "PD78"),
                             "PD79",
                             c("PD80", "PD81"),
                             "PD82",
                             "PD83",
                             c("PD84", "PD85", "PD86"),
                             "PD87")
    ##
    aq20 <-  as.numeric(x = raw$DVTotal)
    ##
    pd_mat <-  as.matrix(x = raw[, pd_items])
    pd_allmis <-  rowSums( x = pd_mat == 1L | pd_mat == 2L,
                           na.rm = TRUE) == 0L
    ##
    bpd_required <-  c(1L, 1L, 4L, 1L, 2L, 1L, 1L, 2L, 1L)
    ##
    bpd_scr <-  Reduce( f = `+`,
                        x = Map( f = function( items,
                                               required) {

                as.integer(x = rowSums( x = pd_mat[, items, drop = FALSE] == 1,
                                        na.rm = TRUE) >= required)

            },
                                 bpd_crit_items,
                                 bpd_required))
    ##
    bpd_scr[pd_allmis] <-  NA_real_
    ##
    ados <-  as.numeric(x = raw$comrsitot2)
    ##
    bpd_crit <-  as.numeric(x = raw$St2BDSc)
    ##
    obs_R_A <-  as.integer(x = raw$ADOSDone == 1 & ados >= 0)
    obs_R_B <-  as.integer(x = raw$SCIDDone == 1 & bpd_crit >= 0)
    obs_Y_A <-  as.integer(x = !is.na(x = aq20) & aq20 >= 0)
    obs_Y_B <-  as.integer(x = !is.na(x = bpd_scr))
    ##
    max_score <-  c(R_A = 24L, R_B = 9L, Y_A = 20L, Y_B = 9L)
    ##
    n_cat <-  max_score + 1L  ## 25, 10, 21, 10
    ##
    to_cat <-  function(x) as.integer(x = x + 1L)
    ##
    ref_thr_cat <-  c(R_A = 11L, R_B = 6L)
    ##
    scr_thr_cat <-  c(Y_A = 11L, Y_B = 6L)
    ##
    y <-  cbind( R_A = to_cat(x = ados),
                 R_B = to_cat(x = bpd_crit),
                 Y_A = to_cat(x = aq20),
                 Y_B = to_cat(x = bpd_scr))
    ##
    obs_mask <-  cbind( obs_R_A,
                        obs_R_B,
                        obs_Y_A,
                        obs_Y_B)
    ##
    y[obs_mask == 0L] <-  -1L
    ##
    stopifnot(!anyNA(x = y))
    ##
    for (j in 1:4) stopifnot( max(y[, j]) <= n_cat[j],
                              all(y[obs_mask[, j] == 1L, j] >= 1L))
    ##
    cat( "N =",
         N_all,
         "\n")
    cat( "observed per test:",
         colSums(x = obs_mask),
         "\n")
    ##
    both <-  obs_R_A == 1L & obs_R_B == 1L
    ##
    s_A <-  (y[, "Y_A"] - 1) / (n_cat[["Y_A"]] - 1)
    s_B <-  (y[, "Y_B"] - 1) / (n_cat[["Y_B"]] - 1)
    s_max <-  pmax( s_A,
                    s_B)
    ##
    sel_ok <-  obs_Y_A == 1L & obs_Y_B == 1L
    sel_by_smax <-  tapply( X = both[sel_ok],
                            INDEX = s_max[sel_ok],
                            FUN = mean)
}
##
## -| --------- 1b. Phase-one modelling groups and explicit target population ------------------------------------------------------------------
##
REQUIRE_REFERENCE_COVERAGE <-  TRUE
##
{
    grouping <-  fn_apms_phase_one_groups(raw = raw)
    ##
    psych_crit <-  grouping$psych_crit
    ##
    pop <-  grouping$pop
    ##
    n_pops <-  length(x = grouping$labels)
    ##
    stratum_levels <-  grouping$labels
    ##
    eligible <-  as.integer(x = v(nm = "eligible"))
    ##
    consent <-  as.integer(x = v(nm = "FollUpGrp") == 1)
    ##
    stopifnot( !anyNA(x = eligible),
               all(eligible %in% 0:1),
               !anyNA(x = consent),
               all(v(nm = "FollUpGrp") %in% 1:2))
    ##
    fpop <-  factor( x = pop,
                     levels = seq_len(length.out = n_pops))
    ##
    sample_pop_count <-  as.numeric(x = table(fpop))
    ##
    target_row_weight <-  as.numeric(x = v(nm = PHASE_ONE_WEIGHT_VAR))
    ##
    stopifnot( all(is.finite(x = target_row_weight)),
               all(target_row_weight > 0))
    ##
    cat( "\nWeighted standardisation assumes within-group transportability of fitted class probabilities.\n",
         "It does not itself account for residual informative sampling or survey clustering.\n")
    ##
    pop_weight <-  as.numeric(x = tapply( X = target_row_weight,
                                          INDEX = fpop,
                                          FUN = sum))
    ##
    strata_table <-  data.frame( pop = seq_len(length.out = n_pops),
                                 group = stratum_levels,
                                 n = sample_pop_count,
                                 target_share = pop_weight / sum(pop_weight),
                                 both_refs = as.numeric(x = tapply( X = both,
                                                                    INDEX = fpop,
                                                                    FUN = sum)))
    ##
    cat("\nPhase-one modelling groups:\n")
    print( x = strata_table,
           row.names = FALSE,
           digits = 3)
    ##
    stopifnot(!any((obs_R_A == 1L | obs_R_B == 1L) & (eligible == 0L | consent == 0L)))
}
##
## -| --------- 1c. DESIGN-WEIGHTED targets ----------------------------------------------------------------------------------------------------
##
{
    w_B <-  as.numeric(x = v(nm = "bdpd_wt"))
    w_A <-  as.numeric(x = v(nm = "ASDwt"))
    ##
    ok_B <-  obs_R_B == 1L & !is.na(x = w_B) & w_B > 0
    ok_A <-  obs_R_A == 1L & !is.na(x = w_A) & w_A > 0
    ##
    weighted_benchmark <-  function( case_var,
                                     positive_value,
                                     weights) {

        if (is.null(x = case_var)) return(NA_real_)
        ##
        column_index <-  which(x = tolower(x = names(x = raw)) == tolower(x = case_var))
        ##
        if (length(x = column_index) != 1L) {
            warning( "benchmark case variable not found: ",
                     case_var)
            ##
            return(NA_real_)
        }
        ##
        case_indicator <-  as.numeric(x = raw[[column_index]])
        ##
        valid_weight_rows <-  is.finite(x = weights) & weights > 0 & is.finite(x = case_indicator) & case_indicator >= 0
        ##
        if (!any(valid_weight_rows)) return(NA_real_)
        ##
        sum(weights[valid_weight_rows] * (case_indicator[valid_weight_rows] == positive_value)) / sum(weights[valid_weight_rows])

    }
    ##
    DESIGN_WEIGHTED <-  c( prev_A = weighted_benchmark( case_var = "com10pl",
                                                        positive_value = 1,
                                                        weights = w_A),
                           prev_B = weighted_benchmark( case_var = "BPDPH2",
                                                        positive_value = 1,
                                                        weights = w_B))
    ##
    cat("\nReference-based weighted benchmarks:\n")
    print(x = DESIGN_WEIGHTED)
}
##
## -| --------- 2. Covariates: POOLED model (the established application configuration) --------------------------------------------------------
##
STRATUM_COVARIATES <-  TRUE
STRATUM_COVARIATE_TESTS <-  c(3L, 4L)
##
USE_OR_PRIOR <-  TRUE
##
COVARIATE_MODEL <-  "pooled" ## ------------------------------
##
POOL_CORRELATIONS_ACROSS_CLASSES <-  FALSE
##
CORRELATION_BOUNDS <-  "force_pos_corr_only_for_same_condition_or_same_test_type"
##
{
    stopifnot( CORRELATION_BOUNDS %in% c("force_pos_corr_only_for_same_condition",
                                         "force_pos_corr_only_for_same_condition_or_same_test_type",
                                         "force_pos_corr_for_everything"))
    ##
    CORR_FORCE_POSITIVE <-  0L
    ##
    lb_corr <-  to3d( m = matrix( data = -1.0,
                                  nrow = n_tests,
                                  ncol = n_tests),
                      k = n_class)
    ##
    ub_corr <-  to3d( m = matrix( data = +1.0,
                                  nrow = n_tests,
                                  ncol = n_tests),
                      k = n_class)
    ##
    is_reference_test <-  seq_len(length.out = n_tests) <= 2L
    ##
    for (i in 2:n_tests) for (j in 1:(i - 1)) {

        same_condition <-  test_condition[i] == test_condition[j]
        same_test_type <-  is_reference_test[i] == is_reference_test[j]
        ##
        floor_this_pair <-  switch( CORRELATION_BOUNDS,
                                    "force_pos_corr_only_for_same_condition" = same_condition,
                                    "force_pos_corr_only_for_same_condition_or_same_test_type" = same_condition ||
                    same_test_type,
                                    "force_pos_corr_for_everything" = TRUE)
        ##
        if (floor_this_pair) for (cc in seq_len(length.out = n_class)) lb_corr[cc, i, j] <-  0.0

    }
    ##
    cat( sprintf( fmt = "\n==== CORRELATION_BOUNDS = %s ====\n",
                  CORRELATION_BOUNDS))
}
##
if (COVARIATE_MODEL == "pooled") {

    n_covariates_max <-  1L
    ##
    X_cov <-  array( data = 0.0,
                     dim = c(4L, N_all, 1L))
    ##
    n_covs_per_outcome <-  rep( x = 0L,
                                times = 4L)
    ##
    covariate_active <-  array( data = 0L,
                                dim = c(2L, 1L, 4L))
    ##
    prior_beta_cov_mean <-  array( data = 0.0,
                                   dim = c(2L, 1L, 4L))
    prior_beta_cov_sd <-  array( data = 0.50,
                                 dim = c(2L, 1L, 4L))
    ##
    COVARIATE_EFFECTS_BY_OWN_STATUS <-  FALSE
    covariate_labels <-  character(length = 0)
    ##
    X_prevalence <-  matrix( data = 0.0,
                             nrow = N_all,
                             ncol = 0L)
    ##
    n_prevalence_covariates <-  0L
    ##
    prior_beta_prevalence_A_mean <-  numeric(length = 0)
    prior_beta_prevalence_A_sd <-  numeric(length = 0)
    prior_beta_prevalence_B_mean <-  numeric(length = 0)
    prior_beta_prevalence_B_sd <-  numeric(length = 0)
    ##
    prevalence_covariate_labels <-  character(length = 0)
    ##
    COVARIATE_PROFILE <-  "pooled_no_covariates"
    ##

} else {
    stop("COVARIATE_MODEL must be 'pooled' in this BayesMVP bridge; the adjusted profile needs R_bridge_covariates_v2.R.")
}
##
## -| --------- Priors (UNCHANGED from R_bridge_APMS2007_styled.R) -----------------------------------------------------------------------------
##
{
    CONDITION_LABELS <-  c("autism", "BPD")
    ##
    TEST_LABELS <-  c("Ref_autism_ADOS4", "Ref_BPD_SCID2", "Screen_autism_AQ20", "Screen_BPD_criteria")
    ##
    CLINICAL_CUTOFF_CATEGORY <-  c(11L, 6L, 11L, 6L)
}
# ##
# REF_ASD_Se_prior <-  list( target_mean = 0.925,
#                            target_interval = c(0.80, 0.95))
# REF_ASD_Sp_prior <-  list( target_mean = 0.99,
#                            target_interval = c(0.95, 0.995))
# REF_BPD_Se_prior <-  list( target_mean = 0.925,
#                            target_interval = c(0.80, 0.95))
# REF_BPD_Sp_prior <-  list( target_mean = 0.99,
#                            target_interval = c(0.95, 0.995))
##
N <-  nrow(x = y)
##
## ---- Ordinal (induced-Dirichlet) accuracy factors.
##
## Three things differ from the earlier settings, all deliberate:
##
##  1. alpha_floor 0.50 -> 0.10. The floor must be affordable on the MINORITY side of the split:
##     C >= n_categories_that_side * alpha_floor / mass_fraction_that_side. For the ADOS absent
##     class that is 15 * 0.50 / 0.01 = 750, which is why asking for 500 pseudo-cases silently
##     produced 750 and pinned all 15 above-cutoff categories flat at 0.50. At 0.10 the same
##     requirement is 150, so the concentration below is honoured as written.
##
##  2. floor_policy "increase_concentration" -> "error". The prior strength is a belief, not
##     something to be adjusted behind our backs; if the floor and the concentration are
##     incompatible we want to be told.
##
##  3. Reference ABSENT concentration 500 -> 200. The references are phase-2: the ADOS is observed
##     on 617 of 7403 subjects and the SCID-BPD on 606, so 750 pseudo-cases outweighed the data
##     that informs reference specificity. 200 is informative without dominating. The PRESENT
##     concentrations are left alone on purpose - only 19 ADOS-positive and 20 BPD-positive
##     subjects are verified, so sensitivity is necessarily prior-led, and that should be stated
##     in the paper and checked for sensitivity rather than hidden.
##
## UNITS: shape_centre is a CATEGORY index like default_cutoff (1 = raw score 0). It used to be
## applied as a raw score, so every centre sat one score too high and the absent classes peaked
## at raw score 1 instead of 0; fixed in fn_alpha_from_target() - the numbers below now mean what
## they say.
##
## shape_kind = "plateau" (present class): normal rise to the centre, then FLAT to the maximum
## score - a true case scores at least around the centre and anywhere above is equally plausible.
## The symmetric bell said a case was 4x less likely to score the ADOS maximum than 15, and 23x
## less likely to meet all nine SCID criteria than six, which is not the belief. Used for the two
## references and both screens (the BPD screen looked monotone only while its centre was mis-placed
## one score too high by the old units bug; at raw 7 the bell drops to 0.56 of its peak by 9).
## For the absent classes with centre 1 (raw 0) bell and plateau coincide, so bell is kept.
##
## shape_join = "smooth" lays ONE shape across all categories and solves its width for the target,
## instead of shaping the two sides of the cutoff independently and rescaling each. The old
## per-side construction stepped at the clinical cutoff - for the AQ-20 absent class it actually
## ROSE across it (0.40 -> 1.91), i.e. the prior said a non-autistic person was likelier to score
## 10 than 9. Drop the shape_join line from any call to go back to the per-side behaviour.
##
## ---- The accuracy-prior specification is SHARED with the cmdstanr bridge (one file, one
## function) so that both fits use identical priors and an algorithm comparison is like for
## like. Edit R_fn_APMS_prior_spec.R, not this block.
##
source(file = file.path( BASE_DIR, "R_fn_APMS_prior_spec.R"), local = TRUE)
##
APMS_PRIOR_SPEC <-  fn_APMS_accuracy_prior_spec( fn_alpha_from_target = fn_alpha_from_target,
                                                 fn_probit_prior_from_target = fn_probit_prior_from_target,
                                                 verbose = TRUE,
                                                 plot = TRUE,
                                                 fn_plot_alpha = fn_plot_alpha)
##
ALPHA_PRESENT <-  APMS_PRIOR_SPEC$ALPHA_PRESENT
ALPHA_ABSENT <-  APMS_PRIOR_SPEC$ALPHA_ABSENT
##
REF_ASD_Se_binary_prior <-  APMS_PRIOR_SPEC$REF_ASD_Se_binary_prior
REF_ASD_Sp_binary_prior <-  APMS_PRIOR_SPEC$REF_ASD_Sp_binary_prior
REF_BPD_Se_binary_prior <-  APMS_PRIOR_SPEC$REF_BPD_Se_binary_prior
REF_BPD_Sp_binary_prior <-  APMS_PRIOR_SPEC$REF_BPD_Sp_binary_prior
##
ALPHA_FLOOR_MIN <-  APMS_PRIOR_SPEC$ALPHA_FLOOR_MIN
##
GRID_THR_A <-  APMS_PRIOR_SPEC$GRID_THR_A
GRID_THR_B <-  APMS_PRIOR_SPEC$GRID_THR_B
##
USE_COND_PREV_PRIOR <-  TRUE
##
PREV_B_BACKGROUND_CENTRE <-  0.01
##
PR_B_GIVEN_A_BELIEF <-  c(2.5, 50.0) / 100
##
PR_B_GIVEN_A_PRIOR <-  fn_logit_normal_from_interval(target_interval = PR_B_GIVEN_A_BELIEF)
##
cat( sprintf( fmt = "\nPr(BPD | autism) prior: median %.4f, central 95%% %.4f - %.4f  [logit N(%.4f, %.4f)]\n",
              PR_B_GIVEN_A_PRIOR$median,
              PR_B_GIVEN_A_PRIOR$achieved_interval[1],
              PR_B_GIVEN_A_PRIOR$achieved_interval[2],
              PR_B_GIVEN_A_PRIOR$logit_mean,
              PR_B_GIVEN_A_PRIOR$logit_sd))
##
## -| --------- 2b. BASE Stan data list (the M6 view: all ordinal, accuracy estimated) ---------------------------------------------------------
##
{
    n_binary_tests  <- 0
    n_ordinal_tests <- 4
    ##
    class_D <-  rbind( c(0L, 0L),
                       c(1L, 0L),
                       c(0L, 1L),
                       c(1L, 1L))
    ##
    test_condition <-  c(1L, 2L, 1L, 2L)
    ##
    free_pairs <-  list( c(2, 1),
                         c(3, 1),
                         c(4, 1),
                         c(3, 2),
                         c(4, 2),
                         c(4, 3))
    ##
    known_values_indicator <-  matrix( data = 0,
                                       nrow = n_tests,
                                       ncol = n_tests)
    known_values <-  matrix( data = 0.0,
                             nrow = n_tests,
                             ncol = n_tests)
    ##
    for (i in 2:n_tests) {
        for (j in 1:(i - 1)) {
            known_values_indicator[i, j] <-  1
        }
    }
    ##
    for (correlation_pair in free_pairs) {
        known_values_indicator[correlation_pair[1], correlation_pair[2]] <-  0
    }
    ##
    known_values_list <-  to3d( m = known_values,
                                k = n_class)
    known_values_indicator_list <-  to3d( m = known_values_indicator,
                                          k = n_class)
    ##
    ## ---- NOTE: prior_dirichlet_alpha is NOT built here any more. It is built - flat where no
    ## factor is supplied, informative where one is - by fn_attach_accuracy_priors() below, which
    ## attaches the accuracy priors to the data itself so that they travel with it into every
    ## model, every subset and every parallel worker (see the block after this list).
    ##
    stan_data_base <-  list( N = N_all,
                             n_tests = n_tests,
                             n_class = n_class,
                             class_D = class_D,
                             test_condition = test_condition,
                             ##
                             allow_cross_loading = rep(0, n_tests),
                             is_perfect_ref = rep(0, n_tests),
                             perfect_ref_M = 8.0,
                             ##
                             n_binary_tests = n_binary_tests,
                             n_ordinal_tests = n_ordinal_tests,
                             n_cat_per_ord_test = unname(obj = n_cat),
                             n_thr_per_ord_test = unname(obj = n_cat) - 1,
                             ##
                             y = y,
                             n_pops = n_pops,
                             pop = pop,
                             pop_weight = pop_weight,
                             ##
                             n_covariates_max = n_covariates_max,
                             X = X_cov,
                             n_covs_per_outcome = n_covs_per_outcome,
                             ##
                             corr_force_positive = CORR_FORCE_POSITIVE,
                             known_values_list = known_values_list,
                             known_values_indicator_list = known_values_indicator_list,
                             lb_corr = lb_corr,
                             ub_corr = ub_corr,
                             ##
                             overflow_threshold  = +7.5,
                             underflow_threshold = -7.5,
                             C_raw_lower = -10.0,
                             C_raw_upper = +2.5,
                             ##
                             prior_only = 0,
                             prior_beta_mean = rbind( rep(0.0, n_tests),
                                                      rep(0.0, n_tests)),
                             prior_beta_sd = rbind( rep(1.0, n_tests),
                                                    rep(1.0, n_tests)),
                             ##
                             prior_delta_cross_sd = rep(0.25, n_tests),
                             ##
                             prior_beta_cov_mean = prior_beta_cov_mean,
                             prior_beta_cov_sd = prior_beta_cov_sd,
                             covariate_active = covariate_active,
                             covariate_effects_by_own_status = as.integer(x = COVARIATE_EFFECTS_BY_OWN_STATUS),
                             ##
                             X_prevalence = X_prevalence,
                             prior_beta_prevalence_A_mean = prior_beta_prevalence_A_mean,
                             prior_beta_prevalence_A_sd = prior_beta_prevalence_A_sd,
                             prior_beta_prevalence_B_mean = prior_beta_prevalence_B_mean,
                             prior_beta_prevalence_B_sd = prior_beta_prevalence_B_sd,
                             ##
                             pool_correlations_across_classes = as.integer(x = POOL_CORRELATIONS_ACROSS_CLASSES),
                             prior_LKJ_global    = 2.0,
                             prior_LKJ_departure = 10.0,
                             prior_lambda_shape_a = 1.0,
                             prior_lambda_shape_b = 4.0,
                             ##
                             covariate_labels = covariate_labels,
                             prevalence_covariate_labels = prevalence_covariate_labels,
                             covariate_profile = COVARIATE_PROFILE,
                             ##
                             condition_labels = CONDITION_LABELS,
                             test_labels = TEST_LABELS,
                             clinical_cutoff_category = CLINICAL_CUTOFF_CATEGORY,
                             ##
                             ## prior_dirichlet_alpha: set by fn_attach_accuracy_priors() below.
                             ##
                             ref_cutoff_category_A = unname(obj = ref_thr_cat[["R_A"]]),
                             ref_cutoff_category_B = unname(obj = ref_thr_cat[["R_B"]]),
                             ##
                             prior_LKJ = matrix( data = c(4.0, 4.0),
                                                 nrow = n_class,
                                                 ncol = 1),
                             ##
                             use_OR_prior = as.integer(x = USE_OR_PRIOR),
                             log_OR_shared = 1,
                             ##
                             prior_mu_logit_A_mean = qlogis(p = 0.05),
                             prior_mu_logit_A_sd = 0.5,
                             prior_mu_logit_B_mean = qlogis(p = if (USE_COND_PREV_PRIOR) PREV_B_BACKGROUND_CENTRE else 0.02),
                             prior_mu_logit_B_sd =  0.75,
                             prior_sigma_logit_sd = 0.75,
                             ##
                             prior_log_OR_mean = 0.0,
                             prior_log_OR_sd   = 2.0,
                             ##
                             use_cond_prior = as.integer(x = USE_COND_PREV_PRIOR),
                             prior_logit_Q_mean = PR_B_GIVEN_A_PRIOR$logit_mean,
                             prior_logit_Q_sd = PR_B_GIVEN_A_PRIOR$logit_sd,
                             ##
                             prior_prev_dirichlet = c(25, 2, 2, 1),
                             ##
                             Phi_type = PHI_TYPE,   ## set next to USE_AVX_STAN_EXTERNALS at the top
                             baseline_case = matrix( data = 0.0,
                                                     nrow = n_tests,
                                                     ncol = n_covariates_max),
                             ##
                             n_grid_A = length(x = GRID_THR_A),
                             n_grid_B = length(x = GRID_THR_B),
                             grid_thr_A = GRID_THR_A,
                             grid_thr_B = GRID_THR_B,
                             n_quad_nodes = 64,   ## MUST match the cmdstanr bridge and the sim study (64) for a like-for-like comparison; was 16
                             ##
                             save_subject_probs = 0,
                             ##
                             grainsize = fn_grainsize_from_chunks( N = N_all,
                                                                   num_chunks = sampler_settings$num_chunks_burnin),
                             ##
                             n_loo_draws = 0)
    ##
    ## ---- Accuracy priors go IN THE DATA, here, once.
    ##
    ## fn_build_stan_data_real() used to reach into the global environment for ALPHA_PRESENT /
    ## ALPHA_ABSENT / REF_*_binary_prior / ALPHA_FLOOR_MIN through exists(). Anything that built
    ## model data without those objects in scope - a PSOCK worker, a re-used helper, a fresh
    ## session - silently got FLAT priors instead. Attaching them to stan_data_base means they are
    ## carried by the data through subsetting, through the per-model builders and across workers,
    ## and a base that lacks them is a hard error rather than a quiet coin-flip prior.
    ##
    ## Ordinal (induced-Dirichlet) factors, one per test in column order R_A, R_B, Y_A, Y_B; and
    ## the probit-location priors for the BINARY reference view (M5), derived at the top of this
    ## script from the same target beliefs:
    ##
    stan_data_base <-  fn_attach_accuracy_priors( stan_data = stan_data_base,
                                                  ##
                                                  alpha_present = list( ALPHA_PRESENT$ref_ASD,
                                                                        ALPHA_PRESENT$ref_BPD,
                                                                        ALPHA_PRESENT$screen_ASD,
                                                                        ALPHA_PRESENT$screen_BPD),
                                                  alpha_absent = list( ALPHA_ABSENT$ref_ASD,
                                                                       ALPHA_ABSENT$ref_BPD,
                                                                       ALPHA_ABSENT$screen_ASD,
                                                                       ALPHA_ABSENT$screen_BPD),
                                                  ##
                                                  binary_reference_priors = list( R_A = list( Se = REF_ASD_Se_binary_prior,
                                                                                              Sp = REF_ASD_Sp_binary_prior),
                                                                                  R_B = list( Se = REF_BPD_Se_binary_prior,
                                                                                              Sp = REF_BPD_Sp_binary_prior)),
                                                  ##
                                                  alpha_floor_min = ALPHA_FLOOR_MIN)
    ##
    stan_data_base <-  fn_complete_stan_data( stan_data = stan_data_base,
                                              survey_weight = target_row_weight)
    stan_data_base <-  fn_build_standardisation_set( stan_data = stan_data_base,
                                                     survey_weight = target_row_weight)
    ##
    stan_data_base <-  fn_build_grid_standardisation_set( stan_data = stan_data_base,
                                                          n_grid_profiles = 1000L)
    ##
    invisible(x = fn_prior_check_cond_prevalence( stan_data = stan_data_base,
                                                  n_prior_draws = 1000L))
    ##
    cat( sprintf( fmt = paste0( "reduce_sum_static: N = %d; burn-in %d chunks / %d WCP threads (initial grainsize %d); ",
                                "sampling %d chunks / %d WCP threads. Phase grainsizes are recalculated after subsetting.\n"),
                  N_all,
                  sampler_settings$num_chunks_burnin,
                  sampler_settings$n_threads_WCP_burnin,
                  stan_data_base$grainsize,
                  sampler_settings$num_chunks_sampling,
                  sampler_settings$n_threads_WCP_sampling))
}
##
## -| --------- 3. Sampler settings + model / subset configuration -----------------------------------------------------------------------------
##
sampler_settings$Stan_model_file_path <-  STAN_FILE
##
## external C++ for the AVX variant (NULL = none). BayesMVP turns a header into
## stanc --allow-undefined + make USER_HEADER=<path> for BridgeStan:
sampler_settings$Stan_cpp_user_header <-  if (USE_AVX_STAN_EXTERNALS) STAN_USER_HEADER_AVX else NULL
##
## ---- per-model sampling counts / "treedepth" (established APMS settings, see section 4b of
## ---- the styled bridge; fn_treedepth_for_model returns the max leapfrog count = 2^8):
##
{
    fn_sampling_for_model <-  function(model_i) {
    
          stopifnot(model_i %in% MODEL_CONFIGS)
          ##
          as.integer(x = sampler_settings$n_iter)
    
    }
    ##
    fn_treedepth_for_model <-  function(model_i) {
    
          stopifnot(model_i %in% MODEL_CONFIGS)
          ##
          as.integer(x = sampler_settings$max_L)
    
    }
    ##
    ## ---- Capacity check for the parallel layout. The chain counts are the user's
    ## (sampler_settings); this verifies n_models x max(chains x WCP threads for each phase)
    ## fits in the configured cores, and stops loudly if not. Nothing
    ## is capped or replaced.
    ## reserve = 0: the master R process only waits on the PSOCK workers while they sample, so
    ## it needs no core of its own; 4 models x 32 chains on N_CORES = 128 is a legitimate layout.
    fn_check_parallel_capacity <-  function( n_models,
                                             settings,
                                             n_cores = N_CORES,
                                             threads = 1,
                                             reserve = 0) {

          cores_per_model <-  max(  settings$n_chains_burnin * settings$n_threads_WCP_burnin,
                                    settings$n_chains_sampling * settings$n_threads_WCP_sampling)
          ##
          cores_needed <-  n_models * cores_per_model * threads
          ##
          if (cores_needed > n_cores - reserve)
              stop( sprintf( fmt = paste0( "Parallel layout needs %d cores (%d models x max(%d burn-in chains x %d WCP, %d sampling chains x %d WCP) x %d) ",
                                           "but N_CORES - reserve = %d. Lower chain / WCP thread counts in fn_sampler_settings_APMS_BayesMVP(), ",
                                           "fit fewer models at once, or set RUN_PARALLEL = FALSE. (No cap is applied silently.)"),
                             cores_needed,
                             n_models,
                             settings$n_chains_burnin,
                             settings$n_threads_WCP_burnin,
                             settings$n_chains_sampling,
                             settings$n_threads_WCP_sampling,
                             threads,
                             n_cores - reserve))
          ##
          invisible(x = cores_needed)

    }
}
##
{
    fits_by_model <-  list()
    gates_by_model <-  list()
    prior_checks <-  list()
}
##
## The full-fit benchmark preparation helper stops HERE, before any prior checks, compilation or fitting.
APMS_PREPARATION_COMPLETE <-  TRUE
##
## ---- Resolve the selected implementation only for a bridge run; benchmark preparation stops above ----------------------------------------
## Generate from the current AVX source and use its matching header. Content-based paths prevent reuse of an older fused model.
##
if (BAYESMVP_CONFIGURATION %in% c("BayesMVP_AVX_fused", "BayesMVP_AVX_whole_vector")) {
    source(file = file.path(BASE_DIR, "stan_ext", "R_make_APMS_AVX_fused_variant.R"), local = TRUE)
    ##
    avx_variant <-  fn_make_APMS_AVX_fused_variant(
        model_file = STAN_FILE_AVX,
        extension_dir = file.path(BASE_DIR, "stan_ext"),
        implementation = if (BAYESMVP_CONFIGURATION == "BayesMVP_AVX_fused") "fused" else "whole_vector")
    ##
    STAN_FILE <-  avx_variant$model_file
    STAN_USER_HEADER_AVX <-  avx_variant$user_header
}
##
sampler_settings$Stan_model_file_path <-  STAN_FILE
sampler_settings$Stan_cpp_user_header <-  if (USE_AVX_STAN_EXTERNALS) STAN_USER_HEADER_AVX else NULL
cat("\nBayesMVP configuration: ", BAYESMVP_CONFIGURATION, "\nStan model: ", STAN_FILE,
    "\nUser header: ", if (USE_AVX_STAN_EXTERNALS) STAN_USER_HEADER_AVX else "none", "\n", sep = "")
##
## Match the benchmark: select once from the ordinal base, before building model-specific views; retain all GQ/LOO settings.
## The prior check continues to use the full base and its own independent subset controls.
##
stan_data_posterior <-  stan_data_base
if (subset) {
    if (N_subset > stan_data_base$N) stop("N_subset exceeds the prepared APMS data size.")
    ##
    stan_data_posterior <-  fn_subset_stan_data( stan_data = stan_data_base,
                                                 n_total = N_subset,
                                                 seed = subset_seed,
                                                 oversample_verified = 1,
                                                 min_per_cell = 0,
                                                 verbose = TRUE)
}
##
## -| --------- 4. THE PRIOR-ONLY RUN (its OWN subset configuration) ---------------------------------------------------------------------------
##
{
    RUN_PRIOR_CHECK <-  FALSE
    ##
    PRIOR_CHECK_CHAINS <- NULL   ## NULL = the prior run keeps sampler_settings' own burn-in / sampling chain counts; a number sets BOTH to it
    PRIOR_CHECK_WARMUP   <- 125
    PRIOR_CHECK_SAMPLING <- 200
    ##
    PRIOR_CHECK_ROWS <- 1000
    ##
    ## standardisation / PPV-grid rows for the PRIOR run only (the Se / Sp priors do not depend
    ## on the grid, and it is ~95% of a prior-only draw's cost). NULL = every kept row.
    PRIOR_CHECK_STANDARDISATION_ROWS <- 50
    ##
    MODELS_TO_FIT_FOR_PRIOR_PRED_CHECK <-  "M6_imperfect_ord"
}
##
if (RUN_PRIOR_CHECK) {
    ##
    ## Require a subset: NULL or a row count >= N would let the helper use the full data.
    if (!is.numeric(x = PRIOR_CHECK_ROWS) || length(x = PRIOR_CHECK_ROWS) != 1L ||
        !is.finite(x = PRIOR_CHECK_ROWS) || PRIOR_CHECK_ROWS < 1 ||
        PRIOR_CHECK_ROWS != floor(x = PRIOR_CHECK_ROWS) || PRIOR_CHECK_ROWS >= stan_data_base$N) {

        stop("Set PRIOR_CHECK_ROWS to a positive integer smaller than stan_data_base$N; prior checks must use a subset.")

    }
    ##
    for (model_i in MODELS_TO_FIT_FOR_PRIOR_PRED_CHECK) {
        ##
        stan_data_model_i <-  fn_build_stan_data_real( base = stan_data_base,
                                                       model_config = model_i)
        ##
        prior_checks[[model_i]] <-  fn_run_prior_check_BayesMVP( model_i = model_i,
                                                                 stan_data = stan_data_model_i,
                                                                 settings = sampler_settings,
                                                                 n_prior_rows = PRIOR_CHECK_ROWS,
                                                                 n_chains = PRIOR_CHECK_CHAINS,
                                                                 n_warmup = PRIOR_CHECK_WARMUP,
                                                                 n_sampling = PRIOR_CHECK_SAMPLING,
                                                                 n_standardisation_prior = PRIOR_CHECK_STANDARDISATION_ROWS,
                                                                 seed = SEED,
                                                                 ##
                                                                 ## same report as the cmdstanr prior check:
                                                                 test_labels = TEST_LABELS,
                                                                 condition_labels = CONDITION_LABELS,
                                                                 accuracy_window = NULL,        ## NULL = every threshold
                                                                 show_cutpoints = FALSE,
                                                                 show_accuracy_table = TRUE)
    }
    ##
    saveRDS( object = prior_checks,
             file = file.path( RUN_DIR,
                               "prior_checks.RDS"))
}
##
if (!is.null(x = prior_checks[["M6_imperfect_ord"]])) {
    fn_reprint_prior_check_BayesMVP( prior_check = prior_checks[["M6_imperfect_ord"]],
                                     test_labels = TEST_LABELS,
                                     condition_labels = CONDITION_LABELS)
}
##
## -| --------- 5. Fit each model to the (subsetted) data --------------------------------------------------------------------------------------
##
##   - Subsets are for computation checks. The full sample is needed for the application.
##   - Each worker builds its own BayesMVP model object (an R6 wrapper around a compiled
##     BridgeStan model cannot be exported across PSOCK workers - same constraint as the
##     cmdstanr bridge, which re-attached via exe_file).
##
{
    jobs <-  lapply( X = MODELS_TO_FIT,
                     FUN = function(model_i) {

        stan_data_model_i <-  fn_build_stan_data_real( base = stan_data_posterior,
                                                       model_config = model_i)
        ##
        fn_reference_coverage( stan_data = stan_data_model_i,
                               label = model_i,
                               require_coverage = REQUIRE_REFERENCE_COVERAGE && !subset,
                               output_file = file.path( RUN_DIR,
                                                        paste0( model_i, "_reference_coverage.csv")))
        ##
        list( model_i = model_i,
              stan_data = stan_data_model_i,
              prev0 = c(0.985, 0.010, 0.004, 0.001))

    })
    ##
    names(x = jobs) <-  MODELS_TO_FIT
    ##
    for (model_i in MODELS_TO_FIT) fn_print_alpha( stan_data_local = jobs[[model_i]]$stan_data,
                                                   label = model_i)
    ##
    ## ---- SERIAL path (RUN_PARALLEL = FALSE): one model at a time
    if (!RUN_PARALLEL || length(x = MODELS_TO_FIT) == 1L) {

        for (model_i in MODELS_TO_FIT) {

            cat( "\n\n================================  ",
                 model_i,
                 "  ================================\n")
            ##
            fits_by_model[[model_i]] <-  fn_fit_APMS_model_BayesMVP( model_i = model_i,
                                                                     stan_data = jobs[[model_i]]$stan_data,
                                                                     settings = sampler_settings,
                                                                     prev0 = jobs[[model_i]]$prev0)
            ##
            gates_by_model[[model_i]] <-  fn_report_fit_BayesMVP( outs_fit = fits_by_model[[model_i]],
                                                                  stan_data_local = jobs[[model_i]]$stan_data,
                                                                  max_rhat_thr = max_rhat_thr,
                                                                  min_ESS_thr = min_ESS_thr)

        }

    } else {
        ##
        ## ---- PARALLEL path: one PSOCK worker per model
        ##
        require(RcppParallel)
        require(foreach)
        require(doParallel)
        require(parallel)
        ##
        n_models <-  length(x = MODELS_TO_FIT)
        ##
        cores_needed <-  fn_check_parallel_capacity( n_models = n_models,
                                                     settings = sampler_settings)
        ##
        cat( sprintf( fmt = "\n==== PARALLEL: %d models x max(%d burn-in chains x %d WCP, %d sampling chains x %d WCP) = %d of %d cores ====\n",
                      n_models,
                      sampler_settings$n_chains_burnin,
                      sampler_settings$n_threads_WCP_burnin,
                      sampler_settings$n_chains_sampling,
                      sampler_settings$n_threads_WCP_sampling,
                      cores_needed,
                      N_CORES))
        ##
        worker_export_environment <-  new.env()
        ##
        worker_export_environment$jobs <-  jobs
        worker_export_environment$sampler_settings <-  sampler_settings
        worker_export_environment$STAN_FILE <-  normalizePath( path = STAN_FILE,
                                                               mustWork = TRUE)
        worker_export_environment$BASE_DIR <-  BASE_DIR
        worker_export_environment$HELPER_FILE <-  normalizePath( path = file.path( BASE_DIR,
                                                                                    "../R_packages/BayesMVP/APMS_bridge/R_fn_APMS_BayesMVP.R"),
                                                                 mustWork = TRUE)
        worker_export_environment$SIM_HELPER_FILE <-  normalizePath( path = file.path( BASE_DIR,
                                                                                        "R_fn_sim_4class_joint_LC_MVOP_data.R"),
                                                                     mustWork = TRUE)
        ##
        worker_cluster <-  makeCluster( spec = n_models,
                                        type = "PSOCK",
                                        outfile = "")
        ##
        registerDoParallel(cl = worker_cluster)
        ##
        clusterExport( cl = worker_cluster,
                       varlist = ls(envir = worker_export_environment),
                       envir = worker_export_environment)
        ##
        parallel_results <-  foreach( job_index = seq_len(length.out = n_models),
                                      .errorhandling = "pass") %dopar% {
            tryCatch( expr = {
              
                    require(RcppParallel)
                    require(BayesMVP)
                    require(bridgestan)
                    ##
                    source(file = SIM_HELPER_FILE)
                    source(file = HELPER_FILE)
                    ##
                    job <-  jobs[[job_index]]
                    ##
                    ## this worker's settings are the user's sampler_settings UNCHANGED (burn-in
                    ## and sampling chain counts included); capacity was checked on the master.
                    settings_local <-  sampler_settings
                    settings_local$Stan_model_file_path <-  STAN_FILE
                    ##
                    assign( x = "Stan_model_file_path",
                            value = STAN_FILE,
                            envir = globalenv())
                    ##
                    outs_fit <-  fn_fit_APMS_model_BayesMVP( model_i = job$model_i,
                                                             stan_data = job$stan_data,
                                                             settings = settings_local,
                                                             prev0 = job$prev0)
                    ##
                    ## return plain-R summaries only: the R6 objects wrap BridgeStan external
                    ## pointers and the draws array, none of which can be serialized back over
                    ## PSOCK without breaking the worker connection:
                    list( model_i = job$model_i,
                          status = "completed",
                          outs_fit = list( model_i = outs_fit$model_i,
                                           summary_main = outs_fit$summary_main,
                                           summary_transformed = outs_fit$summary_transformed,
                                           summary_generated_quantities = outs_fit$summary_generated_quantities,
                                           divergences = outs_fit$divergences,
                                           wall_time_min = outs_fit$wall_time_min,
                                           settings = outs_fit$settings,
                                           timing = outs_fit$timing,
                                           stan_data = outs_fit$stan_data))
                
            },
                      error = function(error_condition) {

                cat(sprintf( fmt = "\n  [worker %d] ERROR: %s\n",
                             job_index,
                             error_condition$message))
                ##
                list( model_i = jobs[[job_index]]$model_i,
                      status = "error",
                      error = error_condition$message)

            })
        }
        ##
        stopCluster(cl = worker_cluster)
        ##
        gc( reset = TRUE,
            full = TRUE)
        ##
        for (parallel_result in parallel_results) {

            if (is.null(x = parallel_result$status) || parallel_result$status != "completed") {
                cat( sprintf( fmt = "\n  MODEL FAILED: %s - %s\n",
                              if (is.null(x = parallel_result$model_i)) "?" else parallel_result$model_i,
                              if (is.null(x = parallel_result$error)) paste(
                                 utils::capture.output(print(x = parallel_result)),
                                 collapse = " "
                             ) else parallel_result$error))
                ##
                next
            }
            ##
            fits_by_model[[parallel_result$model_i]] <-  parallel_result$outs_fit
            ##
            gates_by_model[[parallel_result$model_i]] <-  fn_report_fit_BayesMVP(
                outs_fit = parallel_result$outs_fit,
                stan_data_local = jobs[[parallel_result$model_i]]$stan_data,
                max_rhat_thr = max_rhat_thr,
                min_ESS_thr = min_ESS_thr)

        }
    }
}
##
## -| --------- 6. Application table + DGM extraction ------------------------------------------------------------------------------------------
##
reportable_models <-  names(x = gates_by_model)[vapply( X = gates_by_model,
                                                        FUN = function(convergence_gate)
            isTRUE(x = convergence_gate[["gate_passed"]] == 1),
                                                        FUN.VALUE = logical(length = 1))]
##
if (length(x = fits_by_model) > 0) {
    ##
    application_table <-  fn_application_table_BayesMVP( fits_by_model = fits_by_model,
                                                         target = ESTIMAND_TARGET)
    ##
    cat("\n==== APPLICATION: APMS 2007, all models (BayesMVP/BridgeStan) ====\n")
    print( x = application_table,
           row.names = FALSE,
           right = FALSE)
    ##
    DGM_params_by_model <-  lapply( X = names(x = fits_by_model),
                                    FUN = function(model_i)
        fn_extract_DGM_params_BayesMVP( outs_fit = fits_by_model[[model_i]],
                                        seed = SEED))
    ##
    names(x = DGM_params_by_model) <-  names(x = fits_by_model)
    ##
    saveRDS( object = list( DGM_params_by_model = DGM_params_by_model,
                            bayesmvp_configuration = BAYESMVP_CONFIGURATION,
                            sampler_settings = sampler_settings,
                            N_subset = N_subset,
                            subset_seed = if (subset) subset_seed else NULL,
                            summaries_by_model = lapply( X = fits_by_model,
                                                         FUN = function(outs_fit) list(
                                         main = outs_fit$summary_main,
                                         transformed = outs_fit$summary_transformed,
                                         generated_quantities = outs_fit$summary_generated_quantities,
                                         divergences = outs_fit$divergences)),
                            gates = gates_by_model,
                            prior_checks = prior_checks,
                            target = ESTIMAND_TARGET,
                            phase_one_weight_var = PHASE_ONE_WEIGHT_VAR,
                            strata = strata_table,
                            design_weighted = DESIGN_WEIGHTED,
                            run_dir = RUN_DIR,
                            stan_data_by_model = jobs,
                            app_tbl = application_table),
             file = file.path( BASE_DIR,
                               "apms07_fit_and_DGM_params_using_BayesMVP.rds"))
    ##
    cat( "\n==== written:",
         file.path( BASE_DIR,
                    "apms07_fit_and_DGM_params_using_BayesMVP.rds"),
         "  (DGM params for:",
         paste( names(x = DGM_params_by_model),
                collapse = ", "),
         ") ====\n")
    ##
}
##
if (subset) message("Computational subset fit complete. Set N_subset <- NULL in EDIT HERE for a full-data fit.")
##
cat( "\nReference-based weighted estimates are comparators, not lower bounds on latent prevalence.\n",
     "Adding dependence need not lower prevalence. No change can also be an informative result.\n")




















# 
# 
# 
# packageVersion(pkg = "bridgestan")
# packageVersion(pkg = "cmdstanr")
# cmdstanr::cmdstan_version()
# # Sys.getenv(x = "BRIDGESTAN")
# BayesMVP:::bridgestan_path()
# ##
# system2( command = path.expand(path = "~/.bridgestan/bridgestan-2.5.0/bin/stanc"),
#          args = "--version",
#          stdout = TRUE,
#          stderr = TRUE)
# 
# 
# 
# 









