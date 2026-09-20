#### ================================================================================================================================================================
## test_07_flow_kick_flow.R
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
## Flow-kick-flow (FKF) joint diffusion integrator: selector validation + posterior-equivalence smoke.
##
## (A) R-level selector validation (init_EHMC_args_as_Rcpp_List) - pure R, no native calls.
## (B) Bridge settings + settings gate (fn_sampler_settings_APMS_BayesMVP /
##     fn_validate_settings_APMS_BayesMVP from R_fn_APMS_BayesMVP.R) - pure R.
## (C) Native smoke: joint diffusion HMC (partitioned_HMC = FALSE, diffusion_HMC = TRUE) on the
##     nuisance_present fixture, once with kick-flow-kick and once with flow-kick-flow. Both must
##     run, record their selector, and give the same posterior (smoke-level tolerance, NOT a
##     convergence test). L + 1 vs L + 2 evaluation counts are a C++-level property and are not
##     measurable from R; the wrapper-level reuse is what makes consecutive FKF trajectories
##     cost the same as KFK.
##
## Requires the REBUILT native package (the diffusion_HMC_integrator selector lives in C++ and in
## the R plumbing). If the installed package predates the selector, this test SKIPS with exit 0.
## ================================================================================================================================================================
##
suppressPackageStartupMessages({
    loadNamespace("RcppParallel")
    require(BayesMVP)
    require(bridgestan)
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
MODEL_DIR <- file.path( TESTS_DIR,
                        "fixtures/stan_models")
##
BRIDGE_DIR <- normalizePath(path = file.path( TESTS_DIR,
                                              ".."))
##
OUT_DIR <- file.path( TESTS_DIR,
                      "test_output")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
##
source(file = file.path( TESTS_DIR,
                         "fixtures/settings_smoke.R"))
##
source(file = file.path( BRIDGE_DIR,
                         "R_fn_APMS_BayesMVP.R"))
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
fn_expect_error <- function(name,
                            expr) {
  
    raised <- tryCatch({ force(expr); FALSE },
                       error = function(e) TRUE)
    if (isTRUE(raised)) { n_pass <<- n_pass + 1; cat(sprintf("  PASS: %s (error as expected)\n", name)) }
    else                { n_fail <<- n_fail + 1; cat(sprintf("  FAIL: %s (no error raised)\n", name)) }
    results[[name]] <<- raised
    
}
##
cat("\n======== test_07: flow-kick-flow integrator (selector + equivalence smoke) ========\n")
##
## ---- capability detection: the selector must exist in the installed package -------------
##
has_selector <- tryCatch({
    "diffusion_HMC_integrator" %in% formalArgs(BayesMVP:::init_EHMC_args_as_Rcpp_List)
}, error = function(e) FALSE)
##
if (!isTRUE(has_selector)) {
    cat("SKIP: the installed BayesMVP package does not have the diffusion_HMC_integrator selector.\n")
    cat("      Rebuild/reinstall the inner native package first (see FLOW_KICK_FLOW.md).\n")
    results <- list(skipped = TRUE,
                    reason = "installed package predates the diffusion_HMC_integrator selector")
    saveRDS(results, file = file.path(OUT_DIR, "test_07_results.RDS"))
    quit(status = 0)
}
##
## ---- (A) R-level selector validation -----------------------------------------------------
##
cat("\n(A) init_EHMC_args_as_Rcpp_List selector validation:\n")
##
args_default <- BayesMVP:::init_EHMC_args_as_Rcpp_List(diffusion_HMC = TRUE)
fn_check("A.1 default integrator is 'kick_flow_kick'",
         identical(args_default$diffusion_HMC_integrator, "kick_flow_kick"))
##
args_fkf <- BayesMVP:::init_EHMC_args_as_Rcpp_List( diffusion_HMC = TRUE,
                                                    diffusion_HMC_integrator = "flow_kick_flow")
fn_check("A.2 explicit 'flow_kick_flow' stored",
         identical(args_fkf$diffusion_HMC_integrator, "flow_kick_flow"))
##
fn_expect_error("A.3 invalid string rejected",
                BayesMVP:::init_EHMC_args_as_Rcpp_List(diffusion_HMC = TRUE,
                                                       diffusion_HMC_integrator = "flow_then_kick"))
##
fn_expect_error("A.4 non-character rejected",
                BayesMVP:::init_EHMC_args_as_Rcpp_List(diffusion_HMC = TRUE,
                                                       diffusion_HMC_integrator = 1))
##
fn_expect_error("A.5 NA rejected",
                BayesMVP:::init_EHMC_args_as_Rcpp_List(diffusion_HMC = TRUE,
                                                       diffusion_HMC_integrator = NA_character_))
##
## ---- (B) bridge settings + settings gate -------------------------------------------------
##
cat("\n(B) bridge settings gate (fn_validate_settings_APMS_BayesMVP):\n")
##
settings_app <- fn_sampler_settings_APMS_BayesMVP()
##
fn_check("B.1 bridge defaults to the paper algorithm (partitioned_HMC = FALSE)",
         isFALSE(settings_app$partitioned_HMC))
fn_check("B.2 bridge defaults to diffusion_HMC = TRUE",
         isTRUE(settings_app$diffusion_HMC))
fn_check("B.3 bridge default integrator is 'kick_flow_kick'",
         identical(settings_app$diffusion_HMC_integrator, "kick_flow_kick"))
fn_check("B.4 full application settings validate",
         isTRUE(fn_validate_settings_APMS_BayesMVP(settings = settings_app)))
##
settings_bad <- settings_app
settings_bad$partitioned_HMC <- NULL
fn_expect_error("B.5 missing partitioned_HMC rejected",
                fn_validate_settings_APMS_BayesMVP(settings = settings_bad))
##
settings_bad <- settings_app
settings_bad$diffusion_HMC <- NULL
fn_expect_error("B.6 missing diffusion_HMC rejected",
                fn_validate_settings_APMS_BayesMVP(settings = settings_bad))
##
settings_bad <- settings_app
settings_bad$use_disk <- NULL
fn_expect_error("B.7 missing use_disk rejected",
                fn_validate_settings_APMS_BayesMVP(settings = settings_bad))
##
settings_bad <- settings_app
settings_bad$M_decay_type <- NULL
fn_expect_error("B.8 missing M_decay_type rejected",
                fn_validate_settings_APMS_BayesMVP(settings = settings_bad))
##
settings_bad <- settings_app
settings_bad$M_decay_power <- NULL
fn_expect_error("B.9 missing M_decay_power rejected",
                fn_validate_settings_APMS_BayesMVP(settings = settings_bad))
##
settings_ok <- settings_app
settings_ok$M_decay_scale <- NULL   ## NULL is the documented default (n_adapt / 5)
fn_check("B.10 M_decay_scale = NULL stays allowed",
         isTRUE(fn_validate_settings_APMS_BayesMVP(settings = settings_ok)))
##
settings_bad <- settings_app
settings_bad$diffusion_HMC_integrator <- "flow_kick_flow"
settings_bad$partitioned_HMC <- TRUE
fn_expect_error("B.11 flow_kick_flow + partitioned_HMC = TRUE rejected",
                fn_validate_settings_APMS_BayesMVP(settings = settings_bad))
##
settings_bad <- settings_app
settings_bad$diffusion_HMC_integrator <- "flow_kick_flow"
settings_bad$diffusion_HMC <- FALSE
fn_expect_error("B.12 flow_kick_flow + diffusion_HMC = FALSE rejected",
                fn_validate_settings_APMS_BayesMVP(settings = settings_bad))
##
## ---- (C) native smoke: FKF vs KFK on the nuisance-present fixture ------------------------
##
cat("\n(C) joint diffusion smoke: flow_kick_flow vs kick_flow_kick:\n")
##
settings_c <- fn_smoke_settings()
{
    settings_c$n_iter   <- 250
    settings_c$n_burnin <- 250
    settings_c$partitioned_HMC <- FALSE
    settings_c$diffusion_HMC   <- TRUE
    settings_c$n_chains_burnin <- 4
    settings_c$n_chains_sampling <- 16
    settings_c$interval_width_main <- 1
    settings_c$ratio_M_main  <- 0.9
    settings_c$ratio_M_nuisance <- 0.9
    settings_c$max_L <- 1024
    settings_c$learning_rate <- 0.025
}
##
fn_make_inits_c <- function(n_chains) {
    lapply( 1:n_chains,
            function(chain_id)
                list( u_raw = matrix(rnorm(6, sd = 0.1), nrow = 2, ncol = 3),
                      pi = rep(0.25, 4),
                      mu = 0.0,
                      sigma = 1.0))
}
##
fn_run_integrator <- function(integrator) {
  
    model_obj <- BayesMVP::MVP_model$new( Model_type = "Stan",
                                          sample_nuisance = TRUE,
                                          Stan_data_list = list(n_u = 2L, n_tests = 3L),
                                          Stan_model_file_path = file.path(MODEL_DIR, "nuisance_present.stan"))
    ##
    model_obj$sample( n_chains_burnin = settings_c$n_chains_burnin,
                      init_lists_per_chain = fn_make_inits_c(settings_c$n_chains_burnin),
                      parallel_method = settings_c$parallel_method,
                      sample_nuisance = TRUE,
                      seed = settings_c$seed,
                      n_burnin = settings_c$n_burnin,
                      n_iter = settings_c$n_iter,
                      n_chains_sampling = settings_c$n_chains_sampling,
                      n_superchains = settings_c$n_superchains,
                      adapt_delta = settings_c$adapt_delta,
                      learning_rate = settings_c$learning_rate,
                      partitioned_HMC = settings_c$partitioned_HMC,
                      diffusion_HMC = settings_c$diffusion_HMC,
                      diffusion_HMC_integrator = integrator,
                      metric_type_main = settings_c$metric_type_main,
                      metric_shape_main = settings_c$metric_shape_main,
                      ratio_M_main = settings_c$ratio_M_main,
                      interval_width_main = settings_c$interval_width_main,
                      metric_type_nuisance = settings_c$metric_type_nuisance,
                      metric_shape_nuisance = settings_c$metric_shape_nuisance,
                      ratio_M_nuisance = settings_c$ratio_M_nuisance,
                      interval_width_nuisance = settings_c$interval_width_nuisance,
                      M_decay_type = settings_c$M_decay_type,
                      M_decay_power = settings_c$M_decay_power,
                      M_decay_scale = settings_c$M_decay_scale,
                      max_tau_main = settings_c$max_tau_main,
                      max_tau_nuisance = settings_c$max_tau_nuisance,
                      max_eps_main = settings_c$max_eps_main,
                      max_eps_nuisance = settings_c$max_eps_nuisance,
                      max_L = settings_c$max_L,
                      use_disk = settings_c$use_disk,
                      n_threads_WCP_burnin = settings_c$n_threads_WCP_burnin,
                      n_threads_WCP_sampling = settings_c$n_threads_WCP_sampling,
                      reorder_cols_MVP = settings_c$reorder_cols_MVP)
    ##
    fit_obj <- model_obj$summary(compute_nested_rhat = FALSE)
    ##
    list( model_obj = model_obj,
          fit_obj = fit_obj,
          summary_main = fit_obj$get_summary_main(),
          draws = fit_obj$get_posterior_draws())
    
}
##
cat("  (C.1) running kick_flow_kick...\n")
run_kfk <- tryCatch( fn_run_integrator(integrator = "kick_flow_kick"), ## BayesMVP-original
                     error = function(e) { cat("  ERROR:", conditionMessage(e), "\n"); NULL })
##
cat("  (C.2) running flow_kick_flow...\n")
run_fkf <- tryCatch( fn_run_integrator(integrator = "flow_kick_flow"), ## Alenlöv, Doucet & Lindsten 2021
                     error = function(e) { cat("  ERROR:", conditionMessage(e), "\n"); NULL })
##
fn_check("C.3 kick_flow_kick run completed", !is.null(run_kfk))
fn_check("C.4 flow_kick_flow run completed", !is.null(run_fkf))
##
if (!is.null(run_kfk) && !is.null(run_fkf)) {
  
      fn_check("C.5 recorded selector: KFK",
               tryCatch( identical(run_kfk$model_obj$result$diffusion_HMC_integrator, "kick_flow_kick"),
                         error = function(e) FALSE))
      fn_check("C.6 recorded selector: FKF",
               tryCatch( identical(run_fkf$model_obj$result$diffusion_HMC_integrator, "flow_kick_flow"),
                         error = function(e) FALSE))
      ##
      names_kfk <- run_kfk$summary_main$parameter
      names_fkf <- run_fkf$summary_main$parameter
      ##
      fn_check("C.7 KFK summary names: pi[1..4], mu, sigma",
               all(c("pi[1]", "pi[2]", "pi[3]", "pi[4]", "mu", "sigma") %in% names_kfk))
      fn_check("C.8 FKF summary names: pi[1..4], mu, sigma",
               all(c("pi[1]", "pi[2]", "pi[3]", "pi[4]", "mu", "sigma") %in% names_fkf))
      ##
      fn_check("C.9 KFK draws all finite",
               all(is.finite(run_kfk$draws)))
      fn_check("C.10 FKF draws all finite",
               all(is.finite(run_fkf$draws)))
      ##
      ## Smoke-level posterior equivalence: both samplers target the SAME distribution,
      ## so each parameter mean should agree within 3 posterior SDs of the KFK reference.
      ## (This is an execution/consistency check, NOT a convergence test.)
      common_params <- intersect(names_kfk, names_fkf)
      ##
      tol_violations <- character(0)
      for (param in common_params) {
          mean_kfk <- run_kfk$summary_main$mean[run_kfk$summary_main$parameter == param]
          sd_kfk   <- run_kfk$summary_main$sd[run_kfk$summary_main$parameter == param]
          mean_fkf <- run_fkf$summary_main$mean[run_fkf$summary_main$parameter == param]
          sd_fkf   <- run_fkf$summary_main$sd[run_fkf$summary_main$parameter == param]
          if (length(mean_kfk) == 1 && length(mean_fkf) == 1 &&
              is.finite(mean_kfk) && is.finite(mean_fkf)) {
              tol <- 3 * max(sd_kfk, sd_fkf, 1e-6)
              if (abs(mean_fkf - mean_kfk) > tol) {
                  tol_violations <- c(tol_violations, sprintf("%s (%.4f vs %.4f)", param, mean_kfk, mean_fkf))
              }
          }
      }
      ##
      fn_check("C.11 posterior means agree within 3 SDs",
               length(tol_violations) == 0)
      if (length(tol_violations) > 0) {
          cat("       violated params:", paste(tol_violations, collapse = "; "), "\n")
      }
      ##
      rm(run_kfk, run_fkf); gc()
      
}
##
## ---- save + summary ---------------------------------------------------------------------
##
cat(sprintf("\n==== test_07: %d passed, %d failed ====\n", n_pass, n_fail))
saveRDS(results, file = file.path(OUT_DIR, "test_07_results.RDS"))
if (n_fail > 0) quit(status = 1)




