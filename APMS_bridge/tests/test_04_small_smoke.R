#### ============================================================================================================================================
## test_04_small_smoke.R
## ----------------------------------------------------------------------------------------------------------------------------------------------
## END-TO-END smoke tests through the REAL package + REAL BridgeStan models.
## Reduced settings come from fixtures/settings_smoke.R ONLY - the application
## settings are untouched. A short run verifies EXECUTION (initialisation, lp/grad,
## sampling loop, trace storage, reconstruction, summaries) - NOT convergence.
##
## Cases:
##   A  nuisance-present model (n_nuisance = 6), partitioned + diffusion HMC
##   B  no-nuisance scalar model (sample_nuisance = FALSE)
##   C  no-nuisance model whose FIRST declaration is a simplex (dimension-changing)
##   D  zero-length FIRST nuisance declaration (sample_nuisance = TRUE -> 0)
##   E  SMALL nuisance block (n = 3, below the old <=10 shortcut)
##   F  nuisance + transformed parameters + generated quantities (naming check)
## ==============================================================================================================================================
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
OUT_DIR <- file.path( TESTS_DIR,
                      "test_output")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
##
source(file = file.path( TESTS_DIR,
                         "fixtures/settings_smoke.R"))
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
cat("\n======== test_04: end-to-end smoke (real package + BridgeStan) ========\n")
##
## ---- (A) nuisance-present model, partitioned + diffusion HMC --------------------------------------------------------------------------------
##
cat("\n(A) nuisance-present model, partitioned + diffusion HMC:\n")
##
settings_a <- fn_smoke_settings()
{
    settings_a$n_iter   <- 250
    settings_a$n_burnin <- 250
    settings_a$partitioned_HMC <- FALSE
    settings_a$diffusion_HMC   <- TRUE
    settings_a$n_chains_burnin   <- 4
    settings_a$n_chains_sampling <- 16
    settings_a$interval_width_main <- 1
    settings_a$ratio_M_main  <- 0.9
    settings_a$ratio_M_nuisance <- 0.9
    settings_a$max_L <- 1024
    settings_a$learning_rate <- 0.025
    ##
    settings_a$n_iter   <- 250
    settings_a$n_burnin <- 250
    ##
    settings_b$partitioned_HMC <- FALSE
}
##
model_a <- BayesMVP::MVP_model$new( Model_type = "Stan",
                                    sample_nuisance = TRUE,
                                    Stan_data_list = list(n_u = 2L, n_tests = 3L),
                                    Stan_model_file_path = file.path(MODEL_DIR, "nuisance_present.stan"))
##
fn_check("A.1 n_nuisance auto-detected = 6", model_a$n_nuisance == 6L)
fn_check("A.2 n_params_main auto-detected = 5", model_a$n_params_main == 5L)
##
model_a$sample( n_chains_burnin = settings_a$n_chains_burnin,
                init_lists_per_chain = lapply( 1:settings_a$n_chains_burnin,
                                                function(chain_id)
                    list( u_raw = matrix(rnorm(6, sd = 0.1), nrow = 2, ncol = 3),
                          pi = rep(0.25, 4),
                          mu = 0.0,
                          sigma = 1.0)),
                parallel_method = settings_a$parallel_method,
                sample_nuisance = TRUE,
                seed = settings_a$seed,
                n_burnin = settings_a$n_burnin,
                n_iter = settings_a$n_iter,
                n_chains_sampling = settings_a$n_chains_sampling,
                n_superchains = settings_a$n_superchains,
                adapt_delta = settings_a$adapt_delta,
                learning_rate = settings_a$learning_rate,
                partitioned_HMC = TRUE,
                diffusion_HMC   = TRUE,
                metric_type_main = settings_a$metric_type_main,
                metric_shape_main = settings_a$metric_shape_main,
                ratio_M_main = settings_a$ratio_M_main,
                interval_width_main = settings_a$interval_width_main,
                metric_type_nuisance = settings_a$metric_type_nuisance,
                metric_shape_nuisance = settings_a$metric_shape_nuisance,
                ratio_M_nuisance = settings_a$ratio_M_nuisance,
                interval_width_nuisance = settings_a$interval_width_nuisance,
                M_decay_type = settings_a$M_decay_type,
                M_decay_power = settings_a$M_decay_power,
                M_decay_scale = settings_a$M_decay_scale,
                max_tau_main = settings_a$max_tau_main,
                max_tau_nuisance = settings_a$max_tau_nuisance,
                max_eps_main = settings_a$max_eps_main,
                max_eps_nuisance = settings_a$max_eps_nuisance,
                max_L = settings_a$max_L,
                use_disk = settings_a$use_disk,
                n_threads_WCP_burnin = settings_a$n_threads_WCP_burnin,
                n_threads_WCP_sampling = settings_a$n_threads_WCP_sampling,
                reorder_cols_MVP = settings_a$reorder_cols_MVP)
##
main_trace_a <- model_a$result$sampling_object[[1]]
##
fn_check("A.3 main trace: 5 params x n_iter per chain", dim(main_trace_a[[1]])[1] == 5L && dim(main_trace_a[[1]])[2] == settings_a$n_iter)
##
nuisance_trace_a <- model_a$result$sampling_object[[3]]
##
fn_check("A.4 nuisance trace stored: 6 x n_iter per chain", length(nuisance_trace_a) == settings_a$n_chains_sampling && dim(nuisance_trace_a[[1]])[1] == 6L)
fn_check("A.5 init_object refreshed after sampling", model_a$n_nuisance == 6L)
##
fit_a <- model_a$summary(compute_nested_rhat = FALSE)
##
names_main_a <- fit_a$get_summary_main()$parameter
##
fn_check("A.6 constrained main names correct (pi[1..4], mu, sigma)",
         all(c("pi[1]", "pi[2]", "pi[3]", "pi[4]", "mu", "sigma") %in% names_main_a))
if (!all(c("pi[1]", "pi[2]", "pi[3]", "pi[4]", "mu", "sigma") %in% names_main_a)) {
    cat("  actual A summary main names:\n")
    print(names_main_a)
}
##
draws_a <- fit_a$get_posterior_draws()
##
fn_check("A.7 draws array has 6 constrained main params", dim(draws_a)[3] == 6L)
##
rm(fit_a); gc()
##
## ---- (B) no-nuisance scalar model -----------------------------------------------------------------------------------------------------------
##
cat("\n(B) no-nuisance scalar model (sample_nuisance = FALSE):\n")
##
settings_b <- fn_smoke_settings()
##
settings_b$n_iter   <- 250
settings_b$n_burnin <- 250
##
settings_b$sample_nuisance <- FALSE
settings_b$partitioned_HMC <- FALSE
settings_b$diffusion_HMC   <- FALSE
##
model_b <- BayesMVP::MVP_model$new( Model_type = "Stan",
                                    sample_nuisance = FALSE,
                                    Stan_data_list = list(),
                                    Stan_model_file_path = file.path(MODEL_DIR, "no_nuisance_scalar.stan"))
##
fn_check("B.1 n_nuisance = 0", model_b$n_nuisance == 0L)
fn_check("B.2 n_params_main = 1", model_b$n_params_main == 1L)
##
model_b$sample( n_chains_burnin = settings_b$n_chains_burnin,
                init_lists_per_chain = lapply(1:settings_b$n_chains_burnin, function(chain_id) list(mu = 0.0)),
                parallel_method = settings_b$parallel_method,
                sample_nuisance = FALSE,
                seed = settings_b$seed,
                n_burnin = settings_b$n_burnin,
                n_iter = settings_b$n_iter,
                n_chains_sampling = settings_b$n_chains_sampling,
                n_superchains = settings_b$n_superchains,
                adapt_delta = settings_b$adapt_delta,
                learning_rate = settings_b$learning_rate,
                partitioned_HMC = FALSE,
                diffusion_HMC = FALSE,
                metric_type_main = settings_b$metric_type_main,
                metric_shape_main = settings_b$metric_shape_main,
                ratio_M_main = settings_b$ratio_M_main,
                interval_width_main = settings_b$interval_width_main,
                metric_type_nuisance = settings_b$metric_type_nuisance,
                metric_shape_nuisance = settings_b$metric_shape_nuisance,
                ratio_M_nuisance = settings_b$ratio_M_nuisance,
                interval_width_nuisance = settings_b$interval_width_nuisance,
                M_decay_type = settings_b$M_decay_type,
                M_decay_power = settings_b$M_decay_power,
                M_decay_scale = settings_b$M_decay_scale,
                max_tau_main = settings_b$max_tau_main,
                max_tau_nuisance = settings_b$max_tau_nuisance,
                max_eps_main = settings_b$max_eps_main,
                max_eps_nuisance = settings_b$max_eps_nuisance,
                max_L = settings_b$max_L,
                use_disk = settings_b$use_disk,
                n_threads_WCP_burnin = settings_b$n_threads_WCP_burnin,
                n_threads_WCP_sampling = settings_b$n_threads_WCP_sampling,
                reorder_cols_MVP = settings_b$reorder_cols_MVP)
##
main_trace_b <- model_b$result$sampling_object[[1]]
##
fn_check("B.3 main trace is 1 x n_iter (every coordinate sampled as main)", dim(main_trace_b[[1]])[1] == 1L)
##
fit_b <- model_b$summary(compute_nested_rhat = FALSE)
##
fn_check("B.4 summary names contain mu", "mu" %in% fit_b$get_summary_main()$parameter)
fn_check("B.5 draws array dim 1", dim(fit_b$get_posterior_draws())[3] == 1L)
##
rm(fit_b); gc()
##
## ---- (C) no-nuisance model with SIMPLEX first declaration -----------------------------------------------------------------------------------
##
cat("\n(C) no-nuisance model, simplex as FIRST declaration (all main):\n")
##
settings_c <- fn_smoke_settings()
settings_c$sample_nuisance <- FALSE
settings_c$partitioned_HMC <- FALSE
settings_c$diffusion_HMC <- FALSE
##
model_c <- BayesMVP::MVP_model$new( Model_type = "Stan",
                                    sample_nuisance = FALSE,
                                    Stan_data_list = list(y = rep(1L, 10)),
                                    Stan_model_file_path = file.path(MODEL_DIR, "no_nuisance_simplex_first.stan"))
##
fn_check("C.1 n_nuisance = 0 (no nuisance block)", model_c$n_nuisance == 0L)
fn_check("C.2 n_params_main = 4 (3 unc simplex + mu)", model_c$n_params_main == 4L)
##
model_c$sample( n_chains_burnin = settings_c$n_chains_burnin,
                init_lists_per_chain = lapply(1:settings_c$n_chains_burnin,
                                              function(chain_id) list(theta = rep(0.25, 4), mu = 0.0)),
                parallel_method = settings_c$parallel_method,
                sample_nuisance = FALSE,
                seed = settings_c$seed,
                n_burnin = settings_c$n_burnin,
                n_iter = settings_c$n_iter,
                n_chains_sampling = settings_c$n_chains_sampling,
                n_superchains = settings_c$n_superchains,
                adapt_delta = settings_c$adapt_delta,
                learning_rate = settings_c$learning_rate,
                partitioned_HMC = FALSE,
                diffusion_HMC = FALSE,
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
fit_c <- model_c$summary(compute_nested_rhat = FALSE)
##
names_c <- fit_c$get_summary_main()$parameter
##
fn_check("C.3 summary has 4 constrained theta names + mu (dimension-changing handled)",
         all(c("theta[1]", "theta[2]", "theta[3]", "theta[4]", "mu") %in% names_c))
fn_check("C.4 draws array dim = 5 constrained names", dim(fit_c$get_posterior_draws())[3] == 5L)
##
rm(fit_c); gc()
##
## ---- (D) ZERO-LENGTH first nuisance declaration ---------------------------------------------------------------------------------------------
##
cat("\n(D) zero-length FIRST nuisance declaration (sample_nuisance = TRUE):\n")
##
settings_d <- fn_smoke_settings()
##
model_d <- BayesMVP::MVP_model$new( Model_type = "Stan",
                                    sample_nuisance = TRUE,
                                    Stan_data_list = list(),
                                    Stan_model_file_path = file.path(MODEL_DIR, "zero_length_first.stan"))
##
fn_check("D.1 n_nuisance = 0 (zero-length declaration detected via stanc)", model_d$n_nuisance == 0L)
fn_check("D.2 n_params_main = 2", model_d$n_params_main == 2L)
##
model_d$sample( n_chains_burnin = settings_d$n_chains_burnin,
                init_lists_per_chain = lapply(1:settings_d$n_chains_burnin,
                                              function(chain_id) list(mu = 0.0, sigma = 1.0)),
                parallel_method = settings_d$parallel_method,
                sample_nuisance = TRUE,   ## nuisance declaration EXISTS, but has zero length
                seed = settings_d$seed,
                n_burnin = settings_d$n_burnin,
                n_iter = settings_d$n_iter,
                n_chains_sampling = settings_d$n_chains_sampling,
                n_superchains = settings_d$n_superchains,
                adapt_delta = settings_d$adapt_delta,
                learning_rate = settings_d$learning_rate,
                partitioned_HMC = TRUE,   ## guard inside R_fn_sample_model switches these
                diffusion_HMC = TRUE,     ## off because n_nuisance == 0
                metric_type_main = settings_d$metric_type_main,
                metric_shape_main = settings_d$metric_shape_main,
                ratio_M_main = settings_d$ratio_M_main,
                interval_width_main = settings_d$interval_width_main,
                metric_type_nuisance = settings_d$metric_type_nuisance,
                metric_shape_nuisance = settings_d$metric_shape_nuisance,
                ratio_M_nuisance = settings_d$ratio_M_nuisance,
                interval_width_nuisance = settings_d$interval_width_nuisance,
                M_decay_type = settings_d$M_decay_type,
                M_decay_power = settings_d$M_decay_power,
                M_decay_scale = settings_d$M_decay_scale,
                max_tau_main = settings_d$max_tau_main,
                max_tau_nuisance = settings_d$max_tau_nuisance,
                max_eps_main = settings_d$max_eps_main,
                max_eps_nuisance = settings_d$max_eps_nuisance,
                max_L = settings_d$max_L,
                use_disk = settings_d$use_disk,
                n_threads_WCP_burnin = settings_d$n_threads_WCP_burnin,
                n_threads_WCP_sampling = settings_d$n_threads_WCP_sampling,
                reorder_cols_MVP = settings_d$reorder_cols_MVP)
##
main_trace_d <- model_d$result$sampling_object[[1]]
##
fn_check("D.3 main trace has ALL 2 coordinates", dim(main_trace_d[[1]])[1] == 2L)
##
fit_d <- model_d$summary(compute_nested_rhat = FALSE)
##
fn_check("D.4 summary names: mu + sigma", all(c("mu", "sigma") %in% fit_d$get_summary_main()$parameter))
##
rm(fit_d); gc()
##
## ---- (E) SMALL nuisance block (n = 3, below the old <=10 shortcut) ---------------------------------------------------------------------------
##
cat("\n(E) small nuisance block (n = 3):\n")
##
settings_e <- fn_smoke_settings()
##
model_e <- BayesMVP::MVP_model$new( Model_type = "Stan",
                                    sample_nuisance = TRUE,
                                    Stan_data_list = list(y = rnorm(20)),
                                    Stan_model_file_path = file.path(MODEL_DIR, "small_nuisance.stan"))
##
fn_check("E.1 n_nuisance = 3 (small block detected)", model_e$n_nuisance == 3L)
fn_check("E.2 n_params_main = 2", model_e$n_params_main == 2L)
##
model_e$sample( n_chains_burnin = settings_e$n_chains_burnin,
                init_lists_per_chain = lapply(1:settings_e$n_chains_burnin,
                                              function(chain_id) list(u = rep(0.0, 3), mu = 0.0, sigma = 1.0)),
                parallel_method = settings_e$parallel_method,
                sample_nuisance = TRUE,
                seed = settings_e$seed,
                n_burnin = settings_e$n_burnin,
                n_iter = settings_e$n_iter,
                n_chains_sampling = settings_e$n_chains_sampling,
                n_superchains = settings_e$n_superchains,
                adapt_delta = settings_e$adapt_delta,
                learning_rate = settings_e$learning_rate,
                partitioned_HMC = TRUE,
                diffusion_HMC = TRUE,
                metric_type_main = settings_e$metric_type_main,
                metric_shape_main = settings_e$metric_shape_main,
                ratio_M_main = settings_e$ratio_M_main,
                interval_width_main = settings_e$interval_width_main,
                metric_type_nuisance = settings_e$metric_type_nuisance,
                metric_shape_nuisance = settings_e$metric_shape_nuisance,
                ratio_M_nuisance = settings_e$ratio_M_nuisance,
                interval_width_nuisance = settings_e$interval_width_nuisance,
                M_decay_type = settings_e$M_decay_type,
                M_decay_power = settings_e$M_decay_power,
                M_decay_scale = settings_e$M_decay_scale,
                max_tau_main = settings_e$max_tau_main,
                max_tau_nuisance = settings_e$max_tau_nuisance,
                max_eps_main = settings_e$max_eps_main,
                max_eps_nuisance = settings_e$max_eps_nuisance,
                max_L = settings_e$max_L,
                use_disk = settings_e$use_disk,
                n_threads_WCP_burnin = settings_e$n_threads_WCP_burnin,
                n_threads_WCP_sampling = settings_e$n_threads_WCP_sampling,
                reorder_cols_MVP = settings_e$reorder_cols_MVP)
##
nuisance_trace_e <- model_e$result$sampling_object[[3]]
##
fn_check("E.3 nuisance trace stored (3 x n_iter)", length(nuisance_trace_e) > 0 && dim(nuisance_trace_e[[1]])[1] == 3L)
##
fit_e <- model_e$summary(compute_nested_rhat = FALSE)
##
fn_check("E.4 summary names: mu + sigma", all(c("mu", "sigma") %in% fit_e$get_summary_main()$parameter))
##
rm(fit_e); gc()
##
## ---- (F) nuisance + transformed parameters + generated quantities ---------------------------------------------------------------------------
##
cat("\n(F) nuisance + tp + gq naming/indexing:\n")
##
settings_f <- fn_smoke_settings()
##
model_f <- BayesMVP::MVP_model$new( Model_type = "Stan",
                                    sample_nuisance = TRUE,
                                    Stan_data_list = list(n_u = 2L, n_tests = 2L),
                                    Stan_model_file_path = file.path(MODEL_DIR, "nuisance_tp_gq.stan"))
##
fn_check("F.1 n_nuisance = 4", model_f$n_nuisance == 4L)
##
model_f$sample( n_chains_burnin = settings_f$n_chains_burnin,
                init_lists_per_chain = lapply(1:settings_f$n_chains_burnin,
                                              function(chain_id)
                    list( u_raw = matrix(rnorm(4, sd = 0.1), nrow = 2, ncol = 2),
                          pi = rep(0.25, 4),
                          mu = 0.0)),
                parallel_method = settings_f$parallel_method,
                sample_nuisance = TRUE,
                seed = settings_f$seed,
                n_burnin = settings_f$n_burnin,
                n_iter = settings_f$n_iter,
                n_chains_sampling = settings_f$n_chains_sampling,
                n_superchains = settings_f$n_superchains,
                adapt_delta = settings_f$adapt_delta,
                learning_rate = settings_f$learning_rate,
                partitioned_HMC = TRUE,
                diffusion_HMC = TRUE,
                metric_type_main = settings_f$metric_type_main,
                metric_shape_main = settings_f$metric_shape_main,
                ratio_M_main = settings_f$ratio_M_main,
                interval_width_main = settings_f$interval_width_main,
                metric_type_nuisance = settings_f$metric_type_nuisance,
                metric_shape_nuisance = settings_f$metric_shape_nuisance,
                ratio_M_nuisance = settings_f$ratio_M_nuisance,
                interval_width_nuisance = settings_f$interval_width_nuisance,
                M_decay_type = settings_f$M_decay_type,
                M_decay_power = settings_f$M_decay_power,
                M_decay_scale = settings_f$M_decay_scale,
                max_tau_main = settings_f$max_tau_main,
                max_tau_nuisance = settings_f$max_tau_nuisance,
                max_eps_main = settings_f$max_eps_main,
                max_eps_nuisance = settings_f$max_eps_nuisance,
                max_L = settings_f$max_L,
                use_disk = settings_f$use_disk,
                n_threads_WCP_burnin = settings_f$n_threads_WCP_burnin,
                n_threads_WCP_sampling = settings_f$n_threads_WCP_sampling,
                reorder_cols_MVP = settings_f$reorder_cols_MVP)
##
fit_f <- model_f$summary(compute_nested_rhat = FALSE)
##
names_main_f <- fit_f$get_summary_main()$parameter
names_tp_f <- fit_f$get_summary_transformed()$parameter
names_gq_f <- fit_f$get_summary_generated_quantities()$parameter
##
fn_check("F.2 main names: pi.1..4 + mu", all(c("pi.1", "pi.2", "pi.3", "pi.4", "mu") %in% names_main_f))
fn_check("F.3 transformed parameter present (mu_tp)", "mu_tp" %in% names_tp_f)
fn_check("F.4 generated quantities present (mu_gq, pi1)", all(c("mu_gq", "pi1") %in% names_gq_f))
fn_check("F.5 nuisance NOT in main summary", !any(grepl("u_raw", names_main_f)))
##
rm(fit_f); gc()
##
## ---- save + summary --------------------------------------------------------------------------------------------------------------------------
##
cat(sprintf("\n==== test_04: %d passed, %d failed ====\n", n_pass, n_fail))
saveRDS(results, file = file.path(OUT_DIR, "test_04_results.RDS"))
if (n_fail > 0) quit(status = 1)







