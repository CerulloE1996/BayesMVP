#### ================================================================================================================================================================
## test_03_bridgestan_init.R
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
## REAL BridgeStan initialisation tests for R_fn_init_initial_values():
##   - COMPLETE init list (nuisance + main): unconstrained, then split by position
##   - PARTIAL init list: completed with model defaults (with a warning/message)
##   - zero-length nuisance declaration: n_nuisance = 0 -> empty nuisance block
##   - no-nuisance model, sample_nuisance = FALSE: ALL params main
##   - round-trip check: param_constrain(nuisance | main) reproduces the inits
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
OUT_DIR <- file.path( TESTS_DIR,
                      "test_output")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
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
cat("\n======== test_03: BridgeStan initialisation (complete + partial) ========\n")
##
## ---- (a) COMPLETE init, nuisance + main --------------------------------------------------------------------------------------------------------------------------
##
bs <- bridgestan::StanModel$new(lib = file.path(MODEL_DIR, "nuisance_present.stan"),
                                data = jsonlite::toJSON(list(n_u = 2L, n_tests = 3L), auto_unbox = TRUE),
                                seed = 1)
## n_nuisance = 6 (2 x 3), main = simplex(3 unc) + mu + sigma = 5 unc
##
init_list_1 <- list( u_raw = matrix( c(0.1, 0.2, 0.3, -0.1, -0.2, -0.3),
                                     nrow = 2,
                                     ncol = 3,
                                     byrow = TRUE),
                     pi = c(0.4, 0.3, 0.2, 0.1),
                     mu = 0.5,
                     sigma = 1.5)
##
outs_a <- BayesMVP:::R_fn_init_initial_values( Model_type = "Stan",
                                               bs_model = bs,
                                               n_chains_burnin = 2,
                                               init_lists_per_chain = list(init_list_1, init_list_1),
                                               sample_nuisance = TRUE,
                                               n_nuisance = 6L,
                                               n_params_main = 5L)
##
cat("\n(a) complete init (n_nuisance = 6, n_params_main = 5):\n")
##
fn_check("a.1 nuisance matrix dims 6 x 2", identical(dim(outs_a$theta_nuisance_vectors_all_chains_input_from_R), c(6L, 2L)))
fn_check("a.2 main matrix dims 5 x 2", identical(dim(outs_a$theta_main_vectors_all_chains_input_from_R), c(5L, 2L)))
fn_check("a.3 complete unconstrained vector length 11", length(outs_a$inits_unconstrained_vec_per_chain[[1]]) == 11L)
##
## round-trip: [nuisance | main] -> constrain -> equals the constrained inits.
## Constrained name order: u_raw (6), pi (4), mu, sigma.
##
theta_full <- c( outs_a$theta_nuisance_vectors_all_chains_input_from_R[, 1],
                 outs_a$theta_main_vectors_all_chains_input_from_R[, 1])
##
con_round_trip <- bs$param_constrain(theta_full, include_tp = FALSE, include_gq = FALSE,
                                     rng = bs$new_rng(seed = 7))
##
fn_check("a.4 round-trip pi[1] == 0.4 (constrained pos 7 after u_raw)", abs(con_round_trip[7] - 0.4) < 1e-10)
fn_check("a.5 round-trip mu == 0.5 (constrained pos 11)", abs(con_round_trip[11] - 0.5) < 1e-8)
##
## ---- (b) PARTIAL init (only mu given) ----------------------------------------------------------------------------------------------------------------------------
##
cat("\n(b) partial init (mu only):\n")
##
init_partial <- list(mu = 0.25)
##
outs_b <- tryCatch(
    BayesMVP:::R_fn_init_initial_values( Model_type = "Stan",
                                         bs_model = bs,
                                         n_chains_burnin = 2,
                                         init_lists_per_chain = list(init_partial, init_partial),
                                         sample_nuisance = TRUE,
                                         n_nuisance = 6L,
                                         n_params_main = 5L),
    error = function(e) { cat("  ERROR:", conditionMessage(e), "\n"); NULL })
##
fn_check("b.1 partial init does not error", !is.null(outs_b))
##
if (!is.null(outs_b)) {
    ##
    theta_b <- c( outs_b$theta_nuisance_vectors_all_chains_input_from_R[, 1],
                  outs_b$theta_main_vectors_all_chains_input_from_R[, 1])
    ##
    con_b <- bs$param_constrain(theta_b, include_tp = FALSE, include_gq = FALSE, rng = bs$new_rng(seed = 7))
    ##
    fn_check("b.2 mu from partial init recovered (constrained pos 11)", abs(con_b[11] - 0.25) < 1e-8)
    fn_check("b.3 missing params completed (finite constrained draw)", all(is.finite(con_b)))
    
}
##
## ---- (c) zero-length nuisance declaration ------------------------------------------------------------------------------------------------------------------------
##
bs_z <- bridgestan::StanModel$new(lib = file.path(MODEL_DIR, "zero_length_first.stan"),
                                  data = "{}",
                                  seed = 1)
##
outs_z <- BayesMVP:::R_fn_init_initial_values( Model_type = "Stan",
                                               bs_model = bs_z,
                                               n_chains_burnin = 2,
                                               init_lists_per_chain = list(list(mu = 0.1, sigma = 2.0),
                                                                            list(mu = -0.1, sigma = 1.0)),
                                               sample_nuisance = TRUE,
                                               n_nuisance = 0L,
                                               n_params_main = 2L)
##
cat("\n(c) zero-length nuisance declaration (n_nuisance = 0):\n")
##
fn_check("c.1 nuisance matrix dims 0 x 2", identical(dim(outs_z$theta_nuisance_vectors_all_chains_input_from_R), c(0L, 2L)))
fn_check("c.2 main matrix dims 2 x 2 (COMPLETE vector is main)", identical(dim(outs_z$theta_main_vectors_all_chains_input_from_R), c(2L, 2L)))
##
## ---- (d) no-nuisance model, sample_nuisance = FALSE --------------------------------------------------------------------------------------------------------------
##
bs_d <- bridgestan::StanModel$new(lib = file.path(MODEL_DIR, "no_nuisance_multi.stan"),
                                  data = "{}",
                                  seed = 1)
##
outs_d <- BayesMVP:::R_fn_init_initial_values( Model_type = "Stan",
                                               bs_model = bs_d,
                                               n_chains_burnin = 2,
                                               init_lists_per_chain = list(list(mu = 0.5, sigma = 1.5, beta = c(0, 0, 0)),
                                                                            list(mu = -0.5, sigma = 1.0, beta = c(1, 1, 1))),
                                               sample_nuisance = FALSE,
                                               n_nuisance = 0L,
                                               n_params_main = 5L)
##
cat("\n(d) no-nuisance model (sample_nuisance = FALSE):\n")
##
fn_check("d.1 nuisance matrix dims 0 x 2", identical(dim(outs_d$theta_nuisance_vectors_all_chains_input_from_R), c(0L, 2L)))
fn_check("d.2 main matrix dims 5 x 2 (ALL params main)", identical(dim(outs_d$theta_main_vectors_all_chains_input_from_R), c(5L, 2L)))
##
## ---- save + summary --------------------------------------------------------------------------------------------------------------------------------------------
##
cat(sprintf("\n==== test_03: %d passed, %d failed ====\n", n_pass, n_fail))
saveRDS(results, file = file.path(OUT_DIR, "test_03_results.RDS"))
if (n_fail > 0) quit(status = 1)
