#### ================================================================================================================================================================
## test_05_reconstruction.R
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
## Checks the constrained-draw reconstruction against BridgeStan on KNOWN
## unconstrained draws: split each complete draw into [nuisance | main] the way
## the sampler stores them, run BayesMVP:::fn_compute_param_constrain_from_trace_parallel(),
## and compare against bs_model$param_constrain() on the COMPLETE vector.
## Also checks the length validation for a main-only vector when the model
## requires nuisance + main, and the no-nuisance (n_nuisance = 0) path.
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
cat("\n======== test_05: constrain reconstruction vs BridgeStan ========\n")
##
## ---- model: nuisance_tp_gq (n_nuisance = 4 unc, main = 4 unc; constrained+tp+gq = 8) ---------------------------------------------------------------------------
##
data_list_tp_gq <- list(n_u = 2L, n_tests = 2L)
##
json_file <- BayesMVP:::convert_stan_data_list_to_JSON(stan_data_list = data_list_tp_gq)
##
bs <- bridgestan::StanModel$new(lib = file.path(MODEL_DIR, "nuisance_tp_gq.stan"),
                                data = paste(readLines(json_file), collapse = ""),
                                seed = 1)
##
n_nuisance <- 4L
n_params_main <- 4L
n_params_full <- length(bs$param_names(include_tp = TRUE, include_gq = TRUE))  ## 8
##
## ---- known complete unconstrained draws (3 draws, 2 chains):
##
set.seed(42)
##
n_iter <- 3
n_chains <- 2
##
main_trace <- list()
nuisance_trace <- list()
reference_draws <- list()
##
for (chain_id in 1:n_chains) {
    ##
    draw_mat <- matrix( rnorm(n = n_iter * (n_nuisance + n_params_main), sd = 0.5),
                        nrow = n_iter)
    ##
    nuisance_trace[[chain_id]] <- t(draw_mat[, 1:n_nuisance])                                    ## n_nuisance x n_iter
    main_trace[[chain_id]] <- t(draw_mat[, (n_nuisance + 1):(n_nuisance + n_params_main)])       ## main x n_iter
    reference_draws[[chain_id]] <- draw_mat
    
}
##
## ---- json for the C++ loader -------------------------------------------------------------------------------------------------------------------------------------
##
model_so_file <- BayesMVP:::transform_stan_path(file.path(MODEL_DIR, "nuisance_tp_gq.stan"))
##
pars_indicies_to_track <- 0:(n_params_full - 1)
##
cat("running fn_compute_param_constrain_from_trace_parallel (RAM)...\n")
##
outs <- BayesMVP:::fn_compute_param_constrain_from_trace_parallel(
    unc_params_trace_input_main = main_trace,
    unc_params_trace_input_nuisance = nuisance_trace,
    pars_indicies_to_track = pars_indicies_to_track,
    n_params_full = n_params_full,
    n_params_main = n_params_main,
    n_nuisance = n_nuisance,
    model_so_file = model_so_file,
    json_file_path = json_file,
    use_disk = FALSE)
##
fn_check("1. output has one matrix per chain", length(outs) == n_chains)
fn_check("2. output dims n_params_full x n_iter", dim(outs[[1]])[1] == n_params_full && dim(outs[[1]])[2] == n_iter)
##
max_abs_diff <- 0
##
for (chain_id in 1:n_chains) {
    for (ii in 1:n_iter) {
        ##
        ref <- bs$param_constrain( reference_draws[[chain_id]][ii, ],
                                   include_tp = TRUE,
                                   include_gq = TRUE,
                                   rng = bs$new_rng(seed = 123 + chain_id))
        ##
        max_abs_diff <- max(max_abs_diff, max(abs(outs[[chain_id]][, ii] - ref)))
        
    }
}
##
cat(sprintf("    max |reconstruction - BridgeStan constrain| = %.3e\n", max_abs_diff))
##
fn_check("3. reconstruction matches BridgeStan constrain (exact)", max_abs_diff < 1e-10)
##
## ---- disk mode ---------------------------------------------------------------------------------------------------------------------------------------------------
##
cat("running fn_compute_param_constrain_from_trace_parallel (disk)...\n")
##
trace_dir_disk <- tempfile(pattern = "constrain_disk_test_")
##
outs_disk <- BayesMVP:::fn_compute_param_constrain_from_trace_parallel(
    unc_params_trace_input_main = main_trace,
    unc_params_trace_input_nuisance = nuisance_trace,
    pars_indicies_to_track = pars_indicies_to_track,
    n_params_full = n_params_full,
    n_params_main = n_params_main,
    n_nuisance = n_nuisance,
    model_so_file = model_so_file,
    json_file_path = json_file,
    use_disk = TRUE,
    trace_dir = trace_dir_disk)
##
read_binary_trace <- function(filepath,
                              n_params,
                              n_iter) {
  
    con <- file(filepath, "rb")
    data <- readBin(con, "double", n = n_params * n_iter)
    close(con)
    matrix(data, nrow = n_params, ncol = n_iter)
    
}
##
disk_trace <- read_binary_trace( filepath = file.path(trace_dir_disk, "chain_0_constrained.bin"),
                                 n_params = n_params_full,
                                 n_iter = n_iter)
##
fn_check("4. disk output matches the RAM output", max(abs(disk_trace - outs[[1]])) < 1e-12)
##
## ---- error path: main-only vector for a model that needs nuisance + main -----------------------------------------------------------------------------------------
##
cat("checking the length validation (main-only vector must not continue)...\n")
##
error_raised <- tryCatch({
    BayesMVP:::fn_compute_param_constrain_from_trace_parallel(
        unc_params_trace_input_main = main_trace,
        unc_params_trace_input_nuisance = list(),
        pars_indicies_to_track = pars_indicies_to_track,
        n_params_full = n_params_full,
        n_params_main = n_params_main,
        n_nuisance = n_nuisance,
        model_so_file = model_so_file,
        json_file_path = json_file,
        use_disk = FALSE)
    FALSE
}, error = function(e) TRUE)
##
fn_check("5. missing nuisance trace raises an actionable error", error_raised)
##
## ---- no-nuisance model through the same reconstruction path ------------------------------------------------------------------------------------------------------
##
json_file_0 <- BayesMVP:::convert_stan_data_list_to_JSON(stan_data_list = list())
##
bs_0 <- bridgestan::StanModel$new(lib = file.path(MODEL_DIR, "no_nuisance_scalar.stan"),
                                  data = paste(readLines(json_file_0), collapse = ""),
                                  seed = 1)
##
main_trace_0 <- list(matrix(rnorm(5, sd = 0.5), nrow = 1))
##
model_so_file_0 <- BayesMVP:::transform_stan_path(file.path(MODEL_DIR, "no_nuisance_scalar.stan"))
##
outs_0 <- BayesMVP:::fn_compute_param_constrain_from_trace_parallel(
    unc_params_trace_input_main = main_trace_0,
    unc_params_trace_input_nuisance = list(),
    pars_indicies_to_track = 0:0,
    n_params_full = 1L,
    n_params_main = 1L,
    n_nuisance = 0L,
    model_so_file = model_so_file_0,
    json_file_path = json_file_0,
    use_disk = FALSE)
##
ref_0 <- bs_0$param_constrain( as.numeric(main_trace_0[[1]][, 1]),
                               include_tp = TRUE,
                               include_gq = TRUE,
                               rng = bs_0$new_rng(seed = 123))
##
fn_check("6. no-nuisance reconstruction matches (n_nuisance = 0)", max(abs(outs_0[[1]][, 1] - ref_0)) < 1e-12)
##
## ---- save + summary --------------------------------------------------------------------------------------------------------------------------------------------
##
cat(sprintf("\n==== test_05: %d passed, %d failed ====\n", n_pass, n_fail))
saveRDS(results, file = file.path(OUT_DIR, "test_05_results.RDS"))
if (n_fail > 0) quit(status = 1)
