#### ================================================================================================================================================================
## test_01_detect_nuisance.R
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
## Tests the nuisance-block DETECTION for external Stan models:
##   - nuisance-present (first declaration = nuisance, correct unconstrained dims)
##   - dimension-changing first declaration (simplex: 4 constrained names, 3 unc)
##   - zero-length FIRST nuisance declaration followed by non-empty main
##   - no-nuisance models (sample_nuisance = FALSE -> 0)
##   - small nuisance block (size 3, below the old n_nuisance <= 10 shortcut)
##
## Uses REAL BridgeStan compilation + stanc metadata (no mocks).
## ================================================================================================================================================================
##
suppressPackageStartupMessages({
    ## RcppParallel must be loaded first: BayesMVP's .so references
    ## RcppParallel::tbbParallelFor, which resolves from RcppParallel's own DLL
    ## once it is in the session (the source Makevars also links it directly for
    ## the next fresh reinstall).
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
fn_compile <- function(stan_file,
                       data_list) {
  
    ## an EMPTY data list must be sent as an empty JSON OBJECT ({}) - jsonlite
    ## would return an empty ARRAY ([]), which BridgeStan rejects:
    json <- if (length(data_list) == 0) "{}" else jsonlite::toJSON(data_list, auto_unbox = TRUE)
    bridgestan::StanModel$new(lib = file.path(MODEL_DIR, stan_file),
                              data = json,
                              seed = 1)
    
}
##
cat("\n======== test_01: nuisance-block detection ========\n")
##
## ---- (a) nuisance present --------------------------------------------------------------------------------------------------------------------------------------
##
bs_a <- fn_compile(stan_file = "nuisance_present.stan",
                   data_list = list(n_u = 5L, n_tests = 3L))
##
Stan_model_file_path <- file.path(MODEL_DIR, "nuisance_present.stan")
##
info_a <- BayesMVP:::detect_nuisance_block_info_from_stan_model( bs_model = bs_a,
                                                                 Stan_model_file_path = Stan_model_file_path)
cat("\n(a) nuisance_present (n_u=5, n_tests=3):\n")
##
cat(sprintf("    n_nuisance = %d, names_constrained = %d, base = %s\n",
            info_a$n_nuisance, info_a$n_nuisance_names_constrained, info_a$nuisance_base_name))
##
fn_check("a.1 first declared base is u_raw", info_a$nuisance_base_name == "u_raw")
fn_check("a.2 n_nuisance = 15 (5*3, unconstrained)", info_a$n_nuisance == 15L)
fn_check("a.3 n_nuisance_names_constrained = 15", info_a$n_nuisance_names_constrained == 15L)
fn_check("a.4 param_unc_num = 15 + 3 + 2", bs_a$param_unc_num() == 20L)
##
fn_check("a.5 detect() wrapper matches", BayesMVP:::detect_n_nuisance_from_stan_model(bs_model = bs_a,
                                                                                      sample_nuisance = TRUE,
                                                                                      Stan_model_file_path = Stan_model_file_path) == 15L)
##
fn_check("a.6 sample_nuisance=FALSE -> 0", BayesMVP:::detect_n_nuisance_from_stan_model(bs_model = bs_a,
                                                                                        sample_nuisance = FALSE) == 0L)
##
## ---- (b) simplex as FIRST declared parameter (dimension-changing) ----------------------------------------------------------------------------------------------
##
bs_b <- fn_compile(stan_file = "no_nuisance_simplex_first.stan",
                   data_list = list(y = rep(1L, 10)))
##
Stan_model_file_path <- file.path(MODEL_DIR, "no_nuisance_simplex_first.stan")
##
info_b <- BayesMVP:::detect_nuisance_block_info_from_stan_model( bs_model = bs_b,
                                                                 Stan_model_file_path = Stan_model_file_path)
cat("\n(b) simplex-first declaration:\n")
cat(sprintf("    n_nuisance = %d, names_constrained = %d, base = %s\n",
            info_b$n_nuisance, info_b$n_nuisance_names_constrained, info_b$nuisance_base_name))
fn_check("b.1 base is theta", info_b$nuisance_base_name == "theta")
fn_check("b.2 n_nuisance = 3 (unconstrained simplex)", info_b$n_nuisance == 3L)
fn_check("b.3 names_constrained = 4", info_b$n_nuisance_names_constrained == 4L)
##
## ---- (c) ZERO-LENGTH first nuisance declaration ----------------------------------------------------------------------------------------------------------------
##
bs_c <- fn_compile(stan_file = "zero_length_first.stan",
                   data_list = list())
##
Stan_model_file_path <- file.path(MODEL_DIR, "zero_length_first.stan")
##
info_c <- BayesMVP:::detect_nuisance_block_info_from_stan_model( bs_model = bs_c,
                                                                 Stan_model_file_path = Stan_model_file_path)
cat("\n(c) zero-length FIRST declaration (u_raw empty, then mu/sigma):\n")
cat(sprintf("    n_nuisance = %d, names_constrained = %d, base = %s (first unc name would be 'mu')\n",
            info_c$n_nuisance, info_c$n_nuisance_names_constrained, info_c$nuisance_base_name))
fn_check("c.1 stanc metadata finds the FIRST DECLARED param (u_raw), not the first name",
         info_c$nuisance_base_name == "u_raw")
fn_check("c.2 n_nuisance = 0 (zero-length declaration)", info_c$n_nuisance == 0L)
fn_check("c.3 names_constrained = 0", info_c$n_nuisance_names_constrained == 0L)
fn_check("c.4 param_unc_num = 2 (mu + sigma)", bs_c$param_unc_num() == 2L)
##
## ---- (d) no-nuisance model (scalar) ----------------------------------------------------------------------------------------------------------------------------
##
bs_d <- fn_compile(stan_file = "no_nuisance_scalar.stan",
                   data_list = list())
cat("\n(d) no-nuisance scalar model:\n")
fn_check("d.1 sample_nuisance=FALSE -> n_nuisance 0",
         BayesMVP:::detect_n_nuisance_from_stan_model(bs_model = bs_d, sample_nuisance = FALSE) == 0L)
fn_check("d.2 param_unc_num = 1", bs_d$param_unc_num() == 1L)
##
## ---- (e) no-nuisance multi-declaration model -------------------------------------------------------------------------------------------------------------------
##
bs_e <- fn_compile(stan_file = "no_nuisance_multi.stan",
                   data_list = list())
cat("\n(e) no-nuisance multi-declaration model:\n")
fn_check("e.1 sample_nuisance=FALSE -> 0", BayesMVP:::detect_n_nuisance_from_stan_model(bs_model = bs_e, sample_nuisance = FALSE) == 0L)
fn_check("e.2 param_unc_num = 5 (mu + sigma + 3 beta)", bs_e$param_unc_num() == 5L)
##
## ---- (f) SMALL nuisance block (n = 3, below the old <=10 shortcut) ---------------------------------------------------------------------------------------------
##
bs_f <- fn_compile(stan_file = "small_nuisance.stan",
                   data_list = list(y = rnorm(20)))
##
Stan_model_file_path <- file.path(MODEL_DIR, "small_nuisance.stan")
##
info_f <- BayesMVP:::detect_nuisance_block_info_from_stan_model( bs_model = bs_f,
                                                                 Stan_model_file_path = Stan_model_file_path)
cat("\n(f) small nuisance block (size 3):\n")
fn_check("f.1 base is u", info_f$nuisance_base_name == "u")
fn_check("f.2 n_nuisance = 3", info_f$n_nuisance == 3L)
fn_check("f.3 param_unc_num = 5", bs_f$param_unc_num() == 5L)
##
## ---- save + summary --------------------------------------------------------------------------------------------------------------------------------------------
##
cat(sprintf("\n==== test_01: %d passed, %d failed ====\n", n_pass, n_fail))
saveRDS(results, file = file.path(OUT_DIR, "test_01_results.RDS"))
if (n_fail > 0) quit(status = 1)
