#### ================================================================================================================================================================
## test_06_lp_grad.R
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
## Verifies the external-Stan likelihood/gradient path with a meaningful
## numerical comparison: BayesMVP:::Rcpp_wrapper_fn_lp_grad() (the C++ BridgeStan
## path that sampling goes through) must agree with bridgestan::StanModel
## $log_density_gradient() on the COMPLETE [nuisance | main] vector, including
## for SMALL nuisance blocks (the old n_nuisance <= 10 shortcut corrupted these)
## and for models without a nuisance block (n_nuisance = 0).
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
cat("\n======== test_06: lp/grad path vs BridgeStan ========\n")
##
fn_build_model_args <- function(model_so_file,
                                json_file,
                                n_nuisance,
                                n_params_main) {
  
    ## the same shape initialise_model() builds for external Stan models (the C++
    ## list->struct conversion reads N / n_nuisance / n_params_main /
    ## n_binary_tests / n_ordinal_tests / model_so_file / json_file_path
    ## unconditionally):
    list( n_params_main = n_params_main,
          n_nuisance = n_nuisance,
          N = 100,
          n_tests = 5,
          n_class = 2,
          y = array(0, dim = c(100, 5)),
          n_binary_tests = 0L,
          n_ordinal_tests = 0L,
          json_file_path = json_file,
          model_so_file = model_so_file)
    
}
##
## ---- (a) SMALL nuisance block (n = 3) ----------------------------------------------------------------------------------------------------------------------------
##
cat("\n(a) small nuisance block (n = 3):\n")
##
data_a <- list(y = rnorm(20))
##
json_a <- BayesMVP:::convert_stan_data_list_to_JSON(stan_data_list = data_a)
##
## compile the reference model with the EXACT SAME JSON string the C++ loader uses
## (avoids serialisation-precision differences between jsonlite and cmdstanr):
bs_a <- bridgestan::StanModel$new(lib = file.path(MODEL_DIR, "small_nuisance.stan"),
                                  data = paste(readLines(json_a), collapse = ""),
                                  seed = 1)
##
so_a <- BayesMVP:::transform_stan_path(file.path(MODEL_DIR, "small_nuisance.stan"))
##
model_args_a <- fn_build_model_args( so_a,
                                     json_a,
                                     n_nuisance = 3L,
                                     n_params_main = 2L)
##
set.seed(7)
##
theta_us <- matrix(rnorm(3 * 3), nrow = 3, ncol = 3)
theta_main <- matrix(rnorm(2 * 3), nrow = 2, ncol = 3)
##
for (ii in 1:3) {
    ##
    full_theta <- c(theta_us[, ii], theta_main[, ii])
    ##
    ## BayesMVP C++ path (grad_option = "all"):
    lp_grad_mvp <- BayesMVP:::Rcpp_wrapper_fn_lp_grad( Model_type = "Stan",
                                                       force_autodiff = FALSE,
                                                       force_PartialLog = FALSE,
                                                       multi_attempts = FALSE,
                                                       theta_main_vec = theta_main[, ii, drop = FALSE],
                                                       theta_us_vec = theta_us[, ii, drop = FALSE],
                                                       y = matrix(0, 100, 5),
                                                       grad_option = "all",
                                                       Model_args_as_Rcpp_List = model_args_a)
    ##
    ## BridgeStan reference:
    bs_outs <- bs_a$log_density_gradient(full_theta, propto = TRUE, jacobian = TRUE)
    ##
    ## lp_grad_outs layout: [lp | grad_us | grad_main | log_lik]
    lp_diff <- abs(lp_grad_mvp[1] - bs_outs$val)
    grad_diff <- max(abs(lp_grad_mvp[2:6] - bs_outs$gradient))
    ##
    cat(sprintf("    draw %d: |dlp| = %.3e, max|dgrad| = %.3e\n", ii, lp_diff, grad_diff))
    ##
    fn_check(sprintf("a.%d lp matches (small nuisance)", ii), lp_diff < 1e-8)
    fn_check(sprintf("a.%d gradient matches (small nuisance)", ii), grad_diff < 1e-8)
    
}
##
## ---- (b) larger nuisance block (n = 12 > 10, regression) --------------------------------------------------------------------------------------------------------
##
cat("\n(b) larger nuisance block (n = 12):\n")
##
data_b <- list(n_u = 4L, n_tests = 3L)
##
json_b <- BayesMVP:::convert_stan_data_list_to_JSON(stan_data_list = data_b)
##
bs_b <- bridgestan::StanModel$new(lib = file.path(MODEL_DIR, "nuisance_present.stan"),
                                  data = paste(readLines(json_b), collapse = ""),
                                  seed = 1)
##
so_b <- BayesMVP:::transform_stan_path(file.path(MODEL_DIR, "nuisance_present.stan"))
##
model_args_b <- fn_build_model_args( so_b,
                                     json_b,
                                     n_nuisance = 12L,
                                     n_params_main = 5L)
##
set.seed(8)
##
theta_us_b <- matrix(rnorm(12 * 2), nrow = 12, ncol = 2)
theta_main_b <- matrix(rnorm(5 * 2), nrow = 5, ncol = 2)
##
for (ii in 1:2) {
    ##
    full_theta_b <- c(theta_us_b[, ii], theta_main_b[, ii])
    ##
    lp_grad_b <- BayesMVP:::Rcpp_wrapper_fn_lp_grad( Model_type = "Stan",
                                                     force_autodiff = FALSE,
                                                     force_PartialLog = FALSE,
                                                     multi_attempts = FALSE,
                                                     theta_main_vec = theta_main_b[, ii, drop = FALSE],
                                                     theta_us_vec = theta_us_b[, ii, drop = FALSE],
                                                     y = matrix(0, 100, 5),
                                                     grad_option = "all",
                                                     Model_args_as_Rcpp_List = model_args_b)
    ##
    bs_outs_b <- bs_b$log_density_gradient(full_theta_b, propto = TRUE, jacobian = TRUE)
    ##
    lp_diff_b <- abs(lp_grad_b[1] - bs_outs_b$val)
    grad_diff_b <- max(abs(lp_grad_b[2:18] - bs_outs_b$gradient))
    ##
    cat(sprintf("    draw %d: |dlp| = %.3e, max|dgrad| = %.3e\n", ii, lp_diff_b, grad_diff_b))
    ##
    fn_check(sprintf("b.%d lp matches", ii), lp_diff_b < 1e-8)
    fn_check(sprintf("b.%d gradient matches", ii), grad_diff_b < 1e-8)
    
}
##
## ---- (c) no-nuisance model (n_nuisance = 0) ---------------------------------------------------------------------------------------------------------------------
##
cat("\n(c) no-nuisance model (n_nuisance = 0):\n")
##
data_c <- list()
##
json_c <- BayesMVP:::convert_stan_data_list_to_JSON(stan_data_list = data_c)
##
bs_c <- bridgestan::StanModel$new(lib = file.path(MODEL_DIR, "no_nuisance_multi.stan"),
                                  data = paste(readLines(json_c), collapse = ""),
                                  seed = 1)
##
so_c <- BayesMVP:::transform_stan_path(file.path(MODEL_DIR, "no_nuisance_multi.stan"))
##
model_args_c <- fn_build_model_args( so_c,
                                     json_c,
                                     n_nuisance = 0L,
                                     n_params_main = 5L)
##
theta_main_c <- matrix(rnorm(5), nrow = 5, ncol = 1)
##
lp_grad_c <- BayesMVP:::Rcpp_wrapper_fn_lp_grad( Model_type = "Stan",
                                                 force_autodiff = FALSE,
                                                 force_PartialLog = FALSE,
                                                 multi_attempts = FALSE,
                                                 theta_main_vec = theta_main_c,
                                                 theta_us_vec = matrix(numeric(0), nrow = 0, ncol = 1),
                                                 y = matrix(0, 100, 5),
                                                 grad_option = "all",
                                                 Model_args_as_Rcpp_List = model_args_c)
##
bs_outs_c <- bs_c$log_density_gradient(as.numeric(theta_main_c), propto = TRUE, jacobian = TRUE)
##
lp_diff_c <- abs(lp_grad_c[1] - bs_outs_c$val)
grad_diff_c <- max(abs(lp_grad_c[2:6] - bs_outs_c$gradient))
##
cat(sprintf("    |dlp| = %.3e, max|dgrad| = %.3e\n", lp_diff_c, grad_diff_c))
##
fn_check("c.1 lp matches (no nuisance)", lp_diff_c < 1e-8)
fn_check("c.2 gradient matches (no nuisance)", grad_diff_c < 1e-8)
##
## ---- save + summary --------------------------------------------------------------------------------------------------------------------------------------------
##
cat(sprintf("\n==== test_06: %d passed, %d failed ====\n", n_pass, n_fail))
saveRDS(results, file = file.path(OUT_DIR, "test_06_results.RDS"))
if (n_fail > 0) quit(status = 1)
