


#' Compute Se/Sp at custom baseline covariate values for LC-MVOP models.
#'
#' Mirrors MetaOrdDTA::R_fn_using_Rcpp_compute_MR_Se_Sp_AUC_baseline
#' but adapted for the LC-MVOP (latent class multivariate probit) parameter structure.
#'
#' No between-study random effects => no predictive intervals (single-study LC model).
#' All heavy computation (Se/Sp, summaries, AUC, AUC diffs) done in Eigen/C++.
#'
#' @param trace_beta  3D array [n_iter, n_chains, n_beta_params] with dimnames on 3rd dim
#' @param trace_C     3D array [n_iter, n_chains, n_C_params]    with dimnames on 3rd dim
#' @param baseline_case_nd  List of n_tests vectors: baseline covariate values for non-diseased class
#' @param baseline_case_d   List of n_tests vectors: baseline covariate values for diseased class
#' @param n_covs_per_outcome  Integer matrix [n_class, n_tests] giving # covariates per class per test
#' @param n_binary_tests  Integer
#' @param n_ordinal_tests Integer
#' @param n_thr_per_ord_test  Integer vector (length n_ordinal_tests) of thresholds per ordinal test
#' @param C_param_name  Character: "C_vec" (default)
#' @param test_names  Optional character vector of test names (length n_tests)
#' @return List with Se_baseline, Sp_baseline, Fp_baseline arrays and summary tibbles
#' @export
R_fn_compute_LC_MVOP_Se_Sp_baseline <- function( trace_beta,
                                                trace_C,
                                                baseline_case_nd,
                                                baseline_case_d,
                                                n_covs_per_outcome,   ## [n_class, n_tests] matrix
                                                n_binary_tests,
                                                n_ordinal_tests,
                                                n_thr_per_ord_test,
                                                C_param_name = "C_vec",
                                                test_names = NULL,
                                                debugging = FALSE
) {
  
        require(tibble)
        
        n_tests <- n_binary_tests + n_ordinal_tests
        
        ## ---- Validate ----
        stopifnot(is.array(trace_beta) && length(dim(trace_beta)) == 3)
        stopifnot(is.array(trace_C)    && length(dim(trace_C))    == 3)
        stopifnot(length(baseline_case_nd) == n_tests)
        stopifnot(length(baseline_case_d)  == n_tests)
        
        n_iter   <- dim(trace_beta)[1]
        n_chains <- dim(trace_beta)[2]
        
        ## n_covs_per_outcome is [n_class, n_tests]: class 1 = nd, class 2 = d
        n_covs_nd <- as.integer(n_covs_per_outcome[1, ])
        n_covs_d  <- as.integer(n_covs_per_outcome[2, ])
        
        if (debugging) {
          cat("n_tests:", n_tests, "\n")
          cat("n_binary_tests:", n_binary_tests, "\n")
          cat("n_ordinal_tests:", n_ordinal_tests, "\n")
          cat("n_thr_per_ord_test:", n_thr_per_ord_test, "\n")
          cat("n_covs_nd:", n_covs_nd, "\n")
          cat("n_covs_d:", n_covs_d, "\n")
          cat("beta param names (first 20):", head(dimnames(trace_beta)[[3]], 20), "\n")
          cat("C param names (first 20):",    head(dimnames(trace_C)[[3]], 20), "\n")
        }
        
        ## ---- Call Eigen-based Rcpp function ----
        rcpp_out <- Rcpp_compute_LC_MVOP_Se_Sp_baseline( trace_beta_flat    = as.vector(trace_beta),
                                                        beta_dims          = dim(trace_beta),
                                                        beta_names         = dimnames(trace_beta)[[3]],
                                                        trace_C_flat       = as.vector(trace_C),
                                                        C_dims             = dim(trace_C),
                                                        C_names            = dimnames(trace_C)[[3]],
                                                        baseline_case_nd   = baseline_case_nd,
                                                        baseline_case_d    = baseline_case_d,
                                                        n_covs_nd          = n_covs_nd,
                                                        n_covs_d           = n_covs_d,
                                                        n_binary_tests     = n_binary_tests,
                                                        n_ordinal_tests    = n_ordinal_tests,
                                                        n_thr_per_ord_test = n_thr_per_ord_test,
                                                        C_param_name       = C_param_name)
        
        ## ---- Reshape 4D arrays and add dimnames ----
        n_thr_max <- rcpp_out$n_thr_max
        n_thr_vec <- rcpp_out$n_thr
        
        Se_baseline <- array(rcpp_out$Se_baseline, dim = c(n_iter, n_chains, n_tests, n_thr_max))
        Sp_baseline <- array(rcpp_out$Sp_baseline, dim = c(n_iter, n_chains, n_tests, n_thr_max))
        Fp_baseline <- array(rcpp_out$Fp_baseline, dim = c(n_iter, n_chains, n_tests, n_thr_max))
        
        if (is.null(test_names)) {
          test_names <- c(
            if (n_binary_tests > 0)  paste0("Bin", 1:n_binary_tests)  else character(0),
            if (n_ordinal_tests > 0) paste0("Ord", 1:n_ordinal_tests) else character(0)
          )
        }
        
        dn <- list(
          iteration = as.character(1:n_iter),
          chain     = as.character(1:n_chains),
          test      = test_names,
          threshold = paste0("Thr", 1:n_thr_max)
        )
        dimnames(Se_baseline) <- dn
        dimnames(Sp_baseline) <- dn
        dimnames(Fp_baseline) <- dn
        
        ## ---- Convert C++ summary matrices -> tibbles with test/threshold labels ----
        thresh_map <- rcpp_out$thresh_mapping  ## [n_slots, 2]: test, threshold
        
        make_summary_tibble <- function(summary_mat, 
                                        prefix) {
              
              tibble::tibble(
                variable = paste0(prefix, "[", thresh_map[,1], ",", thresh_map[,2], "]"),
                mean     = summary_mat[, 1],
                sd       = summary_mat[, 2],
                `2.5%`   = summary_mat[, 3],
                `50%`    = summary_mat[, 4],
                `97.5%`  = summary_mat[, 5]
              )
          
        }
        
        Se_summaries <- make_summary_tibble(rcpp_out$Se_summary, "Se_baseline")
        Sp_summaries <- make_summary_tibble(rcpp_out$Sp_summary, "Sp_baseline")
        Fp_summaries <- make_summary_tibble(rcpp_out$Fp_summary, "Fp_baseline")
        
        ## ---- AUC outputs ----
        AUC_baseline     <- NULL
        AUC_summary      <- NULL
        AUC_diff         <- NULL
        AUC_diff_summary <- NULL
        
        if (n_ordinal_tests > 0 && !is.null(rcpp_out$AUC_baseline)) {
          
          AUC_baseline <- array(rcpp_out$AUC_baseline, dim = c(n_iter, n_chains, n_ordinal_tests))
          ord_names <- test_names[(n_binary_tests + 1):n_tests]
          dimnames(AUC_baseline) <- list(
            iteration = as.character(1:n_iter),
            chain     = as.character(1:n_chains),
            test      = ord_names)
          
          AUC_summary <- tibble::tibble(
            test   = ord_names,
            mean   = rcpp_out$AUC_summary[, 1],
            sd     = rcpp_out$AUC_summary[, 2],
            `2.5%` = rcpp_out$AUC_summary[, 3],
            `50%`  = rcpp_out$AUC_summary[, 4],
            `97.5%`= rcpp_out$AUC_summary[, 5])
          
          if (!is.null(rcpp_out$AUC_diff)) {
            
                n_pairs <- n_ordinal_tests * (n_ordinal_tests - 1) / 2
                AUC_diff <- array(rcpp_out$AUC_diff, dim = c(n_iter, n_chains, n_pairs))
                
                ## Build pair labels
                pair_labels <- character(n_pairs)
                p <- 1
                for (t1 in 1:(n_ordinal_tests - 1)) {
                  for (t2 in (t1 + 1):n_ordinal_tests) {
                    pair_labels[p] <- paste0(ord_names[t1], " - ", ord_names[t2])
                    p <- p + 1
                  }
                }
                
                AUC_diff_summary <- tibble::tibble(
                  comparison = pair_labels,
                  mean       = rcpp_out$AUC_diff_summary[, 1],
                  sd         = rcpp_out$AUC_diff_summary[, 2],
                  `2.5%`     = rcpp_out$AUC_diff_summary[, 3],
                  `50%`      = rcpp_out$AUC_diff_summary[, 4],
                  `97.5%`    = rcpp_out$AUC_diff_summary[, 5],
                  prob_gt_0  = rcpp_out$AUC_diff_summary[, 6],
                  prob_lt_0  = rcpp_out$AUC_diff_summary[, 7]
                )
            
          }
          
        }
        
        ## ---- Return ----
        list(
          Se_baseline = Se_baseline,
          Sp_baseline = Sp_baseline,
          Fp_baseline = Fp_baseline,
          ##
          Se_summaries = Se_summaries,
          Sp_summaries = Sp_summaries,
          Fp_summaries = Fp_summaries,
          ##
          AUC_baseline     = AUC_baseline,
          AUC_summary      = AUC_summary,
          AUC_diff         = AUC_diff,
          AUC_diff_summary = AUC_diff_summary,
          ##
          n_thr_vec  = n_thr_vec,
          n_thr_max  = n_thr_max,
          test_names = test_names,
          baseline_case_nd = baseline_case_nd,
          baseline_case_d  = baseline_case_d
        )
        
}



#' Helper: extract named parameters from a 3D trace array by prefix match.
#' @keywords internal
extract_params_from_trace <- function( trace_3d,
                                       param_prefix
) {
  
        param_names <- dimnames(trace_3d)[[3]]
        idx <- grep(paste0("^", param_prefix, "\\["), param_names)
        
        if (length(idx) == 0)
          stop(paste0("No parameters found matching prefix '", param_prefix, "'"))
        
        trace_3d[, , idx, drop = FALSE]
  
}




## ===================================================================================
## Example usage (with language covariate after expand_categorical_to_dummies):
## ===================================================================================
##
## ## After fitting LC-MVOP model and extracting traces:
## trace_beta <- extract_params_from_trace(trace_main, "beta")
## trace_C    <- extract_params_from_trace(trace_tp,   "C_vec")
##
## ## Baseline = English (reference lang), so dummies = c(0, 0), intercept = 1
## baseline_english <- list(c(1, 0, 0), c(1, 0, 0), c(1, 0, 0))
##
## ## Language 2 (e.g. Spanish):
## baseline_spanish <- list(c(1, 1, 0), c(1, 1, 0), c(1, 1, 0))
##
## ## Language 3 (e.g. Mandarin):
## baseline_mandarin <- list(c(1, 0, 1), c(1, 0, 1), c(1, 0, 1))
##
## results_english <- R_fn_compute_LCMVOP_Se_Sp_baseline(
##   trace_beta         = trace_beta,
##   trace_C            = trace_C,
##   baseline_case_nd   = baseline_english,
##   baseline_case_d    = baseline_english,
##   n_covs_per_outcome = matrix(c(3,3,3, 3,3,3), nrow = 2, byrow = TRUE),
##   n_binary_tests     = 1,
##   n_ordinal_tests    = 2,
##   n_thr_per_ord_test = c(27, 30),
##   C_param_name       = "C_vec",
##   test_names         = c("SCID", "PHQ9", "CESD10")
## )
##
## results_english$Se_summaries
## results_english$AUC_summary
## results_english$AUC_diff_summary













