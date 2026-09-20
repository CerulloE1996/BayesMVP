
## R_fns_init_hard_coded_models.R


# json <- jsonlite::fromJSON("/home/enzocerullo/R/R-4.3.3/lib/R/library/BayesMVP/stan_data/data_6e479c5a0b94fe3f633a31e26aa5dbcc_1000.json")
# json$n_thr_per_ord_test
# json$n_cat_per_ord_test
# json$n_ordinal_tests
# 
# file.remove("/home/enzocerullo/R/R-4.3.3/lib/R/library/BayesMVP/stan_data/data_6e479c5a0b94fe3f633a31e26aa5dbcc_1000.json")


#' init_hard_coded_model_finalise_model_args_list
#' @export
init_hard_coded_model_args <- function( Model_type,
                                        model_args_list
) {
  
        ##
        ## ---- Get basic model info for internal/hard-coded models:
        ##
        y <- model_args_list$y
        if (is.null(y)) {
          stop("model_args_list must contain 'y'")
        }
        ##
        outs <- get_basic_dims_for_internal_models( Model_type = Model_type, 
                                                    y = y)
        N <- outs$N
        model_args_list$N <- outs$N
        ##
        n_tests <- outs$n_tests
        model_args_list$n_tests <- outs$n_tests
        ##
        n_class <- outs$n_class
        model_args_list$n_class <- outs$n_class
        ##
        n_binary_tests  <- outs$n_binary_tests
        model_args_list$n_binary_tests <- n_binary_tests
        ##
        n_ordinal_tests <- outs$n_ordinal_tests
        model_args_list$n_ordinal_tests <- n_ordinal_tests
        ##
        model_args_list$n_obs <- N * n_tests
        ##
        is_binary_slot <- sapply(1:n_tests, 
                         function(t) all(y[, t] %in% c(0, 1)))
        ##
        n_binary_tests  <- sum(is_binary_slot)
        n_ordinal_tests <- sum(!is_binary_slot)
        ##
        if (n_ordinal_tests > 0) {
          
              n_cat_per_ord_test <- model_args_list$n_cat_per_ord_test
              ##
              if (is.null(n_cat_per_ord_test)) {
                stop(paste0("n_cat_per_ord_test must be supplied for ordinal models: ", n_ordinal_tests,
                            " ordinal test(s) detected in y. It CANNOT be inferred from the data, ",
                            "since the top category is often unobserved."))
              }
              ##
              n_cat_per_ord_test <- as.integer(n_cat_per_ord_test)
              ##
              if (length(n_cat_per_ord_test) != n_ordinal_tests) {
                stop(paste0("n_cat_per_ord_test must have length n_ordinal_tests (", n_ordinal_tests,
                            "), got ", length(n_cat_per_ord_test)))
              }
              ##
              ## ---- Validate against y (observing FEWER categories than declared is fine):
              ##
              ord_slots <- which(!is_binary_slot)
              for (k in seq_along(ord_slots)) {
                    t <- ord_slots[k]
                    if (max(y[, t]) > n_cat_per_ord_test[k]) {
                      stop(paste0("slot ", t, ": observed max category (", max(y[, t]),
                                  ") exceeds declared n_cat_per_ord_test[", k, "] = ", n_cat_per_ord_test[k]))
                    }
                    if (min(y[, t]) < 1) {
                      stop(paste0("slot ", t, " is ordinal so y must be 1-indexed; found min = ", min(y[, t])))
                    }
                    if (n_cat_per_ord_test[k] < 3L) {
                      stop(paste0("slot ", t, ": n_cat_per_ord_test[", k, "] = ", n_cat_per_ord_test[k],
                                  " but an ordinal test needs >= 3 categories (2 == binary)"))
                    }
              }
              ##
              ## ---- Derive n_cat_per_test (SLOT order; 2 for binary slots):
              ##
              n_cat_per_test <- rep(2L, n_tests)
              n_cat_per_test[ord_slots] <- n_cat_per_ord_test
          
        } else {
          
              n_cat_per_ord_test <- integer(0)
              n_cat_per_test     <- rep(2L, n_tests)
          
        }
        ##
        model_args_list$n_cat_per_test  <- n_cat_per_test
        model_args_list$n_binary_tests  <- n_binary_tests
        model_args_list$n_ordinal_tests <- n_ordinal_tests
        ##
        if (n_ordinal_tests > 0) {
          model_args_list$n_cat_per_ord_test <- n_cat_per_ord_test        ## SLOT order
          model_args_list$n_thr_per_ord_test <- n_cat_per_ord_test - 1L   ## SLOT order
        }
        ##
        # n_pops <- outs$n_pops
        # model_args_list$n_pops <- n_pops
        n_pops <- model_args_list$n_pops  ## either user-supplied or the default (1 pop).
        ##
        ## ---- Calculate basic dims:
        ##
        n_covariates_total <- model_args_list$n_covariates_total
        ##
        dims <- get_dims_for_internal_models(  y = y,
                                               Model_type = Model_type,
                                               n_tests = n_tests,
                                               n_class = n_class,
                                               n_pops = n_pops,
                                               N = N,
                                               n_covariates_total = n_covariates_total,
                                               n_cat_per_ord_test = model_args_list$n_cat_per_ord_test)
        n_nuisance    <- dims$n_nuisance
        n_params_main <- dims$n_params_main
        n_params      <- dims$n_params
        n_corrs       <- dims$n_corrs
        ##
        # n_cat_per_ord_test   <- dims$n_cat_per_ord_test
        # n_thr_per_ord_test   <- dims$n_thr_per_ord_test
        ##
        model_args_list$n_nuisance    <- n_nuisance
        model_args_list$n_params_main <- n_params_main
        model_args_list$n_params      <- n_params
        model_args_list$n_corrs       <- n_corrs
        ##
        # model_args_list$n_cat_per_ord_test       <- n_cat_per_ord_test
        # model_args_list$n_thr_per_ord_test       <- n_thr_per_ord_test
        ##
        # if (is.null(model_args_list$prior_dirichlet_alpha)) {
        #   if (n_ordinal_tests > 0) {
        #     model_args_list$prior_dirichlet_alpha <- rep(list(matrix(1.0, nrow = max(n_cat_per_ord_test), ncol = n_ordinal_tests)), n_class)
        #   } else {
        #     model_args_list$prior_dirichlet_alpha <- rep(list(matrix(1.0, nrow = 1, ncol = 1)), n_class)  # dummy
        #   }
        # }
        ##
        if (n_ordinal_tests > 0) {
            expected_nrow <- max(n_cat_per_ord_test)
            if (is.null(model_args_list$prior_dirichlet_alpha)) {
                  model_args_list$prior_dirichlet_alpha <- rep(list(matrix(1.0, nrow = expected_nrow, 
                                                                           ncol = n_ordinal_tests)), n_class)
            } else {
                  for (c in 1:n_class) {
                        m <- model_args_list$prior_dirichlet_alpha[[c]]
                        if (nrow(m) != expected_nrow || ncol(m) != n_ordinal_tests) {
                          stop(paste0("prior_dirichlet_alpha[[", c, "]] is ", nrow(m), "x", ncol(m),
                                      " but must be max(n_cat_per_ord_test) x n_ordinal_tests = ",
                                      expected_nrow, "x", n_ordinal_tests, ". ",
                                      "Likely stale from a previous init with different n_cat_per_ord_test ",
                                      "(e.g. ungrouped y) -- set it to NULL before re-initialising."))
                        }
                  }
            }
        } else { 
            model_args_list$prior_dirichlet_alpha <- rep(list(matrix(1.0, nrow = 1, ncol = 1)), n_class)  # dummy
        }
        ##
        # if (Model_type %in% c("MVP", "LC_MVP")) {
        #       
        #       model_args_list$n_binary_tests <- n_tests
        #       model_args_list$n_ordinal_tests <- 0
        #       model_args_list$n_cutpoints_total <- 0
        #   
        # } else if (Model_type %in% c("MVOP", "LC_MVOP")) { 
        #   
        #       ## put ordinal stuff here (need to detect "n_bin_tests" from y!)
        # }
        ##
        outs <- process_covariates_X( Model_type = Model_type,
                                      X = model_args_list$X,
                                      N = model_args_list$N,
                                      n_tests = model_args_list$n_tests,
                                      n_class = model_args_list$n_class)
        X <- outs$X
        model_args_list$X <- outs$X
        ##
        n_covariates_per_outcome_mat <- outs$n_covariates_per_outcome_mat
        model_args_list$n_covariates_per_outcome_mat <- outs$n_covariates_per_outcome_mat
        ##
        n_covariates_max <- outs$n_covariates_max
        model_args_list$n_covariates_max <- outs$n_covariates_max
        ##
        n_covariates_max_nd <- outs$n_covariates_max_nd
        model_args_list$n_covariates_max_nd <- outs$n_covariates_max_nd
        ##
        n_covariates_max_d <- outs$n_covariates_max_d
        model_args_list$n_covariates_max_d <- outs$n_covariates_max_d
        ##
        n_covariates_total <- outs$n_covariates_total
        model_args_list$n_covariates_total <- outs$n_covariates_total
        
        # Set defaults for all parameters
        defaults <- list( prior_only = FALSE,
                          vect_type = R_fn_detect_vectorisation_support(),
                          handle_numerical_issues = TRUE,
                          ##
                          nuisance_transformation = "Phi",
                          Phi_type = "Phi",
                          inv_Phi_type = "inv_Phi",
                          ##
                          overflow_threshold  = +7.5,
                          underflow_threshold = -7.5,
                          ##
                          C_raw_lower = -7.5,   ## ---- ordinal-only
                          C_raw_upper = +2.5,   ## ---- ordinal-only
                          ##
                          num_chunks = find_num_chunks_MVP(N, n_tests),
                          ##
                          corr_force_positive = FALSE,
                          corr_param = "Sean",
                          ##
                          corr_prior_norm = FALSE,
                          corr_prior_beta = FALSE,
                          ##
                          debug = FALSE,
                          ##
                          model_so_file = "none",
                          json_file_path = "none")
        
        # Apply defaults
        for (name in names(defaults)) {
          if (is.null(model_args_list[[name]])) {
            model_args_list[[name]] <- defaults[[name]]
          }
        }
        ##
        ## ---- Multiple-pop stuff:
        ##
        if (is.null(model_args_list$n_pops)) {
          model_args_list$n_pops <- 1L
        }
        if (is.null(model_args_list$pop)) {
          model_args_list$pop <- rep(1L, N)  # all same population, 1-indexed
        }
        # if (Model_type %in% c("LC_MVP", "latent_trait")) {
        # Class prevalence priors
        if (is.null(model_args_list$prior_prev_a)) {
          model_args_list$prior_prev_a <- rep(1.0, model_args_list$n_pops)
        }
        if (is.null(model_args_list$prior_prev_b)) {
          model_args_list$prior_prev_b <- rep(1.0, model_args_list$n_pops)
        }
        ##
        ## ---- Handle covariates:
        ##
        if (is.null(model_args_list$X)) {
          model_args_list$X <- matrix(1, nrow = N, ncol = 1)
        }
        
        # CORRECTED: known_values_indicator_list - list of n_class matrices
        if (is.null(model_args_list$known_values_indicator_list)) {
          model_args_list$known_values_indicator_list <- list()
          for (c in 1:n_class) {
            model_args_list$known_values_indicator_list[[c]] <- matrix(0L, nrow = n_tests, ncol = n_tests)
          }
        }
        
        # CORRECTED: known_values_list - list of n_class matrices
        if (is.null(model_args_list$known_values_list)) {
          model_args_list$known_values_list <- list()
          for (c in 1:n_class) {
            model_args_list$known_values_list[[c]] <- matrix(0.0, nrow = n_tests, ncol = n_tests)
          }
        }
        
        #  ub_corr and lb_corr are LISTS OF MATRICES!
        # if (Model_type %in% c("MVP", "LC_MVP")) {
          # These are Model_args_vecs_of_mats_double - std::vector of matrices
          if (is.null(model_args_list$ub_corr)) {
            model_args_list$ub_corr <- list()
            for (c in 1:n_class) {
              model_args_list$ub_corr[[c]] <- matrix(1.0, nrow = n_tests, ncol = n_tests)  # All +1.0
            }
          }
          
          if (is.null(model_args_list$lb_corr)) {
            model_args_list$lb_corr <- list()
            for (c in 1:n_class) {
              model_args_list$lb_corr[[c]] <- matrix(-1.0, nrow = n_tests, ncol = n_tests)  # All -1.0
            }
          }
        # }
        
        # Initialize prior matrices
        if (is.null(model_args_list$prior_coeffs_mean_mat)) {
          model_args_list$prior_coeffs_mean_mat <- list()
          for (c in 1:n_class) {
            model_args_list$prior_coeffs_mean_mat[[c]] <- matrix(0.0, nrow = n_covariates_max, ncol = n_tests)
          }
        }
        if (is.null(model_args_list$prior_coeffs_sd_mat)) {
          model_args_list$prior_coeffs_sd_mat <- list()
          for (c in 1:n_class) {
            model_args_list$prior_coeffs_sd_mat[[c]] <- matrix(1.0, nrow = n_covariates_max, ncol = n_tests)
          }
        }
        
        # Model-specific parameters
        # if (Model_type %in% c("MVP", "LC_MVP")) {
            # LKJ prior
            if (is.null(model_args_list$lkj_cholesky_eta)) {
              model_args_list$lkj_cholesky_eta <- matrix(2.0, nrow = n_class, ncol = 1)
            }
            model_args_list$prior_LKJ <- model_args_list$lkj_cholesky_eta
       
            # # Skewed LKJ priors
            # if (is.null(model_args_list$prior_for_skewed_LKJ_a)) {
            #   model_args_list$prior_for_skewed_LKJ_a <- matrix(5.0, nrow = n_class, ncol = 1)
            # }
            # if (is.null(model_args_list$prior_for_skewed_LKJ_b)) {
            #   model_args_list$prior_for_skewed_LKJ_b <- matrix(5.0, nrow = n_class, ncol = 1)
            # }
            # 
            # model_args_list$list_prior_for_corr_a <-  model_args_list$list_prior_for_corr_b <- list()
            
            if (is.null(model_args_list$prior_for_corr_a)) {
              model_args_list$prior_for_corr_a <- rep(list(array(1.0, dim = c(n_tests, n_tests))), 2)
            }
            if (is.null(model_args_list$prior_for_corr_b)) {
              model_args_list$prior_for_corr_b <- rep(list(array(1.0, dim = c(n_tests, n_tests))), 2)
            }
        # }
        
        if (is.null(model_args_list$J_grad_option)) { 
          model_args_list$J_grad_option <- "num_diff"
        }
        if (is.null(model_args_list$LT_prior_corr_bs)) { 
          model_args_list$LT_prior_corr_bs <- "gamma"
        }

        
        if (Model_type == "latent_trait") {
                # Latent trait specific
                if (is.null(model_args_list$LT_b_priors_1)) {
                  model_args_list$LT_b_priors_1 <- matrix(1.0, nrow = n_class, ncol = n_tests)
                }
                if (is.null(model_args_list$LT_b_priors_2)) {
                  model_args_list$LT_b_priors_2 <- matrix(1.0, nrow = n_class, ncol = n_tests)
                }
                if (is.null(model_args_list$LT_known_bs_values)) {
                  model_args_list$LT_known_bs_values <- matrix(0.00001, nrow = n_class, ncol = n_tests)
                }
                if (is.null(model_args_list$LT_known_bs_indicator)) {
                  model_args_list$LT_known_bs_indicator <- matrix(0L, nrow = n_class, ncol = n_tests)
                }
        } else { ## need to make dummy mat's for C++:
                model_args_list$LT_b_priors_1 <- matrix(1.0, nrow = n_class, ncol = n_tests)
                model_args_list$LT_b_priors_2 <- matrix(1.0, nrow = n_class, ncol = n_tests)
                model_args_list$LT_known_bs_values <- matrix(0.00001, nrow = n_class, ncol = n_tests)
                model_args_list$LT_known_bs_indicator <- matrix(0L, nrow = n_class, ncol = n_tests)
        }
        
        
        # Set all vectorization types consistently
        vect_types <- c("vect_type_exp", "vect_type_tanh", "vect_type_log", 
                        "vect_type_lse", "vect_type_Phi", "vect_type_inv_Phi")
        for (vt in vect_types) {
          model_args_list[[vt]] <- model_args_list$vect_type
        }
        
        return(model_args_list)
        
}





#' Helper for null defaults
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}





#' build_Model_args_as_Rcpp_Lists
#' @export
build_Model_args_as_Rcpp_List <- function(Model_type,
                                          model_args_list, 
                                          n_nuisance,
                                          n_params_main
) {
  
        # Extract from model_args_list
        y <- model_args_list$y
        N <- model_args_list$N
        n_tests <- model_args_list$n_tests
        n_class <- model_args_list$n_class
        n_params <- n_nuisance + n_params_main
        ##
        n_binary_tests    <- model_args_list$n_binary_tests
        n_ordinal_tests   <- model_args_list$n_ordinal_tests
        ##
        print(paste("n_binary_tests = ", n_binary_tests))
        print(paste("n_ordinal_tests = ", n_ordinal_tests))
        ##
        n_cat_per_ord_test <- model_args_list$n_cat_per_ord_test
        n_thr_per_ord_test <- model_args_list$n_thr_per_ord_test
        ##
        n_cat_per_ord_test   <- matrix(as.integer(n_cat_per_ord_test), ncol = 1)
        n_thr_per_ord_test   <- matrix(as.integer(n_thr_per_ord_test), ncol = 1)
        ##
        # n_cat_per_test <- c( rep(2L, n_binary_tests), as.integer(n_cat_per_ord_test) )
        n_cat_per_test <- matrix(as.integer(
          model_args_list$n_cat_per_test %||% c(rep(2L, n_binary_tests), as.integer(model_args_list$n_cat_per_ord_test))
        ), ncol = 1)
        ##
        print(paste("n_cat_per_ord_test = ", n_cat_per_ord_test))
        print(paste("n_thr_per_ord_test = ", n_thr_per_ord_test))
        ##
        # Model_args_bools - ORDER MATTERS! Must match C++ indexing
        Model_args_bools <- c(
          model_args_list$exclude_priors %||% FALSE,           # [0]
          model_args_list$CI %||% FALSE,                       # [1]
          model_args_list$corr_force_positive,                 # [2]
          model_args_list$corr_prior_beta %||% FALSE,          # [3]
          model_args_list$corr_prior_norm %||% FALSE,          # [4]
          model_args_list$handle_numerical_issues %||% TRUE,   # [5]
          model_args_list$skip_checks_exp %||% FALSE,          # [6]
          model_args_list$skip_checks_log %||% FALSE,          # [7]
          model_args_list$skip_checks_lse %||% FALSE,          # [8]
          model_args_list$skip_checks_tanh %||% FALSE,         # [9]
          model_args_list$skip_checks_Phi %||% FALSE,          # [10]
          model_args_list$skip_checks_log_Phi %||% FALSE,      # [11]
          model_args_list$skip_checks_inv_Phi %||% FALSE,      # [12]
          model_args_list$skip_checks_inv_Phi_approx_from_logit_prob %||% FALSE, # [13]
          model_args_list$debug %||% FALSE                     # [14]
        )
        Model_args_bools <- matrix(Model_args_bools, ncol = 1, nrow = length(Model_args_bools))
        # Model_args_ints - ORDER MATTERS!
        Model_args_ints <- c(
          model_args_list$n_cores %||% 1L,                     # [0]
          n_class,                                             # [1]
          model_args_list$ub_threshold_phi_approx %||% 10L,    # [2]
          model_args_list$num_chunks %||% 1L,                  # [3]
          model_args_list$n_binary_tests,                      # [4] ## ---- ordinal-only
          model_args_list$n_ordinal_tests,                      # [5] ## ---- ordinal-only
          model_args_list$n_pops ## ---- multiple-pops
        )
        Model_args_ints <- matrix(Model_args_ints, ncol = 1, nrow = length(Model_args_ints))
        # Model_args_doubles - ORDER MATTERS!
        Model_args_doubles <- c(
          # model_args_list$prior_prev_a,                        # [0]
          # model_args_list$prior_prev_b,                        # [1]
          model_args_list$overflow_threshold,                  # [0]
          model_args_list$underflow_threshold,                 # [1]
          model_args_list$C_raw_lower %||% -7.5,               # [2]  NEW ## ---- ordinal-only
          model_args_list$C_raw_upper %||% +2.5                # [3]  NEW ## ---- ordinal-only
        )
        Model_args_doubles <- matrix(Model_args_doubles, ncol = 1, nrow = length(Model_args_doubles))
        
        # Model_args_strings - ORDER MATTERS!
        Model_args_strings <- c(
          as.character(model_args_list$vect_type),              # [0]
          "Phi",                                                # [1] Phi_type
          "inv_Phi",                                            # [2] inv_Phi_type
          ##
          as.character(model_args_list$vect_type_exp %||% model_args_list$vect_type),      # [3]
          as.character(model_args_list$vect_type_log %||% model_args_list$vect_type),      # [4]
          as.character(model_args_list$vect_type_lse %||% model_args_list$vect_type),      # [5]
          as.character(model_args_list$vect_type_tanh %||% model_args_list$vect_type),     # [6]
          ##
          as.character(model_args_list$vect_type_Phi %||% model_args_list$vect_type),      # [7]
          as.character(model_args_list$vect_type_log_Phi %||% model_args_list$vect_type),  # [8]
          as.character(model_args_list$vect_type_inv_Phi %||% model_args_list$vect_type),  # [9]
          as.character(model_args_list$vect_type_inv_Phi_approx_from_logit_prob %||% model_args_list$vect_type), # [10]
          ##
          model_args_list$J_grad_option %||% "num_diff" ,                                          # [11] J_grad_option
          model_args_list$nuisance_transformation %||% "Phi",              # [12]
          model_args_list$LT_prior_corr_bs %||% "gamma" # [13]
        )
        
        Model_args_strings <- matrix(Model_args_strings, ncol = 1, nrow = length(Model_args_strings))
        # Model_args_col_vecs_double - ORDERED LIST
        Model_args_col_vecs_double <- list(
          matrix(model_args_list$lkj_cholesky_eta, ncol = 1),  # [0]
          model_args_list$prior_prev_a,                                      # [1] 
          model_args_list$prior_prev_b                                       # [2] 
        )
        ##
        ## Model_args_col_vecs_int:
        ##
        pop_ind_0indexed <- matrix(as.integer(model_args_list$pop - 1L), ncol = 1)  # convert 1-indexed → 0-indexed
        ##
        Model_args_col_vecs_int <- list( n_cat_per_ord_test,  ## [0] ## ---- ordinal-only
                                         n_thr_per_ord_test,   ## [1] ## ---- ordinal-only
                                         pop_ind_0indexed,      ## [2] ## ----
                                         n_cat_per_test   ## [3] length n_tests, SLOT order, 2 for binary tests
                                         )
        
        # Model_args_mats_double - ORDERED LIST (y goes elsewhere)
        Model_args_mats_double <- list(
          model_args_list$LT_b_priors_1,         ## [0]
          model_args_list$LT_b_priors_2,         ## [1]
          model_args_list$LT_known_bs_indicator, ## [2]
          model_args_list$LT_known_bs_values    ## [3]
          # model_args_list$prior_dirichlet_alpha  ## [4] ## ---- ordinal-only
        )
        
        # Model_args_mats_int - ORDERED LIST
        Model_args_mats_int <- list(
          model_args_list$n_covariates_per_outcome_mat        # [0]
        )
        
        # Model_args_vecs_of_col_vecs_double - empty
        Model_args_vecs_of_col_vecs_double <- list()
        
        # Model_args_vecs_of_col_vecs_int - empty
        Model_args_vecs_of_col_vecs_int <- list()
        
        # Model_args_vecs_of_mats_double - ORDERED LIST - CRITICAL!
        Model_args_vecs_of_mats_double <- list(
          model_args_list$prior_coeffs_mean_mat,              # [0]
          model_args_list$prior_coeffs_sd_mat,                # [1]
          model_args_list$prior_for_corr_a,                   # [2]
          model_args_list$prior_for_corr_b,                   # [3]
          model_args_list$lb_corr,                            # [4]
          model_args_list$ub_corr,                            # [5]
          model_args_list$known_values_list,                   # [6]
          model_args_list$prior_dirichlet_alpha               # [7] ## ---- ordinal-only
        )
        
        # Model_args_vecs_of_mats_int - ORDERED LIST
        Model_args_vecs_of_mats_int <- list(
          model_args_list$known_values_indicator_list         # [0]
        )
        
        # Model_args_2_layer_vecs_of_col_vecs_double - empty
        Model_args_2_layer_vecs_of_col_vecs_double <- list()
        
        # Model_args_2_layer_vecs_of_col_vecs_int - empty
        Model_args_2_layer_vecs_of_col_vecs_int <- list()
        
        # Model_args_2_layer_vecs_of_mats_double - X GOES HERE!
        # Process X to correct format
        if (!is.null(model_args_list$X)) {
          # X should already be a list of lists of matrices
          # For MVP: X[[1]][[t]] is the matrix for outcome t
          # For LC_MVP: X[[c]][[t]] is the matrix for class c, outcome t
          Model_args_2_layer_vecs_of_mats_double <- list(
            model_args_list$X                                  
          )
        } else {
          Model_args_2_layer_vecs_of_mats_double <- list()
        }
        # Model_args_2_layer_vecs_of_mats_int - empty
        Model_args_2_layer_vecs_of_mats_int <- list()
        ## typos in c++:
        Model_args_2_later_vecs_of_col_vecs_double <- Model_args_2_layer_vecs_of_col_vecs_double
        Model_args_2_later_vecs_of_col_vecs_int <- Model_args_2_layer_vecs_of_col_vecs_int
        Model_args_2_later_vecs_of_mats_double <- Model_args_2_layer_vecs_of_mats_double
        Model_args_2_later_vecs_of_mats_int <- Model_args_2_layer_vecs_of_mats_int
        ##
        colnames(Model_args_bools) <- c("Model arguments - boolean")
        rownames(Model_args_bools) <- c("exclude_priors", "Cond. indep.", "Force +'ve corr's",
                                        "corr_prior_beta", "corr_prior_norm",
                                        "handle num. issues",
                                        "skip_checks_exp", "skip_checks_log", "skip_checks_lse", "skip_checks_tanh",
                                        "skip_checks_Phi", "skip_checks_log_Phi", "skip_checks_inv_Phi",
                                        "skip_checks_inv_Phi_approx_from_logit_prob",
                                        "debug")
        ##
        colnames(Model_args_ints) <- c("Model arguments - integers")
        rownames(Model_args_ints) <- c("n_cores",
                                       "n_class",
                                       "ub_threshold_phi_approx",
                                       "number of chunks",
                                       "n_binary_tests", 
                                       "n_ordinal_tests", 
                                       "n_pops" ## ----
                                       )
        ##
        colnames(Model_args_doubles) <- c("Model arguments - doubles")
        # rownames(Model_args_doubles) <- c("prior_prev_a", "prior_prev_b", "overflow_threshold", "underflow_threshold")
        rownames(Model_args_doubles) <- c("overflow_threshold", "underflow_threshold", "C_raw_lower", "C_raw_upper")
        ##
        colnames(Model_args_strings) <- c("Model arguments - character strings")
        rownames(Model_args_strings) <- c("vect_type",
                                          "Phi_type", 
                                          "inv_Phi_type",
                                          "vect_type_exp", 
                                          "vect_type_log", 
                                          "vect_type_lse", 
                                          "vect_type_tanh",
                                          "vect_type_Phi",
                                          "vect_type_log_Phi",
                                          "vect_type_inv_Phi",
                                          "vect_type_inv_Phi_approx_from_logit_prob",
                                          "J_grad_option", 
                                          "nuisance_transformation", 
                                          "LT_prior_corr_bs")
        ##
        ## ---- Build final list matching C++ struct EXACTLY:
        ##
        Model_args_as_Rcpp_List <- list(
          N = as.integer(N),
          n_nuisance = as.integer(n_nuisance),
          n_params_main = as.integer(n_params_main),
          ##
          n_binary_tests = as.integer(n_binary_tests),
          n_ordinal_tests = as.integer(n_ordinal_tests),
          ##
          model_so_file = as.character(model_args_list$model_so_file %||% "none"),
          json_file_path = as.character(model_args_list$json_file_path %||% "none"),
          ##
          Model_args_bools = Model_args_bools,
          Model_args_ints = Model_args_ints,
          Model_args_doubles = Model_args_doubles,
          Model_args_strings = Model_args_strings,
          ##
          Model_args_col_vecs_double = Model_args_col_vecs_double,
          Model_args_col_vecs_int = Model_args_col_vecs_int,
          Model_args_mats_double = Model_args_mats_double,
          Model_args_mats_int = Model_args_mats_int,
          ##
          Model_args_vecs_of_col_vecs_double = Model_args_vecs_of_col_vecs_double,
          Model_args_vecs_of_col_vecs_int = Model_args_vecs_of_col_vecs_int,
          Model_args_vecs_of_mats_double = Model_args_vecs_of_mats_double,
          Model_args_vecs_of_mats_int = Model_args_vecs_of_mats_int,
          ##
          Model_args_2_layer_vecs_of_col_vecs_double = Model_args_2_layer_vecs_of_col_vecs_double,
          Model_args_2_layer_vecs_of_col_vecs_int = Model_args_2_layer_vecs_of_col_vecs_int,
          Model_args_2_layer_vecs_of_mats_double = Model_args_2_layer_vecs_of_mats_double,
          Model_args_2_layer_vecs_of_mats_int = Model_args_2_layer_vecs_of_mats_int,
          ##
          Model_args_2_later_vecs_of_col_vecs_double = Model_args_2_later_vecs_of_col_vecs_double,
          Model_args_2_later_vecs_of_col_vecs_int = Model_args_2_later_vecs_of_col_vecs_int,
          Model_args_2_later_vecs_of_mats_double = Model_args_2_later_vecs_of_mats_double,
          Model_args_2_later_vecs_of_mats_int = Model_args_2_later_vecs_of_mats_int
        )
        
        return(Model_args_as_Rcpp_List)
        
}








#' init_hard_coded_model
#' @export
init_hard_coded_model <- function(  Model_type, 
                                    model_args_list
) {
  
        hard_coded_models_vec <- c("LC_MVP", 
                                   "MVP",
                                   "latent_trait", 
                                   "LC_MVOP", 
                                   "MVOP")
        ##
        if (!(Model_type %in% hard_coded_models_vec)) {
          stop("Model_type must be one of: ", paste(hard_coded_models_vec, collapse = ", "))
        }
        ##
        y <- model_args_list$y
        if (is.null(y)) { 
          stop("no data (y) supplied.")
        }
        ##
        if (!is.matrix(y)) { 
          stop("y must be a matrix where #cols = #outcomes and #rows = #individuals")
        }
        ##
        ## ---- Initialize all model args with proper defaults:
        ##
        model_args_list <- init_hard_coded_model_args(  Model_type = Model_type,
                                                        model_args_list = model_args_list)
        N <- model_args_list$N
        n_tests <- model_args_list$n_tests
        n_class <- model_args_list$n_class
        n_covariates_total <- model_args_list$n_covariates_total
        n_pops <- model_args_list$n_pops
        ##
        model_args_list$N <- N
        model_args_list$n_tests <- n_tests
        model_args_list$n_class <- n_class
        model_args_list$n_covariates_total <- n_covariates_total
        ##
        model_args_list$n_pops <- n_pops
        model_args_list$n_pops
        ##
        ## ---- Calculate basic dims:
        ##
        dims <- get_dims_for_internal_models(  y = y,
                                               Model_type = Model_type, 
                                               n_tests = n_tests, 
                                               n_class = n_class,
                                               n_pops = n_pops,
                                               N = N,
                                               n_covariates_total = n_covariates_total,
                                               n_cat_per_ord_test = model_args_list$n_cat_per_ord_test)
        n_nuisance    <- dims$n_nuisance
        n_params_main <- dims$n_params_main
        n_params      <- dims$n_params
        n_corrs       <- dims$n_corrs
        ##
        # n_cat_per_ord_test <- dims$n_cat_per_ord_test
        # n_thr_per_ord_test <- dims$n_thr_per_ord_test
        ##
        model_args_list$n_nuisance    <- n_nuisance
        model_args_list$n_params_main <- n_params_main
        model_args_list$n_params      <- n_params
        model_args_list$n_corrs       <- n_corrs
        ##
        # model_args_list$n_cat_per_ord_test <- n_cat_per_ord_test
        # model_args_list$n_thr_per_ord_test <- n_thr_per_ord_test
        ##
        ##
        ## ---- Slot-order type detection from y (fully general -- interleaving-safe).
        ##      OVERRIDES the binaries-first assumptions above; re-init after any column
        ##      permutation automatically yields correct slot-order metadata:
        ##
        # n_cat_per_test <- sapply(1:n_tests, function(t) {
        #   if (all(y[, t] %in% c(0, 1))) 2L else as.integer(max(y[, t]))
        # })
        ##
        ## ---- Slot-order type detection. n_cat_per_test is USER-SUPPLIED for ordinal tests
        ##      (the observed max is NOT the number of categories -- the top category is often
        ##      unobserved). Detection from y is used ONLY as a fallback when the user gives
        ##      nothing, and warns, since it is unreliable.
        ##
        ##      NOTE: whatever the source, this vector is in SLOT order, so re-init after a
        ##      column permutation must be given the PERMUTED vector (see R_fn_sample.R).
        ##
        ##
        ## ---- Slot-order metadata.
        ##
        ##      BINARY vs ORDINAL is detected from y (safe: a column is binary iff all values
        ##      are in {0,1}, which does not depend on unobserved categories).
        ##
        ##      The NUMBER OF CATEGORIES for the ordinal tests is USER-SUPPLIED via
        ##      n_cat_per_ord_test -- it CANNOT be inferred, because max(y[, t]) undercounts
        ##      whenever the top category is unobserved.
        ##
        ##      n_cat_per_test (length n_tests, 2 for binary slots) is DERIVED from these two.
        ##      Everything here is in SLOT order.
        ##
        # is_binary_slot <- sapply(1:n_tests, 
        #                          function(t) all(y[, t] %in% c(0, 1)))
        # ##
        # n_binary_tests  <- sum(is_binary_slot)
        # n_ordinal_tests <- sum(!is_binary_slot)
        # ##
        # if (n_ordinal_tests > 0) {
        #   
        #       n_cat_per_ord_test <- model_args_list$n_cat_per_ord_test
        #       ##
        #       if (is.null(n_cat_per_ord_test)) {
        #         stop(paste0("n_cat_per_ord_test must be supplied for ordinal models: ", n_ordinal_tests,
        #                     " ordinal test(s) detected in y. It CANNOT be inferred from the data, ",
        #                     "since the top category is often unobserved."))
        #       }
        #       ##
        #       n_cat_per_ord_test <- as.integer(n_cat_per_ord_test)
        #       ##
        #       if (length(n_cat_per_ord_test) != n_ordinal_tests) {
        #         stop(paste0("n_cat_per_ord_test must have length n_ordinal_tests (", n_ordinal_tests,
        #                     "), got ", length(n_cat_per_ord_test)))
        #       }
        #       ##
        #       ## ---- Validate against y (observing FEWER categories than declared is fine):
        #       ##
        #       ord_slots <- which(!is_binary_slot)
        #       for (k in seq_along(ord_slots)) {
        #             t <- ord_slots[k]
        #             if (max(y[, t]) > n_cat_per_ord_test[k]) {
        #               stop(paste0("slot ", t, ": observed max category (", max(y[, t]),
        #                           ") exceeds declared n_cat_per_ord_test[", k, "] = ", n_cat_per_ord_test[k]))
        #             }
        #             if (min(y[, t]) < 1) {
        #               stop(paste0("slot ", t, " is ordinal so y must be 1-indexed; found min = ", min(y[, t])))
        #             }
        #             if (n_cat_per_ord_test[k] < 3L) {
        #               stop(paste0("slot ", t, ": n_cat_per_ord_test[", k, "] = ", n_cat_per_ord_test[k],
        #                           " but an ordinal test needs >= 3 categories (2 == binary)"))
        #             }
        #       }
        #       ##
        #       ## ---- Derive n_cat_per_test (SLOT order; 2 for binary slots):
        #       ##
        #       n_cat_per_test <- rep(2L, n_tests)
        #       n_cat_per_test[ord_slots] <- n_cat_per_ord_test
        #   
        # } else {
        #   
        #       n_cat_per_ord_test <- integer(0)
        #       n_cat_per_test     <- rep(2L, n_tests)
        #   
        # }
        # ##
        # model_args_list$n_cat_per_test  <- n_cat_per_test
        # model_args_list$n_binary_tests  <- n_binary_tests
        # model_args_list$n_ordinal_tests <- n_ordinal_tests
        # ##
        # if (n_ordinal_tests > 0) {
        #   model_args_list$n_cat_per_ord_test <- n_cat_per_ord_test        ## SLOT order
        #   model_args_list$n_thr_per_ord_test <- n_cat_per_ord_test - 1L   ## SLOT order
        # }
        ##
        # model_args_list$n_cat_per_test  <- n_cat_per_test
        # model_args_list$n_binary_tests  <- sum(n_cat_per_test == 2L)
        # model_args_list$n_ordinal_tests <- sum(n_cat_per_test >  2L)
        ##
        # if (model_args_list$n_ordinal_tests > 0) {
        #   model_args_list$n_cat_per_ord_test <- n_cat_per_test[n_cat_per_test > 2L]   ## SLOT order
        #   model_args_list$n_thr_per_ord_test <- model_args_list$n_cat_per_ord_test - 1L
        # }
        # model_args_list$n_cat_per_test  <- n_cat_per_test
        # model_args_list$n_binary_tests  <- sum(n_cat_per_test == 2L)
        # model_args_list$n_ordinal_tests <- sum(n_cat_per_test >  2L)
        ##
        # if (model_args_list$n_ordinal_tests > 0) {
        #   model_args_list$n_cat_per_ord_test <- n_cat_per_test[n_cat_per_test > 2L]   ## SLOT order
        #   model_args_list$n_thr_per_ord_test <- model_args_list$n_cat_per_ord_test - 1L
        # }
        ##
        ## ---- Build the "Model_args_as_Rcpp_List" C++ list:
        ##
        Model_args_as_Rcpp_List <- build_Model_args_as_Rcpp_List( Model_type = Model_type, 
                                                                  model_args_list = model_args_list, 
                                                                  n_nuisance = n_nuisance,
                                                                  n_params_main = n_params_main)
        
        return(list(
          Model_args_as_Rcpp_List = Model_args_as_Rcpp_List,
          model_args_list = model_args_list,
          n_params_main = n_params_main,
          n_nuisance = n_nuisance
        ))
        
}









# 
# 
# n_covariates_per_outcome_mat <- matrix(3, nrow = 2, ncol = 6)
# model_args_list$n_covariates_per_outcome_mat <- n_covariates_per_outcome_mat
# ##
# n_covariates_max <- n_covariates_max_nd <- n_covariates_max_d <- 3
# model_args_list$n_covariates_max <- n_covariates_max
# model_args_list$n_covariates_max_nd <- n_covariates_max_nd
# model_args_list$n_covariates_max_d <- n_covariates_max_d
# model_args_list$n_covariates_total <- sum(n_covariates_per_outcome_mat)

#' make_Stan_data_list_for_internal_models
#' @export
make_Stan_data_list_for_internal_models <- function( Model_type, 
                                                     model_args_list
                                                      
) {
        
        y <- model_args_list$y
        N <- model_args_list$N
        n_tests <- model_args_list$n_tests
        n_class <- model_args_list$n_class
        ##
        X <- model_args_list$X
        n_covariates_per_outcome_mat <- model_args_list$n_covariates_per_outcome_mat
        ##
        n_covariates_max <- model_args_list$n_covariates_max
        n_covariates_max_nd <- model_args_list$n_covariates_max_nd
        n_covariates_max_d <- model_args_list$n_covariates_max_d
        n_covariates_total <- model_args_list$n_covariates_total
        ##
        outs <- convert_X_to_padded_for_Stan(X = X,
                                             N = N,
                                             n_tests = n_tests,
                                             n_class = n_class,
                                             n_covariates_max = n_covariates_max,
                                             n_covariates_max_nd = n_covariates_max_nd,
                                             n_covariates_max_d = n_covariates_max_d,
                                             n_covariates_per_outcome_mat = n_covariates_per_outcome_mat,
                                             padding_value = -999)
        X_nd <- outs$X_nd
        X_d  <- outs$X_d
        
        str(X)
        str(X_nd)
        str(X_d)
        
        if (Model_type %in% c("MVP", "LC_MVP", "MVOP", "LC_MVOP")) {
          
              known_num <- 0
              if (!is.null(model_args_list$known_values_indicator_list)) {
                for (c in 1:n_class) {
                  known_num <- known_num + sum(model_args_list$known_values_indicator_list[[c]])
                }
              }
              known_values_indicator_list <- if_null_then_set_to(
                model_args_list$known_values_indicator_list,
                lapply(1:n_class, function(c) matrix(0L, n_tests, n_tests))
              )
              known_values_list <- if_null_then_set_to(
                model_args_list$known_values_list,
                lapply(1:n_class, function(c) matrix(0.0, n_tests, n_tests))
              )
              
        }
        
        Stan_data_list <- list()
        
        if (Model_type == "LC_MVP") {
          
                prior_only <-  as.integer(model_args_list$prior_only)
                
                prior_coeffs_mean_mat <- model_args_list$prior_coeffs_mean_mat
                prior_coeffs_sd_mat <- model_args_list$prior_coeffs_sd_mat
                
                corr_force_positive <- as.integer(model_args_list$corr_force_positive)
                lb_corr <- model_args_list$lb_corr
                ub_corr <- model_args_list$ub_corr
                ##
                prior_LKJ <- model_args_list$prior_LKJ
                if (!(is.matrix(prior_LKJ))) {
                  prior_LKJ <-  matrix(prior_LKJ, ncol = 1)
                }
                
                overflow_threshold <-  model_args_list$overflow_threshold
                underflow_threshold <- model_args_list$underflow_threshold
                handle_numerical_issues <- if_null_then_set_to(model_args_list$handle_numerical_issues, 1)
                fully_vectorised <- if_null_then_set_to(model_args_list$fully_vectorised, 1)
                
                Phi_type <- model_args_list$Phi_type
                Phi_type_int <- ifelse(Phi_type == "Phi", 1, 2)
                
                n_pops <- if_null_then_set_to(model_args_list$n_pops, 1)
                pop <- if_null_then_set_to(model_args_list$pop,  c(rep(1, N)))
                prior_prev_a <- model_args_list$prior_prev_a
                prior_prev_b <- model_args_list$prior_prev_b
                ##
                if (!(is.matrix(prior_prev_a))) {
                  prior_prev_a <-  matrix(prior_prev_a, ncol = 1)
                }
                ##
                if (!(is.matrix(prior_prev_b))) {
                  prior_prev_b <-  matrix(prior_prev_b, ncol = 1)
                }
                ##
                baseline_case_nd <- baseline_case_d <- list()
                ##
                for (t in 1:n_tests) {
                  baseline_case_nd[[t]] <- rep(0.0, n_covariates_max)
                  baseline_case_d[[t]]  <- rep(0.0, n_covariates_max)
                  ##
                  baseline_case_nd[[t]][1] <- 1.0
                  baseline_case_d[[t]][1]  <- 1.0
                }
                ##
                Stan_data_list <- list(N = N,
                                       n_tests = n_tests,
                                       y = y,
                                       n_class = n_class,
                                       n_pops =  n_pops,  ## multi-pop not supported yet (currently only in Stan version)
                                       pop =  pop,
                                       ##
                                       test_perm = as.integer(model_args_list$test_perm %||% 1:n_tests),
                                       ##
                                       n_covariates_max_nd = n_covariates_max_nd,
                                       n_covariates_max_d = n_covariates_max_d,
                                       n_covariates_max = n_covariates_max,
                                       X_nd = X_nd,
                                       X_d = X_d,
                                       n_covs_per_outcome = n_covariates_per_outcome_mat,
                                       ##
                                       corr_force_positive = corr_force_positive,
                                       known_num = known_num,
                                       ##
                                       # known_values_indicator = known_values_indicator_list,
                                       known_values_indicator_list = known_values_indicator_list,
                                       ##
                                       # known_values = known_values_list,
                                       known_values_list = known_values_list,
                                       ##
                                       lb_corr = lb_corr,
                                       ub_corr = ub_corr,
                                       prior_LKJ = prior_LKJ,
                                       lkj_cholesky_eta = prior_LKJ,
                                       ##
                                       overflow_threshold =  overflow_threshold,
                                       underflow_threshold = underflow_threshold,
                                       ##
                                       prior_only = prior_only,
                                       ##
                                       prior_beta_mean = prior_coeffs_mean_mat,
                                       prior_beta_sd = prior_coeffs_sd_mat,
                                       ##
                                       prior_prev_a = prior_prev_a,
                                       prior_prev_b = prior_prev_b,
                                       ##
                                       Phi_type =  Phi_type_int,
                                       handle_numerical_issues = handle_numerical_issues,
                                       fully_vectorised = fully_vectorised,
                                       ##
                                       baseline_case_nd = baseline_case_nd,
                                       baseline_case_d = baseline_case_d)
                
                # print(paste("Stan_data_list = "))
                # print(str(Stan_data_list))
          
        } else if (Model_type == "MVP") {
          
                prior_only <-  as.integer(model_args_list$prior_only)
                ##
                prior_coeffs_mean_mat <- model_args_list$prior_coeffs_mean_mat
                prior_coeffs_sd_mat   <- model_args_list$prior_coeffs_sd_mat
                ##
                corr_force_positive <- as.integer(model_args_list$corr_force_positive)
                ub_corr <- model_args_list$ub_corr
                lb_corr <- model_args_list$lb_corr
                ##
                prior_LKJ <- model_args_list$prior_LKJ
                if (!(is.matrix(prior_LKJ))) {
                  prior_LKJ <-  matrix(prior_LKJ, ncol = 1)
                }
                ##
                overflow_threshold <-  model_args_list$overflow_threshold
                underflow_threshold <- model_args_list$underflow_threshold
                handle_numerical_issues <- if_null_then_set_to(model_args_list$handle_numerical_issues, 1)
                fully_vectorised <- if_null_then_set_to(model_args_list$fully_vectorised, 1)
                ##
                Phi_type <- model_args_list$Phi_type
                Phi_type_int <- ifelse(Phi_type == "Phi", 1, 2)
                ##
                baseline_case_nd <- baseline_case_d <- list()
                ##
                for (t in 1:n_tests) {
                  baseline_case_nd[[t]] <- rep(0.0, n_covariates_max)
                  baseline_case_d[[t]]  <- rep(0.0, n_covariates_max)
                  ##
                  baseline_case_nd[[t]][1] <- 1.0
                  baseline_case_d[[t]][1]  <- 1.0
                }
                ##
                baseline_case_nd
                ##
                Stan_data_list <- list(N = N,
                                       n_tests = n_tests,
                                       y = y,
                                       ##
                                       test_perm = as.integer(model_args_list$test_perm %||% 1:n_tests),
                                       ##
                                       n_covariates_max = n_covariates_max,
                                       X = X,
                                       n_covs_per_outcome = n_covariates_per_outcome_mat,
                                       ##
                                       corr_force_positive = corr_force_positive,
                                       known_num = known_num,
                                       known_values_indicator_list = known_values_indicator_list,
                                       known_values_list = known_values_list,
                                       lb_corr = lb_corr,
                                       ub_corr = ub_corr,
                                       prior_LKJ = prior_LKJ,
                                       ##
                                       overflow_threshold =  overflow_threshold,
                                       underflow_threshold = underflow_threshold,
                                       ##
                                       prior_only = prior_only,
                                       ##
                                       prior_beta_mean = prior_coeffs_mean_mat,
                                       prior_beta_sd = prior_coeffs_sd_mat,
                                       ##
                                       Phi_type =  Phi_type_int,
                                       handle_numerical_issues =  handle_numerical_issues,
                                       fully_vectorised =  fully_vectorised,
                                       ##
                                       baseline_case_nd = baseline_case_nd,
                                       baseline_case_d = baseline_case_d)
          
        } else if (Model_type == "latent_trait") {
          
                prior_only <-  as.integer(model_args_list$prior_only)
                
                LT_b_priors_1 <- model_args_list$LT_b_priors_1
                LT_b_priors_2 <- model_args_list$LT_b_priors_2
                LT_known_bs_values <- model_args_list$LT_known_bs_values
                LT_known_bs_indicator <- model_args_list$LT_known_bs_indicator
                
                prior_coeffs_mean_mat <- model_args_list$prior_coeffs_mean_mat
                prior_coeffs_sd_mat   <- model_args_list$prior_coeffs_sd_mat
                
                ## these vars need conversion to different types compatible w/ Stan
                corr_force_positive <- as.integer(model_args_list$corr_force_positive)
                prior_only <-  as.integer(model_args_list$prior_only)
                
                Phi_type <- model_args_list$Phi_type
                print(paste("Phi_type = ", Phi_type))
                ##
                Phi_type_int <- ifelse(Phi_type == "Phi", 1, 2)
                
                overflow_threshold <-  model_args_list$overflow_threshold
                underflow_threshold <- model_args_list$underflow_threshold
                handle_numerical_issues <- if_null_then_set_to(model_args_list$handle_numerical_issues, 1)
                fully_vectorised <- if_null_then_set_to(model_args_list$fully_vectorised, 1)
                
                Phi_type <- model_args_list$Phi_type
                Phi_type_int <- ifelse(Phi_type == "Phi", 1, 2)
                
                n_pops <- if_null_then_set_to(model_args_list$n_pops, 1)
                pop <- if_null_then_set_to(model_args_list$pop,  c(rep(1, N)))
                prior_prev_a <- model_args_list$prior_prev_a
                prior_prev_b <- model_args_list$prior_prev_b
                ##
                if (!(is.matrix(prior_prev_a))) {
                  prior_prev_a <-  matrix(prior_prev_a, ncol = 1)
                }
                ##
                if (!(is.matrix(prior_prev_b))) {
                  prior_prev_b <-  matrix(prior_prev_b, ncol = 1)
                }
                ##
                Stan_data_list <- list(N = N,
                                       n_tests = n_tests,
                                       y = y,
                                       n_class = n_class,
                                       n_pops =  n_pops,
                                       pop =   pop,
                                       # # ##
                                       # n_covariates_max_nd = n_covariates_max_nd,
                                       # n_covariates_max_d = n_covariates_max_d,
                                       # n_covariates_max = n_covariates_max
                                       # X_nd = X_nd,
                                       # X_d = X_d,
                                       # n_covs_per_outcome = n_covariates_per_outcome_mat
                                       ##
                                       overflow_threshold =  overflow_threshold,
                                       underflow_threshold = underflow_threshold,
                                       ##
                                       prior_only = prior_only,
                                       ##
                                       prior_beta_mean = prior_coeffs_mean_mat,
                                       prior_beta_sd = prior_coeffs_sd_mat,
                                       ##
                                       prior_prev_a =  prior_prev_a,
                                       prior_prev_b = prior_prev_b,
                                       ##
                                       Phi_type =  Phi_type_int,
                                       handle_numerical_issues =  handle_numerical_issues,
                                       fully_vectorised =  fully_vectorised,
                                       ##
                                       LT_b_priors_1 = LT_b_priors_1,
                                       LT_b_priors_2 = LT_b_priors_2,
                                       LT_known_bs_values = LT_known_bs_values,
                                       LT_known_bs_indicator = LT_known_bs_indicator
                                       )
          
        } else if (Model_type == "LC_MVOP") {
          
                prior_only <-  as.integer(model_args_list$prior_only)
                ##
                prior_coeffs_mean_mat <- model_args_list$prior_coeffs_mean_mat
                prior_coeffs_sd_mat <- model_args_list$prior_coeffs_sd_mat
                ##
                corr_force_positive <- as.integer(model_args_list$corr_force_positive)
                lb_corr <- model_args_list$lb_corr
                ub_corr <- model_args_list$ub_corr
                ##
                prior_LKJ <- model_args_list$prior_LKJ
                if (!(is.matrix(prior_LKJ))) {
                  prior_LKJ <-  matrix(prior_LKJ, ncol = 1)
                }
                ##
                overflow_threshold <-  model_args_list$overflow_threshold
                underflow_threshold <- model_args_list$underflow_threshold
                handle_numerical_issues <- if_null_then_set_to(model_args_list$handle_numerical_issues, 1)
                fully_vectorised <- if_null_then_set_to(model_args_list$fully_vectorised, 1)
                ##
                Phi_type <- model_args_list$Phi_type
                Phi_type_int <- ifelse(Phi_type == "Phi", 1, 2)
                ##
                n_pops <- if_null_then_set_to(model_args_list$n_pops, 1)
                pop <- if_null_then_set_to(model_args_list$pop,  c(rep(1, N)))
                ##
                prior_prev_a <- model_args_list$prior_prev_a
                prior_prev_b <- model_args_list$prior_prev_b
                ##
                if (!(is.matrix(prior_prev_a))) {
                  prior_prev_a <-  matrix(prior_prev_a, ncol = 1)
                }
                ##
                if (!(is.matrix(prior_prev_b))) {
                  prior_prev_b <-  matrix(prior_prev_b, ncol = 1)
                }
                ##
                ## ---- Ordinal-only stuff:
                ##
                n_binary_tests    <- model_args_list$n_binary_tests
                n_ordinal_tests   <- model_args_list$n_ordinal_tests
                ##
                n_cat_per_ord_test   <- model_args_list$n_cat_per_ord_test
                n_thr_per_ord_test   <- model_args_list$n_thr_per_ord_test
                ##
                prior_dirichlet_alpha <- model_args_list$prior_dirichlet_alpha
                ##
                print(paste("n_ordinal_tests = ", n_ordinal_tests))
                print(n_cat_per_ord_test)
                str(n_cat_per_ord_test)
                ##
                if (is.null(prior_dirichlet_alpha)) { 
                  prior_dirichlet_alpha <- rep(list(matrix(1.0, nrow = max(n_cat_per_ord_test), ncol = n_ordinal_tests)), n_class)
                }
                ##
                baseline_case_nd <- baseline_case_d <- list()
                ##
                for (t in 1:n_tests) {
                  baseline_case_nd[[t]] <- rep(0.0, n_covariates_max)
                  baseline_case_d[[t]]  <- rep(0.0, n_covariates_max)
                  ##
                  baseline_case_nd[[t]][1] <- 1.0
                  baseline_case_d[[t]][1]  <- 1.0
                }
                ##
                C_raw_lower = model_args_list$C_raw_lower
                C_raw_upper = model_args_list$C_raw_upper
                ##
                print(paste0("model_args_list$C_raw_lower = ", model_args_list$C_raw_lower))
                # print(-5.0 %||% NULL)   ## sanity check the operator exists
                ##
                ## ---- Make "Stan_data_list":
                ##
                Stan_data_list <- list(N = N,
                                       n_tests = n_tests,
                                       ##
                                       n_binary_tests = n_binary_tests, ## ---- ordinal-only (int)
                                       n_ordinal_tests = n_ordinal_tests, ## ---- ordinal-only (int)
                                       ##
                                       n_cat_per_ord_test = n_cat_per_ord_test, ## ---- ordinal-only (col vec, int)
                                       n_thr_per_ord_test = n_thr_per_ord_test, ## ---- ordinal-only (col vec, int)
                                       ##
                                       prior_dirichlet_alpha = prior_dirichlet_alpha, ## ---- ordinal-only (col vec, double)
                                       ##
                                       y = y,
                                       n_class = n_class,
                                       n_pops =  n_pops,  ## multi-pop not supported yet (currently only in Stan version)
                                       pop = pop,
                                       ##
                                       test_perm = as.integer(model_args_list$test_perm %||% 1:n_tests),
                                       n_cat_per_test = as.integer(model_args_list$n_cat_per_test), ## ---- ordinal-only
                                       ##
                                       n_covariates_max_nd = n_covariates_max_nd,
                                       n_covariates_max_d = n_covariates_max_d,
                                       n_covariates_max = n_covariates_max,
                                       X_nd = X_nd,
                                       X_d = X_d,
                                       n_covs_per_outcome = n_covariates_per_outcome_mat,
                                       ##
                                       corr_force_positive = corr_force_positive,
                                       known_num = known_num,
                                       ##
                                       # known_values_indicator = known_values_indicator_list,
                                       known_values_indicator_list = known_values_indicator_list,
                                       ##
                                       # known_values = known_values_list,
                                       known_values_list = known_values_list,
                                       ##
                                       lb_corr = lb_corr,
                                       ub_corr = ub_corr,
                                       prior_LKJ = prior_LKJ,
                                       lkj_cholesky_eta = prior_LKJ,
                                       ##
                                       overflow_threshold =  overflow_threshold,
                                       underflow_threshold = underflow_threshold,
                                       ##
                                       C_raw_lower = C_raw_lower, ## ---- ordinal-only
                                       C_raw_upper = C_raw_upper, ## ---- ordinal-only
                                       ##
                                       prior_only = prior_only,
                                       ##
                                       prior_beta_mean = prior_coeffs_mean_mat,
                                       prior_beta_sd = prior_coeffs_sd_mat,
                                       ##
                                       prior_prev_a = prior_prev_a,
                                       prior_prev_b = prior_prev_b,
                                       ##
                                       Phi_type =  Phi_type_int,
                                       handle_numerical_issues = handle_numerical_issues,
                                       fully_vectorised = fully_vectorised,
                                       ##
                                       baseline_case_nd = baseline_case_nd,
                                       baseline_case_d = baseline_case_d)
                
                # print(paste("Stan_data_list = "))
                # print(str(Stan_data_list))
          
        } else if (Model_type == "MVOP") {
          
                prior_only <-  as.integer(model_args_list$prior_only)
                ##
                prior_coeffs_mean_mat <- model_args_list$prior_coeffs_mean_mat
                prior_coeffs_sd_mat   <- model_args_list$prior_coeffs_sd_mat
                ##
                corr_force_positive <- as.integer(model_args_list$corr_force_positive)
                ub_corr <- model_args_list$ub_corr
                lb_corr <- model_args_list$lb_corr
                ##
                prior_LKJ <- model_args_list$prior_LKJ
                if (!(is.matrix(prior_LKJ))) {
                  prior_LKJ <-  matrix(prior_LKJ, ncol = 1)
                }
                ##
                overflow_threshold <-  model_args_list$overflow_threshold
                underflow_threshold <- model_args_list$underflow_threshold
                handle_numerical_issues <- if_null_then_set_to(model_args_list$handle_numerical_issues, 1)
                fully_vectorised <- if_null_then_set_to(model_args_list$fully_vectorised, 1)
                ##
                Phi_type <- model_args_list$Phi_type
                Phi_type_int <- ifelse(Phi_type == "Phi", 1, 2)
                ##
                ## ---- Ordinal-only stuff:
                ##
                n_binary_tests    <- model_args_list$n_binary_tests
                n_ordinal_tests   <- model_args_list$n_ordinal_tests
                ##
                n_cat_per_ord_test   <- model_args_list$n_cat_per_ord_test
                n_thr_per_ord_test   <- model_args_list$n_thr_per_ord_test
                ##
                prior_dirichlet_alpha <- model_args_list$prior_dirichlet_alpha
                ##
                if (is.null(prior_dirichlet_alpha)) { 
                  prior_dirichlet_alpha <- matrix(1.0, nrow = max(n_cat_per_ord_test), ncol = n_ordinal_tests)
                }
                ##
                baseline_case_nd <- baseline_case_d <- list()
                ##
                for (t in 1:n_tests) {
                  baseline_case_nd[[t]] <- rep(0.0, n_covariates_max)
                  baseline_case_d[[t]]  <- rep(0.0, n_covariates_max)
                  ##
                  baseline_case_nd[[t]][1] <- 1.0
                  baseline_case_d[[t]][1]  <- 1.0
                }
                ##
                C_raw_lower = model_args_list$C_raw_lower
                C_raw_upper = model_args_list$C_raw_upper
                ##
                ## ---- Make "Stan_data_list":
                ##
                Stan_data_list <- list(N = N,
                                       n_tests = n_tests,
                                       ##
                                       n_binary_tests = n_binary_tests,   ## ---- ordinal-only (int)
                                       n_ordinal_tests = n_ordinal_tests, ## ---- ordinal-only (int)
                                       ##
                                       n_cat_per_ord_test = n_cat_per_ord_test, ## ---- ordinal-only (col vec, int)
                                       n_thr_per_ord_test = n_thr_per_ord_test, ## ---- ordinal-only (col vec, int)
                                       ##
                                       prior_dirichlet_alpha = prior_dirichlet_alpha, ## ---- ordinal-only (col vec, double)
                                       ##
                                       y = y,
                                       ##
                                       test_perm = as.integer(model_args_list$test_perm %||% 1:n_tests),
                                       n_cat_per_test = as.integer(model_args_list$n_cat_per_test), ## ---- ordinal-only
                                       ##
                                       n_covariates_max = n_covariates_max,
                                       X = X,
                                       n_covs_per_outcome = n_covariates_per_outcome_mat,
                                       ##
                                       corr_force_positive = corr_force_positive,
                                       known_num = known_num,
                                       known_values_indicator_list = known_values_indicator_list,
                                       known_values_list = known_values_list,
                                       lb_corr = lb_corr,
                                       ub_corr = ub_corr,
                                       prior_LKJ = prior_LKJ,
                                       ##
                                       overflow_threshold =  overflow_threshold,
                                       underflow_threshold = underflow_threshold,
                                       ##
                                       C_raw_lower = C_raw_lower, ## ---- ordinal-only
                                       C_raw_upper = C_raw_upper, ## ---- ordinal-only
                                       ##
                                       prior_only = prior_only,
                                       ##
                                       prior_beta_mean = prior_coeffs_mean_mat,
                                       prior_beta_sd = prior_coeffs_sd_mat,
                                       ##
                                       Phi_type =  Phi_type_int,
                                       handle_numerical_issues =  handle_numerical_issues,
                                       fully_vectorised =  fully_vectorised,
                                       ##
                                       baseline_case_nd = baseline_case_nd,
                                       baseline_case_d = baseline_case_d)
          
        }

        return(Stan_data_list)
  
}










# 
# 


