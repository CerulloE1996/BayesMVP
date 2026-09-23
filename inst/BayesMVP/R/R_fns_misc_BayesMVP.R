
## R_fns_misc_BayesMVP.R



 
# auto_reorder_tests <- function(Y) {
# 
#         K <- ncol(Y)
#         R <- cor(Y)
#         diag(R) <- 0
# 
#         placed <- integer(K)
#         remaining <- 1:K
# 
#         # Root = node with highest total squared correlation (most "central")
#         total_r2 <- colSums(R^2)
#         placed[1] <- which.max(total_r2)
#         remaining <- setdiff(remaining, placed[1])
# 
#         # Second = most correlated with root
#         placed[2] <- remaining[which.max(abs(R[placed[1], remaining]))]
#         remaining <- setdiff(remaining, placed[2])
# 
#         # Greedy fill: max sum of R^2 with placed tests
#         for (pos in 3:K) {
#           scores <- sapply(remaining, function(j) sum(R[j, placed[1:(pos-1)]]^2))
#           placed[pos] <- remaining[which.max(scores)]
#           remaining <- setdiff(remaining, placed[pos])
#         }
# 
#         return(placed)
# 
# }


#' @export
auto_reorder_tests_MVP <- function(Y) {
        
        R <- psych::tetrachoric(Y)$rho  # or polycor::hetcor
        diag(R) <- 0
        K <- ncol(Y)
        placed <- integer(K)
        max_idx <- which(abs(R) == max(abs(R)), arr.ind = TRUE)[1, ]
        placed[1] <- max_idx[1]
        placed[2] <- max_idx[2]
        remaining <- setdiff(1:K, placed[1:2])
        
        for (pos in 3:K) {
          scores <- sapply(remaining, function(j) sum(R[j, placed[1:(pos-1)]]^2))
          placed[pos] <- remaining[which.max(scores)]
          remaining <- setdiff(remaining, placed[pos])
        }
        
        placed
        
}


#' @export
auto_reorder_tests <- function(Y) {

        K <- ncol(Y)
        R <- cor(Y)
        diag(R) <- 0

        placed <- integer(K)
        remaining <- 1:K

        # Root pair = strongest pairwise correlation
        max_idx <- which(abs(R) == max(abs(R)), arr.ind = TRUE)[1, ]
        placed[1] <- max_idx[1]
        placed[2] <- max_idx[2]
        remaining <- setdiff(remaining, placed[1:2])

        # Greedy fill: max sum of R^2 with placed tests
        for (pos in 3:K) {
          scores <- sapply(remaining, function(j) sum(R[j, placed[1:(pos-1)]]^2))
          placed[pos] <- remaining[which.max(scores)]
          remaining <- setdiff(remaining, placed[pos])
        }

        return(placed)

}


# ==============================================================================
# Automatic test reordering for LDL correlation parameterisation
# ==============================================================================
#
# Flow:
#   1. Short pre-burnin (50 iter) to find approximate mode
#   2. BridgeStan param_constrain() to get Omega_hat (no manual index tracking)
#   3. Compute optimal permutation
#   4. If non-identity: permute Y/lb/ub/known_values, zero Omega params
#   5. Full burnin with correctly ordered data
#   6. Sampling
#   7. Un-permute posteriors
# ==============================================================================


#' Compute optimal test ordering from estimated correlation matrices
#' Dissmann (2013) greedy pair-first algorithm for C-vine structure selection.
#' @export
compute_optimal_test_order <- function(Omega_hat_list) {
  
        max_abs_r <- sapply(Omega_hat_list, function(O) {
          R <- abs(O); diag(R) <- 0; max(R)
        })
        
        R <- Omega_hat_list[[which.max(max_abs_r)]]
        diag(R) <- 0
        K <- nrow(R)
        if (K <= 2) return(1:K)
        
        placed <- integer(K)
        remaining <- 1:K
        max_idx <- which(abs(R) == max(abs(R)), arr.ind = TRUE)[1, ]
        placed[1] <- max_idx[1]
        placed[2] <- max_idx[2]
        remaining <- setdiff(remaining, placed[1:2])
        
        for (pos in 3:K) {
          scores <- sapply(remaining, function(j) sum(R[j, placed[1:(pos - 1)]]^2))
          placed[pos] <- remaining[which.max(scores)]
          remaining <- setdiff(remaining, placed[pos])
        }
        
        placed
        
}





#' Infer the dimensions of the "Omega" parameter(s) of a compiled BridgeStan model
#'
#' The column-reordering pre-burnin needs to know how many latent classes and
#' how many tests/outcomes the model's correlation matrix has. For the built-in
#' models this is known from 'model_args_list'; for a USER-SUPPLIED Stan model
#' nothing is known about the model apart from its parameter names, so the
#' dimensions are read off the (transformed) parameter names directly:
#' "Omega[c,i,j]" -> n_class = max(c), n_tests = max(i, j);
#' "Omega[i,j]"   -> n_class = 1,      n_tests = max(i, j).
#'
#' @param bs_model BridgeStan model object
#' @return list(n_class, n_tests, n_found) - all NULL/0 if no "Omega" parameter exists
#' @export
infer_Omega_dims_from_bs_model <- function(bs_model) {

        pnames <- tryCatch(bs_model$param_names(include_tp = TRUE, include_gq = TRUE),
                           error = function(e) character(0))
        ##
        ## exact name "Omega" only - a bare grepl("Omega") also matched Omega_unconstrained_vec, L_Omega, Omega_orig, ...
        omega_names <- pnames[sub("(\\.|\\[).*$", "", pnames) == "Omega"]
        ##
        if (length(omega_names) == 0) {
          return(list(n_class = NULL, n_tests = NULL, n_found = 0L))
        }
        ##
        n_class <- 1L
        n_tests <- 1L
        ##
        for (nm in omega_names) {
              nums <- as.integer(regmatches(nm, gregexpr("[0-9]+", nm))[[1]])
              if (length(nums) == 3) {
                n_class <- max(n_class, nums[1])
                n_tests <- max(n_tests, nums[2], nums[3])
              } else if (length(nums) == 2) {
                n_tests <- max(n_tests, nums[1], nums[2])
              }
        }
        ##
        list(n_class = as.integer(n_class),
             n_tests = as.integer(n_tests),
             n_found = length(omega_names))

}





#' Decide whether 'reorder_cols_MVP' may be used with a user-supplied Stan model
#'
#' For the built-in models BayesMVP owns the data layout, so it knows which
#' columns of which objects are test-indexed. For an EXTERNAL Stan model it does
#' not: the only thing it can permute is the outcome matrix in the user-supplied
#' Stan data, which MUST therefore be present and be called "y" (an N x n_tests
#' matrix, n_tests >= 2). If it is absent (or is not a usable matrix), the option
#' is REFUSED - the run continues WITHOUT reordering rather than permuting
#' something arbitrary.
#'
#' @param Stan_data_list the user-supplied Stan data list
#' @return TRUE if reordering may proceed, FALSE if it must be disabled
#' @export
check_reorder_cols_MVP_for_Stan <- function(Stan_data_list) {

        ##
        ## ---- Caveat banner - printed whenever reordering is REQUESTED for an
        ## external Stan model, whether or not it ends up being used:
        ##
        big_warning_banner(
              title = "this only has been tested to work and/or be beneficial for multivariate probit-based models",
              body = c("'reorder_cols_MVP = TRUE' with Model_type = 'Stan' reorders the columns of the",
                       "outcome matrix 'y' in your Stan data (Dissmann pair-first ordering of the",
                       "estimated correlation matrix), then re-initialises and fits the model on the",
                       "permuted data.",
                       "",
                       "It requires your Stan model to: (i) take the outcome matrix as data named 'y',",
                       "with one COLUMN per test/outcome; and (ii) expose a correlation matrix named",
                       "'Omega' (parameter, transformed parameter or generated quantity).",
                       "",
                       "NOTE: test-indexed OUTPUTS are NOT un-permuted for external Stan models (there is",
                       "no way to know which of your parameters are test-indexed): slot j of any",
                       "test-indexed parameter corresponds to your ORIGINAL test test_perm[j]. The",
                       "permutation used is returned as 'test_perm' (and its inverse as 'test_inv_perm').",
                       "Any user-supplied test-indexed INITIAL VALUES are likewise left un-permuted."),
              warn = FALSE)
        ##
        y <- Stan_data_list$y
        ##
        if (is.null(y)) {
              big_warning_banner(
                    title = "y must be supplied",
                    body = c("'reorder_cols_MVP = TRUE' was requested with Model_type = 'Stan', but there is",
                             "NO variable called 'y' in your Stan data list / data file.",
                             "",
                             "Column reordering for external Stan models permutes the columns of 'y' and",
                             "nothing else, so without it there is nothing to reorder.",
                             "",
                             "==> reorder_cols_MVP has been DISABLED for this run; sampling continues with",
                             "    your data in its original column order."))
              return(FALSE)
        }
        ##
        if (!(is.matrix(y) || (is.array(y) && length(dim(y)) == 2L)) || (ncol(y) < 2L)) {
              big_warning_banner(
                    title = "y must be supplied",
                    body = c("'reorder_cols_MVP = TRUE' was requested with Model_type = 'Stan', and a variable",
                             "called 'y' WAS found in your Stan data - but it is not an N x n_tests matrix",
                             "with at least 2 columns.",
                             paste0("    found: class = ", paste(class(y), collapse = "/"),
                                    ", dim = ", paste(if (is.null(dim(y))) length(y) else dim(y), collapse = " x ")),
                             "",
                             "Column reordering needs the outcome matrix with one COLUMN per test/outcome.",
                             "",
                             "==> reorder_cols_MVP has been DISABLED for this run; sampling continues with",
                             "    your data in its original column order."))
              return(FALSE)
        }
        ##
        ## ---- REFUSAL: any OTHER entry in the Stan data with a dimension of n_tests may
        ## be test-indexed - per-test priors, covariates, flags, correlation bounds, the
        ## induced-Dirichlet alpha, and so on. BayesMVP permutes 'y' and NOTHING else,
        ## because for an external model it cannot know which of the user's data objects
        ## are indexed by test nor along which margin. If any such entry exists, reordering
        ## 'y' alone would silently DESYNC the responses from their own metadata - test j's
        ## data scored against test k's priors - which is far worse than not reordering at
        ## all. So this is refused outright rather than warned about.
        ##
        ## The scan is deliberately conservative: it flags anything merely SHAPED like
        ## n_tests (so e.g. a matrix whose n_class happens to equal n_tests is flagged
        ## too). Over-flagging costs a reordering; under-flagging corrupts a fit.
        ##
        n_tests <- ncol(y)
        ##
        candidate_names <- character(0)
        candidate_descriptions <- character(0)
        ##
        for (entry_name in setdiff(x = names(x = Stan_data_list), y = "y")) {

              entry <- Stan_data_list[[entry_name]]
              ##
              entry_dim <- if (is.null(x = dim(x = entry))) length(x = entry) else dim(x = entry)
              ##
              if (any(entry_dim == n_tests)) {
                    candidate_names <- c(candidate_names, entry_name)
                    candidate_descriptions <- c( candidate_descriptions,
                                                 paste0("    ", entry_name, "  [",
                                                        paste(entry_dim, collapse = " x "), "]"))
              }

        }
        ##
        if (length(x = candidate_names) > 0L) {

              shown_descriptions <- utils::head(x = candidate_descriptions, n = 12L)
              ##
              if (length(x = candidate_descriptions) > 12L) {
                shown_descriptions <- c( shown_descriptions,
                                         paste0("    ... and ",
                                                length(x = candidate_descriptions) - 12L, " more"))
              }
              ##
              big_warning_banner(
                    title = "reorder_cols_MVP: other test-indexed data would be left behind",
                    body = c( paste0("Reordering permutes the columns of 'y' and NOTHING ELSE. Your Stan data has ",
                                     length(x = candidate_names), " other"),
                              paste0("entries shaped like n_tests = ", n_tests,
                                     ", any of which may be indexed by test:"),
                              "",
                              shown_descriptions,
                              "",
                              "If any of those ARE test-indexed, permuting 'y' on its own would pair each test's",
                              "responses with a DIFFERENT test's priors / covariates / flags, silently, and the",
                              "fit would be wrong rather than merely un-reordered.",
                              "",
                              "==> reorder_cols_MVP has been DISABLED for this run; sampling continues with",
                              "    your data in its original column order.",
                              "",
                              "Column reordering of an external Stan model is only safe when 'y' is the ONLY",
                              "test-indexed thing in the data. (The built-in LC_MVP model permutes all of its",
                              "test-indexed arguments together, which is why it is supported there.)"))
              ##
              return(FALSE)

        }
        ##
        TRUE

}



#' Extract Omega correlation matrices from BridgeStan constrained output
#' @param bs_model BridgeStan model object
#' @param theta_main unconstrained main parameter vector (median across chains)
#' @param n_tests number of tests K
#' @param n_class number of latent classes
#' @return list of K x K correlation matrices (one per class)
#' @export
extract_Omega_via_bridgestan <- function(bs_model, 
                                         theta_main, 
                                         n_tests, 
                                         n_class) {
  
        # Get constrained parameters and their names
        constrained <- bs_model$param_constrain(theta_main,
                                                include_tp = TRUE,
                                                include_gq = TRUE,
                                                rng = bs_model$new_rng(seed = 1234L))
        pnames <- bs_model$param_names(include_tp = TRUE, include_gq = TRUE)
        
        # Find the Omega entries by name
        # Typical naming: "Omega[c,i,j]" for 3D array or "Omega.c.i.j"
        # Match the variable named EXACTLY "Omega". A bare grepl("Omega") also matches
        # Omega_unconstrained_vec, L_Omega, Omega_orig and L_Omega_orig; the last of these
        # (a Cholesky factor, not a correlation matrix) overwrote the others, so the test
        # order was computed from Cholesky entries instead of correlations.
        omega_mask <- sub("(\\.|\\[).*$", "", pnames) == "Omega"

        if (sum(omega_mask) == 0) {
          stop("No parameter named exactly 'Omega' found in BridgeStan param_names(). ",
               "Check your skeleton Stan model's transformed parameters block.")
        }
        
        omega_names <- pnames[omega_mask]
        omega_vals  <- constrained[omega_mask]
        
        # Parse indices from names like "Omega[1,2,3]" or "Omega.1.2.3"
        Omega_hat_list <- vector("list", n_class)
        for (c_idx in 1:n_class) {
          Omega_hat_list[[c_idx]] <- diag(n_tests)
        }
        
        for (k in seq_along(omega_names)) {
          nm <- omega_names[k]
          # Extract numeric indices from name
          nums <- as.integer(regmatches(nm, gregexpr("[0-9]+", nm))[[1]])
          
          if (length(nums) == 3) {
            # 3D array: Omega[class, row, col]
            c_idx <- nums[1]; i <- nums[2]; j <- nums[3]
          } else if (length(nums) == 2) {
            # 2D matrix (single class): Omega[row, col]
            c_idx <- 1; i <- nums[1]; j <- nums[2]
          } else {
            next
          }
          if (c_idx >= 1 && c_idx <= n_class && i >= 1 && i <= n_tests && j >= 1 && j <= n_tests) {
            Omega_hat_list[[c_idx]][i, j] <- omega_vals[k]
          }
        }
        
        Omega_hat_list
  
}


#' Position of every (observation, test) nuisance entry under a given chunking
#'
#' The built-in C++ log densities (MVP, MVOP, LC_LT) store the nuisance vector CHUNK BY CHUNK:
#' chunk k occupies the contiguous segment starting at row_start_k * n_tests, laid out column-major
#' as a (chunk_size_k x n_tests) block, e.g. MVP_lp_grad_MD_AD_fns_WCP.hpp:
#'     theta_us_vec_ref.segment(row_start * n_tests, chunk_size * n_tests).reshaped(chunk_size, n_tests)
#' so which (observation, test) an entry belongs to depends on the chunk size. The chunk sizes here
#' mirror calculate_chunk_sizes() in src/general_functions/structures.hpp exactly.
#'
#' @param N number of observations
#' @param n_tests number of tests (outcomes)
#' @param n_chunks number of chunks requested (Model_args_ints[4])
#' @param vect_type the vectorisation string the C++ reads (Model_args_strings[1])
#' @return N x n_tests integer matrix; entry [n, t] is the 1-based position of (n, t) in the nuisance vector
#' @keywords internal
fn_nuisance_chunk_layout_positions <- function( N,
                                                n_tests,
                                                n_chunks,
                                                vect_type) {

        ## ---- chunk sizes, exactly as calculate_chunk_sizes() in src/general_functions/structures.hpp:
        ##
        vec_size <- switch(as.character(vect_type), "AVX512" = 8L, "AVX2" = 4L, "AVX" = 2L, 1L)
        N        <- as.integer(N)
        n_tests  <- as.integer(n_tests)
        effective_n_chunks <- max(as.integer(n_chunks), 1L)
        ##
        normal_chunk_size <- vec_size * (N %/% (vec_size * effective_n_chunks))
        if (normal_chunk_size == 0L) normal_chunk_size <- (N %/% vec_size) * vec_size
        ##
        if (normal_chunk_size == 0L) {   ## N < vec_size: one scalar chunk
          chunk_row_starts <- 0L
          chunk_sizes      <- N
        } else {
          n_full_chunks    <- N %/% normal_chunk_size
          last_chunk_size  <- N - n_full_chunks * normal_chunk_size
          chunk_row_starts <- (seq_len(n_full_chunks) - 1L) * normal_chunk_size
          chunk_sizes      <- rep(normal_chunk_size, n_full_chunks)
          if (last_chunk_size > 0L) {
            chunk_row_starts <- c(chunk_row_starts, n_full_chunks * normal_chunk_size)
            chunk_sizes      <- c(chunk_sizes, last_chunk_size)
          }
        }
        ##
        ## ---- chunk c holds rows [row_start, row_start + chunk_size) as a column-major (chunk_size x n_tests)
        ##      block starting at position row_start * n_tests of the nuisance vector:
        ##
        position_of_observation_and_test <- matrix(NA_integer_, nrow = N, ncol = n_tests)
        for (chunk_index in seq_along(chunk_row_starts)) {
              chunk_row_start <- chunk_row_starts[chunk_index]
              chunk_size      <- chunk_sizes[chunk_index]
              rows_in_chunk   <- chunk_row_start + seq_len(chunk_size)
              for (test_index in seq_len(n_tests)) {
                position_of_observation_and_test[rows_in_chunk, test_index] <- chunk_row_start * n_tests + (test_index - 1L) * chunk_size + seq_len(chunk_size)
              }
        }
        stopifnot(!anyNA(position_of_observation_and_test), !anyDuplicated(as.vector(position_of_observation_and_test)))
        position_of_observation_and_test

}


#' Permutation taking a nuisance vector from one chunk layout to another
#'
#' Returns remap_index such that nuisance_vec_in_new_layout <- nuisance_vec_in_old_layout[remap_index]; apply it to
#' every nuisance-indexed quantity when the number of chunks changes (e.g. between burn-in and sampling).
#' Identity when the two layouts agree.
#' @keywords internal
fn_nuisance_chunk_remap_index <- function( N,
                                           n_tests,
                                           n_chunks_from,
                                           n_chunks_to,
                                           vect_type) {

        positions_in_old_layout <- fn_nuisance_chunk_layout_positions(N = N, n_tests = n_tests, n_chunks = n_chunks_from, vect_type = vect_type)
        positions_in_new_layout <- fn_nuisance_chunk_layout_positions(N = N, n_tests = n_tests, n_chunks = n_chunks_to,   vect_type = vect_type)
        remap_index <- integer(N * n_tests)
        remap_index[as.vector(positions_in_new_layout)] <- as.vector(positions_in_old_layout)
        remap_index

}


#' Permute correlation-related model data
#' @export
permute_corr_data <- function(lb_corr, ub_corr,
                              known_values_indicator, known_values, perm) {
  list(
    lb_corr = lapply(lb_corr, function(m) m[perm, perm]),
    ub_corr = lapply(ub_corr, function(m) m[perm, perm]),
    known_values_indicator = lapply(known_values_indicator, function(m) m[perm, perm]),
    known_values = lapply(known_values, function(m) m[perm, perm])
  )
}


#' Zero out Omega unconstrained params in theta_main vectors.
#' Uses BridgeStan param_names to find Omega-related raw params automatically.
#' @export
zero_omega_in_theta <- function(theta_main_vectors, 
                                bs_model) {
  
        # Unconstrained param names (the raw parameters)
        raw_names <- bs_model$param_unc_names()
        omega_raw_mask <- grepl("Omega|corr|omega", raw_names, ignore.case = TRUE)
        
        if (sum(omega_raw_mask) == 0) {
          warning("Could not identify Omega unconstrained params by name. ",
                  "Zeroing nothing. You may need to adapt this function to your ",
                  "Stan model's parameter naming convention.")
          return(theta_main_vectors)
        }
        
        omega_indices <- which(omega_raw_mask)
        
        theta_main_vectors[omega_indices, ] <- 0.0
        
        # for (ch in seq_along(theta_main_vectors)) {
        #   theta_main_vectors[[ch]][omega_indices] <- 0.0
        # }
        
        theta_main_vectors
  
}


#' Un-permute posterior correlation matrices
#' @export
unpermute_Omega <- function(Omega_samples, 
                            inv_perm) {
  
        if (is.list(Omega_samples)) {
          return(lapply(Omega_samples, function(m) m[inv_perm, inv_perm]))
        }
        if (is.matrix(Omega_samples)) {
          return(Omega_samples[inv_perm, inv_perm])
        }
        if (is.array(Omega_samples) && length(dim(Omega_samples)) == 3) {
          out <- Omega_samples
          for (s in seq_len(dim(out)[3])) out[, , s] <- Omega_samples[inv_perm, inv_perm, s]
          return(out)
        }
        
        Omega_samples
  
}





#' expand_categorical_to_dummies
#' @description Convert an M-level categorical variable (integer-coded 1:M) to M-1 binary dummy indicators.
#'              Reference category is level 1 by default.
#' @param x Integer vector coded 1, 2, ..., M
#' @param ref_level Which level to use as reference (default = 1)
#' @param prefix Column name prefix (default = "cat")
#' @return Matrix with M-1 columns of 0/1 indicators
#' @export
expand_categorical_to_dummies <- function( x,
                                           ref_level = 1L,
                                           prefix = "cat"
) {
        
        levels <- sort(unique(x))
        M <- length(levels)
        stopifnot(M >= 2)
        stopifnot(ref_level %in% levels)
        
        non_ref_levels <- setdiff(levels, ref_level)
        out <- matrix(0L, nrow = length(x), ncol = M - 1L)
        colnames(out) <- paste0(prefix, "_", non_ref_levels)
        
        for (j in seq_along(non_ref_levels)) {
          out[, j] <- as.integer(x == non_ref_levels[j])
        }
        
        return(out)
  
}





#' validate_y_tests_ordinal
#' @export
validate_y_tests_ordinal <- function(y) {
  
        y <- as.matrix(y)
        n_tests <- ncol(y)
        
        is_binary  <- logical(n_tests)
        is_ordinal <- logical(n_tests)
        
        for (t in seq_len(n_tests)) {
          
              vals <- sort(unique(y[, t]))
              
              if (length(vals) == 2 && all(vals == c(0, 1))) {
                is_binary[t] <- TRUE
                next
              }
              
              if (length(vals) == 2 && all(vals == c(0, 1))) {
                is_binary[t] <- TRUE
                next
              }
              
              if (0 %in% vals) {
                stop(sprintf("Column %d contains 0. Ordinal outcomes must be coded 1, 2, ..., K (not 0-indexed).", t))
              }
              
              K <- max(vals)
              
              if (K >= 3) {
                is_ordinal[t] <- TRUE
              } else if (K == 2) {
                is_binary[t] <- TRUE
              } else {
                stop(sprintf("Column %d has only one unique value (%d). Need at least 2 categories.", t, K))
              }
              
              # ## From here on, ordinal — no zeros allowed
              # if (0 %in% vals) {
              #   stop(sprintf("Column %d contains 0. Ordinal outcomes must be coded 1, 2, ..., K (not 0-indexed).", t))
              # }
              # 
              # K <- max(vals)
              # if (!identical(vals, seq_len(K))) {
              #   stop(sprintf("Column %d has non-contiguous categories: {%s}. Must be 1, 2, ..., K.",
              #                t, paste(vals, collapse = ", ")))
              # }
              # 
              # if (K >= 3) {
              #   is_ordinal[t] <- TRUE
              # } else {
              #   stop(sprintf("Column %d has only one unique value (%d). Need at least 2 categories.", t, K))
              # }
              
        }
        
        n_binary_tests  <- sum(is_binary)
        n_ordinal_tests <- sum(is_ordinal)
        
        ## Check ordering: all binary first, then all ordinal
        if (n_binary_tests > 0 && n_ordinal_tests > 0) {
              
              last_binary   <- max(which(is_binary))
              first_ordinal <- min(which(is_ordinal))
              if (first_ordinal < last_binary) {
                stop("Binary tests must come before ordinal tests in y. ",
                     sprintf("Found ordinal test in column %d but binary test in column %d.", 
                             first_ordinal, last_binary))
              }
          
        }
        
        list(n_binary_tests  = n_binary_tests,
             n_ordinal_tests = n_ordinal_tests,
             n_tests         = n_tests)
        
}


 
# compute_ordinal_info <- function(y) {
#   
#         y <- as.matrix(y)
#         ##
#         # n_tests <- ncol(y)
#         # n_ordinal_tests <- n_tests - n_binary_tests
#         ##
#         outs_y_validation <- validate_y_tests_ordinal(y)
#         ##
#         n_binary_tests  <- outs_y_validation$n_binary_tests
#         n_ordinal_tests <- outs_y_validation$n_ordinal_tests
#         n_tests         <- outs_y_validation$n_tests
#         
#         if (n_ordinal_tests == 0) {
#           return(list(
#             n_cat_per_ord_test   = integer(0),
#             n_thr_per_ord_test   = integer(0)
#           ))
#         }
#         
#         n_cat <- integer(n_ordinal_tests)
#         n_thr <- integer(n_ordinal_tests)
#         ##
#         for (t in seq_len(n_ordinal_tests)) {
#           col_idx <- n_binary_tests + t
#           n_cat[t] <- max(y[, col_idx], na.rm = TRUE)
#           n_thr[t] <- n_cat[t] - 1L
#         }
#         
#         list(
#           n_cat_per_ord_test   = as.integer(n_cat),
#           n_thr_per_ord_test   = as.integer(n_thr)
#         )
#   
# }


#' get_basic_dims_for_internal_models
#' @export
get_basic_dims_for_internal_models <- function( Model_type,
                                                y
) {
  
        ## For hard-coded models
        N <- nrow(y)
        n_tests <- ncol(y)
        ##
        n_class <- ifelse(grepl("LC", Model_type) || Model_type == "latent_trait", 2, 1)
        ##
        if (Model_type %in% c("LC_MVOP", "MVOP")) {
          
            outs_y_validation <- validate_y_tests_ordinal(y)
            ##
            n_binary_tests  <- outs_y_validation$n_binary_tests
            n_ordinal_tests <- outs_y_validation$n_ordinal_tests
            n_tests         <- outs_y_validation$n_tests
            
        } else {
          
            n_binary_tests  <- n_tests
            n_ordinal_tests <- 0
            n_tests         <- n_tests
          
        }
        ##
        return(list(
          N = N,
          n_class = n_class,
          n_tests = n_tests,
          ##
          n_binary_tests = n_binary_tests,
          n_ordinal_tests = n_ordinal_tests
        ))
        
}




#' get_dims_for_internal_models
#' @export
get_dims_for_internal_models <- function( y,
                                          Model_type,
                                          n_tests,
                                          n_class,
                                          n_pops,
                                          N,
                                          n_covariates_total,
                                          n_cat_per_ord_test = NULL   ## REQUIRED for MVOP/LC_MVOP; ignored otherwise
) {
  
        if (Model_type %in% c("MVP", "MVOP")) {
          n_corrs <- choose(n_tests, 2)
          n_params_main <- n_covariates_total + n_corrs
        } else if (Model_type %in% c("LC_MVP", "LC_MVOP")) {
          n_corrs <- n_class * choose(n_tests, 2)
          n_params_main <- n_pops + n_covariates_total + n_corrs
        } else if (Model_type == "latent_trait") {
          n_corrs <- n_tests * n_class
          n_params_main <- n_pops + n_covariates_total + n_corrs
        }
        
        if (Model_type %in% c("MVOP", "LC_MVOP")) {
          
              ##
              ## ---- n_cat_per_ord_test MUST be supplied by the user. It CANNOT be inferred:
              ##      the observed max UNDERCOUNTS whenever the top category is unobserved, and
              ##      n_params_main depends on it -- so an inferred value silently produces a
              ##      parameter vector that is SHORT and mismatches the Stan skeleton.
              ##
              if (is.null(n_cat_per_ord_test)) {
                stop(paste0("get_dims_for_internal_models: n_cat_per_ord_test MUST be supplied for ",
                            Model_type, ". It cannot be inferred from y (the top category is often ",
                            "unobserved, so the observed max undercounts)."))
              }
              ##
              n_cat_per_ord_test <- as.integer(n_cat_per_ord_test)
              ##
              if (any(is.na(n_cat_per_ord_test)) || any(n_cat_per_ord_test < 3L)) {
                stop(paste0("get_dims_for_internal_models: n_cat_per_ord_test must be integers >= 3 ",
                            "(2 == binary). Got: ", paste(n_cat_per_ord_test, collapse = ", ")))
              }
              ##
              n_thr_per_ord_test <- n_cat_per_ord_test - 1L
              ##
              if (Model_type == "LC_MVOP") {
                n_params_main <- n_pops + n_covariates_total + n_corrs + n_class * sum(n_thr_per_ord_test)
              } else if (Model_type == "MVOP") {
                n_params_main <- n_covariates_total + n_corrs + sum(n_thr_per_ord_test)   ## FIXED: was "n_thr_per_ord_t"
              }
          
        } else {
          
              n_cat_per_ord_test <- 0
              n_thr_per_ord_test <- 0
          
        }
        
        n_nuisance <- N * n_tests
        n_params <- n_params_main + n_nuisance
        
        return(list( n_params = n_params,
                     n_params_main = n_params_main,
                     n_nuisance = n_nuisance,
                     n_corrs = n_corrs,
                     n_cat_per_ord_test = n_cat_per_ord_test,
                     n_thr_per_ord_test = n_thr_per_ord_test))
  
}




#' process_covariates_X
#' @export
process_covariates_X <- function(Model_type,
                                 X, 
                                 N,
                                 n_tests,
                                 n_class
) {
        
        if (is.null(X)) {
          
              # Intercept-only model
              warning("X not provided - assuming intercept-only model")
              
              # Create n_covariates_per_outcome_mat - all 1's
              n_covariates_per_outcome_mat <- matrix(1, nrow = n_class, ncol = n_tests)
              
              # Create X as list of lists for C++ compatibility
              X <- list()
              for (c in 1:n_class) {
                X[[c]] <- list()
                for (t in 1:n_tests) {
                  X[[c]][[t]] <- matrix(1, nrow = N, ncol = 1)  # Intercept only
                }
              }
              
              n_covariates_max <- 1
              n_covariates_max_nd <- 1
              n_covariates_max_d  <- 1
              n_covariates_total <- n_class * n_tests
          
        } else {
          
              # X is provided - need to process it
              
              # Check if X is already in the correct format
              if ((Model_type %in% c("MVP", "MVOP")) && (!is.list(X[[1]]))) {
                  # X is provided as list of matrices - wrap it for C++ compatibility
                  X_temp <- X
                  X <- list()
                  X[[1]] <- X_temp
              }
              
              # Validate X structure
              if (!is.list(X) || length(X) != n_class) {
                stop("X must be a list of length n_class")
              }
              
              # Build n_covariates_per_outcome_mat from X
              n_covariates_per_outcome_mat <- matrix(NA, nrow = n_class, ncol = n_tests)
              
              for (c in 1:n_class) {
                
                    if (!is.list(X[[c]]) || length(X[[c]]) != n_tests) {
                      stop(paste("X[[", c, "]] must be a list of length n_tests"))
                    }
                    
                    for (t in 1:n_tests) {
                      if (!is.matrix(X[[c]][[t]])) {
                        stop(paste("X[[", c, "]][[", t, "]] must be a matrix"))
                      }
                      
                      if (nrow(X[[c]][[t]]) != N) {
                        stop(paste("X[[", c, "]][[", t, "]] must have N rows"))
                      }
                      
                      n_covariates_per_outcome_mat[c, t] <- ncol(X[[c]][[t]])
                    }
                
              }
              
              n_covariates_max <- max(n_covariates_per_outcome_mat)
              n_covariates_total <- sum(n_covariates_per_outcome_mat)
              ##
              if (Model_type %in% c("MVP", "MVOP")) {
                n_covariates_max_nd <- n_covariates_max
                n_covariates_max_d  <- n_covariates_max
              } else { 
                n_covariates_max_nd <- max(n_covariates_per_outcome_mat[1, ])
                n_covariates_max_d  <- max(n_covariates_per_outcome_mat[2, ])
              }

        }
        
        return(list(
          X = X,
          ##
          n_covariates_per_outcome_mat = n_covariates_per_outcome_mat,
          ##
          n_covariates_max_nd = n_covariates_max_nd,
          n_covariates_max_d = n_covariates_max_d,
          n_covariates_max = n_covariates_max,
          n_covariates_total = n_covariates_total
        ))
  
}







#' Convert X to padded array format for Stan 
#' @description Pads X matrices to n_covariates_max and converts to array format
#' @keywords internal
#' @export
convert_X_to_padded_for_Stan <- function(X, 
                                N,
                                n_tests, 
                                n_class, 
                                n_covariates_max, 
                                n_covariates_max_nd,
                                n_covariates_max_d,
                                n_covariates_per_outcome_mat, 
                                padding_value = -999) {
        
        # X is already a list of lists from process_covariates_X
        # X[[c]][[t]] is the matrix for class c, outcome t
        
        X_padded <- list()
        
        for (c in 1:n_class) {
          X_padded[[c]] <- list()
          
          # Determine max covariates for this class
          if (n_class == 1) {
            n_cov_max_this_class <- n_covariates_max
          } else {
            n_cov_max_this_class <- ifelse(c == 1, n_covariates_max_nd, n_covariates_max_d)
          }
          
          for (t in 1:n_tests) {
            # Get actual number of covariates
            n_cov_actual <- n_covariates_per_outcome_mat[c, t]
            
            # Create padded matrix
            X_padded_mat <- matrix(padding_value, nrow = N, ncol = n_cov_max_this_class)
            
            # Fill in actual values
            X_padded_mat[, 1:n_cov_actual] <- X[[c]][[t]]
            
            X_padded[[c]][[t]] <- X_padded_mat
          }
        }
        
        # Return in the same format for different model types
        if (n_class == 1) {
          # For MVP, return with single name
          return(list(X = X_padded[[1]]))
        } else {
          # For LC_MVP/latent_trait, split into X_nd and X_d
          return(list(
            X_nd = X_padded[[1]],  # List of matrices for class 1
            X_d = X_padded[[2]]    # List of matrices for class 2
          ))
        }
  
}
