
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

#' Print a large, hard-to-miss warning banner
#'
#' Used by the 'reorder_cols_MVP' machinery so that refusals / caveats are not
#' lost in the (very verbose) sampler output.
#'
#' @param title single-line headline (printed in CAPS between rules)
#' @param body character vector of additional lines (wrapped, may be empty)
#' @param warn if TRUE, also raise an R warning() with the same headline so the
#'        message survives into warnings() / worker error handling
#' @export
big_warning_banner <- function(title,
                               body = character(0),
                               warn = TRUE) {

        rule <- paste(rep("!", 100), collapse = "")
        ##
        message("\n", rule)
        message("!!!!  ", toupper(title))
        message(rule)
        ##
        for (line in body) {
          message("!!!!  ", line)
        }
        ##
        if (length(body) > 0) {
          message(rule)
        }
        message("")
        ##
        if (isTRUE(warn)) {
          warning(title, call. = FALSE, immediate. = TRUE)
        }
        ##
        invisible(NULL)

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



#' @export
remove_duplicates_by_name <- function(my_list) {
  
      if (is.null(names(my_list))) return(my_list)
      my_list[!duplicated(names(my_list))]
  
}


#' is_NaN_or_Inf_vec
#' @export
is_NaN_or_Inf_vec <- function(x) { 
  
        if ((any(is.infinite(x))) || (any(is.nan(x)))) { 
          return(TRUE)
        } else { 
          return(FALSE)
        }
  
}





#' convert_bridgestan_par_names_to_stan
#' @export
convert_bridgestan_par_names_to_stan <- function(names) {
        
        # Handle the conversion more carefully
        result <- names
        
        # Find first dot and replace with [
        result <- sub("\\.", "[", result)
        
        # Replace all remaining dots with commas
        result <- gsub("\\.", ",", result)
        
        # Add closing bracket if we added an opening one
        needs_bracket <- grepl("\\[", result) & !grepl("\\]", result)
        result[needs_bracket] <- paste0(result[needs_bracket], "]")
        
        return(result)
  
}





#' transform_stan_path
#' @export
transform_stan_path <- function(stan_path) {
  
        # Remove any leading ~/ if present
        path_no_tilde <- sub("^~/", "", stan_path)
        
        # Extract directory and filename
        dir_path <- dirname(path_no_tilde)
        filename <- basename(path_no_tilde)
        
        # Remove .stan extension and add _model.so or _model.dll
        if (.Platform$OS.type == "windows") {
          
          model_name <- sub("\\.stan$", "_model.dll", filename)
          final_path <- file.path(dir_path, model_name)
          
        } else { 
          
          model_name <- sub("\\.stan$", "_model.so", filename)
          final_path <- if (startsWith(x = dir_path, prefix = "/")) {
                            file.path(dir_path, model_name)
                        } else {
                            file.path("/", dir_path, model_name)
                        }
          
        }
        
        # final_path <- file.path("~", dir_path, model_name)
        
        return(final_path)
  
}




#' copy_json_with_worker_id
#' @export
copy_json_with_worker_id <- function(original_path, 
                                     ii) {
  
        # Create new path with worker ID
        base_path <- sub("\\.json$", "", original_path)
        new_path <- paste0(base_path, "_", ii, ".json")
        
        # Copy the file to the new path
        if (file.exists(original_path)) {
          file.copy(original_path, new_path, overwrite = TRUE)
        } else {
          stop(paste("Original file doesn't exist:", original_path))
        }
        
        return(new_path)
  
}



#' copy_.so_with_worker_id
#' @export
copy_.so_with_worker_id <- function(original_path, 
                                    ii) {
  
        # Create new path with worker ID
        base_path <- sub("\\.so$", "", original_path)
        new_path <- paste0(base_path, "_", ii, ".so")
        
        # Copy the file to the new path
        if (file.exists(original_path)) {
          file.copy(original_path, new_path, overwrite = TRUE)
        } else {
          stop(paste("Original file doesn't exist:", original_path))
        }
        
        return(new_path)
  
}



#' copy_.so_with_worker_id
#' @export
copy_.stan_with_worker_id <- function( original_path, 
                                       ii) {
  
        # Create new path with worker ID
        base_path <- sub("\\.stan$", "", original_path)
        new_path <- paste0(base_path, "_", ii, ".stan")
        
        # Copy the file to the new path
        if (file.exists(original_path)) {
          file.copy(original_path, new_path, overwrite = TRUE)
        } else {
          stop(paste("Original file doesn't exist:", original_path))
        }
        
        return(new_path)
  
}



#' if_null_then_set_to
#' @export
if_null_then_set_to <- function(x, 
                                set_to_thif_null, 
                                debugging = FALSE
)  {
  
        if (is.null(x)) {
          y <- set_to_thif_null
          return(y)
        } else { 
          y <- x
          return(y)
        }
        
        # if (debugging) print(paste(y))
  
  
}





#' Detect n_nuisance from Stan model (UNCONSTRAINED coordinates of the FIRST
#' declared parameter block)
#'
#' BayesMVP's convention: the nuisance / high-dimensional latent parameter block is
#' the FIRST declaration in the `parameters` block of the Stan model. Everything
#' after it is "main". The dimension must be counted in UNCONSTRAINED coordinates
#' (the space the sampler integrates over): a 4-category simplex occupies four
#' constrained names but only three unconstrained coordinates, so counting
#' constrained names would overstate the nuisance dimension.
#'
#' The first DECLARED parameter is identified from stanc compiler metadata
#' (`stanc --info`), not from the first returned name: if the first declaration has
#' ZERO length (e.g. an empty array under a switched-off branch) it contributes no
#' names at all and the first returned name belongs to the SECOND declaration -
#' a name-based heuristic would wrongly treat that second block as nuisance.
#' Comments, `#include`-d files and data-dependent dimensions are all handled by
#' stanc / BridgeStan rather than by a handwritten regex parser.
#'
#' `sample_nuisance = FALSE` means the model has NO nuisance block: every declared
#' parameter belongs to the main block and 0 is returned (this is distinct from a
#' nuisance declaration that exists but evaluates to zero length - see
#' detect_nuisance_block_info_from_stan_model()).
#'
#' @param bs_model a BridgeStan StanModel (with its data already loaded)
#' @param sample_nuisance whether a nuisance block exists and should be sampled
#' @param Stan_model_file_path path to the .stan source (used for stanc metadata)
#' @param stanc_args optional stanc arguments (e.g. list(include_paths = ...))
#' @export
detect_n_nuisance_from_stan_model <- function(bs_model, 
                                              sample_nuisance = TRUE,
                                              Stan_model_file_path = NULL,
                                              stanc_args = NULL) {
  
        if (!isTRUE(sample_nuisance)) {
          return(as.integer(0))
        }
        ##
        as.integer(detect_nuisance_block_info_from_stan_model(bs_model = bs_model,
                                                              Stan_model_file_path = Stan_model_file_path,
                                                              stanc_args = stanc_args)$n_nuisance)
  
}



#' Full nuisance-block metadata for an external Stan model
#'
#' Returns the UNCONSTRAINED nuisance dimension (n_nuisance), the number of
#' CONSTRAINED names belonging to the nuisance block (n_nuisance_names_constrained -
#' used to locate the main block inside constrained draws), and the declared base
#' name of the nuisance block (nuisance_base_name). See
#' detect_n_nuisance_from_stan_model() for the conventions.
#' @export
detect_nuisance_block_info_from_stan_model <- function(bs_model,
                                                       Stan_model_file_path = NULL,
                                                       stanc_args = NULL) {
  
        bs_names <- bs_model$param_names()
        bs_unc_names <- bs_model$param_unc_names()
        ##
        if (length(bs_unc_names) == 0) {
          return(list( n_nuisance = as.integer(0),
                       n_nuisance_names_constrained = as.integer(0),
                       nuisance_base_name = NA_character_))
        }
        ##
        fn_param_base <- function(bridgestan_names) {
          
              sub( pattern = "\\..*$",
                   replacement = "",
                   x = bridgestan_names)
              
        }
        ##
        ## ---- Identify the FIRST DECLARED parameter from stanc compiler metadata
        ## (declaration order, with comments / includes / data-dependent dims handled
        ## by the compiler itself):
        ##
        first_declared_param <- NULL
        first_declared_param_used_stanc <- FALSE
        ##
        if (!is.null(Stan_model_file_path) && file.exists(Stan_model_file_path)) {
          
              stanc_bin <- file.path( cmdstanr::cmdstan_path(),
                                      "bin",
                                      "stanc")
              ##
              if (file.exists(stanc_bin)) {
                
                    stanc_cmd_args <- c( "--info",
                                         shQuote(Stan_model_file_path))
                    ##
                    include_paths <- stanc_args$include_paths
                    ##
                    if (!is.null(include_paths)) {
                      stanc_cmd_args <- c( stanc_cmd_args,
                                           paste0( "--include-paths=",
                                                   paste( include_paths,
                                                          collapse = ",")))
                    }
                    ##
                    stanc_out <- tryCatch( expr = system2( command = stanc_bin,
                                                           args = stanc_cmd_args,
                                                           stdout = TRUE,
                                                           stderr = TRUE),
                                           error = function(e) NULL)
                    ##
                    stanc_json_text <- paste( stanc_out,
                                              collapse = "\n")
                    ##
                    if (!is.null(stanc_out) && length(stanc_out) > 0 &&
                        grepl( pattern = "\"parameters\"",
                               x = stanc_json_text,
                               fixed = TRUE)) {
                      
                          stanc_info <- jsonlite::fromJSON( txt = stanc_json_text,
                                                            simplifyVector = FALSE)
                          ##
                          if (!is.null(stanc_info$parameters) && length(stanc_info$parameters) > 0) {
                            first_declared_param <- names(stanc_info$parameters)[1]
                            first_declared_param_used_stanc <- TRUE
                          }
                          
                    }
              }
          
        }
        ##
        if (is.null(first_declared_param)) {
          ##
          ## fallback: first UNCONSTRAINED name (cannot distinguish a zero-length
          ## first declaration from "no first declaration", so the dimension of the
          ## first NON-EMPTY block is used):
          ##
          first_declared_param <- fn_param_base(bs_unc_names[1])
          ##
          warning( "detect_nuisance_block_info_from_stan_model: stanc metadata unavailable; ",
                   "the first declared parameter was inferred from the first unconstrained name ",
                   "('", first_declared_param, "'). A ZERO-LENGTH first declaration cannot be ",
                   "detected this way - supply a working cmdstan installation (stanc --info) or ",
                   "use n_nuisance_override.")
          
        }
        ##
        ## ---- count the coordinates belonging to that declaration:
        ##
        n_nuisance <- sum(fn_param_base(bs_unc_names) == first_declared_param)
        n_nuisance_names_constrained <- sum(fn_param_base(bs_names) == first_declared_param)
        ##
        list( n_nuisance = as.integer(n_nuisance),
              n_nuisance_names_constrained = as.integer(n_nuisance_names_constrained),
              nuisance_base_name = first_declared_param,
              first_declared_param_used_stanc = first_declared_param_used_stanc)
  
}





#' get_n_params_main_from_n_nuisance_using_Stan_bs_model
#' @export
get_n_params_main_from_n_nuisance_using_Stan_bs_model <- function(  n_nuisance, 
                                                                 Stan_data_list,
                                                                 Stan_model_file_path,
                                                                 stanc_args = NULL
) { 
  
  
        if (is.null(stanc_args)) { 
          outs <- init_bs_model_external(  Stan_data_list = Stan_data_list,
                                           Stan_model_file_path = Stan_model_file_path)
        } else { 
          outs <- init_bs_model_external(  Stan_data_list = Stan_data_list,
                                          Stan_model_file_path = Stan_model_file_path,
                                          stanc_args = stanc_args)
        }
        ##
        bs_model <- outs$bs_model
        ##
        bs_names  <- bs_model$param_names() ## bookmark_2 (previous code)
        bs_names_inc_tp <-  (bs_model$param_names(include_tp = TRUE))
        bs_names_inc_tp_and_gq <-  (bs_model$param_names(include_tp = TRUE, include_gq = TRUE))
        ##
        n_params <- length(bs_names)
        n_params_main <- n_params - n_nuisance
        ##
        return(list(n_params = n_params,
                    n_params_main = n_params_main,
                    n_nuisance = n_nuisance,
                    ##
                    bs_model = bs_model,
                    bs_names = bs_names,
                    bs_names_inc_tp = bs_names_inc_tp,
                    bs_names_inc_tp_and_gq = bs_names_inc_tp_and_gq))
  
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
        n_class <- ifelse(grepl("LC", Model_type), 2, 1)
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
          return(list(X = X_padded))
        } else {
          # For LC_MVP/latent_trait, split into X_nd and X_d
          return(list(
            X_nd = X_padded[[1]],  # List of matrices for class 1
            X_d = X_padded[[2]]    # List of matrices for class 2
          ))
        }
  
}





#' check_stan_data
#' @export
check_stan_data <- function(stan_model,
                            stan_data_list
) {
  
        # Get the data variables from the Stan model
        vars <- stan_model$variables()
        data_vars <- vars$data
        
        # Extract variable names from the data block
        required_vars <- names(data_vars)
        
        # Check which variables are in the stan_data_list
        provided_vars <- names(stan_data_list)
        
        # Find missing variables
        missing_vars <- setdiff(required_vars, provided_vars)
        
        # Find extra variables (not in model but in data list)
        extra_vars <- setdiff(provided_vars, required_vars)
        
        # Report results
        if (length(missing_vars) > 0) {
          missing_details <- character(length(missing_vars))
          
          # Add details about each missing variable
          for (i in seq_along(missing_vars)) {
            var_name <- missing_vars[i]
            var_info <- data_vars[[var_name]]
            var_type <- var_info$type
            var_dims <- var_info$dimensions
            
            # Format dimension info
            if (length(var_dims) == 0 || all(var_dims == 0)) {
              dim_str <- "scalar"
            } else {
              dim_str <- paste0(var_dims, "D array/matrix")
            }
            
            missing_details[i] <- sprintf("  - %s (type: %s, dimensions: %s)", 
                                          var_name, var_type, dim_str)
          }
          
          error_msg <- paste0(
            "Missing required Stan data variables:\n",
            paste(missing_details, collapse = "\n"),
            "\n\nPlease add these variables to your Stan_data_list."
          )
          
          stop(error_msg, call. = FALSE)
        }
        
        # Optional: warn about extra variables
        if (length(extra_vars) > 0) {
          warning(sprintf(
            "The following variables in Stan_data_list are not in the model's data block:\n  %s\n",
            paste(extra_vars, collapse = ", ")
          ), call. = FALSE)
        }
        
        # If we get here, all required variables are present
        message("✓ All required Stan data variables are present!")
        
        # Return summary
        invisible(list(
          required = required_vars,
          provided = provided_vars,
          missing = missing_vars,
          extra = extra_vars,
          check_passed = length(missing_vars) == 0
        ))
        
}










#' Get model dimensions (n_params, n_params_main, n_nuisance)
#' @export
get_model_info <- function(  Model_type, 
                             ##
                             stream,
                             ##
                             Stan_model_name = NULL,
                             ##
                             Stan_data_list = NULL,
                             Stan_model_file_path = NULL,
                             stanc_args = NULL,
                             make_args = NULL,
                             ##
                             sample_nuisance = NULL,
                             n_nuisance_override = NULL,
                             ##
                             model_args_list
) {
  
        n_tests <- NULL
        n_class <- NULL
        N <- NULL
        n_covariates_total <- NULL
        ##
        n_cat_per_ord_test <- NULL
        n_thr_per_ord_test <- NULL
        
        if (Model_type == "Stan") {
                  
               outs_bs_model <- init_bs_model_external(   stream = stream,
                                                          Stan_data_list = Stan_data_list,
                                                          Stan_model_file_path = Stan_model_file_path,
                                                          stanc_args = stanc_args,
                                                          make_args = make_args)
          
                bs_model <- outs_bs_model$bs_model
                
                # Total number of parameters is the UNCONSTRAINED dimension - that is
                # what the sampler integrates over. Counting constrained names is wrong
                # whenever a parameter is constrained (a 4-category simplex has 4
                # constrained names but only 3 unconstrained coordinates).
                n_params <- bs_model$param_unc_num()
                
                nuisance_info <- NULL
                
                # Detect or use provided n_nuisance
                if (!is.null(n_nuisance_override)) {
                  n_nuisance <- n_nuisance_override
                  ## constrained-name count of the nuisance block cannot be inferred
                  ## from an override alone; fall back to the unc count (exact whenever
                  ## the nuisance block has no dimension-changing constraints):
                  n_nuisance_names_constrained <- n_nuisance
                } else {
                  nuisance_info <- detect_nuisance_block_info_from_stan_model( bs_model = bs_model,
                                                                               Stan_model_file_path = Stan_model_file_path,
                                                                               stanc_args = stanc_args)
                  if (!isTRUE(sample_nuisance)) {
                    ## ---- Models WITHOUT a nuisance block: every declared parameter is
                    ## main. (distinct from sample_nuisance = TRUE with a zero-length
                    ## first declaration, which detect_... resolves to 0 by itself)
                    n_nuisance <- 0L
                    n_nuisance_names_constrained <- 0L
                  } else {
                    n_nuisance <- nuisance_info$n_nuisance
                    n_nuisance_names_constrained <- nuisance_info$n_nuisance_names_constrained
                  }
                }
                
                if (n_nuisance < 0 || n_nuisance > n_params) {
                  stop( "get_model_info (Stan): n_nuisance = ", n_nuisance,
                        " is outside [0, n_params = ", n_params, "]. ",
                        "Check n_nuisance_override and the declaration order of the parameters block ",
                        "(the nuisance block must be the FIRST declaration).")
                }
                
                n_params_main <- n_params - n_nuisance
          
        } else {
          
                outs_bs_model <- init_bs_model_internal(   stream = stream,
                                                           Stan_data_list = Stan_data_list,
                                                           Stan_model_name = Stan_model_name)
                # ##
#                 # # make_Stan_data_list_for_internal_models(Model_type = "LC_MVP", model_args_list = list(y = y))
#                 # Stan_data_list <- make_Stan_data_list_for_internal_models( Model_type = "LC_MVOP",
#                 #                                                            model_args_list = model_args_list)
#                 #                                                            # model_args_list = list(y = y))
#                 ##
#                 Stan_data_list <- make_Stan_data_list_for_internal_models( Model_type = "LC_MVP",
#                                                                            model_args_list = model_args_list)
#                 #
#                 Stan_data_list$overflow_threshold  <- +5
#                 Stan_data_list$underflow_threshold <- -5
#                 ##
#                 Stan_data_list$prior_only <- 0
#                 Stan_data_list$Phi_type <- 1
#                 ##
#                 Stan_data_list$pop
#                 Stan_data_list$n_pops
#                 ##
#                 model_args_list$prior_prev_a
#                 model_args_list$prior_prev_b
#                 ##
#                 Stan_data_list$prior_prev_a
#                 Stan_data_list$prior_prev_b
#                 ## 
#                 Stan_data_list$baseline_case_nd <- baseline_case_nd
#                 Stan_data_list$baseline_case_d <- baseline_case_d
#                 ##
#                 # library(cmdstanr)
#                 # mod <- cmdstan_model("/home/enzocerullo/R/R-4.3.3/lib/R/library/BayesMVP/stan_models/LC_MVP_cpp_skeleton.stan")
#                 # mod <- cmdstan_model("/home/enzocerullo/R/R-4.3.3/lib/R/library/BayesMVP/stan_models/LC_MVOP_cpp_skeleton.stan")
#                 # mod <- cmdstan_model("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP/inst/stan_models/LC_MVOP_cpp_skeleton.stan")
#                 mod <- cmdstan_model("/home/enzocerullo/Documents/Work/PhD_work/R_packages/BayesMVP/inst/BayesMVP/inst/stan_models/LC_MVP_bin_cpp_skeleton.stan")
#                 ##
#                 fit <- mod$sample(
#                   data = Stan_data_list,  # your R list before JSON conversion
#                   chains = 1,
#                   iter_warmup = 10,
#                   iter_sampling = 10,
#                   seed = 123
#                 )
#                 # # #
#                 # # Stan_data_list
# #                 ##
#                 ## For hard-coded models:
                ##
                y <- model_args_list$y
                ##
                if (is.null(y)) { 
                  stop("input data y (inside 'model_args_list') is needed if using a built-in/hard-coded model
                 (i.e. for Model_type = 'LC_MVP', 'MVP', 'latent_trait')")
                }
                ##
                n_tests <- ncol(y)
                N <- nrow(y)
                n_class <- ifelse(Model_type %in% c("LC_MVP", "LC_MVOP"), 2, 1)
                n_pops <- model_args_list$n_pops
                n_pops
                ##
                # Calculate n_covariates_total
                # if (is.null(model_args_list$n_covariates_per_outcome_mat)) {
                #   n_covariates_total <- n_tests  # Intercept-only
                # } else {
                  n_covariates_total <- sum(model_args_list$n_covariates_per_outcome_mat)
                # }
                
                # Calculate dimensions based on model type
                if (Model_type %in% c("MVP")) {
                    n_params_main <- n_covariates_total + choose(n_tests, 2)
                    n_nuisance <- n_tests * N
                } else if (Model_type %in% c("LC_MVP")) {
                    n_params_main <- n_pops + n_covariates_total + 2 * choose(n_tests, 2)
                    n_nuisance <- n_tests * N
                } else if (Model_type == "latent_trait") {
                    n_params_main <- n_pops + n_covariates_total + 2 * n_tests
                    n_nuisance <- n_tests * N
                }
                ##
##
                n_cat_per_ord_test <- 0
                n_thr_per_ord_test <- 0
                ##
                n_corrs <- choose(n_tests, 2)
                ##
                if (Model_type %in% c("LC_MVOP", "MVOP")) {
                  
                      ##
                      ## ---- n_cat_per_ord_test MUST be user-supplied (see get_dims_for_internal_models):
                      ##
                      n_cat_per_ord_test <- model_args_list$n_cat_per_ord_test
                      ##
                      if (is.null(n_cat_per_ord_test)) {
                        stop(paste0("get_model_info: model_args_list$n_cat_per_ord_test MUST be supplied for ",
                                    Model_type, ". It cannot be inferred from y (the top category is often ",
                                    "unobserved, so the observed max undercounts)."))
                      }
                      ##
                      n_cat_per_ord_test <- as.integer(n_cat_per_ord_test)
                      n_thr_per_ord_test <- n_cat_per_ord_test - 1L
                      ##
                      if (Model_type == "LC_MVOP") {
                        n_params_main <- n_pops + n_covariates_total + n_class*n_corrs + n_class*sum(n_thr_per_ord_test)
                      } else {  ## MVOP
                        n_params_main <- n_covariates_total + n_corrs + sum(n_thr_per_ord_test)   ## FIXED: was "n_thr_per_ord_t"
                      }
                      ##
                      n_nuisance <- n_tests * N
                  
                }
                ##
                n_params <- n_params_main + n_nuisance
                
        }

        return(list(
          outs_bs_model = outs_bs_model,
          ##
          n_params = n_params,
          n_params_main = n_params_main,
          n_nuisance = n_nuisance,
          ##
          n_nuisance_names_constrained = if (Model_type == "Stan") n_nuisance_names_constrained else n_nuisance,
          nuisance_base_name = if (Model_type == "Stan") (if (is.null(nuisance_info)) NA_character_ else nuisance_info$nuisance_base_name) else NA_character_,
          ##
          N = N,
          n_class = n_class,
          n_tests = n_tests,
          n_covariates_total = n_covariates_total,
          ##
          n_cat_per_ord_test = n_cat_per_ord_test,
          n_thr_per_ord_test = n_thr_per_ord_test
        ))
  
}


#' Check and validate Model_args_as_Rcpp_List
#' @export
validate_rcpp_list <- function(Model_args_as_Rcpp_List, 
                               Model_type) {
        
        required_fields <- c(
          "N", "n_nuisance", "n_params_main",
          "model_so_file", "json_file_path"
        )
        
        if (Model_type %in% c("MVP", "LC_MVP", "latent_trait", "MVOP", "LC_MVOP")) {
          required_fields <- c(required_fields,
                               "Model_args_bools", "Model_args_ints", "Model_args_doubles",
                               "Model_args_strings", "Model_args_mats_double", "Model_args_mats_int"
          )
        }
        
        missing <- setdiff(required_fields, names(Model_args_as_Rcpp_List))
        
        if (length(missing) > 0) {
          stop("Missing required fields in Model_args_as_Rcpp_List: ", 
               paste(missing, collapse = ", "))
        }
        
        # Check types
        if (!is.numeric(Model_args_as_Rcpp_List$N)) {
          stop("N must be numeric")
        }
        
        if (!is.numeric(Model_args_as_Rcpp_List$n_nuisance)) {
          stop("n_nuisance must be numeric")
        }
        
        if (!is.numeric(Model_args_as_Rcpp_List$n_params_main)) {
          stop("n_params_main must be numeric")
        }
        
        return(TRUE)
  
}



#' Debug helper to inspect model structure
#' @export
inspect_model <- function(model_obj) {
        
        cat("Model Type:", model_obj$Model_type, "\n")
        cat("N:", model_obj$init_object$N, "\n")
        cat("n_params:", model_obj$init_object$n_params, "\n")
        cat("n_params_main:", model_obj$init_object$n_params_main, "\n")
        cat("n_nuisance:", model_obj$init_object$n_nuisance, "\n")
        cat("sample_nuisance:", model_obj$init_object$sample_nuisance, "\n")
        
        if (model_obj$Model_type == "Stan") {
          
              cat("\nStan model info:\n")
              cat("  Model file:", model_obj$init_object$Stan_model_file_path, "\n")
              cat("  JSON file:", model_obj$init_object$json_file_path, "\n")
              cat("  SO file:", model_obj$init_object$model_so_file, "\n")
              
              if (!is.null(model_obj$init_object$bs_model)) {
                param_names <- model_obj$init_object$bs_model$param_names()
                cat("  Total parameters:", length(param_names), "\n")
                cat("  First 5 params:", paste(head(param_names, 5), collapse = ", "), "\n")
              }
          
        } else {
              
              cat("\nHard-coded model info:\n")
              cat("  n_tests:", model_obj$init_object$model_args_list$n_tests, "\n")
              cat("  n_class:", model_obj$init_object$model_args_list$n_class, "\n")
              cat("  n_covariates_total:", model_obj$init_object$model_args_list$n_covariates_total, "\n")
          
        }
        
        # Check Rcpp list
        if (!is.null(model_obj$init_object$Model_args_as_Rcpp_List)) {
          cat("\nModel_args_as_Rcpp_List fields:\n")
          cat("  ", paste(names(model_obj$init_object$Model_args_as_Rcpp_List), collapse = ", "), "\n")
        }
  
}




#' compute_equivalent_ESS
#' @export
compute_equivalent_ESS <- function(N, 
                                   N_new, 
                                   ESS_observed, 
                                   scaling_powers = c(0.33, 0.5, 1.0)
) {
  
        # For each scaling assumption, compute equivalent ESS
        # posterior_SD ∝ 1/N^power
        # So SD_new/SD_old = (N/N_new)^power
        
        equivalent_ESS <- numeric(length(scaling_powers))
        
        for (i in seq_along(scaling_powers)) {
          power <- scaling_powers[i]
          
          # Ratio of posterior SDs
          SD_ratio <- (N / N_new)^power
          
          # For equal MC error: ESS_new = ESS_old * SD_ratio^2
          equivalent_ESS[i] <- ESS_observed * SD_ratio^2
        }
        
        # Create nice output
        cat(sprintf("Original: N = %d studies, ESS = %.0f\n", N, ESS_observed))
        cat(sprintf("New:      N = %d studies\n", N_new))
        cat("\nEquivalent ESS for same MC error:\n")
        cat(sprintf("  Conservative (SD ∝ 1/N^0.33): ESS ≈ %.0f\n", equivalent_ESS[1]))
        cat(sprintf("  Typical      (SD ∝ 1/N^0.50): ESS ≈ %.0f\n", equivalent_ESS[2]))
        cat(sprintf("  Optimistic   (SD ∝ 1/N^1.00): ESS ≈ %.0f\n", equivalent_ESS[3]))
        cat(sprintf("\nRange: %.0f - %.0f\n", min(equivalent_ESS), max(equivalent_ESS)))
        
        # Return invisibly for programmatic use
        invisible(list(
          N = N,
          N_new = N_new,
          ESS_observed = ESS_observed,
          equivalent_ESS = equivalent_ESS,
          range = c(min = min(equivalent_ESS), max = max(equivalent_ESS))
        ))
  
}



#' validate_and_extrapolate_ESS
#' @export
validate_and_extrapolate_ESS <- function(N1, 
                                         ESS1,
                                         N2, 
                                         ESS2,
                                         N_target
) {
        # Estimate scaling power
        p <- log(ESS2/ESS1) / (2*log(N1/N2))
        
        # Extrapolate
        ESS_target <- ESS1 * (N1/N_target)^(2*p)
        
        cat(sprintf("Observed scaling: SD ∝ 1/N^%.2f\n", p))
        cat(sprintf("Target ESS for N=%d: %.0f\n", N_target, ESS_target))
        
        # Add safety margin
        cat(sprintf("With 20%% safety margin: %.0f\n", ESS_target * 0.8))
        
        return(list(p = p,
                    ESS_target = ESS_target))
  
}

# If SD ∝ 1/N^p, then from N=500 to N=2500:
# ESS should scale by (500/2500)^(2p) = (1/5)^(2p)
##
# So if you observe ESS_500 = X and ESS_2500 = Y:
# p = log(Y/X) / (2*log(5))
##
# Then for N=10,000:
# ESS_10000 = ESS_500 * (500/10000)^(2p)
##
# For N=25,000:
# ESS_25000 = ESS_500 * (500/25000)^(2p)



#' resize_init_list
#' @export
resize_init_list <- function(init_lists_per_chain, 
                             n_chains_new
) {
        
        n_chains_old <- length(init_lists_per_chain)
        
        if (n_chains_new == n_chains_old) {
          # No change needed
          return(init_lists_per_chain)
        } else if (n_chains_new < n_chains_old) {
          # Fewer chains - just take first n_chains_new
          return(init_lists_per_chain[1:n_chains_new])
        } else {
          # More chains - repeat cyclically
          indices <- rep(1:n_chains_old, length.out = n_chains_new)
          return(init_lists_per_chain[indices])
        }
  
}


#' @export
permute_3d_draws_array <- function(array,
                                   iter_index = 1,
                                   chain_index = 2,
                                   param_index = 3,
                                   new_iter_index = 2,
                                   new_chain_index = 3,
                                   new_param_index = 1
) {
  
        # Create the permutation vector
        # We need to map old positions to new positions
        perm_vector <- numeric(3)
        
        # Find which old index goes to position 1
        if (new_iter_index == 1) perm_vector[1] <- iter_index
        else if (new_chain_index == 1) perm_vector[1] <- chain_index
        else if (new_param_index == 1) perm_vector[1] <- param_index
        
        # Find which old index goes to position 2
        if (new_iter_index == 2) perm_vector[2] <- iter_index
        else if (new_chain_index == 2) perm_vector[2] <- chain_index
        else if (new_param_index == 2) perm_vector[2] <- param_index
        
        # Find which old index goes to position 3
        if (new_iter_index == 3) perm_vector[3] <- iter_index
        else if (new_chain_index == 3) perm_vector[3] <- chain_index
        else if (new_param_index == 3) perm_vector[3] <- param_index
        
        # Apply the permutation
        new_array <- aperm(array, perm = perm_vector)
        
        # Update dimension names if they exist
        if (!is.null(dimnames(array))) {
          old_names <- dimnames(array)
          new_names <- list(
            old_names[[perm_vector[1]]],
            old_names[[perm_vector[2]]],
            old_names[[perm_vector[3]]]
          )
          
          # Rename based on new positions
          names(new_names) <- c(
            if (new_iter_index == 1) "iteration" else if (new_chain_index == 1) "chain" else "parameter",
            if (new_iter_index == 2) "iteration" else if (new_chain_index == 2) "chain" else "parameter",
            if (new_iter_index == 3) "iteration" else if (new_chain_index == 3) "chain" else "parameter"
          )
          
          dimnames(new_array) <- new_names
        }
        
        return(new_array)
  
}












