#### =============================================================================================================
## R_fns_initial_values_in_fitted_test_order_BayesMVP.R
##
## ---- LC_MVP initial values in the FITTED test order, after reorder_cols_MVP has picked a non-identity test
##      order (fitted slot j holds ORIGINAL test test_perm[j]). Called from the re-layout block of
##      R_fn_sample_model (the shared NicoStan R body), next to the permutation of y, the priors and the beta
##      initial values:
##      * u_raw (the nuisance initial values, N x n_tests, one column per test): columns permuted with the tests;
##      * Omega_unconstrained_vec (per class, the bounded LDL ("Pinkney") raw coordinates of the class's
##        correlation matrix): mapped through the correlation matrix itself, raw (original order) -> Omega
##        (original order) -> Omega[test_perm, test_perm] -> raw (fitted order).
##




#' Cholesky factor of a correlation matrix from its bounded LDL ("Pinkney") raw coordinates
#'
#' Exact R version of Pinkney_LDL_bounds_opt_dbl in src/general_functions/var_fns.hpp (the transform that
#' Pinkney_corr_master_dbl uses), without its log Jacobian. The raw coordinates are read row by row (i > j), as
#' fn_convert_std_vec_of_corrs_to_3d_array_double reads one class block of the main vector. Only the strictly
#' lower triangles of the bound and known-value matrices are read, as in the C++.
#'
#' @param raw_vector numeric vector of length n_tests * (n_tests - 1) / 2 (one class)
#' @param lb_corr,ub_corr n_tests x n_tests lower / upper bounds of the correlations
#' @param known_values_indicator n_tests x n_tests, 1 where the correlation is known (fixed)
#' @param known_values n_tests x n_tests known correlation values
#' @return the n_tests x n_tests lower-triangular Cholesky factor of the correlation matrix
#' @keywords internal
fn_cholesky_factor_of_correlation_matrix_from_bounded_LDL_raw_vector <-  function( raw_vector,
                                                                                   lb_corr,
                                                                                   ub_corr,
                                                                                   known_values_indicator,
                                                                                   known_values
) {

        n_tests <-  nrow(lb_corr)
        stopifnot(length(raw_vector) == n_tests * (n_tests - 1) / 2)
        ##
        fn_bounded_value <-  function( raw_value,
                                       lower,
                                       upper) {

              return(lower + (upper - lower) * 0.5 * (1 + tanh(raw_value)))

        }
        ##
        ## ---- the raw coordinates, row by row (i > j):
        ##
        raw_matrix <-  matrix(0, n_tests, n_tests)
        counter <-  1
        for (i in 2:n_tests) {
              for (j in 1:(i - 1)) {
                    raw_matrix[i, j] <-  raw_vector[counter]
                    counter <-  counter + 1
              }
        }
        ##
        ## ---- first column:
        ##
        L <-  diag(n_tests)
        D <-  numeric(n_tests)
        D[1] <-  1
        for (i in 2:n_tests) {
              if (known_values_indicator[i, 1] == 1) {
                    L[i, 1] <-  known_values[i, 1]
              } else {
                    L[i, 1] <-  fn_bounded_value(raw_matrix[i, 1], lb_corr[i, 1], ub_corr[i, 1])
              }
              D[i] <-  1 - L[i, 1]^2
        }
        ##
        ## ---- the other columns (rows 3 onwards), each bounded by what is left of the row:
        ##
        if (n_tests >= 3) {
              for (i in 3:n_tests) {
                    D[i] <-  1 - L[i, 1]^2
                    L[i, 2:(i - 1)] <-  1 - L[i, 1]^2
                    remainder <-  L[i, 2]
                    ##
                    for (j in 2:(i - 1)) {
                          b1 <-  sum(L[j, 1:(j - 1)] * D[1:(j - 1)] * L[i, 1:(j - 1)])
                          ##
                          if (known_values_indicator[i, j] == 1) {
                                L[i, j] <-  (known_values[i, j] - b1) / D[j]
                          } else {
                                half_width <-  sqrt(remainder * D[j])
                                lower <-  max(-half_width, lb_corr[i, j] - b1)
                                upper <-  min( half_width, ub_corr[i, j] - b1)
                                L[i, j] <-  fn_bounded_value(raw_matrix[i, j], lower, upper) / D[j]
                          }
                          ##
                          remainder <-  remainder - D[j] * L[i, j]^2
                    }
                    ##
                    D[i] <-  remainder
              }
        }
        ##
        return(L %*% diag(sqrt(D), n_tests))

}




#' Bounded LDL ("Pinkney") raw coordinates of a correlation matrix
#'
#' The inverse of fn_cholesky_factor_of_correlation_matrix_from_bounded_LDL_raw_vector: the LDL factors of Omega,
#' then each free coordinate from its bounded value through atanh. A known (fixed) correlation has no raw
#' coordinate in the model (the C++ never reads it): it is returned as 0.
#'
#' @param Omega n_tests x n_tests correlation matrix (positive definite, within the bounds)
#' @inheritParams fn_cholesky_factor_of_correlation_matrix_from_bounded_LDL_raw_vector
#' @return numeric vector of length n_tests * (n_tests - 1) / 2, row by row (i > j)
#' @keywords internal
fn_bounded_LDL_raw_vector_from_correlation_matrix <-  function( Omega,
                                                                lb_corr,
                                                                ub_corr,
                                                                known_values_indicator,
                                                                known_values
) {

        n_tests <-  nrow(Omega)
        ##
        fn_raw_value <-  function( value,
                                   lower,
                                   upper) {

              return(atanh(2 * (value - lower) / (upper - lower) - 1))

        }
        ##
        ## ---- unit lower-triangular L and diagonal D with Omega = L diag(D) t(L):
        ##
        L <-  diag(n_tests)
        D <-  numeric(n_tests)
        for (j in 1:n_tests) {
              ##
              sum_of_earlier_columns <-  0
              if (j > 1) {
                    sum_of_earlier_columns <-  sum(L[j, 1:(j - 1)]^2 * D[1:(j - 1)])
              }
              D[j] <-  Omega[j, j] - sum_of_earlier_columns
              ##
              if (j < n_tests) {
                    for (i in (j + 1):n_tests) {
                          before_j <-  seq_len(j - 1)
                          L[i, j] <-  (Omega[i, j] - sum(L[i, before_j] * L[j, before_j] * D[before_j])) / D[j]
                    }
              }
        }
        ##
        ## ---- the raw coordinates, row by row (i > j):
        ##
        raw_vector <-  numeric(n_tests * (n_tests - 1) / 2)
        counter <-  1
        for (i in 2:n_tests) {
              for (j in 1:(i - 1)) {
                    if (known_values_indicator[i, j] == 1) {
                          raw_vector[counter] <-  0
                    } else if (j == 1) {
                          raw_vector[counter] <-  fn_raw_value(L[i, 1], lb_corr[i, 1], ub_corr[i, 1])
                    } else {
                          b1 <-  sum(L[j, 1:(j - 1)] * D[1:(j - 1)] * L[i, 1:(j - 1)])
                          remainder <-  1 - sum(D[1:(j - 1)] * L[i, 1:(j - 1)]^2)
                          half_width <-  sqrt(remainder * D[j])
                          lower <-  max(-half_width, lb_corr[i, j] - b1)
                          upper <-  min( half_width, ub_corr[i, j] - b1)
                          raw_vector[counter] <-  fn_raw_value(L[i, j] * D[j], lower, upper)
                    }
                    counter <-  counter + 1
              }
        }
        ##
        return(raw_vector)

}




#' u_raw (nuisance) initial values in the fitted test order
#'
#' @param u_raw the u_raw initial values of one chain: an N x n_tests matrix (column t = test t), or a vector of
#'   length N * n_tests read column by column
#' @param test_perm the test order: fitted slot j holds ORIGINAL test test_perm[j]
#' @param N number of observations
#' @return u_raw with its columns in the fitted test order, in the format it was given in
#' @keywords internal
fn_u_raw_initial_values_in_fitted_test_order <-  function( u_raw,
                                                           test_perm,
                                                           N
) {

        n_tests <-  length(test_perm)
        ##
        if (is.matrix(u_raw)) {
              stopifnot(nrow(u_raw) == N, ncol(u_raw) == n_tests)
              return(u_raw[, test_perm, drop = FALSE])
        }
        ##
        ## ---- a vector, read column by column:
        ##
        stopifnot(length(u_raw) == N * n_tests)
        u_raw_as_matrix <-  matrix(u_raw, nrow = N, ncol = n_tests)
        ##
        return(as.vector(u_raw_as_matrix[, test_perm, drop = FALSE]))

}




#' Omega_unconstrained_vec initial values in the fitted test order
#'
#' Each class's raw vector (the bounded LDL coordinates of its correlation matrix in the ORIGINAL test order, with
#' the original-order bounds and known values) is mapped to the raw vector of the SAME correlation matrix in the
#' fitted order, with the bounds and known values the model reads in the fitted order:
#' raw -> Omega -> Omega[test_perm, test_perm] -> raw. The mapped vector is checked by the forward transform
#' (largest difference in Omega at most 1e-8).
#'
#' @param Omega_unconstrained_vec the initial values of one chain: a list with one vector per class, an
#'   n_class x (n_tests * (n_tests - 1) / 2) matrix, or one vector holding the class blocks one after the other
#' @param test_perm the test order: fitted slot j holds ORIGINAL test test_perm[j]
#' @param n_class number of latent classes
#' @param bounds_and_known_values_in_original_test_order list(lb_corr, ub_corr, known_values_indicator_list,
#'   known_values_list), one matrix per class, in the original test order
#' @param bounds_and_known_values_in_fitted_test_order the same, in the fitted test order (as the model reads
#'   them)
#' @param corr_force_positive TRUE: the lower bound of every correlation is 0, whatever lb_corr holds, as the C++
#'   log densities set it (lb_corr[c](i, j) = 0 for i > j when corr_force_positive)
#' @return Omega_unconstrained_vec in the fitted test order, in the format it was given in
#' @keywords internal
fn_Omega_unconstrained_vec_initial_values_in_fitted_test_order <-  function(
                                            Omega_unconstrained_vec,
                                            test_perm,
                                            n_class,
                                            bounds_and_known_values_in_original_test_order,
                                            bounds_and_known_values_in_fitted_test_order,
                                            corr_force_positive
) {

        n_tests <-  length(test_perm)
        n_pairs <-  n_tests * (n_tests - 1) / 2
        ##
        ## ---- one raw vector per class, whatever the format given:
        ##
        raw_per_class <-  list()
        ##
        if (is.list(Omega_unconstrained_vec)) {
              for (class_index in seq_along(Omega_unconstrained_vec)) {
                    raw_per_class[[class_index]] <-  as.numeric(Omega_unconstrained_vec[[class_index]])
              }
        } else if (is.matrix(Omega_unconstrained_vec)) {
              for (class_index in seq_len(nrow(Omega_unconstrained_vec))) {
                    raw_per_class[[class_index]] <-  Omega_unconstrained_vec[class_index, ]
              }
        } else {
              stopifnot(length(Omega_unconstrained_vec) == n_class * n_pairs)
              for (class_index in seq_len(n_class)) {
                    positions_of_class <-  (class_index - 1) * n_pairs + seq_len(n_pairs)
                    raw_per_class[[class_index]] <-  Omega_unconstrained_vec[positions_of_class]
              }
        }
        ##
        stopifnot(length(raw_per_class) == n_class, all(lengths(raw_per_class) == n_pairs))
        ##
        ## ---- the bounds and known values of one class (with corr_force_positive, the lower bound of every
        ##      correlation is 0, as in the C++):
        ##
        fn_bounds_and_known_values_of_class <-  function( bounds_and_known_values,
                                                          class_index) {

              lb_corr_of_class <-  bounds_and_known_values$lb_corr[[class_index]]
              if (isTRUE(corr_force_positive)) {
                    lb_corr_of_class[lower.tri(lb_corr_of_class)] <-  0
              }
              known_values_indicator_list <-  bounds_and_known_values$known_values_indicator_list
              known_values_list <-  bounds_and_known_values$known_values_list
              ##
              return(list( lb_corr = lb_corr_of_class,
                           ub_corr = bounds_and_known_values$ub_corr[[class_index]],
                           known_values_indicator = known_values_indicator_list[[class_index]],
                           known_values = known_values_list[[class_index]]))

        }
        ##
        ## ---- each class: raw (original order) -> Omega -> Omega[test_perm, test_perm] -> raw (fitted order):
        ##
        raw_per_class_in_fitted_order <-  vector("list", n_class)
        ##
        for (class_index in seq_len(n_class)) {

              in_original_order <-  fn_bounds_and_known_values_of_class(
                                        bounds_and_known_values = bounds_and_known_values_in_original_test_order,
                                        class_index = class_index)
              in_fitted_order <-  fn_bounds_and_known_values_of_class(
                                        bounds_and_known_values = bounds_and_known_values_in_fitted_test_order,
                                        class_index = class_index)
              ##
              L_in_original_order <-  fn_cholesky_factor_of_correlation_matrix_from_bounded_LDL_raw_vector(
                                            raw_vector = raw_per_class[[class_index]],
                                            lb_corr = in_original_order$lb_corr,
                                            ub_corr = in_original_order$ub_corr,
                                            known_values_indicator = in_original_order$known_values_indicator,
                                            known_values = in_original_order$known_values)
              ##
              Omega_in_fitted_order <-  (L_in_original_order %*% t(L_in_original_order))[test_perm, test_perm]
              ##
              raw_in_fitted_order <-  fn_bounded_LDL_raw_vector_from_correlation_matrix(
                                            Omega = Omega_in_fitted_order,
                                            lb_corr = in_fitted_order$lb_corr,
                                            ub_corr = in_fitted_order$ub_corr,
                                            known_values_indicator = in_fitted_order$known_values_indicator,
                                            known_values = in_fitted_order$known_values)
              raw_per_class_in_fitted_order[[class_index]] <-  raw_in_fitted_order
              ##
              ## ---- check: the forward transform of the mapped vector gives back the permuted matrix:
              ##
              L_check <-  fn_cholesky_factor_of_correlation_matrix_from_bounded_LDL_raw_vector(
                                raw_vector = raw_in_fitted_order,
                                lb_corr = in_fitted_order$lb_corr,
                                ub_corr = in_fitted_order$ub_corr,
                                known_values_indicator = in_fitted_order$known_values_indicator,
                                known_values = in_fitted_order$known_values)
              ##
              largest_difference <-  max(abs(L_check %*% t(L_check) - Omega_in_fitted_order))
              ##
              if (!is.finite(largest_difference) || largest_difference > 1e-8) {
                    stop(paste0("Omega_unconstrained_vec initial values of class ", class_index, " could not be ",
                                "mapped into the fitted test order (largest difference in Omega ",
                                largest_difference, ")."))
              }

        }
        ##
        ## ---- back in the format given:
        ##
        if (is.list(Omega_unconstrained_vec)) {
              return(raw_per_class_in_fitted_order)
        }
        if (is.matrix(Omega_unconstrained_vec)) {
              return(do.call(rbind, raw_per_class_in_fitted_order))
        }
        return(unlist(raw_per_class_in_fitted_order))

}
























