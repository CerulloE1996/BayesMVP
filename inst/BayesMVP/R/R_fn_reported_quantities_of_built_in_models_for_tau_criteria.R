##
## ==== R_fn_reported_quantities_of_built_in_models_for_tau_criteria.R ===========================================
##
## ---- The reported quantities (Se, Sp, prevalence) of BayesMVP's built-in latent-class models as a function of ----
## ---- the main parameters (for the tau criteria of the burn-in) --------------------------------------------------
##
## BayesMVP's own definition of NicoStan's general hook fn_reported_quantities_function_of_the_model() (NicoStan's
## default, in R_fn_model_backend.R, returns NULL): NicoStan's backend for BayesMVP takes BayesMVP's functions first,
## so the sampler uses this one for BayesMVP's models. Se, Sp and prevalence belong to BayesMVP, never to NicoStan.
##
## fn_reported_quantities_function_of_the_model(init_object, n_params_main) returns, for Model_type "LC_MVP",
## "LC_MVOP" or "CLC_MVOP", a function f(theta_main) (NULL for every other model). theta_main is a matrix of main parameters
## on the sampler's unconstrained scale: n_params_main rows in the sampler's main-block layout, one column per
## chain (a vector is one chain). f returns the reported quantities of every column: a matrix with one named row
## per quantity and one column per chain.
##
## The reported quantities are the generated quantities over which BayesMVP / NicoStan compute the efficiency
## endpoint, i.e. Min_ESS_gq, Min_ESS_tail_gq and Min_ESS_sd_gq of create_summary_and_traces (returned by
## $summary()$get_efficiency_metrics(), and used by fn_gradient_evaluations_to_target_ESS(quantities =
## "generated_quantities", ESS_type = "min_of_bulk_tail_and_sd")): the same set, in the same order, with the same
## names and the same values.
##
## Where the quantities are defined: create_summary_and_traces (NicoStan, R_fn_create_summary_and_traces.R)
## computes the generated quantities of a built-in model with BridgeStan's param_constrain() on the model's Stan
## skeleton, i.e. the generated quantities block (and the transformed parameters it reads) of
##   LC_MVP:   BayesMVP/inst/BayesMVP/inst/stan_models/LC_MVP_bin_cpp_skeleton.stan,
##   LC_MVOP:  BayesMVP/inst/BayesMVP/inst/stan_models/LC_MVOP_cpp_skeleton.stan,
## with the data init_object$Stan_data_list (make_Stan_data_list_for_internal_models in BayesMVP's
## R_fns_init_hard_coded_models.R). This file reproduces those blocks in vectorised R over the chains. Class 1 is
## the non-diseased class (nd), class 2 the diseased class (d):
##   prevalence of population g:     p[g] = (1 + tanh(p_raw[g])) / 2;
##   baseline linear predictor of original test t in class c:
##                                   X_c[t] = sum over k of baseline_case_c[s][k] x beta[c][k, s], over the first
##                                   n_covs_per_outcome[c, s] coefficients of the fitted slot s that holds test t
##                                   (test_perm[s] = t); with the baseline case (1, 0, ..., 0), the intercept;
##   binary test:                    Se = Phi(X_d), Fp = Phi(X_nd), Sp = 1 - Fp;
##   ordinal test, threshold k:      the cutpoints of class c, from the test's block of C_unc_vec[c]:
##                                   C_raw = C_raw_lower + (C_raw_upper - C_raw_lower) (1 + tanh(C_unc)) / 2,
##                                   C[1] = C_raw[1], C[k] = C[k - 1] + exp(C_raw[k]);
##                                   Se[k] = Phi(X_d - C_d[k]), Fp[k] = Phi(X_nd - C_nd[k]), Sp[k] = 1 - Fp[k].
## The set, in the skeletons' output order (Stan names, as in the summaries):
##   LC_MVP:   p[g] (g = 1, ..., n_pops); Se_baseline[t], Sp_baseline[t], Fp_baseline[t] (t = 1, ..., n_tests);
##   LC_MVOP:  p[g]; Se_baseline_bin[b], Sp_baseline_bin[b], Fp_baseline_bin[b] (the b-th binary test);
##             Se_baseline_ord[o,k], Sp_baseline_ord[o,k], Fp_baseline_ord[o,k] (the o-th ordinal test and its
##             threshold k, o varying fastest). The skeleton pads each ordinal test to the largest number of
##             thresholds with the constant -1; the ESS of a constant is undefined, so the efficiency endpoint
##             leaves these entries out, and so does f.
## The tests are numbered in the ORIGINAL test order, as the skeletons return them, whilst the main parameters are
## in the fitted test order of reorder_cols_MVP. Everything f needs is read from init_object once, so init_object
## must be the one the main burn-in runs with (after any test re-ordering, with test_perm in its Stan_data_list).
## Phi is stats::pnorm, which agrees with Stan's Phi to rounding error (absolute differences of about 1e-16).
##
## The sampler's main vector is the skeleton's unconstrained vector: Omega_unconstrained_vec, beta, p_raw and (for
## LC_MVOP) C_unc_vec, in that order, each array stored element by element (class by class) and each matrix column
## by column, so beta[c][k, s] is at position (c - 1) K S + (s - 1) K + k of the beta block (K = n_covariates_max,
## S = n_tests). Note that the names in bs_main_param_names (BridgeStan's param_names(), the constrained names)
## list the first index fastest, so within a family they do not name the sampler's coordinates (with two classes
## and one coefficient, the second coordinate of the beta block is beta[1][1, 2], not "beta.2.1.1"); only the
## family blocks coincide, and these are checked against the names.
##
## (Its directional derivative along a velocity is NicoStan's general fn_directional_derivative_of_reported_quantities.)
##
##
##
##
## ---- fn_reported_quantities_function_of_the_model --------------------------------------------------------------
##
fn_reported_quantities_function_of_the_model <-  function( init_object,
                                                           n_params_main) {

        Model_type <-  init_object$Model_type
        if (!(identical(Model_type, "LC_MVP") || identical(Model_type, "LC_MVOP") ||
              (Model_type %in% c("CLC_MVOP", "CLC-MVOP")))) return(NULL)
        is_ordinal_model <-  identical(Model_type, "LC_MVOP") || (Model_type %in% c("CLC_MVOP", "CLC-MVOP"))
        ##
        fn_stop <-  function(...) {

              stop(paste0("fn_reported_quantities_function_of_the_model (", Model_type, "): ", ...),
                   call. = FALSE)
        }

        if (!is.numeric(n_params_main) || (length(n_params_main) != 1) || !is.finite(n_params_main) ||
            (n_params_main < 1) || (n_params_main != round(n_params_main))) {
              fn_stop("n_params_main must be one positive whole number.")
        }
        if (Model_type %in% c("CLC_MVOP", "CLC-MVOP")) {
              bs_model_clc <-  init_object$bs_model
              if (is.null(bs_model_clc)) {
                    fn_stop("init_object has no bs_model for CLC_MVOP.")
              }
              names_of_main_tp_and_gq <-  init_object$stan_main_and_tp_and_gq_param_names
              names_of_main_and_tp <-  init_object$stan_main_and_tp_param_names
              if (is.null(names_of_main_tp_and_gq) || is.null(names_of_main_and_tp)) {
                    fn_stop("init_object is missing stan_main_and_tp_and_gq_param_names for CLC_MVOP.")
              }
              all_bs_names <-  convert_bridgestan_par_names_to_stan(
                bs_model_clc$param_names(
                  include_tp = TRUE,
                  include_gq = TRUE
                )
              )
              n_main_and_tp <-  length(names_of_main_and_tp)
              gq_positions <-  seq.int(
                from = n_main_and_tp + 1L,
                to = length(all_bs_names)
              )
              gq_names_all <-  all_bs_names[gq_positions]
              gq_family_prefix <-  sub(
                pattern = "\\[.*$",
                replacement = "",
                x = gq_names_all
              )
              reported_families_clc <-  c(
                "prev_overall",
                "prev_A_overall",
                "prev_B_overall",
                "Pr_B_given_A_overall",
                "Pr_A_given_B_overall",
                "log_OR_comorbid_overall",
                "OR_comorbid_overall",
                "Se_baseline_bin",
                "Sp_baseline_bin",
                "Fp_baseline_bin",
                "Se_baseline_ord",
                "Sp_baseline_ord",
                "Fp_baseline_ord",
                "Se_bin",
                "Sp_bin",
                "Se_ord",
                "Sp_ord"
              )
              probe_values <-  bs_model_clc$param_constrain(
                theta_unc = rep(0.0, times = n_params_main),
                include_tp = TRUE,
                include_gq = TRUE,
                rng = bs_model_clc$new_rng(seed = 1L)
              )
              gq_probe_values <-  probe_values[gq_positions]
              keep_gq_mask <-  (gq_family_prefix %in% reported_families_clc) &
                               is.finite(gq_probe_values) &
                               !((gq_family_prefix %in% c(
                                 "Se_baseline_ord", "Sp_baseline_ord", "Fp_baseline_ord",
                                 "Se_bin", "Sp_bin", "Se_ord", "Sp_ord"
                               )) & (gq_probe_values == -1.0))
              kept_constrain_positions <-  gq_positions[keep_gq_mask]
              reported_quantity_names <-  gq_names_all[keep_gq_mask]
              n_reported_quantities <-  length(reported_quantity_names)
              rm(list = c("init_object", "names_of_main_tp_and_gq", "names_of_main_and_tp",
                          "all_bs_names", "gq_positions", "gq_names_all", "gq_family_prefix",
                          "probe_values", "gq_probe_values", "keep_gq_mask"))
              fn_reported_quantities_clc <-  function(theta_main) {

                    if (is.null(dim(theta_main))) theta_main <-  matrix(theta_main, ncol = 1)
                    if (nrow(theta_main) != n_params_main) {
                          stop(paste0("reported quantities (", Model_type, "): theta_main has ", nrow(theta_main),
                                      " rows; the main block has ", n_params_main, "."), call. = FALSE)
                    }
                    n_chains_eval <-  ncol(theta_main)
                    reported_mat <-  matrix(
                      0.0,
                      nrow = n_reported_quantities,
                      ncol = n_chains_eval,
                      dimnames = list(reported_quantity_names, NULL)
                    )
                    for (chain_index in seq_len(n_chains_eval)) {
                          constrained_vec <-  bs_model_clc$param_constrain(
                            theta_unc = as.numeric(theta_main[, chain_index]),
                            include_tp = TRUE,
                            include_gq = TRUE,
                            rng = bs_model_clc$new_rng(seed = 1L)
                          )
                          reported_mat[, chain_index] <-  constrained_vec[kept_constrain_positions]
                    }
                    return(reported_mat)
              }
              attr(fn_reported_quantities_clc, "reported_quantity_names") <-  reported_quantity_names
              return(fn_reported_quantities_clc)
        }
        Stan_data_list <-  init_object$Stan_data_list
        if (!is.list(Stan_data_list)) {
              fn_stop("init_object has no Stan_data_list (the data of the model's Stan skeleton).")
        }
        fn_data_value <-  function(name) {

              value <-  Stan_data_list[[name]]
              if (is.null(value)) fn_stop("the skeleton's data (init_object$Stan_data_list) have no ", name, ".")
              return(value)
        }

        ##
        ## ---- the dimensions, and the test order of the fit (fitted slot s holds original test test_perm[s]):
        ##
        n_tests           <-  as.numeric(fn_data_value("n_tests"))
        n_class           <-  as.numeric(fn_data_value("n_class"))
        n_pops            <-  as.numeric(fn_data_value("n_pops"))
        n_covariates_max  <-  as.numeric(fn_data_value("n_covariates_max"))
        if (n_class != 2) {
              fn_stop("the skeleton computes Se, Sp and Fp for two latent classes only, not n_class = ",
                      n_class, ".")
        }
        test_perm <-  if (is.null(Stan_data_list$test_perm)) {
              seq_len(n_tests)
        } else as.numeric(Stan_data_list$test_perm)
        if ((length(test_perm) != n_tests) || !setequal(test_perm, seq_len(n_tests))) {
              fn_stop("test_perm (", paste(test_perm, collapse = " "), ") is not an order of the ", n_tests,
                      " tests.")
        }
        fitted_slot_of_original_test <-  match(seq_len(n_tests), test_perm)
        ##
        ## ---- the covariates of the baseline case (one column per fitted slot) and the number of coefficients of
        ##      each class and fitted slot:
        ##
        n_covs_per_outcome <-  as.matrix(fn_data_value("n_covs_per_outcome"))
        if (!identical(as.numeric(dim(n_covs_per_outcome)), c(n_class, n_tests))) {
              fn_stop("n_covs_per_outcome is not a ", n_class, " x ", n_tests, " matrix.")
        }
        fn_baseline_case_by_fitted_slot <-  function(name) {

              value <-  fn_data_value(name)
              ## (a list of one vector per test, or a matrix with one row per test)
              baseline_case <-  if (is.list(value)) {
                    do.call(cbind, lapply(value, as.numeric))
              } else t(as.matrix(value))
              if (!identical(as.numeric(dim(baseline_case)), c(n_covariates_max, n_tests))) {
                    fn_stop(name, " does not hold ", n_tests, " vectors of ", n_covariates_max, " covariates.")
              }
              return(baseline_case)
        }

        baseline_case_by_class <-  list( fn_baseline_case_by_fitted_slot("baseline_case_nd"),
                                         fn_baseline_case_by_fitted_slot("baseline_case_d"))
        ##
        ## ---- the main-block layout (the skeleton's parameters block, in the sampler's order):
        ##
        n_corrs_per_class <-  n_tests * (n_tests - 1) / 2
        n_thresholds_of_fitted_ordinal_slot <-  if (is_ordinal_model) {
              as.numeric(fn_data_value("n_thr_per_ord_test"))
        } else numeric(0)
        n_main_rows_of_family <-  c( Omega_unconstrained_vec = n_class * n_corrs_per_class,
                                     beta                    = n_class * n_covariates_max * n_tests,
                                     p_raw                   = n_pops)
        if (is_ordinal_model) {
              n_main_rows_of_family <-  c( n_main_rows_of_family,
                                           C_unc_vec = n_class * sum(n_thresholds_of_fitted_ordinal_slot))
        }
        if (sum(n_main_rows_of_family) != n_params_main) {
              fn_stop("the skeleton's parameters (",
                      paste(names(n_main_rows_of_family), n_main_rows_of_family, sep = ": ", collapse = ", "),
                      ") fill ", sum(n_main_rows_of_family), " main rows, not n_params_main = ",
                      n_params_main, ".")
        }
        n_main_rows_before_family <-  structure(cumsum(c(0, n_main_rows_of_family))[
                                                      seq_along(n_main_rows_of_family)],
                                                names = names(n_main_rows_of_family))
        ##
        ## ---- the family blocks against the names of the main parameters (the family of each name):
        ##
        if (length(init_object$bs_main_param_names) >= n_params_main) {
              family_of_main_row_by_name <-  sub("\\..*$", "", utils::tail(init_object$bs_main_param_names,
                                                                           n_params_main))
              family_of_main_row_by_layout <-  rep(names(n_main_rows_of_family), times = n_main_rows_of_family)
              if (!identical(family_of_main_row_by_name, family_of_main_row_by_layout)) {
                    fn_stop("the families of bs_main_param_names do not follow the layout ",
                            paste(names(n_main_rows_of_family), collapse = ", "), ".")
              }
        }
        ##
        ## ---- the baseline linear predictor of each ORIGINAL test, per class: a sum of terms, each a coefficient
        ##      index k with the main rows of beta[c][k, s] (s = the fitted slot of each original test) and the
        ##      weights baseline_case_c[s][k] (0 beyond n_covs_per_outcome[c, s]); coefficients with no weight are
        ##      left out:
        ##
        fn_main_row_of_beta <-  function(class_index,
                                         coefficient_index,
                                         fitted_slot) {

              return(n_main_rows_before_family[["beta"]] + (class_index - 1) * n_covariates_max * n_tests +
                           (fitted_slot - 1) * n_covariates_max + coefficient_index)
        }

        fn_baseline_linear_predictor_terms <-  function(class_index) {

              terms <-  list()
              for (coefficient_index in seq_len(n_covariates_max)) {
                    weights <-  baseline_case_by_class[[class_index]][coefficient_index,
                                                                      fitted_slot_of_original_test]
                    coefficient_is_used <-  coefficient_index <= n_covs_per_outcome[class_index,
                                                                                    fitted_slot_of_original_test]
                    weights[!coefficient_is_used] <-  0
                    if (all(weights == 0)) next
                    terms[[length(terms) + 1]] <-  list( main_rows = fn_main_row_of_beta(
                                                                           class_index,
                                                                           coefficient_index,
                                                                           fitted_slot_of_original_test),
                                                         weights = weights)
              }
              return(terms)
        }

        baseline_linear_predictor_terms_of_class <-  list( fn_baseline_linear_predictor_terms(1),
                                                           fn_baseline_linear_predictor_terms(2))
        main_rows_of_p_raw <-  n_main_rows_before_family[["p_raw"]] + seq_len(n_pops)
        ##
        ## ---- the quantities that f computes, family by family (names as in the summaries):
        ##
        computed_names_by_family <-  list(p = paste0("p[", seq_len(n_pops), "]"))
        if (!is_ordinal_model) {
              for (family in c("Se_baseline", "Sp_baseline", "Fp_baseline")) {
                    computed_names_by_family[[family]] <-  paste0(family, "[", seq_len(n_tests), "]")
              }
        }
        ##
        ## ---- LC_MVOP: the binary and ordinal tests in the original order, and the cutpoints. Each cutpoint row
        ##      is one (original ordinal test o, threshold k), o slowest; its main row in C_unc_vec[c] is in the
        ##      block of the fitted ordinal slot that holds test o. Both classes are stacked (class 1 rows, then
        ##      class 2 rows), so that the cumulative sums of both run together:
        ##
        n_binary_tests <-  0
        n_ordinal_tests <-  0
        constant_padding_names <-  character(0)
        if (is_ordinal_model) {
              n_cat_per_test_in_fitted_order <-  as.numeric(fn_data_value("n_cat_per_test"))
              if (length(n_cat_per_test_in_fitted_order) != n_tests) {
                    fn_stop("n_cat_per_test does not have ", n_tests, " entries.")
              }
              is_ordinal_fitted_slot <-  n_cat_per_test_in_fitted_order > 2
              if (sum(is_ordinal_fitted_slot) != length(n_thresholds_of_fitted_ordinal_slot)) {
                    fn_stop("n_cat_per_test has ", sum(is_ordinal_fitted_slot), " ordinal tests, ",
                            "n_thr_per_ord_test ", length(n_thresholds_of_fitted_ordinal_slot), ".")
              }
              ordinal_index_of_fitted_slot <-  cumsum(is_ordinal_fitted_slot)
              is_ordinal_original_test <-  is_ordinal_fitted_slot[fitted_slot_of_original_test]
              original_binary_tests <-  which(!is_ordinal_original_test)
              original_ordinal_tests <-  which(is_ordinal_original_test)
              n_binary_tests <-  length(original_binary_tests)
              n_ordinal_tests <-  length(original_ordinal_tests)
              ##
              for (family in c("Se_baseline_bin", "Sp_baseline_bin", "Fp_baseline_bin")) {
                    computed_names_by_family[[family]] <-  if (n_binary_tests > 0) {
                          paste0(family, "[", seq_len(n_binary_tests), "]")
                    } else character(0)
              }
              ##
              if (n_ordinal_tests > 0) {
                    fitted_ordinal_slot_of_original_ordinal_test <-
                          ordinal_index_of_fitted_slot[fitted_slot_of_original_test[original_ordinal_tests]]
                    n_thresholds_of_original_ordinal_test <-
                          n_thresholds_of_fitted_ordinal_slot[fitted_ordinal_slot_of_original_ordinal_test]
                    n_cutpoints_of_class <-  sum(n_thresholds_of_fitted_ordinal_slot)
                    n_main_rows_before_fitted_ordinal_slot_block <-
                          cumsum(c(0, n_thresholds_of_fitted_ordinal_slot))[seq_along(
                                n_thresholds_of_fitted_ordinal_slot)]
                    ##
                    ordinal_test_of_cutpoint_row <-  rep(seq_len(n_ordinal_tests),
                                                         times = n_thresholds_of_original_ordinal_test)
                    threshold_of_cutpoint_row <-  unlist(lapply(n_thresholds_of_original_ordinal_test, seq_len))
                    original_test_of_cutpoint_row <-  original_ordinal_tests[ordinal_test_of_cutpoint_row]
                    main_rows_of_cutpoints_of_class_1 <-  n_main_rows_before_family[["C_unc_vec"]] +
                          n_main_rows_before_fitted_ordinal_slot_block[
                                fitted_ordinal_slot_of_original_ordinal_test[ordinal_test_of_cutpoint_row]] +
                          threshold_of_cutpoint_row
                    main_rows_of_stacked_cutpoints <-  c( main_rows_of_cutpoints_of_class_1,
                                                          main_rows_of_cutpoints_of_class_1 +
                                                                n_cutpoints_of_class)
                    threshold_of_stacked_row <-  rep(threshold_of_cutpoint_row, times = 2)
                    stacked_rows_after_first_threshold <-  which(threshold_of_stacked_row > 1)
                    largest_number_of_thresholds <-  max(n_thresholds_of_fitted_ordinal_slot)
                    stacked_rows_of_threshold <-  lapply(seq_len(largest_number_of_thresholds),
                                                         function(threshold) {

                                                               which(threshold_of_stacked_row == threshold)
                                                         })

                    stacked_rows_of_class_1 <-  seq_len(n_cutpoints_of_class)
                    stacked_rows_of_class_2 <-  n_cutpoints_of_class + seq_len(n_cutpoints_of_class)
                    ##
                    C_raw_lower <-  as.numeric(fn_data_value("C_raw_lower"))
                    C_raw_upper <-  as.numeric(fn_data_value("C_raw_upper"))
                    half_width_of_C_raw_interval <-  (C_raw_upper - C_raw_lower) * 0.5
                    ##
                    for (family in c("Se_baseline_ord", "Sp_baseline_ord", "Fp_baseline_ord")) {
                          computed_names_by_family[[family]] <-  paste0(family, "[", ordinal_test_of_cutpoint_row,
                                                                        ",", threshold_of_cutpoint_row, "]")
                          ## (the skeleton's constant -1 entries beyond each test's number of thresholds)
                          for (ordinal_test in seq_len(n_ordinal_tests)) {
                                padded_thresholds <-  seq_len(largest_number_of_thresholds)[
                                      seq_len(largest_number_of_thresholds) >
                                            n_thresholds_of_original_ordinal_test[ordinal_test]]
                                if (length(padded_thresholds) > 0) {
                                      constant_padding_names <-  c(constant_padding_names,
                                                                   paste0(family, "[", ordinal_test, ",",
                                                                          padded_thresholds, "]"))
                                }
                          }
                    }
              } else {
                    for (family in c("Se_baseline_ord", "Sp_baseline_ord", "Fp_baseline_ord")) {
                          computed_names_by_family[[family]] <-  character(0)
                    }
              }
        }
        ##
        ## ---- the generated quantities of the summaries, in their order (the names after the main parameters and
        ##      transformed parameters), without the constant padding:
        ##
        names_of_main_tp_and_gq <-  init_object$stan_main_and_tp_and_gq_param_names
        names_of_main_and_tp <-  init_object$stan_main_and_tp_param_names
        if (is.null(names_of_main_tp_and_gq) || is.null(names_of_main_and_tp) ||
            !identical(utils::head(names_of_main_tp_and_gq, length(names_of_main_and_tp)),
                       names_of_main_and_tp)) {
              fn_stop("init_object has no stan_main_and_tp_and_gq_param_names starting with ",
                      "stan_main_and_tp_param_names (the names of the summaries).")
        }
        generated_quantity_names <-  names_of_main_tp_and_gq[seq_along(names_of_main_tp_and_gq) >
                                                                  length(names_of_main_and_tp)]
        padding_names_missing <-  setdiff(constant_padding_names, generated_quantity_names)
        if (length(padding_names_missing) > 0) {
              fn_stop("the skeleton has no generated quantities ", paste(padding_names_missing, collapse = ", "),
                      " (expected as constant padding).")
        }
        reported_quantity_names <-  generated_quantity_names[!(generated_quantity_names %in%
                                                                    constant_padding_names)]
        computed_names <-  unlist(computed_names_by_family, use.names = FALSE)
        names_not_computed <-  setdiff(reported_quantity_names, computed_names)
        names_not_generated <-  setdiff(computed_names, reported_quantity_names)
        if ((length(names_not_computed) > 0) || (length(names_not_generated) > 0) ||
            (anyDuplicated(reported_quantity_names) > 0)) {
              fn_stop("the skeleton's generated quantities and the quantities of this function differ; not ",
                      "computed here: ", paste(names_not_computed, collapse = ", "), "; not generated by the ",
                      "skeleton: ", paste(names_not_generated, collapse = ", "), ".")
        }
        n_reported_quantities <-  length(reported_quantity_names)
        output_rows_of_family <-  lapply(computed_names_by_family, function(names_of_family) {

              match(names_of_family, reported_quantity_names)
        })

        ##
        ## (f keeps only what it needs: init_object and the data are not kept in its environment)
        rm(list = c("init_object", "Stan_data_list"))
        ##
        ##
        ## ---- the reported quantities of every column of theta_main:
        ##
        fn_baseline_linear_predictor_of_original_tests <-  function(theta_main,
                                                                    class_index) {

              linear_predictor <-  matrix(0, nrow = n_tests, ncol = ncol(theta_main))
              for (term in baseline_linear_predictor_terms_of_class[[class_index]]) {
                    linear_predictor <-  linear_predictor +
                                         term$weights * theta_main[term$main_rows, , drop = FALSE]
              }
              return(linear_predictor)
        }

        ##
        fn_reported_quantities <-  function(theta_main) {

              if (is.null(dim(theta_main))) theta_main <-  matrix(theta_main, ncol = 1)
              if (nrow(theta_main) != n_params_main) {
                    stop(paste0("reported quantities (", Model_type, "): theta_main has ", nrow(theta_main),
                                " rows; the main block has ", n_params_main, "."), call. = FALSE)
              }
              n_chains <-  ncol(theta_main)
              reported <-  matrix(0, nrow = n_reported_quantities, ncol = n_chains,
                                  dimnames = list(reported_quantity_names, NULL))
              ##
              reported[output_rows_of_family$p, ] <-  0.5 * (1 + tanh(theta_main[main_rows_of_p_raw, ,
                                                                                 drop = FALSE]))
              X_nd <-  fn_baseline_linear_predictor_of_original_tests(theta_main, 1)
              X_d  <-  fn_baseline_linear_predictor_of_original_tests(theta_main, 2)
              ##
              if (!is_ordinal_model) {
                    Fp <-  stats::pnorm(X_nd)
                    reported[output_rows_of_family$Se_baseline, ] <-  stats::pnorm(X_d)
                    reported[output_rows_of_family$Sp_baseline, ] <-  1 - Fp
                    reported[output_rows_of_family$Fp_baseline, ] <-  Fp
                    return(reported)
              }
              ##
              if (n_binary_tests > 0) {
                    Fp <-  stats::pnorm(X_nd[original_binary_tests, , drop = FALSE])
                    reported[output_rows_of_family$Se_baseline_bin, ] <-
                          stats::pnorm(X_d[original_binary_tests, , drop = FALSE])
                    reported[output_rows_of_family$Sp_baseline_bin, ] <-  1 - Fp
                    reported[output_rows_of_family$Fp_baseline_bin, ] <-  Fp
              }
              ##
              if (n_ordinal_tests > 0) {
                    ## (C_raw, then the gaps exp(C_raw[k]) for k > 1, then the cumulative sums within each test)
                    cutpoints <-  C_raw_lower + half_width_of_C_raw_interval *
                                                  (1 + tanh(theta_main[main_rows_of_stacked_cutpoints, ,
                                                                       drop = FALSE]))
                    cutpoints[stacked_rows_after_first_threshold, ] <-
                          exp(cutpoints[stacked_rows_after_first_threshold, , drop = FALSE])
                    for (threshold in seq_len(largest_number_of_thresholds)[-1]) {
                          rows <-  stacked_rows_of_threshold[[threshold]]
                          cutpoints[rows, ] <-  cutpoints[rows - 1, , drop = FALSE] +
                                                cutpoints[rows, , drop = FALSE]
                    }
                    Fp <-  stats::pnorm(X_nd[original_test_of_cutpoint_row, , drop = FALSE] -
                                              cutpoints[stacked_rows_of_class_1, , drop = FALSE])
                    reported[output_rows_of_family$Se_baseline_ord, ] <-
                          stats::pnorm(X_d[original_test_of_cutpoint_row, , drop = FALSE] -
                                             cutpoints[stacked_rows_of_class_2, , drop = FALSE])
                    reported[output_rows_of_family$Sp_baseline_ord, ] <-  1 - Fp
                    reported[output_rows_of_family$Fp_baseline_ord, ] <-  Fp
              }
              return(reported)

        }
        attr(fn_reported_quantities, "reported_quantity_names") <-  reported_quantity_names
        ##
        return(fn_reported_quantities)

}
