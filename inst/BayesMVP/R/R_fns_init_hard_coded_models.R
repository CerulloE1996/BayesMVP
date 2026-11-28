
## R_fns_init_hard_coded_models.R


# json <- jsonlite::fromJSON("/home/enzocerullo/R/R-4.3.3/lib/R/library/BayesMVP/stan_data/data_6e479c5a0b94fe3f633a31e26aa5dbcc_1000.json")
# json$n_thr_per_ord_test
# json$n_cat_per_ord_test
# json$n_ordinal_tests
# 
# file.remove("/home/enzocerullo/R/R-4.3.3/lib/R/library/BayesMVP/stan_data/data_6e479c5a0b94fe3f633a31e26aa5dbcc_1000.json")


#### =====================================================================================================================================
#' fn_get_compiled_SIMD_level_of_BayesMVP
#'
#' Returns the HIGHEST SIMD level that the INSTALLED BayesMVP binary was compiled with: "AVX512", "AVX2" or "none".
#' It reuses the existing C++ export detect_vectorization_support() (generated into BayesMVP's own main.cpp from the
#' NicoStan template), which reports the compile-time macros. That C++ function returns "Stan" when neither AVX-512 nor
#' AVX2 was compiled; this helper maps that to "none". This is NOT the CPU's runtime capability
#' (R_fn_detect_vectorisation_support() reports that); only compiled levels can be used by the native kernels.
#' It is the default vect_type of the native models. An AVX-512 build ALSO contains the AVX2 kernels;
#' fn_get_compiled_SIMD_levels_of_BayesMVP() returns every compiled level.
#' @noRd
fn_get_compiled_SIMD_level_of_BayesMVP <-  function() {

        ##
        ## ---- Always ask BayesMVP's own binary (NicoStan exports a function with the same name for ITS binary):
        ##
        compiled_SIMD_level_from_cpp <-  BayesMVP:::detect_vectorization_support()
        ##
        if (identical(compiled_SIMD_level_from_cpp, "AVX512") || identical(compiled_SIMD_level_from_cpp, "AVX2")) {
              return(compiled_SIMD_level_from_cpp)
        }
        ##
        if (identical(compiled_SIMD_level_from_cpp, "Stan")) {
              return("none")
        }
        ##
        stop(paste0("BayesMVP:::detect_vectorization_support() returned an unexpected value ('",
                    paste(compiled_SIMD_level_from_cpp, collapse = ", "),
                    "'); expected 'AVX512', 'AVX2' or 'Stan'."))

}




#### =====================================================================================================================================
#' fn_stop_if_BayesMVP_SIMD_dispatch_info_not_compiled
#'
#' Stops with a clear message if the installed BayesMVP binary predates the SIMD dispatch change (both kernel sets on
#' AVX-512 builds, src/Rcpp_SIMD_dispatch_info.cpp). The R files can be sourced without reinstalling; C++ changes need a rebuild.
#' @noRd
fn_stop_if_BayesMVP_SIMD_dispatch_info_not_compiled <-  function() {

        ##
        BayesMVP_namespace <-  getNamespace("BayesMVP")
        ##
        for (required_cpp_export_name in c("Rcpp_BayesMVP_compiled_SIMD_levels", "Rcpp_BayesMVP_SIMD_lane_width_for_vect_type")) {
              if (!exists(required_cpp_export_name, envir = BayesMVP_namespace, inherits = FALSE)) {
                    stop(paste0("The installed BayesMVP binary has no ", required_cpp_export_name, "() (added with the ",
                                "AVX2 + AVX-512 SIMD dispatch). The R code that was sourced needs the matching C++ build: reinstall ",
                                "NicoStan, then BayesMVP."))
              }
        }
        ##
        return(value_not_printed <- TRUE)

}




#### =====================================================================================================================================
#' fn_probe_BayesMVP_SIMD_lane_width_of_dispatch
#'
#' MEASURES the kernel width that BayesMVP's string dispatch fn_EIGEN_Ref_double (fn_wrappers_overall.hpp, the function every
#' native likelihood calls for exp / log / Phi / ...) really uses for one vect_type string, by calling it through
#' BayesMVP:::Rcpp_wrapper_EIGEN_double_colvec. How: a SIMD processor with lane width W computes a vector shorter than W with its
#' scalar fallback, and the FIRST element of any vector of length >= W with the SIMD kernel (its scalar remainder pass only redoes
#' the last W elements). So the smallest length L at which element 1 differs from its length-1 value is W. The SIMD kernels and the
#' scalar fallbacks differ in the last digits for generic inputs (e.g. log ~1e-9, Phi ~1e-8, exp at the ulp level), so several
#' functions and inputs are probed and the first difference found decides. Returns 1 if element 1 never changes up to length 9.
#' Only meaningful for the SIMD strings "AVX512" / "AVX2": "Stan" runs Stan math on Eigen arrays, which Eigen itself may
#' vectorise in packets (8 doubles on an AVX-512 build; the first test run of this probe measured 8 for "Stan"), and that is not a
#' BayesMVP kernel, so R_fn_BayesMVP_SIMD_lane_width_for_vect_type() does not probe "Stan" / "Loop".
#' @param vect_type one SIMD vect_type string ("AVX512" or "AVX2").
#' @return integer: measured lane width (8, 4, or 1 if no SIMD kernel ran).
#' @noRd
fn_probe_BayesMVP_SIMD_lane_width_of_dispatch <-  function(vect_type) {

        ##
        probe_first_values_by_function_name <-  list( log       = c(0.37, 1.9, 7.3),
                                                      Phi       = c(-1.3, 0.37, 2.2),
                                                      exp       = c(-1.3, 0.37, 2.2),
                                                      tanh      = c(-1.3, 0.37, 2.2),
                                                      inv_logit = c(-1.3, 0.37, 2.2))
        ##
        for (probe_vector_length in 2:9) {
              for (probe_function_name in names(probe_first_values_by_function_name)) {
                    for (probe_first_value in probe_first_values_by_function_name[[probe_function_name]]) {

                          value_from_length_1 <-  BayesMVP:::Rcpp_wrapper_EIGEN_double_colvec( x           = probe_first_value,
                                                                                               fn          = probe_function_name,
                                                                                               vect_type   = vect_type,
                                                                                               skip_checks = FALSE)[1]
                          probe_vector <-  c(probe_first_value, probe_first_value + 0.01 * seq_len(probe_vector_length - 1))
                          value_from_length_L <-  BayesMVP:::Rcpp_wrapper_EIGEN_double_colvec( x           = probe_vector,
                                                                                               fn          = probe_function_name,
                                                                                               vect_type   = vect_type,
                                                                                               skip_checks = FALSE)[1]
                          if (!identical(value_from_length_1, value_from_length_L)) {
                                return(as.integer(probe_vector_length))
                          }

                    }
              }
        }
        ##
        return(1L)

}




#### =====================================================================================================================================
#' R_fn_BayesMVP_SIMD_lane_width_for_vect_type
#'
#' Lane width that the installed BayesMVP binary's native-model SIMD dispatch ACTUALLY uses for each vect_type string:
#' 8 = AVX-512 kernels, 4 = AVX2 kernels, 1 = no BayesMVP SIMD kernel ("Stan", "Loop"). A self-check, not a table:
#'   - it asks the C++ resolver BayesMVP:::Rcpp_BayesMVP_SIMD_lane_width_for_vect_type() (fn_SIMD_level_resolver.hpp, which also
#'     drives vec_from_string, the templated dispatch), and
#'   - for "AVX512" / "AVX2" it MEASURES the width on the real string dispatch, fn_EIGEN_Ref_double
#'     (fn_probe_BayesMVP_SIMD_lane_width_of_dispatch()), and stops if the two disagree.
#' On an AVX-512 build "AVX2" gives 4 (the genuine 256-bit kernels run; previously the 8-lane kernels ran), and on an
#' AVX2-only build "AVX512" stops with the C++ error message (those kernels were not compiled).
#' @param vect_type character vector of vect_type strings.
#' @return integer vector of lane widths, named by vect_type.
#' @export
R_fn_BayesMVP_SIMD_lane_width_for_vect_type <-  function(vect_type) {

        ##
        fn_stop_if_BayesMVP_SIMD_dispatch_info_not_compiled()
        ##
        if (!(is.character(vect_type) && length(vect_type) >= 1)) {
              stop("vect_type must be a non-empty character vector, e.g. 'AVX512', 'AVX2', 'Stan' or 'Loop'.")
        }
        ##
        SIMD_lane_width_by_vect_type <-  unlist(sapply( X         = vect_type,
                                                        FUN       = function(vect_type_value) {

                                                              ##
                                                              ## ---- C++ resolver (stops with the C++ message for a level this build did not compile):
                                                              ##
                                                              lane_width_from_resolver <-  as.integer(BayesMVP:::Rcpp_BayesMVP_SIMD_lane_width_for_vect_type(vect_type_value))
                                                              ##
                                                              ## ---- measured on the real string dispatch (SIMD strings only; see fn_probe_BayesMVP_SIMD_lane_width_of_dispatch):
                                                              ##
                                                              if (!(vect_type_value %in% c("AVX512", "AVX2"))) return(lane_width_from_resolver)
                                                              lane_width_measured <-  fn_probe_BayesMVP_SIMD_lane_width_of_dispatch(vect_type = vect_type_value)
                                                              ##
                                                              if (!identical(lane_width_measured, lane_width_from_resolver)) {
                                                                    stop(paste0("vect_type = '", vect_type_value, "': BayesMVP's string dispatch (fn_EIGEN_Ref_double) ",
                                                                                "ran ", lane_width_measured, "-lane code, but the C++ resolver says ",
                                                                                lane_width_from_resolver, " lanes. The build is inconsistent; reinstall NicoStan, ",
                                                                                "then BayesMVP."))
                                                              }
                                                              ##
                                                              return(lane_width_measured)

                                                        }))
        ##
        return(SIMD_lane_width_by_vect_type)

}




#### =====================================================================================================================================
#' fn_get_compiled_SIMD_levels_of_BayesMVP
#'
#' Returns ALL SIMD kernel sets compiled into the INSTALLED BayesMVP binary, highest first: c("AVX512", "AVX2") on an
#' AVX-512 build (which compiles the AVX2 kernels too, so vect_type = "AVX2" runs them),
#' "AVX2" on an AVX2-only build (e.g. the AVX2-only machine), character(0) if neither. It asks the C++ export
#' Rcpp_BayesMVP_compiled_SIMD_levels() (src/Rcpp_SIMD_dispatch_info.cpp), and stops if its highest level disagrees
#' with fn_get_compiled_SIMD_level_of_BayesMVP() (main.cpp's detect_vectorization_support()), since the two translation
#' units must describe the same build.
#' @noRd
fn_get_compiled_SIMD_levels_of_BayesMVP <-  function() {

        ##
        ## ---- The R files can be sourced without reinstalling BayesMVP; this helper needs the C++ build:
        ##
        fn_stop_if_BayesMVP_SIMD_dispatch_info_not_compiled()
        ##
        compiled_SIMD_levels_from_cpp <-  as.character(BayesMVP:::Rcpp_BayesMVP_compiled_SIMD_levels())
        ##
        if (!all(compiled_SIMD_levels_from_cpp %in% c("AVX512", "AVX2"))) {
              stop(paste0("BayesMVP:::Rcpp_BayesMVP_compiled_SIMD_levels() returned unexpected value(s): ",
                          paste(compiled_SIMD_levels_from_cpp, collapse = ", "), " (expected 'AVX512' and/or 'AVX2')."))
        }
        ##
        best_compiled_SIMD_level_from_main_cpp <-  fn_get_compiled_SIMD_level_of_BayesMVP()
        best_compiled_SIMD_level_from_levels   <-  if (length(compiled_SIMD_levels_from_cpp) > 0) compiled_SIMD_levels_from_cpp[1] else "none"
        ##
        if (!identical(best_compiled_SIMD_level_from_main_cpp, best_compiled_SIMD_level_from_levels)) {
              stop(paste0("BayesMVP build is inconsistent: detect_vectorization_support() (main.cpp) reports '",
                          best_compiled_SIMD_level_from_main_cpp, "' but Rcpp_BayesMVP_compiled_SIMD_levels() reports '",
                          paste(compiled_SIMD_levels_from_cpp, collapse = ", "), "'. Reinstall BayesMVP."))
        }
        ##
        return(compiled_SIMD_levels_from_cpp)

}




#### =====================================================================================================================================
#' fn_check_native_model_vect_types_and_Phi_types
#'
#' Stops (fail loudly) if a native (hard-coded C++) BayesMVP model is asked for a math setting it cannot honour.
#'
#' (1) vect_type and every per-kernel vect_type_* string must be one of the values that the C++ dispatch
#'     (fn_EIGEN_Ref_double in fn_wrappers_overall.hpp, fn_log_sum_exp_2d_double, vec_from_string) actually implements
#'     in THIS build: "Stan", "Loop", or a compiled SIMD level (see fn_get_compiled_SIMD_levels_of_BayesMVP()).
#'     An AVX-512 build compiles BOTH kernel sets, so "AVX512" (8-lane kernels) and "AVX2" (genuine
#'     4-lane kernels) are both accepted there; an AVX2-only build (e.g. the laptop) accepts "AVX2" and STOPS on "AVX512".
#'     (History: previously the build compiled only one level, so "AVX2" on an AVX-512 build was first returned
#'     UNCHANGED (audit item D3) and then, after the D3 fix, stopped here.) Strings the dispatch does not know (e.g. "AVX",
#'     which R_fn_detect_vectorisation_support() returns on AVX-only CPUs) stop too.
#'     Self-check: for every accepted string, BayesMVP:::Rcpp_BayesMVP_SIMD_lane_width_for_vect_type() - the same C++
#'     resolver the dispatchers switch on - must report the expected kernel width (AVX512 = 8, AVX2 = 4, Stan / Loop = 1),
#'     otherwise this stops.
#'
#' (2) Phi_type / inv_Phi_type are no longer restricted here: they are passed through to the native C++ models, which
#'     accept "Phi" / "Phi_approx" / "Phi_approx_2" and "inv_Phi" / "inv_Phi_approx" and stop on anything else.
#'
#' @param Model_type the native model type.
#' @param model_args_list the (user-supplied or resolved) model argument list.
#' @param compiled_SIMD_levels output of fn_get_compiled_SIMD_levels_of_BayesMVP().
#' @param per_kernel_vect_type_names_overwritten_by_vect_type names of per-kernel strings that init_hard_coded_model_args
#'        overwrites with vect_type (a user value that differs would be silently ignored, so it stops).
#' @noRd
fn_check_native_model_vect_types_and_Phi_types <-  function( Model_type,
                                                             model_args_list,
                                                             compiled_SIMD_levels,
                                                             per_kernel_vect_type_names_overwritten_by_vect_type
) {

        ##
        ## ---- Values the C++ dispatch implements in this build (every compiled SIMD level, plus the scalar paths):
        ##
        accepted_vect_type_values <-  c("Stan", "Loop", compiled_SIMD_levels)
        ##
        compiled_SIMD_levels_as_text <-  if (length(compiled_SIMD_levels) > 0) paste0("'", compiled_SIMD_levels, "'", collapse = " and ") else "none"
        ##
        ## ---- Kernel width each accepted string must run (checked against the C++ resolver below):
        ##
        expected_SIMD_lane_width_by_vect_type <-  c( AVX512 = 8,
                                                     AVX2   = 4,
                                                     Stan   = 1,
                                                     Loop   = 1)
        ##
        all_vect_type_names <-  c( "vect_type",
                                   "vect_type_exp",
                                   "vect_type_log",
                                   "vect_type_lse",
                                   "vect_type_tanh",
                                   "vect_type_Phi",
                                   "vect_type_log_Phi",
                                   "vect_type_inv_Phi",
                                   "vect_type_inv_Phi_approx_from_logit_prob")
        ##
        ## ---- (1a) every supplied vect_type* must be a single string the build can honour:
        ##
        effective_SIMD_lane_width_by_vect_type <-  list()
        ##
        for (vect_type_field_name in all_vect_type_names) {

              requested_vect_type_value <-  model_args_list[[vect_type_field_name]]
              ##
              if (is.null(requested_vect_type_value)) next
              ##
              if (!(is.character(requested_vect_type_value) && length(requested_vect_type_value) == 1)) {
                    stop(paste0( vect_type_field_name, " must be a single character string (one of ",
                                 paste0("'", accepted_vect_type_values, "'", collapse = ", "), "), got: ",
                                 paste(format(requested_vect_type_value), collapse = ", ")))
              }
              ##
              if (!(requested_vect_type_value %in% accepted_vect_type_values)) {
                    ##
                    if (requested_vect_type_value %in% c("AVX512", "AVX2")) {
                          build_description <-  if (length(compiled_SIMD_levels) == 0) "a build of BayesMVP without AVX2/AVX-512 kernels"
                                                else paste0("this ", if ("AVX512" %in% compiled_SIMD_levels) "AVX-512" else "AVX2 (no AVX-512)", " build of BayesMVP")
                          stop(paste0( vect_type_field_name, " = '", requested_vect_type_value, "' is not available in ",
                                       build_description, " (Model_type = '", Model_type, "'): its ",
                                       if (identical(requested_vect_type_value, "AVX512")) "AVX-512 (8-lane)" else "AVX2 (4-lane)",
                                       " kernels were not compiled (compiled SIMD levels: ", compiled_SIMD_levels_as_text, "). Use ",
                                       paste0("'", setdiff(accepted_vect_type_values, "Loop"), "'", collapse = " or "),
                                       ", or rebuild BayesMVP on a machine whose compiler flags provide ", requested_vect_type_value,
                                       " (an AVX-512 build contains both the AVX-512 and the AVX2 kernels)."))
                    }
                    ##
                    stop(paste0( vect_type_field_name, " = '", requested_vect_type_value, "' is not a vectorisation type the native ",
                                 "BayesMVP models implement (Model_type = '", Model_type, "'); the C++ wrapper would ignore it. ",
                                 "Accepted values in this build (compiled SIMD levels: ", compiled_SIMD_levels_as_text, "): ",
                                 paste0("'", accepted_vect_type_values, "'", collapse = ", "), "."))
              }
              ##
              ## ---- self-check: the width the dispatch really runs (measured on fn_EIGEN_Ref_double and checked against the C++
              ##      resolver that vec_from_string uses) must be the one this string asks for. Measured once per distinct string:
              ##
              if (is.null(effective_SIMD_lane_width_by_vect_type[[requested_vect_type_value]])) {
                    effective_SIMD_lane_width_by_vect_type[[requested_vect_type_value]] <-  R_fn_BayesMVP_SIMD_lane_width_for_vect_type(vect_type = requested_vect_type_value)
              }
              effective_SIMD_lane_width <-  effective_SIMD_lane_width_by_vect_type[[requested_vect_type_value]]
              expected_SIMD_lane_width  <-  unname(expected_SIMD_lane_width_by_vect_type[requested_vect_type_value])
              ##
              if (!identical(unname(as.numeric(effective_SIMD_lane_width)), as.numeric(expected_SIMD_lane_width))) {
                    stop(paste0( vect_type_field_name, " = '", requested_vect_type_value, "': the C++ SIMD dispatch of this BayesMVP ",
                                 "build would run ", effective_SIMD_lane_width, "-lane kernels, but '", requested_vect_type_value,
                                 "' must run ", expected_SIMD_lane_width, "-lane kernels. The build is inconsistent; reinstall BayesMVP."))
              }

        }
        ##
        ## ---- (1b) per-kernel strings that init_hard_coded_model_args overwrites with vect_type must agree with it:
        ##
        for (vect_type_field_name in per_kernel_vect_type_names_overwritten_by_vect_type) {

              requested_vect_type_value <-  model_args_list[[vect_type_field_name]]
              ##
              if (is.null(requested_vect_type_value) || is.null(model_args_list[["vect_type"]])) next
              ##
              if (!identical(requested_vect_type_value, model_args_list[["vect_type"]])) {
                    stop(paste0( vect_type_field_name, " = '", requested_vect_type_value, "' differs from vect_type = '",
                                 model_args_list[["vect_type"]], "', but init_hard_coded_model_args sets ", vect_type_field_name,
                                 " to vect_type for native models, so it would be silently ignored. Set vect_type instead ",
                                 "(and leave ", vect_type_field_name, " NULL)."))
              }

        }
        ##
        return(value_not_printed <- TRUE)

}




#' init_hard_coded_model_finalise_model_args_list
#'
#' Math-setting rules for the native models (see fn_check_native_model_vect_types_and_Phi_types):
#'   - vect_type defaults to the highest SIMD level COMPILED into the installed BayesMVP ("AVX512"/"AVX2"; "Stan" if
#'     neither), not the CPU's runtime capability; a requested vect_type / vect_type_* that this build cannot honour
#'     stops. An AVX-512 build accepts both "AVX512" and "AVX2"; an AVX2-only build stops on "AVX512".
#'   - Phi_type / inv_Phi_type: "Phi" / "inv_Phi" by default; "Phi_approx" / "inv_Phi_approx" are passed through to the C++.
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
        ## ---- missing test results are coded y = -1 (handled in the C++ by the GHK method: the untruncated
        ##      draw, no likelihood term); every other value must be a category. The type of each test and its
        ##      categories are read from its observed (non-missing) results only:
        ##
        if (any(is.na(y)) || any((y < 0) & (y != -1))) {
          stop("y: a missing test result must be coded -1 (no NA, and no other negative value)")
        }
        # is_binary_slot <- sapply(1:n_tests, 
        #                  function(t) all(y[, t] %in% c(0, 1)))
        is_binary_slot <- sapply(1:n_tests, 
                         function(t) all(y[y[, t] != -1, t] %in% c(0, 1)))
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
                    ## (the observed results only: y = -1 is a missing result)
                    observed_y_of_slot_t <- y[y[, t] != -1, t]
                    # if (max(y[, t]) > n_cat_per_ord_test[k]) {
                    #   stop(paste0("slot ", t, ": observed max category (", max(y[, t]),
                    #               ") exceeds declared n_cat_per_ord_test[", k, "] = ", n_cat_per_ord_test[k]))
                    # }
                    # if (min(y[, t]) < 1) {
                    #   stop(paste0("slot ", t, " is ordinal so y must be 1-indexed; found min = ", min(y[, t])))
                    # }
                    if (max(observed_y_of_slot_t) > n_cat_per_ord_test[k]) {
                      stop(paste0("slot ", t, ": observed max category (", max(observed_y_of_slot_t),
                                  ") exceeds declared n_cat_per_ord_test[", k, "] = ", n_cat_per_ord_test[k]))
                    }
                    if (min(observed_y_of_slot_t) < 1) {
                      stop(paste0("slot ", t, " is ordinal so y must be 1-indexed; found min = ",
                                  min(observed_y_of_slot_t)))
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
        if (!(Model_type %in% c("CLC_MVOP", "CLC-MVOP"))) {
          dims <- get_dims_for_internal_models(  y = y,
                                                 Model_type = Model_type,
                                                 n_tests = n_tests,
                                                 n_class = n_class,
                                                 n_pops = n_pops,
                                                 N = N,
                                                 n_covariates_total = n_covariates_total,
                                                 n_cat_per_ord_test = model_args_list$n_cat_per_ord_test,
                                                 model_args_list = model_args_list)
          n_nuisance    <- dims$n_nuisance
          n_params_main <- dims$n_params_main
          n_params      <- dims$n_params
          n_corrs       <- dims$n_corrs
          ##
          model_args_list$n_nuisance    <- n_nuisance
          model_args_list$n_params_main <- n_params_main
          model_args_list$n_params      <- n_params
          model_args_list$n_corrs       <- n_corrs
        }
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
        n_dirichlet_slices <- n_class
        if (n_ordinal_tests > 0) {
            expected_nrow <- max(n_cat_per_ord_test)
            if (is.null(model_args_list$prior_dirichlet_alpha)) {
                  model_args_list$prior_dirichlet_alpha <- rep(list(matrix(1.0, nrow = expected_nrow, 
                                                                           ncol = n_ordinal_tests)), n_dirichlet_slices)
            } else {
                  if (is.array(model_args_list$prior_dirichlet_alpha) &&
                      length(dim(model_args_list$prior_dirichlet_alpha)) == 3L) {
                        raw_alpha_arr <- model_args_list$prior_dirichlet_alpha
                        model_args_list$prior_dirichlet_alpha <- lapply(
                              seq_len(n_dirichlet_slices),
                              function(slice_index) {
                                    alpha_mat <- matrix(
                                          as.numeric(raw_alpha_arr[slice_index, , ]),
                                          nrow = dim(raw_alpha_arr)[2L],
                                          ncol = dim(raw_alpha_arr)[3L]
                                    )
                                    return(alpha_mat)
                              }
                        )
                  }
                  for (c in 1:n_dirichlet_slices) {
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
            model_args_list$prior_dirichlet_alpha <- rep(list(matrix(1.0, nrow = 1, ncol = 1)), n_dirichlet_slices)
        }
        ##
        if (!(Model_type %in% c("CLC_MVOP", "CLC-MVOP"))) {
            outs <- process_covariates_X( Model_type = Model_type,
                                          X = model_args_list$X,
                                          N = model_args_list$N,
                                          n_tests = model_args_list$n_tests,
                                          n_class = model_args_list$n_class,
                                          n_covariates_per_outcome_mat_given =
                                                model_args_list$n_covariates_per_outcome_mat)
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
        } else {
            raw_covariate_X <- model_args_list$X
            if (is.null(raw_covariate_X)) {
                n_covariates_max <- if (is.null(model_args_list$n_covariates_max)) 1L else max(1L, as.integer(model_args_list$n_covariates_max))
                X_class_test_list <- lapply(
                    seq_len(n_class),
                    function(class_index) {
                        return(lapply(
                            seq_len(n_tests),
                            function(test_index) {
                                mat_cur <- matrix(
                                    0.0,
                                    nrow = N,
                                    ncol = n_covariates_max
                                )
                                mat_cur[, 1L] <- 1.0
                                return(mat_cur)
                            }
                        ))
                    }
                )
            } else if (is.array(raw_covariate_X) && length(dim(raw_covariate_X)) == 4L) {
                dim_X <- dim(raw_covariate_X)
                if (dim_X[1L] != n_class || dim_X[2L] != n_tests || dim_X[3L] != N) {
                    stop("CLC_MVOP: 4D array X must have dimensions [4, n_tests, N, n_covariates_max].")
                }
                n_covariates_max <- max(1L, as.integer(dim_X[4L]))
                X_class_test_list <- lapply(
                    seq_len(n_class),
                    function(class_index) {
                        return(lapply(
                            seq_len(n_tests),
                            function(test_index) {
                                mat_cur <- matrix(
                                    as.numeric(raw_covariate_X[class_index, test_index, , ]),
                                    nrow = N,
                                    ncol = n_covariates_max
                                )
                                return(mat_cur)
                            }
                        ))
                    }
                )
            } else if (is.list(raw_covariate_X) && length(raw_covariate_X) == n_class && is.list(raw_covariate_X[[1L]])) {
                max_cols <- 1L
                for (class_index in seq_len(n_class)) {
                    for (test_index in seq_len(n_tests)) {
                        mat_cur <- as.matrix(raw_covariate_X[[class_index]][[test_index]])
                        if (ncol(mat_cur) > max_cols) {
                            max_cols <- as.integer(ncol(mat_cur))
                        }
                    }
                }
                n_covariates_max <- if (is.null(model_args_list$n_covariates_max)) max_cols else max(max_cols, as.integer(model_args_list$n_covariates_max))
                X_class_test_list <- lapply(
                    seq_len(n_class),
                    function(class_index) {
                        return(lapply(
                            seq_len(n_tests),
                            function(test_index) {
                                mat_src <- as.matrix(raw_covariate_X[[class_index]][[test_index]])
                                storage.mode(mat_src) <- "double"
                                if (ncol(mat_src) == n_covariates_max) {
                                    return(mat_src)
                                }
                                mat_pad <- matrix(
                                    0.0,
                                    nrow = N,
                                    ncol = n_covariates_max
                                )
                                if (ncol(mat_src) > 0L) {
                                    mat_pad[, seq_len(ncol(mat_src))] <- mat_src
                                }
                                return(mat_pad)
                            }
                        ))
                    }
                )
            } else if (is.list(raw_covariate_X) && length(raw_covariate_X) == n_tests && is.matrix(raw_covariate_X[[1L]])) {
                max_cols <- max(
                    1L,
                    unlist(sapply(
                               raw_covariate_X,
                               function(mat_item) as.integer(ncol(as.matrix(mat_item)))
                           ))
                )
                n_covariates_max <- if (is.null(model_args_list$n_covariates_max)) max_cols else max(max_cols, as.integer(model_args_list$n_covariates_max))
                X_per_test <- lapply(
                    seq_len(n_tests),
                    function(test_index) {
                        mat_src <- as.matrix(raw_covariate_X[[test_index]])
                        storage.mode(mat_src) <- "double"
                        if (ncol(mat_src) == n_covariates_max) {
                            return(mat_src)
                        }
                        mat_pad <- matrix(
                            0.0,
                            nrow = N,
                            ncol = n_covariates_max
                        )
                        if (ncol(mat_src) > 0L) {
                            mat_pad[, seq_len(ncol(mat_src))] <- mat_src
                        }
                        return(mat_pad)
                    }
                )
                X_class_test_list <- rep(
                    list(X_per_test),
                    n_class
                )
            } else if (is.matrix(raw_covariate_X)) {
                covariate_matrix <- as.matrix(raw_covariate_X)
                storage.mode(covariate_matrix) <- "double"
                n_covariates_max <- max(1L, as.integer(ncol(covariate_matrix)))
                X_class_test_list <- lapply(
                    seq_len(n_class),
                    function(class_index) {
                        return(lapply(
                            seq_len(n_tests),
                            function(test_index) {
                                return(covariate_matrix)
                            }
                        ))
                    }
                )
            } else {
                stop("CLC_MVOP: X must be a 4D array [4, n_tests, N, K], nested list [[4]][[n_tests]], list of n_tests matrices, or single [N x K] matrix.")
            }
            model_args_list$X <- X_class_test_list
            model_args_list$n_covariates_max <- n_covariates_max
            model_args_list$n_covariates_max_nd <- n_covariates_max
            model_args_list$n_covariates_max_d <- n_covariates_max
            raw_ncov <- if (!is.null(model_args_list$n_covs_per_outcome)) {
                model_args_list$n_covs_per_outcome
            } else {
                model_args_list$n_covariates_per_outcome_mat
            }
            if (is.null(raw_ncov)) {
                n_covs_mat <- matrix(
                    as.integer(n_covariates_max),
                    nrow = n_class,
                    ncol = n_tests
                )
            } else if (is.matrix(raw_ncov) && nrow(raw_ncov) == n_class && ncol(raw_ncov) == n_tests) {
                n_covs_mat <- matrix(
                    as.integer(raw_ncov),
                    nrow = n_class,
                    ncol = n_tests
                )
            } else if (length(raw_ncov) == n_tests) {
                n_covs_mat <- matrix(
                    rep(
                        as.integer(raw_ncov),
                        each = n_class
                    ),
                    nrow = n_class,
                    ncol = n_tests,
                    byrow = FALSE
                )
            } else {
                stop("CLC_MVOP: n_covs_per_outcome must be a [4 x n_tests] matrix or length-n_tests integer vector.")
            }
            model_args_list$n_covs_per_outcome <- n_covs_mat
            model_args_list$n_covariates_per_outcome_mat <- n_covs_mat
            n_covariates_total <- as.integer(sum(n_covs_mat))
            model_args_list$n_covariates_total <- n_covariates_total
        }
        
        ##
        ## ---- Native-model math settings: check the USER-SUPPLIED vect_type / vect_type_* / Phi_type /
        ##      inv_Phi_type BEFORE the defaults and the per-kernel overwrite below, so that nothing requested is
        ##      silently ignored (see fn_check_native_model_vect_types_and_Phi_types):
        ##
        compiled_SIMD_levels_of_BayesMVP <-  fn_get_compiled_SIMD_levels_of_BayesMVP()   ## e.g. c("AVX512", "AVX2") on an AVX-512 build
        compiled_SIMD_level_of_BayesMVP  <-  fn_get_compiled_SIMD_level_of_BayesMVP()    ## the highest one ("none" if neither)
        ##
        per_kernel_vect_type_names_overwritten_by_vect_type <-  c( "vect_type_exp",
                                                                   "vect_type_tanh",
                                                                   "vect_type_log",
                                                                   "vect_type_lse",
                                                                   "vect_type_Phi",
                                                                   "vect_type_inv_Phi")
        ##
        ##
        ## ---- Default vect_type = the SIMD level compiled into this BayesMVP binary (previously the CPU's runtime
        ##      capability via R_fn_detect_vectorisation_support(), which can name a level the binary does not contain,
        ##      e.g. "AVX" on an AVX-only CPU, which the C++ wrapper silently ignored):
        ##
        default_vect_type_for_native_models <-  if (compiled_SIMD_level_of_BayesMVP %in% c("AVX512", "AVX2")) compiled_SIMD_level_of_BayesMVP else "Stan"
        ##
        ## ---- (the per-kernel strings are compared with the vect_type they will be overwritten by, i.e. the default if
        ##      vect_type itself was not supplied):
        ##
        model_args_list_as_requested_with_default_vect_type <-  model_args_list
        model_args_list_as_requested_with_default_vect_type[["vect_type"]] <-  if_null_then_set_to( model_args_list[["vect_type"]],
                                                                                                    default_vect_type_for_native_models,
                                                                                                    debugging = FALSE)
        ##
        fn_check_native_model_vect_types_and_Phi_types( Model_type                                          = Model_type,
                                                        model_args_list                                     = model_args_list_as_requested_with_default_vect_type,
                                                        compiled_SIMD_levels                                = compiled_SIMD_levels_of_BayesMVP,
                                                        per_kernel_vect_type_names_overwritten_by_vect_type = per_kernel_vect_type_names_overwritten_by_vect_type)
        ##
        # Set defaults for all parameters
        defaults <- list( prior_only = FALSE,
                          vect_type = default_vect_type_for_native_models,
                          handle_numerical_issues = TRUE,
                          ##
                          nuisance_transformation = "Phi",
                          Phi_type = "Phi",
                          inv_Phi_type = "inv_Phi",
                          ##
                          overflow_threshold  = +7.5,
                          underflow_threshold = -7.5,
                          ##
                          C_raw_lower = -15,    ## ---- ordinal-only
                          C_raw_upper = +2.5,   ## ---- ordinal-only
                          ##
                          num_chunks = find_num_chunks_MVP(N,
                                                           n_tests),
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
        ## ---- corr_prior_beta and corr_prior_norm are alternatives (a Beta or a normal prior on every
        ##      correlation; both FALSE = the LKJ prior): both TRUE is not a model, so stop:
        ##
        if (isTRUE(as.logical(model_args_list$corr_prior_beta)) &&
            isTRUE(as.logical(model_args_list$corr_prior_norm))) {
          stop(paste0("corr_prior_beta = TRUE and corr_prior_norm = TRUE: set at most one of them to TRUE ",
                      "(corr_prior_beta: Beta prior on (rho + 1) / 2; corr_prior_norm: normal prior on rho; ",
                      "both FALSE: LKJ prior)."))
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
        if (!(Model_type %in% c("CLC_MVOP", "CLC-MVOP")) && is.null(model_args_list$X)) {
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
        if (!(Model_type %in% c("CLC_MVOP", "CLC-MVOP"))) {
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
        }
        
        # Model-specific parameters
        # if (Model_type %in% c("MVP", "LC_MVP")) {
            # LKJ prior
            if (is.null(model_args_list$lkj_cholesky_eta)) {
              if (!is.null(model_args_list$prior_LKJ)) {
                prior_LKJ_numeric <- as.numeric(model_args_list$prior_LKJ)
                if (length(prior_LKJ_numeric) == 1L) {
                  prior_LKJ_numeric <- rep(
                    prior_LKJ_numeric,
                    n_class
                  )
                }
                model_args_list$lkj_cholesky_eta <- matrix(
                  prior_LKJ_numeric,
                  nrow = n_class,
                  ncol = 1L
                )
              } else {
                model_args_list$lkj_cholesky_eta <- matrix(
                  2.0,
                  nrow = n_class,
                  ncol = 1L
                )
              }
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
        ##
        ## ---- Re-check the RESOLVED settings (defaults included):
        ##
        fn_check_native_model_vect_types_and_Phi_types( Model_type                                          = Model_type,
                                                        model_args_list                                     = model_args_list,
                                                        compiled_SIMD_levels                                = compiled_SIMD_levels_of_BayesMVP,
                                                        per_kernel_vect_type_names_overwritten_by_vect_type = per_kernel_vect_type_names_overwritten_by_vect_type)

        if (Model_type %in% c("CLC_MVOP", "CLC-MVOP")) {
                y_integer_matrix <- as.matrix(model_args_list$y)
                storage.mode(y_integer_matrix) <- "integer"
                model_args_list$y <- y_integer_matrix
                ##
                ## ---- Convert any 3D array correlation bounds / known matrices into lists of 4 matrices:
                ##
                for (corr_field_name in c("lb_corr", "ub_corr", "known_values_list", "known_values_indicator_list")) {
                        raw_corr_obj <- model_args_list[[corr_field_name]]
                        if (is.array(raw_corr_obj) && length(dim(raw_corr_obj)) == 3L) {
                                is_ind_field <- identical(corr_field_name, "known_values_indicator_list")
                                model_args_list[[corr_field_name]] <- lapply(
                                        seq_len(n_class),
                                        function(class_index) {
                                                mat_cur <- matrix(
                                                        if (is_ind_field) as.integer(raw_corr_obj[class_index, , ]) else as.numeric(raw_corr_obj[class_index, , ]),
                                                        nrow = n_tests,
                                                        ncol = n_tests
                                                )
                                                return(mat_cur)
                                        }
                                )
                        }
                }
                ##
                ## ---- CLC_MVOP defaults:
                ##
                clc_defaults <- list(
                        test_perm = seq_len(n_tests),
                        is_perfect_ref = rep(0L, n_tests),
                        perfect_ref_M = 8.0,
                        pop_weight = rep(1.0 / as.integer(model_args_list$n_pops), as.integer(model_args_list$n_pops)),
                        n_standardisation = as.integer(N),
                        standardisation_index = seq_len(N),
                        use_label_constraint = 0L,
                        cutoff_category = rep(2L, n_tests),
                        save_subject_probs = 0L,
                        pool_correlations_across_classes = 0L,
                        prior_LKJ_global = 2.0,
                        prior_LKJ_departure = 2.0,
                        prior_lambda_shape_a = 2.0,
                        prior_lambda_shape_b = 2.0,
                        use_pop_prev_prior = 0L,
                        prior_mu_logit_A_mean = -3.0,
                        prior_mu_logit_A_sd = 1.0,
                        prior_mu_logit_B_mean = -4.0,
                        prior_mu_logit_B_sd = 1.0,
                        prior_sigma_logit_sd = 1.0,
                        use_Se_Sp_prior = 0L,
                        Se_Sp_prior_type = 4L,
                        use_pop_Se_Sp_prior = 0L
                )
                for (default_name in names(clc_defaults)) {
                        if (is.null(model_args_list[[default_name]])) {
                                model_args_list[[default_name]] <- clc_defaults[[default_name]]
                        }
                }
                ##
                ## ---- Validate test_perm and interval-censored upper matrix:
                ##
                test_perm_int <- as.integer(model_args_list$test_perm)
                if (length(test_perm_int) != n_tests || !identical(sort(test_perm_int), seq_len(n_tests))) {
                        stop("init_hard_coded_model_args (CLC_MVOP): test_perm must be a permutation of 1:n_tests.")
                }
                model_args_list$test_perm <- test_perm_int
                ##
                if (is.null(model_args_list$y_upper_when_interval_censored)) {
                        y_upper_matrix <- y_integer_matrix
                        has_y_upper_flag <- FALSE
                } else {
                        y_upper_matrix <- as.matrix(model_args_list$y_upper_when_interval_censored)
                        storage.mode(y_upper_matrix) <- "integer"
                        has_y_upper_flag <- any(y_upper_matrix != y_integer_matrix & y_integer_matrix >= 0L)
                }
                model_args_list$y_upper_when_interval_censored <- y_upper_matrix
                model_args_list$has_y_upper <- has_y_upper_flag
                ##
                ## ---- Factorial class_D, test_condition, reference tests, and cutoff_category:
                ##
                if (is.null(model_args_list$class_D)) {
                        class_D_matrix <- matrix(
                                c(0L, 1L, 0L, 1L,
                                  0L, 0L, 1L, 1L),
                                nrow = 4L,
                                ncol = 2L
                        )
                } else {
                        class_D_matrix <- as.matrix(model_args_list$class_D)
                        storage.mode(class_D_matrix) <- "integer"
                }
                model_args_list$class_D <- class_D_matrix
                ##
                if (is.null(model_args_list$test_condition)) {
                        default_condition <- rep(1L, n_tests)
                        if (n_tests >= 2L) {
                                default_condition[seq(2L, n_tests, by = 2L)] <- 2L
                        }
                        model_args_list$test_condition <- default_condition
                } else {
                        model_args_list$test_condition <- as.integer(model_args_list$test_condition)
                }
                model_args_list$is_perfect_ref <- as.integer(model_args_list$is_perfect_ref)
                model_args_list$perfect_ref_M <- as.numeric(model_args_list$perfect_ref_M)
                if (is.null(model_args_list$reference_test)) {
                        cond_A_slots <- which(model_args_list$test_condition == 1L)
                        cond_B_slots <- which(model_args_list$test_condition == 2L)
                        ref_A_orig <- if (length(cond_A_slots) > 0L) test_perm_int[cond_A_slots[1L]] else 1L
                        ref_B_orig <- if (length(cond_B_slots) > 0L) test_perm_int[cond_B_slots[1L]] else min(2L, n_tests)
                        model_args_list$reference_test <- c(
                                as.integer(ref_A_orig),
                                as.integer(ref_B_orig)
                        )
                } else {
                        model_args_list$reference_test <- as.integer(model_args_list$reference_test)
                }
                model_args_list$cutoff_category <- as.integer(model_args_list$cutoff_category)
                model_args_list$use_label_constraint <- as.integer(model_args_list$use_label_constraint)
                ##
                ## ---- Populations and standardisation weights:
                ##
                n_pops_clc <- as.integer(model_args_list$n_pops)
                model_args_list$n_pops <- n_pops_clc
                model_args_list$pop <- as.integer(model_args_list$pop)
                model_args_list$pop_weight <- as.numeric(model_args_list$pop_weight)
                n_standardisation_clc <- as.integer(model_args_list$n_standardisation)
                model_args_list$n_standardisation <- n_standardisation_clc
                model_args_list$standardisation_index <- as.integer(model_args_list$standardisation_index)
                if (is.null(model_args_list$standardisation_weight)) {
                        model_args_list$standardisation_weight <- rep(1.0, n_standardisation_clc)
                } else {
                        model_args_list$standardisation_weight <- as.numeric(model_args_list$standardisation_weight)
                }
                model_args_list$save_subject_probs <- as.integer(model_args_list$save_subject_probs)
                ##
                ## ---- Per-class baseline_case (stored as list of 4 [K_max x n_tests] matrices):
                ##
                if (is.null(model_args_list$baseline_case)) {
                        baseline_case_list <- lapply(
                                seq_len(n_class),
                                function(class_index) {
                                        mat_cur <- matrix(
                                                0.0,
                                                nrow = n_covariates_max,
                                                ncol = n_tests
                                        )
                                        mat_cur[1L, ] <- 1.0
                                        return(mat_cur)
                                }
                        )
                } else if (is.array(model_args_list$baseline_case) && length(dim(model_args_list$baseline_case)) == 3L) {
                        raw_bc <- model_args_list$baseline_case
                        baseline_case_list <- lapply(
                                seq_len(n_class),
                                function(class_index) {
                                        mat_cur <- matrix(
                                                0.0,
                                                nrow = n_covariates_max,
                                                ncol = n_tests
                                        )
                                        for (test_index in seq_len(n_tests)) {
                                                mat_cur[, test_index] <- as.numeric(raw_bc[class_index, test_index, seq_len(n_covariates_max)])
                                        }
                                        return(mat_cur)
                                }
                        )
                } else if (is.list(model_args_list$baseline_case) &&
                           length(model_args_list$baseline_case) == n_class &&
                           is.matrix(model_args_list$baseline_case[[1L]])) {
                        baseline_case_list <- lapply(
                                seq_len(n_class),
                                function(class_index) {
                                        mat_cur <- as.matrix(model_args_list$baseline_case[[class_index]])
                                        storage.mode(mat_cur) <- "double"
                                        return(mat_cur)
                                }
                        )
                } else if (is.list(model_args_list$baseline_case) &&
                           length(model_args_list$baseline_case) == n_class &&
                           is.list(model_args_list$baseline_case[[1L]])) {
                        baseline_case_list <- lapply(
                                seq_len(n_class),
                                function(class_index) {
                                        mat_cur <- matrix(
                                                0.0,
                                                nrow = n_covariates_max,
                                                ncol = n_tests
                                        )
                                        for (test_index in seq_len(n_tests)) {
                                                vec_cur <- as.numeric(model_args_list$baseline_case[[class_index]][[test_index]])
                                                len_copy <- min(
                                                        length(vec_cur),
                                                        n_covariates_max
                                                )
                                                if (len_copy > 0L) {
                                                        mat_cur[seq_len(len_copy), test_index] <- vec_cur[seq_len(len_copy)]
                                                }
                                        }
                                        return(mat_cur)
                                }
                        )
                } else {
                        stop("init_hard_coded_model_args (CLC_MVOP): baseline_case must be a [4, n_tests, K_max] array, list of 4 [K_max x n_tests] matrices, or nested list [[4]][[n_tests]].")
                }
                model_args_list$baseline_case <- baseline_case_list
                ##
                ## ---- Prevalence covariates:
                ##
                if (is.null(model_args_list$X_prevalence)) {
                        n_prev_cov_clc <- if (is.null(model_args_list$n_prevalence_covariates)) {
                                0L
                        } else {
                                as.integer(model_args_list$n_prevalence_covariates)
                        }
                        model_args_list$X_prevalence <- matrix(
                                0.0,
                                nrow = N,
                                ncol = n_prev_cov_clc
                        )
                } else {
                        X_prev_matrix <- as.matrix(model_args_list$X_prevalence)
                        storage.mode(X_prev_matrix) <- "double"
                        if (is.null(model_args_list$n_prevalence_covariates)) {
                                n_prev_cov_clc <- as.integer(ncol(X_prev_matrix))
                        } else {
                                n_prev_cov_clc <- as.integer(model_args_list$n_prevalence_covariates)
                                if (ncol(X_prev_matrix) != n_prev_cov_clc) {
                                        stop("init_hard_coded_model_args (CLC_MVOP): ncol(X_prevalence) does not match n_prevalence_covariates.")
                                }
                        }
                        model_args_list$X_prevalence <- X_prev_matrix
                }
                model_args_list$n_prevalence_covariates <- n_prev_cov_clc
                if (is.null(model_args_list$prior_beta_prevalence_A_mean)) {
                        model_args_list$prior_beta_prevalence_A_mean <- numeric(n_prev_cov_clc)
                } else {
                        model_args_list$prior_beta_prevalence_A_mean <- as.numeric(model_args_list$prior_beta_prevalence_A_mean)
                }
                if (is.null(model_args_list$prior_beta_prevalence_A_sd)) {
                        model_args_list$prior_beta_prevalence_A_sd <- rep(0.5, n_prev_cov_clc)
                } else {
                        model_args_list$prior_beta_prevalence_A_sd <- as.numeric(model_args_list$prior_beta_prevalence_A_sd)
                }
                if (is.null(model_args_list$prior_beta_prevalence_B_mean)) {
                        model_args_list$prior_beta_prevalence_B_mean <- numeric(n_prev_cov_clc)
                } else {
                        model_args_list$prior_beta_prevalence_B_mean <- as.numeric(model_args_list$prior_beta_prevalence_B_mean)
                }
                if (is.null(model_args_list$prior_beta_prevalence_B_sd)) {
                        model_args_list$prior_beta_prevalence_B_sd <- rep(0.5, n_prev_cov_clc)
                } else {
                        model_args_list$prior_beta_prevalence_B_sd <- as.numeric(model_args_list$prior_beta_prevalence_B_sd)
                }
                ##
                ## ---- Per-class beta prior mean/sd (list of 4 [K_max x n_tests] matrices):
                ##
                normalise_clc_beta_prior <- function(raw_prior, default_fill) {
                        if (is.null(raw_prior)) {
                                return(lapply(
                                        seq_len(n_class),
                                        function(class_index) {
                                                return(matrix(
                                                        default_fill,
                                                        nrow = n_covariates_max,
                                                        ncol = n_tests
                                                ))
                                        }
                                ))
                        }
                        if (is.array(raw_prior) && length(dim(raw_prior)) == 3L) {
                                return(lapply(
                                        seq_len(n_class),
                                        function(class_index) {
                                                mat_cur <- matrix(
                                                        as.numeric(raw_prior[class_index, , ]),
                                                        nrow = dim(raw_prior)[2L],
                                                        ncol = dim(raw_prior)[3L]
                                                )
                                                return(mat_cur)
                                        }
                                ))
                        }
                        if (is.list(raw_prior) && length(raw_prior) == n_class) {
                                return(lapply(
                                        seq_len(n_class),
                                        function(class_index) {
                                                mat_cur <- as.matrix(raw_prior[[class_index]])
                                                storage.mode(mat_cur) <- "double"
                                                return(mat_cur)
                                        }
                                ))
                        }
                        stop("init_hard_coded_model_args (CLC_MVOP): prior_beta_mean / prior_beta_sd must be a [4, K_max, n_tests] array or list of 4 [K_max x n_tests] matrices.")
                }
                raw_bm <- if (!is.null(model_args_list$prior_beta_mean)) model_args_list$prior_beta_mean else model_args_list$prior_coeffs_mean_mat
                raw_bs <- if (!is.null(model_args_list$prior_beta_sd)) model_args_list$prior_beta_sd else model_args_list$prior_coeffs_sd_mat
                model_args_list$prior_beta_mean <- normalise_clc_beta_prior(
                        raw_bm,
                        0.0
                )
                model_args_list$prior_beta_sd <- normalise_clc_beta_prior(
                        raw_bs,
                        1.0
                )
                model_args_list$prior_coeffs_mean_mat <- model_args_list$prior_beta_mean
                model_args_list$prior_coeffs_sd_mat <- model_args_list$prior_beta_sd
                ##
                ## ---- Correlation pooling and prevalence switches:
                ##
                pool_corr_clc <- as.integer(model_args_list$pool_correlations_across_classes)
                if (!(pool_corr_clc %in% 0:1)) {
                        stop("init_hard_coded_model_args (CLC_MVOP): pool_correlations_across_classes must be 0 or 1.")
                }
                model_args_list$pool_correlations_across_classes <- pool_corr_clc
                model_args_list$prior_LKJ_global <- as.numeric(model_args_list$prior_LKJ_global)
                model_args_list$prior_LKJ_departure <- as.numeric(model_args_list$prior_LKJ_departure)
                model_args_list$prior_lambda_shape_a <- as.numeric(model_args_list$prior_lambda_shape_a)
                model_args_list$prior_lambda_shape_b <- as.numeric(model_args_list$prior_lambda_shape_b)
                ##
                model_args_list$use_pop_prev_prior <- as.integer(model_args_list$use_pop_prev_prior)
                model_args_list$prior_mu_logit_A_mean <- as.numeric(model_args_list$prior_mu_logit_A_mean)
                model_args_list$prior_mu_logit_A_sd <- as.numeric(model_args_list$prior_mu_logit_A_sd)
                model_args_list$prior_mu_logit_B_mean <- as.numeric(model_args_list$prior_mu_logit_B_mean)
                model_args_list$prior_mu_logit_B_sd <- as.numeric(model_args_list$prior_mu_logit_B_sd)
                prior_sigma_logit_sd_clc <- as.numeric(model_args_list$prior_sigma_logit_sd)
                model_args_list$prior_sigma_logit_sd <- prior_sigma_logit_sd_clc
                if (is.null(model_args_list$prior_sigma_logit_A_sd)) {
                        model_args_list$prior_sigma_logit_A_sd <- prior_sigma_logit_sd_clc
                } else {
                        model_args_list$prior_sigma_logit_A_sd <- as.numeric(model_args_list$prior_sigma_logit_A_sd)
                }
                if (is.null(model_args_list$prior_sigma_logit_B_sd)) {
                        model_args_list$prior_sigma_logit_B_sd <- prior_sigma_logit_sd_clc
                } else {
                        model_args_list$prior_sigma_logit_B_sd <- as.numeric(model_args_list$prior_sigma_logit_B_sd)
                }
                if (is.null(model_args_list$prior_logit_prob_B_given_A_mean)) {
                        if (!is.null(model_args_list$prior_logit_Q_mean)) {
                                model_args_list$prior_logit_prob_B_given_A_mean <- as.numeric(model_args_list$prior_logit_Q_mean)
                        } else {
                                model_args_list$prior_logit_prob_B_given_A_mean <- -1.5
                        }
                } else {
                        model_args_list$prior_logit_prob_B_given_A_mean <- as.numeric(model_args_list$prior_logit_prob_B_given_A_mean)
                }
                if (is.null(model_args_list$prior_logit_prob_B_given_A_sd)) {
                        if (!is.null(model_args_list$prior_logit_Q_sd)) {
                                model_args_list$prior_logit_prob_B_given_A_sd <- as.numeric(model_args_list$prior_logit_Q_sd)
                        } else {
                                model_args_list$prior_logit_prob_B_given_A_sd <- 1.0
                        }
                } else {
                        model_args_list$prior_logit_prob_B_given_A_sd <- as.numeric(model_args_list$prior_logit_prob_B_given_A_sd)
                }
                model_args_list$use_Se_Sp_prior <- as.integer(model_args_list$use_Se_Sp_prior)
                model_args_list$Se_Sp_prior_type <- as.integer(model_args_list$Se_Sp_prior_type)
                model_args_list$use_pop_Se_Sp_prior <- as.integer(model_args_list$use_pop_Se_Sp_prior)
                ##
                ## ---- Normalise 4-class n_chosen_cutoffs (accepting n_chosen alias):
                ##
                n_ord_pad <- max(
                        1L,
                        n_ordinal_tests
                )
                raw_n_chosen <- if (!is.null(model_args_list$n_chosen_cutoffs)) {
                        model_args_list$n_chosen_cutoffs
                } else {
                        model_args_list$n_chosen
                }
                if (is.null(raw_n_chosen)) {
                        n_chosen_mat <- matrix(
                                0L,
                                nrow = n_class,
                                ncol = n_ord_pad
                        )
                } else {
                        n_chosen_mat <- matrix(
                                as.integer(raw_n_chosen),
                                nrow = n_class,
                                ncol = n_ord_pad
                        )
                }
                model_args_list$n_chosen_cutoffs <- n_chosen_mat
                model_args_list$n_chosen <- n_chosen_mat
                n_chosen_max_clc <- if (!is.null(model_args_list$n_chosen_cutoffs_max)) {
                        max(
                                1L,
                                as.integer(model_args_list$n_chosen_cutoffs_max),
                                as.integer(max(n_chosen_mat))
                        )
                } else {
                        max(
                                1L,
                                as.integer(max(n_chosen_mat))
                        )
                }
                model_args_list$n_chosen_cutoffs_max <- n_chosen_max_clc
                ##
                ## ---- Recompute exact CLC_MVOP parameter dimensions after all defaults are resolved:
                ##
                all_correlations_zero <- TRUE
                for (class_index in seq_len(n_class)) {
                        known_indicator_mat <- model_args_list$known_values_indicator_list[[class_index]]
                        known_value_mat <- model_args_list$known_values_list[[class_index]]
                        for (row_index in 2:n_tests) {
                                for (col_index in 1:(row_index - 1L)) {
                                        if (as.integer(known_indicator_mat[row_index, col_index]) == 0L) {
                                                all_correlations_zero <- FALSE
                                        } else if (as.numeric(known_value_mat[row_index, col_index]) != 0.0) {
                                                all_correlations_zero <- FALSE
                                        }
                                }
                        }
                }
                dims_clc <- get_dims_for_internal_models(
                        y = model_args_list$y,
                        Model_type = Model_type,
                        n_tests = n_tests,
                        n_class = n_class,
                        n_pops = n_pops_clc,
                        N = N,
                        n_covariates_total = model_args_list$n_covariates_total,
                        n_cat_per_ord_test = model_args_list$n_cat_per_ord_test,
                        model_args_list = model_args_list
                )
                model_args_list$CI <- isTRUE(all_correlations_zero)
                model_args_list$n_corrs <- dims_clc$n_corrs
                model_args_list$n_params_main <- dims_clc$n_params_main
                model_args_list$n_nuisance <- dims_clc$n_nuisance
                model_args_list$n_params <- dims_clc$n_params
        }
        return(model_args_list)
        
}









#' fn_Se_Sp_prior_Model_args
#' @keywords internal
fn_Se_Sp_prior_Model_args <- function(model_args_list,
                                      n_class,
                                      n_tests,
                                      n_cat_per_test,
                                      n_covariates_max) {

        ##
        ## ---- Priors directly on Sp (c = 1) and Se (c = 2) at chosen cutoffs, for the C++ (0-based cutoffs, -1 =
        ##      unused). Types 1 and 3 on the scale P = Sp (c = 1) and P = 1 - Se (c = 2); type 2 Beta(a, b) on Sp
        ##      and on Se; type 4 on link(Sp) and link(Se). Ordinal tests (second index = ordinal test slot):
        ##      n_chosen [class, ordinal test], chosen_cutoffs [class, ordinal test, j] (1-based within the test),
        ##      prior_Se_Sp_dirichlet_A [class, ordinal test, j],
        ##      prior_Se_Sp_beta [class, ordinal test, j, a / b],
        ##      prior_Se_Sp_link_mean_SD [class, ordinal test, j, mean / SD]
        ##      (unset: every cutoff, mean 0 and SD 1).
        ##
        use_Se_Sp_prior <- if (is.null(model_args_list$use_Se_Sp_prior)) 0L else model_args_list$use_Se_Sp_prior
        dummy_int <- rep(list(matrix(-1L, nrow = 1, ncol = 1)), n_class)
        dummy_double <- rep(list(matrix(0.0, nrow = 1, ncol = 1)), n_class)
        if (use_Se_Sp_prior == 0) {
              return(list(chosen_cutoffs = dummy_int,
                          A = dummy_double,
                          beta_a = dummy_double,
                          beta_b = dummy_double,
                          link_mean = dummy_double,
                          link_SD = dummy_double))
        }
        ##
        ## ---- ordinal tests:
        ##
        ord_slots <- which(n_cat_per_test > 2)
        n_ordinal_tests <- length(ord_slots)
        n_thr_per_ord_test <- n_cat_per_test[ord_slots] - 1L
        n_chosen <- if (is.null(model_args_list$n_chosen)) matrix(n_thr_per_ord_test, nrow = n_class, ncol = n_ordinal_tests, byrow = TRUE) else model_args_list$n_chosen
        n_chosen_max <- max(1L, n_chosen)
        chosen_cutoffs_of_user <- model_args_list$chosen_cutoffs
        if (is.null(chosen_cutoffs_of_user)) {
              chosen_cutoffs_of_user <- array(1L, dim = c(n_class, max(1L, n_ordinal_tests), n_chosen_max))
              for (c in 1:n_class) {
                    for (t_ord in seq_len(n_ordinal_tests)) {
                          n_thr_t <- n_thr_per_ord_test[t_ord]
                          chosen_cutoffs_of_user[c, t_ord, 1:n_thr_t] <- 1:n_thr_t
                    }
              }
        }
        link_mean_SD_of_user <- model_args_list$prior_Se_Sp_link_mean_SD
        if (is.null(link_mean_SD_of_user)) {
              link_mean_SD_of_user <- array(0.0, dim = c(n_class, max(1L, n_ordinal_tests), n_chosen_max, 2))
              link_mean_SD_of_user[, , , 2] <- 1.0
        }
        chosen_cutoffs <- A <- beta_a <- beta_b <- link_mean <- link_SD <- vector("list", n_class)
        for (c in 1:n_class) {

              chosen_cutoffs[[c]] <- matrix(-1L, nrow = n_chosen_max, ncol = max(1L, n_ordinal_tests))
              A[[c]] <- matrix(0.0, nrow = n_chosen_max + 1, ncol = max(1L, n_ordinal_tests))
              beta_a[[c]] <- beta_b[[c]] <- matrix(0.0, nrow = n_chosen_max, ncol = max(1L, n_ordinal_tests))
              link_mean[[c]] <- link_SD[[c]] <- matrix(0.0, nrow = n_chosen_max, ncol = max(1L, n_ordinal_tests))
              for (t_ord in seq_len(n_ordinal_tests)) {

                    m <- n_chosen[c, t_ord]
                    if (m == 0) next
                    chosen <- chosen_cutoffs_of_user[c, t_ord, 1:m]
                    chosen_cutoffs[[c]][1:m, t_ord] <- as.integer(chosen - 1L)
                    if (!is.null(model_args_list$prior_Se_Sp_dirichlet_A)) {
                          A[[c]][1:(m + 1), t_ord] <- model_args_list$prior_Se_Sp_dirichlet_A[c, t_ord, 1:(m + 1)]
                    }
                    if (!is.null(model_args_list$prior_Se_Sp_beta)) {
                          beta_a[[c]][1:m, t_ord] <- model_args_list$prior_Se_Sp_beta[c, t_ord, 1:m, 1]
                          beta_b[[c]][1:m, t_ord] <- model_args_list$prior_Se_Sp_beta[c, t_ord, 1:m, 2]
                    }
                    link_mean[[c]][1:m, t_ord] <- link_mean_SD_of_user[c, t_ord, 1:m, 1]
                    link_SD[[c]][1:m, t_ord] <- link_mean_SD_of_user[c, t_ord, 1:m, 2]

              }

        }
        return(list(chosen_cutoffs = chosen_cutoffs,
                    A = A,
                    beta_a = beta_a,
                    beta_b = beta_b,
                    link_mean = link_mean,
                    link_SD = link_SD))

}



#' build_Model_args_as_Rcpp_Lists
#' @export
build_Model_args_as_Rcpp_List <- function(Model_type,
                                          model_args_list, 
                                          n_nuisance,
                                          n_params_main
) {

        is_CLC_MVOP <- Model_type %in% c("CLC_MVOP", "CLC-MVOP")
  
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
        # print(paste("n_binary_tests = ", n_binary_tests))
        # print(paste("n_ordinal_tests = ", n_ordinal_tests))
        ##
        n_cat_per_ord_test <- model_args_list$n_cat_per_ord_test
        n_thr_per_ord_test <- model_args_list$n_thr_per_ord_test
        ##
        if (is_CLC_MVOP && n_ordinal_tests == 0L) {
          n_cat_per_ord_test <- matrix(2L, ncol = 1L)
          n_thr_per_ord_test <- matrix(1L, ncol = 1L)
        } else {
          n_cat_per_ord_test <- matrix(as.integer(n_cat_per_ord_test), ncol = 1)
          n_thr_per_ord_test <- matrix(as.integer(n_thr_per_ord_test), ncol = 1)
        }
        ##
        ## ---- n_cat_per_test (length n_tests, column order of y, 2 for a binary test) is set from y by
        ##      init_hard_coded_model_args. The fallback above assumed that the binary tests come first in y;
        ##      a missing n_cat_per_test is now built from y in the same way as there (a column is binary iff
        ##      all its values are in {0, 1}; the ordinal columns take n_cat_per_ord_test, in column order):
        ##
        n_cat_per_test_in_column_order_of_y <-  model_args_list$n_cat_per_test
        ##
        if (is.null(n_cat_per_test_in_column_order_of_y)) {
              ## (the observed results only: y = -1 is a missing result)
              is_binary_column_of_y <-  apply(y, 2, function(column_of_y) all(column_of_y[column_of_y != -1] %in%
                                                                                c(0, 1)))
              n_cat_per_test_in_column_order_of_y <-  rep(2, ncol(y))
              n_cat_per_test_in_column_order_of_y[!is_binary_column_of_y] <-  model_args_list$n_cat_per_ord_test
        }
        ##
        n_cat_per_test <-  matrix(as.integer(n_cat_per_test_in_column_order_of_y), ncol = 1)
        ##
        # Model_args_bools - ordered to match C++ indexing
        Model_args_bools <- c(
          if (is.null(model_args_list$exclude_priors)) FALSE else model_args_list$exclude_priors,           # [0]
          if (is.null(model_args_list$CI)) FALSE else model_args_list$CI,                       # [1]
          model_args_list$corr_force_positive,                 # [2]
          if (is.null(model_args_list$corr_prior_beta)) FALSE else model_args_list$corr_prior_beta,          # [3]
          if (is.null(model_args_list$corr_prior_norm)) FALSE else model_args_list$corr_prior_norm,          # [4]
          if (is.null(model_args_list$handle_numerical_issues)) TRUE else model_args_list$handle_numerical_issues,   # [5]
          if (is.null(model_args_list$skip_checks_exp)) FALSE else model_args_list$skip_checks_exp,          # [6]
          if (is.null(model_args_list$skip_checks_log)) FALSE else model_args_list$skip_checks_log,          # [7]
          if (is.null(model_args_list$skip_checks_lse)) FALSE else model_args_list$skip_checks_lse,          # [8]
          if (is.null(model_args_list$skip_checks_tanh)) FALSE else model_args_list$skip_checks_tanh,         # [9]
          if (is.null(model_args_list$skip_checks_Phi)) FALSE else model_args_list$skip_checks_Phi,          # [10]
          if (is.null(model_args_list$skip_checks_log_Phi)) FALSE else model_args_list$skip_checks_log_Phi,      # [11]
          if (is.null(model_args_list$skip_checks_inv_Phi)) FALSE else model_args_list$skip_checks_inv_Phi,      # [12]
          if (is.null(model_args_list$skip_checks_inv_Phi_approx_from_logit_prob)) FALSE else model_args_list$skip_checks_inv_Phi_approx_from_logit_prob, # [13]
          if (is.null(model_args_list$debug)) FALSE else model_args_list$debug                     # [14]
        )
        if (is_CLC_MVOP) {
          ## TRUE only if the covariate matrix of every class and test is the same matrix as that of class 1, test 1
          X_is_the_same_matrix_for_every_class_and_test <- !is.null(model_args_list$X)
          for (class_index in seq_along(model_args_list$X)) {
            for (test_index in seq_along(model_args_list$X[[class_index]])) {
              if (!identical(model_args_list$X[[class_index]][[test_index]], model_args_list$X[[1]][[1]])) {
                X_is_the_same_matrix_for_every_class_and_test <- FALSE
              }
            }
          }
          Model_args_bools <- c(
            Model_args_bools,
            isTRUE(as.logical(model_args_list$prior_only)),
            (as.integer(model_args_list$use_pop_Se_Sp_prior) == 1L),
            isTRUE(as.logical(model_args_list$has_y_upper)),
            X_is_the_same_matrix_for_every_class_and_test
          )
        }
        Model_args_bools <- matrix(Model_args_bools, ncol = 1, nrow = length(Model_args_bools))
        # Model_args_ints - ordered to match C++ indexing
        Model_args_ints <- c(
          if (is.null(model_args_list$n_cores)) 1L else model_args_list$n_cores,                     # [0]
          n_class,                                             # [1]
          if (is.null(model_args_list$ub_threshold_phi_approx)) 10L else model_args_list$ub_threshold_phi_approx,    # [2]
          if (is.null(model_args_list$num_chunks)) 1L else model_args_list$num_chunks,                  # [3]
          model_args_list$n_binary_tests,                      # [4] ## ---- ordinal-only
          model_args_list$n_ordinal_tests,                      # [5] ## ---- ordinal-only
          model_args_list$n_pops, ## ---- multiple-pops
          if (is.null(model_args_list$use_Se_Sp_prior)) 0L else model_args_list$use_Se_Sp_prior,             # [7] priors directly on Se / Sp
          if (is.null(model_args_list$Se_Sp_prior_type)) 4L else model_args_list$Se_Sp_prior_type             # [8] 1 = Dirichlet, 2 / 3 = Betas, 4 = link Normal
        )
        if (is_CLC_MVOP) {
          Model_args_ints <- c(
            Model_args_ints,
            as.integer(model_args_list$n_covariates_max),
            as.integer(model_args_list$n_prevalence_covariates),
            as.integer(model_args_list$pool_correlations_across_classes),
            as.integer(model_args_list$use_pop_prev_prior),
            as.integer(model_args_list$n_standardisation),
            as.integer(model_args_list$use_label_constraint),
            as.integer(model_args_list$n_chosen_cutoffs_max)
          )
        }
        Model_args_ints <- matrix(Model_args_ints, ncol = 1, nrow = length(Model_args_ints))
        if (!is_CLC_MVOP) {
          Se_Sp_args <- fn_Se_Sp_prior_Model_args(model_args_list = model_args_list,
                                                  n_class = n_class,
                                                  n_tests = n_tests,
                                                  n_cat_per_test = if (is.null(model_args_list$n_cat_per_test)) rep(2L, n_tests) else model_args_list$n_cat_per_test,
                                                  n_covariates_max = max(1L, model_args_list$n_covariates_max))
        }
        # Model_args_doubles - ordered to match C++ indexing
        Model_args_doubles <- c(
          model_args_list$overflow_threshold,                  # [0]
          model_args_list$underflow_threshold,                 # [1]
          if (is.null(model_args_list$C_raw_lower)) -7.5 else model_args_list$C_raw_lower,               # [2]  NEW ## ---- ordinal-only
          if (is.null(model_args_list$C_raw_upper)) +2.5 else model_args_list$C_raw_upper                # [3]  NEW ## ---- ordinal-only
        )
        if (is_CLC_MVOP) {
          Model_args_doubles <- c(
            Model_args_doubles,
            as.numeric(model_args_list$perfect_ref_M),
            as.numeric(model_args_list$prior_LKJ_global),
            as.numeric(model_args_list$prior_LKJ_departure),
            as.numeric(model_args_list$prior_lambda_shape_a),
            as.numeric(model_args_list$prior_lambda_shape_b),
            as.numeric(model_args_list$prior_mu_logit_A_mean),
            as.numeric(model_args_list$prior_mu_logit_A_sd),
            as.numeric(model_args_list$prior_mu_logit_B_mean),
            as.numeric(model_args_list$prior_mu_logit_B_sd),
            as.numeric(model_args_list$prior_sigma_logit_A_sd),
            as.numeric(model_args_list$prior_sigma_logit_B_sd),
            as.numeric(model_args_list$prior_logit_prob_B_given_A_mean),
            as.numeric(model_args_list$prior_logit_prob_B_given_A_sd)
          )
        }
        Model_args_doubles <- matrix(Model_args_doubles, ncol = 1, nrow = length(Model_args_doubles))
        
        ##
        ## ---- final guard on the strings the C++ reads. Phi_type / inv_Phi_type are passed through as set in
        ##      model_args_list (exact by default), and every vect_type* must be honoured by this
        ##      build - see fn_check_native_model_vect_types_and_Phi_types. (No per-kernel overwrite happens here, so no
        ##      per-kernel/vect_type agreement is required at this point.)
        ##
        fn_check_native_model_vect_types_and_Phi_types( Model_type                                          = Model_type,
                                                        model_args_list                                     = model_args_list,
                                                        compiled_SIMD_levels                                = fn_get_compiled_SIMD_levels_of_BayesMVP(),
                                                        per_kernel_vect_type_names_overwritten_by_vect_type = character(0))
        ##
        if (is.null(model_args_list[["vect_type"]])) {
              stop("build_Model_args_as_Rcpp_List: model_args_list$vect_type is NULL; run init_hard_coded_model_args() first.")
        }
        ##
        # Model_args_strings - ordered to match C++ indexing
        # [1] Phi_type and [2] inv_Phi_type as set in model_args_list ("Phi" / "inv_Phi" by default; the C++ stops on unknown values).
        Model_args_strings <- c(
          as.character(model_args_list$vect_type),              # [0]
          as.character(if (is.null(model_args_list$Phi_type)) "Phi" else model_args_list$Phi_type),         # [1] Phi_type
          as.character(if (is.null(model_args_list$inv_Phi_type)) "inv_Phi" else model_args_list$inv_Phi_type), # [2] inv_Phi_type
          ##
          as.character(if (is.null(model_args_list$vect_type_exp)) model_args_list$vect_type else model_args_list$vect_type_exp),      # [3]
          as.character(if (is.null(model_args_list$vect_type_log)) model_args_list$vect_type else model_args_list$vect_type_log),      # [4]
          as.character(if (is.null(model_args_list$vect_type_lse)) model_args_list$vect_type else model_args_list$vect_type_lse),      # [5]
          as.character(if (is.null(model_args_list$vect_type_tanh)) model_args_list$vect_type else model_args_list$vect_type_tanh),     # [6]
          ##
          as.character(if (is.null(model_args_list$vect_type_Phi)) model_args_list$vect_type else model_args_list$vect_type_Phi),      # [7]
          as.character(if (is.null(model_args_list$vect_type_log_Phi)) model_args_list$vect_type else model_args_list$vect_type_log_Phi),  # [8]
          as.character(if (is.null(model_args_list$vect_type_inv_Phi)) model_args_list$vect_type else model_args_list$vect_type_inv_Phi),  # [9]
          as.character(if (is.null(model_args_list$vect_type_inv_Phi_approx_from_logit_prob)) model_args_list$vect_type else model_args_list$vect_type_inv_Phi_approx_from_logit_prob), # [10]
          ##
          if (is.null(model_args_list$J_grad_option)) "num_diff" else model_args_list$J_grad_option ,                                          # [11] J_grad_option
          if (is.null(model_args_list$nuisance_transformation)) "Phi" else model_args_list$nuisance_transformation,              # [12]
          if (is.null(model_args_list$LT_prior_corr_bs)) "gamma" else model_args_list$LT_prior_corr_bs # [13]
        )
        
        Model_args_strings <- matrix(Model_args_strings, ncol = 1, nrow = length(Model_args_strings))
        pop_ind_0indexed <- matrix(as.integer(model_args_list$pop - 1L), ncol = 1)  # convert 1-indexed -> 0-indexed
        ##
        if (!is_CLC_MVOP) {
          # Model_args_col_vecs_double - ordered list
          Model_args_col_vecs_double <- list(
            matrix(model_args_list$lkj_cholesky_eta, ncol = 1),  # [0]
            model_args_list$prior_prev_a,                                      # [1] 
            model_args_list$prior_prev_b                                       # [2] 
          )
          ##
          ## Model_args_col_vecs_int:
          ##
          Model_args_col_vecs_int <- list( n_cat_per_ord_test,  ## [0] ## ---- ordinal-only
                                           n_thr_per_ord_test,   ## [1] ## ---- ordinal-only
                                           pop_ind_0indexed,      ## [2] ## ----
                                           n_cat_per_test   ## [3] length n_tests, SLOT order, 2 for binary tests
                                           )
          
          # Model_args_mats_double - ordered list (y goes elsewhere)
          Model_args_mats_double <- list(
            model_args_list$LT_b_priors_1,         ## [0]
            model_args_list$LT_b_priors_2,         ## [1]
            model_args_list$LT_known_bs_indicator, ## [2]
            model_args_list$LT_known_bs_values    ## [3]
          )
          
          # Model_args_mats_int - ordered list
          Model_args_mats_int <- list(
            model_args_list$n_covariates_per_outcome_mat        # [0]
          )
          
          # Model_args_vecs_of_col_vecs_double - empty
          Model_args_vecs_of_col_vecs_double <- list()
          
          # Model_args_vecs_of_col_vecs_int - empty
          Model_args_vecs_of_col_vecs_int <- list()
          
          # Model_args_vecs_of_mats_double - ordered list
          Model_args_vecs_of_mats_double <- list(
            model_args_list$prior_coeffs_mean_mat,              # [0]
            model_args_list$prior_coeffs_sd_mat,                # [1]
            model_args_list$prior_for_corr_a,                   # [2]
            model_args_list$prior_for_corr_b,                   # [3]
            model_args_list$lb_corr,                            # [4]
            model_args_list$ub_corr,                            # [5]
            model_args_list$known_values_list,                   # [6]
            model_args_list$prior_dirichlet_alpha,              # [7] ## ---- ordinal-only
            Se_Sp_args$A,                                       # [8]  Se / Sp prior: Dirichlet A
            Se_Sp_args$beta_a,                                  # [9]  Se / Sp prior: Beta a
            Se_Sp_args$beta_b,                                  # [10] Se / Sp prior: Beta b
            Se_Sp_args$link_mean,                               # [11] Se / Sp prior: link mean (Sp; Se)
            Se_Sp_args$link_SD                                  # [12] Se / Sp prior: link SD
          )
          
          # Model_args_vecs_of_mats_int - ordered list
          Model_args_vecs_of_mats_int <- list(
            model_args_list$known_values_indicator_list,        # [0]
            Se_Sp_args$chosen_cutoffs                           # [1] Se / Sp prior: chosen cutoffs (0-based)
          )
          
          # Model_args_2_layer_vecs_of_col_vecs_double - empty
          Model_args_2_layer_vecs_of_col_vecs_double <- list()
          
          # Model_args_2_layer_vecs_of_col_vecs_int - empty
          Model_args_2_layer_vecs_of_col_vecs_int <- list()
          
          # Model_args_2_layer_vecs_of_mats_double - X matrices
          if (!is.null(model_args_list$X)) {
            Model_args_2_layer_vecs_of_mats_double <- list(
              model_args_list$X                                  
            )
          } else {
            Model_args_2_layer_vecs_of_mats_double <- list()
          }
          # Model_args_2_layer_vecs_of_mats_int - empty
          Model_args_2_layer_vecs_of_mats_int <- list()
        } else {
          n_prev_cov_clc <- as.integer(model_args_list$n_prevalence_covariates)
          std_index_1based <- as.integer(model_args_list$standardisation_index)
          std_weight_raw <- as.numeric(model_args_list$standardisation_weight)
          std_weight_normalised <- std_weight_raw / sum(std_weight_raw)
          pop_std_0based <- as.integer(pop_ind_0indexed[std_index_1based, 1L])
          test_perm_clc <- as.integer(model_args_list$test_perm)
          inv_perm_clc <- integer(n_tests)
          for (slot_index in seq_len(n_tests)) {
            inv_perm_clc[test_perm_clc[slot_index]] <- slot_index
          }
          ref_test_orig <- as.integer(model_args_list$reference_test)
          ref_test_fitted_0based <- as.integer(inv_perm_clc[ref_test_orig] - 1L)
          ##
          if (n_prev_cov_clc > 0L) {
            X_prev_matrix <- as.matrix(model_args_list$X_prevalence)
            storage.mode(X_prev_matrix) <- "double"
          } else {
            X_prev_matrix <- matrix(
              0.0,
              nrow = N,
              ncol = 1L
            )
          }
          X_prev_std_matrix <- X_prev_matrix[std_index_1based, , drop = FALSE]
          ##
          prior_dir_alpha_list <- lapply(
            seq_len(n_class),
            function(class_index) {
              if (n_ordinal_tests > 0L) {
                alpha_matrix <- as.matrix(model_args_list$prior_dirichlet_alpha[[class_index]])
                storage.mode(alpha_matrix) <- "double"
                return(alpha_matrix)
              }
              return(matrix(
                1.0,
                nrow = 1L,
                ncol = 1L
              ))
            }
          )
          ##
          ## ---- 4-class Se/Sp prior arrays:
          ##
          use_Se_Sp_prior_clc <- as.integer(model_args_list$use_Se_Sp_prior)
          n_chosen_mat_clc <- model_args_list$n_chosen_cutoffs
          n_chosen_max_clc <- as.integer(model_args_list$n_chosen_cutoffs_max)
          n_ord_pad <- max(
            1L,
            n_ordinal_tests
          )
          Se_Sp_chosen_list <- lapply(
            seq_len(n_class),
            function(class_index) {
              return(matrix(
                -1L,
                nrow = n_chosen_max_clc,
                ncol = n_ord_pad
              ))
            }
          )
          Se_Sp_A_list <- lapply(
            seq_len(n_class),
            function(class_index) {
              return(matrix(
                0.0,
                nrow = n_chosen_max_clc + 1L,
                ncol = n_ord_pad
              ))
            }
          )
          Se_Sp_beta_a_list <- lapply(
            seq_len(n_class),
            function(class_index) {
              return(matrix(
                0.0,
                nrow = n_chosen_max_clc,
                ncol = n_ord_pad
              ))
            }
          )
          Se_Sp_beta_b_list <- lapply(
            seq_len(n_class),
            function(class_index) {
              return(matrix(
                0.0,
                nrow = n_chosen_max_clc,
                ncol = n_ord_pad
              ))
            }
          )
          Se_Sp_link_mean_list <- lapply(
            seq_len(n_class),
            function(class_index) {
              return(matrix(
                0.0,
                nrow = n_chosen_max_clc,
                ncol = n_ord_pad
              ))
            }
          )
          Se_Sp_link_SD_list <- lapply(
            seq_len(n_class),
            function(class_index) {
              return(matrix(
                1.0,
                nrow = n_chosen_max_clc,
                ncol = n_ord_pad
              ))
            }
          )
          if ((use_Se_Sp_prior_clc == 1L) && is.null(model_args_list$prior_Se_Sp_link_mean_SD)) {
            stop("CLC-MVOP: use_Se_Sp_prior = 1 (the prior on the marginal Se / Sp) needs prior_Se_Sp_link_mean_SD [class, ordinal test, j, mean / SD]; no prior is filled in for the user.")
          }
          if (use_Se_Sp_prior_clc == 1L && n_ordinal_tests > 0L) {
            for (class_index in seq_len(n_class)) {
              for (ord_index in seq_len(n_ordinal_tests)) {
                n_chosen_value <- as.integer(n_chosen_mat_clc[class_index, ord_index])
                if (n_chosen_value == 0L) {
                  next
                }
                Se_Sp_chosen_list[[class_index]][1:n_chosen_value, ord_index] <-
                  as.integer(model_args_list$chosen_cutoffs[class_index, ord_index, 1:n_chosen_value] - 1L)
                if (!is.null(model_args_list$prior_Se_Sp_dirichlet_A)) {
                  raw_dirichlet_A <- model_args_list$prior_Se_Sp_dirichlet_A
                  dirichlet_A_vec <- if (is.list(raw_dirichlet_A)) {
                    raw_dirichlet_A[[class_index]][[ord_index]]
                  } else {
                    raw_dirichlet_A[class_index, ord_index, ]
                  }
                  Se_Sp_A_list[[class_index]][1:(n_chosen_value + 1L), ord_index] <-
                    as.numeric(dirichlet_A_vec[1:(n_chosen_value + 1L)])
                }
                if (!is.null(model_args_list$prior_Se_Sp_beta)) {
                  raw_beta_array <- model_args_list$prior_Se_Sp_beta
                  beta_slice <- if (is.list(raw_beta_array)) {
                    raw_beta_array[[class_index]][[ord_index]]
                  } else {
                    raw_beta_array[class_index, ord_index, , ]
                  }
                  beta_matrix <- matrix(
                    as.numeric(beta_slice),
                    ncol = 2L
                  )
                  Se_Sp_beta_a_list[[class_index]][1:n_chosen_value, ord_index] <- beta_matrix[1:n_chosen_value, 1L]
                  Se_Sp_beta_b_list[[class_index]][1:n_chosen_value, ord_index] <- beta_matrix[1:n_chosen_value, 2L]
                }
                if (!is.null(model_args_list$prior_Se_Sp_link_mean_SD)) {
                  raw_link_array <- model_args_list$prior_Se_Sp_link_mean_SD
                  link_slice <- if (is.list(raw_link_array)) {
                    raw_link_array[[class_index]][[ord_index]]
                  } else {
                    raw_link_array[class_index, ord_index, , ]
                  }
                  link_matrix <- matrix(
                    as.numeric(link_slice),
                    ncol = 2L
                  )
                  Se_Sp_link_mean_list[[class_index]][1:n_chosen_value, ord_index] <- link_matrix[1:n_chosen_value, 1L]
                  Se_Sp_link_SD_list[[class_index]][1:n_chosen_value, ord_index] <- link_matrix[1:n_chosen_value, 2L]
                }
              }
            }
          }
          ##
          ## ---- the prior on the GAP (Se_Sp_prior_type 5, priors on the marginal Se / Sp): for each ordinal test and chosen cutoff j,
          ##      probit of Pr(positive) in the class WITH the other condition minus that in the class WITHOUT it ~ normal(mean, SD);
          ##      positive = the other condition pushes scores up (for Se: higher Se; for Sp: lower Sp, as Pr(positive) = 1 - Sp).
          ##      User data: prior_Se_Sp_gap_link_mean_SD [ordinal test, j, Se / Sp, mean / SD]. Lists [[1]] = Se, [[2]] = Sp, each
          ##      [j, ordinal test]. It is data the user always gives with type 5 (an error otherwise); unused (NA) for every other type.
          ##
          Se_Sp_gap_mean_list <- list(matrix(NA_real_, nrow = n_chosen_max_clc, ncol = n_ord_pad),
                                      matrix(NA_real_, nrow = n_chosen_max_clc, ncol = n_ord_pad))
          Se_Sp_gap_SD_list <- list(matrix(NA_real_, nrow = n_chosen_max_clc, ncol = n_ord_pad),
                                    matrix(NA_real_, nrow = n_chosen_max_clc, ncol = n_ord_pad))
          if (use_Se_Sp_prior_clc == 1L) {
            if (is.null(model_args_list$prior_Se_Sp_gap_link_mean_SD)) {
              stop("CLC-MVOP: use_Se_Sp_prior = 1 (the prior on the marginal Se / Sp) needs prior_Se_Sp_gap_link_mean_SD [ordinal test, j, Se / Sp, mean / SD] (the prior on the probit gap with / without the other condition).")
            }
            raw_gap_array <- model_args_list$prior_Se_Sp_gap_link_mean_SD
            for (ord_index in seq_len(n_ordinal_tests)) {
              for (acc_index in 1:2) {
                Se_Sp_gap_mean_list[[acc_index]][1:n_chosen_max_clc, ord_index] <- as.numeric(raw_gap_array[ord_index, 1:n_chosen_max_clc, acc_index, 1L])
                Se_Sp_gap_SD_list[[acc_index]][1:n_chosen_max_clc, ord_index] <- as.numeric(raw_gap_array[ord_index, 1:n_chosen_max_clc, acc_index, 2L])
              }
            }
          }
          ##
          prior_LKJ_vec <- as.numeric(model_args_list$prior_LKJ)
          if (length(prior_LKJ_vec) == 1L) {
            prior_LKJ_vec <- rep(
              prior_LKJ_vec,
              n_class
            )
          }
          Model_args_col_vecs_double <- list(
            matrix(prior_LKJ_vec, ncol = 1L),
            matrix(if (n_prev_cov_clc > 0L) as.numeric(model_args_list$prior_beta_prevalence_A_mean) else 0.0, ncol = 1L),
            matrix(if (n_prev_cov_clc > 0L) as.numeric(model_args_list$prior_beta_prevalence_A_sd) else 1.0, ncol = 1L),
            matrix(if (n_prev_cov_clc > 0L) as.numeric(model_args_list$prior_beta_prevalence_B_mean) else 0.0, ncol = 1L),
            matrix(if (n_prev_cov_clc > 0L) as.numeric(model_args_list$prior_beta_prevalence_B_sd) else 1.0, ncol = 1L),
            matrix(as.numeric(std_weight_normalised), ncol = 1L)
          )
          Model_args_col_vecs_int <- list(
            n_cat_per_ord_test,
            n_thr_per_ord_test,
            pop_ind_0indexed,
            n_cat_per_test,
            matrix(as.integer(model_args_list$test_condition - 1L), ncol = 1L),
            matrix(as.integer(model_args_list$is_perfect_ref), ncol = 1L),
            matrix(as.integer(std_index_1based - 1L), ncol = 1L),
            matrix(pop_std_0based, ncol = 1L),
            matrix(ref_test_fitted_0based, ncol = 1L),
            matrix(as.integer(model_args_list$cutoff_category), ncol = 1L)
          )
          Model_args_mats_double <- list(
            X_prev_matrix,
            X_prev_std_matrix
          )
          Model_args_mats_int <- list(
            model_args_list$n_covariates_per_outcome_mat,
            model_args_list$class_D,
            model_args_list$y_upper_when_interval_censored
          )
          Model_args_vecs_of_col_vecs_double <- list()
          Model_args_vecs_of_col_vecs_int <- list()
          Model_args_vecs_of_mats_double <- list(
            model_args_list$prior_beta_mean,
            model_args_list$prior_beta_sd,
            model_args_list$lb_corr,
            model_args_list$ub_corr,
            model_args_list$known_values_list,
            prior_dir_alpha_list,
            Se_Sp_A_list,
            Se_Sp_beta_a_list,
            Se_Sp_beta_b_list,
            Se_Sp_link_mean_list,
            Se_Sp_link_SD_list,
            model_args_list$baseline_case,
            Se_Sp_gap_mean_list,
            Se_Sp_gap_SD_list
          )
          Model_args_vecs_of_mats_int <- list(
            model_args_list$known_values_indicator_list,
            Se_Sp_chosen_list
          )
          Model_args_2_layer_vecs_of_col_vecs_double <- list()
          Model_args_2_layer_vecs_of_col_vecs_int <- list()
          Model_args_2_layer_vecs_of_mats_double <- list(
            model_args_list$X
          )
          ## one matrix is handed over if the covariate matrix of every class and test is the same matrix
          if (is_CLC_MVOP && X_is_the_same_matrix_for_every_class_and_test) {
            Model_args_2_layer_vecs_of_mats_double <- list(
              list(
                list(
                  model_args_list$X[[1]][[1]]
                )
              )
            )
          }
          Model_args_2_layer_vecs_of_mats_int <- list()
        }
        ## typos in c++:
        Model_args_2_later_vecs_of_col_vecs_double <- Model_args_2_layer_vecs_of_col_vecs_double
        Model_args_2_later_vecs_of_col_vecs_int <- Model_args_2_layer_vecs_of_col_vecs_int
        Model_args_2_later_vecs_of_mats_double <- Model_args_2_layer_vecs_of_mats_double
        Model_args_2_later_vecs_of_mats_int <- Model_args_2_layer_vecs_of_mats_int
        ##
        colnames(Model_args_bools) <- c("Model arguments - boolean")
        bool_row_names <- c("exclude_priors", "Cond. indep.", "Force +'ve corr's",
                            "corr_prior_beta", "corr_prior_norm",
                            "handle num. issues",
                            "skip_checks_exp", "skip_checks_log", "skip_checks_lse", "skip_checks_tanh",
                            "skip_checks_Phi", "skip_checks_log_Phi", "skip_checks_inv_Phi",
                            "skip_checks_inv_Phi_approx_from_logit_prob",
                            "debug")
        if (is_CLC_MVOP) {
          bool_row_names <- c(bool_row_names, "prior_only", "use_pop_Se_Sp_prior", "has_y_upper",
                              "X_is_the_same_matrix_for_every_class_and_test")
        }
        rownames(Model_args_bools) <- bool_row_names
        ##
        colnames(Model_args_ints) <- c("Model arguments - integers")
        int_row_names <- c("n_cores",
                           "n_class",
                           "ub_threshold_phi_approx",
                           "number of chunks",
                           "n_binary_tests", 
                           "n_ordinal_tests", 
                           "n_pops", ## ----
                           "use_Se_Sp_prior",
                           "Se_Sp_prior_type"
                           )
        if (is_CLC_MVOP) {
          int_row_names <- c(int_row_names,
                             "n_covariates_max",
                             "n_prevalence_covariates",
                             "pool_correlations_across_classes",
                             "use_pop_prev_prior",
                             "n_standardisation",
                             "use_label_constraint",
                             "n_chosen_cutoffs_max")
        }
        rownames(Model_args_ints) <- int_row_names
        ##
        colnames(Model_args_doubles) <- c("Model arguments - doubles")
        double_row_names <- c("overflow_threshold", "underflow_threshold", "C_raw_lower", "C_raw_upper")
        if (is_CLC_MVOP) {
          double_row_names <- c(double_row_names,
                                "perfect_ref_M",
                                "prior_LKJ_global",
                                "prior_LKJ_departure",
                                "prior_lambda_shape_a",
                                "prior_lambda_shape_b",
                                "prior_mu_logit_A_mean",
                                "prior_mu_logit_A_sd",
                                "prior_mu_logit_B_mean",
                                "prior_mu_logit_B_sd",
                                "prior_sigma_logit_A_sd",
                                "prior_sigma_logit_B_sd",
                                "prior_logit_prob_B_given_A_mean",
                                "prior_logit_prob_B_given_A_sd")
        }
        rownames(Model_args_doubles) <- double_row_names
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
          model_so_file = as.character(if (is.null(model_args_list$model_so_file)) "none" else model_args_list$model_so_file),
          json_file_path = as.character(if (is.null(model_args_list$json_file_path)) "none" else model_args_list$json_file_path),
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
                                   "MVOP",
                                   "CLC-MVOP",
                                   "CLC_MVOP")
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
                                               n_cat_per_ord_test = model_args_list$n_cat_per_ord_test,
                                               model_args_list = model_args_list)
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
        if (!(Model_type %in% c("CLC_MVOP", "CLC-MVOP"))) {
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
                if (n_class == 1L) X <-  outs$X
        }
        
        if (Model_type %in% c("MVP", "LC_MVP", "MVOP", "LC_MVOP", "CLC_MVOP", "CLC-MVOP")) {
          
              known_num <- 0
              if (!is.null(model_args_list$known_values_indicator_list)) {
                for (c in 1:n_class) {
                  known_num <- known_num + sum(model_args_list$known_values_indicator_list[[c]])
                }
              }
              known_values_indicator_list <- if_null_then_set_to( model_args_list$known_values_indicator_list,
                                                                  lapply(1:n_class, function(c) matrix(0L, n_tests, n_tests)),
                                                                  debugging = FALSE)
              known_values_list <- if_null_then_set_to( model_args_list$known_values_list,
                                                        lapply(1:n_class, function(c) matrix(0.0, n_tests, n_tests)),
                                                        debugging = FALSE)
              
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
                handle_numerical_issues <- if_null_then_set_to( model_args_list$handle_numerical_issues,
                                                                1,
                                                                debugging = FALSE)
                fully_vectorised <- if_null_then_set_to(model_args_list$fully_vectorised,
                                                        1,
                                                        debugging = FALSE)
                
                Phi_type <- model_args_list$Phi_type
                Phi_type_int <- ifelse(Phi_type == "Phi", 1, 2)
                
                n_pops <- if_null_then_set_to(model_args_list$n_pops,
                                              1,
                                              debugging = FALSE)
                pop <- if_null_then_set_to(model_args_list$pop,
                                           c(rep(1, N)),
                                           debugging = FALSE)
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
                  ## (the baseline cases given by the user, test by test, padded with 0)
                  for (baseline_name in c("baseline_case_nd", "baseline_case_d")) {
                        given <- model_args_list[[baseline_name]]
                        if (!is.null(given) && (length(given) >= t) && length(given[[t]])) {
                              values <- as.numeric(given[[t]])[seq_len(min(length(given[[t]]), n_covariates_max))]
                              if (baseline_name == "baseline_case_nd") baseline_case_nd[[t]][seq_along(values)] <- values
                              else                                      baseline_case_d[[t]][seq_along(values)]  <- values
                        }
                  }
                }
                ##
                Stan_data_list <- list(N = N,
                                       n_tests = n_tests,
                                       y = y,
                                       n_class = n_class,
                                       n_pops =  n_pops,  ## multi-pop not supported yet (currently only in Stan version)
                                       pop =  pop,
                                       ##
                                       test_perm = as.integer(if (is.null(model_args_list$test_perm)) 1:n_tests else model_args_list$test_perm),
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
                handle_numerical_issues <- if_null_then_set_to( model_args_list$handle_numerical_issues,
                                                                1,
                                                                debugging = FALSE)
                fully_vectorised <- if_null_then_set_to(model_args_list$fully_vectorised,
                                                        1,
                                                        debugging = FALSE)
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
                  ## (the baseline cases given by the user, test by test, padded with 0)
                  for (baseline_name in c("baseline_case_nd", "baseline_case_d")) {
                        given <- model_args_list[[baseline_name]]
                        if (!is.null(given) && (length(given) >= t) && length(given[[t]])) {
                              values <- as.numeric(given[[t]])[seq_len(min(length(given[[t]]), n_covariates_max))]
                              if (baseline_name == "baseline_case_nd") baseline_case_nd[[t]][seq_along(values)] <- values
                              else                                      baseline_case_d[[t]][seq_along(values)]  <- values
                        }
                  }
                }
                ##
                baseline_case_nd
                ##
                Stan_data_list <- list(N = N,
                                       n_tests = n_tests,
                                       y = y,
                                       ##
                                       test_perm = as.integer(if (is.null(model_args_list$test_perm)) 1:n_tests else model_args_list$test_perm),
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
                handle_numerical_issues <- if_null_then_set_to( model_args_list$handle_numerical_issues,
                                                                1,
                                                                debugging = FALSE)
                fully_vectorised <- if_null_then_set_to(model_args_list$fully_vectorised,
                                                        1,
                                                        debugging = FALSE)
                
                Phi_type <- model_args_list$Phi_type
                Phi_type_int <- ifelse(Phi_type == "Phi", 1, 2)
                
                n_pops <- if_null_then_set_to(model_args_list$n_pops,
                                              1,
                                              debugging = FALSE)
                pop <- if_null_then_set_to(model_args_list$pop,
                                           c(rep(1, N)),
                                           debugging = FALSE)
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
          
        } else if (Model_type %in% c("LC_MVOP", "CLC_MVOP", "CLC-MVOP")) {
          
          if (Model_type == "LC_MVOP") {
          
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
                handle_numerical_issues <- if_null_then_set_to( model_args_list$handle_numerical_issues,
                                                                1,
                                                                debugging = FALSE)
                fully_vectorised <- if_null_then_set_to(model_args_list$fully_vectorised,
                                                        1,
                                                        debugging = FALSE)
                ##
                Phi_type <- model_args_list$Phi_type
                Phi_type_int <- ifelse(Phi_type == "Phi", 1, 2)
                ##
                n_pops <- if_null_then_set_to(model_args_list$n_pops,
                                              1,
                                              debugging = FALSE)
                pop <- if_null_then_set_to(model_args_list$pop,
                                           c(rep(1, N)),
                                           debugging = FALSE)
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
                  ## (the baseline cases given by the user, test by test, padded with 0)
                  for (baseline_name in c("baseline_case_nd", "baseline_case_d")) {
                        given <- model_args_list[[baseline_name]]
                        if (!is.null(given) && (length(given) >= t) && length(given[[t]])) {
                              values <- as.numeric(given[[t]])[seq_len(min(length(given[[t]]), n_covariates_max))]
                              if (baseline_name == "baseline_case_nd") baseline_case_nd[[t]][seq_along(values)] <- values
                              else                                      baseline_case_d[[t]][seq_along(values)]  <- values
                        }
                  }
                }
                ##
                C_raw_lower = model_args_list$C_raw_lower
                C_raw_upper = model_args_list$C_raw_upper
                ##
                print(paste0("model_args_list$C_raw_lower = ", model_args_list$C_raw_lower))
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
                                       test_perm = as.integer(if (is.null(model_args_list$test_perm)) 1:n_tests else model_args_list$test_perm),
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
          
          } else {
          
                prior_only <- as.integer(isTRUE(as.logical(model_args_list$prior_only)))
                ##
                corr_force_positive <- as.integer(isTRUE(as.logical(model_args_list$corr_force_positive)))
                lb_corr <- model_args_list$lb_corr
                ub_corr <- model_args_list$ub_corr
                ##
                prior_LKJ <- model_args_list$prior_LKJ
                if (!(is.matrix(prior_LKJ))) {
                  prior_LKJ <- matrix(
                    as.numeric(prior_LKJ),
                    ncol = 1L
                  )
                }
                ##
                overflow_threshold <- as.numeric(model_args_list$overflow_threshold)
                underflow_threshold <- as.numeric(model_args_list$underflow_threshold)
                handle_numerical_issues <- as.integer(isTRUE(as.logical(if_null_then_set_to( model_args_list$handle_numerical_issues,
                                                                                             TRUE,
                                                                                             debugging = FALSE))))
                fully_vectorised <- as.integer(if_null_then_set_to( model_args_list$fully_vectorised,
                                                                    1L,
                                                                    debugging = FALSE))
                ##
                Phi_type <- model_args_list$Phi_type
                Phi_type_int <- if (identical(Phi_type, "Phi")) 1L else 2L
                ##
                n_pops_clc <- as.integer(model_args_list$n_pops)
                pop_clc <- array(
                  as.integer(model_args_list$pop),
                  dim = N
                )
                ##
                ## ---- Ordinal-only stuff:
                ##
                n_binary_tests <- as.integer(model_args_list$n_binary_tests)
                n_ordinal_tests <- as.integer(model_args_list$n_ordinal_tests)
                n_ord_pad <- max(
                  1L,
                  n_ordinal_tests
                )
                ##
                n_cat_per_ord_test <- if (n_ordinal_tests > 0L) {
                  array(
                    as.integer(model_args_list$n_cat_per_ord_test),
                    dim = n_ordinal_tests
                  )
                } else {
                  array(
                    integer(0L),
                    dim = 0L
                  )
                }
                n_thr_per_ord_test <- if (n_ordinal_tests > 0L) {
                  array(
                    as.integer(model_args_list$n_thr_per_ord_test),
                    dim = n_ordinal_tests
                  )
                } else {
                  array(
                    integer(0L),
                    dim = 0L
                  )
                }
                ##
                prior_dirichlet_alpha <- lapply(
                  seq_len(n_class),
                  function(class_index) {
                    if (n_ordinal_tests > 0L) {
                      alpha_matrix <- as.matrix(model_args_list$prior_dirichlet_alpha[[class_index]])
                      storage.mode(alpha_matrix) <- "double"
                      return(alpha_matrix)
                    }
                    return(matrix(
                      1.0,
                      nrow = 1L,
                      ncol = 1L
                    ))
                  }
                )
                ##
                ## ---- Build 4D X, 3D baseline_case, and 4-class Se/Sp arrays for Stan JSON:
                ##
                X_stan_arr <- array(
                  0.0,
                  dim = c(n_class, n_tests, N, n_covariates_max)
                )
                for (class_index in seq_len(n_class)) {
                  for (test_index in seq_len(n_tests)) {
                    X_stan_arr[class_index, test_index, , ] <- model_args_list$X[[class_index]][[test_index]]
                  }
                }
                baseline_case_stan_arr <- array(
                  0.0,
                  dim = c(n_class, n_tests, n_covariates_max)
                )
                for (class_index in seq_len(n_class)) {
                  for (test_index in seq_len(n_tests)) {
                    baseline_case_stan_arr[class_index, test_index, ] <-
                      as.numeric(model_args_list$baseline_case[[class_index]][, test_index])
                  }
                }
                n_chosen_mat_clc <- model_args_list$n_chosen_cutoffs
                n_chosen_max_clc <- as.integer(model_args_list$n_chosen_cutoffs_max)
                chosen_cutoffs_stan_arr <- array(
                  1L,
                  dim = c(n_class, n_ord_pad, n_chosen_max_clc)
                )
                link_ms_stan_arr <- array(
                  0.0,
                  dim = c(n_class, n_ord_pad, n_chosen_max_clc, 2L)
                )
                link_ms_stan_arr[, , , 2L] <- 1.0
                if (n_ordinal_tests > 0L) {
                  for (class_index in seq_len(n_class)) {
                    for (ord_index in seq_len(n_ordinal_tests)) {
                      m_val <- as.integer(n_chosen_mat_clc[class_index, ord_index])
                      if (m_val > 0L) {
                        if (!is.null(model_args_list$chosen_cutoffs)) {
                          chosen_cutoffs_stan_arr[class_index, ord_index, 1:m_val] <-
                            as.integer(model_args_list$chosen_cutoffs[class_index, ord_index, 1:m_val])
                        }
                        if (!is.null(model_args_list$prior_Se_Sp_link_mean_SD)) {
                          raw_link <- model_args_list$prior_Se_Sp_link_mean_SD
                          link_slice <- if (is.list(raw_link)) {
                            raw_link[[class_index]][[ord_index]]
                          } else {
                            raw_link[class_index, ord_index, , ]
                          }
                          link_mat <- matrix(
                            as.numeric(link_slice),
                            ncol = 2L
                          )
                          link_ms_stan_arr[class_index, ord_index, 1:m_val, 1L] <- link_mat[1:m_val, 1L]
                          link_ms_stan_arr[class_index, ord_index, 1:m_val, 2L] <- link_mat[1:m_val, 2L]
                        }
                      }
                    }
                  }
                }
                ##
                C_raw_lower <- as.numeric(model_args_list$C_raw_lower)
                C_raw_upper <- as.numeric(model_args_list$C_raw_upper)
                n_prev_cov_clc <- as.integer(model_args_list$n_prevalence_covariates)
                n_standardisation_clc <- as.integer(model_args_list$n_standardisation)
                ##
                ## ---- Make "Stan_data_list" matching CLC_MVOP_cpp_skeleton.stan:
                ##
                Stan_data_list <- list(
                  N = as.integer(N),
                  n_tests = as.integer(n_tests),
                  n_binary_tests = n_binary_tests,
                  n_ordinal_tests = n_ordinal_tests,
                  n_cat_per_ord_test = n_cat_per_ord_test,
                  n_thr_per_ord_test = n_thr_per_ord_test,
                  prior_dirichlet_alpha = prior_dirichlet_alpha,
                  y = model_args_list$y,
                  y_upper_when_interval_censored = model_args_list$y_upper_when_interval_censored,
                  n_class = as.integer(n_class),
                  test_perm = array(
                    as.integer(model_args_list$test_perm),
                    dim = n_tests
                  ),
                  n_cat_per_test = array(
                    as.integer(model_args_list$n_cat_per_test),
                    dim = n_tests
                  ),
                  n_pops = n_pops_clc,
                  pop = pop_clc,
                  pop_weight = array(
                    as.numeric(model_args_list$pop_weight),
                    dim = n_pops_clc
                  ),
                  class_D = model_args_list$class_D,
                  test_condition = array(
                    as.integer(model_args_list$test_condition),
                    dim = n_tests
                  ),
                  is_perfect_ref = array(
                    as.integer(model_args_list$is_perfect_ref),
                    dim = n_tests
                  ),
                  perfect_ref_M = as.numeric(model_args_list$perfect_ref_M),
                  n_covariates_max = as.integer(n_covariates_max),
                  X = X_stan_arr,
                  n_covs_per_outcome = model_args_list$n_covariates_per_outcome_mat,
                  n_prevalence_covariates = n_prev_cov_clc,
                  X_prevalence = model_args_list$X_prevalence,
                  corr_force_positive = corr_force_positive,
                  known_num = as.integer(known_num),
                  pool_correlations_across_classes = as.integer(model_args_list$pool_correlations_across_classes),
                  known_values_list = known_values_list,
                  known_values_indicator_list = known_values_indicator_list,
                  lb_corr = lb_corr,
                  ub_corr = ub_corr,
                  prior_LKJ = prior_LKJ,
                  overflow_threshold = overflow_threshold,
                  underflow_threshold = underflow_threshold,
                  C_raw_lower = C_raw_lower,
                  C_raw_upper = C_raw_upper,
                  prior_only = prior_only,
                  prior_beta_mean = model_args_list$prior_beta_mean,
                  prior_beta_sd = model_args_list$prior_beta_sd,
                  Phi_type = Phi_type_int,
                  handle_numerical_issues = handle_numerical_issues,
                  fully_vectorised = fully_vectorised,
                  use_pop_prev_prior = as.integer(model_args_list$use_pop_prev_prior),
                  baseline_case = baseline_case_stan_arr,
                  use_Se_Sp_prior = as.integer(model_args_list$use_Se_Sp_prior),
                  n_chosen_cutoffs_max = n_chosen_max_clc,
                  n_chosen_cutoffs = n_chosen_mat_clc,
                  chosen_cutoffs = chosen_cutoffs_stan_arr,
                  prior_Se_Sp_link_mean_SD = link_ms_stan_arr,
                  use_pop_Se_Sp_prior = as.integer(model_args_list$use_pop_Se_Sp_prior),
                  n_standardisation = n_standardisation_clc,
                  standardisation_index = array(
                    as.integer(model_args_list$standardisation_index),
                    dim = n_standardisation_clc
                  ),
                  standardisation_weight = array(
                    as.numeric(model_args_list$standardisation_weight),
                    dim = n_standardisation_clc
                  ),
                  use_label_constraint = as.integer(model_args_list$use_label_constraint),
                  reference_test = array(
                    as.integer(model_args_list$reference_test),
                    dim = 2L
                  ),
                  cutoff_category = array(
                    as.integer(model_args_list$cutoff_category),
                    dim = n_tests
                  ),
                  save_subject_probs = as.integer(model_args_list$save_subject_probs)
                )
          
          }
          
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
                handle_numerical_issues <- if_null_then_set_to( model_args_list$handle_numerical_issues,
                                                                1,
                                                                debugging = FALSE)
                fully_vectorised <- if_null_then_set_to(model_args_list$fully_vectorised,
                                                        1,
                                                        debugging = FALSE)
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
                  ## (the baseline cases given by the user, test by test, padded with 0)
                  for (baseline_name in c("baseline_case_nd", "baseline_case_d")) {
                        given <- model_args_list[[baseline_name]]
                        if (!is.null(given) && (length(given) >= t) && length(given[[t]])) {
                              values <- as.numeric(given[[t]])[seq_len(min(length(given[[t]]), n_covariates_max))]
                              if (baseline_name == "baseline_case_nd") baseline_case_nd[[t]][seq_along(values)] <- values
                              else                                      baseline_case_d[[t]][seq_along(values)]  <- values
                        }
                  }
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
                                       test_perm = as.integer(if (is.null(model_args_list$test_perm)) 1:n_tests else model_args_list$test_perm),
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

        if (Model_type %in% c("LC_MVOP", "MVOP", "CLC_MVOP", "CLC-MVOP")) {
          ## These remain Stan arrays when there is exactly one ordinal test.
          Stan_data_list$n_cat_per_ord_test <-  array(Stan_data_list$n_cat_per_ord_test, dim = n_ordinal_tests)
          Stan_data_list$n_thr_per_ord_test <-  array(Stan_data_list$n_thr_per_ord_test, dim = n_ordinal_tests)
        }
        return(Stan_data_list)
  
}










# 
# 






















