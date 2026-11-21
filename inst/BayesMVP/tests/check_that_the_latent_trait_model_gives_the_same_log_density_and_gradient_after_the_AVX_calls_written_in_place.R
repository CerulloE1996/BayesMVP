#### =========================================================================================================
## check_that_the_latent_trait_model_gives_the_same_log_density_and_gradient_after_the_AVX_calls_written_in_place.R
##
## The C++ of BayesMVP's latent trait model (LC_LT_functions/ and the MVP functions it calls) used
## fn_EIGEN_double(), which makes a new vector on each call; it now uses fn_EIGEN_double_into() and
## fn_EIGEN_double_in_a_per_thread_buffer() (fn_wrappers_overall.hpp), which give the same values with no new
## vector. This script evaluates the model's log density and gradient (BayesMVP:::Rcpp_wrapper_fn_lp_grad, one
## thread) at fixed points, for every math backend of the machine, both Phi types, 1 and 3 chunks, the gradient
## as sampled and the PartialLog gradient, with the BayesMVP of the library given, and saves them; in compare
## mode it compares two such files (the previous install against the new one).
##
## Reads:   the BayesMVP installed in library_of_the_BayesMVP=<path> (first in .libPaths())
## Writes:  save_as=<file>.rds (the log densities and gradients); in compare mode, a line per case and a verdict
##
## Usage:   Rscript check_that_the_latent_trait_model_gives_the_same_log_density_and_gradient_after_the_AVX_
##                  calls_written_in_place.R library_of_the_BayesMVP=<path> save_as=<file>.rds
##          Rscript check_that_the_latent_trait_model_gives_the_same_log_density_and_gradient_after_the_AVX_
##                  calls_written_in_place.R compare=<before>.rds,<after>.rds
#### =========================================================================================================
##
##
## ------- Settings:  --------------------------------------------------------------------------------------------
##
{
  arguments <-  commandArgs(trailingOnly = TRUE)
  fn_argument <-  function(name) {
        argument <-  grep(paste0("^", name, "="), arguments, value = TRUE)
        if (length(argument) == 0) return(NA_character_)
        return(sub(paste0("^", name, "="), "", argument[1]))
  }
  library_of_the_BayesMVP <-  fn_argument("library_of_the_BayesMVP")
  save_as <-  fn_argument("save_as")
  files_to_compare <-  fn_argument("compare")
  ##
  N <-  1000
  n_tests <-  5
  n_class <-  2
  seed <-  123
  num_chunks_values <-  c(1, 3)
  Phi_types <-  c("Phi", "Phi_approx")
  evaluations <-  c("as_sampled", "PartialLog")
  SD_of_the_points <-  c(0.3, 1, 3)   #### (3: intercepts far out, so that the underflow / overflow rows are used)
  largest_relative_difference_allowed <-  1e-12
}
##
##
## ------- Compare mode:  ----------------------------------------------------------------------------------------
##
if (!is.na(files_to_compare)) {
      ##
      files <-  strsplit(files_to_compare, ",")[[1]]
      results_before <-  readRDS(files[1])
      results_after <-  readRDS(files[2])
      stopifnot(identical(names(results_before), names(results_after)))
      ##
      largest_relative_difference_of_every_case <-  sapply(names(results_before), function(case_name) {
            before <-  results_before[[case_name]]
            after <-  results_after[[case_name]]
            if (length(before) != length(after)) return(Inf)
            finite_both <-  is.finite(before) & is.finite(after)
            same_non_finite <-  !is.finite(before) & !is.finite(after) & (is.na(before) == is.na(after))
            same_non_finite <-  same_non_finite & (is.na(before) | before == after)
            if (any(!same_non_finite & !finite_both)) return(Inf)
            if (!any(finite_both)) return(0)
            scale <-  max(abs(before[finite_both]), 1e-300)
            return(max(abs(before[finite_both] - after[finite_both])) / scale)
      })
      n_identical <-  sum(sapply(names(results_before), function(case_name) {
            identical(results_before[[case_name]], results_after[[case_name]])
      }))
      for (case_name in names(largest_relative_difference_of_every_case)) {
            message(paste0(case_name, ": largest relative difference ",
                           formatC(largest_relative_difference_of_every_case[[case_name]], format = "e",
                               digits = 2)))
      }
      every_case_passed <-  all(largest_relative_difference_of_every_case <= largest_relative_difference_allowed)
      message(paste0(n_identical, " of ", length(results_before),
                     " cases identical bit for bit; largest relative ",
                     "difference ", formatC(max(largest_relative_difference_of_every_case), format = "e",
                                            digits = 2),
                     " (allowed: ", largest_relative_difference_allowed, ")"))
      message(if (every_case_passed) "PASSED" else "FAILED")
      quit(status = if (every_case_passed) 0 else 1)
##
}
##
##
## ------- The BayesMVP of the library given:  -------------------------------------------------------------------
##
.libPaths(c(library_of_the_BayesMVP, .libPaths()))
message(paste0("BayesMVP from: ", find.package("BayesMVP")))
vect_types <-  c("Stan", "AVX2")
if (any(grepl("avx512f", readLines("/proc/cpuinfo", warn = FALSE)))) vect_types <-  c(vect_types, "AVX512")
##
##
## ------- Data: two latent classes with a latent trait, as in BayesMVP's Run_latent_trait.R example:  -----------
##
set.seed(seed, kind = "L'Ecuyer-CMRG")
latent_class <-  stats::rbinom(n = N, size = 1, prob = 0.30)
latent_trait <-  stats::rnorm(n = N)
class_shift <-  ifelse(latent_class == 1, 0.70, -0.70)
trait_loading <-  ifelse(latent_class == 1, 1.10, 0.75)
test_shift <-  seq(-0.50, 0.50, length.out = n_tests)
latent_values <-  outer(class_shift + trait_loading * latent_trait, rep(1, n_tests))
latent_values <-  latent_values + matrix(rep(test_shift, each = N), nrow = N, ncol = n_tests)
latent_values <-  latent_values + matrix(stats::rnorm(N * n_tests), nrow = N, ncol = n_tests)
y <-  1 * (latent_values > 0)
storage.mode(y) <-  "integer"
##
##
## ------- The log density and gradient of every case:  ----------------------------------------------------------
##
results <-  list()
for (vect_type in vect_types) {
      for (Phi_type in Phi_types) {
            for (num_chunks in num_chunks_values) {
                  ##
                  model_args_list <-  list( y = y,
                                            N = N,
                                            n_class = n_class,
                                            n_pops = 1,
                                            X = lapply(1:n_class, function(class_index) {
                                                  lapply(1:n_tests, function(test_index) matrix(1, N, 1))
                                            }),
                                            n_covariates_per_outcome_mat = matrix(1, n_class, n_tests),
                                            num_chunks = num_chunks,
                                            prior_coeffs_mean_mat = lapply(1:n_class, function(class_index) {
                                                  matrix(if (class_index == 1) -0.70 else 0.70, 1, n_tests)
                                            }),
                                            prior_coeffs_sd_mat = lapply(1:n_class, function(class_index) {
                                                  matrix(0.75, 1, n_tests)
                                            }),
                                            LT_b_priors_shape = matrix(1, n_class, n_tests),
                                            LT_b_priors_scale = matrix(1, n_class, n_tests),
                                            prev_prior_a = 1,
                                            prev_prior_b = 1,
                                            vect_type = vect_type,
                                            Phi_type = Phi_type,
                                            inv_Phi_type = if (Phi_type == "Phi_approx") "inv_Phi_approx" else
                                                "inv_Phi")
                  init <-  NULL
                  invisible(utils::capture.output(
                        init <-  BayesMVP::init_hard_coded_model(Model_type = "latent_trait",
                                                                 model_args_list = model_args_list)))
                  n_params_main <-  init$model_args_list$n_params_main
                  n_nuisance <-  init$model_args_list$n_nuisance
                  y_of_the_model <-  init$model_args_list$y
                  storage.mode(y_of_the_model) <-  "integer"
                  ##
                  for (SD_of_the_point in SD_of_the_points) {
                        set.seed(seed + round(10 * SD_of_the_point))
                        theta_main <-  stats::rnorm(n_params_main, mean = 0, sd = SD_of_the_point)
                        theta_us <-  stats::rnorm(n_nuisance, mean = 0, sd = 1)
                        for (evaluation in evaluations) {
                              case_name <-  paste0(vect_type, "_", Phi_type, "_chunks_", num_chunks, "_SD_",
                                                   SD_of_the_point, "_", evaluation)
                              lp_grad <-  BayesMVP:::Rcpp_wrapper_fn_lp_grad(
                                                Model_type = "latent_trait",
                                                force_autodiff = FALSE,
                                                force_PartialLog = (evaluation == "PartialLog"),
                                                multi_attempts = TRUE,
                                                theta_main_vec = theta_main,
                                                theta_us_vec = theta_us,
                                                y = y_of_the_model,
                                                grad_option = "all",
                                                Model_args_as_Rcpp_List = init$Model_args_as_Rcpp_List)
                              results[[case_name]] <-  as.numeric(lp_grad)
                              message(paste0(case_name, ": log density ", format(lp_grad[1], digits = 12)))
                        }
                  }
            ##
            }
      }
}
saveRDS(results, save_as)
message(paste0(length(results), " cases saved in ", save_as))






















