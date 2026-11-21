##
## ==== check_Se_Sp_prior_files_vs_parents_log_density_gradient_LC_MVOP.R =============================================
##
## ---- Gate 4(a): every <parent>_Se_Sp_prior.stan with use_Se_Sp_prior = 0 against its parent, the log density and
## ---- the gradient at 6 saved points drawn once; gate: log density within 1e-8 (relative), gradient within 1e-8 of
## ---- its largest element.
##
{
      roots_of_the_work_folder <-  c("/home/enzocerullo/Documents/Work/PhD_work",
                                     "/home/enzo/Documents/Work/PhD_work")
      root_of_the_work_folder <-  roots_of_the_work_folder[dir.exists(roots_of_the_work_folder)][1]
      models_folder <-  file.path(root_of_the_work_folder, "R_packages", "BayesMVP", "inst", "BayesMVP", "inst",
                                  "stan_models")
      shared_folder <-  file.path(root_of_the_work_folder, "Alg_paper_analysis", "0_utilities", "shared_functions")
      PS6_functions_folder <-  file.path(root_of_the_work_folder, "Alg_paper_analysis", "1_appendix_pilot_studies",
                                         "ps_6_basic_MCMC_settings_Stan", "functions")
      checks_folder <-  file.path("/tmp", "w13_gate")
      libraries_folder <-  file.path(checks_folder, "libraries")
      results_file <-  file.path(models_folder, "Se_Sp_prior_gate_4a_results.csv")
      model_names <-  c("LC_MVOP_PartialLog", "LC_MVOP_PartialLog_v1", "LC_MVOP_PartialLog_v2",
                        "LC_MVOP_PartialLog_v2_reduce_sum", "LC_MVOP_bin_PartialLog_v5")
      n_points <-  12
      SD_of_the_noise_of_the_points <-  0.3
      tolerance_of_the_rule <-  1e-8
      seed <-  1
      dir.create(libraries_folder, showWarnings = FALSE, recursive = TRUE)
}
##
##
## ---- The NicoStan routine this check runs: ------------------------------------------------------------------
##
loadNamespace("NicoStan")
fn_log_density_at_points <-  get("fn_log_density_and_gradient_of_compiled_Stan_model_at_points",
                                 envir = asNamespace("NicoStan"))
##
##
## ---- The data of one LC-MVOP data set, as the PS6 harness builds it: ----------------------------------------
##
PS6_functions <-  new.env()
sys.source(file.path(shared_folder, "R_fn_sim_mixed_bin_ord_LC_MVOP_data.R"), envir = PS6_functions)
sys.source(file.path(PS6_functions_folder, "ps_6_MCMC_settings_Stan_functions.R"), envir = PS6_functions)
ordinal_simulation <-  PS6_functions$R_fn_simulate_mixed_binary_ordinal_LC_MVOP_data(
                             study_type = "algorithm_benchmarking", seed = 123, grouping = FALSE,
                             corr_force_positive_DGM = FALSE, dirichlet_prior_conc = 50)
n_tests <-  ordinal_simulation$n_tests
n_pops <-  ordinal_simulation$DGM_info$n_pops
prior_beta_mean <-  lapply(1:2, function(class) matrix(0, nrow = 3, ncol = n_tests))
prior_beta_sd <-  lapply(1:2, function(class) matrix(1, nrow = 3, ncol = n_tests))
data_of_each_N <-  PS6_functions$R_fn_prep_Stan_ordinal_data(
                          ord_sim_outs = ordinal_simulation, n_covariates_max = 3,
                          prior_beta_mean = prior_beta_mean, prior_beta_sd = prior_beta_sd,
                          prior_LKJ = matrix(c(4, 4), ncol = 1),
                          prior_prev_a = matrix(1, nrow = n_pops, ncol = 1),
                          prior_prev_b = matrix(1, nrow = n_pops, ncol = 1))
base_data <-  data_of_each_N[[1]]
n_ordinal_tests <-  base_data$n_ordinal_tests
message(paste0("Data: N = ", base_data$N, ", n_tests = ", base_data$n_tests, " (", base_data$n_binary_tests,
               " binary, ", n_ordinal_tests, " ordinal)"))
##
##
## ---- The Se/Sp prior fields with the flag off, in the shape of each file's data block: ----------------------
##
fn_Se_Sp_fields_with_the_flag_off <-  function(model_name) {

      n_chosen_max <-  1
      fields <-  list(use_Se_Sp_prior = 0,
                      Se_Sp_prior_type = 1,
                      n_chosen_max = n_chosen_max)
      if (grepl("bin_PartialLog", model_name)) {
            fields$n_chosen <-  array(0L, dim = 2)
            fields$chosen_tests <-  array(1L, dim = c(2, n_chosen_max))
            fields$prior_Se_Sp_dirichlet_A <-  array(1, dim = c(2, n_chosen_max, 2))
            fields$prior_Se_Sp_beta <-  array(1, dim = c(2, n_chosen_max, 2))
            fields$prior_Se_Sp_link_mean_SD <-  array(0, dim = c(2, n_chosen_max, 2))
      } else {
            fields$n_chosen <-  array(0L, dim = c(2, n_ordinal_tests))
            fields$chosen_cutoffs <-  array(1L, dim = c(2, n_ordinal_tests, n_chosen_max))
            fields$prior_Se_Sp_dirichlet_A <-  array(1, dim = c(2, n_ordinal_tests, n_chosen_max + 1))
            fields$prior_Se_Sp_beta <-  array(1, dim = c(2, n_ordinal_tests, n_chosen_max, 2))
            fields$prior_Se_Sp_link_mean_SD <-  array(0, dim = c(2, n_ordinal_tests, n_chosen_max, 2))
      }
      fields

}
##
##
## ---- The data as JSON text (a one-value container written as an array of length 1): -------------------------
##
fn_containers_of_one_value_as_arrays <-  function(Stan_data_list) {

      lines_of_the_data_block <-  fn_lines_of_the_data_block(stan_file_of_the_model)
      for (name in names(Stan_data_list)) {
            declared_as_a_container <-  any(grepl(paste0("^ *(vector|row_vector|array)[^;]*[] >]", name, " *;"),
                                                  lines_of_the_data_block))
            if (declared_as_a_container && (length(Stan_data_list[[name]]) == 1) &&
                is.null(dim(Stan_data_list[[name]]))) {
                  Stan_data_list[[name]] <-  array(Stan_data_list[[name]], dim = 1)
            }
      }
      Stan_data_list

}
fn_JSON_text_of <-  function(Stan_data_list) {

      Stan_data_list <-  fn_containers_of_one_value_as_arrays(Stan_data_list)
      JSON_file <-  tempfile(fileext = ".json")
      cmdstanr::write_stan_json(data = Stan_data_list, file = JSON_file)
      paste(readLines(JSON_file, warn = FALSE), collapse = "\n")

}
fn_names_in_the_data_block <-  function(stan_file) {

      lines_of_the_declarations <-  grep(";", fn_lines_of_the_data_block(stan_file), value = TRUE)
      sub(".*[<> ]([A-Za-z_][A-Za-z_0-9]*) *;.*$", "\\1", lines_of_the_declarations)

}
fn_fields_the_data_set_does_not_carry <-  function() {

      list(chunk_size = 1L)

}
fn_data_for_model <-  function(stan_files,
                               Stan_data_list) {

      declared <-  unique(unlist(lapply(stan_files, fn_names_in_the_data_block)))
      stan_file <-  stan_files[1]
      ## (the parents that carry one shared prior_dirichlet_alpha take one class's matrix; the two classes'
      ##  alphas are the same in this data set, checked rather than assumed)
      line_of_the_alpha <-  grep("prior_dirichlet_alpha", fn_lines_of_the_data_block(stan_file), value = TRUE)
      if ((length(line_of_the_alpha) == 1) && grepl("^ *matrix", line_of_the_alpha) &&
          ("prior_dirichlet_alpha" %in% names(Stan_data_list))) {
            alpha_of_each_class <-  Stan_data_list$prior_dirichlet_alpha
            stopifnot(identical(alpha_of_each_class[[1]], alpha_of_each_class[[2]]))
            Stan_data_list$prior_dirichlet_alpha <-  alpha_of_each_class[[1]]
      }
      Stan_data_list$prior_p_alpha <-  Stan_data_list$prior_prev_a
      Stan_data_list$prior_p_beta <-  Stan_data_list$prior_prev_b
      missing <-  setdiff(declared, names(Stan_data_list))
      fields_of_the_model <-  fn_fields_the_data_set_does_not_carry()
      for (name in missing) {
            if (!name %in% names(fields_of_the_model)) {
                  stop(paste0("The data set carries no value for the declared field ", name, " of ", stan_file))
            }
            Stan_data_list[[name]] <-  fields_of_the_model[[name]]
      }
      Stan_data_list[intersect(names(Stan_data_list), declared)]

}
fn_lines_of_the_data_block <-  function(stan_file) {

      lines_of_the_model <-  readLines(stan_file, warn = FALSE)
      first_line <-  grep("^data *[{]", lines_of_the_model)[1]
      last_line <-  grep("^transformed data", lines_of_the_model)[1]
      sub("//.*$", "", lines_of_the_model[first_line:last_line])

}
##
##
## ---- The builds, one at a time, from a copy in the checks folder: -------------------------------------------
##
fn_build_unless_built <-  function(stan_file) {

      copy_file <-  file.path(libraries_folder, basename(stan_file))
      already_same <-  file.exists(copy_file) &&
                       identical(unname(tools::md5sum(stan_file)), unname(tools::md5sum(copy_file)))
      if (already_same) {
            message(paste0("Source unchanged: ", basename(copy_file)))
      } else {
            stopifnot(file.copy(stan_file, copy_file, overwrite = TRUE, copy.date = TRUE))
      }
      so_file <-  sub("[.]stan$", "_model.so", copy_file)
      if (file.exists(so_file) && (file.info(so_file)$mtime > file.info(copy_file)$mtime)) {
            message(paste0("Already built: ", so_file))
            return(so_file)
      }
      start <-  Sys.time()
      message(paste0("Building ", basename(stan_file)))
      built <-  bridgestan::compile_model(copy_file, stanc_args = NULL,
                                          make_args = c("STAN_THREADS=true"))
      stopifnot(file.exists(built))
      message(paste0("Built ", built, " in ",
                     formatC(as.numeric(difftime(Sys.time(), start, units = "secs")), format = "f", digits = 1),
                     " s"))
      built

}
##
##
## ---- The points (the model's ordering of the parameters): ---------------------------------------------------
##
fn_absolute_cutpoints <-  function() {

      lapply(n_thr_per_ord_test, function(n_thr) {
            c(-2.0 + stats::rnorm(1, mean = 0, sd = 0.02),
              -2.0 + stats::rnorm(1, mean = 0, sd = 0.02) +
                    cumsum(exp(stats::rnorm(n_thr - 1, mean = log(4.0 / n_thr), sd = 0.02))))
      })

}
fn_points <-  function(parameter_names) {

      lapply(seq_len(n_points), function(point_index) {
            cutpoints_of_class_1 <-  fn_absolute_cutpoints()
            point <-  rep(0, length(parameter_names))
            is_beta <-  grepl("^beta_vec", parameter_names)
            point[is_beta] <-  stats::rnorm(sum(is_beta), mean = 0, sd = 0.2)
            is_p <-  grepl("^p_raw", parameter_names)
            point[is_p] <-  atanh(2 * stats::runif(sum(is_p), 0.05, 0.3) - 1)
            is_u <-  grepl("^u_raw", parameter_names)
            point[is_u] <-  fn_u_raw_values(parameter_names[is_u], cutpoints_of_class_1)
            fn_cutpoint_terms(point, parameter_names, cutpoints_of_class_1)
      })

}
fn_u_raw_values <-  function(names_of_u,
                             cutpoints_of_class_1) {

      ## (u_raw drawn from the standard normal, as in the model's own prior on it; drawing it inside
      ##  each observation's category interval pushes the model's internal normal-from-uniform step to
      ##  a zero probability and returns NaN)
      rep(stats::rnorm(length(names_of_u), mean = 0, sd = 0.7))

}
fn_cutpoint_terms <-  function(point,
                               parameter_names,
                               cutpoints_of_class_1) {

      is_C <-  grepl("^C_raw", parameter_names) | grepl("^C_unc", parameter_names)
      if (!any(is_C)) return(point)
      names_of_C <-  parameter_names[is_C]
      positions <-  which(is_C)
      class_of_each <-  as.integer(sub("^[A-Za-z_]*[.]", "", sub("[.][0-9]+$", "", names_of_C)))
      index_of_each <-  as.integer(sub("^.*[.]", "", names_of_C))
      unconstrained_is_the_log_spacing <-  all(grepl("_vec$|_vec[.]", names_of_C))
      starts_of_block <-  cumsum(c(1, utils::head(n_thr_per_ord_test, -1)))
      for (c in sort(unique(class_of_each))) {
            for (block in seq_along(n_thr_per_ord_test)) {
                  n_thr <-  n_thr_per_ord_test[block]
                  rows_of_block <-  which((class_of_each == c) &
                                          (index_of_each >= starts_of_block[block]) &
                                          (index_of_each < starts_of_block[block] + n_thr))
                  values_of_C <-  cutpoints_of_class_1[[block]]
                  if (c > 1) {
                        values_of_C <-  values_of_C + stats::rnorm(1, mean = 0, sd = 0.05)
                  }
                  values_of_C <-  if (unconstrained_is_the_log_spacing) {
                        c(values_of_C[1], log(diff(values_of_C)))
                  } else {
                        atanh(2 * (values_of_C - bounds_of_the_cutpoints[1]) /
                              (bounds_of_the_cutpoints[2] - bounds_of_the_cutpoints[1]) - 1)
                  }
                  point[positions[rows_of_block]] <-  values_of_C
            }
      }
      point

}
fn_log_density_in_a_fresh_R_process <-  function(so_file,
                                                 JSON_text,
                                                 points) {

      callr::r(func  = fn_log_density_at_points,
               args  = list(model_so_file          = so_file,
                            JSON_text_of_Stan_data = JSON_text,
                            points                 = points,
                            propto                 = FALSE,
                            jacobian               = TRUE,
                            seed                   = 1))

}
##
##
## ---- Gate 4(a), model by model: -----------------------------------------------------------------------------
##
RNGkind("L'Ecuyer-CMRG")
set.seed(seed)
rows <-  list()
for (model_name in model_names) {

      parent_file <-  file.path(models_folder, paste0(model_name, ".stan"))
      new_file <-  file.path(models_folder, paste0(model_name, "_Se_Sp_prior.stan"))
      stopifnot(file.exists(parent_file), file.exists(new_file))
      stan_file_of_the_model <-  parent_file
      n_thr_per_ord_test <-  as.integer(base_data$n_thr_per_ord_test)
      n_binary_tests_of_the_data <-  as.integer(base_data$n_binary_tests)
      n_cat_per_ord_test_of_the_data <-  as.integer(base_data$n_cat_per_ord_test)
      y_of_the_data <-  base_data$y
      bounds_of_the_cutpoints <-  if (all(c("C_raw_lower", "C_raw_upper") %in% names(base_data))) {
            c(base_data$C_raw_lower[1], base_data$C_raw_upper[1])
      } else {
            numeric(0)
      }
      message(paste0("Cutpoint thresholds: ", paste(n_thr_per_ord_test, collapse = " "),
                     "; bounds: ", paste(bounds_of_the_cutpoints, collapse = " ")))
      message(paste0("=== ", model_name))
      data_list <-  c(base_data, fn_Se_Sp_fields_with_the_flag_off(model_name))
      data_list <-  fn_data_for_model(c(parent_file, new_file), data_list)
      message(paste0("Data fields: ", paste(sort(names(data_list)), collapse = " ")))
      JSON_text <-  fn_JSON_text_of(data_list)
      writeLines(JSON_text, file.path(checks_folder, paste0(model_name, "_data.json")))
      parent_so <-  fn_build_unless_built(parent_file)
      new_so <-  fn_build_unless_built(new_file)
      probe <-  tryCatch(fn_log_density_in_a_fresh_R_process(parent_so, JSON_text, list()),
                         error = function(condition) {
                               cat("\nXXXX construction failed XXXX\n")
                               cat(conditionMessage(condition))
                               cat("\nXXXX end XXXX\n")
                               stop(condition)
                         })
      names_of_C <-  grep("^C_raw|^C_unc", probe$unconstrained_parameter_names, value = TRUE)
      message(paste0("Cutpoint parameter names: ", length(names_of_C), ", first: ",
                     paste(utils::head(names_of_C, 4), collapse = " "), "; last: ",
                     paste(utils::tail(names_of_C, 2), collapse = " ")))
      points <-  fn_points(probe$unconstrained_parameter_names)
      reference <-  fn_log_density_in_a_fresh_R_process(parent_so, JSON_text, points)
      other <-  fn_log_density_in_a_fresh_R_process(new_so, JSON_text, points)
      same_names <-  identical(reference$unconstrained_parameter_names, other$unconstrained_parameter_names)
      if (!same_names) {
            stop(paste0("The unconstrained parameters of ", model_name, " differ from its parent's"))
      }
      finite_in_both <-  is.finite(reference$log_density) & is.finite(other$log_density)
      both_infinite_and_equal <-  (!is.finite(reference$log_density)) & (!is.finite(other$log_density)) &
                                  (!is.na(reference$log_density)) & (!is.na(other$log_density)) &
                                  (reference$log_density == other$log_density)
      unexplained <-  (!finite_in_both) & (!both_infinite_and_equal)
      if (any(unexplained)) {
            stop(paste0("A point disagrees or failed in ", model_name, ": parent ",
                        paste(signif(reference$log_density, 6), collapse = " "), "; new ",
                        paste(signif(other$log_density, 6), collapse = " "), "; messages ",
                        paste(c(reference$error_message_of_each_point,
                                other$error_message_of_each_point), collapse = " | ")))
      }
      if (sum(finite_in_both) < 4) {
            stop(paste0("Only ", sum(finite_in_both), " of ", length(points), " points are finite in both ",
                        model_name, "; the comparison needs at least 4"))
      }
      message(paste0("Points finite in both: ", sum(finite_in_both), " of ", length(points),
                     "; infinite and equal in both: ", sum(both_infinite_and_equal)))
      absolute_difference_of_the_log_density <-  max(abs(other$log_density[finite_in_both] -
                                                         reference$log_density[finite_in_both]))
      relative_difference_of_the_log_density <-  max(abs(other$log_density[finite_in_both] -
                                                         reference$log_density[finite_in_both]) /
                                                     pmax(abs(reference$log_density[finite_in_both]),
                                                          .Machine$double.eps))
      largest_gradient_element <-  max(abs(reference$gradient[finite_in_both, ]))
      largest_gradient_difference <-  max(abs(other$gradient[finite_in_both, ] -
                                              reference$gradient[finite_in_both, ]))
      relative_difference_of_the_gradient <-  largest_gradient_difference / largest_gradient_element
      passes <-  (relative_difference_of_the_log_density <= tolerance_of_the_rule) &&
                 (relative_difference_of_the_gradient <= tolerance_of_the_rule)
      rows[[length(rows) + 1]] <-  data.frame(model = model_name,
                                              n_points = length(points),
                                              n_unconstrained_parameters = length(reference$unconstrained_parameter_names),
                                              largest_absolute_log_density_difference = absolute_difference_of_the_log_density,
                                              largest_relative_log_density_difference = relative_difference_of_the_log_density,
                                              largest_gradient_element = largest_gradient_element,
                                              largest_gradient_difference = largest_gradient_difference,
                                              gradient_difference_over_largest_element = relative_difference_of_the_gradient,
                                              tolerance = tolerance_of_the_rule,
                                              passes_gate_4a = passes,
                                              stringsAsFactors = FALSE)
      message(paste0(model_name, ": log density relative difference ",
                     formatC(relative_difference_of_the_log_density, format = "e", digits = 3),
                     ", gradient difference over its largest element ",
                     formatC(relative_difference_of_the_gradient, format = "e", digits = 3),
                     " -> ", if (passes) "PASS" else "FAIL"))
      utils::write.csv(do.call(rbind, rows), results_file, row.names = FALSE)

}
message(paste0("Saved: ", results_file, " (", sum(do.call(rbind, rows)$passes_gate_4a), " of ", length(rows),
               " models pass gate 4(a))"))
##
##
##
