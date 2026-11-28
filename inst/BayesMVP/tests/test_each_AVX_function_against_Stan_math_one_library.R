##
## ==== test_each_AVX_function_against_Stan_math_one_library.R ==================================================
##
## ---- Evaluates the log density and gradient of one library of the timing model of each elementwise function
##      (each_elementwise_function_on_<backend>.stan: which_function 0-19, on a vector and on a matrix) at the
##      same points for every function, with the library loaded once in this fresh R process, and saves them, so
##      that the AVX2 and AVX-512 libraries can be compared with the Stan math library -
##
## Usage:   Rscript test_each_AVX_function_against_Stan_math_one_library.R <library .so> <output .rds> <N>
##
{
      arguments <-  commandArgs(trailingOnly = TRUE)
      library_file <-  normalizePath(arguments[1], mustWork = TRUE)
      output_file <-  arguments[2]
      N <-  as.numeric(arguments[3])
      library_name <-  tools::file_path_sans_ext(basename(library_file))
      stopifnot(!is.loaded("bs_model_construct_R", PACKAGE = library_name))
      dyn.load(library_file)
}
##
fn_log_density_and_gradient <- function(JSON_text,
                                        point) {
        construction <-  .C("bs_model_construct_R", as.character(JSON_text), as.integer(1), ptr_out = raw(8),
                            err_msg = as.character(""), err_ptr = raw(8), PACKAGE = library_name)
        model_pointer <-  construction$ptr_out
        n_unconstrained <-  .C("bs_param_unc_num_R", model_pointer, return_value = as.integer(0),
                               PACKAGE = library_name)$return_value
        if (is.null(point)) point <-  rep(0, n_unconstrained)
        evaluation <-  .C("bs_log_density_gradient_R", model_pointer, as.logical(FALSE), as.logical(TRUE),
                          as.double(point), log_density = double(1), gradient = double(n_unconstrained),
                          return_code = as.integer(0), err_msg = as.character(""), err_ptr = raw(8),
                          PACKAGE = library_name)
        .C("bs_model_destruct_R", model_pointer, PACKAGE = library_name)
        return(list(n_unconstrained = n_unconstrained, log_density = evaluation$log_density,
                    gradient = evaluation$gradient, return_code = evaluation$return_code))
}
##
results <-  list()
for (which_function in c(0:12, 15:19)) {
      for (shape_is_matrix in c(0, 1)) {
            if ((shape_is_matrix == 1) && !(which_function %in% c(1:12, 15:17))) next
            data_list <-  list(N = as.integer(N), chunk_size = as.integer(N), which_function = as.integer(which_function),
                               shape_is_matrix = as.integer(shape_is_matrix), N_rows = as.integer(N / 5),
                               M = as.integer(50), x_gp = seq(from = -2, to = 2, length.out = 50))
            JSON_text <-  as.character(jsonlite::toJSON(data_list, auto_unbox = TRUE, digits = NA))
            size <-  fn_log_density_and_gradient(JSON_text, NULL)$n_unconstrained
            set.seed(2026)
            for (point_index in 1:3) {
                  point <-  stats::rnorm(size, mean = 0, sd = c(0.3, 1, 2)[point_index])
                  evaluation <-  fn_log_density_and_gradient(JSON_text, point)
                  results[[length(results) + 1]] <-  list(which_function = which_function,
                                                          shape_is_matrix = shape_is_matrix,
                                                          point_index = point_index, evaluation = evaluation)
            }
      }
}
saveRDS(results, output_file)
message(paste0("saved ", length(results), " evaluations of ", library_name))
