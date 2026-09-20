#### =====================================================================================================================================
## test_17_autodiff_fallback.R - same-position dispatch and R option checks, with stub evaluators; no MCMC
## =====================================================================================================================================
##
{
    require(Rcpp)
    require(RcppEigen)
}
##
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
package_directory <-  normalizePath(path = file.path(tests_directory, "..", "..", "inst", "BayesMVP"))
##
fallback_source <-  readLines(con = file.path(package_directory, "src", "MVP_functions", "MVP_lp_grad_multi_attempts.hpp"))
fallback_start <-  grep(pattern = "^inline void fn_lp_grad_MVP_multi_attempts_InPlace_process", x = fallback_source)
fallback_end <-  length(fallback_source)
fallback_source <-  paste(fallback_source[seq.int(from = fallback_start, to = fallback_end)], collapse = "\n")
##
latent_trait_source <-  readLines(con = file.path(package_directory, "src", "LC_LT_functions", "LT_LC_lp_grad_multi_attempts.hpp"))
latent_trait_start <-  grep(pattern = "fn_lp_grad_LT_LC_multi_attempts_InPlace_process", x = latent_trait_source, fixed = TRUE)
latent_trait_source <-  paste(latent_trait_source[seq.int(from = latent_trait_start, to = length(latent_trait_source))], collapse = "\n")
##
main_source <-  readLines(con = file.path(package_directory, "src", "main.cpp"))
option_reader_start <-  grep(pattern = "^bool fn_read_autodiff_fallback_option", x = main_source)
option_reader_end <-  grep(pattern = "^Model_fn_args_struct   convert_R_List_to_Model_fn_args_struct", x = main_source) - 1
option_reader_source <-  paste(main_source[seq.int(from = option_reader_start, to = option_reader_end)], collapse = "\n")
##
## ---- Compile the actual wrapper and option reader, replacing only the expensive model evaluations with controlled stubs ----------------
##
Rcpp::sourceCpp(code = paste0('
#include <RcppEigen.h>
#include <limits>
#include <stdexcept>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp17)]]
struct Model_fn_args_struct {};
struct LC_MVP_workspace_struct {};
int failure_mode;
std::vector<int> attempted_evaluators;
template <typename MatrixType>
bool is_NaN_or_Inf_Eigen(const MatrixType &values) { return !values.allFinite(); }
void fn_stub_evaluation(int evaluator, Eigen::Ref<Eigen::VectorXd> output,
                        const Eigen::Ref<const Eigen::VectorXd> main_parameters,
                        const Eigen::Ref<const Eigen::VectorXd> nuisance_parameters,
                        const Eigen::Ref<const Eigen::MatrixXi> data) {
    if (main_parameters(0) != 1.25 || nuisance_parameters(0) != -0.5 || data(0, 0) != 1) {
        throw std::logic_error("The fallback changed its input position or data.");
    }
    attempted_evaluators.push_back(evaluator);
    output.setConstant(evaluator);
    if ((evaluator == 1 && failure_mode >= 1) || (evaluator == 2 && failure_mode >= 2) ||
        (evaluator == 3 && failure_mode == 3)) output(1) = std::numeric_limits<double>::infinity();
    if (evaluator == 3 && failure_mode == 4) throw std::domain_error("Test autodiff failure");
}
#define STUB_EVALUATOR(function_name, evaluator) \\
void function_name(Eigen::Ref<Eigen::VectorXd> output, \\
                  const Eigen::Ref<const Eigen::VectorXd> main_parameters, \\
                  const Eigen::Ref<const Eigen::VectorXd> nuisance_parameters, \\
                  const Eigen::Ref<const Eigen::MatrixXi> data, const std::string, \\
                  const Model_fn_args_struct &, std::vector<LC_MVP_workspace_struct> &, const int) { \\
    fn_stub_evaluation(evaluator, output, main_parameters, nuisance_parameters, data); \\
}
STUB_EVALUATOR(fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process, 1)
STUB_EVALUATOR(fn_lp_grad_MVP_LC_Pinkney_PartialLog_MD_and_AD_InPlace_process, 2)
STUB_EVALUATOR(fn_lp_and_grad_MVP_Pinkney_AD_log_scale_InPlace_process, 3)
STUB_EVALUATOR(fn_lp_and_grad_std_MVP_Pinkney_AD_log_scale_InPlace_process, 3)
STUB_EVALUATOR(fn_lp_grad_MVOP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process, 1)
STUB_EVALUATOR(fn_lp_grad_MVOP_LC_Pinkney_PartialLog_MD_and_AD_Inplace_process, 2)
STUB_EVALUATOR(fn_lp_and_grad_MVOP_Pinkney_AD_log_scale_InPlace_process, 3)
void fn_lp_grad_LT_LC_NoLog_MD_and_AD_InPlace_process(Eigen::Ref<Eigen::VectorXd> output,
        const Eigen::Ref<const Eigen::VectorXd> main_parameters, const Eigen::Ref<const Eigen::VectorXd> nuisance_parameters,
        const Eigen::Ref<const Eigen::MatrixXi> data, const std::string, const Model_fn_args_struct &) {
    fn_stub_evaluation(1, output, main_parameters, nuisance_parameters, data);
}
void fn_lp_and_grad_LC_LT_AD_log_scale_InPlace_process(Eigen::Ref<Eigen::VectorXd> output,
        const Eigen::Ref<const Eigen::VectorXd> main_parameters, const Eigen::Ref<const Eigen::VectorXd> nuisance_parameters,
        const Eigen::Ref<const Eigen::MatrixXi> data, const std::string, const Model_fn_args_struct &) {
    fn_stub_evaluation(3, output, main_parameters, nuisance_parameters, data);
}
', fallback_source, '\n', latent_trait_source, '\n', option_reader_source, '
// [[Rcpp::export]]
Rcpp::List fn_test_same_position_fallback(int test_failure_mode, bool enable_autodiff, std::string model_type) {
    failure_mode = test_failure_mode;
    attempted_evaluators.clear();
    Eigen::VectorXd output = Eigen::VectorXd::Constant(4, 42.0);
    Eigen::VectorXd main_parameters(1), nuisance_parameters(1);
    Eigen::MatrixXi data(1, 1);
    main_parameters << 1.25;
    nuisance_parameters << -0.5;
    data << 1;
    Model_fn_args_struct model_arguments;
    std::vector<LC_MVP_workspace_struct> workspaces;
    if (model_type == "latent_trait") {
        fn_lp_grad_LT_LC_multi_attempts_InPlace_process(output, main_parameters, nuisance_parameters, data,
                                                       "all", model_arguments, enable_autodiff);
    } else if (model_type == "MVOP" || model_type == "LC_MVOP") {
        fn_lp_grad_MVOP_multi_attempts_InPlace_process(output, main_parameters, nuisance_parameters, data,
                                                      "all", model_arguments, workspaces, 1, enable_autodiff);
    } else if (enable_autodiff) {
        fn_lp_grad_MVP_multi_attempts_InPlace_process(output, main_parameters, nuisance_parameters, data,
                                                     "all", model_arguments, workspaces, 1, true, model_type == "MVP");
    } else {
        fn_lp_grad_MVP_multi_attempts_InPlace_process(output, main_parameters, nuisance_parameters, data,
                                                     "all", model_arguments, workspaces, 1);
    }
    return Rcpp::List::create(Rcpp::Named("attempts") = attempted_evaluators, Rcpp::Named("output") = output);
}
// [[Rcpp::export]]
bool fn_test_fallback_option_reader(Rcpp::List model_arguments) {
    return fn_read_autodiff_fallback_option(model_arguments);
}
'), verbose = FALSE)
##
fn_run_fallback_unit_checks <-  function() {

        previous_options <-  options(BayesMVP_autodiff_fallback = NULL)
        on.exit(expr = options(previous_options), add = TRUE)
        ##
        stopifnot(!fn_test_fallback_option_reader(model_arguments = list()))
        options(BayesMVP_autodiff_fallback = TRUE)
        stopifnot(fn_test_fallback_option_reader(model_arguments = list()))
        stopifnot(!fn_test_fallback_option_reader(model_arguments = list(autodiff_fallback = FALSE)))
        ##
        for (invalid_option in list(NA, numeric(), 1, "TRUE", c(TRUE, FALSE))) {
            options(BayesMVP_autodiff_fallback = invalid_option)
            option_error <-  tryCatch(expr = fn_test_fallback_option_reader(model_arguments = list()),
                                      error = function(condition) condition)
            stopifnot(inherits(x = option_error, what = "error"))
        }
        ##
        for (model_type in c("LC_MVP", "MVP", "LC_MVOP", "MVOP", "latent_trait")) {
        for (enable_autodiff in c(FALSE, TRUE)) {
            for (test_failure_mode in 0:4) {
                test_result <-  fn_test_same_position_fallback( test_failure_mode = test_failure_mode,
                                                               enable_autodiff = enable_autodiff,
                                                               model_type = model_type)
                expected_attempts <-  seq_len(length.out = if (test_failure_mode == 0) 1 else if (test_failure_mode == 1 || !enable_autodiff) 2 else 3)
                if (model_type == "latent_trait") expected_attempts <-  if (test_failure_mode == 0 || !enable_autodiff) 1 else c(1, 3)
                stopifnot(identical(as.numeric(test_result$attempts), as.numeric(expected_attempts)))
                ##
                expected_failure <-  if (model_type == "latent_trait") (test_failure_mode >= 1 && !enable_autodiff) || test_failure_mode >= 3 else
                                     test_failure_mode >= 2 && (!enable_autodiff || test_failure_mode >= 3)
                if (expected_failure) {
                    stopifnot(is.nan(test_result$output[1]), all(test_result$output[-1] == 42))
                } else {
                    stopifnot(all(test_result$output == tail(x = expected_attempts, n = 1)))
                }
            }
        }
        }
        ##
        return(invisible(TRUE))

}
##
fn_run_fallback_unit_checks()
##
## ---- Validate and exercise the actual R option snapshot, including the model and multi_attempts gates ----------------------------------
##
sampler_source <-  readLines(con = file.path(package_directory, "R", "R_fn_sample.R"))
invisible(parse(text = sampler_source))
invisible(tools::parse_Rd(file = file.path(package_directory, "man", "R_fn_sample_model.Rd")))
snapshot_start <-  grep(pattern = "autodiff_fallback <-  getOption", x = sampler_source, fixed = TRUE)
snapshot_end <-  grep(pattern = "force_autodiff_for_metric <-", x = sampler_source, fixed = TRUE) - 1
snapshot_expression <-  parse(text = sampler_source[seq.int(from = snapshot_start, to = snapshot_end)])
##
fn_test_R_snapshot <-  function() {

        previous_options <-  options(BayesMVP_autodiff_fallback = TRUE)
        on.exit(expr = options(previous_options), add = TRUE)
        ##
        for (model_type in c("LC_MVP", "MVP", "LC_MVOP", "MVOP", "latent_trait", "Stan")) {
            for (multi_attempts in c(TRUE, FALSE)) {
                snapshot_environment <-  list2env(x = list(init_object = list(Model_type = model_type), multi_attempts = multi_attempts))
                eval(expr = snapshot_expression, envir = snapshot_environment)
                stopifnot(identical(snapshot_environment$autodiff_fallback, model_type != "Stan" && multi_attempts))
            }
        }
        ##
        return(invisible(TRUE))

}
##
fn_test_R_snapshot()
cat("PASS: fallback order, unchanged positions, disabled default, option validation, frozen list override, failure restoration and R syntax. No MCMC run.\n")
