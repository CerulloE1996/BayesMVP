#### =====================================================================================================================================
## test_13_KE_endpoint_capture.R - compile the actual HMCResult class and check proposal derivatives across rejection; no MCMC
## =====================================================================================================================================
##
{
    require(Rcpp)
    require(RcppEigen)
}
##
test_script_arguments <- grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <- dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
package_src_directory <- normalizePath(path = file.path(tests_directory, "..", "..", "inst", "BayesMVP", "src"))
class_source <- readLines(con = file.path(package_src_directory, "general_functions", "classes.hpp"))
class_start <- grep(pattern = "^class HMCResult \\{", x = class_source)
next_class_start <- grep(pattern = "^class HMC_output_single_chain \\{", x = class_source)
stopifnot(length(class_start) == 1, length(next_class_start) == 1)
class_source <- paste(class_source[seq.int(from = class_start, to = next_class_start - 1)], collapse = "\n")
##
## ---- Compile only the state container, using the source class verbatim -----------------------------------------------------------------
##
Rcpp::sourceCpp(code = paste0('
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp17)]]
', class_source, '
// [[Rcpp::export]]
bool fn_test_KE_endpoint_capture() {
    HMCResult result(2, 3, 1);
    result.main_velocity_vec_proposed() << 2.0, 3.0;
    result.us_velocity_vec_proposed() << 1.0, 2.0, 3.0;
    result.lp_and_grad_outs().segment(1, 3) << 4.0, 5.0, 6.0;
    result.lp_and_grad_outs().segment(4, 2) << 7.0, 8.0;
    result.lp_and_grad_outs_0().setConstant(100.0);
    result.record_kinetic_energy_rate_main();
    result.record_kinetic_energy_rate_us();
    if (result.kinetic_energy_rate_main_proposed != 38.0 || result.kinetic_energy_rate_us_proposed != 32.0) return false;
    result.reject_proposal_main();
    result.reject_proposal_us();
    return result.lp_and_grad_outs()(1) == 100.0 && result.kinetic_energy_rate_main_proposed == 38.0 &&
           result.kinetic_energy_rate_us_proposed == 32.0;
}
'), verbose = FALSE)
stopifnot(fn_test_KE_endpoint_capture())
cat("PASS: native endpoint dot products use the correct gradient segments and survive rejection/cache restoration. No MCMC run.\n")
