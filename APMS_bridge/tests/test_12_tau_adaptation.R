#### =====================================================================================================================================
## test_12_tau_adaptation.R - deterministic algebra and actual R scheduling code; no MCMC or package installation
## =====================================================================================================================================
##
test_script_arguments <- grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <- dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
package_R_directory <- normalizePath(path = file.path(tests_directory, "..", "..", "inst", "BayesMVP", "R"))
source(file = file.path(package_R_directory, "R_fn_update_tau_ADAM.R"))
source(file = file.path(package_R_directory, "R_fns_ChESSR_HMC.R"))
is_NaN_or_Inf_vec <- function(x) any(!is.finite(x))
##
## ---- Extract actual scheduling blocks, without evaluating the sampler ---------------------------------------------------------------
##
fn_find_tau_expressions <- function(expression, predicate) {
    matches <- list()
    if (is.call(expression) && predicate(expression)) matches <- list(expression)
    if (is.call(expression) || is.expression(expression)) {
        for (child in as.list(expression)) {
            if (missing(child)) next
            if (is.call(child) || is.expression(child)) matches <- c(matches, fn_find_tau_expressions(expression = child, predicate = predicate))
        }
    }
    matches
}
burnin_expressions <- parse(file = file.path(package_R_directory, "R_fn_init_and_run_burnin_CHESS.R"))
ramp_block <- fn_find_tau_expressions(expression = burnin_expressions, predicate = function(expression) {
    identical(expression[[1]], as.name("if")) && identical(expression[[2]], quote(!isTRUE(manual_tau) && ii >= clip_iter && ii < gap))
})
handover_block <- fn_find_tau_expressions(expression = burnin_expressions, predicate = function(expression) {
    identical(expression[[1]], as.name("if")) && identical(expression[[2]], quote(ii == gap))
})
stopifnot(length(ramp_block) == 1, length(handover_block) == 1)
if_not_NA_or_INF_else <- function(x, fallback) if (length(x) == 1 && is.finite(x)) x else fallback
##
## ---- Both ramp shapes cover their window and respect tau_initial, including below pi ------------------------------------------------
##
for (tau_ramp in c("original", "staged")) {
    for (tau_initial in c(0.01, pi / 2, pi, 2 * pi)) {
        for (n_burnin in c(125, 250)) {
            clip_iter <- if (n_burnin == 125) 20 else 50
            gap <- if (n_burnin == 125) 50 else 125
            for (partitioned_HMC in c(FALSE, TRUE)) {
                manual_tau <- FALSE
                EHMC_args_as_Rcpp_List <- list(tau_main = 0.01, tau_us = 0.02, eps_main = 0.01, eps_us = 0.02)
                for (ii in seq.int(from = clip_iter, to = gap - 1)) {
                    eval(expr = ramp_block[[1]])
                    stopifnot(EHMC_args_as_Rcpp_List$tau_main > 0, EHMC_args_as_Rcpp_List$tau_main <= tau_initial,
                              EHMC_args_as_Rcpp_List$tau_us > 0, EHMC_args_as_Rcpp_List$tau_us <= tau_initial)
                    if (ii == clip_iter) stopifnot(EHMC_args_as_Rcpp_List$tau_main == min(tau_initial, 0.05))
                }
                stopifnot(EHMC_args_as_Rcpp_List$tau_main == tau_initial, EHMC_args_as_Rcpp_List$tau_us == tau_initial)
                EHMC_burnin_as_Rcpp_List <- list(tau_m_adam_main = 99, tau_v_adam_main = 99, tau_m_adam_us = 99, tau_v_adam_us = 99)
                for (ii in seq.int(from = gap, to = gap + 12)) {
                    suppressMessages(eval(expr = handover_block[[1]]))
                    if (ii == gap) stopifnot(EHMC_args_as_Rcpp_List$tau_main == tau_initial,
                                             all(unlist(EHMC_burnin_as_Rcpp_List) == 0))
                    else stopifnot(EHMC_args_as_Rcpp_List$tau_main == tau_initial / 3)
                    EHMC_args_as_Rcpp_List$tau_main <- tau_initial / 3
                }
                manual_tau <- TRUE
                ii <- clip_iter
                EHMC_args_as_Rcpp_List$tau_main <- 0.123
                eval(expr = ramp_block[[1]])
                stopifnot(EHMC_args_as_Rcpp_List$tau_main == 0.123)
            }
        }
    }
}
cat("PASS: both ramps, pi/2/pi/2pi and small tau, both blocks, 125/250 schedules, one reset, manual ramp bypass.\n")
##
## ---- ADAM receives its own iteration count and uses the full requested LR on the first update ----------------------------------------
##
adam_calls <- fn_find_tau_expressions(expression = burnin_expressions, predicate = function(expression) {
    identical(expression[[1]], as.name("R_fn_update_tau_using_ADAM"))
})
stopifnot(length(adam_calls) == 3)
for (adam_call in adam_calls) {
    stopifnot(identical(adam_call$ii, as.name("tau_adaptation_iteration")),
              identical(adam_call$n_burnin, as.name("n_tau_adaptation_iterations")),
              identical(adam_call$aggregation, "weighted_mean"))
}
adam_arguments <- list(debug = FALSE, n_chains = 2, noisy_grads_prop_per_chain = c(-1, -1), valid_chains = 2,
                      tau = pi, LR = 0.075, ii = 1, n_burnin = 63, tau_m_adam = 0, tau_v_adam = 0,
                      beta1_adam = 0, beta2_adam = 0.95, eps_adam = 1e-8, aggregation = "weighted_mean")
adam_result <- do.call(what = R_fn_update_tau_using_ADAM, args = adam_arguments)
stopifnot(abs(log(adam_result[1] / pi) + 0.075) < 1e-7)
adam_arguments$noisy_grads_prop_per_chain <- c(NA_real_, Inf)
adam_arguments$tau_m_adam <- 0.3
adam_arguments$tau_v_adam <- 0.4
stopifnot(identical(do.call(what = R_fn_update_tau_using_ADAM, args = adam_arguments), c(pi, 0.3, 0.4)))
adam_arguments$noisy_grads_prop_per_chain <- c(1, -1)
adam_arguments$weights_per_chain <- c(0, 0)
stopifnot(identical(do.call(what = R_fn_update_tau_using_ADAM, args = adam_arguments), c(pi, 0.3, 0.4)))
cat("PASS: local tau clock, mean aggregation, first-step LR, no momentum drift with missing/zero-weight evidence.\n")
##
## ---- Finite differences against exact Gaussian dynamics, with diagonal and non-diagonal mass -----------------------------------------
##
initial_position <- c(0.6, 1.3)
initial_velocity <- c(1.1, -0.3)
for (metric_shape in c("diag", "dense")) {
    mass_matrix <- if (metric_shape == "diag") c(2, 5) else matrix(c(2, 0.4, 0.4, 1.5), nrow = 2)
    mass_dense <- if (metric_shape == "diag") diag(mass_matrix) else mass_matrix
    fn_exact_KE_criterion <- function(log_tau) {
        trajectory_length <- exp(log_tau)
        endpoint_velocity <- initial_velocity * cos(trajectory_length) - initial_position * sin(trajectory_length)
        kinetic_change <- 0.5 * sum(endpoint_velocity * c(mass_dense %*% endpoint_velocity)) -
                          0.5 * sum(initial_velocity * c(mass_dense %*% initial_velocity))
        0.5 * kinetic_change^2
    }
    for (tau_ii in c(0.1, 0.8, 2.5)) {
        endpoint_position <- initial_position * cos(tau_ii) + initial_velocity * sin(tau_ii)
        endpoint_velocity <- initial_velocity * cos(tau_ii) - initial_position * sin(tau_ii)
        endpoint_force <- -c(mass_dense %*% endpoint_position)
        exact_result <- R_fn_compute_gradients_for_tau_using_KE(velocity_initial = initial_velocity,
                                                               velocity_proposed = endpoint_velocity,
                                                               metric_shape = metric_shape,
                                                               mass_matrix = mass_matrix,
                                                               kinetic_energy_rate_proposed = sum(endpoint_velocity * endpoint_force),
                                                               tau_ii = tau_ii,
                                                               return_criterion = TRUE)
        numerical_gradient <- (fn_exact_KE_criterion(log_tau = log(tau_ii) + 1e-6) -
                               fn_exact_KE_criterion(log_tau = log(tau_ii) - 1e-6)) / 2e-6
        stopifnot(abs(exact_result$gradient - numerical_gradient) < 1e-7,
                  abs(exact_result$criterion - fn_exact_KE_criterion(log_tau = log(tau_ii))) < 1e-12)
        rejected_gradient <- R_fn_compute_gradients_for_tau_using_KE(velocity_initial = initial_velocity,
                                                                    velocity_proposed = initial_velocity,
                                                                    metric_shape = metric_shape,
                                                                    mass_matrix = mass_matrix,
                                                                    kinetic_energy_rate_proposed = 123,
                                                                    tau_ii = tau_ii)
        stopifnot(rejected_gradient == 0)
    }
}
cat("PASS: squared KE value and log-tau derivative match finite differences; rejected endpoints contribute zero.\n")
##
## ---- At Gaussian stationarity, signed energy has zero mean but squared KE has the expected nonzero gradient --------------------------
##
quadrature_nodes <- c(-sqrt(3), 0, sqrt(3))
quadrature_weights <- c(1 / 6, 2 / 3, 1 / 6)
tau_ii <- 0.8
mean_signed_energy_change <- mean_KE_criterion <- mean_KE_gradient <- 0
for (position_index in seq_along(quadrature_nodes)) {
    for (velocity_index in seq_along(quadrature_nodes)) {
        initial_position <- quadrature_nodes[position_index]
        initial_velocity <- quadrature_nodes[velocity_index]
        endpoint_position <- initial_position * cos(tau_ii) + initial_velocity * sin(tau_ii)
        endpoint_velocity <- initial_velocity * cos(tau_ii) - initial_position * sin(tau_ii)
        joint_weight <- quadrature_weights[position_index] * quadrature_weights[velocity_index]
        objective_result <- R_fn_compute_gradients_for_tau_using_KE(velocity_initial = initial_velocity,
                                                                   velocity_proposed = endpoint_velocity,
                                                                   metric_shape = "diag",
                                                                   mass_matrix = 1,
                                                                   kinetic_energy_rate_proposed = -endpoint_velocity * endpoint_position,
                                                                   tau_ii = tau_ii,
                                                                   return_criterion = TRUE)
        mean_signed_energy_change <- mean_signed_energy_change + joint_weight * 0.5 * (endpoint_velocity^2 - initial_velocity^2)
        mean_KE_criterion <- mean_KE_criterion + joint_weight * objective_result$criterion
        mean_KE_gradient <- mean_KE_gradient + joint_weight * objective_result$gradient
    }
}
stopifnot(abs(mean_signed_energy_change) < 1e-12,
          abs(mean_KE_criterion - 0.5 * sin(tau_ii)^2) < 1e-12,
          abs(mean_KE_gradient - tau_ii * sin(tau_ii) * cos(tau_ii)) < 1e-12)
cat("PASS: squared KE has a nonzero, correct stationary gradient where the old signed-energy signal averages to zero.\n")
##
## ---- Cost-aware normalisation: stationary criterion does not imply zero cost gradient -------------------------------------------------
##
normalised_result <- fn_normalise_ChEES_per_tau(gradients = c(4, 4), criteria = c(2, 2), criterion_ema = 2)
stopifnot(normalised_result$informative, normalised_result$gradient == 1)
normalised_result <- fn_normalise_ChEES_per_tau(gradients = c(0, 0), criteria = c(2, 2), criterion_ema = 2)
stopifnot(normalised_result$gradient == -1)
cat("PASS: ChEES-per-tau retains the cost derivative. No MCMC was run.\n")
