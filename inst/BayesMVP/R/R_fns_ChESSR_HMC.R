




#' Super simple tau adaptation based on jumping distance
#' R_fn_update_tau_simple
#' @keywords internal
#' @export
R_fn_update_tau_simple <- function( debug = FALSE,
                                    theta_vec_initial,
                                    theta_vec_prop,
                                    tau,
                                    LR,
                                    target_jump_dist = NULL,
                                    ii = 1
) {
  
          # Compute jumping distance
          jump_dist <- sqrt(sum((theta_vec_prop - theta_vec_initial)^2))
          
          # Set target based on dimension if not provided
          if (is.null(target_jump_dist)) {
            d <- length(theta_vec_initial)
            target_jump_dist <- sqrt(d) * 0.5  # heuristic
          }
          
          # Simple proportional control
          # If jumping too far, reduce tau; if too close, increase tau
          ratio <- jump_dist / target_jump_dist
          
          # Log-space update
          log_tau_new <- log(tau) + LR * (log(ratio))
          tau_new <- exp(log_tau_new)
          
          # Bound tau
          tau_new <- max(0.01, min(25, tau_new))
          
          # Debug
          if (debug) {
            if (ii %% 50 == 0) {
              cat("Jump dist:", jump_dist, "Target:", target_jump_dist, 
                  "Tau:", tau, "->", tau_new, "\n")
            }
          }
          
          return(tau_new)
  
}







## Legacy signed-energy helpers below are retained for old scripts, but are NOT used by tau_objective = "KE".
## The corrected KE objective uses R_fn_compute_gradients_for_tau_using_KE (below).
#' R_fn_compute_gradients_for_tau_using_ChEES
#' @keywords internal
#' @export
R_fn_compute_gradients_for_tau_using_ChEES <- function( debug = FALSE,
                                                        theta_vec_initial,
                                                        theta_vec_prop,
                                                        ##
                                                        metric_shape_main,
                                                        ##
                                                        # snaper_m_vec,
                                                        ##
                                                        # velocity_0,
                                                        # velocity_prop,
                                                        ##
                                                        velocity_main_0,
                                                        velocity_main_prop,
                                                        velocity_nuisance_0,
                                                        velocity_nuisance_prop,
                                                        ##
                                                        # LR,
                                                        ii,
                                                        ##
                                                        M_inv_diag_main,
                                                        M_inv_dense_main,
                                                        M_inv_diag_nuisance,
                                                        ##
                                                        # tau,  # This is the global tau (not used in gradient)
                                                        tau_ii      # THIS is the chain-specific tau to use!
) {
        
        # M_diag <- 1.0 / M_inv_diag
        # sigma_sq <- sum(M_inv_diag)  # This is σ²
        # 
        # # Trace of covariance
        # Centered positions (assuming you have mean estimate)
        # z_0 <- theta_vec_initial - snaper_m_vec  # or some mean estimate
        # z_prop <- theta_vec_prop - snaper_m_vec
        ##
        # The ChEES gradient is based on the change in kinetic energy
        # which serves as a proxy for the autocorrelation
        ##
        # Kinetic energies (using M_inv as the mass matrix)
        # K = 0.5 * v^T * M^(-1) * v
        # KE_initial <- 0.5 * sum((z_0^2) )
        # KE_final <- 0.5 * sum((z_prop^2) )
        ##
        if (metric_shape_main == "diag") {
              # KE_initial <- 0.5 * sum((velocity_main_0^2) * M_inv_diag)
              # KE_final   <- 0.5 * sum((velocity_main_0^2) * M_inv_diag)
              # ##
              # delta_K <- KE_final - KE_initial # Change in kinetic energy
              ##
              KE_main_initial <- 0.5 * sum((velocity_main_0^2) * M_inv_diag_main)
              KE_main_final   <- 0.5 * sum((velocity_main_prop^2) * M_inv_diag_main)
              ##
              KE_nuisance_initial <- 0.5 * sum((velocity_nuisance_0^2) * M_inv_diag_nuisance)
              KE_nuisance_final   <- 0.5 * sum((velocity_nuisance_prop^2) * M_inv_diag_nuisance)
              ##
              delta_E <- (KE_main_final - KE_main_initial) +  (KE_nuisance_final - KE_nuisance_initial) # Change in kinetic energy
        } else { 
              KE_main_initial <- 0.5 * sum((velocity_main_0^2) %*% M_inv_dense_main)
              KE_main_final   <- 0.5 * sum((velocity_main_prop^2) %*% M_inv_dense_main)
              ##
              KE_nuisance_initial <- 0.5 * sum((velocity_nuisance_0^2) * M_inv_diag_nuisance)
              KE_nuisance_final   <- 0.5 * sum((velocity_nuisance_prop^2) * M_inv_diag_nuisance)
              ##
              delta_E <- (KE_main_final - KE_main_initial) +  (KE_nuisance_final - KE_nuisance_initial) # Change in kinetic energy
        }
        
        # The ChEES gradient with respect to log(tau)
        # Based on BlackJAX: gradient aims to decorrelate kinetic energy
        g_ChEES <- delta_E / tau_ii
        
        # Clip gradient for stability
        g_k <- max(-100000000000, min(100000000000, g_ChEES))

        # if (debug) {
        #   if (ii %% 50 == 0) {
        #     cat("\n=== Tau gradient components ===\n")
        #     # cat("delta_norm_sq:", delta_norm_sq, "\n")
        #     cat("g_ChEES:", g_ChEES, "\n")
        #     cat("g_k (final gradient):", g_k, "\n")
        #   }
        # }
        ##
        
        # Return just the gradient (ADAM is done globally)
        if (is_NaN_or_Inf_vec(g_k) == TRUE) {
          g_k <- 0.0
        }
        ##
        return(g_k)

}



 
#' R_fn_compute_gradients_for_tau_using_ChEES_combined
#' @keywords internal
#' @export
R_fn_compute_gradients_for_tau_using_ChEES_combined <- function( debug = FALSE,
                                                                ## Main params (standard HMC):
                                                                velocity_main_0,
                                                                velocity_main_prop,
                                                                metric_shape_main,
                                                                M_inv_diag_main,
                                                                M_inv_dense_main,
                                                                ## Nuisance params (diffusion-pathspace HMC):
                                                                velocity_nuisance_0,
                                                                velocity_nuisance_prop,
                                                                theta_nuisance_initial,
                                                                theta_nuisance_prop,
                                                                theta_hat_nuisance_vec,
                                                                M_inv_diag_nuisance,
                                                                ##
                                                                # LR,
                                                                ii,
                                                                tau_ii
) {
  
  # --- Main params: kinetic only ---
  if (metric_shape_main == "diag") {
        KE_main_initial <- 0.5 * sum((velocity_main_0^2) * M_inv_diag_main)
        KE_main_final   <- 0.5 * sum((velocity_main_prop^2) * M_inv_diag_main)
  } else { 
        KE_main_initial <- 0.5 * sum((velocity_main_0^2) %*% M_inv_dense_main)
        KE_main_final   <- 0.5 * sum((velocity_main_prop^2) %*% M_inv_dense_main)
  }
  
  # --- Nuisance params: kinetic + spring ---
  KE_nuis_initial <- 0.5 * sum((velocity_nuisance_0^2) * M_inv_diag_nuisance)
  KE_nuis_final   <- 0.5 * sum((velocity_nuisance_prop^2) * M_inv_diag_nuisance)
  
  M_diag_nuisance <- 1.0 / M_inv_diag_nuisance
  SE_nuis_initial <- 0.5 * sum(((theta_nuisance_initial - theta_hat_nuisance_vec)^2) * M_diag_nuisance)
  SE_nuis_final   <- 0.5 * sum(((theta_nuisance_prop - theta_hat_nuisance_vec)^2) * M_diag_nuisance)
  
  # # ## --- Total non-potential energy change ---
  # delta_E <- (KE_main_final - KE_main_initial) +
  #   (KE_nuis_final - KE_nuis_initial) +
  #   (SE_nuis_final - SE_nuis_initial)
  
  # # # --- Total non-potential energy change ---
  delta_E <- (KE_main_final - KE_main_initial)
  
  g_ChEES <- delta_E / tau_ii
  
  g_k <- max(-100000000000, min(100000000000, g_ChEES))
  
  # if (debug) {
  #   if (ii %% 50 == 0) {
  #     cat("\n=== Tau gradient components (combined) ===\n")
  #     cat("delta_KE_main:", KE_main_final - KE_main_initial, "\n")
  #     cat("delta_KE_nuis:", KE_nuis_final - KE_nuis_initial, "\n")
  #     cat("delta_SE_nuis:", SE_nuis_final - SE_nuis_initial, "\n")
  #     cat("delta_E (total):", delta_E, "\n")
  #     cat("g_k:", g_k, "\n")
  #   }
  # }
  
  if (is_NaN_or_Inf_vec(g_k) == TRUE) {
    g_k <- 0.0
  }
  
  return(g_k)

}









##
## ---- Squared kinetic-energy change, on the block whose trajectory length is being adapted ----------------------------------------
##
## C = 0.5 * (K_proposed - K_initial)^2, K(v) = 0.5 * v' M v because these are VELOCITIES, not momenta.
## Along exact Hamiltonian dynamics, dK/dt = v' grad(log pi), including diffusion dynamics.
## Therefore dC/dlog(tau) = tau_ii * (K_proposed - K_initial) * dK/dt at the proposal.
## The native sampler records this scalar BEFORE rejection restores its cached gradient.
## Joint HMC scores the main block; partitioned HMC scores each updated block separately.
## As with ChEES, this is an endpoint/pathwise surrogate at finite epsilon: rounding L and differentiating
## acceptance are not included. It is an energy-decorrelation criterion, not a direct ESS/sec objective.
##
R_fn_compute_gradients_for_tau_using_KE <- function( velocity_initial,
                                                     velocity_proposed,
                                                     metric_shape,
                                                     mass_matrix,
                                                     kinetic_energy_rate_proposed,
                                                     tau_ii,
                                                     return_criterion = FALSE
) {

        if (length(velocity_initial) != length(velocity_proposed)) stop("KE velocities must have the same dimension.")
        if (identical(metric_shape, "diag")) {
            if (length(mass_matrix) != length(velocity_initial)) stop("KE diagonal mass must match the velocity dimension.")
            kinetic_energy_initial <- 0.5 * sum(velocity_initial^2 * c(mass_matrix))
            kinetic_energy_proposed <- 0.5 * sum(velocity_proposed^2 * c(mass_matrix))
        } else {
            kinetic_energy_initial <- 0.5 * sum(velocity_initial * c(mass_matrix %*% velocity_initial))
            kinetic_energy_proposed <- 0.5 * sum(velocity_proposed * c(mass_matrix %*% velocity_proposed))
        }
        ##
        kinetic_energy_change <- kinetic_energy_proposed - kinetic_energy_initial
        gradient <- tau_ii * kinetic_energy_change * kinetic_energy_rate_proposed
        criterion <- 0.5 * kinetic_energy_change^2
        if (length(gradient) != 1 || !is.finite(gradient) || !is.finite(criterion)) gradient <- NA_real_
        if (isTRUE(return_criterion)) return(list(gradient = gradient, criterion = criterion))
        return(gradient)

}

#' R_fn_compute_gradients_for_tau_using_ChEES_position
#'
#' The ChEES criterion of Hoffman, Radul and Sountsov (2021), on the MAIN block only.
#'
#' ---- Why this exists, alongside the kinetic-energy objective above ----
#'
#' The objectives above score a trajectory by its change in kinetic energy,
#' delta_E = KE(v_prop) - KE(v_0), and that statistic carries no information about tau once the
#' chains are stationary. Kinetic energy is a function of the state alone and is invariant to the
#' momentum flip, so writing Phi for the involution "L leapfrog steps, then flip" and alpha for the
#' Metropolis acceptance probability, detailed balance gives
#'
#'     E[ alpha(z) h(Phi z) ]  =  E[ alpha(z) h(z) ]        for ANY function h
#'
#' and therefore E[alpha * delta_E] = 0 exactly, for every tau and every eps. A LINEAR difference of
#' a state function cannot have an optimum in tau. The one-dimensional standard normal makes it
#' concrete: delta_E = 0.5*sin(t)^2*(theta_0^2 - v_0^2) - v_0*theta_0*sin(t)*cos(t), whose
#' expectation is 0.5*sin(t)^2*(E[theta_0^2] - 1) -> 0 at stationarity.
#'
#' ChEES escapes this because the criterion is a SQUARE, not a difference:
#'
#'     ChEES(t) = 0.25 * E[ alpha * ( ||x_prop - m||^2 - ||x_0 - m||^2 )^2 ]
#'
#' Detailed balance maps that expression to itself rather than to zero, so it is a genuine
#' objective with an interior maximum. Differentiating under the integral and using
#' d/dt ||x(t) - m||^2 = 2 (x(t) - m) . xdot(t) gives the gradient estimated here:
#'
#'     dChEES/dt = E[ alpha * ( ||x_prop - m||^2 - ||x_0 - m||^2 ) * ( (x_prop - m) . xdot_prop ) ]
#'
#' ---- Coordinates ----
#'
#' Everything is evaluated in the METRIC-STANDARDISED coordinates x = M^(1/2) theta, in which the
#' target is roughly isotropic and the velocity is standard normal (the sampler draws
#' v ~ N(0, M_inv), so xdot = M^(1/2) v ~ N(0, I)). Without that standardisation the criterion is
#' dominated by whichever main coordinate happens to have the largest raw scale. Concretely:
#'
#'     ||x - m_x||^2      = (theta - m)' M (theta - m)
#'     (x - m_x) . xdot   = (theta - m)' M v
#'
#' so the caller passes M itself (M_main_vec for a diagonal metric, M_dense_main for a dense one),
#' NOT M_inv. Note that the kinetic-energy objectives above weight by M_inv where the C++ Metropolis
#' step weights by M; that inconsistency is not reproduced here.
#'
#' ---- Jitter and the log-tau chain rule ----
#'
#' The sampler draws tau_ii ~ Uniform(0, 2*tau) per trajectory, so tau_ii = 2*u*tau and
#' d tau_ii / d log(tau) = tau_ii. The gradient with respect to log(tau) - which is what
#' R_fn_update_tau_using_ADAM ascends - therefore carries a factor of tau_ii:
#'
#'     g = tau_ii * ( ||x_prop - m||^2 - ||x_0 - m||^2 ) * ( (x_prop - m) . xdot_prop )
#'
#' The acceptance weighting is applied by the CALLER, either as the accept indicator (which is what
#' use_proposed = FALSE already supplies, since a rejected chain has theta_prop reset to theta_0 and
#' contributes exactly zero) or as p_jump. Both are unbiased for the alpha-weighted expectation;
#' p_jump has the lower variance.
#'
#' @keywords internal
#' @export
R_fn_compute_gradients_for_tau_using_ChEES_position <- function( debug = FALSE,
                                                                 ##
                                                                 theta_main_0,
                                                                 theta_main_prop,
                                                                 velocity_main_prop,
                                                                 ##
                                                                 snaper_m_vec_main,
                                                                 ##
                                                                 metric_shape_main,
                                                                 ##
                                                                 ## M itself (NOT M_inv): a vector when metric_shape_main == "diag",
                                                                 ## a matrix when it is "dense".
                                                                 M_main_for_ChEES,
                                                                 ##
                                                                 ii,
                                                                 tau_ii,
                                                                 return_criterion = FALSE
) {

        ## centre on the running posterior mean of the main block (SNAPER's Welford mean):
        x_0_centred    <- c(theta_main_0)    - c(snaper_m_vec_main)
        x_prop_centred <- c(theta_main_prop) - c(snaper_m_vec_main)
        v_prop         <- c(velocity_main_prop)
        ##
        if (metric_shape_main == "diag") {
              ##
              M_vec <- c(M_main_for_ChEES)
              ##
              d_sq_0    <- sum((x_0_centred^2)    * M_vec)
              d_sq_prop <- sum((x_prop_centred^2) * M_vec)
              ##
              dot_prop  <- sum(x_prop_centred * v_prop * M_vec)
              ##
        } else {
              ##
              M_mat <- M_main_for_ChEES
              ##
              d_sq_0    <- sum(x_0_centred    * c(M_mat %*% x_0_centred))
              d_sq_prop <- sum(x_prop_centred * c(M_mat %*% x_prop_centred))
              ##
              dot_prop  <- sum(x_prop_centred * c(M_mat %*% v_prop))
              ##
        }
        ##
        ## ChEES gradient w.r.t. log(tau); the tau_ii factor is the jitter chain rule, NOT a
        ## normalisation, so it MULTIPLIES here where the kinetic-energy objective divides:
        delta_d_sq <- d_sq_prop - d_sq_0
        ##
        g_ChEES <- tau_ii * delta_d_sq * dot_prop
        ##
        g_k <- max(-100000000000, min(100000000000, g_ChEES))
        ##
        if (debug) {
          if (ii %% 50 == 0) {
            cat("\n=== ChEES (position) gradient components ===\n")
            cat("d_sq_0:", d_sq_0, " d_sq_prop:", d_sq_prop, "\n")
            cat("delta_d_sq:", delta_d_sq, " dot_prop:", dot_prop, "\n")
            cat("tau_ii:", tau_ii, " g_k:", g_k, "\n")
          }
        }
        ##
        if (is_NaN_or_Inf_vec(g_k) == TRUE) {
          g_k <- 0.0
        }
        ##
        ## Normalise the cost-aware arm BEFORE any clipping, so the ratio is scale-invariant.
        ## Its caller excludes non-finite gradient/criterion pairs together.
        if (isTRUE(x = return_criterion)) return(list(gradient = g_ChEES, criterion = 0.25 * delta_d_sq^2))
        return(g_k)

}









##
## ---- Cost-aware ChEES: gradient of log(E[ChEES]) - log(tau) -------------------------------------------------------------------
##
## Mean aggregation is required here, not the historical median. Apply the SAME chain weights
## to the criterion and its gradient. The acceptance derivative is ignored, as in the existing
## ChEES gradient estimator; this is an experimental adaptation arm, not an exact finite-epsilon derivative.
## The running criterion uses an EMA with decay 0.9. No positive finite signal means no ADAM update.
##
fn_normalise_ChEES_per_tau <- function( gradients,
                                        criteria,
                                        criterion_ema = NA_real_,
                                        weights = NULL,
                                        criterion_ema_decay = 0.9
) {
  
      if (length(x = gradients) != length(x = criteria)) stop("ChEES gradients and criteria must be aligned by chain.")
      if (length(x = criterion_ema_decay) != 1 || !is.finite(x = criterion_ema_decay) ||
          criterion_ema_decay < 0 || criterion_ema_decay >= 1) stop("criterion_ema_decay must be in [0, 1).")
      if (is.null(x = weights)) weights <- rep(x = 1.0, times = length(x = gradients))
      if (length(x = weights) != length(x = gradients)) stop("ChEES weights must be aligned by chain.")
      ##
      valid <- is.finite(x = gradients) & is.finite(x = criteria) & criteria >= 0 &
               is.finite(x = weights) & weights >= 0
      unchanged <- list(gradient = NA_real_, criterion_ema = criterion_ema, informative = FALSE)
      if (!any(valid)) return(unchanged)
      ## Estimate E[alpha * criterion], not the criterion conditional on acceptance.
      ## In particular, zero-acceptance chains contribute zero but remain in the mean denominator.
      ## Normalising by sum(alpha) would change the EMA's scale as acceptance changes.
      weights_used <- weights[valid] / sum(valid)
      criterion_mean <- sum(weights_used * criteria[valid])
      gradient_mean <- sum(weights_used * gradients[valid])
      if (!is.finite(x = criterion_mean) || criterion_mean <= 0 || !is.finite(x = gradient_mean)) return(unchanged)
      ##
      criterion_updated <- if (is.finite(x = criterion_ema) && criterion_ema > 0)
          criterion_ema_decay * criterion_ema + (1.0 - criterion_ema_decay) * criterion_mean else criterion_mean
      gradient <- gradient_mean / criterion_updated - 1.0
      ##
      if (!is.finite(x = gradient)) return(unchanged)
      ##
      return(list(gradient = gradient,
                  criterion_ema = criterion_updated,
                  informative = TRUE))
    
}















