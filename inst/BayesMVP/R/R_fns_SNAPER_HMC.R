 








#' R_fn_compute_noisy_grad_diag_M_ADAM
#' @export
R_fn_compute_gradients_for_tau_using_SNAPER <- function(  debug = FALSE,
                                                          ##
                                                          eigen_vector, 
                                                          eigen_max,
                                                          ##
                                                          theta_vec_initial,
                                                          theta_vec_prop,
                                                          ##
                                                          snaper_m_vec,
                                                          ##
                                                          velocity_prop,
                                                          velocity_0,
                                                          ##
                                                          LR,
                                                          ii,
                                                          n_burnin,
                                                          ##
                                                          sqrt_M_vec,
                                                          M_inv_diag,
                                                          ##
                                                          tau_m_adam,
                                                          tau_v_adam,
                                                          tau_ii,
                                                          ##
                                                          beta1_adam,
                                                          beta2_adam,
                                                          eps_adam
) { 
  
        eta_w = 3.0;
        rho = 1.0;
        ##
        z <- eigen_vector
        m <- snaper_m_vec
        ##
        g_SNAPER <- 0.0
        ##
        if (ii > eta_w && eigen_max > 0) {
          
              x_0 <- theta_vec_initial
              x_tau <- theta_vec_prop
              ##
              v_0 <- velocity_0
              v_tau <- velocity_prop
              ##
              psi_x_0   <- (sum(z * sqrt_M_vec * (x_0 - m)))^2   # SCALAR
              psi_x_tau <- (sum(z * sqrt_M_vec * (x_tau - m)))^2 # SCALAR
              ##
              grad_psi_x_0   <- 2*sum(z*sqrt_M_vec*(x_0 - m))*sqrt_M_vec*z  # VECTOR
              grad_psi_x_tau <- 2*sum(z*sqrt_M_vec*(x_tau - m))*sqrt_M_vec*z  # VECTOR
              ##
              delta_x_0   <- 2*sum(grad_psi_x_0*M_inv_diag*(-v_0))*(psi_x_0 - psi_x_tau) # SCALAR
              delta_x_tau <- 2*sum(grad_psi_x_tau*M_inv_diag*(+v_tau))*(psi_x_tau - psi_x_0) # SCALAR
              ##
              g <- 0.5*(delta_x_tau + delta_x_0) # SCALAR
              ##
              g_SNAPER <- g - (1/tau_ii)*(psi_x_tau - psi_x_0)^2 # SCALAR
              ##
              # Clip gradient for stability
              g_k <- max(-100000000000, min(100000000000, g_SNAPER))
              
        }
        ##
        if (debug) {
          if (ii %% 50 == 0) {
            cat("\n=== Tau gradient components ===\n")
            cat("psi_x_0:", psi_x_0, "\n")
            cat("psi_x_tau:", psi_x_tau, "\n")
            cat("psi diff:", psi_x_tau - psi_x_0, "\n")
            cat("psi diff squared:", (psi_x_tau - psi_x_0)^2, "\n")
            cat("delta_x_0:", delta_x_0, "\n")
            cat("delta_x_tau:", delta_x_tau, "\n")
            cat("g:", g, "\n")
            cat("penalty term:", (1/tau_ii)*(psi_x_tau - psi_x_0)^2, "\n")
            cat("g_k (final gradient):", g_k, "\n")
          }
        }
        ##
        ##
        if (is_NaN_or_Inf_vec(g_k) == TRUE) {
          g_k <- 0.0
        }
        ##
        return(g_k)
  
}




 

 



#' After updating snaper_w but before computing eigen_max
#' fn_stabilize_snaper_w
#' @export
fn_stabilize_snaper_w <- function(snaper_w_vec, 
                                  ii) {
  
        # Remove NaNs first
        if (any(is.nan(snaper_w_vec))) {
          cat("NaN detected in snaper_w, resetting\n")
          snaper_w_vec[is.nan(snaper_w_vec)] <- 0.01
        }
        
        # # # Option 1: Normalize by iteration (reduces influence of early iterations)
        # if (ii > 10) {
        #   scale_factor <- sqrt(ii) / 10  # Grows slowly
        #   snaper_w_vec <- snaper_w_vec / scale_factor
        # }
        # # 
        # # Option 2: Clip extreme values
        # w_magnitude <- sqrt(sum(snaper_w_vec^2))
        # ##
        # if (w_magnitude > 10) {
        #   snaper_w_vec <- snaper_w_vec * (10 / w_magnitude)
        # }
        # 
        # # Option 3: Use more aggressive decay for large w
        # if (w_magnitude > 5) {
        #   beta_extra <- 0.9  # Extra decay
        #   snaper_w_vec <- snaper_w_vec * beta_extra
        # }
        
        return(snaper_w_vec)
  
}






 






