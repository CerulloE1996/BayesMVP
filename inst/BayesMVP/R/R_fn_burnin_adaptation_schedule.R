#### =====================================================================================================================================
## R_fn_burnin_adaptation_schedule.R
##
## ---- Resolve MAIN burn-in phases in one place; the ordering pre-burn-in keeps its own schedule ----------------------------------------
##
#' Resolve the burn-in adaptation schedule
#'
#' The automatic schedule freezes both metrics and the nuisance centre before a
#' terminal epsilon/tau calibration phase. Iterations are one-based and end
#' points are inclusive. n_adapt is the FIRST iteration with epsilon/tau fixed.
#' Metric and centre updates occur before a transition; epsilon/tau updates occur
#' after it. A final metric update is requested at the boundary even when it is
#' not a multiple of the usual update interval (provided the estimate is ready).
#'
#' @param n_burnin Number of main burn-in iterations, at least 5 in automatic mode.
#' @param n_adapt First iteration with epsilon/tau fixed. NULL uses
#'   n_burnin - round(n_burnin / 10), with at least one fixed iteration.
#' @param burnin_schedule "automatic" or "legacy". Legacy retains the previous
#'   metric cutoff and independently frozen centre.
#' @param metric_adaptation_end_iter Last permitted update of BOTH metrics.
#'   NULL uses round(0.8 * n_adapt) in automatic mode.
#' @param theta_hat_us_freeze_iter Last centre update. NULL uses
#'   min(metric_adaptation_end_iter, round(0.7 * n_adapt)) in automatic
#'   mode, or round(0.6 * n_adapt) in legacy mode.
#' @param theta_hat_us_rule Centre rule. Automatic mode requires
#'   "running_mean_frozen" or "zero"; use legacy for an unfrozen running mean.
#' @param clip_iter End of the initial epsilon-only ramp. NULL interpolates the
#'   existing 125/250/500/1000-iteration reference schedules in automatic mode.
#' @param clip_iter_tau Start of adaptive tau. NULL uses the same interpolation.
#' @return Named list of resolved iteration boundaries, a phase table and a
#'   filename token recording the schedule version and effective boundaries.
#' @export
fn_burnin_adaptation_schedule <-  function( n_burnin,
                                            n_adapt = NULL,
                                            burnin_schedule = "automatic",
                                            metric_adaptation_end_iter = NULL,
                                            theta_hat_us_freeze_iter = NULL,
                                            theta_hat_us_rule = "running_mean_frozen",
                                            clip_iter = NULL,
                                            clip_iter_tau = NULL
) {
  
        burnin_schedule <-  match.arg(arg = burnin_schedule, choices = c("automatic", "legacy"))
        theta_hat_us_rule <-  match.arg(arg = theta_hat_us_rule, choices = c("running_mean_frozen", "zero", "running_mean"))
        ##
        fn_check_iteration <-  function(value, 
                                        name, 
                                        minimum, 
                                        maximum) {
          
                if (!is.numeric(x = value) || length(x = value) != 1 || !is.finite(x = value) ||
                    value != floor(x = value) || value < minimum || value > maximum) {
                    stop(name, " must be one integer between ", minimum, " and ", maximum, ".")
                }
                ##
                return(as.integer(x = value))
                
        }
        ##
        n_burnin <-  fn_check_iteration( value = n_burnin, 
                                         name = "n_burnin",
                                         minimum = if (burnin_schedule == "automatic") 5 else 2,
                                         maximum = .Machine$integer.max)
        ##
        if (is.null(x = n_adapt)) n_adapt <-  n_burnin - max(1, round(x = n_burnin / 10))
        n_adapt <-  fn_check_iteration( value = n_adapt, 
                                        name = "n_adapt",
                                        minimum = if (burnin_schedule == "automatic") 4 else 2,
                                        maximum = n_burnin)
        ##
        if (burnin_schedule == "automatic" && theta_hat_us_rule == "running_mean") {
            stop("The automatic schedule requires a frozen or zero centre. Use burnin_schedule = 'legacy' for 'running_mean'.")
        }
        ##
        if (is.null(x = metric_adaptation_end_iter)) {
            metric_adaptation_end_iter <-  if (burnin_schedule == "automatic") {
                max(2, min(round(x = 0.8 * n_adapt), n_adapt - 2))
            } else n_adapt - 1
        }
        ##
        metric_adaptation_end_iter <-  fn_check_iteration( value = metric_adaptation_end_iter,
                                                           name = "metric_adaptation_end_iter",
                                                           minimum = max(1, round(x = n_burnin / 10) + 1),
                                                           maximum = n_adapt - if (burnin_schedule == "automatic") 2 else 1)
        ##
        # if (is.null(x = theta_hat_us_freeze_iter)) {
        #     theta_hat_us_freeze_iter <-  if (burnin_schedule == "automatic") metric_adaptation_end_iter else round(x = 0.6 * n_adapt)
        # }
        if (is.null(x = theta_hat_us_freeze_iter)) {
          
              theta_hat_us_freeze_iter <-   if (burnin_schedule == "automatic") {
                                                min(metric_adaptation_end_iter, round(x = 0.7 * n_adapt))
                                            } else {
                                                round(x = 0.6 * n_adapt)
                                            }

        }
        ##
        theta_hat_us_freeze_iter <-  fn_check_iteration( value = theta_hat_us_freeze_iter,
                                                         name = "theta_hat_us_freeze_iter",
                                                         minimum = 1,
                                                         maximum = if (burnin_schedule == "automatic") n_adapt - 2 else n_burnin)
        ##
        ## ---- Ramp anchors reproduce PS7's existing values; other lengths interpolate or scale at the endpoints -------------------------
        ##
        if (burnin_schedule == "automatic") {
          
                reference_lengths <-  c(125, 250, 500, 1000)
                ##
                fn_scale_ramp_boundary <-  function(reference_iterations) {
                      
                        if (n_burnin < reference_lengths[1]) return(round(x = n_burnin * reference_iterations[1] / reference_lengths[1]))
                        if (n_burnin > tail(x = reference_lengths, n = 1)) {
                            return(round(x = n_burnin * tail(x = reference_iterations, n = 1) / tail(x = reference_lengths, n = 1)))
                        }
                        round(x = stats::approx(x = reference_lengths, y = reference_iterations, xout = n_burnin)$y)
                    
                }
                ##
                if (is.null(x = clip_iter)) clip_iter <-  max(1, min(fn_scale_ramp_boundary(reference_iterations = c(20, 50, 75, 150)),
                                                                   metric_adaptation_end_iter - 1,
                                                                   if (theta_hat_us_rule == "running_mean_frozen") theta_hat_us_freeze_iter - 1 else Inf))
                ##
                if (is.null(x = clip_iter_tau)) clip_iter_tau <-  max(clip_iter + 1, min(n_adapt - 1,
                                                                     fn_scale_ramp_boundary(reference_iterations = c(50, 125, 175, 350))))
                ##
                clip_iter <-  fn_check_iteration( value = clip_iter,
                                                  name = "clip_iter", 
                                                  minimum = 1, 
                                                  maximum = n_adapt - 2)
                ##
                clip_iter_tau <-  fn_check_iteration( value = clip_iter_tau, 
                                                      name = "clip_iter_tau", 
                                                      minimum = clip_iter + 1, 
                                                      maximum = n_adapt - 1)
                ##
                if (theta_hat_us_rule == "running_mean_frozen" && theta_hat_us_freeze_iter <= clip_iter) {
                    stop("theta_hat_us_freeze_iter must follow clip_iter so that a running-mean centre can be learned.")
                }
        }
        ##
        centre_adaptation_end_iter <-  switch(theta_hat_us_rule, 
                                              zero = 0, 
                                              running_mean = n_burnin,
                                              running_mean_frozen = theta_hat_us_freeze_iter)
        ##
        geometry_adaptation_end_iter <-  max(metric_adaptation_end_iter, centre_adaptation_end_iter)
        fn_iteration_range <-  function( first,
                                         last) {
          
                if (first > last) return("none")
                if (first == last) return(as.character(x = first))
                return(paste0(first, "-", last))
                
        }
        phase_table <-  data.frame(
            phase = c("Final update to centre", "Final update to both metrics", "Adapt epsilon and tau only", "Everything fixed"),
            iterations = c(as.character(x = centre_adaptation_end_iter),
                           as.character(x = metric_adaptation_end_iter),
                           fn_iteration_range(first = geometry_adaptation_end_iter + 1,
                                              last = n_adapt - 1),
                           fn_iteration_range(first = max(n_adapt, geometry_adaptation_end_iter + 1),
                                              last = n_burnin)))
        ##
        if (centre_adaptation_end_iter == metric_adaptation_end_iter) {
            phase_table <-  phase_table[-1, , drop = FALSE]
            phase_table$phase[1] <-  "Final update to centre and both metrics"
            rownames(x = phase_table) <-  NULL
        }
        ##
        ## Keep the token short: PS7 filenames are already long. a = first fixed epsilon/tau iteration;
        ## m = final metric update; c = final centre update. Version 1 also forces the final metric update.
        ##
        schedule_version <-  if (burnin_schedule == "automatic") 1 else 0
        filename_token <-  paste0("_bs", schedule_version, "a", n_adapt, "m", metric_adaptation_end_iter, "c", centre_adaptation_end_iter)
        ##
        return(list( burnin_schedule = burnin_schedule,
                     schedule_version = schedule_version,
                     n_burnin = n_burnin, 
                     n_adapt = n_adapt,
                     metric_adaptation_end_iter = metric_adaptation_end_iter,
                     theta_hat_us_freeze_iter = theta_hat_us_freeze_iter, 
                     centre_adaptation_end_iter = centre_adaptation_end_iter,
                     theta_hat_us_rule = theta_hat_us_rule, 
                     clip_iter = clip_iter,
                     clip_iter_tau = clip_iter_tau,
                     epsilon_tau_adaptation_end_iter = n_adapt - 1, 
                     fixed_kernel_start_iter = n_adapt,
                     phase_table = phase_table,
                     filename_token = filename_token))
        
}






















