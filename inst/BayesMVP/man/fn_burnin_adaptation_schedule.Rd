\name{fn_burnin_adaptation_schedule}
\alias{fn_burnin_adaptation_schedule}
\title{Resolve the Main Burn-in Adaptation Schedule}
\usage{
fn_burnin_adaptation_schedule(
  n_burnin, n_adapt = NULL, burnin_schedule = "automatic",
  metric_adaptation_end_iter = NULL, theta_hat_us_freeze_iter = NULL,
  theta_hat_us_rule = "running_mean_frozen", clip_iter = NULL,
  clip_iter_tau = NULL
)
}
\arguments{
\item{n_burnin}{Main burn-in length. Automatic mode requires at least five iterations.}
\item{n_adapt}{First iteration with epsilon and tau fixed. NULL derives it from the burn-in length.}
\item{burnin_schedule}{"automatic" for terminal epsilon/tau calibration, or "legacy" for the previous metric cutoff.}
\item{metric_adaptation_end_iter}{Last update to both metrics. NULL uses round(0.8 * n_adapt) in automatic mode.}
\item{theta_hat_us_freeze_iter}{Last running-mean centre update. NULL follows the metric cutoff in automatic mode.}
\item{theta_hat_us_rule}{"running_mean_frozen", "zero", or (legacy only) "running_mean".}
\item{clip_iter}{End of the initial ramp. NULL interpolates the existing 125/250/500/1000-iteration anchors in automatic mode.}
\item{clip_iter_tau}{Start of adaptive tau. NULL uses the same interpolation in automatic mode.}
}
\description{Derives and validates one-based inclusive phase boundaries. The default gives metric/centre endpoints 90 and 180,
epsilon/tau endpoints 112 and 224, and fixed phases 113--125 and 225--250 for burn-in lengths 125 and 250 respectively.}
\details{Metric and centre updates precede the transition; epsilon/tau updates follow it. A final metric update is requested at
the cutoff even if the usual interval does not divide it, provided the estimate is ready. The cutoff is a scheduling heuristic,
not an optimality guarantee. Very short schedules are validated but are not guaranteed to learn useful metrics. The ordering
pre-burn-in keeps its separate legacy schedule. Explicit ramp and centre overrides are validated and retained.}
\value{A list of resolved settings, phase_table, schedule_version and filename_token. The token encodes the first fixed
epsilon/tau iteration and final metric and centre updates, so changed boundaries identify different PS7 fits.}
\examples{
fn_burnin_adaptation_schedule(n_burnin = 125)$phase_table
fn_burnin_adaptation_schedule(n_burnin = 250)$phase_table
}
