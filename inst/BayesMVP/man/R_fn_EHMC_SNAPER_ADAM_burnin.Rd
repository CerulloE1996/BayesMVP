% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/R_fn_EHMC_burnin.R
\name{R_fn_EHMC_SNAPER_ADAM_burnin}
\alias{R_fn_EHMC_SNAPER_ADAM_burnin}
\title{R_fn_EHMC_SNAPER_ADAM_burnin}
\usage{
R_fn_EHMC_SNAPER_ADAM_burnin(
  Model_type,
  vect_type,
  parallel_method,
  Stan_data_list,
  y,
  N,
  sample_nuisance,
  n_params_main,
  n_nuisance,
  n_chains_burnin,
  seed,
  manual_tau,
  tau_if_manual,
  n_burnin,
  LR_main,
  LR_us,
  n_adapt,
  partitioned_HMC,
  clip_iter,
  gap,
  metric_type_main,
  metric_shape_main,
  ape_main,
  metric_type_nuisance,
  metric_shape_nuisance,
  max_eps_main,
  max_eps_us,
  max_L,
  tau_mult,
  ratio_M_us,
  ratio_M_main,
  interval_width_main,
  interval_width_nuisance,
  force_autodiff,
  force_PartialLog,
  multi_attempts,
  theta_main_vectors_all_chains_input_from_R,
  theta_us_vectors_all_chains_input_from_R,
  n_nuisance_to_track,
  Model_args_as_Rcpp_List,
  EHMC_args_as_Rcpp_List,
  EHMC_Metric_as_Rcpp_List,
  EHMC_burnin_as_Rcpp_List
)
}
\description{
R_fn_EHMC_SNAPER_ADAM_burnin
}
\keyword{internal}
