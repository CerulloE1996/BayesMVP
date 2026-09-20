#### ================================================================================================================================================================
## settings_smoke.R  -  REDUCED, TEST-ONLY sampler settings.
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------
## These exist ONLY so the smoke tests can verify EXECUTION (initialisation,
## log-density/gradient calls, sampling loop, trace storage, reconstruction,
## summaries) in seconds. They are NOT the application settings and must never
## be imported by the APMS bridge (which keeps its own settings).
## ================================================================================================================================================================
##
fn_smoke_settings <- function() {
  
    list( seed = 123,
          n_chains_burnin   = 4,
          n_chains_sampling = 16,
          n_superchains = 16,
          n_burnin = 250,
          n_iter   = 500,
          ##
          adapt_delta   = 0.80,
          learning_rate = 0.025,
          max_L = 1024,
          parallel_method = "RcppParallel",
          n_threads_WCP_burnin   = 1,
          n_threads_WCP_sampling = 1,
          use_disk = FALSE,
          ##
          partitioned_HMC = FALSE,
          diffusion_HMC   = TRUE,
          sample_nuisance = TRUE,
          ##
          metric_type_main = "Empirical",
          metric_shape_main = "diag",
          ratio_M_main = 0.90,
          interval_width_main = 1,
          ##
          metric_type_nuisance = "Empirical",
          metric_shape_nuisance = "diag",
          ratio_M_nuisance = 0.90,
          interval_width_nuisance = 1,
          ##
          M_decay_type = "inverse",
          M_decay_power = 0.50,
          M_decay_scale = NULL,
          ##
          max_tau_main     = 25.0,
          max_tau_nuisance = 25.0,
          max_eps_main = 0.75,
          max_eps_nuisance = 0.75,
          ##
          reorder_cols_MVP = FALSE)
    
}
