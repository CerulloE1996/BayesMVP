#### =====================================================================================================================================
## Run_MVP.R
## Small self-contained standard multivariate probit example without latent classes.
## Data are generated locally so the script has no external data-path dependency.
##
## Editable demonstration settings are grouped below.
#### =====================================================================================================================================

## ---- Editable demonstration settings -----------------------------------------------------------------------------------------------
Model_type <-  "MVP"
seed <-  20260923L
N <-  48L
n_tests <-  3L
n_class <-  1L
n_pops <-  1L
num_chunks <-  1L
vect_type <-  "Stan"
Phi_type <-  "Phi"
inv_Phi_type <-  "inv_Phi"

n_burnin <-  200L
n_iter <-  250L
n_chains_burnin <-  4L
n_chains_sampling <-  4L
n_superchains <-  2L
burnin_algorithm <-  "CHESSR"
n_threads_WCP_burnin <-  1L
n_threads_WCP_sampling <-  1L
n_nuisance_to_track <-  0L

## ---- Generate a small correlated binary data set ----------------------------------------------------------------------------------
set.seed(seed, kind = "L'Ecuyer-CMRG")
common_signal <-  stats::rnorm(n = N)
test_shift <-  c(-0.40, 0.00, 0.40)
latent_values <-  matrix(0.45 * common_signal, nrow = N, ncol = n_tests)
latent_values <-  latent_values + matrix(rep(test_shift, each = N), nrow = N, ncol = n_tests)
latent_values <-  latent_values + matrix(stats::rnorm(N * n_tests), nrow = N, ncol = n_tests)
y <-  1L * (latent_values > 0)
storage.mode(y) <-  "integer"

## ---- Model arguments and initial values --------------------------------------------------------------------------------------------
n_covariates_per_outcome_mat <-  matrix(1L, nrow = n_class, ncol = n_tests)
X <-  lapply(seq_len(n_class), function(class_index) {
    lapply(seq_len(n_tests), function(test_index) {
        matrix(1, nrow = N, ncol = 1L)
    })
})
prior_coeffs_mean_mat <-  list( matrix(-0.40, nrow = 1L, ncol = n_tests))
prior_coeffs_sd_mat <-  list( matrix(0.75, nrow = 1L, ncol = n_tests))

model_args_list <-  list( y = y,
                           N = N,
                           n_class = n_class,
                           n_pops = n_pops,
                           X = X,
                           n_covariates_per_outcome_mat = n_covariates_per_outcome_mat,
                           num_chunks = num_chunks,
                           prior_coeffs_mean_mat = prior_coeffs_mean_mat,
                           prior_coeffs_sd_mat = prior_coeffs_sd_mat,
                           lkj_cholesky_eta = matrix(2, nrow = 1L, ncol = 1L),
                           vect_type = vect_type,
                           Phi_type = Phi_type,
                           inv_Phi_type = inv_Phi_type
)

initial_values <-  list( u_raw = matrix(0, nrow = N, ncol = n_tests),
                          Omega_unconstrained_vec = list(rep(0, choose(n_tests, 2L))),
                          beta = prior_coeffs_mean_mat
)
init_lists_per_chain <-  rep(list(initial_values), n_chains_burnin)

message(paste0("Preparing ", Model_type, " example with N = ", N, ", n_tests = ", n_tests,
               ", n_class = ", n_class, "."))

## ---- Initialise and sample ---------------------------------------------------------------------------------------------------------
model <-  BayesMVP::MVP_model$new( Model_type = Model_type,
                                    model_args_list = model_args_list,
                                    sample_nuisance = TRUE
)

fit <-  model$sample( init_lists_per_chain = init_lists_per_chain,
                      n_chains_burnin = n_chains_burnin,
                      n_burnin = n_burnin,
                      n_iter = n_iter,
                      n_chains_sampling = n_chains_sampling,
                      n_superchains = n_superchains,
                      burnin_algorithm = burnin_algorithm,
                      partitioned_HMC = FALSE,
                      diffusion_HMC = TRUE,
                      metric_type_main = "Empirical",
                      metric_shape_main = "diag",
                      metric_type_nuisance = "uniform_diag",
                      M_decay_type = "inverse",
                      M_decay_power = 0.5,
                      M_decay_scale = 1.13,
                      ratio_M_main = 0.90,
                      ratio_M_nuisance = 0.90,
                      learning_rate = 0.05,
                      learning_rate_initial = 0.10,
                      n_threads_WCP_burnin = n_threads_WCP_burnin,
                      n_threads_WCP_sampling = n_threads_WCP_sampling,
                      n_refresh = 100L,
                      adapt_delta = 0.90,
                      n_nuisance_to_track = n_nuisance_to_track,
                      seed = seed,
                      stream = seed
)

model_fit <-  fit$summary( save_log_lik_trace = FALSE,
                           compute_nested_rhat = FALSE
)
main_summary <-  model_fit$get_summary_main()
generated_summary <-  model_fit$get_summary_generated_quantities()
divergences <-  model_fit$get_divergences()

message(paste0("Completed ", Model_type, " example."))
invisible(list(model = model, fit = fit, summary = model_fit,
               main_summary = main_summary, generated_summary = generated_summary,
               divergences = divergences))























