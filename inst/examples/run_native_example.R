#### =====================================================================================================================================
## Small simulated examples for the five native BayesMVP model identifiers.
## Model_type = "latent_trait" is the current two-class latent-trait implementation.
## Internal *_cpp_skeleton.stan files only reconstruct parameters; the native backend evaluates the target.
##
run_native_example <-  function( Model_type = c("MVP", "LC_MVP", "MVOP", "LC_MVOP", "latent_trait"),
                                  burnin_algorithm = "CHESSR",
                                  n_burnin = 200L,
                                  n_iter = 250L,
                                  seed = 2026L,
                                  N = 48L) {
        Model_type <-  match.arg(Model_type)
        set.seed(seed)
        stopifnot(length(N) == 1L, is.finite(N), N >= 12L, N == as.integer(N))
        n <-  as.integer(N)
        n_tests <-  3L
        n_class <-  if (Model_type %in% c("LC_MVP", "LC_MVOP", "latent_trait")) 2L else 1L
        ordinal <-  Model_type %in% c("MVOP", "LC_MVOP")
        latent_class <-  if (n_class == 2L) rbinom(n, size = 1L, prob = 0.3) else rep(0L, n)
        common <-  rnorm(n)
        latent <-  sapply(seq_len(n_tests), function(test) -0.8 + 1.6 * latent_class + 0.4 * common + rnorm(n))
        y <-  1L * (latent > 0)
        if (ordinal) y[, n_tests] <-  1L + (latent[, n_tests] > -0.5) + (latent[, n_tests] > 0.5)
        storage.mode(y) <-  "integer"
        prior_means <-  lapply(seq_len(n_class), function(class) matrix(if (class == 1L) -0.8 else 0.8, nrow = 1L, ncol = n_tests))
        model_args <-  list(y = y, num_chunks = 1L,
                            prior_coeffs_mean_mat = prior_means,
                            prior_coeffs_sd_mat = lapply(seq_len(n_class), function(class) matrix(0.5, nrow = 1L, ncol = n_tests)))
        if (ordinal) model_args$n_cat_per_ord_test <-  3L
        model <-  BayesMVP::MVP_model$new(Model_type = Model_type, model_args_list = model_args, sample_nuisance = TRUE)
        initial <-  list(u_raw = matrix(0, nrow = n, ncol = n_tests))
        if (Model_type == "latent_trait") {
                initial$LT_b_raw_vec <-  rep(-2, n_class * n_tests)
                initial$LT_a_vec <-  rep(c(-0.8, 0.8), each = n_tests)
        } else {
                initial$Omega_unconstrained_vec <-  lapply(seq_len(n_class), function(class) rep(0, choose(n_tests, 2L)))
                initial$beta <-  prior_means
        }
        if (n_class == 2L) initial$p_raw <-  array(atanh(2 * 0.3 - 1), dim = 1L)
        if (ordinal) {
                raw_cutpoints <-  c(-0.5, 0)
                initial$C_unc_vec <-  lapply(seq_len(n_class), function(class) atanh(2 * (raw_cutpoints + 7.5) / 10 - 1))
        }
        model$sample(init_lists_per_chain = rep(list(initial), 4L), n_burnin = n_burnin, n_iter = n_iter,
                      n_chains_burnin = 4L, n_chains_sampling = 4L, n_superchains = 2L,
                      burnin_algorithm = burnin_algorithm, diffusion_HMC = TRUE, partitioned_HMC = FALSE,
                      metric_type_main = "Empirical", metric_shape_main = "diag", metric_type_nuisance = "uniform_diag",
                      M_decay_type = "inverse", M_decay_power = 0.5, M_decay_scale = 1.13,
                      ratio_M_main = 0.9, ratio_M_nuisance = 0.9, learning_rate = 0.05, learning_rate_initial = 0.1,
                      n_threads_WCP_burnin = 1L, n_threads_WCP_sampling = 1L, adapt_delta = 0.9,
                      n_refresh = 100L, seed = seed, stream = seed, reorder_cols_MVP = FALSE)
        model$summary(save_log_lik_trace = FALSE, compute_nested_rhat = FALSE)
        model
}
