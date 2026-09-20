#### ======================================================================================================= test_15_nested_rhat_grouping.R
##
## ---- Diagnostic-only tests: no MCMC, no installation, no changes to saved fits -----------------------------------------------------
##
{
    require(RcppParallel)
    require(BayesMVP)
    require(posterior)
}
test_script_arguments <-  grep(pattern = "^--file=", x = commandArgs(trailingOnly = FALSE), value = TRUE)
tests_directory <-  dirname(path = normalizePath(path = sub(pattern = "^--file=", replacement = "", x = test_script_arguments[1])))
repository_directory <-  dirname(path = dirname(path = tests_directory))
workspace_directory <-  dirname(path = dirname(path = repository_directory))
package_R_directory <-  file.path(repository_directory, "inst", "BayesMVP", "R")
ps7_directory <-  file.path(workspace_directory, "Alg_paper_analysis", "1_appendix_pilot_studies", "ps_7_basic_MCMC_settings_BayesMVP")
source(file = file.path(package_R_directory, "R_fn_create_superchain_ids.R"))
source(file = file.path(package_R_directory, "R_fn_generate_summary_tibble.R"))
source(file = file.path(ps7_directory, "functions", "fn_ps7_extract_tau_sweep.R"))
##
## ---- (A) Group identities agree with the installed native initializer, including its trailing block -------------------------------
##
for (configuration in list(c(180, 180, 4), c(180, 180, 8), c(10, 3, 3), c(12, 6, 4))) {
    n_chains_sampling <-  configuration[1]
    n_superchains <-  configuration[2]
    n_chains_burnin <-  configuration[3]
    grouping <-  fn_nested_rhat_grouping_from_burnin(n_chains_sampling = n_chains_sampling,
                                                    n_superchains = n_superchains,
                                                    n_chains_burnin = n_chains_burnin,
                                                    nuisance_jitter_scale = 0)
    endpoint_main <-  rbind(seq_len(length.out = n_chains_burnin), 10 + seq_len(length.out = n_chains_burnin))
    endpoint_nuisance <-  endpoint_main + 100
    native_initial_states <-  BayesMVP:::cpp_fn_post_burnin_prep_for_sampling(
        n_chains_burnin = n_chains_burnin,
        n_chains_sampling = n_chains_sampling,
        n_superchains = n_superchains,
        n_params_main = 2,
        n_nuisance = 2,
        theta_main_in = endpoint_main,
        theta_us_in = endpoint_nuisance,
        nuisance_jitter_scale = 0,
        seed = 123)
    stopifnot(identical(x = native_initial_states$theta_main_vectors_all_chains_input_from_R,
                       y = endpoint_main[, grouping$superchain_ids, drop = FALSE]),
              identical(x = native_initial_states$theta_us_vectors_all_chains_input_from_R,
                       y = endpoint_nuisance[, grouping$superchain_ids, drop = FALSE]))
}
grouping_four <-  fn_nested_rhat_grouping_from_burnin(n_chains_sampling = 180, n_superchains = 180,
                                                    n_chains_burnin = 4, nuisance_jitter_scale = 0)
grouping_eight <-  fn_nested_rhat_grouping_from_burnin(n_chains_sampling = 180, n_superchains = 180,
                                                     n_chains_burnin = 8, nuisance_jitter_scale = 0)
stopifnot(grouping_four$n_chains_used == 180, grouping_four$n_chains_per_superchain == 45,
          grouping_eight$n_chains_used == 176, grouping_eight$n_chains_per_superchain == 22,
          grouping_eight$n_chains_omitted == 4,
          all(grouping_eight$chain_indices == seq_len(length.out = 176)),
          all(grouping_eight$superchain_ids == rep(x = seq_len(length.out = 8), length.out = 180)))
cat("PASS: native initial states, interleaved endpoint IDs, and balanced 4x45 / 8x22 subsets.\n")
##
## ---- (B) Match posterior::rhat_nested; unavailable full-state nesting must not look converged --------------------------------------
##
set.seed(seed = 918)
test_draws <-  array(data = rnorm(n = 50 * 180 * 2), dim = c(50, 180, 2))
for (chain_index in seq_len(length.out = 180)) {
    test_draws[, chain_index, ] <-  test_draws[, chain_index, ] + grouping_eight$superchain_ids[chain_index] / 4
}
nested_rhat <-  fn_nested_rhat_from_draws_array(draws_array = test_draws, nested_rhat_grouping = grouping_eight)
reference_nested_rhat <-  vapply(X = seq_len(length.out = 2), FUN.VALUE = 0, FUN = function(parameter_index) {
    posterior::rhat_nested(x = test_draws[, seq_len(length.out = 176), parameter_index],
                           superchain_ids = grouping_eight$selected_superchain_ids)
})
stopifnot(identical(x = nested_rhat, y = reference_nested_rhat),
          identical(x = fn_nested_rhat_from_draws_array(draws_array = test_draws[, , 1, drop = FALSE],
                                                        nested_rhat_grouping = grouping_eight), y = nested_rhat[1]))
jittered_grouping <-  fn_nested_rhat_grouping_from_burnin(n_chains_sampling = 180, n_superchains = 180,
                                                        n_chains_burnin = 8, nuisance_jitter_scale = 0.01)
one_state_grouping <-  fn_prepare_nested_rhat_grouping(superchain_ids = rep(x = 1, times = 180))
for (unavailable_grouping in list(NULL, list(status = "unavailable"), jittered_grouping, one_state_grouping)) {
    stopifnot(all(is.na(x = fn_nested_rhat_from_draws_array(draws_array = test_draws,
                                                          nested_rhat_grouping = unavailable_grouping))))
}
cat("PASS: posterior estimator agreement, one-parameter arrays, unknown/jittered/single-state safeguards.\n")
##
## ---- (C) Production summaries change nested R-hat ONLY; ordinary R-hat and ESS still use 180 chains --------------------------------
##
old_summary_function <-  get(x = "generate_summary_tibble", envir = asNamespace(ns = "BayesMVP"))
old_summary <-  old_summary_function(trace = aperm(a = test_draws, perm = c(3, 1, 2)), param_names = c("a", "b"),
                                     n_to_compute = 2, compute_nested_rhat = FALSE, n_chains = 180, n_superchains = 180)
new_summary <-  generate_summary_tibble(trace = aperm(a = test_draws, perm = c(3, 1, 2)), param_names = c("a", "b"),
                                       n_to_compute = 2, compute_nested_rhat = TRUE, n_chains = 180, n_superchains = 180,
                                       nested_rhat_grouping = grouping_eight)
unchanged_columns <-  setdiff(x = names(x = old_summary), y = "n_Rhat")
stopifnot(identical(x = old_summary[unchanged_columns], y = new_summary[unchanged_columns]),
          identical(x = new_summary$n_Rhat, y = nested_rhat))
cat("PASS: ESS, ordinary R-hat, means, intervals and SDs unchanged.\n")
##
## ---- (D) Old-run repair is explicit, recorded maps take precedence, and raw fit files stay untouched -------------------------------
##
fixture_directory <-  tempfile(pattern = "nested_rhat_grouping_")
dir.create(path = fixture_directory)
fixture_path <-  file.path(fixture_directory, "ps7_run_LC_MVP_N2500_ta2_cb8_s180_run1")
fixture_fit <-  list(settings = list(n_chains_sampling = 180, n_chains_burnin = 8),
                     HMC_info = list(n_superchains = 180),
                     trace_gq = test_draws, trace_main = test_draws,
                     min_ESS = 123, max_Rhat = 1.2, max_nRhat = 1.1,
                     efficiency_info = list(Min_ESS_main = 321, Max_rhat_main = 1.3, Max_nested_rhat_main = 1.2))
saveRDS(object = fixture_fit, file = fixture_path)
original_checksum <-  tools::md5sum(files = fixture_path)
unreconstructed <-  fn_ps7_extract_tau_sweep(dir = fixture_directory, verbose = FALSE)
reconstructed <-  fn_ps7_extract_tau_sweep(dir = fixture_directory, verbose = FALSE, reconstruct_ta2_superchains = TRUE)
stopifnot(is.na(x = unreconstructed$max_nRhat), reconstructed$max_nRhat == max(nested_rhat),
          reconstructed$max_nRhat_main == max(nested_rhat), reconstructed$max_nRhat_stored == 1.1,
          reconstructed$n_chains_nested_rhat == 176, reconstructed$min_ESS == 123,
          reconstructed$min_ESS_main == 321,
          identical(x = tools::md5sum(files = fixture_path), y = original_checksum))
fixture_fit$HMC_info$nested_rhat_grouping <-  grouping_four
saveRDS(object = fixture_fit, file = fixture_path)
recorded <-  fn_ps7_extract_tau_sweep(dir = fixture_directory, verbose = FALSE, reconstruct_ta2_superchains = TRUE)
stopifnot(recorded$n_superchains_diagnostic == 4, recorded$n_chains_nested_rhat == 180)
cat("PASS: explicit legacy reconstruction, recorded-map precedence, audit columns, no raw-fit mutation.\n")
