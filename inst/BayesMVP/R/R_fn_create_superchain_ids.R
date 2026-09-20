
## Create superchain IDs for chain grouping when many chains are used


#' #' create_superchain_ids
#' @keywords internal
#' @export
create_superchain_ids <- function(n_superchains,
                                  n_chains) {
  
      for (count in list(n_superchains, n_chains)) {
          if (!is.numeric(x = count) || length(x = count) != 1 || !is.finite(x = count) || count < 1 || count != floor(x = count)) {
              stop("n_superchains and n_chains must be positive whole numbers.")
          }
      }
      ## Match cpp_fn_post_burnin_prep_for_sampling: floor-sized blocks, with the remainder in the LAST block.
      n_superchains <- min(n_superchains, n_chains)
      chains_per_superchain <- floor(n_chains / n_superchains)
      superchain_ids <- pmin((seq_len(length.out = n_chains) - 1) %/% chains_per_superchain + 1, n_superchains)
      
      return(superchain_ids)
  
}

##
## ---- Nested R-hat groups must represent common FULL starting states, not arbitrary chain blocks ---------------------------------
##
fn_nested_rhat_grouping_from_burnin <-  function(n_chains_sampling,
                                                n_superchains,
                                                n_chains_burnin,
                                                nuisance_jitter_scale,
                                                source = "recorded_burnin_endpoint_mapping") {
    if (!is.numeric(x = n_chains_burnin) || length(x = n_chains_burnin) != 1 || !is.finite(x = n_chains_burnin) ||
        n_chains_burnin < 1 || n_chains_burnin != floor(x = n_chains_burnin)) {
        stop("n_chains_burnin must be one positive whole number.")
    }
    nominal_superchain_ids <-  create_superchain_ids(n_superchains = n_superchains, n_chains = n_chains_sampling)
    ## C++ assigns burnin_col = superchain_id % n_chains_burnin, using zero-based indices.
    initial_state_ids <-  (nominal_superchain_ids - 1) %% n_chains_burnin + 1
    common_initial_states <-  is.numeric(x = nuisance_jitter_scale) && length(x = nuisance_jitter_scale) == 1 &&
        is.finite(x = nuisance_jitter_scale) && nuisance_jitter_scale == 0
    grouping <-  fn_prepare_nested_rhat_grouping(superchain_ids = initial_state_ids,
                                                common_initial_states = common_initial_states,
                                                source = source)
    grouping$n_nominal_superchains <-  length(x = unique(x = nominal_superchain_ids))
    grouping$nuisance_jitter_scale <-  nuisance_jitter_scale
    grouping
}
##
fn_prepare_nested_rhat_grouping <-  function(superchain_ids,
                                            common_initial_states = TRUE,
                                            source = "explicit_superchain_ids") {
    if (!is.numeric(x = superchain_ids) || length(x = superchain_ids) == 0 ||
        any(!is.finite(x = superchain_ids)) || any(superchain_ids < 1 | superchain_ids != floor(x = superchain_ids))) {
        stop("superchain_ids must identify every chain with a positive whole-number starting-state ID.")
    }
    chain_indices_by_group <-  split(x = seq_along(along.with = superchain_ids), f = superchain_ids)
    n_chains_per_superchain <-  min(lengths(x = chain_indices_by_group))
    ## Select by original chain index ONLY, never by samples, acceptance or diagnostic quality.
    chain_indices <-  sort(x = unlist(x = lapply(X = chain_indices_by_group, FUN = function(indices) {
        head(x = indices, n = n_chains_per_superchain)
    }), use.names = FALSE))
    status <-  if (!isTRUE(x = common_initial_states)) "unavailable_nonidentical_or_unknown_initial_states" else
        if (length(x = chain_indices_by_group) < 2) "unavailable_only_one_starting_state" else
        if (n_chains_per_superchain == 1) "singleton_groups" else "ok"
    list(version = 1,
         source = source,
         status = status,
         superchain_ids = superchain_ids,
         chain_indices = chain_indices,
         selected_superchain_ids = superchain_ids[chain_indices],
         n_superchains = length(x = chain_indices_by_group),
         n_chains_per_superchain = n_chains_per_superchain,
         n_chains_used = length(x = chain_indices),
         n_chains_omitted = length(x = superchain_ids) - length(x = chain_indices))
}
##
fn_nested_rhat_from_draws_array <-  function(draws_array, nested_rhat_grouping) {
    ## Input is iterations x chains x parameters, matching PS7's saved trace_main/trace_gq arrays.
    if (length(x = dim(x = draws_array)) != 3) stop("Nested R-hat draws must be an iterations x chains x parameters array.")
    if (is.null(x = nested_rhat_grouping) || !isTRUE(x = nested_rhat_grouping$status %in% c("ok", "singleton_groups"))) {
        return(rep(x = NA_real_, times = dim(x = draws_array)[3]))
    }
    if (dim(x = draws_array)[2] != length(x = nested_rhat_grouping$superchain_ids)) {
        stop("Nested R-hat grouping does not match the number of stored chains.")
    }
    vapply(X = seq_len(length.out = dim(x = draws_array)[3]), FUN.VALUE = 0, FUN = function(parameter_index) {
        parameter_draws <-  matrix(data = draws_array[, nested_rhat_grouping$chain_indices, parameter_index, drop = FALSE],
                                    nrow = dim(x = draws_array)[1], ncol = nested_rhat_grouping$n_chains_used)
        posterior::rhat_nested(x = parameter_draws, superchain_ids = nested_rhat_grouping$selected_superchain_ids)
    })
}










