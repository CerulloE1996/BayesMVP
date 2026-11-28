

## ---- The parameter families that Se, Sp and prevalence depend on, for the built-in latent-class models --------
##
## interest_only (R_fn_sample_model): every tau criterion computes its statistic over the main-block rows of a set
## of parameter families only. For the built-in latent-class models of diagnostic accuracy (BayesMVP's LC_MVP and
## LC_MVOP) the reported quantities - Se, Sp and prevalence - are the same for every user, so the package supplies
## the set here. Every other model (user-supplied Stan models, and the built-in MVP / MVOP / latent_trait models,
## which report no Se, Sp or prevalence) has no built-in set: NULL is returned, and interest_only needs a
## user-supplied character vector of parameter families there.
##
## Derivation (the reported quantities and the main parameter families they are computed from):
##   LC_MVP  (binary tests):   the prevalence from p_raw; Se_j and Sp_j (and Fp_j = 1 - Sp_j) of test j from the
##                             latent-class means of test j in beta; hence {beta, p_raw};
##   LC_MVOP (ordinal tests):  the prevalence from p_raw; Se and Sp of test j at each threshold from the
##                             latent-class means in beta AND the cutpoints of test j in C_unc_vec; hence
##                             {beta, p_raw, C_unc_vec}.
##   Neither depends on Omega_unconstrained_vec (the within-class correlations), the only other main family: the
##   marginal of one test of a multivariate probit has unit latent variance whatever the correlations.
##
## BayesMVP's own definition of NicoStan's general hook fn_parameter_families_of_the_reported_quantities() (NicoStan's
## default, in R_fn_model_backend.R, returns NULL): NicoStan's backend for BayesMVP takes BayesMVP's functions first,
## so the sampler uses this one for BayesMVP's models. Se, Sp and prevalence belong to BayesMVP, never to NicoStan.
##
## ---- Renamed (no alias); the previous names (in NicoStan):
##
## parameters_that_Se_Sp_and_prevalence_depend_on_for_built_in_latent_class_models <-  function(Model_type) {
## parms_that_Se_Sp_and_prev_depend_on_for_built_in_LCMs <-  function(Model_type) {
fn_parameter_families_of_the_reported_quantities <-  function(Model_type) {

        if (identical(Model_type, "LC_MVP"))   return(c("beta", "p_raw"))
        if (identical(Model_type, "LC_MVOP"))  return(c("beta", "p_raw", "C_unc_vec"))
        if (Model_type %in% c("CLC_MVOP", "CLC-MVOP")) {
              return(c(
                "beta_vec",
                "beta_prevalence_A",
                "beta_prevalence_B",
                "z_logit_A",
                "z_logit_B",
                "mu_logit_A",
                "mu_logit_B",
                "sigma_logit_A",
                "sigma_logit_B",
                "log_OR",
                "C_unc_vec"
              ))
        }
        return(NULL)

}
##
##
## ---- The Se / Sp prior inputs of BayesMVP's built-in models, put into fitted ordinal-slot order ----------------
##
## BayesMVP's own definition of NicoStan's general hook fn_put_model_inputs_in_fitted_ordinal_slot_order() (NicoStan's
## default, in R_fn_model_backend.R, changes nothing), called by the sampler's reorder_cols_MVP once the ordinal-indexed
## metadata are in the fitted order. Each Se / Sp prior input of the models that read them (CLC_MVOP, LC_MVOP, MVOP) is
## indexed by ordinal test, as [class, ordinal test, ...] (n_chosen_cutoffs / n_chosen, chosen_cutoffs,
## prior_Se_Sp_dirichlet_A, prior_Se_Sp_beta, prior_Se_Sp_link_mean_SD) or [ordinal test, ...]
## (prior_Se_Sp_gap_link_mean_SD), as an array, or as a list per class of lists per ordinal test. All go into fitted
## ordinal-slot order, once. ord_of_fitted[k] is the ORIGINAL ordinal index of the k-th ordinal fitted slot.
##
fn_put_model_inputs_in_fitted_ordinal_slot_order <-  function( model_args_list,
                                                               ord_of_fitted) {

        fn_Se_Sp_prior_input_in_fitted_ordinal_slot_order <- function(x,
                                                                      ordinal_axis) {
              if (is.null(x)) return(x)
              if (is.list(x)) {
                    if (ordinal_axis == 1L) return(x[ord_of_fitted])
                    return(lapply(x, function(per_class) if (is.list(per_class)) per_class[ord_of_fitted] else per_class))
              }
              n_dims <- length(dim(x))
              if (n_dims < ordinal_axis) return(x)
              index_list <- rep(list(TRUE), n_dims)
              index_list[[ordinal_axis]] <- ord_of_fitted
              return(do.call(`[`, c(list(x), index_list, list(drop = FALSE))))
        }
        for (Se_Sp_input_name in c("n_chosen_cutoffs", "n_chosen", "chosen_cutoffs", "prior_Se_Sp_dirichlet_A",
                                   "prior_Se_Sp_beta", "prior_Se_Sp_link_mean_SD")) {
              model_args_list[[Se_Sp_input_name]] <- fn_Se_Sp_prior_input_in_fitted_ordinal_slot_order(model_args_list[[Se_Sp_input_name]],
                                                                                                       ordinal_axis = 2L)
        }
        model_args_list[["prior_Se_Sp_gap_link_mean_SD"]] <- fn_Se_Sp_prior_input_in_fitted_ordinal_slot_order(model_args_list[["prior_Se_Sp_gap_link_mean_SD"]],
                                                                                                               ordinal_axis = 1L)
        return(model_args_list)

}





















