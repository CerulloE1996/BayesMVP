#### =====================================================================================================================================
## zz_shared_backend.R
##
## ---- BayesMVP keeps NO copy of NicoStan's functions. Every time BayesMVP loads (.onLoad), NicoStan's shared R interface
##      (R_fn_sample_model, MVP_model, ...) is taken from the NicoStan installed NOW and bound to BayesMVP's own native
##      (C++) model operations; any change to NicoStan's R code is used by BayesMVP straight away, with no BayesMVP reinstall.
##
R_fn_bind_NicoStan_shared_functions_into_BayesMVP <- function() {

        BayesMVP_namespace <-  environment(R_fn_bind_NicoStan_shared_functions_into_BayesMVP)
        ##
        bayesmvp_backend <-  NicoStan::fn_create_model_backend( provider = BayesMVP_namespace,
                                                                model_types = c( "Stan",
                                                                                 "MVP",
                                                                                 "MVOP",
                                                                                 "LC_MVP",
                                                                                 "LC_MVOP",
                                                                                 "CLC-MVOP",
                                                                                 "CLC_MVOP",
                                                                                 "latent_trait"))
        ##
        ## ---- every object of NicoStan's shared backend that BayesMVP does not define itself is bound into its namespace:
        ##
        for (shared_name in ls(bayesmvp_backend, all.names = TRUE)) {
                if (!exists(shared_name, envir = BayesMVP_namespace, inherits = FALSE)) {
                        assign(shared_name, get(shared_name, envir = bayesmvp_backend, inherits = FALSE), envir = BayesMVP_namespace)
                }
        }
        assign(".bayesmvp_backend", bayesmvp_backend, envir = BayesMVP_namespace)

        return(value_not_printed <- NULL)

}
