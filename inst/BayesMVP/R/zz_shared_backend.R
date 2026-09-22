#### =====================================================================================================================================
## zz_shared_backend.R
##
## ---- Instantiate NicoStan's shared R interface with BayesMVP's existing native/model operations --------------------------------------
##
.bayesmvp_backend <-  NicoStan::fn_create_model_backend(
    provider = environment(),
    model_types = c("Stan", "MVP", "MVOP", "LC_MVP", "LC_MVOP", "latent_trait"))
##
for (.shared_name in ls(.bayesmvp_backend, all.names = TRUE)) {
        if (!exists(.shared_name, envir = environment(), inherits = FALSE)) {
                assign(.shared_name, get(.shared_name, envir = .bayesmvp_backend, inherits = FALSE))
        }
}
rm(.shared_name)
