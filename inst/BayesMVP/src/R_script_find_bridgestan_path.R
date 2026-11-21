##
## ---- BridgeStan path for the Makevars, from NicoStan::bridgestan_path(). Anything printed to stdout whilst
##      NicoStan loads is captured, so that the only output (read by make) is the path:
##
NicoStan_startup_output <-  utils::capture.output(BridgeStan_path <-  NicoStan::bridgestan_path())
##
cat(BridgeStan_path)






















