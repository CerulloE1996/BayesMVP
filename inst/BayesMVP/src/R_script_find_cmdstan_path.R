##
## ---- CmdStan path for the Makevars, from NicoStan::cmdstanr_path(). Anything printed to stdout whilst NicoStan
##      loads is captured, so that the only output (read by make) is the path:
##
NicoStan_startup_output <-  utils::capture.output(CmdStan_path <-  NicoStan::cmdstanr_path(
                                                                                           preferred_version = "2.36.0"))
##
cat(CmdStan_path)






















