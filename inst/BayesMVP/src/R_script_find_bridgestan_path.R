## Build-time path discovery uses the maintained helper in installed NicoStan.
## Capture namespace startup output so stdout contains only the path consumed by make.
invisible(capture.output(build_dependency_path <- NicoStan::bridgestan_path()))
cat(build_dependency_path)
