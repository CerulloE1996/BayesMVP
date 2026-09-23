## ---- BayesMVP installer ------------------------------------------------------

## The outer package is an installer shell.  The compiled package has the same
## package name, so its installation must happen in a fresh R process after the
## outer namespace has stopped being used.  This avoids the development
## namespace-reuse failure that made the old installer unreliable.


## ---- Runtime library helpers -------------------------------------------------

.bayesmvp_runtime_dll_paths <- function(libname = NULL,
                                         pkgname = "BayesMVP") {

    package_root <- if (is.null(libname)) {
        system.file(package = pkgname)
    } else {
        file.path(libname, pkgname)
    }

    paths <- character()

    if (.Platform$OS.type == "windows") {
        paths <- c(
            file.path(package_root, "inst", "BayesMVP", "inst", "tbb_stan", "tbb.dll"),
            file.path(package_root, "inst", "BayesMVP", "inst", "dummy_stan_model_win_model.so"),
            file.path(package_root, "inst", "BayesMVP", "inst", "dummy_stan_model_win_model.dll")
        )
    } else {
        paths <- c(
            file.path(package_root, "inst", "BayesMVP", "inst", "tbb_stan", "libtbb.so.2"),
            file.path(package_root, "inst", "BayesMVP", "inst", "dummy_stan_model_model.so")
        )
    }

    paths[nzchar(paths) & file.exists(paths)]
}


.bayesmvp_try_load_runtime <- function(libname = NULL,
                                       pkgname = "BayesMVP") {

    dll_paths <- .bayesmvp_runtime_dll_paths(libname = libname,
                                             pkgname = pkgname)

    for (dll_path in dll_paths) {
        try(dyn.load(dll_path), silent = TRUE)
    }

    invisible(dll_paths)
}


#' setup_env_pre_install
#' @export
setup_env_pre_install <- function() {

    ## This helper is retained for compatibility.  It only attempts to preload
    ## files that are already present and never changes compiler flags or package
    ## libraries.
    .bayesmvp_try_load_runtime()
    invisible(TRUE)
}


#' setup_env_post_install
#' @export
setup_env_post_install <- function() {

    .bayesmvp_try_load_runtime()
    invisible(TRUE)
}


## ---- Fresh-process inner installation ---------------------------------------

.bayesmvp_validate_custom_flags <- function(CUSTOM_FLAGS) {

    if (is.null(CUSTOM_FLAGS)) {
        return(NULL)
    }

    if (!is.list(CUSTOM_FLAGS) || is.null(names(CUSTOM_FLAGS)) ||
        any(!nzchar(names(CUSTOM_FLAGS)))) {
        stop("CUSTOM_FLAGS must be a named list.")
    }

    allowed_flags <- c(
        "CCACHE_PATH",
        "CXX_COMPILER",
        "CPP_COMPILER",
        "CXX_STD",
        "CPU_BASE_FLAGS",
        "FMA_FLAGS",
        "AVX_FLAGS",
        "OMP_FLAGS",
        "OMP_LIB_PATH",
        "OMP_LIB_FLAGS",
        "PKG_CPPFLAGS",
        "PKG_CXXFLAGS",
        "CPPFLAGS",
        "CXXFLAGS",
        "PKG_LIBS"
    )

    invalid_flags <- setdiff(names(CUSTOM_FLAGS), allowed_flags)
    if (length(invalid_flags) > 0L) {
        stop(
            paste0(
                "Invalid custom flags: ", paste(invalid_flags, collapse = ", "),
                ". Allowed flags: ", paste(allowed_flags, collapse = ", ")
            )
        )
    }

    bad_values <- vapply(CUSTOM_FLAGS, function(value) {
        length(value) != 1L || is.na(value)
    }, logical(1L))
    if (any(bad_values)) {
        stop("Each CUSTOM_FLAGS value must be one non-missing character or numeric value.")
    }

    CUSTOM_FLAGS
}


.bayesmvp_write_custom_flags <- function(staged_inner,
                                          CUSTOM_FLAGS) {

    if (is.null(CUSTOM_FLAGS)) {
        return(invisible(FALSE))
    }

    makevars_files <- file.path(staged_inner, "src", c("Makevars", "Makevars.win"))
    custom_lines <- vapply(names(CUSTOM_FLAGS), function(flag_name) {
        flag_value <- as.character(CUSTOM_FLAGS[[flag_name]])
        flag_value <- gsub('"', '\\"', flag_value, fixed = TRUE)
        paste0("USER_SUPPLIED_", flag_name, ' = "', flag_value, '"')
    }, character(1L))

    for (makevars_file in makevars_files[file.exists(makevars_files)]) {
        writeLines(c(custom_lines, readLines(makevars_file, warn = FALSE)), makevars_file)
    }

    invisible(TRUE)
}


.bayesmvp_sync_outer_examples <- function(outer_root,
                                           staged_inner) {

    outer_examples <- file.path(outer_root, "examples")
    if (!dir.exists(outer_examples)) {
        ## This branch supports a development/source-loaded outer package.
        outer_examples <- file.path(outer_root, "inst", "examples")
    }
    if (!dir.exists(outer_examples)) {
        stop(paste0("The outer BayesMVP package has no examples directory: ", outer_root))
    }

    staged_examples <- file.path(staged_inner, "inst", "examples")
    unlink(staged_examples, recursive = TRUE, force = TRUE)
    dir.create(staged_examples, recursive = TRUE, showWarnings = FALSE)

    runtime_extensions <- c("so", "o", "dll", "lib", "rds", "rda", "RData")
    outer_files <- list.files(outer_examples,
                              recursive = TRUE,
                              full.names = TRUE,
                              include.dirs = FALSE,
                              all.files = TRUE)
    outer_files <- outer_files[file.info(outer_files)$isdir %in% FALSE]
    relative_files <- substring(outer_files, nchar(outer_examples) + 2L)
    keep_files <- !tolower(tools::file_ext(relative_files)) %in% tolower(runtime_extensions)

    for (relative_file in relative_files[keep_files]) {
        source_file <- file.path(outer_examples, relative_file)
        target_file <- file.path(staged_examples, relative_file)
        dir.create(dirname(target_file), recursive = TRUE, showWarnings = FALSE)
        if (!file.copy(source_file, target_file, overwrite = TRUE, copy.date = TRUE)) {
            stop(paste0("Could not copy BayesMVP example file into the staged inner package: ", relative_file))
        }
    }

    invisible(relative_files[keep_files])
}


.bayesmvp_run_inner_install <- function(staged_inner,
                                         lib,
                                         output_script,
                                         nicostan_lib) {

    r_executable <- if (.Platform$OS.type == "windows") {
        file.path(R.home("bin"), "Rscript.exe")
    } else {
        file.path(R.home("bin"), "Rscript")
    }

    child_lines <- c(
        "args <- commandArgs(trailingOnly = TRUE)",
        "staged_inner <- normalizePath(args[[1L]], mustWork = TRUE)",
        "install_library <- normalizePath(args[[2L]], mustWork = FALSE)",
        "nicostan_library <- normalizePath(args[[3L]], mustWork = TRUE)",
        "dir.create(install_library, recursive = TRUE, showWarnings = FALSE)",
        ".libPaths(c(install_library, nicostan_library, .libPaths()))",
        "Sys.setenv(R_LIBS = paste(c(install_library, nicostan_library, .libPaths()), collapse = .Platform$path.sep))",
        "if (any(grepl('BayesMVP', loadedNamespaces()))) stop('BayesMVP must not be loaded in the installer child process.')",
        "status <- system2(command = file.path(R.home('bin'), if (.Platform$OS.type == 'windows') 'R.exe' else 'R'), args = c('CMD', 'INSTALL', paste0('--library=', shQuote(install_library)), '--no-test-load', '--preclean', shQuote(staged_inner)))",
        "if (!identical(as.integer(status), 0L)) stop(paste0('R CMD INSTALL failed with status ', status, '.'))",
        "loadNamespace('BayesMVP', lib.loc = install_library)",
        "required_exports <- c('MVP_model', 'initialise_model', 'setup_env_post_install')",
        "actual_exports <- getNamespaceExports('BayesMVP')",
        "if (!all(required_exports %in% actual_exports)) stop(paste0('Installed inner BayesMVP is missing exports: ', paste(setdiff(required_exports, actual_exports), collapse = ', ')))",
        "message(paste0('BayesMVP inner package installed and verified in: ', install_library))"
    )

    writeLines(child_lines, output_script)

    status <- system2(
        command = r_executable,
        args = c("--vanilla", shQuote(output_script), shQuote(staged_inner), shQuote(lib), shQuote(nicostan_lib))
    )

    if (!identical(as.integer(status), 0L)) {
        stop(paste0("The fresh-process BayesMVP inner installation failed with status ", status, "."))
    }

    invisible(TRUE)
}


## ---- Public installer --------------------------------------------------------

#' Install the compiled BayesMVP package from the bundled inner source
#'
#' The outer BayesMVP package is an installer shell.  This function stages the
#' bundled inner source and installs it from a fresh R process, so a development
#' namespace already loaded by the caller cannot be reused accidentally.
#'
#' @param CUSTOM_FLAGS Optional named list of Makevars overrides.  The default
#'   uses the package's ordinary compiler and CPU detection.
#' @param lib Target R package library.  Defaults to the first current library.
#' @param force Passed through as an installation intent.  The installer never
#'   removes a package directory itself; the fresh R CMD INSTALL process handles
#'   replacement while no BayesMVP namespace is loaded in that process.
#' @return Invisibly, the target library path.
#' @export
install_BayesMVP <- function(CUSTOM_FLAGS = NULL,
                             lib = .libPaths()[1L],
                             nicostan_lib = .libPaths()[1L],
                             force = TRUE) {

    if (!is.character(lib) || length(lib) != 1L || !nzchar(lib)) {
        stop("lib must be one non-empty directory path.")
    }
    if (!isTRUE(force)) {
        stop("force must remain TRUE for the outer-to-inner package replacement.")
    }

    if (!is.character(nicostan_lib) || length(nicostan_lib) != 1L || !nzchar(nicostan_lib)) {
        stop("nicostan_lib must be one non-empty directory path.")
    }

    nicostan_lib <- normalizePath(nicostan_lib, mustWork = FALSE)
    if (!requireNamespace("NicoStan", quietly = TRUE, lib.loc = nicostan_lib)) {
        stop("Install NicoStan into nicostan_lib before compiling BayesMVP.")
    }
    nicostan_header <- system.file("include", "NicoStan",
                                   package = "NicoStan",
                                   lib.loc = nicostan_lib)
    if (!nzchar(nicostan_header) || !dir.exists(nicostan_header)) {
        stop(paste0("The NicoStan installation has no exported include directory: ", nicostan_header))
    }
    nicostan_version <- utils::packageVersion("NicoStan", lib.loc = nicostan_lib)
    if (nicostan_version < "0.1.9000") {
        stop(paste0("BayesMVP requires NicoStan >= 0.1.9000; found ", nicostan_version))
    }

    outer_root <- system.file(package = "BayesMVP")
    inner_pkg <- system.file("BayesMVP", package = "BayesMVP")

    if (!nzchar(outer_root) || !file.exists(file.path(outer_root, "DESCRIPTION"))) {
        stop("The outer BayesMVP package is not installed in the current R session.")
    }
    if (!nzchar(inner_pkg) || !file.exists(file.path(inner_pkg, "DESCRIPTION"))) {
        stop(paste0(
            "The installed outer BayesMVP package has no bundled inner source. ",
            "Install the outer source package in a fresh R session first."
        ))
    }

    CUSTOM_FLAGS <- .bayesmvp_validate_custom_flags(CUSTOM_FLAGS)
    lib <- normalizePath(lib, mustWork = FALSE)
    dir.create(lib, recursive = TRUE, showWarnings = FALSE)

    staging_root <- tempfile("BayesMVP_inner_install_")
    dir.create(staging_root, recursive = TRUE, showWarnings = FALSE)
    on.exit(unlink(staging_root, recursive = TRUE, force = TRUE), add = TRUE)

    staged_inner <- file.path(staging_root, "BayesMVP")
    if (!file.copy(inner_pkg, staging_root, recursive = TRUE, copy.date = TRUE)) {
        stop("Could not stage the bundled inner BayesMVP source package.")
    }
    if (!file.exists(file.path(staged_inner, "DESCRIPTION"))) {
        stop("The staged inner BayesMVP source is incomplete.")
    }

    .bayesmvp_sync_outer_examples(outer_root = outer_root,
                                  staged_inner = staged_inner)

    .bayesmvp_write_custom_flags(staged_inner = staged_inner,
                                 CUSTOM_FLAGS = CUSTOM_FLAGS)

    child_script <- file.path(staging_root, "install_inner_in_fresh_process.R")
    .bayesmvp_run_inner_install(staged_inner = staged_inner,
                                lib = lib,
                                output_script = child_script,
                                nicostan_lib = nicostan_lib)

    message(paste0(
        "BayesMVP inner package installation complete in ", lib,
        ". Restart R before using the newly installed compiled namespace."
    ))

    invisible(lib)
}
