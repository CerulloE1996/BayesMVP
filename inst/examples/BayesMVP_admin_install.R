## ---- BayesMVP local administrator installer -------------------------------

## Source this file directly from RStudio to install or reinstall BayesMVP.
## The complete build runs automatically in a clean R process, without loading
## or unloading packages in the calling session. Source it again to reinstall.
## To define the functions without installing, first set
## options(BayesMVP.admin.autorun = FALSE).


.bayesmvp_admin_script_path <- function() {

    source_files <- unlist(sapply(sys.frames(), function(frame) {
               value <- frame$ofile
               if (is.null(value)) "" else as.character(value)
           }))
    source_files <- source_files[source_files != ""]

    if (length(source_files) > 0L) {
        return(normalizePath(tail(source_files, 1L), mustWork = TRUE))
    }

    ## not sourced: no script path; the source is then source_root, BAYESMVP_SOURCE_ROOT, or found from the working directory
    ""
}


.bayesmvp_admin_find_source <- function(script_path,
                                        source_root = NULL) {

    env_source_root <- Sys.getenv("BAYESMVP_SOURCE_ROOT", unset = "")
    requested_root <- source_root
    if (is.null(requested_root) && env_source_root != "") {
        requested_root <- env_source_root
    }

    if (!is.null(requested_root) && requested_root != "") {
        requested_root <- normalizePath(requested_root, mustWork = FALSE)
        if (!file.exists(file.path(requested_root, "DESCRIPTION")) ||
            !file.exists(file.path(requested_root, "inst", "BayesMVP", "DESCRIPTION"))) {
            stop(paste0("BAYESMVP_SOURCE_ROOT is not a BayesMVP source package: ", requested_root))
        }
        return(requested_root)
    }

    search_roots <- c(
        if (script_path != "") dirname(script_path) else character(),
        getwd(),
        file.path(path.expand("~"), "Documents", "Work", "PhD_work", "R_packages", "BayesMVP")
    )

    for (search_root in unique(search_roots)) {
        candidate <- normalizePath(search_root, mustWork = FALSE)
        for (iteration in seq_len(8L)) {
            if (file.exists(file.path(candidate, "DESCRIPTION")) &&
                file.exists(file.path(candidate, "inst", "BayesMVP", "DESCRIPTION"))) {
                return(candidate)
            }
            parent <- dirname(candidate)
            if (identical(parent, candidate)) break
            candidate <- parent
        }
    }

    stop(paste0(
        "Could not locate the BayesMVP source package. Source this file from ",
        "R_packages/BayesMVP/inst/examples or set BAYESMVP_SOURCE_ROOT."
    ))
}


.bayesmvp_admin_r_executable <- function() {

    if (.Platform$OS.type == "windows") {
        file.path(R.home("bin"), "R.exe")
    } else {
        file.path(R.home("bin"), "R")
    }
}


.bayesmvp_admin_install_outer <- function(source_root,
                                          lib) {

    status <- system2(
        command = .bayesmvp_admin_r_executable(),
        args = c(
            "CMD",
            "INSTALL",
            paste0("--library=", shQuote(lib)),
            "--no-test-load",
            "--preclean",
            shQuote(source_root)
        )
    )

    if (!identical(as.integer(status), 0L)) {
        stop(paste0("The outer BayesMVP source installation failed with status ", status, "."))
    }

    value_not_printed <- TRUE
}


.bayesmvp_admin_ensure_nicostan <- function(nicostan_source,
                                            nicostan_lib) {
    tryCatch({
        run_at_exit_1 <- FALSE

        if (!dir.exists(nicostan_lib)) {
            dir.create(nicostan_lib, recursive = TRUE, showWarnings = FALSE)
        }

        package_root <- file.path(nicostan_lib, "NicoStan")
        description_file <- file.path(package_root, "DESCRIPTION")
        installed_version <- if (file.exists(description_file)) {
            tryCatch(package_version(read.dcf(description_file, fields = "Version")[[1L]]),
                     error = function(error) package_version("0.0.0"))
        } else {
            package_version("0.0.0")
        }
        native_files <- list.files(file.path(package_root, "libs"),
                                   pattern = paste0("NicoStan", .Platform$dynlib.ext, "$"),
                                   recursive = TRUE)
        compiled_install <- installed_version >= "0.1.9000" &&
            dir.exists(file.path(package_root, "include", "NicoStan")) &&
            length(native_files) > 0L
        required_exports <- c("MVP_model", "initialise_model", "setup_env_post_install")
        if (compiled_install) {
            compiled_install <- requireNamespace("NicoStan", quietly = TRUE, lib.loc = nicostan_lib) &&
                all(required_exports %in% getNamespaceExports("NicoStan"))
        }

        if (!compiled_install) {
            if (is.null(nicostan_source) || !dir.exists(nicostan_source)) {
                stop("A compiled NicoStan >= 0.1.9000 is required; its local source package was not found.")
            }
            ## This worker is already isolated from the user's R session.
            if ("NicoStan" %in% loadedNamespaces()) unloadNamespace("NicoStan")
            message("Installing the missing, outdated or incomplete NicoStan dependency.")
            installer_environment <- list2env(list(), parent = globalenv())
            old_autorun <- getOption("NicoStan.admin.autorun")
            options(NicoStan.admin.autorun = FALSE)
            run_at_exit_1 <- TRUE
            source(file.path(nicostan_source, "inst", "examples", "NicoStan_admin_install.R"), local = installer_environment)
            installer_environment$run_NicoStan_admin_install(source_root = nicostan_source,
                                                              lib = nicostan_lib)
        }

        if (!requireNamespace("NicoStan", quietly = TRUE, lib.loc = nicostan_lib) ||
            !all(required_exports %in% getNamespaceExports("NicoStan"))) {
            stop("The compiled NicoStan dependency could not be loaded or is missing required exports.")
        }

        header_dir <- system.file("include", "NicoStan",
                                  package = "NicoStan",
                                  lib.loc = nicostan_lib)
        if (header_dir == "" || !dir.exists(header_dir)) {
            stop(paste0("NicoStan is installed but its exported headers are missing: ", header_dir))
        }

        version <- utils::packageVersion("NicoStan", lib.loc = nicostan_lib)
        if (version < "0.1.9000") {
            stop(paste0("BayesMVP requires NicoStan >= 0.1.9000; found ", version))
        }

        value_not_printed <- version
    }, finally = if (run_at_exit_1) options(NicoStan.admin.autorun = old_autorun))
}


.bayesmvp_admin_run_clean <- function(arguments) {

    work_dir <- tempfile("BayesMVP_admin_install_")
    dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
    tryCatch({

        config_file <- file.path(work_dir, "install_config.rds")
        child_script <- file.path(work_dir, "install.R")
        saveRDS(list(arguments = arguments,
                     library_paths = .libPaths(),
                     script = file.path(arguments$source_root, "inst", "examples", "BayesMVP_admin_install.R")),
                config_file)
        writeLines(c(
            "config <- readRDS(commandArgs(trailingOnly = TRUE)[[1L]])",
            "options(BayesMVP.admin.autorun = FALSE)",
            ".libPaths(unique(c(config$arguments$lib, config$arguments$nicostan_lib, config$library_paths, .libPaths())))",
            "Sys.setenv(R_LIBS = paste(.libPaths(), collapse = .Platform$path.sep))",
            "source(config$script, local = TRUE)",
            "do.call(.bayesmvp_admin_install_worker, config$arguments)"
        ), child_script)

        rscript <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
        status <- system2(command = rscript,
                          args = c("--vanilla", shQuote(child_script), shQuote(config_file)))
        if (!identical(as.integer(status), 0L)) {
            stop(paste0("BayesMVP installation failed. See the build error above (status ", status, ")."), call. = FALSE)
        }

        value_not_printed <- TRUE
    }, finally = unlink(work_dir, recursive = TRUE, force = TRUE))
}


#' Install NicoStan first, then the compiled BayesMVP extension
#'
#' @param source_root Path to `R_packages/BayesMVP`.
#' @param lib Target BayesMVP library.
#' @param nicostan_lib Library containing the compiled NicoStan dependency.
#'   Defaults to `lib`, unless `NICOSTAN_INSTALL_LIB` is set.
#' @param nicostan_source Optional local `R_packages/NicoStan` source path if
#'   NicoStan is missing, outdated or incomplete in `nicostan_lib`.
#' @param CUSTOM_FLAGS Optional named compiler/Makevars overrides passed to
#'   `BayesMVP::install_BayesMVP`.
#' @return Invisibly, the BayesMVP target library path.
run_BayesMVP_admin_install <- function(source_root = NULL,
                                       lib = NULL,
                                       nicostan_lib = NULL,
                                       nicostan_source = NULL,
                                       CUSTOM_FLAGS = NULL) {

    script_path <- .bayesmvp_admin_script_path()
    source_root <- .bayesmvp_admin_find_source(script_path = script_path,
                                               source_root = source_root)

    env_library <- Sys.getenv("BAYESMVP_INSTALL_LIB", unset = "")
    if (is.null(lib) && env_library != "") lib <- env_library
    if (is.null(lib)) lib <- .libPaths()[1L]

    env_nicostan_library <- Sys.getenv("NICOSTAN_INSTALL_LIB", unset = "")
    if (is.null(nicostan_lib) && env_nicostan_library != "") nicostan_lib <- env_nicostan_library
    if (is.null(nicostan_lib)) nicostan_lib <- lib

    if (is.null(nicostan_source)) {
        candidate <- file.path(dirname(source_root), "NicoStan")
        if (dir.exists(candidate)) nicostan_source <- candidate
    }

    if (!is.character(lib) || length(lib) != 1L || lib == "" ||
        !is.character(nicostan_lib) || length(nicostan_lib) != 1L || nicostan_lib == "") {
        stop("lib and nicostan_lib must each be one non-empty directory path.")
    }

    lib <- normalizePath(lib, mustWork = FALSE)
    nicostan_lib <- normalizePath(nicostan_lib, mustWork = FALSE)
    dir.create(lib, recursive = TRUE, showWarnings = FALSE)
    dir.create(nicostan_lib, recursive = TRUE, showWarnings = FALSE)

    package_was_loaded <- any(c("BayesMVP", "NicoStan") %in% loadedNamespaces())
    message("Installing BayesMVP in a clean R process. Build output follows below.")
    .bayesmvp_admin_run_clean(arguments = list(
        source_root = source_root,
        lib = lib,
        nicostan_lib = nicostan_lib,
        nicostan_source = nicostan_source,
        CUSTOM_FLAGS = CUSTOM_FLAGS
    ))

    message(paste0("BayesMVP installation completed in: ", lib))
    if (package_was_loaded) {
        message("Restart R when ready to use the new build; the current session still has its previous package loaded.")
    }

    value_not_printed <- lib
}


.bayesmvp_admin_install_worker <- function(source_root,
                                          lib,
                                          nicostan_lib,
                                          nicostan_source,
                                          CUSTOM_FLAGS) {

    message(paste0("Checking NicoStan in: ", nicostan_lib))
    .bayesmvp_admin_ensure_nicostan(nicostan_source = nicostan_source,
                                    nicostan_lib = nicostan_lib)

    ## Keep the installer shell out of the target library. R CMD INSTALL can
    ## then preserve the previous compiled package if the new build fails.
    outer_library <- tempfile("BayesMVP_installer_shell_")
    dir.create(outer_library, recursive = TRUE, showWarnings = FALSE)
    tryCatch({
        .bayesmvp_admin_install_outer(source_root = source_root, lib = outer_library)

        library("BayesMVP", lib.loc = outer_library)
        BayesMVP::install_BayesMVP(
            CUSTOM_FLAGS = CUSTOM_FLAGS,
            lib = lib,
            nicostan_lib = nicostan_lib,
            force = TRUE
        )

        value_not_printed <- lib
    }, finally = unlink(outer_library, recursive = TRUE, force = TRUE))
}


if (isTRUE(getOption("BayesMVP.admin.autorun", TRUE))) {
    run_BayesMVP_admin_install()
}






















