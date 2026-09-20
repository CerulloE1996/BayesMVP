##
## ---- summarise_decomposition.R -----------------------------------------------------------------------------------------------
##
## Usage:   Rscript summarise_decomposition.R <output_root_dir>
##
## Prints one row per run and, per arm, mean [min, max] over seeds. ESS/grad is x 1000 (the scale ps7
## prints; your historical level at this configuration is ~15-20). Also writes
## <output_root_dir>/decomposition_summary.csv. The three saved 09-17 runs are included as the reference.
##
{
      output_root_dir <- commandArgs(trailingOnly = TRUE)[1]
      if (is.na(output_root_dir) || !dir.exists(output_root_dir)) stop("usage: Rscript summarise_decomposition.R <output_root_dir>")
      ##
      reference_runs_dir <- "/home/enzocerullo/Documents/Work/PhD_work/Alg_paper_analysis/1_appendix_pilot_studies/ps_7_basic_MCMC_settings_BayesMVP/outputs/DGM_3/17th_Sept_2026_backup"
      reference_run_files <- Sys.glob(file.path(reference_runs_dir,
                                                "ps7_run_LC_MVP_N10000_np2_dH1_pH0_cb4_s180_wb10_s1_kb10_s25_b250_it100_LR.05_AD.8_clip50*_run[123]"))
      ##
      run_files_table <- data.frame(arm_name = character(0), run_file_path = character(0), stringsAsFactors = FALSE)
      if (length(reference_run_files) > 0) {
        run_files_table <- data.frame(arm_name = "09-17 saved", run_file_path = reference_run_files, stringsAsFactors = FALSE)
      } else {
        message("note: the saved 09-17 reference runs were not found in ", reference_runs_dir, " - summarising the new runs only")
      }
      ##
      ## ---- one sub-folder per "<arm_label>_seed<k>". Each run writes run_record.rds naming the ONE file it
      ##      produced; only that file is used. (Folders from before run records existed fall back to their
      ##      ps7_run_* files, with a warning if there is more than one.)
      ##
      for (arm_seed_dir in list.dirs(output_root_dir, recursive = FALSE)) {
            arm_label_of_dir <- sub(pattern = "_seed[0-9]+$", replacement = "", x = basename(arm_seed_dir))
            if (identical(arm_label_of_dir, basename(arm_seed_dir))) next     ## not an "<arm>_seed<k>" folder (e.g. logs)
            run_record_path <- file.path(arm_seed_dir, "run_record.rds")
            if (file.exists(run_record_path)) {
              run_file_paths_in_dir <- readRDS(run_record_path)$run_file_path
            } else {
              run_file_paths_in_dir <- Sys.glob(file.path(arm_seed_dir, "ps7_run_*"))
              if (length(run_file_paths_in_dir) > 1) {
                warning(arm_seed_dir, " has ", length(run_file_paths_in_dir), " ps7_run_* files and no run_record.rds - ",
                        "they may be DIFFERENT configurations; all are included under '", arm_label_of_dir, "'.", call. = FALSE)
              }
            }
            if (length(run_file_paths_in_dir) > 0) {
              run_files_table <- rbind(run_files_table,
                                       data.frame(arm_name = arm_label_of_dir, run_file_path = run_file_paths_in_dir, stringsAsFactors = FALSE))
            }
      }
      if (nrow(run_files_table) == 0) stop("no runs found under ", output_root_dir)
}
##
## ---- one row per run:
##
{
      per_run_rows <- lapply(seq_len(nrow(run_files_table)), function(run_row_index) {
            run_object <- readRDS(run_files_table$run_file_path[run_row_index])
            data.frame( arm_name           = run_files_table$arm_name[run_row_index],
                        MCMC_seed          = run_object$MCMC_seed,
                        ESS_per_grad_x1000 = 1000 * run_object$ESS_per_grad_samp,
                        ESS_per_sec        = run_object$ESS_per_sec_samp,
                        min_ESS            = run_object$min_ESS,
                        max_Rhat           = run_object$max_Rhat,
                        max_Rhat_main      = run_object$efficiency_info$Max_rhat_main,
                        pct_divergences    = run_object$divergences$pct_divs,
                        eps_main           = run_object$HMC_info$eps_main,
                        L_main_sampling    = run_object$efficiency_info$L_main_during_sampling,
                        grad_evals_per_sec = run_object$efficiency_info$grad_evals_per_sec,
                        test_perm          = paste(run_object$HMC_info$test_perm, collapse = ""),
                        stringsAsFactors   = FALSE)
      })
      all_runs_table <- do.call(rbind, per_run_rows)
      ##
      arm_display_order <- unique(c("09-17 saved", "D", "D0", "E0", "F", "G", "H", "A0", "B0", all_runs_table$arm_name))
      all_runs_table <- all_runs_table[order(match(all_runs_table$arm_name, arm_display_order), all_runs_table$MCMC_seed), ]
      write.csv(all_runs_table, file.path(output_root_dir, "decomposition_summary.csv"), row.names = FALSE)
      ##
      cat("\n==== per run ====\n")
      print(transform( all_runs_table,
                       ESS_per_grad_x1000 = round(ESS_per_grad_x1000, 1),
                       ESS_per_sec        = round(ESS_per_sec),
                       min_ESS            = round(min_ESS),
                       max_Rhat           = round(max_Rhat, 3),
                       max_Rhat_main      = round(max_Rhat_main, 2),
                       pct_divergences    = round(pct_divergences, 2),
                       eps_main           = round(eps_main, 3),
                       L_main_sampling    = round(L_main_sampling, 1),
                       grad_evals_per_sec = round(grad_evals_per_sec)),
            row.names = FALSE)
}
##
## ---- per arm: mean [min, max] over seeds:
##
{
      fn_format_mean_min_max <- function(values, n_decimals) {
            sprintf(paste0("%.", n_decimals, "f [%.", n_decimals, "f, %.", n_decimals, "f]"), mean(values), min(values), max(values))
      }
      ##
      cat("\n==== per arm: mean [min, max] over seeds ====\n")
      for (arm_name in unique(all_runs_table$arm_name)) {
            runs_of_this_arm <- all_runs_table[all_runs_table$arm_name == arm_name, ]
            cat(sprintf("%-12s n=%d | ESS/grad %-19s | ESS/s %-17s | min ESS %-19s | max Rhat %-21s | eps %-21s | L %s\n",
                        arm_name,
                        nrow(runs_of_this_arm),
                        fn_format_mean_min_max(runs_of_this_arm$ESS_per_grad_x1000, 1),
                        fn_format_mean_min_max(runs_of_this_arm$ESS_per_sec, 0),
                        fn_format_mean_min_max(runs_of_this_arm$min_ESS, 0),
                        fn_format_mean_min_max(runs_of_this_arm$max_Rhat, 3),
                        fn_format_mean_min_max(runs_of_this_arm$eps_main, 3),
                        fn_format_mean_min_max(runs_of_this_arm$L_main_sampling, 1)))
      }
}
