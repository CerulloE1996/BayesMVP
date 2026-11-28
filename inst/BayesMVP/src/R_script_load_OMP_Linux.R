


get_OMP <- function() {
  
        omp_path_1 <- "/usr/lib/llvm-14/lib/libgomp.so"
        omp_path_2 <- " " ##"/usr/lib/x86_64-linux-gnu/libomp.so.5"
        omp_path_3 <- " " ##"/usr/lib/llvm-14/lib/libomp.so.5"
        omp_path_4 <- " " ##"/opt/AMD/aocc-compiler-5.0.0/lib/libomp.so"
        omp_path_5 <- "/usr/lib/gcc/x86_64-linux-gnu/12/libgomp.so"
        omp_path_6 <- "/usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so"
        ##
        omp_path_vec <- c(omp_path_1, omp_path_2, 
                          omp_path_3, omp_path_4,
                          omp_path_5, omp_path_6)
        
        counter <- 0
        found <- 0
        for (i in 1:length(omp_path_vec)) {
              if (file.exists(omp_path_vec[i])) {
                  counter <- counter + 1
                  if (counter == 1) {
                        if (found == 0) {
                            found <- 1
                            ## current_ld_path <- Sys.getenv("LD_LIBRARY_PATH")
                            ## new_ld_path <- omp_path_vec[i]
                            # new_ld_path <- if (current_ld_path == "") omp_path_vec[i] else paste0(omp_path_vec[i], ":", current_ld_path)
                            ##Sys.setenv(LD_LIBRARY_PATH = new_ld_path)
                            return(as.character(omp_path_vec[i]))
                        }
                  }
              }
        }
        
        ## (none of the paths above: the libgomp of the C++ compiler R builds with; else -fopenmp, which links it)
        CXX_of_R <-  tryCatch(system2(file.path(R.home("bin"), "R"),
                                      args = c("CMD", "config", "CXX"),
                                      stdout = TRUE,
                                      stderr = FALSE),
                              error = function(e) character(0))
        words_of_CXX <-  strsplit(trimws(paste(CXX_of_R, collapse = " ")), " +")[[1]]
        words_of_CXX <-  words_of_CXX[!grepl("ccache$", words_of_CXX)]
        if (length(words_of_CXX) > 0) {
              omp_path_of_the_compiler <-  tryCatch(system2(words_of_CXX[1],
                                                            args = "-print-file-name=libgomp.so",
                                                            stdout = TRUE,
                                                            stderr = FALSE),
                                                    error = function(e) character(0))
              if ((length(omp_path_of_the_compiler) == 1) && (dirname(omp_path_of_the_compiler) != ".") &&
                  file.exists(omp_path_of_the_compiler)) {
                    return(as.character(omp_path_of_the_compiler))
              }
        }

        return("-fopenmp")   ## no libgomp file found: the compiler's own OpenMP flag, which links its library

}

## the only output make reads: one line, the library path or the flag; never an auto-printed value such as "[1] 0",
## which went into PKG_LIBS and broke the link on a machine without the paths above
writeLines(get_OMP())

## writeLines(as.character(Sys.getenv("LD_LIBRARY_PATH")))






















