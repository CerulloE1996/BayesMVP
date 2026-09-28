 
cat("Setting up and checking R Session Environment:\n")

options(devtools.install.args = c("--no-test-load"))

# setwd("C:\\Users\\enzoc\\Documents")
# 
#  append_to_path <- "C:\\Windows;C:\\Windows\\System32\\wbem;C:\\Windows\\System32\\WindowsPowerShell\\v1.0;C:\\Windows\\System32\\OpenSSH;C:\\Program Files (x86)\\NVIDIA Corporation\\PhysX\\Common;C:\\Program Files\\NVIDIA Corporation\\NVIDIA NvDLISR;C:\\Windows\\System32;C:\\Windows;C:\\Windows\\System32\\wbem;C:\\Windows\\System32\\WindowsPowerShell\\v1.0;C:\\Windows\\System32\\OpenSSH;C:\\Program Files\\Git\\cmd;C:\\rtools44\\x86_64-w64-mingw32.static.posix\\bin;C:\\rtools44\\usr\\bin;C:\\usr\\bin;C:\\usr\\bin;C:\\Windows\\System32;C:\\Windows;C:\\Windows\\System32\\wbem;C:\\Windows\\System32\\WindowsPowerShell\\v1.0;C:\\Windows\\System32\\OpenSSH;C:\\Program Files (x86)\\NVIDIA Corporation\\PhysX\\Common;C:\\Program Files\\NVIDIA Corporation\\NVIDIA NvDLISR;C:\\Windows\\System32;C:\\Windows;C:\\Windows\\System32\\wbem;C:\\Windows\\System32\\WindowsPowerShell\\v1.0;C:\\Windows\\System32\\OpenSSH;C:\\Program Files\\Git\\cmd;C:\\Users\\enzoc\\AppData\\Local\\Microsoft\\WindowsApps;C:\\Program Files\\RStudio\\resources\\app\\bin\\quarto\\bin;C:\\Program Files\\RStudio\\resources\\app\\bin\\postback;C:\\Program Files\\Git\\bin;C:\\Program Files\\RStudio\\resources\\app\\bin\\quarto\\bin;C:\\Program Files\\RStudio\\resources\\app\\bin\\postback"
# 
# # Update PATH
# Sys.setenv(PATH = paste(
#   "C:\\rtools44\\x86_64-w64-mingw32.static.posix\\bin",
#   "C:\\rtools44\\usr\\bin",
#   "C:\\usr\\bin",
#   "C:\\Windows\\System32",
#   "C:\\Program Files\\R\\R-4.4.2\\bin\\x64",
#   "C:\\Users\\enzoc\\.bridgestan/bridgestan-2.5.0/stan/lib/stan_math/lib/tbb",
#   "C:/Users/enzoc/.bridgestan/bridgestan-2.5.0/stan/lib/stan_math/lib/tbb_2020.3/include",
#   Sys.getenv("PATH"),
#   sep = ";"
# ))
# 
# Sys.setenv(PATH = paste(append_to_path,   Sys.getenv("PATH"),  sep = ";"))

cat("  PATH: ", Sys.getenv("PATH"), "\n")
cat("  libPaths: ", paste(.libPaths(), collapse = "; "), "\n")
cat("  Working Directory: ", getwd(), "\n")

cat("Preloading critical DLLs for BayesMVP package\n")

Sys.getenv("BRIDGESTAN")
Sys.setenv(BRIDGESTAN="C:/Users/enzoc/.bridgestan/bridgestan-2.5.0")

# List of DLLs to preload
dll_paths <- c(
 # "C:/Users/enzoc/Documents/BayesMVP/inst/tbb12.dll",
  "C:/Users/enzoc/Documents/BayesMVP/inst/tbb.dll",
  "C:/Users/enzoc/Documents/BayesMVP/inst/dummy_stan_model_win_model.so",
  "C:/Users/enzoc/Documents/BayesMVP/inst/dummy_stan_model_win_model.dll",
  "C:/Users/enzoc/Documents/BayesMVP/inst/BayesMVP.dll"
)

# Attempt to load each DLL
for (dll in dll_paths) {
  
      tryCatch(
        {
          dyn.load(dll)
          cat("  Loaded:", dll, "\n")
        },
        error = function(e) {
          cat("  Failed to load:", dll, "\n  Error:", e$message, "\n")
        }
      )
  
}




 

















