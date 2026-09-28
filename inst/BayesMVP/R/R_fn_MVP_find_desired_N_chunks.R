

#' find_num_chunks_MVP
#' @keywords internal
#' @export
find_num_chunks_MVP  <- function(N, 
                                 n_tests) {


         num_chunks <- 1
         n_obs <- N * n_tests
        
      
        if (parallel::detectCores() < 33) { # (most) laptops / consumer-level computer's (up to 32 threads e.g. even a niche 16-core AMD laptop w/ SMT would be included here)
              
                  if (n_obs <= 2500)              num_chunks <-    1    # e.g. if 5 tests then N = 500
                  if (n_obs %in% c(2501:5000))    num_chunks <-    2    # e.g. if 5 tests then N = 1000
                  if (n_obs %in% c(5001:12500))   num_chunks <-    5    # e.g. if 5 tests then N = 2500
                  if (n_obs %in% c(12501:25000))   num_chunks <-   10   # e.g. if 5 tests then N = 5000
                  if (n_obs %in% c(25001:62500))   num_chunks <-   25   # e.g. if 5 tests then N = 12,500 
                  if (n_obs > 62500)   num_chunks <-               50   # e.g. if 5 tests then N = 25,000
                  if (n_obs > 125000)  num_chunks <-  125 # untested
                  if (n_obs > 250000)  num_chunks <-  250 # untested
                  if (n_obs > 500000)  num_chunks <-  500 # untested
              
        } else { # HPC's / server's 
          
                  if (n_obs <= 2500)               num_chunks <-   1
                  if (n_obs %in% c(2501:5000))     num_chunks <-   2
                  if (n_obs %in% c(5001:12500))    num_chunks <-   4
                  if (n_obs %in% c(12501:25000))   num_chunks <-   5
                  if (n_obs %in% c(25001:62500))   num_chunks <-   40  
                  if (n_obs > 62500)   num_chunks <-               25
                  if (n_obs > 125000)  num_chunks <-  125 # untested
                  if (n_obs > 250000)  num_chunks <-  250 # untested
                  if (n_obs > 500000)  num_chunks <-  500 # untested
              
        }
         
         
         return(num_chunks)
      
  
}






















