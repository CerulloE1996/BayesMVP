

.fg_colours <- c(
  "black" = "0;30",
  "blue" = "0;34",
  "green" = "0;32",
  "cyan" = "0;36",
  "red" = "0;31",
  "purple" = "0;35",
  "brown" = "0;33",
  "light gray" = "0;37",
  "dark gray" = "1;30",
  "light blue" = "1;34",
  "light green" = "1;32",
  "light cyan" = "1;36",
  "light red" = "1;31",
  "light purple" = "1;35",
  "yellow" = "1;33",
  "white" = "1;37"
)

.bg_colours <- c(
  "black" = "40",
  "red" = "41",
  "green" = "42",
  "brown" = "43",
  "blue" = "44",
  "purple" = "45",
  "cyan" = "46",
  "light gray" = "47"
)



#' rcmd_running
#' @export
rcmd_running <- function() {
  nchar(Sys.getenv('R_TESTS')) != 0
}



#' colourise
#' @export
colourise <- function(text, fg = "black", bg = NULL) {
  
  term <- Sys.getenv()["TERM"]
  colour_terms <- c("xterm-color","xterm-256color", "screen", "screen-256color")
  
  if(rcmd_running() || !any(term %in% colour_terms, na.rm = TRUE)) {
    return(text)
  }
  
  col_escape <- function(col) {
    paste0("\033[", col, "m")
  }
  
  col <- .fg_colours[tolower(fg)]
  if (!is.null(bg)) {
    col <- paste0(col, .bg_colours[tolower(bg)], sep = ";")
  }
  
  init <- col_escape(col)
  reset <- col_escape("0")
  paste0(init, text, reset)
  
}






