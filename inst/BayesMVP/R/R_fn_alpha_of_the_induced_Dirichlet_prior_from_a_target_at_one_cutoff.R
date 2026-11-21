##
## ==== R_fn_alpha_of_the_induced_Dirichlet_prior_from_a_target_at_one_cutoff.R ================================
##
## ---- Priors on ordinal category probabilities, from beliefs at ONE clinical cutoff: the alphas of an induced
##      Dirichlet prior of an ordinal test's category probabilities (one latent class or one disease status), such
##      that the implied Se (present class) or Sp (absent class) at the clinical cutoff has the target mean and the
##      target interval (or the given number of pseudo-cases). The cutoff belief fixes the TOTAL mass on either
##      side; shape is a separate belief: where scores fall WITHIN each side, and therefore how accuracy behaves at
##      the other thresholds. These are NOMINAL Dirichlet factors: a model jointly restricts the categories (e.g.
##      by stochastic ordering, or by the location prior of the cutpoints), so the real induced prior of Se and Sp
##      is read only from the model itself with its likelihood switched off (MetaOrdDTA:
##      run_prior_predictive_check_of_Se_and_Sp(); BayesMVP: a prior-only fit). Used for MetaOrdDTA's
##      informative priors from the literature (the sim study of arXiv 2505.23393, the DIAQANDI network
##      meta-analysis) and the APMS latent class models. ----
##
#' fn_alpha_from_target
#'
#' @description
#' The alphas of an induced Dirichlet prior of an ordinal test's category probabilities, from a target accuracy at
#' one clinical cutoff: Se (for the class or status in which the condition is present) or Sp (absent), its mean
#' and its interval (or a number of pseudo-cases), and where the scores fall within each side of the cutoff.
#'
#' @param n_categories The number of categories of the test (the number of possible scores).
#' @param default_cutoff The clinical cutoff, as a CATEGORY index (category 1 is the lowest possible score).
#' @param target_mean The target mean of Se (latent_class "present") or Sp ("absent") at the cutoff.
#' @param target_interval The target interval of Se or Sp at the cutoff, at CI_level (give exactly one of
#'   target_interval and prior_n_pseudo_cases).
#' @param prior_n_pseudo_cases The concentration (number of pseudo-cases), in place of target_interval.
#' @param shape_centre The category (index) at which the scores are centred; NULL: uniform within each side.
#' @param shape_sd The SD of the shape (shape_join "per_side" only); NULL: each side's length / 3.
#' @param shape_join "per_side" (each side shaped on its own) or "smooth" (one shape over every category, its width
#'   solved for the target).
#' @param shape_kind "bell" or "plateau" ("plateau" only with shape_join "smooth").
#' @param latent_class "present" (the target is Se) or "absent" (the target is Sp).
#' @param alpha_floor The smallest alpha allowed.
#' @param floor_policy "error" (stop when the floor is hit) or "increase_concentration".
#' @param CI_level The level of target_interval.
#' @param verbose Whether to print what was achieved.
#'
#' @return A list: alpha_vector (the alphas), achieved_mean, achieved_interval, n_pseudo_used,
#'   n_pseudo_requested, target_mean, target_interval, CI_level, shape_centre, shape_sd, shape_join, shape_kind,
#'   flatten_weight_below, flatten_weight_above, default_cutoff, latent_class.
#' @export
fn_alpha_from_target <-  function( n_categories,
                                   default_cutoff,
                                   ##
                                   target_mean,
                                   target_interval = NULL,
                                   ##
                                   prior_n_pseudo_cases = NULL,
                                   ##
                                   shape_centre = NULL,
                                   shape_sd = NULL,
                                   ##
                                   shape_join = c("per_side", "smooth"),
                                   shape_kind = c("bell", "plateau"),
                                   ##
                                   latent_class = c("present", "absent"),
                                   ##
                                   alpha_floor = 1.0,
                                   floor_policy = c("error", "increase_concentration"),
                                   ##
                                   CI_level = 0.95,
                                   ##
                                   verbose = TRUE) {

    latent_class <-  match.arg(arg = latent_class)
    ##
    floor_policy <-  match.arg(arg = floor_policy)
    ##
    shape_join <-  match.arg(arg = shape_join)
    shape_kind <-  match.arg(arg = shape_kind)
    ##
    fn_is_finite_scalar <-  function(candidate_value)
        is.numeric(x = candidate_value) && length(x = candidate_value) == 1L && is.finite(x = candidate_value)
    ##
    stopifnot( fn_is_finite_scalar(candidate_value = n_categories),
               n_categories >= 2,
               n_categories == round(x = n_categories),
               fn_is_finite_scalar(candidate_value = default_cutoff),
               default_cutoff == round(x = default_cutoff),
               default_cutoff >= 2,
               default_cutoff <= n_categories,
               fn_is_finite_scalar(candidate_value = target_mean),
               target_mean > 0,
               target_mean < 1,
               fn_is_finite_scalar(candidate_value = alpha_floor),
               alpha_floor > 0,
               fn_is_finite_scalar(candidate_value = CI_level),
               CI_level > 0,
               CI_level < 1)
    ##
    if (!is.null(x = shape_centre)) {
        stopifnot( fn_is_finite_scalar(candidate_value = shape_centre),
                   shape_centre >= 1,
                   shape_centre <= n_categories)
    }
    ##
    ## ---- UNITS. shape_centre is a CATEGORY index, exactly like default_cutoff: category 1 is
    ## raw score 0, so shape_centre = 1 puts the mode at a raw score of 0 and shape_centre = 16 at
    ## a raw score of 15. The kernels below run over raw scores 0:(n_categories - 1), so convert
    ## ONCE here. (This used to be dropped straight into dnorm() over raw scores, i.e. it was
    ## silently a raw score while default_cutoff was a category - every centre landed one score
    ## above where it was put, which is why an absent class with centre 1 peaked at raw score 1.)
    ##
    shape_centre_raw <-  if (is.null(x = shape_centre)) NULL else shape_centre - 1
    ##
    if (shape_kind == "plateau" && shape_join != "smooth")
        stop("shape_kind = 'plateau' is only defined for shape_join = 'smooth'.")
    ##
    if (!is.null(x = shape_sd)) {
        stopifnot( fn_is_finite_scalar(candidate_value = shape_sd),
                   shape_sd > 0)
        ##
        if (is.null(x = shape_centre)) stop("shape_sd requires shape_centre; otherwise the shape is uniform.")
    }
    ##
    ## ---- shape_join = "smooth": ONE shape across all categories instead of one per side.
    ##
    ## "per_side" (the original, and still the default) shapes the below-cutoff and above-cutoff
    ## categories INDEPENDENTLY - each gets its own centre (clamped into that side) and its own SD
    ## (side length / 3) - and then rescales each side to its target mass. That hits the target
    ## Se/Sp exactly, but the two shapes are unrelated, so the alpha vector STEPS at the clinical
    ## cutoff; for an absent-class test it can even rise across the cutoff, which says a
    ## disease-free subject is MORE likely to score just above the threshold than just below it.
    ##
    ## "smooth" instead lays a single normal shape over every category at the requested centre and
    ## SOLVES for its width so that the implied mass split equals the target. The result is
    ## monotone away from the centre with no step, and still hits the target exactly - the price is
    ## that the width is no longer yours to choose (it is what buys the target), and that a target
    ## far from what any single bump can produce is refused rather than forced.
    ##
    if (shape_join == "smooth") {

        if (is.null(x = shape_centre))
            stop("shape_join = 'smooth' requires shape_centre (a uniform shape has no step to remove).")
        ##
        if (!is.null(x = shape_sd))
            stop( "shape_join = 'smooth' SOLVES for the width, so shape_sd cannot also be supplied. ",
                  "Drop shape_sd, or use shape_join = 'per_side'.")

    }
    ##
    if (is.null(x = target_interval) == is.null(x = prior_n_pseudo_cases))
        stop("give exactly ONE of target_interval or prior_n_pseudo_cases.")
    ##
    tail_probabilities <-  c((1 - CI_level) / 2, 1 - (1 - CI_level) / 2)
    ##
    n_categories_above_cutoff <-  n_categories - default_cutoff + 1L
    n_categories_below_cutoff <-  default_cutoff - 1L
    ##
    if (!is.null(x = target_interval)) {

        stopifnot( is.numeric(x = target_interval),
                   length(x = target_interval) == 2L,
                   all(is.finite(x = target_interval)),
                   target_interval[1] > 0,
                   target_interval[1] < target_mean,
                   target_interval[2] > target_mean,
                   target_interval[2] < 1)
        ##
        ## A fixed mean leaves ONE Beta parameter free. Match interval WIDTH, and print the
        ## achieved endpoints: an arbitrary mean and both endpoints cannot all be matched exactly.
        fn_interval_width_gap <-  function(log_concentration) {

            concentration <-  exp(x = log_concentration)
            ##
            diff(x = qbeta( p = tail_probabilities,
                            shape1 = concentration * target_mean,
                            shape2 = concentration * (1 - target_mean))) - diff(x = target_interval)

        }
        ##
        log_concentration_search_range <-  c(log(x = 1e-6), log(x = 1e12))
        ##
        if (fn_interval_width_gap(log_concentration = log_concentration_search_range[1]) * fn_interval_width_gap(
            log_concentration = log_concentration_search_range[2]
        ) > 0)
            stop("target interval width is outside the Beta search range for this mean.")
        ##
        concentration_requested <-  exp(x = uniroot( f = fn_interval_width_gap,
                                                     interval = log_concentration_search_range,
                                                     tol = 1e-10)$root)

    } else {
        stopifnot( fn_is_finite_scalar(candidate_value = prior_n_pseudo_cases),
                   prior_n_pseudo_cases > 0)
        ##
        concentration_requested <-  prior_n_pseudo_cases
    }
    ##
    prior_mass_above_cutoff <-  if (latent_class == "present") target_mean else 1 - target_mean
    ##
    concentration_needed_for_floor <-  alpha_floor * max( n_categories_above_cutoff / prior_mass_above_cutoff,
                                                          n_categories_below_cutoff / (1 - prior_mass_above_cutoff))
    ##
    if (concentration_requested < concentration_needed_for_floor * (1 - 1e-10) && floor_policy == "error")
        stop( sprintf( fmt = paste0( "alpha_floor %.4g needs concentration >= %.3f, but the belief requests %.3f. ",
                                     "Lower alpha_floor, or explicitly choose floor_policy = 'increase_concentration'."),
                       alpha_floor,
                       concentration_needed_for_floor,
                       concentration_requested))
    ##
    concentration_used <-  max( concentration_requested,
                                concentration_needed_for_floor)
    ##
    ## Each side is flattened independently ONLY when needed to respect the chosen floor.
    ## Evaluate the shape in log space so a centre far from a side does not underflow to uniform.
    fn_solve_one_side <-  function( raw_scores_this_side,
                                    total_mass_this_side) {

        if (is.null(x = shape_centre)) within_side_share <-  rep( x = 1 / length(x = raw_scores_this_side),
                                                                  times = length(x = raw_scores_this_side)) else {

            ## NULL retains the original within-side shape. An explicit shape_sd uses
            ## the supplied centre / SD in raw-score units, independently on each side.
            shape_sd_used <-  if (is.null(x = shape_sd)) max( length(x = raw_scores_this_side) / 3,
                                                              0.75) else shape_sd
            shape_centre_used <-  if (is.null(x = shape_sd)) {
                if (max(raw_scores_this_side) < default_cutoff - 1L) min(
                    shape_centre_raw,
                    max(raw_scores_this_side)
                ) else max( shape_centre_raw,
                            min(raw_scores_this_side))
            } else shape_centre_raw
            ##
            log_shape_density <-  dnorm( x = raw_scores_this_side,
                                         mean = shape_centre_used,
                                         sd = shape_sd_used,
                                         log = TRUE)
            ##
            if (!all(is.finite(x = log_shape_density))) stop("shape_centre / shape_sd is outside a numerically usable range.")
            ##
            within_side_share <-  exp(x = log_shape_density - max(log_shape_density))
            within_side_share <-  within_side_share / sum(within_side_share)

        }
        ##
        smallest_alpha_this_side <-  total_mass_this_side * min(within_side_share)
        ##
        uniform_alpha_this_side <-  total_mass_this_side / length(x = raw_scores_this_side)
        ##
        flattening_weight <-  if (smallest_alpha_this_side >= alpha_floor) 0 else if (uniform_alpha_this_side <= alpha_floor * (1 + 1e-10))
            1
        else(alpha_floor - smallest_alpha_this_side) / (uniform_alpha_this_side - smallest_alpha_this_side)
        ##
        list( alpha_vector = total_mass_this_side * ((1 - flattening_weight) * within_side_share + flattening_weight / length(
                  x = raw_scores_this_side
              )),
              flatten = flattening_weight)

    }
    ##
    shape_sd_solved <-  NULL
    ##
    if (shape_join == "smooth") {

        raw_scores_all <-  0:(n_categories - 1L)
        index_above_cutoff <-  default_cutoff:n_categories
        ##
        ## One shape over ALL categories, flattened (globally) only as far as the floor demands.
        ## The mass split is read off AFTER flattening, so the width is solved against the alpha
        ## vector actually returned rather than against the unflattened shape:
        fn_smooth_alpha <-  function(shape_sd_candidate) {

            ## "bell": one normal over every category.
            ## "plateau": the same normal on the side that should DECAY, held at its peak on the
            ## other side - present class rises to the centre then stays flat to the max score (a
            ## case scores at least around the centre; anywhere above is equally plausible); absent
            ## class is flat from raw score 0 up to the centre then decays. The symmetric bell says
            ## a true case is LESS likely to score the maximum than the centre, which for a
            ## reference test is usually the opposite of the belief.
            log_shape_density <-  if (shape_kind == "bell") {
                dnorm( x = raw_scores_all, mean = shape_centre_raw, sd = shape_sd_candidate, log = TRUE)
            } else if (latent_class == "present") {
                ifelse( raw_scores_all < shape_centre_raw,
                        dnorm( x = raw_scores_all,  mean = shape_centre_raw, sd = shape_sd_candidate, log = TRUE),
                        dnorm( x = shape_centre_raw, mean = shape_centre_raw, sd = shape_sd_candidate, log = TRUE))
            } else {
                ifelse( raw_scores_all > shape_centre_raw,
                        dnorm( x = raw_scores_all,  mean = shape_centre_raw, sd = shape_sd_candidate, log = TRUE),
                        dnorm( x = shape_centre_raw, mean = shape_centre_raw, sd = shape_sd_candidate, log = TRUE))
            }
            ##
            if (!all(is.finite(x = log_shape_density))) return(NULL)
            ##
            within_share <-  exp(x = log_shape_density - max(log_shape_density))
            within_share <-  within_share / sum(within_share)
            ##
            smallest_alpha <-  concentration_used * min(within_share)
            uniform_alpha <-  concentration_used / n_categories
            ##
            flattening_weight <-  if (smallest_alpha >= alpha_floor) 0 else if (uniform_alpha <= alpha_floor * (1 + 1e-10))
                1
            else(alpha_floor - smallest_alpha) / (uniform_alpha - smallest_alpha)
            ##
            alpha_candidate <-  concentration_used * ((1 - flattening_weight) * within_share +
                                                      flattening_weight / n_categories)
            ##
            list( alpha_vector = alpha_candidate,
                  flatten = flattening_weight,
                  mass_above = sum(alpha_candidate[index_above_cutoff]) / concentration_used)

        }
        ##
        fn_smooth_gap <-  function(log_shape_sd) {

            candidate <-  fn_smooth_alpha(shape_sd_candidate = exp(x = log_shape_sd))
            ##
            if (is.null(x = candidate)) return(NA_real_)
            ##
            candidate$mass_above - prior_mass_above_cutoff

        }
        ##
        log_shape_sd_grid <-  seq( from = log(x = 0.05),
                                   to = log(x = 10 * n_categories),
                                   length.out = 400L)
        ##
        gap_grid <-  vapply( X = log_shape_sd_grid,
                             FUN = fn_smooth_gap,
                             FUN.VALUE = numeric(length = 1))
        ##
        usable <-  which(is.finite(x = gap_grid))
        ##
        sign_change <-  if (length(x = usable) >= 2L)
            which(gap_grid[usable[-length(x = usable)]] * gap_grid[usable[-1]] <= 0)[1] else NA_integer_
        ##
        if (is.na(x = sign_change)) stop( sprintf( fmt = paste0(
                  "shape_join = 'smooth': no width reproduces %s = %.4f with centre category %.2f (raw score %.2f) over %d categories ",
                  "at alpha_floor %.4g (attainable mass above the cutoff spans [%.4f, %.4f]). Move shape_centre ",
                  "towards the cutoff, lower alpha_floor, raise the concentration, or use shape_join = 'per_side'."),
                  if (latent_class == "present") "Se" else "Sp",
                  target_mean,
                  shape_centre,
                  shape_centre_raw,
                  n_categories,
                  alpha_floor,
                  min(gap_grid[usable]) + prior_mass_above_cutoff,
                  max(gap_grid[usable]) + prior_mass_above_cutoff))
        ##
        shape_sd_solved <-  exp(x = uniroot( f = fn_smooth_gap,
                                             interval = c( log_shape_sd_grid[usable[sign_change]],
                                                           log_shape_sd_grid[usable[sign_change + 1]]),
                                             tol = 1e-12)$root)
        ##
        smooth_solution <-  fn_smooth_alpha(shape_sd_candidate = shape_sd_solved)
        ##
        alpha_vector <-  smooth_solution$alpha_vector
        ##
        ## reported per-side for the printout, but the flattening is global here:
        below_side <-  list(flatten = smooth_solution$flatten)
        above_side <-  list(flatten = smooth_solution$flatten)

    } else {

        below_side <-  fn_solve_one_side( raw_scores_this_side = 0:(n_categories_below_cutoff - 1L),
                                          total_mass_this_side = concentration_used * (1 - prior_mass_above_cutoff))
        ##
        above_side <-  fn_solve_one_side( raw_scores_this_side = n_categories_below_cutoff:(n_categories - 1L),
                                          total_mass_this_side = concentration_used * prior_mass_above_cutoff)
        ##
        alpha_vector <-  c(below_side$alpha_vector, above_side$alpha_vector)

    }
    alpha_mass_above_cutoff <-  sum(alpha_vector[default_cutoff:n_categories])
    alpha_mass_below_cutoff <-  sum(alpha_vector[seq_len(length.out = n_categories_below_cutoff)])
    alpha_reported_side <-  if (latent_class == "present") alpha_mass_above_cutoff else alpha_mass_below_cutoff
    alpha_complement_side <-  if (latent_class == "present") alpha_mass_below_cutoff else alpha_mass_above_cutoff
    ##
    achieved_interval <-  qbeta( p = tail_probabilities,
                                 shape1 = alpha_reported_side,
                                 shape2 = alpha_complement_side)
    ##
    quantity_label <-  if (latent_class == "present") "Se" else "Sp"
    ##
    if (verbose) {

        cat( sprintf( fmt = "\nfn_alpha_from_target: %s (%s class), %d categories, raw-score cutoff >= %d\n",
                      quantity_label,
                      latent_class,
                      n_categories,
                      default_cutoff - 1L))
        message( (sprintf( fmt = "  nominal %s ~ Beta(%.3f, %.3f): mean %.3f, %.0f%% [%.3f, %.3f], concentration %.3f\n",
                           quantity_label,
                           alpha_reported_side,
                           alpha_complement_side,
                           alpha_reported_side / (alpha_reported_side + alpha_complement_side),
                           100 * CI_level,
                           achieved_interval[1],
                           achieved_interval[2],
                           concentration_used)))
        ##
        if (!is.null(x = target_interval)) cat( sprintf( fmt =
                            "  requested interval [%.3f, %.3f]; width matched before any floor adjustment\n",
                                                         target_interval[1],
                                                         target_interval[2]))
        ##
        cat( sprintf( fmt = "  shape: %s; flattening %.1f%% below / %.1f%% above; min alpha %.4g\n",
                      if (shape_join == "smooth") sprintf( fmt = "centre category %.2f = raw score %.2f, SINGLE smooth %s, solved sd %.3f",
                                                          shape_centre,
                                                          shape_centre_raw,
                                                          shape_kind,
                                                          shape_sd_solved) else
                      if (is.null(x = shape_centre)) "uniform within each side" else if (is.null(x = shape_sd))
                         sprintf( fmt = "centre %.2f, original within-side SDs",
                                  shape_centre) else sprintf( fmt = "centre %.2f, sd %.2f",
                                                              shape_centre,
                                                              shape_sd),
                      100 * below_side$flatten,
                      100 * above_side$flatten,
                      min(alpha_vector)))
        ##
        if (concentration_used > concentration_requested * 1.001)
            cat( sprintf( fmt = "  EXPLICIT floor adjustment: concentration %.3f -> %.3f; prior strengthened.\n",
                          concentration_requested,
                          concentration_used))
        ##
        cat("  actual Stan prior also includes stochastic ordering; inspect the prior-only report.\n")

    }
    ##
    list( alpha_vector = alpha_vector,
          achieved_mean = alpha_reported_side / (alpha_reported_side + alpha_complement_side),
          achieved_interval = achieved_interval,
          n_pseudo_used = concentration_used,
          n_pseudo_requested = concentration_requested,
          target_mean = target_mean,
          target_interval = target_interval,
          CI_level = CI_level,
          shape_centre = shape_centre,
          shape_sd = if (shape_join == "smooth") shape_sd_solved else shape_sd,
          shape_join = shape_join,
          shape_kind = shape_kind,
          flatten_weight_below = below_side$flatten,
          flatten_weight_above = above_side$flatten,
          default_cutoff = default_cutoff,
          latent_class = latent_class)

}






















