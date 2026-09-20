
require(R6)

#' MVP Model Class
#' @title MVP Model Class
#' @description R6 Class for MVP model initialization and sampling.
#'
#' @details 
#' This class handles:
#' * Model initialization and compilation
#' * MCMC sampling with adaptive diffusion-pathspace HMC.
#' * Parameter updates and diagnostics
#' * Support for both built-in models (MVP, LC-MVP, latent_trait) and user-supplied Stan models
#'
#' @section Model Types:
#' * MVP: Multivariate probit model
#' * LC-MVP: Latent class multivariate probit model
#' * latent_trait: Latent trait model
#' * Stan: User-supplied Stan models (files extension must be .stan). 
#'
#' 
#' @section Typical workflow:
#'  \preformatted{
#'  #### NOTE: Please see the ".R" example files for full details & examples, 
#'  #### as this is NOT a complete working example, but is provided just to show 
#'  #### what order you should call things in. 
#'  ####  -----------  Compile + initialise the model using "MVP_model$new(...)"  ---------------- 
#'    require(BayesMVP)
#'    ## NOTE: The "model_args_list" argument is only needed for BUILT-IN (not Stan) models.
#'    model_obj <- BayesMVP::MVP_model$new(   
#'                    Model_type = Model_type,
#'                    y = y,
#'                    N = N,
#'                    model_args_list = model_args_list,
#'                    init_lists_per_chain = init_lists_per_chain,
#'                    sample_nuisance = TRUE,
#'                    n_chains_burnin = n_chains_burnin,
#'                    n_params_main = n_params_main,
#'                    n_nuisance = n_nuisance)  
#'
#'    #### --- SAMPLE MODEL: --------------------------------------------------------------------
#'    model_samples <-  model_obj$sample( 
#'                    partitioned_HMC = FALSE,
#'                    diffusion_HMC = FALSE,
#'                    seed = 123,
#'                    n_burnin = 500,
#'                    n_iter = 1000,
#'                    y = y,
#'                    N = N,
#'                    n_params_main = n_params_main,
#'                    n_nuisance = n_nuisance,
#'                    init_lists_per_chain = init_lists_per_chain,
#'                    n_chains_burnin = 8,
#'                    n_chains_sampling = 16,
#'                    model_args_list = model_args_list,
#'                    adapt_delta = 0.80,
#'                    learning_rate = 0.05)
#'    
#'    #### --- EXTRACT MODEL RESULTS SUMMARY + DIAGNOSTICS ---------------------------------------           
#'    model_fit <- model_samples$summary(
#'                    save_log_lik_trace = FALSE, 
#'                    compute_nested_rhat = FALSE,
#'                    compute_transformed_parameters = FALSE) 
#'    
#'    
#'    ## extract # divergences + % of sampling iterations which have divergences
#'    model_fit$get_divergences()
#'    
#'    #### --- TRACE PLOTS  ----------------------------------------------------------------------
#'    ## trace_plots_all <- model_samples$plot_traces() # if want the trace for all parameters 
#'    trace_plots <- model_fit$plot_traces(
#'                    params = c("beta", "Omega", "p"), 
#'                    batch_size = 12)
#'    
#'    ## you can extract parameters by doing: "trace$param_name()". 
#'    ## For example:
#'    ## display each panel for beta and Omega ("batch_size" controls the # of plots per panel)
#'    trace_plots$beta[[1]] # 1st (and only) panel
#'    
#'    
#'    #### --- POSTERIOR DENSITY PLOTS -----------------------------------------------------------
#'    ## density_plots_all <- model_samples$plot_densities() # if want the densities 
#'    ## for all parameters.
#'    ## Let's plot the densities for: sensitivity, specificity, and prevalence 
#'    density_plots <- model_fit$plot_densities(
#'                      params = c("Se_bin", "Sp_bin", "p"), 
#'                      batch_size = 12)
#'    
#'    ## you can extract parameters by doing: "trace$param_name()". 
#'    ## For example:
#'    ## display each panel for beta and Omega ("batch_size" controls the # of plots per panel)
#'    density_plots$Se[[1]] # Se - 1st (and only) panel
#'    density_plots$Sp[[1]] # Sp - 1st (and only) panel
#'    density_plots$p[[1]] # p (prevelance) - 1st (and only) panel
#'    
#'    
#'    ###### --- EXTRACT PARAMETER SUMMARY TIBBLES: ----------------------------------------------
#'    ## The "model_summary" object (created using the "$summary()" method) contains
#'    ##  many useful objects. 
#'    ## For example:
#'    require(dplyr)
#'    ## nice summary tibble for main parameters, includes ESS/Rhat, etc:
#'    model_fit$get_summary_main() %>% print(n = 50) 
#'    ## nice summary tibble for transformed parameters, includes ESS/Rhat, etc:
#'    model_fit$get_summary_transformed() %>% print(n = 150) 
#'    ## nice summary tibble for generated quantities, includes 
#'    ## ESS/Rhat, etc (for LC-MVP this includes Se/Sp/prevalence):
#'    model_fit$get_summary_generated_quantities () %>% print(n = 150) 
#'    
#' }
#' 
#' 
#'
#' @export
MVP_model <- R6Class("MVP_model",
                    
            public = list(
              
                          ## Store all important parameters as class members / define them first
                          #'@field Model_type Type of model to fit (can be one of: "MVP", "LC_MVP", "latent_trait" for built-in models, or "Stan" for fitting any Stan model via BayesMVP).
                          Model_type = NULL, 
                          #'@field init_object Initialization object.
                          init_object = NULL,
                          #'@field is_compiled Whether the model has been compiled
                          is_compiled = NULL,
                          #'@field n_nuisance_override Override for number of nuisance parameters
                          n_nuisance_override = NULL,
                          #'@field stanc_args Arguments for Stan compiler (for BridgeStan)
                          stanc_args = NULL,
                          #'@field make_args C++/make arguments used to compile the generated C++ of the Stan model (e.g. list("STAN_THREADS=true")); distinct from stanc_args.
                          make_args = NULL,
                          #'@field Stan_cpp_user_header User C++ header file path
                          Stan_cpp_user_header = NULL,
                          #'@field Stan_cpp_flags C++ compiler flags
                          Stan_cpp_flags = NULL,
                          #'@field Stan_model_file_path Path to Stan model file
                          Stan_model_file_path = NULL,
                          #'@field result Model results object.
                          result = NULL, 
                          #'@field model_fit_object Model fit object.
                          model_fit_object = NULL,
                          #'@field y The dataset. Note that for built-in models (i.e., MVP, LC_MVP, or latent_trait) y should be an  N x n_outcomes matrix. 
                          y = NULL,
                          #'@field N The total sample size. Note that for built-in models (i.e., MVP, LC_MVP, or latent_trait) this should be equal to: \eqn{N = nrow(y)}.
                          N = NULL,
                          #'@field n_params_main The total number of main model parameters (i.e., excluding nuisance parameters / high-dimensional latent variables). For Stan models (i.e., if "Model_type" is set to "Stan" and you 
                          #'are running a Stan model), this should equal the number of parameters that you define in the "parameters" block of the Stan model, EXCEPT for any nuisance parameters (i.e., high-dimensional latent
                          #' variables). 
                          n_params_main = NULL,
                          #'@field n_nuisance The total number of nuisance parameters, i.e. the dimension of the high-dimensional latent variable vector. For built-in models (i.e., MVP, LC_MVP, or latent_trait), this should be 
                          #'equal to: \eqn{n_nuisance = N \cdot n_outcomes = nrow(y) \cdot ncol(y) }.
                          n_nuisance = NULL,
                          #'@field init_lists_per_chain A list of dimension n_chains_burnin (NOT n_chains_sampling, unless n_chains_sampling = n_chains_burnin), where each element of the list is another list which contains the #
                          #'initial values for the chain. Note that this is the same format for initial values that Stan (both rstan and cmdstanr) uses. 
                          init_lists_per_chain = NULL,
                          #'@field model_args_list List containing model-specific arguments. This is only relevant for built-in models (i.e., MVP, LC_MVP, or latent_trait).
                          #'All arguments in this list are optional.
                          #' 
                          #'Arguments which are relevant to all three of the built-in models include:
                          #' 
                          #'* \code{n_covariates_per_outcome}: 
                          #'   A matrix of dimension \code{n_class} x \code{n_outcomes},  which contains the number of covariates per outcome. If your model has no covariates (which is often the case for the LC_MVP or
                          #'   latent_trait), this does not need to be included, and it just be a matrix of 1's (since each outcome has 1 intercept). Also note that n_class = 1 for the MVP, and for the LC_MVP and latent_trait 
                          #'   n_class = 2. 
                          #'   
                          #'* \code{prior_coeffs_mean}: 
                          #'   An array of dimension \code{n_class} x \code{n_outcomes} x \code{n_covariates_max}, where \code{n_covariates_max} is equal to the number of covariates which are contained in the outcome which has the 
                          #'   most covariates. This matrix contains the prior mean of each coefficient parameter. In other words, if: \eqn{\beta_{c, t, k} \sim \text{N}\left(\mu_{c, t, k}, \sigma_{c, t, k} \right)}, 
                          #'   then element \eqn{(c, t, k)} in the matrix corresponds to \eqn{\mu_{c, t, k}},  where \eqn{c} is an index for class, \eqn{t} is an index for outcome, and \eqn{k} is an index for the coefficient for 
                          #'   outcome \eqn{t}. The default is an array of zeros.  
                          #'   
                          #'* \code{prior_coeffs_sd}:    
                          #'   This is the same as  \code{prior_coeffs_mean}, except that each element in thr array corresponds to the prior SD. In other words, each element in the array corresponds to:  \eqn{ \sigma_{c, t, k} },
                          #'   where: \eqn{\beta_{c, t, k} \sim \text{N}\left(\mu_{c, t, k}, \sigma_{c, t, k} \right)}. The default is an array of ones. 
                          #'   
                          #'* \code{vect_type}:
                          #'   The SIMD (single-instruction, multiple-data) vectorisation type to use for math functions (such as log, exp, Phi, etc). The default is AVX-512 if available, then AVX2 if not, and if neither AVX-512 
                          #'   nor AVX2 are available (e.g., on ARM-based CPU's such as Apple systems), then BayesMVP will use the Stan C++ math library functions, and they will decide what vectorisation type to use. More 
                          #'   vectorisation types will be supported in the future.  
                          #'   
                          #'*  \code{num_chunks}:
                          #'    The number of chunks to use in the log-probability and gradient function. By default, the number selected will depend on your CPU. 
                          #'    
                          #'*  \code{Phi_type}:
                          #'    Type of \eqn{\Phi()} function implementation to use, where \eqn{\Phi()} is the standard normal CDF. Can be either "Phi" (the default) or "Phi_approx". The default is "Phi". Note that \eqn{\Phi()} 
                          #'    will use a fast, highly accurate polynomial approximation of \eqn{\Phi()} if \code{vect_type} is either AVX-512 or AVX2. Otherwise, it will use the Phi function from the Stan math C++ library. 
                          #'    
                          #' Arguments which are only relevant to the MVP and LC_MVP include:
                          #'      
                          #'*  \code{corr_force_positive}:
                          #'    This will force all elements in the correlation matrix (or matrices if LC_MVP) to be positive. Uses the method proposed by Pinkney et al, 2024. 
                          #'    
                          #'*  \code{lkj_cholesky_eta}:
                          #'    This is a vector of length \code{n_class}, where each element corresponds to the LKJ prior parameter \eqn{\eta_{c}} corresponding  to latent class \eqn{c}. For non-latent class models (i.e., the MVP),
                          #'    this will just be a vector with 1 element e.g. c(4) corresponds to: 
                          #'    \eqn{\Omega \sim \text{LKJ}\left(4\right)}. For latent class models (i.e. the LC_MVP and the latent_trait models), the first element corresponds to the first latent class 
                          #'    (for test accuracy applications this will be the NON-diseased class) and the second element corresponds to the second latent class (for test accuracy applications this will be the diseased class). 
                          #'    
                          #'*  \code{ub_corr}:
                          #'    An array of dimension \code{n_class} x \code{n_outcomes} x \code{n_outcomes}, which contains the upper-bounds for the correlations. Note that only the lower-triangular elements of each of the 
                          #'    supplied \code{n_class} matrices are used, and the default is a matrix with lower-triangular part all equal to 1. 
                          #' 
                          #'*  \code{lb_corr}:
                          #'    An array of dimension \code{n_class} x \code{n_outcomes} x \code{n_outcomes}, which contains the lower-bounds for the correlations. Note that only the lower-triangular elements of each of the
                          #'    supplied \code{n_class} matrices are used, and the default is a matrix with lower-triangular part all equal to 0. 
                          #'    
                          #'*  \code{known_values_indicator}:
                          #'    An array of dimension \code{n_class} x \code{n_outcomes} x \code{n_outcomes}, which contains elements that are either 1 or 0, such that: if element \eqn{(c, t_1, t_2)} is 0, then correlation 
                          #'    \eqn{(c, t_1, t_2)} is unknown and will be estimated, however if element \eqn{(c, t_1, t_2)} is 1, then we know correlation \eqn{(c, t_1, t_2)} a priori and hence it will be fixed - specifically 
                          #'    it will be set equal to the corresponding value in \code{known_values} (see below). In other words, if any correlations are known a priori, they can be passed onto the model via this argument. 
                          #'    Note that only the lower-triangular elements of each of the supplied \code{n_class} matrices are used, and the default is a matrix with lower-triangular part all equal to 0 (i.e., assumes no #
                          #'    correlations are known/fixed). 
                          #' 
                          #'*  \code{known_values}:
                          #'    An array of dimension \code{n_class} x \code{n_outcomes} x \code{n_outcomes}, which contains any known values for the correlations. In other words, if any correlations are known a priori, they can 
                          #'    be passed onto the model via this argument. Note that only the lower-triangular elements of each of the supplied \code{n_class} matrices are used, and the default is a matrix with lower-triangular 
                          #'    part all equal to 0 (note that these values are arbitrary since the elements in known_values_indicator are all equal to zero, so they will be ignored unless one or more elements in
                          #'     known_values_indicator is non-zero). 
                          #'   
                          #' Arguments which are only relevant to latent class models (i.e. the LC_MVP and latent_trait models):
                          #' 
                          #'*  \code{prev_prior_a}: 
                          #'    Shape parameter 1 for prevalence beta prior. Only relevant for latent-class models (i.e. the LC_MVP or latent_trait models).
                          #'   
                          #'*  \code{prev_prior_b}: 
                          #'    Shape parameter 2 for prevalence beta prior. Only relevant for latent-class models (i.e. the LC_MVP or latent_trait models).
                          #'   
                          #' Arguments which are only relevant to the latent trait model (i.e., if \code{Model_type} = \code{"latent_trait"}) include:
                          #' 
                          #'*  \code{LT_b_priors_shape}: 
                          #'    A matrix of dimension \code{n_class} x \code{n_outcomes}, where each element corresponds to the prior Weibull shape parameter of the "b" parameters in the latent trait model - which are denoted 
                          #'    as \code{LT_b}. The default is a matrix with with every value equal to 1.33. Please see LT_b_priors_scale below for a justification of this default choice. 
                          #'           
                          #'*  \code{LT_b_priors_scale}:
                          #'    A matrix of dimension \code{n_class} x \code{n_outcomes}, where each element corresponds to the prior Weibull scale parameter of the "b" parameters in the latent trait model - which are denoted 
                          #'    as \code{LT_b}. The default is a matrix with every value equal to 1.25. Together with the default choice \code{LT_b_priors_shape} (please see description above), these priors correspond to the 
                          #'    following Weibull priors: \eqn{b_{c, t} \sim \text{Weibull}\left(1.33, 1.250\right)}. We chose these as default values because they are equivalent to setting \eqn{\text{truncated-LKJ}\left(1.5\right)}
                          #'     priors in the LC_MVP model, which are very weakly informative, especially if the dimension (i.e. number of outcomes/tests) is small. 
                          #'    
                          #'*  \code{LT_known_bs_indicator}:
                          #'    A matrix of dimension \code{n_class} x \code{n_outcomes}, which contains elements that are either 1 or 0, such that: if element \eqn{(c, t)} is 0, then the corresponding  \code{LT_b} parameter is 
                          #'    unknown and will be estimated; however, if element \eqn{(c, t)} is 1, then we know the corresponding  \code{LT_b} parameter a priori and hence it will be fixed - specifically it will be set equal 
                          #'    to the corresponding value in \code{LT_known_bs_values}. In other words, if any \code{LT_b} are known a priori, they can be passed onto the model via this argument. 
                          #'    
                          #'*  \code{LT_known_bs_values}:
                          #'    A matrix of dimension \code{n_class} x \code{n_outcomes}, which contains any known values for the \code{LT_b} parameters. In other words, if any \code{LT_b} are known a priori, they can be passed 
                          #'    onto the model via this argument.  
                          #'   
                          #'   
                          model_args_list = NULL,
                          #'@field Stan_data_list  List containing data for Stan models (only relevant if "Model_type" is set to "Stan"). The elements of the list should correspond to the variables defined in the "data" block of
                          #' your Stan model. 
                          Stan_data_list = NULL,
                          #'@field sample_nuisance Whether or not to sample the high-dimensional nuisance/latent variable vector. 
                          sample_nuisance = NULL,
                          #'@field n_chains_burnin The total number of burn-in chains. The default is Min(8, n_cores), where n_cores is the number of cores on the CPU. 
                          n_chains_burnin = NULL,
                          #'@field use_disk Whether the sampling traces were stored in RAM (FALSE) or on disk (TRUE). Set by $sample() and re-used by $summary() so the summary reads the same storage mode.
                          use_disk = NULL,
                          #'@field compile_choice The "compile" argument captured at construction (used if initialisation is deferred to $sample()).
                          compile_choice = NULL,
                          #'@field force_recompile_choice The "force_recompile" argument captured at construction (used if initialisation is deferred to $sample()).
                          force_recompile_choice = NULL,
                          
                          ## ---------- constructor - initialize using the initialise_model fn (this wraps the initialise_model function with $new()) - store all important parameters
                          #'@description
                          #'Create a new MVP model object
                          #'@param Model_type Type of model ("MVP", "LC-MVP", etc.). See class documentation for details.
                          #'@param y The dataset. See class documentation for details.
                          #'@param N The sample size. See class documentation for details.
                          #'@param n_params_main Number of main parameters. See class documentation for details.
                          #'@param n_nuisance Number of nuisance parameters. See class documentation for details.
                          #'@param init_lists_per_chain List of initial values for each chain. See class documentation for details.
                          #'@param n_chains_burnin Number of chains used for burnin. See class documentation for details.
                          #'@param compile Compile the (possibly dummy if using built-in models) Stan model. 
                          #'@param force_recompile Force-compile the (possibly dummy if using built-in models) Stan model. 
                          #'@param model_args_list List of model arguments. See class documentation for details.
                          #'@param Stan_data_list List of Stan data (optional). See class documentation for details.
                          #'@param sample_nuisance Whether to sample nuisance parameters. See class documentation for details.
                          #'@param Stan_model_file_path The file path to the Stan model, only needed if \code{Model_type = "Stan"}.
                          #'@param Stan_cpp_user_header The file path to a user-supplied C++ .hpp file to be compiled together with the Stan model. This is optional and only needed if you want to use custom C++ functions in your 
                          #' Stan model, and is only relvant if \code{Model_type = "Stan"}.
                          #'@param Stan_cpp_flags User-supplied R list containing comma-separated values of compiler flags (e.g. CXX_FLAGS, etc) to be passed on
                          #' to cmdstanr - the Stan model will then be compiled using these flags. This is optional and only needed if you want to use custom C++ functions in your Stan model. 
                          #' Only relevant if \code{Model_type = "Stan"}.
                          #'@param ... Additional arguments passed to BayesMVP::initialise_model.
                          #'@return Returns self$init_object, an object generated from the "BayesMVP::initialise_model" function which contains information such as which 
                          #'model type (\code{Model_type}) to use. 
                          initialize = function(Model_type, 
                                                ##
                                                sample_nuisance = NULL,
                                                n_nuisance_override = NULL,
                                                ##
                                                model_args_list = NULL,  # For internal/hard-coded models 
                                                ##
                                                Stan_data_list = NULL,  # Optional AT THIS STAGE for Stan models
                                                Stan_model_file_path = NULL,
                                                ##
                                                Stan_cpp_user_header = NULL,
                                                Stan_cpp_flags = NULL,
                                                stanc_args = NULL,
                                                make_args = NULL,
                                                ##
                                                compile = TRUE,
                                                force_recompile = FALSE) {
                            
                                    if (Model_type == "Stan") {
                                      if (is.null(Stan_model_file_path)) {
                                        stop("Stan_model_file_path required for Stan models")
                                      }
                                    } else {
                                      # For internal models, we just store the model type
                                      # The actual model will be initialized when data is provided in sample()
                                    }
                                    ## ------ store important parameters as class members
                                    self$Model_type <- Model_type
                                    ##
                                    ## set bs environment variable (otherwise it'll try downloading it even if already installed...)
                                    bs_path <- bridgestan_path()
                                    Sys.setenv(BRIDGESTAN = bs_path)
                                    ##
                                    ## ---- PRESERVE the constructor inputs so that deferred
                                    ## initialisation (below) and $sample() can always re-create
                                    ## the model (previously these were overwritten with
                                    ## NULL$... lookups and lost when init was deferred):
                                    self$sample_nuisance <- sample_nuisance
                                    self$n_nuisance_override <- n_nuisance_override
                                    ##
                                    self$model_args_list <- model_args_list
                                    ##
                                    self$Stan_data_list <- Stan_data_list
                                    self$Stan_model_file_path <- Stan_model_file_path
                                    ##
                                    self$Stan_cpp_user_header <- Stan_cpp_user_header
                                    self$Stan_cpp_flags <- Stan_cpp_flags
                                    self$stanc_args <- stanc_args
                                    self$make_args <- make_args
                                    ##
                                    self$compile_choice <- compile
                                    self$force_recompile_choice <- force_recompile
                                    ##
                                    ## If data provided, compile now. Otherwise defer to sample()
                                    if (!is.null(Stan_data_list) || !is.null(model_args_list)) {
                                      
                                          ## Compile with provided data
                                          self$init_object <-   initialise_model(   Model_type = Model_type,
                                                                                    ##
                                                                                    sample_nuisance = sample_nuisance,
                                                                                    n_nuisance_override = n_nuisance_override,
                                                                                    ##
                                                                                    model_args_list = model_args_list, # For internal/hard-coded models 
                                                                                    ##
                                                                                    Stan_data_list = Stan_data_list, ## for user-supplied Stan models
                                                                                    Stan_model_file_path = Stan_model_file_path, ## for user-supplied Stan models
                                                                                    ##
                                                                                    # n_chains_burnin = n_chains_burnin,
                                                                                    # init_lists_per_chain = init_lists_per_chain,
                                                                                    ##
                                                                                    compile = compile,
                                                                                    force_recompile = force_recompile,
                                                                                    ##
                                                                                    cmdstanr_model_fit_obj = NULL, ## ignore this
                                                                                    ##
                                                                                    Stan_cpp_user_header = Stan_cpp_user_header,
                                                                                    Stan_cpp_flags = Stan_cpp_flags,
                                                                                    stanc_args = stanc_args,
                                                                                    make_args = make_args)
                                          self$is_compiled <- TRUE
                                         
                                    } else {
                                           
                                          self$is_compiled <- FALSE ## deferred - initialised in $sample()
                                    }
                                    ##
                                    ## ---- take the RESOLVED state from the fresh init object
                                    ## (kept only when initialisation happened NOW; the raw
                                    ## inputs above survive when it is deferred):
                                    if (!is.null(self$init_object)) {
                                      ##
                                      self$model_args_list <- self$init_object$model_args_list
                                      ##
                                      self$Stan_data_list <- self$init_object$Stan_data_list 
                                      self$Stan_model_file_path <- self$init_object$Stan_model_file_path
                                      ##
                                      self$Stan_cpp_user_header <- self$init_object$Stan_cpp_user_header
                                      self$Stan_cpp_flags <- self$init_object$Stan_cpp_flags
                                      self$stanc_args <- self$init_object$stanc_args
                                      self$make_args <- self$init_object$make_args
                                      ##
                                      self$n_nuisance <- self$init_object$n_nuisance
                                      self$n_params_main <- self$init_object$n_params_main
                                      ##
                                    }
                           },
                          ## --------  wrap the sample fn -----------------------------------------------------------------------------------------------------------------------------
                          #'@description
                          #'Sample from the model
                          #'@param init_lists_per_chain List of initial values for each chain. See class documentation for details.
                          #'@param model_args_list List of model arguments. See class documentation for details.
                          #'@param Stan_data_list List of Stan data (optional). See class documentation for details.
                          #'@param parallel_method The method to use for parallelisation (multithreading) in C++. Default is "RcppParallel". Can be changed to "OpenMP", if available. 
                          #'@param y The dataset. See class documentation for details.
                          #'@param N The sample size. See class documentation for details.
                          #'@param manual_tau If \code{FALSE}, then SNAPER-HMC will be used to adapt \eqn{\tau} during the burnin phase. Otherwise if \code{TRUE}, \eqn{\tau} will be
                          #' fixed to the value given in the \code{tau_if_manual} argument. 
                          #'@param tau_if_manual The HMC path length (\eqn{\tau}) to use for the HMC sampling. This will be used for both the burnin and sampling phases. 
                          #'Note that this only works if \code{manual_tau = TRUE}. Otherwise, \eqn{\tau} will be adapted using SNAPER-HMC. Also note that if only one value is 
                          #'given, then this value of tau will be used for both the main parameter sampling and the nuisance parameter sampling. If you want to specify separate
                          #'path lengths for the main and nuisance parameters, provide a vector instead. E.g.  \code{tau_if_manual = c(0.50, 2.0)} will mean that 
                          #'\eqn{\tau} = 0.50 is used for the main parameters and \eqn{\tau} = 2.0 is used for the nuisance parameters. 
                          #'@param sample_nuisance Whether to sample nuisance parameters. See class documentation for details.
                          #'@param partitioned_HMC Whether to sample all parameters at once (note: wont use diffusion HMC) or whether to alternate between sampling the nuisance 
                          #'parameters and the main model parameters. 
                          #'@param diffusion_HMC Whether to use diffusion-pathspace HMC (Beskos et al) to sample nuisance parameters. Default is TRUE. 
                          #'@param diffusion_HMC_integrator Joint diffusion integrator: "kick_flow_kick" (default) or "flow_kick_flow".
                          #'@param debug_burnin_timing Collect burn-in timing and per-chain integrator step diagnostics. Default FALSE.
                          #'@param vect_type The SIMD (single-nstruction, multiple-data) vectorisation type to use for math functions (such as log, exp, Phi, etc).
                          #'The default is AVX-512 if available, then AVX2 if not, and if neither AVX-512 nor AVX2 are available
                          #'(e.g., on ARM-based CPU's such as Apple systems), then BayesMVP will use the Stan C++ math library functions, and they will decide what
                          #'vectorisation type to use. More vectorisation types will be supported in the future. 
                          #'@param Phi_type Type of Phi() function implementation to use, where Phi() is the standard normal CDF. Can be either "Phi" (the default) or "Phi_approx()". 
                          #'@param n_params_main Number of main parameters. See class documentation for details.
                          #'@param n_nuisance Number of nuisance parameters. See class documentation for details.
                          #'@param n_chains_burnin Number of chains used for burnin. See class documentation for details.
                          #'@param n_chains_sampling Number of chains used for sampling (i.e., post-burnin). Note that this must match the length of "init_lists_per_chain".
                          #'@param n_superchains  Number of superchains for nested R-hat (nR-hat; see Margossian et al, 2023) computation. 
                          #'@param seed Random MCMC seed.
                          #'@param n_burnin The number of burnin iterations. Default is 500. 
                          #'@param n_iter The number of sampling (i.e., post-burnin) iterations. Default is 1000. 
                          #'@param adapt_delta The Metropolis-Hastings target acceptance rate. Default is 0.80. If there are divergences, sometimes increasing this can help, at the cost 
                          #'of efficiency (since this will decrease the step-size (epsilon) and increase the number of leapfrog steps (L) per iteration).
                          #'@param learning_rate The ADAM learning rate (LR) for learning the appropriate HMC step-size (\eqn{\epsilon}) and HMC path length (\eqn{\tau = L \cdot \epsilon})
                          #'during the burnin period. The default depends on the length of the burnin, and is the pairing selected by pilot study 7:
                          #'LR = 0.0125, 0.025, 0.05 and 0.075 for \code{n_burnin} = 1000, 500, 250 and 125 respectively. Burnin lengths BETWEEN those
                          #'values are interpolated linearly in \eqn{\log(n\_burnin)} vs \eqn{\log(LR)} (i.e. a piecewise power law), which over
                          #'[250, 1000] is exactly \eqn{LR = 12.5 / n\_burnin}; the short [125, 250] segment is deliberately shallower than that.
                          #'Outside [125, 1000] the nearest segment's slope is continued, capped at LR = 0.125 so that a very short burnin cannot
                          #'produce an unstable rate.
                          #'@param n_refresh How often to report burnin progress, as an INTERVAL in iterations: progress is printed every
                          #'\code{n_refresh} iterations (the same convention as Stan's \code{refresh}). \code{NULL} (the default) uses
                          #'\code{n_burnin / 20}, i.e. about twenty reports across the burnin. Larger values are quieter.
                          #'@param learning_rate_initial A HIGH ADAM learning rate held for the first \code{learning_rate_initial_iter} burnin
                          #'iterations, then dropped to \code{learning_rate}. The learning rate drives BOTH the step-size (\eqn{\epsilon}) and the
                          #'path-length (\eqn{\tau}) adaptation, so holding it high lets the adaptation move a long way early instead of creeping,
                          #'while the later, finer adaptation still runs at the normal rate. Default is \code{0.10}; pass \code{NULL} (or \code{NA})
                          #'to switch the hold off.
                          #'@param learning_rate_initial_iter How many burnin iterations to hold \code{learning_rate_initial} for. The default is
                          #'\code{n_burnin / 10}, i.e. 50 iterations at \code{n_burnin} = 500, scaled proportionally to other burnin lengths.
                          #'@param eps_initial A deliberately LARGE HMC step-size (\eqn{\epsilon}) held for the first \code{eps_initial_iter} burnin
                          #'iterations, so that the chains cover ground quickly before the usual step-size takes over. During this window the path
                          #'length is one leapfrog step (MALA), so these are large single steps. After the window, \eqn{\epsilon} is handed back to
                          #'the value it would otherwise have started from (the automatic heuristic) and adapted as normal, with the ADAM moments
                          #'reset so the warm-start value is not carried forward. Default is \code{NULL} (OFF); set a positive value to enable it.
                          #'Capped by \code{max_eps_main} / \code{max_eps_nuisance}.
                          #'@param eps_initial_iter How many burnin iterations to hold \code{eps_initial} for. The default is \code{n_burnin / 10},
                          #'i.e. the 50 iterations found to work well at \code{n_burnin} = 500, scaled proportionally to other burnin lengths.
                          #'@param n_nuisance_to_track How many nuisance (latent-variable) coordinates the sampler keeps a trace of.
                          #'The trace is only ever read by the post-hoc \code{param_constrain} step, which needs it solely when the model
                          #'being constrained declares the nuisance block (an external Stan model with a nuisance block, or \code{latent_trait});
                          #'the other built-in skeletons declare the main parameters only, so for them the trace has no reader and storing it is
                          #'pure memory traffic (tens of GB per run at \code{n_nuisance} = 60,000). \code{NULL} (the default) decides this
                          #'automatically from the model's unconstrained dimension; \code{0} never stores it; \code{n_nuisance} always does. Any other value is refused: the trace is all-or-nothing (a partial
                          #'buffer is grown to the full length on the first write, so e.g. 1 would cost exactly as much as keeping everything).
                          #'@param clip_iter The number of iterations to perform MALA (i.e., one leapfrog step) on (first clip_iter iterations) for the burnin period.
                          #'The default depends on the length of the burnin period, and is the configuration selected by pilot study 7:
                          #'\code{clip_iter} = 150, 75, 50 and 30 for \code{n_burnin} = 1000, 500, 250 and 125 respectively (banded, so e.g. any
                          #'\code{n_burnin} in [250, 500) takes 50; outside [125, 1000] the nearest band's ratio is extrapolated).
                          #'Update: at exactly \code{n_burnin = 125}, the default is now \code{clip_iter = 20}; the historical values above
                          #'are retained for reference. Other burnin lengths and explicit overrides are unchanged.
                          #'@param clip_iter_tau The iteration at which the path-length (tau) adaptation stops being clipped during the burnin period.
                          #'The default likewise follows pilot study 7, where it is \code{clip_iter + int}: \code{clip_iter_tau} = 350, 175, 125 and 80
                          #'for \code{n_burnin} = 1000, 500, 250 and 125 respectively. Note this also sets the default \code{gap}, which is
                          #'\code{clip_iter + clip_iter_tau} when \code{gap} is left NULL.
                          #'Update: at exactly \code{n_burnin = 125}, \code{clip_iter_tau = 50} gives the requested 40-percent handover.
                          #'The burn-in routine uses \code{clip_iter_tau} as its effective handover, overriding the incoming \code{gap}.
                          #'@param interval_width_main How often to update the metric (which is either empirical or Hessian-informed, and can be diagonal or dense) for the 
                          #'main parameters during the burnin period. The default used depends on the length of the burnin period. 
                          #'@param interval_width_nuisance How often to update the metric (which is empirical and diagonal) for the nuisance parameters during the burnin period.
                          #'The default used depends on the length of the burnin period. 
                          #'@param force_autodiff Whether to use (force) autodiff instead of the built-in manual gradients. This is only relevant for the built-in models, not 
                          #'Stan models. The default is \code{FALSE}. 
                          #'@param force_PartialLog Whether to use (force) use of gradients on the log-scale (for stability). Note that if \code{force_autodiff = TRUE} then this will 
                          #'automatically be set to \code{TRUE}, since autodiff is only available for partial-log-scale gradients. The default is \code{FALSE}. 
                          #'@param multi_attempts Whether 
                          #'@param max_L The maximum number of leapfrog steps. This is similar to \code{max_treedepth} in Stan. The default is 1024 (which is equivalent to the default
                          #'\code{max_treedepth = 10} in Stan). 
                          #'@param tau_mult The SNAPER-HMC algorithm multiplier to use when adjusting the path length during the burnin phase. 
                          #'@param metric_type_main The type of metric to use for the main parameters, which is adapted during the burnin period. Can be either \code{"Hessian"} or 
                          #'\code{"Empirical"}, where the former uses second derivative information and the latter uses the SD of the posterior computed whilst sampling. The default
                          #'is \code{"Hessian"} if \code{n_params_main < 250} and \code{"Empirical"} otherwise. 
                          #'@param metric_shape_main The shape of the metric to use for the main parameters, which is adapted during the burnin period. Can be either \code{"diag"} or 
                          #'\code{"dense"}. The default is \code{"diag"} (i.e. a diagonal metric).
                          #'@param metric_type_nuisance The type of metric to use for the nuisance parameters, which is adapted during the burnin period. Can be "diag" or "unit". 
                          #'@param ratio_M_main Ratio to use for the metric. Main parameters. 
                          #'@param ratio_M_us Ratio to use for the metric. Nuisance parameters. 
                          #'@param force_recompile Recompile the (possibly dummy if using built-in model) Stan model. 
                          #'@param ... Additional arguments passed to sampling.
                          #'@return Returns self invisibly, allowing for method chaining of this classes (MVP_model) methods. E.g.: model$sample(...)$summary(...). 
                          #'
                          #'@param burnin_TBB_pool_equals_n_chains NULL defaults to TRUE for built-in models, which use OpenMP within chains.
                          #'External Stan models always force FALSE, even when TRUE is supplied, so nested TBB likelihood work can use
                          #'n_chains_burnin * n_threads_WCP_burnin threads. Built-in models can explicitly select FALSE.
                          sample = function(  force_recompile = FALSE,
                                              ##
                                              n_chains_burnin = self$n_chains_burnin,
                                              init_lists_per_chain = self$init_lists_per_chain,
                                              ##
                                              parallel_method = NULL,
                                              ##
                                              Stan_data_list = self$Stan_data_list,
                                              model_args_list = self$model_args_list,
                                              ##
                                              sample_nuisance =   self$sample_nuisance,
                                              n_nuisance_override = self$n_nuisance_override,
                                              ##
                                              seed = NULL,
                                              n_burnin = NULL,
                                              n_adapt = NULL,
                                              gap = NULL,
                                              ##
                                              n_chains_sampling = NULL,
                                              n_superchains = NULL,
                                              n_iter = NULL,
                                              ##
                                              adapt_delta = NULL,
                                              learning_rate = NULL,
                                              ##
                                              tau_mult  = NULL,
                                              tau_initial = NULL,
                                              ##
                                              manual_tau = NULL,
                                              tau_if_manual = NULL,
                                              ##
                                              burnin_algorithm = NULL,
                                              partitioned_HMC = NULL,
                                              diffusion_HMC = NULL,
                                              ##
                                              clip_iter = NULL,
                                              clip_iter_tau = NULL,
                                              ##
                                              n_refresh = NULL,
                                              use_proposed = NULL,
                                              ##
                                              beta1_adam = 0.00,
                                              beta2_adam = 0.95,
                                              eps_adam = 1e-8,
                                              ##
                                              force_autodiff = NULL,
                                              force_PartialLog = NULL,
                                              multi_attempts = NULL,
                                              ##
                                              force_autodiff_for_metric = TRUE,
                                              force_PartialLog_for_metric = FALSE,
                                              force_multi_attempts_for_metric = FALSE,
                                              ##
                                              vect_type = NULL,
                                              Phi_type = "Phi",
                                              inv_Phi_type = "inv_Phi",
                                              ##
                                              metric_type_main = NULL,
                                              metric_shape_main = NULL,
                                              ratio_M_main = NULL,
                                              interval_width_main = NULL,
                                              ##
                                              metric_type_nuisance = NULL,
                                              metric_shape_nuisance = NULL,
                                              ratio_M_nuisance = NULL,
                                              interval_width_nuisance = NULL,
                                              ##
                                              metric_estimator = "pooled",
                                              ##
                                              M_decay_type = NULL,
                                              M_decay_power = NULL,
                                              M_decay_scale = NULL,
                                              ##
                                              max_tau_main = 25.0,
                                              max_tau_nuisance = 25.0,
                                              ##
                                              max_eps_main = NULL,
                                              max_eps_nuisance = NULL,
                                              ##
                                              learning_rate_initial = 0.10,
                                              learning_rate_initial_iter = NULL,
                                              ##
                                              eps_initial = NULL,
                                              eps_initial_iter = NULL,
                                              ##
                                              max_L = NULL,
                                              ##
                                              n_nuisance_to_track = NULL,
                                              ##
                                              use_disk = NULL,
                                              ##
                                              n_threads_WCP_burnin = 1,
                                              n_threads_WCP_sampling = 1,
                                              num_chunks_burnin = NULL,
                                              num_chunks_sampling = NULL,
                                              ##
                                              reorder_cols_MVP = FALSE,
                                              diffusion_HMC_integrator = "kick_flow_kick",
                                              debug_burnin_timing = FALSE,
                                              ##
                                              ## Explicit access to the options already supported by R_fn_sample_model.
                                              debug = FALSE,
                                              stream = NULL,
                                              nuisance_jitter_scale = 0.25,
                                              tau_if_manual_in_L_units = NULL,
                                              tau_objective = NULL,
                                              tau_weight_by_p_jump = NULL,
                                              tau_ramp = NULL,
                                              eps_reinit_at_ChEES_handover = NULL,
                                              theta_hat_us_rule = NULL,
                                              theta_hat_us_freeze_iter = NULL,
                                              burnin_schedule = "automatic",
                                              metric_adaptation_end_iter = NULL,
                                              pre_burnin_n_iter = NULL,
                                              pre_burnin_L = NULL,
                                              share_tau_ii_across_chains_in_burnin = NULL,
                                              burnin_TBB_pool_equals_n_chains = NULL,
                                              store_log_lik_trace = NULL,
                                              use_disk_path = "/tmp/hmc_traces",
                                              test_perm_override = NULL
                                              ##
                                             ) {
                            ## Resolve here for R6 callers; R_fn_sample_model also enforces this for direct callers.
                            if (identical(x = self$Model_type, y = "Stan")) {
                                if (isTRUE(x = burnin_TBB_pool_equals_n_chains)) {
                                    message("burnin_TBB_pool_equals_n_chains = TRUE is overridden to FALSE for Stan models to allow TBB within-chain parallelism.")
                                }
                                burnin_TBB_pool_equals_n_chains <-  FALSE
                            } else {
                                burnin_TBB_pool_equals_n_chains <-  if (is.null(x = burnin_TBB_pool_equals_n_chains)) TRUE else
                                                                      isTRUE(x = burnin_TBB_pool_equals_n_chains)
                            }
                            ##
                            ## ---- DEFERRED initialisation: if $new() was called without data,
                            ## initialise NOW with the preserved constructor inputs (the data must
                            ## be supplied to this $sample() call):
                            if (is.null(self$init_object)) {
                              ##
                              self$init_object <-   initialise_model(   Model_type = self$Model_type,
                                                                        ##
                                                                        sample_nuisance = self$sample_nuisance,
                                                                        n_nuisance_override = self$n_nuisance_override,
                                                                        ##
                                                                        model_args_list = model_args_list,
                                                                        ##
                                                                        Stan_data_list = Stan_data_list,
                                                                        Stan_model_file_path = self$Stan_model_file_path,
                                                                        ##
                                                                        compile = if (is.null(self$compile_choice)) TRUE else self$compile_choice,
                                                                        force_recompile = if (is.null(self$force_recompile_choice)) force_recompile else self$force_recompile_choice,
                                                                        ##
                                                                        cmdstanr_model_fit_obj = NULL,
                                                                        ##
                                                                        Stan_cpp_user_header = self$Stan_cpp_user_header,
                                                                        Stan_cpp_flags = self$Stan_cpp_flags,
                                                                        stanc_args = self$stanc_args,
                                                                        make_args = self$make_args)
                              ##
                              self$is_compiled <- TRUE
                              ##
                              self$model_args_list <- self$init_object$model_args_list
                              ##
                              self$Stan_data_list <- self$init_object$Stan_data_list
                              self$Stan_model_file_path <- self$init_object$Stan_model_file_path
                              ##
                              self$Stan_cpp_user_header <- self$init_object$Stan_cpp_user_header
                              self$Stan_cpp_flags <- self$init_object$Stan_cpp_flags
                              self$stanc_args <- self$init_object$stanc_args
                              self$make_args <- self$init_object$make_args
                              ##
                              self$n_nuisance <- self$init_object$n_nuisance
                              self$n_params_main <- self$init_object$n_params_main
                              ##
                            }
                            ##
                            ## validate initialization:
                            ##
                            if (is.null(self$init_object)) {
                              stop("Model was not properly initialized")
                            }
                            ##
                            ## ---- resolve the storage mode ONCE, and record it so that
                            ## $summary() reads the same storage mode that sampling used:
                            if (is.null(use_disk)) {
                              use_disk <- if (is.null(self$use_disk)) FALSE else self$use_disk
                            }
                            self$use_disk <- use_disk
                            ##
                            if (!(is.null(Stan_data_list))) { 
                              self$Stan_data_list <- Stan_data_list
                            }
                            if (!(is.null(model_args_list))) { 
                              self$model_args_list <- model_args_list
                            }
                            ##
                            if (!(is.null(n_chains_burnin))) { 
                              self$n_chains_burnin <- n_chains_burnin
                            }
                            if (!(is.null(init_lists_per_chain))) { 
                              self$init_lists_per_chain <- init_lists_per_chain
                            }
                            if (!(is.null(sample_nuisance))) { 
                              self$sample_nuisance <- sample_nuisance
                            }
                            if (!(is.null(n_nuisance_override))) { 
                              self$n_nuisance_override <- n_nuisance_override
                            }
                            
                            # if (is.null(clip_iter)) {
                            #       if (n_burnin > 999) {
                            #         clip_iter <-  round(n_burnin/20, 0)  ## round(n_burnin/10, 0)   
                            #       } else if ((n_burnin > 499) && (n_burnin < 1000)) { 
                            #         clip_iter <-  round(n_burnin/20, 0)  ## round(n_burnin/10, 0)  
                            #       } else if (n_burnin %in% c(250:499)) { 
                            #         clip_iter <-  25 #  round(n_burnin/10, 0) # 50 # 15
                            #       } else if (n_burnin %in% c(150:249)) { 
                            #         clip_iter <-  25 # 30 # 2  #  round(n_burnin/20, 0) # 25
                            #       } else {  # 149 or less
                            #         clip_iter <-  15 #  20 # 10 # 5 
                            #       }
                            # }
                            ##
                            # ## Nuisance params:
                            # n_nuisance <- if_null_then_set_to(n_nuisance, self$n_nuisance)
                            # if (n_nuisance == 0) {
                            #   diffusion_HMC <- FALSE ## diffusion_HMC only done for nuisance 
                            #   partitioned_HMC <- FALSE ## nothing to partition if no nuisance params!
                            # }
                            ##
                            # n_adapt <- if_null_then_set_to(n_adapt, n_burnin - round(n_burnin/10))
                            # gap <- if_null_then_set_to(gap, clip_iter  + round(n_adapt / 5))
                            # ##
                            # interval_width_main <- if_null_then_set_to(interval_width_main, round(n_burnin/10))
                            # interval_width_nuisance <- if_null_then_set_to(interval_width_nuisance, round(n_burnin/10))
                            # ##
                            # clip_iter_tau <- if_null_then_set_to(clip_iter_tau, round(0.3*n_burnin))
                            ##
                            ###  partitioned_HMC <- TRUE # currently only TRUE is supported. 
                            ## inv_Phi_type <- ifelse(Phi_type == "Phi", "inv_Phi", "inv_Phi_approx") # inv_Phi_type is not modifiable 

                            # -----------  call R_fn_sample_model fn ---------------------------------------------------------------------------------------------------
                            self$result <-       R_fn_sample_model(   
                                                            init_object = self$init_object,
                                                            ##
                                                            # Model_type = self$Model_type, ## cannot be changed in "$sample()"
                                                            ##
                                                            n_chains_burnin = n_chains_burnin,
                                                            init_lists_per_chain = init_lists_per_chain,
                                                            ##
                                                            parallel_method = parallel_method,
                                                            ##
                                                            Stan_data_list = Stan_data_list, 
                                                            model_args_list = model_args_list,
                                                            ##
                                                            sample_nuisance =   sample_nuisance,
                                                            n_nuisance_override = n_nuisance_override,
                                                            ##
                                                            seed = seed,
                                                            n_burnin = n_burnin,
                                                            n_adapt = n_adapt,
                                                            gap = gap,
                                                            ##
                                                            n_chains_sampling = n_chains_sampling,
                                                            n_superchains = n_superchains,
                                                            n_iter = n_iter,
                                                            ##
                                                            adapt_delta = adapt_delta,
                                                            learning_rate = learning_rate,
                                                            ##
                                                            tau_mult = tau_mult,
                                                            tau_initial = tau_initial,
                                                            ##
                                                            manual_tau = manual_tau,
                                                            tau_if_manual = tau_if_manual,
                                                            ##
                                                            burnin_algorithm = burnin_algorithm,
                                                            diffusion_HMC = diffusion_HMC,
                                                            diffusion_HMC_integrator = diffusion_HMC_integrator,
                                                            debug_burnin_timing = debug_burnin_timing,
                                                            debug = debug,
                                                            stream = stream,
                                                            nuisance_jitter_scale = nuisance_jitter_scale,
                                                            tau_if_manual_in_L_units = tau_if_manual_in_L_units,
                                                            tau_objective = tau_objective,
                                                            tau_weight_by_p_jump = tau_weight_by_p_jump,
                                                            tau_ramp = tau_ramp,
                                                            eps_reinit_at_ChEES_handover = eps_reinit_at_ChEES_handover,
                                                            theta_hat_us_rule = theta_hat_us_rule,
                                                            theta_hat_us_freeze_iter = theta_hat_us_freeze_iter,
                                                            burnin_schedule = burnin_schedule,
                                                            metric_adaptation_end_iter = metric_adaptation_end_iter,
                                                            pre_burnin_n_iter = pre_burnin_n_iter,
                                                            pre_burnin_L = pre_burnin_L,
                                                            share_tau_ii_across_chains_in_burnin = share_tau_ii_across_chains_in_burnin,
                                                            burnin_TBB_pool_equals_n_chains = burnin_TBB_pool_equals_n_chains,
                                                            store_log_lik_trace = store_log_lik_trace,
                                                            use_disk_path = use_disk_path,
                                                            test_perm_override = test_perm_override,
                                                            partitioned_HMC = partitioned_HMC,
                                                            ##
                                                            clip_iter = clip_iter,
                                                            clip_iter_tau = clip_iter_tau,
                                                            ##
                                                            n_refresh = n_refresh,
                                                            use_proposed = use_proposed,
                                                            ##
                                                            beta1_adam = beta1_adam,
                                                            beta2_adam = beta2_adam,
                                                            eps_adam = eps_adam,
                                                            ##
                                                            force_autodiff = force_autodiff,
                                                            force_PartialLog = force_PartialLog,
                                                            multi_attempts = multi_attempts,
                                                            ##
                                                            force_autodiff_for_metric = force_autodiff_for_metric,
                                                            force_PartialLog_for_metric = force_PartialLog_for_metric,
                                                            force_multi_attempts_for_metric = force_multi_attempts_for_metric,
                                                            ##
                                                            vect_type = vect_type,
                                                            Phi_type = Phi_type,
                                                            inv_Phi_type = inv_Phi_type,
                                                            ##
                                                            metric_type_main = metric_type_main,
                                                            metric_shape_main = metric_shape_main,
                                                            ratio_M_main = ratio_M_main,
                                                            interval_width_main = interval_width_main,
                                                            ##
                                                            metric_type_nuisance = metric_type_nuisance,
                                                            metric_shape_nuisance = metric_shape_nuisance,
                                                            ratio_M_nuisance = ratio_M_nuisance,
                                                            interval_width_nuisance = interval_width_nuisance,
                                                            ##
                                                            metric_estimator = metric_estimator,
                                                            ##
                                                            M_decay_type = M_decay_type,
                                                            M_decay_power = M_decay_power,
                                                            M_decay_scale = M_decay_scale,
                                                            ##
                                                            max_tau_main = max_tau_main,
                                                            max_tau_nuisance = max_tau_nuisance,
                                                            ##
                                                            max_eps_main = max_eps_main,
                                                            max_eps_nuisance = max_eps_nuisance,
                                                            ##
                                                            learning_rate_initial = learning_rate_initial,
                                                            learning_rate_initial_iter = learning_rate_initial_iter,
                                                            ##
                                                            eps_initial = eps_initial,
                                                            eps_initial_iter = eps_initial_iter,
                                                            ##
                                                            max_L = max_L,
                                                            ##
                                                            n_nuisance_to_track = n_nuisance_to_track,
                                                            ##
                                                            use_disk = use_disk,
                                                            ##
                                                            n_threads_WCP_burnin = n_threads_WCP_burnin,
                                                            n_threads_WCP_sampling = n_threads_WCP_sampling,
                                                            num_chunks_burnin = num_chunks_burnin,
                                                            num_chunks_sampling = num_chunks_sampling,
                                                            ##
                                                            reorder_cols_MVP = reorder_cols_MVP)
                            
                            ## ---- refresh object state after sampling: the sampler returns an
                            ## UPDATED init_object (it re-initialises the model internally), so
                            ## keep the class in sync (n_nuisance / n_params_main, compiled flag,
                            ## resolved paths):
                            if (!is.null(self$result$init_object)) {
                              self$init_object <- self$result$init_object
                              ##
                              self$n_nuisance <- self$init_object$n_nuisance
                              self$n_params_main <- self$init_object$n_params_main
                              ##
                              self$Stan_model_file_path <- self$init_object$Stan_model_file_path
                              ##
                            }
                            self$is_compiled <- TRUE
                            
                            return(self)
                            
                          },
                          
                          ## --------  wrap the create_summary_and_traces fn + call the "BayesMVP::MVP_class_plot" R6 class  ----------------------------------------------------------
                          #'@description
                          #'Create and compute summary statistics, traces and model diagnostics. 
                          #'@param compute_main_params Whether to compute the main parameter summaries. Default is TRUE. Note that this excludes the high-dimensional nuisance
                          #'parameter vector (for Stan models - this should be defined as the FIRST  parameter in the "parameters" block). For Stan models, this will be for the
                          #'parameters defined  the "parameters" block of the model. For built-in models (i.e. the MVP, LC_MVP, and latent_trait),
                          #'this will compute summaries and traces for the coefficient vector (beta; for all 3 models), the correlation matrix/matrices (Omega;
                          #'for the MVP and LC_MVP only), for the latent_trait latent effect coefficients (i.e. the "b" parameters - denoted "LT_b" - for latent_trait only), 
                          #'and finally for the disease prevalence ("p" or "prev"; for the LC_MVP and latent_trait only). 
                          #'@param compute_transformed_parameters Whether to compute transformed parameter summaries. Default is TRUE. For Stan models, this will be for 
                          #'all of the parameters defined in the "transformed parameters" block, EXCEPT for the (transformed) nuisance parameters and log_lik (see "save_log_lik_trace"
                          #'for more information on log_lik). 
                          #'@param compute_generated_quantities Whether to compute the summaries for generated quantities. Default is TRUE. For Stan models this will exclude
                          #'any log_lik defined in the "generated quantities" block - see "save_log_lik_trace". 
                          #'@param save_log_lik_trace Whether to save the log-likelihood (log_lik) trace. Default is FALSE. For Stan models, this will only work 
                          #'if there is a "log_lik" parameter defined in the "transformed parameters" model block. For built-in models, 
                          #'@param save_nuisance_trace Whether to save the nuisance trace. Default is FALSE. 
                          #'@param compute_nested_rhat Whether to compute the nested rhat diagnostic (nR-hat) (Margossian et al, 2023). This is useful when
                          #'running many (usually short) chains. Also see "n_superchains" argument. 
                          #'@param n_superchains The number of superchains to use for the computation of nR-hat. 
                          #'Only relevant if \code{compute_nested_rhat = TRUE}.
                          #'@param save_trace_tibbles Whether to save the trace as tibble dataframes as well as 3D arrays. 
                          #'Default is FALSE. 
                          #'@param ... Any other arguments to be passed to BayesMVP::create_summary_and_traces.
                          #'@return Returns a new MVP_plot_and_diagnose object (from the "MVP_plot_and_diagnose" R6 class) for creating MCMC diagnostics and plots.
                          summary = function(       compute_main_params = TRUE,
                                                    compute_transformed_parameters = TRUE,
                                                    compute_generated_quantities = TRUE,
                                                    ##
                                                    save_log_lik_trace = TRUE,
                                                    save_nuisance_trace = FALSE,
                                                    ##
                                                    compute_nested_rhat = NULL,
                                                    n_superchains = NULL,
                                                    ##
                                                    save_trace_tibbles = FALSE,
                                                    ##
                                                    n_iter_to_store = NULL,
                                                    ##
                                                    use_disk = NULL,
                                                    use_disk_path = "/tmp/hmc_traces",
                                                    use_disk_path_post_hoc_dir = "/tmp/constrain_traces"
                                                    ) {
                            
                                # validate initialization
                                if (is.null(self$init_object)) {
                                  stop("Model was not properly initialize")
                                }
                                if (is.null(self$result)) {
                                  stop("Model has not been sampled yet - call $sample() before $summary().")
                                }
                                ##
                                ## ---- use the SAME storage mode that sampling used (self$use_disk
                                ## is recorded by $sample()):
                                if (is.null(use_disk)) {
                                  use_disk <- if (is.null(self$use_disk)) FALSE else self$use_disk
                                }
                                
                                # create model fit object (includes model summary tables + traces + divergence info) by calling "BayesMVP::create_summary_and_traces" ----------------------
                                self$model_fit_object <-           create_summary_and_traces(     model_results = self$result,
                                                                                                  ##
                                                                                                  compute_main_params = compute_main_params,
                                                                                                  compute_transformed_parameters = compute_transformed_parameters,
                                                                                                  compute_generated_quantities = compute_generated_quantities,
                                                                                                  ##
                                                                                                  save_log_lik_trace = save_log_lik_trace,
                                                                                                  save_nuisance_trace = save_nuisance_trace,
                                                                                                  ##
                                                                                                  compute_nested_rhat = compute_nested_rhat,
                                                                                                  n_superchains = n_superchains,
                                                                                                  ##
                                                                                                  save_trace_tibbles = save_trace_tibbles,
                                                                                                  ##
                                                                                                  n_iter_to_store = n_iter_to_store,
                                                                                                  ##
                                                                                                  use_disk = use_disk,
                                                                                                  use_disk_path = use_disk_path,
                                                                                                  use_disk_path_post_hoc_dir = use_disk_path_post_hoc_dir)
                                
                                # return the plotting class instance with the summary
                                MVP_class_plot_object <- MVP_plot_and_diagnose$new(  model_summary =   self$model_fit_object,
                                                                                               init_object = self$init_object,
                                                                                               n_nuisance = self$n_nuisance)
                                
                                return(MVP_class_plot_object)
                            
                          }
                          
            )
)


