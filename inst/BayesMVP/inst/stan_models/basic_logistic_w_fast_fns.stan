



functions { 
  
  //// --------------- import fast C++ MATRIX fns
  real mvp_exp_approx(real x);
  matrix mvp_exp_approx(matrix x);
  vector mvp_exp_approx(vector x);
  row_vector mvp_exp_approx(row_vector x);

  real mvp_log_approx(real x);
  matrix mvp_log_approx(matrix x);
  vector mvp_log_approx(vector x);
  row_vector mvp_log_approx(row_vector x);

  real mvp_log1p_approx(real x);
  matrix mvp_log1p_approx(matrix x);
  vector mvp_log1p_approx(vector x);
  row_vector mvp_log1p_approx(row_vector x);

  real mvp_log1m_approx(real x);
  matrix mvp_log1m_approx(matrix x); 
  vector mvp_log1m_approx(vector x);
  row_vector mvp_log1m_approx(row_vector x);

  real mvp_tanh_approx(real x);
  matrix mvp_tanh_approx(matrix x);
  vector mvp_tanh_approx(vector x);
  row_vector mvp_tanh_approx(row_vector x);

  real mvp_Phi(real x);
  matrix mvp_Phi(matrix x);
  vector mvp_Phi(vector x);
  row_vector mvp_Phi(row_vector x);

  real mvp_inv_Phi(real x); 
  vector mvp_inv_Phi(vector x);
  row_vector mvp_inv_Phi(row_vector x);
  matrix mvp_inv_Phi(matrix x);

  real mvp_Phi_approx(real x);
  vector mvp_Phi_approx(vector x);
  row_vector mvp_Phi_approx(row_vector x);
  matrix mvp_Phi_approx(matrix x); 

  real mvp_inv_Phi_approx(real x);
  vector mvp_inv_Phi_approx(vector x);  
  row_vector mvp_inv_Phi_approx(row_vector x);
  matrix mvp_inv_Phi_approx(matrix x);
   
  real mvp_inv_Phi_approx_from_logit_prob(real x);  
  vector mvp_inv_Phi_approx_from_logit_prob(vector x); 
  row_vector mvp_inv_Phi_approx_from_logit_prob(row_vector x);
  matrix mvp_inv_Phi_approx_from_logit_prob(matrix x);  
  
  real mvp_log_Phi_approx(real x); 
  vector mvp_log_Phi_approx(vector x);  
  row_vector mvp_log_Phi_approx(row_vector x);
  matrix mvp_log_Phi_approx(matrix x);

  real mvp_inv_logit(real x);
  vector mvp_inv_logit(vector x);
  row_vector mvp_inv_logit(row_vector x);  
  matrix mvp_inv_logit(matrix x);

  real mvp_log_inv_logit(real x);
  vector mvp_log_inv_logit(vector x);
  row_vector mvp_log_inv_logit(row_vector x);  
  matrix mvp_log_inv_logit(matrix x);
  
  real mvp_logit_approx(real x);
  vector mvp_logit_approx(vector x);
  row_vector mvp_logit_approx(row_vector x);  
  matrix mvp_logit_approx(matrix x);
  
  
   
   vector lb_ub_lp ( vector y,
                     real lb,
                     real ub) {
 
    int N = num_elements(y); 
    vector[N] tanh_y; 
    tanh_y = mvp_tanh_approx(y);
      target +=  - mvp_log_approx(2)  +  mvp_log_approx( (ub - lb) * (1 - square(tanh_y))) ;
    return lb +  (ub - lb) *  0.5 * (1 + tanh_y) ;   
    
  }  
 
    real lb_ub_lp ( real y,
                    real lb,
                    real ub) {
      
       real  tanh_y = mvp_tanh_approx(y);
       target +=  - mvp_log_approx(2)  +  mvp_log_approx( (ub - lb) * (1 - square(tanh_y))) ;
      return lb +  (ub - lb) *  0.5 * (1 + tanh_y) ; 
    
  } 
  
  
  
 
  
}

 

data { 
   
  int<lower=1> N;
  vector[N] y;
  int<lower=1> n_covs;
  matrix[N, n_covs] X;
  
}

parameters {
  
 // real alpha; // intercept
  vector[n_covs] beta; // coefficients for cts covariates 
  /// vector[N] u_raw;
  
} 

 
model {
   
  //// likelihood
  {
    
      // vector[N] u = lb_ub_lp(u_raw, 0.0, 1.0);
      // vector[N] Xbeta = X * beta;
      // vector[N] Bound_Z = - Xbeta;
      // vector[N] Bound_U_Phi_Bound_Z = mvp_Phi(Bound_Z);
      // vector[N] Z_std_norm   =     mvp_inv_Phi((y .*  Bound_U_Phi_Bound_Z  +  (y -   Bound_U_Phi_Bound_Z) .* (y + (y - 1.0) ) .*  u)  );
      // 
      // vector[N] log_lik  =         mvp_log_approx( y .* (1.0 -  Bound_U_Phi_Bound_Z) + (y - 1.0) .*  Bound_U_Phi_Bound_Z .* (y + (y - 1.0)) );
     
     

    vector[N] probs_vector = mvp_inv_logit(X * beta);
    vector[N] lik = y .* probs_vector + (1 - y) .* (1.0 - probs_vector);
    vector[N] log_lik = mvp_log_approx(lik);
      
      target +=  sum(log_lik);
    
  }
  
  /// priors
 ///   alpha ~ normal(0.0, 1.0);
  beta  ~ normal(0.0, 5.0);
   
}





















