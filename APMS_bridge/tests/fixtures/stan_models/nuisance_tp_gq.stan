// nuisance + main + transformed parameters + generated quantities (checks the
// constrained output indexing / naming through the reconstruction path).
data {
  int<lower=0> n_u;
  int<lower=1> n_tests;
}
parameters {
  array[n_u] row_vector[n_tests] u_raw;
  simplex[4] pi;
  real mu;
}
transformed parameters {
  real mu_tp = mu * 2.0;
}
model {
  pi ~ dirichlet(rep_vector(1.0, 4));
  mu ~ normal(0, 1);
  for (i in 1:n_u) target += normal_lpdf(u_raw[i] | mu, 1.0);
}
generated quantities {
  real mu_gq = mu + 1.0;
  real pi1 = pi[1];
}
