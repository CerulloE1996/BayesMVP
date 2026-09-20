// nuisance-present model: FIRST declaration is the nuisance block (u_raw),
// then a simplex (dimension-changing constraint) and unconstrained scalars.
data {
  int<lower=0> n_u;
  int<lower=1> n_tests;
}
parameters {
  array[n_u] row_vector[n_tests] u_raw;   // FIRST declaration -> nuisance block
  simplex[4] pi;
  real mu;
  real<lower=0> sigma;
}
model {
  pi ~ dirichlet(rep_vector(1.0, 4));
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1);
  for (i in 1:n_u) target += normal_lpdf(u_raw[i] | mu, sigma);
}
