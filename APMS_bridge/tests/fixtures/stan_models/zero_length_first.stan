// Nuisance declaration EXISTS but has ZERO length (empty array first),
// followed by non-empty main parameters.
parameters {
  array[0] real u_raw;      // first declaration, zero length
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1);
}
