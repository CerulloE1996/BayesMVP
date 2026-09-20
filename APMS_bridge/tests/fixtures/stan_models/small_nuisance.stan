// SMALL nuisance block (size 3, below the old n_nuisance <= 10 shortcut
// threshold): must still be passed to the log-density in full.
data {
  vector[20] y;
}
parameters {
  vector[3] u;              // first declaration -> nuisance block (size 3)
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1);
  u ~ normal(0, 1);
  y ~ normal(mu + sum(u) / 20.0, sigma);
}
