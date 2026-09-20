// NO nuisance block: multiple main declarations.
parameters {
  real mu;
  real<lower=0> sigma;
  vector[3] beta;
}
model {
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1);
  beta ~ normal(0, 1);
}
