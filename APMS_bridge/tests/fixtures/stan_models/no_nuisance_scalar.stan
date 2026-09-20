// NO nuisance block: a single scalar main parameter.
parameters {
  real mu;
}
model {
  mu ~ normal(0, 1);
}
