// NO nuisance block: the FIRST declaration is a simplex (dimension-changing
// constraint) and it is MAIN, not nuisance.
data {
  array[10] int y;
}
parameters {
  simplex[4] theta;
  real mu;
}
model {
  theta ~ dirichlet(rep_vector(1.0, 4));
  mu ~ normal(0, 1);
  y ~ multinomial(theta);
}
