data {
  int<lower=0> n; // number of observations
  int<lower=0> d; // number of predictors
  int<lower=0> y[n]; // outputs
  matrix[n,d] x; // inputs
  int<lower=0> n_rep;
}

parameters {
  real alpha;
  real<lower=0> phi;
  vector[d] beta;
}

model {
  phi ~ exponential(0.5);
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  target += n_rep * neg_binomial_2_log_glm_lpmf(y | x, alpha, beta, phi + 0.1);
}
