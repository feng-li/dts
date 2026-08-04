data {
  int<lower=3> T;
  int<lower=1> p;
  matrix[T, p] X;
  vector[T] y;
  real<lower=0> likelihood_temper;
  real<lower=0> prior_temper;
}

parameters {
  real<lower=-1, upper=1> kappa1;
  real<lower=-1, upper=1> kappa2;
  vector[p] beta;
  real log_sigma2;
}

transformed parameters {
  real phi1 = kappa1 * (1 - kappa2);
  real phi2 = kappa2;
  real<lower=0> sigma2 = exp(log_sigma2);
  vector[T] residual = y - X * beta;
}

model {
  target += prior_temper * normal_lpdf(beta | 0, 1);
  target += prior_temper * normal_lpdf(log_sigma2 | 0, 1);

  target += likelihood_temper * normal_lpdf(
    residual[3:T]
    | phi1 * residual[2:(T - 1)] + phi2 * residual[1:(T - 2)],
      sqrt(sigma2)
  );
}
