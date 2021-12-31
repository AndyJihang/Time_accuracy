data{
  int<lower = 1> D; // number of dimensions
  int<lower = 1> N; // number of examinees
  int<lower = 1> K; // number of items
  int<lower = 1> Nobs; // number of observed data points
  int<lower = 1, upper = K> jj[Nobs]; // item id
  int<lower = 1, upper = N> ii[Nobs]; // person id
  int<lower = 0, upper = 1> y[Nobs]; //responses
  real logt[Nobs]; //log response times
  vector[D] Zero; // a vector of Zeros (fixed means for person parameter)
  matrix[D,D] W; // scale matrix (a binary unit matrix)
}
parameters{
  cov_matrix[D] Sigma;
  real<lower = 0> alpha_inv;
  row_vector[D] PersPar[N]; //person parameter (ability, time)
  vector[K] TimeIntense;
  vector[K] ItemPar; //difficulty parameter for item k
}

model{
  Sigma ~ inv_wishart (2.0, W);
  // prior person parameter
  PersPar~ multi_normal(Zero,Sigma);
  // prior item parameter
  TimeIntense ~ normal(0,5);
  ItemPar ~ normal(0,5);
  alpha_inv ~ normal(0,1) T[0,];
  for(n in 1:Nobs){
   y[n] ~ bernoulli_logit(PersPar[ii[n],1] - ItemPar[jj[n]]);
   logt[n] ~ normal(TimeIntense[jj[n]] - PersPar[ii[n],2], (alpha_inv^(-2)));
  }
}
