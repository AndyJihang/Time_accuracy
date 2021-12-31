data {
  int<lower=1> N;              // number of students
  int<lower=1> K;              // number of questions
  int<lower=1> Nobs;              // number of observations
  int<lower=1,upper=N> ii[Nobs];  // student for observation n
  int<lower=1,upper=K> jj[Nobs];  // question for observation n
  int<lower=0,upper=1> y[Nobs];   // correctness for observation n
}

parameters {
  vector[N] theta;             // ability for j - mean
  vector[K] beta;              // difficulty for k
}

model {
  theta ~ std_normal(); // normal(y| 0,1)
  beta ~ normal(0, 5);
  for(n in 1:Nobs){
   y[n] ~ bernoulli_logit(theta[ii[n]] - beta[jj[n]]);
  }
}
