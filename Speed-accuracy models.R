setwd("/Users/Jihang/Desktop/Time_accuracy")
### Windows
#install.packages(c("callr", "fs", "processx"))
#install.packages("devtools")
#library(devtools)
#Sys.setenv(PATH = paste("C:\\Rtools\\bin", Sys.getenv("PATH"), sep=";"))
#Sys.setenv(PATH = paste("C:\\Rtools\\mingw_64\\bin", Sys.getenv("PATH"), sep=";"))
#install.packages("rstan")
#library(rstan)

### MAC
#Sys.setenv(MAKEFLAGS = "-j4") # four cores used
#install.packages(c("Rcpp", "RcppEigen", "RcppParallel", "StanHeaders"), type = "source")
#install.packages("rstan", type = "source")

#install.packages("mirt")

library(rstan)
library(foreign)
library(MASS)
library(dplyr)
library(mirt)
library(shinystan)

load("/Users/Jihang/Desktop/Time_accuracy/speed_accuracy_new.Rdata")
speed_accuracy_new <- speed_accuracy_new[1:200,]
x<-log(as.numeric(as.character(speed_accuracy_new[,2]))/1000)
hist(x, breaks=100,prob=F, xlab="Log time", main="Log response time distribution")
#hist(x, breaks=10000,prob=F,xlim=c(0,50), xlab="Time", main="Histogram")

Response <- matrix(0,nrow=nrow(speed_accuracy_new),ncol=11)
Response[,1] <- recode(speed_accuracy_new$CM034Q01S,`Full credit`=1,`No credit`=0,`Not Reached`=0,`No Response`=0)
Response[,2] <- recode(speed_accuracy_new$CM305Q01S,`Full credit`=1,`No credit`=0,`Not Reached`=0,`No Response`=0, `Not Applicable`=0)
Response[,3] <- recode(speed_accuracy_new$CM496Q01S,`Full credit`=1,`No credit`=0,`Not Reached`=0,`No Response`=0)
Response[,4] <- recode(speed_accuracy_new$CM496Q02S,`Full credit`=1,`No credit`=0,`Not Reached`=0,`No Response`=0)
Response[,5] <- recode(speed_accuracy_new$CM423Q01S,`Full credit`=1,`No credit`=0,`Not Reached`=0,`No Response`=0)
Response[,6] <- recode(speed_accuracy_new$DM406Q01C,`1 - Full credit`=1,`0 - No credit`=0,`Not Reached`=0,`No Response`=0, `Not Applicable`=0)
Response[,7] <- recode(speed_accuracy_new$DM406Q02C,`1 - Full credit`=1,`0 - No credit`=0,`Not Reached`=0,`No Response`=0, `Not Applicable`=0)
Response[,8] <- recode(speed_accuracy_new$CM603Q01S,`Full credit`=1,`No credit`=0,`Not Reached`=0,`No Response`=0)
Response[,9] <- recode(speed_accuracy_new$CM571Q01S,`Full credit`=1,`No credit`=0,`Not Reached`=0,`No Response`=0)
Response[,10] <- recode(speed_accuracy_new$CM564Q01S,`Full credit`=1,`No credit`=0,`Not Reached`=0,`No Response`=0)
Response[,11] <- recode(speed_accuracy_new$CM564Q02S,`Full credit`=1,`No credit`=0,`Not Reached`=0,`No Response`=0)

Time <- matrix(0,nrow=nrow(speed_accuracy_new),ncol=11)
Time[,1] <- log(as.numeric(as.character(speed_accuracy_new$CM034Q01T))/1000)
Time[,2] <- log(as.numeric(as.character(speed_accuracy_new$CM305Q01T))/1000)
Time[,3] <- log(as.numeric(as.character(speed_accuracy_new$CM496Q01T))/1000)
Time[,4] <- log(as.numeric(as.character(speed_accuracy_new$CM496Q02T))/1000)
Time[,5] <- log(as.numeric(as.character(speed_accuracy_new$CM423Q01T))/1000)
Time[,6] <- log(as.numeric(as.character(speed_accuracy_new$CM406Q01T))/1000)
Time[,7] <- log(as.numeric(as.character(speed_accuracy_new$CM406Q02T))/1000)
Time[,8] <- log(as.numeric(as.character(speed_accuracy_new$CM603Q01T))/1000)
Time[,9] <- log(as.numeric(as.character(speed_accuracy_new$CM571Q01T))/1000)
Time[,10] <- log(as.numeric(as.character(speed_accuracy_new$CM564Q01T))/1000)
Time[,11] <- log(as.numeric(as.character(speed_accuracy_new$CM564Q02T))/1000)

D <- 2 # number of dimensions
N <- nrow(speed_accuracy_new) # number of respondents
K <- 11 # number of items
Nobs <-N*K # number of observed data points
jj <- c(rep(1,N),rep(2,N),rep(3,N),rep(4,N),rep(5,N),rep(6,N),rep(7,N),rep(8,N),rep(9,N),rep(10,N),rep(11,N)) # item id
ii <- rep(seq(1,N),K) # person id
#responses
y <- c(Response[,1],Response[,2],Response[,3],Response[,4],Response[,5],Response[,6],Response[,7],Response[,8],Response[,9],Response[,10],Response[,11])
#log response times
logt <- c(Time[,1],Time[,2],Time[,3],Time[,4],Time[,5],Time[,6],Time[,7],Time[,8],Time[,9],Time[,10],Time[,11])
Zero <- c(0,0) # a vector of Zeros (fixed means for person parameter)
W <- matrix(c(1,0,0,1),2,2) # binary unit matrix

################# 1PL IRT Model ####################################################################################################
# Data preparation
stan_1pl_data <- list(
  N = N,
  K = K,
  Nobs = Nobs,
  ii = ii,
  jj = jj,
  y = y
)

writeLines(readLines("stan_1pl.stan"))
rstan_options(auto_write = TRUE)

# complie model
model <- stan_model("stan_1pl.stan")

# pass data to stan and run model
fit_1pl <- sampling(model, stan_1pl_data, cores = 2, chains = 2, iter = 2000, refresh = 0)
#fit_2pl <- stan(file = "stan_1pl.stan", data = stan_2pl_data, iter = 10000, chains = 2)
print(fit_1pl)

launch_shinystan(fit_1pl)
# Calculate DIC
fit_ss <- extract(fit_1pl, permuted = TRUE)
loglikelihood<-fit_ss$lp__
log_posterior <- -1229.3
DIC <-2 * (log_posterior) - 4 * mean(loglikelihood)
DIC




################# Traditional Model LKJ prior ###########################################################################################
stan_traditional_data <- list(
    D = D,
    N = N,
    K = K,
    Nobs = Nobs,
    ii = ii,
    jj = jj,
    Zero = Zero,
    y = y,
    logt = logt
  )

writeLines(readLines("traditional_LKJ.stan"))
rstan_options(auto_write = TRUE)

# complie model
model <- stan_model("traditional_LKJ.stan")

fit_traditional_LKJ <- sampling(model, stan_traditional_data, cores = 2, chains = 2, iter = 2000, refresh = 0)
#fit_traditional <- stan(file="speed accuracy.stan", data=stan_traditional_data,iter=1000, chains=2)
print(fit_traditional_LKJ)
launch_shinystan(fit_traditional_LKJ)
# Calculate DIC
fit_ss <- extract(fit_traditional_LKJ, permuted = TRUE)
loglikelihood<-fit_ss$lp__
log_posterior <- -1215.1
DIC <-2 * (log_posterior) - 4 * mean(loglikelihood)
DIC

################# Traditional Model Inverse-Wishart prior ###########################################################################################
stan_inv_wishart <- list(
  D = D,
  N = N,
  K = K,
  Nobs = Nobs,
  W=W,
  ii = ii,
  jj = jj,
  Zero = Zero,
  y = y,
  logt = logt
)
writeLines(readLines("traditional_inverse_wishart.stan"))
rstan_options(auto_write = TRUE)

# complie model
model <- stan_model("traditional_inverse_wishart.stan")

fit_inverse_wishart <- sampling(model, stan_inv_wishart, cores = 4, chains = 2, iter = 2000, refresh = 0)
#fit_traditional <- stan(file="speed accuracy.stan", data=stan_traditional_data,iter=1000, chains=2)
print(fit_inverse_wishart)
launch_shinystan(fit_inverse_wishart)
# Calculate DIC
fit_ss <- extract(fit_inverse_wishart, permuted = TRUE)
loglikelihood<-fit_ss$lp__
log_posterior <- -1222.3
DIC <-2 * (log_posterior) - 4 * mean(loglikelihood)
DIC

################# Speed-Accuracy Tradeoff Model ###########################################################################################
stan_tradeoff <- list(
  D = D,
  N = N,
  K = K,
  Nobs = Nobs,
  W=W,
  ii = ii,
  jj = jj,
  Zero = Zero,
  y = y,
  logt = logt
)
writeLines(readLines("tradeoff.stan"))
rstan_options(auto_write = TRUE)

# complie model
model <- stan_model("tradeoff.stan")

fit_tradeoff <- sampling(model, stan_tradeoff, cores = 4, chains = 2, iter = 2000, refresh = 0)
#fit_traditional <- stan(file="speed accuracy.stan", data=stan_traditional_data,iter=1000, chains=2)
print(fit_tradeoff)
launch_shinystan(fit_tradeoff)
fit_ss <- extract(fit_tradeoff, permuted = TRUE)
loglikelihood<-fit_ss$lp__
log_posterior <- -2320.1	
DIC<-2 * (log_posterior) - 4 * mean(loglikelihood)
DIC

