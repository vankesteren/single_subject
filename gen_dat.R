# Example data generation for single-subject study with count outcomes
set.seed(456)
logmean_control <- 0.2
treatment_effect <- -3.8
ar <- 0.1

treat <- c(rep(1, 28), rep(0, 21), rep(1, 14), rep(0, 21))
N <- length(treat)
lograte <- numeric(N)
y <- integer(N)

lograte[1] <- treatment_effect
y[1] <- rpois(1, exp(logmean_control + treatment_effect))
for (n in 2:N) {
  mu <- logmean_control + treat[n]*treatment_effect
  lograte[n] <- mu + ar*(lograte[n-1]-mu)
  y[n] <- rpois(1, exp(lograte[n]))
}

df <- data.frame(day = 1:N, y = y, treat = factor(treat))

saveRDS(df, "example_data.rds")
