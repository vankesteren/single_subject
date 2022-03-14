# Bayesian analysis of single-subject research design
# with count outcomes
library(rstanarm)
library(ggplot2)
library(patchwork)
library(bridgesampling)

# Load data
df <- readRDS("example_data.rds")

# Basic visualisation
plt_data <-
  ggplot(df, aes(x = day, y = y, fill = treat)) +
  geom_point(pch = 21, colour = "white", size = 2.3) + 
  theme_minimal() + 
  labs(x = "Day", y = "Seizures", fill = "Treatment", title = "Raw data") +
  ylim(0, 5)
plt_data


# Fit the null model: poisson model with only an intercept
fit_int <- stan_glm(
  formula = y ~ 1, 
  family = poisson(), 
  data = df, 
  diagnostic_file = "chains/fit_int.csv" # needed for bayes factors later
)

# Fit model 1: poisson model with intercept and treatment effect
fit_treat <- stan_glm(
  formula = y ~ 1 + treat, 
  family = poisson(), 
  data = df, 
  diagnostic_file = "chains/fit_treat.csv"
)

# two more models to check
# Fit a gaussian model
fit_lm <- stan_glm(
  formula = y ~ 1 + treat, 
  family = gaussian(), 
  data = df, 
  diagnostic_file = "chains/fit_lm.csv" # needed for bayes factors later
)


# Fit model 2: poisson model with intercept and treatment effect and autocorrelation
fit_treat_ar <- stan_glm(
  formula = y ~ 1 + treat + yl, 
  family = poisson(), 
  data = df, 
  diagnostic_file = "chains/fit_treat_ar.csv"
)


# Posterior predictive check for each of these models
ppc_plot <- function(fit, df) {
  ppred <- posterior_predict(fit, df)
  ppdf <- data.frame(t(apply(ppred, 2, \(x) {
    c(quantile(x, prob = 0.05), mean(x), quantile(x, prob = 0.95))
  })))
  names(ppdf) <- c("lower", "est", "upper")
  ppdf$day <- df$day
  ppdf$treat <- df$treat
  ppdf$y <- df$y
  ggplot(ppdf, aes(x = day)) +
    geom_point(aes(y = y, fill = treat), pch = 21, colour = "white", size = 2.3) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "seagreen", alpha = 0.3) +
    geom_line(aes(y = est)) +
    labs(x = "Day", y = "Seizures", fill = "Treatment") +
    theme_minimal() +
    ylim(-1.5, 5) +
    scale_fill_hue(guide = "none")
}

ppc_lm <- ppc_plot(fit_lm, df) + ggtitle("Normal model")
ppc_int <- ppc_plot(fit_int, df) + ggtitle("Intercept-only")
ppc_treat <- ppc_plot(fit_treat, df) + ggtitle("Treatment effect")
ppc_treat_ar <- ppc_plot(fit_treat_ar, df[-1,]) + ggtitle("Treatment and autocorrelation")

(ppc_lm + ppc_int) / (ppc_treat + ppc_treat_ar)

# Model comparison

# should we use poisson outcome?
bf(bridge_sampler(fit_treat), bridge_sampler(fit_lm))
# yes

# is there a treatment effect?
bf(bridge_sampler(fit_treat), bridge_sampler(fit_int))
# yea

# should we use autoregression to correct for time-series correlation?
bf(bridge_sampler(fit_treat_ar), bridge_sampler(fit_treat))
# nope

# plot the posterior distribution of the treatment effect
loo_compare(fit_treat, fit_lm)
