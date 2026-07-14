library("ltsa")
library("gsl")
library("SuperGauss")
library("bayesplot")
library("hexbin")
library("stringr")

library(artfima)
source("_mh_artfima.R")

## --------------------------------------------------
## Output directory
## --------------------------------------------------
root_dir <- "simulations"
if (!dir.exists(root_dir)) dir.create(root_dir)

## --------------------------------------------------
## Simulation settings from Ou et al.
## --------------------------------------------------
param_df <- data.frame(
  case   = c("case1", "case2"),
  lambda = c(0.005, 0.1),
  d      = c(0.1, 0.3),
  phi    = c(0.9, 0.99),
  stringsAsFactors = FALSE
)

## --------------------------------------------------
## MCMC settings
## --------------------------------------------------
n_iter <- 6000

for (i in seq_len(nrow(param_df))) {
  
  pars <- param_df[i, ]
  print(pars)
  
  ## Load simulated data
  data_path <- paste0(pars$case, "_y.csv")
  x <- read.csv(data_path)[[1]]
  
  ## Create output directory
  out_dir <- file.path(root_dir, pars$case)
  if (!dir.exists(out_dir)) dir.create(out_dir)
  
  ## Proposal standard deviations
  phi_prop <- if (pars$phi == 0.9) 0.008 else 0.002
  d_prop   <- 0.02
  
  sd_proposals <- list(
    d = d_prop,
    lambda = 0.3,
    phi = phi_prop,
    sigma2 = 0.01
  )
  
  ## --------------------------------------------------
  ## Full-data posterior
  ## --------------------------------------------------
  cat("\nRunning full-data MCMC\n")
  
  full_mcmc <- mh_artfima(
    x,
    n_iter = n_iter,
    init = list(
      d = pars$d,
      lambda = pars$lambda,
      phi = pars$phi,
      sigma2 = 1
    ),
    sd_prop = sd_proposals,
    incl_sample = list(
      sample_d = TRUE,
      sample_lambda = FALSE,
      sample_phi = TRUE,
      sample_sigma2 = FALSE
    ),
    temper = 1
  )
  
  dir.create(file.path(out_dir, "full"), showWarnings = FALSE)
  
  saveRDS(
    full_mcmc,
    file.path(out_dir, "full", "full_mcmc.RDS")
  )
  
  ## --------------------------------------------------
  ## Divide-and-conquer posterior
  ## --------------------------------------------------
  for (G in c(5, 10, 20)) {
    
    cat(sprintf("\nRunning DC MCMC: G = %d\n", G))
    
    Kdir <- file.path(out_dir, paste0("G", G))
    dir.create(Kdir, showWarnings = FALSE)
    
    n <- length(x)
    m <- floor(n / G)
    
    for (k in 1:G) {
      
      ## Consecutive data partition
      x_sub <- x[((k - 1) * m + 1):(k * m)]
      
      sub_mcmc <- mh_artfima(
        x_sub,
        n_iter = n_iter,
        init = list(
          d = pars$d,
          lambda = pars$lambda,
          phi = pars$phi,
          sigma2 = 1
        ),
        sd_prop = sd_proposals,
        incl_sample = list(
          sample_d = TRUE,
          sample_lambda = FALSE,
          sample_phi = TRUE,
          sample_sigma2 = FALSE
        ),
        temper = G
      )
      
      saveRDS(
        sub_mcmc,
        file.path(
          Kdir,
          paste0("k", sprintf("%02d", k), "_mcmc.RDS")
        )
      )
    }
  }
}