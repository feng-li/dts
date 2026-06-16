library(dplyr)
library(tidyr)
library(stringr)

library(artfima)
source("_mh_artfima.R")   # compute_percentiles + W1_rel

simulation_root <- "simulations"

cases <- c("case1", "case2")
Gs    <- c(5, 10, 20)

results <- lapply(cases, function(case) {
  
  cat("\nProcessing", case, "\n")
  
  ## ---- full posterior quantiles (phi) ----
  full_draws <- readRDS(
    file.path(simulation_root, case, "full", "full_mcmc.RDS")
  )
  q_full <- compute_percentiles(full_draws[, "phi"])
  
  ## ---- DC posteriors ----
  w1_by_G <- sapply(Gs, function(G) {
    
    Gdir <- file.path(simulation_root, case, paste0("G", G))
    shard_files <- list.files(Gdir, full.names = TRUE)
    
    ## per-shard quantiles
    q_shards <- lapply(shard_files, function(f) {
      draws <- readRDS(f)
      compute_percentiles(draws[, "phi"])
    })
    
    ## percentile averaging
    q_dc <- Reduce("+", q_shards) / length(q_shards)
    
    ## W1 distance
    W1_rel(q_full, q_dc, method = "riemann")
  })
  
  tibble(
    case = case,
    G    = Gs,
    W1   = as.numeric(w1_by_G)
  )
})

w1_results <- bind_rows(results)
print(w1_results)