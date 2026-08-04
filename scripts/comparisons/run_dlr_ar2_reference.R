#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(cmdstanr)
  library(optparse)
  library(posterior)
})

options <- list(
  make_option("--data", type = "character"),
  make_option("--stan-file", type = "character",
              default = "scripts/comparisons/dlr_ar2_manuscript.stan"),
  make_option("--output", type = "character",
              default = "artifacts/dlr_ar2_reference.rds"),
  make_option("--time-column", type = "character", default = "time"),
  make_option("--y-column", type = "character", default = "y"),
  make_option("--x-columns", type = "character", default = "x"),
  make_option("--groups", type = "character", default = "10,20"),
  make_option("--chains", type = "integer", default = 4L),
  make_option("--parallel-chains", type = "integer", default = 4L),
  make_option("--warmup", type = "integer", default = 1000L),
  make_option("--samples", type = "integer", default = 1000L),
  make_option("--seed", type = "integer", default = 20260802L)
)

parse_names <- function(value) {
  trimws(strsplit(value, ",", fixed = TRUE)[[1]])
}

make_data <- function(y, X, likelihood_temper, prior_temper) {
  list(
    T = length(y),
    p = ncol(X),
    X = unname(X),
    y = as.numeric(y),
    likelihood_temper = likelihood_temper,
    prior_temper = prior_temper
  )
}

extract_draws <- function(fit) {
  variables <- c("kappa1", "kappa2", "phi1", "phi2", "beta",
                 "log_sigma2", "sigma2")
  as.matrix(posterior::as_draws_matrix(fit$draws(variables = variables)))
}

fit_target <- function(model, y, X, likelihood_temper, prior_temper,
                       opt, seed) {
  fit <- model$sample(
    data = make_data(y, X, likelihood_temper, prior_temper),
    chains = opt$chains,
    parallel_chains = opt$parallel_chains,
    iter_warmup = opt$warmup,
    iter_sampling = opt$samples,
    seed = seed,
    refresh = 100
  )
  list(draws = extract_draws(fit), diagnostics = fit$diagnostic_summary())
}

make_blocks <- function(n, groups) {
  cuts <- floor(seq(0, n, length.out = groups + 1L))
  lapply(seq_len(groups), function(group) {
    seq.int(cuts[group] + 1L, cuts[group + 1L])
  })
}

marginal_wasserstein <- function(fits) {
  draws <- lapply(fits, function(fit) fit$draws)
  common <- Reduce(intersect, lapply(draws, colnames))
  n_draws <- min(vapply(draws, nrow, integer(1)))
  out <- matrix(NA_real_, nrow = n_draws, ncol = length(common),
                dimnames = list(NULL, common))
  for (column in seq_along(common)) {
    ordered <- vapply(draws, function(chain) {
      sort(chain[, common[column]])[seq_len(n_draws)]
    }, numeric(n_draws))
    out[, column] <- rowMeans(ordered)
  }
  out
}

main <- function() {
  opt <- parse_args(OptionParser(option_list = options))
  if (is.null(opt$data) || !file.exists(opt$data)) stop("--data CSV is required")
  if (!file.exists(opt$stan_file)) stop("Stan model not found: ", opt$stan_file)

  frame <- read.csv(opt$data, check.names = FALSE)
  x_columns <- parse_names(opt$x_columns)
  required <- c(opt$time_column, opt$y_column, x_columns)
  missing <- setdiff(required, names(frame))
  if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))
  frame <- frame[order(frame[[opt$time_column]]), , drop = FALSE]
  y <- frame[[opt$y_column]]
  X <- as.matrix(frame[, x_columns, drop = FALSE])
  groups <- unique(as.integer(parse_names(opt$groups)))
  if (any(is.na(groups)) || any(groups < 1L)) stop("Invalid --groups")

  model <- cmdstan_model(opt$stan_file)
  result <- list(
    specification = list(
      intercept = FALSE,
      parameterization = "AR(2) partial autocorrelations",
      beta_prior = "normal(0,1)",
      log_sigma2_prior = "normal(0,1)",
      conditional_start = 3L
    ),
    full = fit_target(model, y, X, 1, 1, opt, opt$seed),
    groups = list()
  )

  for (group_count in groups) {
    blocks <- make_blocks(length(y), group_count)
    naive <- vector("list", group_count)
    dcbats <- vector("list", group_count)
    for (group in seq_len(group_count)) {
      index <- blocks[[group]]
      naive[[group]] <- fit_target(
        model, y[index], X[index, , drop = FALSE],
        1, 1 / group_count, opt, opt$seed + 1000L * group_count + group
      )
      dcbats[[group]] <- fit_target(
        model, y[index], X[index, , drop = FALSE],
        group_count, 1, opt, opt$seed + 2000L * group_count + group
      )
    }
    result$groups[[paste0("G", group_count)]] <- list(
      blocks = blocks,
      naive = naive,
      naive_wasserstein = marginal_wasserstein(naive),
      dcbats = dcbats,
      dcbats_wasserstein = marginal_wasserstein(dcbats)
    )
  }

  dir.create(dirname(opt$output), recursive = TRUE, showWarnings = FALSE)
  saveRDS(result, opt$output)
  message("Saved reference posteriors to ", opt$output)
}

main()
