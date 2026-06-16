mh_artfima <- function(
  x,
  n_iter = 1000,
  init = list(d = 0.4, lambda = 0.01, phi = 0.99, sigma2 = 1),
  sd_prop = list(d = 0.01, lambda = 0.001, phi = 0.001, sigma2 = 0.01),
  incl_sample = list(
    sample_d = TRUE,
    sample_lambda = TRUE,
    sample_phi = TRUE,
    sample_sigma2 = TRUE
  ),
  temper = 1
) {

  acf_maxlag <- length(x) - 1
  acf_n <- length(x) - 1

  ## ------ log-posterior ------
  logpost <- function(par) {
    # parameter bounds
    if (par$d <= 0 ||
        par$d >= 0.5 ||
        par$lambda <= 0 ||
        abs(par$phi) >= 1 ||
        par$sigma2 <= 0) {
          return(-Inf)
        }

    # acf <- artfimaTACVF(
    #   d      = par$d,
    #   lambda = par$lambda,
    #   phi    = par$phi,
    #   theta  = numeric(0),
    #   maxlag = acf_maxlag,
    #   sigma2 = par$sigma2
    # )

    acf <- fast_artfima_acf(
        n = acf_n,
        d = par$d,
        lambda = par$lambda,
        phi = par$phi
      )

    ll <- temper * dnormtz(x, mu = 0, acf = acf, log = TRUE, method = "gschur")

    # just likelihood at the moment, need to include priors
    ll
  }

  ## ------ storage & initial state ------
  out <- matrix(NA_real_, nrow = n_iter, ncol = 4,
                dimnames = list(NULL, c("d", "lambda", "phi", "sigma2")))
  cur <- init
  lcur <- logpost(cur)
  acc  <- 0L

  prop_d <- cur$d
  prop_lambda <- cur$lambda
  prop_phi <- cur$phi
  prop_sigma2 <- cur$sigma2

  ## ------ main MH loop ------
  for (i in seq_len(n_iter)) {

    cat(sprintf('\rIteration %d/%d', i, n_iter))

    if (incl_sample$sample_d) {
      prop_d <- cur$d + rnorm(1, 0, sd_prop$d)
    }

    if (incl_sample$sample_lambda) {
      prop_lambda <- 10^(log10(cur$lambda) + rnorm(1, 0, sd_prop$lambda))
    }

    if (incl_sample$sample_phi) {
      prop_phi <- cur$phi + rnorm(1, 0, sd_prop$phi)
    }

    if (incl_sample$sample_sigma2) {
      prop_sigma2 <- cur$sigma2 + rnorm(1, 0, sd_prop$sigma2)
    }

    prop <- list(
      d = prop_d,
      lambda = prop_lambda,
      phi = prop_phi,
      sigma2 = prop_sigma2
    )

    lprop <- logpost(prop)    # returns –Inf if any bound violated

    # print(cur$phi)
    # print(prop_phi)
    # print(lcur)
    # print(lprop)

    if (log(runif(1)) < lprop - lcur) {  # accept
      cur  <- prop
      lcur <- lprop
      acc  <- acc + 1L
    }

    out[i, ] <- c(cur$d, cur$lambda, cur$phi, cur$sigma2)
  }

  attr(out, "accept_rate") <- acc / n_iter
  out
}

psd1_to_acvf <- function(P_os, df, maxlag = NULL, include_nyq = TRUE) {
  stopifnot(is.numeric(P_os), length(P_os) >= 1, is.numeric(df), df > 0)

  M <- length(P_os)
  if (include_nyq) {
    # N = 2*(M-1); indices: 0,1,...,N/2, ..., N-1
    N <- 2*(M - 1)
    S0   <- P_os[1]/2                 # DC
    mid  <- if (M > 2) P_os[2:(M-1)]/2 else numeric(0)  # halve interior
    Snyq <- P_os[M]/2                 # Nyquist
    S_two <- c(S0, mid, Snyq, rev(mid))
  } else {
    # If no explicit Nyquist bin was provided
    N <- 2*M - 2
    S0  <- P_os[1]
    mid <- if (M > 1) P_os[2:M]/2 else numeric(0)
    S_two <- c(S0, mid, rev(mid))
  }

  # IFFT: R's fft() inverse is unnormalized; divide by N.
  # Multiply by df to approximate the continuous inverse FT (Riemann sum).
  acvf_full <- 2 * Re(fft(S_two, inverse = TRUE))
  acvf_full <- acvf_full * df

  # Return nonnegative lags
  acvf <- acvf_full[1:(if (is.null(maxlag)) length(acvf_full) else (maxlag + 1))]
  acvf
}

fast_artfima_acf <- function(n, d, lambda, phi) {
  psd <- artfimaSDF(n=2*n, d=d, lambda=lambda, phi=phi, plot = "none")
  psd1_to_acvf(psd, 1/(2*n), include_nyq = TRUE)[1:(n+1)]
}

compute_percentiles <- function(x) {
  probs <- seq(0.01, 0.99, by = 0.01)
  quantile(x, probs = probs, type = 7, names = FALSE)
}

wasserstein_from_percentiles <- function(
  qP, qQ, p = 2, method = c("trap","riemann")
) {
  stopifnot(length(qP) == 99L, length(qQ) == 99L)
  method <- match.arg(method)
  du <- 0.01
  d <- abs(qP - qQ)^p
  wp_p <- if (method == "trap") {
    du * (0.5 * d[1] + sum(d[2:98]) + 0.5 * d[99])
  } else {
    du * sum(d)
  }
  wp_p^(1 / p)
}

W1_rel <- function(qP, qQ, method = c("trap","riemann")) {
  method <- match.arg(method)
  du <- 0.01
  # numerator: W1 over 1..99%
  num <- wasserstein_from_percentiles(qP, qQ, p = 1, method = method)
  # denominator: mean abs deviation of Q about its median over same grid
  m <- qQ[50]
  a <- abs(qQ - m)
  denom <- if (method == "trap") du * (0.5*a[1] + sum(a[2:98]) + 0.5*a[99]) else du * sum(a)
  num / denom
}