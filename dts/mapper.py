def mappler(pdf):
    """
    MCMC sampler for each data shard for Spark. The input and output should be both pandas DataFrames
    """

    # Parameters setting
    TFI_term = model_conf['TFI_term']
    G = mcmc_conf['partition_num']
    data = pdf.to_numpy()


    if TFI_term:
        n_params = q + p + 3
        Last_ARMA = 3
    else:
        n_params = q + p + 1
        Last_ARMA = 1

    #Sampling
    draws = []
    log_pi = []
    Acceptance = np.zeros((n_samples*G, 1))
    w = []

    params = 0.01*np.ones(n_params)
    params = params + sps.norm.rvs(0, 0.1, size = len(params))

    omega_full = 2*np.pi*np.arange(1,int(n/2)+1)/len(data)

    # 2.5 posterior distribution function
    #2.9.2 find own MAP(paramsStar), as start point
    log_p = lambda x: log_prior(x, 0, 1, Last_ARMA, TFI_term) / G + np.sum(whittle_log_likelihood(x, q, p, I_pg_shard, TFI_term, omega_shard))
    def obj(params): return -log_p(params)

    jcb = grad(obj)

    r_logp, H_logp = grad(log_p), hessian(log_p)
    hs = hessian(obj)

    lb = [-1]*len(params) # Constrains it to the stationary region after using the partial autocorrelation parameterisation.
    ub = [1]*len(params)


    if TFI_term:
        lb[-3:] = [-30, -30, -30]
        ub[-3:] = [30, 30, 30]
    else:
        lb[-1:] = [-30]
        ub[-1:] = [30]

    bnds = Bounds(lb, ub, keep_feasible=True)

    # 2.9.1 each worker gets its shard of data(periodgram)
    data_shard = data[S[ind]]
    I_pg_shard = I_pg_full[I[ind]]
    omega_shard = omega_full[I[ind]]
    if exact_L:
        def p_gram_shard(x):
            id = int(np.floor((len(x))/2))
            return np.square(np.abs(x[0:(id)]))/(2 * np.pi * len(x))
        if G > 1:
            I_pg_shard = p_gram_shard(fft(data_shard))
            omega_shard = 2*np.pi*np.arange(1,int(len(data_shard)/2)+1)/len(data_shard)

    res = minimize(obj, bounds=bnds, jac = jcb, hess = hs, method = 'trust-constr', x0 = params)
    paramsStar = res.x
    assert(res.success)
    sigma = np.linalg.inv(-H_logp(paramsStar))

    print('\nMAP%s' %(ind+1), paramsStar)
    proposal_width = (2.38/np.sqrt(n_params))*sigma

    # 2.9.3 run sampling, jackknife bias correction and restore all results as draws
    draw, log_pi_g, Acceptance = sampler(q, p, data_shard, I_pg_shard, TFI_term, omega_shard, n_samples, paramsStar, proposal_width, Burn_in, exact=exact_L)

    draws.append(draw)
    log_pi.append(log_pi_g)

    # 2.9.4 calculate sample variance(coveriance) of each shard, inverse them as the weights.
    w.append(np.linalg.inv(np.cov(draw, rowvar=False)))

    AcceptanceRate = np.mean(Acceptance)
    print('\nAcceptanceRate:', AcceptanceRate)

    return out
