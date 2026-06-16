data 
{
    int<lower=0> K;
    real<lower=0> m;
    int<lower=0> p;
    real y[K];
    matrix[p,K] X;
}
parameters 
{
    real alpha;
    vector[2] phi;
    vector[p] beta;
    real<lower=0.00001> sigmasq;
}

/*
model 
{
    target += normal_lpdf(alpha | 0, 10);
    target += normal_lpdf(beta | 0, 10);
    target += normal_lpdf(phi | 0, 10);
    target += inv_gamma_lpdf(sigmasq | 3, 10);
    
    target += m*normal_lpdf(y[1] | alpha+beta'*X[:,1], sqrt(sigmasq));
    target += m*normal_lpdf(y[2] | alpha+beta'*X[:,2], sqrt(sigmasq));
    for (t in 3:K) 
    {
        target += m*normal_lpdf(y[t] | phi[1]*(y[t-1]-alpha-beta'*X[:,t-1])+phi[2]*(y[t-2]-alpha-beta'*X[:,t-2]), sqrt(sigmasq));
    }
}
*/

model 
{
    target += normal_lpdf(alpha | 0, 10);
    target += normal_lpdf(beta  | 0, 10);
    target += normal_lpdf(phi   | 0, 10);
    target += inv_gamma_lpdf(sigmasq | 3, 10);

    target += m * normal_lpdf(y[1] | alpha + beta' * X[:,1], sqrt(sigmasq));
    target += m * normal_lpdf(y[2] | alpha + beta' * X[:,2], sqrt(sigmasq));

    for (t in 3:K) 
    {
        target += m * normal_lpdf(
            y[t] |
            (alpha + beta' * X[:,t])
            + phi[1] * (y[t-1] - alpha - beta' * X[:,t-1])
            + phi[2] * (y[t-2] - alpha - beta' * X[:,t-2]),
            sqrt(sigmasq)
        );
    }
}