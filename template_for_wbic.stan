data {
    int<lower=1> K;
    vector<lower=0>[K] alpha_class;
    int<lower=1> N;
    int<lower=1> n_reg;
    int<lower=1> n_cont;
    vector[n_cont] X_cont[N];
    vector[n_reg] X_reg[N];
    real y[N];
    real<lower=0> alpha_lasso;
    
    vector[n_cont] mu_mu_X_cont;
    cov_matrix[n_cont] sigma_mu_X_cont;
    
    vector<lower=0>[2] alpha_bern;
    real<lower=0> sigma_y;
    
    real y_mean;
    real<lower=0> y_std;
    
// 離散変数のデータを追加します。
// discrete data start //
// discrete data end //

// ゼロ過剰ポアソン分布のデータを追加します。
// zero poi data start //
// zero poi data end //

}

parameters {    
//    real beta1_0;
    vector[K] beta1;
    vector[n_reg] beta2[K];
    
    simplex[K] pi;
    vector[n_cont] phi_mu_X_cont[K];
    vector<lower=0>[n_cont] diag_sigma_X_cont[K];


// 離散変数のパラメータを追加します。
// discrete parameters start //
// discrete parameters end //
    
// ゼロ過剰ポアソン分布のパラメータを追加します。
// zero poi parameters start //
// zero poi parameters end //

}

transformed parameters{
    matrix[n_cont,n_cont] phi_sigma_X_cont[K];
    for (k in 1:K) {
        phi_sigma_X_cont[k] = diag_matrix(diag_sigma_X_cont[k]);
    }
}

model {    
    pi ~ dirichlet(alpha_class);
    
    for (k in 1:K) {    
        beta1[k] ~ normal(y_mean, 5*y_std);
        beta2[k] ~ double_exponential(0, 1/alpha_lasso);
        
        phi_mu_X_cont[k] ~ multi_normal(mu_mu_X_cont, sigma_mu_X_cont);
        
        diag_sigma_X_cont[k] ~ cauchy(0, 2.5);
        

// 離散変数の事前分布を追加します。
// discrete model prior start //
// discrete model prior end //

// ゼロ過剰ポアソン分布の事前分布を追加します。
// zero poi model prior start //
// zero poi model prior end //

    }
    
    for (n in 1:N) {
        real eta[K];
        real mu[K];
        
        for (k in 1:K) {
            mu[k] = beta1[k] + dot_product(X_reg[n], beta2[k]);
            eta[k] = categorical_lpmf(k | pi);
            eta[k] += normal_lpdf(y[n] | mu[k], sigma_y);
            eta[k] += multi_normal_lpdf(X_cont[n] | phi_mu_X_cont[k], phi_sigma_X_cont[k]);


// 離散変数に関する計算を追加します。
// discrete model likelihood start //
// discrete model likelihood end //            


// ゼロ過剰ポアソン分布に関する計算を追加します．
// zero poi model likelihood start //
// zero poi model likelihood end //  


        }
        
        target += 1/log(N) * log_sum_exp(eta);
    }
}

generated quantities {
    real class_lp[N, K];
    real log_lik[N];
    
    for (n in 1:N) {
        log_lik[n] = 0.0;
    }

    for (n in 1:N) {
        real mu[K];
        
        for (k in 1:K) {
            real eta;
                
            mu[k] = beta1[k] + dot_product(X_reg[n], beta2[k]);
            eta = categorical_lpmf(k | pi);
            eta += normal_lpdf(y[n] | mu[k], sigma_y);
            eta += multi_normal_lpdf(X_cont[n] | phi_mu_X_cont[k], phi_sigma_X_cont[k]);

// 離散変数に関する計算を追加します。
// discrete generated quantities start //
// discrete generated quantities end //   

// ゼロ過剰ポアソン分布に関する計算を追加します
// zero poi generated quantities start //
// zero poi generated quantities end //   

            class_lp[n, k] = eta;
        
        }
        log_lik[n] += log_sum_exp(class_lp[n]);
    }
}
