data{
	int N_train; //number of observations training and validation set
	int p; //number of predictors
	real y_train[N_train]; //response vector
	matrix[N_train, p] X_train; //model matrix
	//test set
	int N_test; //number of observations test set
	matrix[N_test, p] X_test; //model matrix test set
}
parameters{
	real mu; //intercept
	real<lower=0> sigma2; //error variance
	vector[5] beta_g1; // regression parameters group 1
	vector[5] beta_g2; // regression parameters group 2
	vector[5] beta_g3; // regression parameters group 3
	vector[15] beta_g4; // regression parameters group 4
	//hyperparameters prior
	real<lower=0> lambda; //penalty parameter
	vector<lower=0>[4] tau2;
}
transformed parameters{
	vector[10] beta1;
	vector[20] beta2;
	vector[p] beta;
	real<lower=0> lambda2; 
	real<lower=0> sigma; //error sd
	vector[N_train] linpred; //mean normal model
	beta1 = append_row(beta_g1, beta_g2); 
	beta2 = append_row(beta_g3, beta_g4);
	beta = append_row(beta1, beta2);
	lambda2 = lambda*lambda;
	sigma = sqrt(sigma2);
	linpred = mu + X_train*beta;
}
model{
 //prior regression coefficients: group lasso
 beta_g1 ~ normal(0, sqrt(sigma2*tau2[1]));
 beta_g2 ~ normal(0, sqrt(sigma2*tau2[2]));
 beta_g3 ~ normal(0, sqrt(sigma2*tau2[3]));
 beta_g4 ~ normal(0, sqrt(sigma2*tau2[4]));
 tau2 ~ gamma((p + 1)/2, lambda2/2);
 lambda ~ cauchy(0, 1);
	
 //priors nuisance parameters: uniform on log(sigma^2) & mu
	target += -2 * log(sigma); 
	
 //likelihood
	y_train ~ normal(linpred, sigma);
}
generated quantities{ //predict responses test set
	real y_test[N_test]; //predicted responses
	for(i in 1:N_test){
		y_test[i] = normal_rng(mu + X_test[i,] * beta, sigma);
	}
}	
