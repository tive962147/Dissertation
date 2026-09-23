############################################################
####### SIMULATION STUDY SHRINKAGE PRIORS FULL BAYES #######
####### Author: Sara van Erp                         #######
############################################################

## Note 1: running the simulation takes long, so the final output is available in the file "results_full_bayes.Rdata"
## Note 2: the bridge prior appears in the code, but did not make it into the final manuscript due to computational inefficiency

set.seed(18082017)

########################
####### PACKAGES #######            
########################

library(rstan)
rstan_options(auto_write = TRUE) # to avoid recompiling stan model
library(bayesplot) # to save traceplots
library(parallel) # to run parallel

#####################################
####### SIMULATION CONDITIONS #######            
#####################################
priors <- c("ridge", "student", "lasso", "elastic_net", "group_lasso", "hyperlasso", "bridge", "horseshoe", 
            "regularized_horseshoe", "regularized_horseshoe_true_p0", "regularized_horseshoe_off_p0",
            "normal_mixture_bernoulli", "normal_mixture_uniform")
cond <- 1:8
meth <- "full_bayes"

conditions <- expand.grid(prior=priors, condition=cond, method=meth)

###################################
####### SIMULATION FUNCTION #######            
###################################
sim.fun <- function(pos, cond, reps){
  
  mypath <- file.path("INSERT")
  
  ### Select condition ###
  prior <- cond$prior[pos]
  condition <- cond$condition[pos]
  method <- cond$method[pos]
  
  ### Load generated data ###
  setwd(paste0(mypath, "/simdata"))
  load(list.files()[grep(condition, list.files())])
  simdata <- simdata.split[1:reps]
  
  ### Compile model ###
  setwd(paste0(mypath, "/full_bayes"))
  # select stanfile corresponding to prior
  stanfls <- list.files()[grep("stan", list.files())]
  sel <- grep(paste0("^", prior), stanfls)
  modFB <- stan_model(stanfls[sel])
  
  ### Run model ###
  counter <- 0 # initialize counter to keep track
  out <- lapply(simdata, function(x){
    counter <<- counter + 1
    print(paste("replication", counter))
    # stan data input
    if(prior == "regularized_horseshoe"){
      standat <- list(N_train=nrow(x$trainX), p=ncol(x$trainX), y_train=c(x$trainY), X_train=x$trainX,
                      N_test=nrow(x$testX), X_test=x$testX, scale_global= 1, nu_global= 1, nu_local= 1, 
                      slab_scale=2, slab_df=4)
    }
    if(prior == "regularized_horseshoe_true_p0"){ # true number of relevant variables
      if(condition==1){p0 <- 3}
      if(condition==2){p0 <- 7} # 1 variable less than true because all variables relevant is not possible for computation 
      if(condition==3 | condition==4){p0 <- 15}
      if(condition==5 | condition==6){p0 <- 20}
      if(condition == 7 | condition==8){p0 <- 10}
      tau0 <- p0/((ncol(x$trainX) - p0) * sqrt(nrow(x$trainX))) # tau0 is based on a prior guess of the # of relevant predictors p0
      standat <- list(N_train=nrow(x$trainX), p=ncol(x$trainX), y_train=c(x$trainY), X_train=x$trainX,
                      N_test=nrow(x$testX), X_test=x$testX, scale_global= tau0, nu_global= 1, nu_local= 1, slab_scale=2, slab_df=4)
    }
    if(prior == "regularized_horseshoe_off_p0"){ # prior guess relevant variables = half # of variables
      if(condition==1){p0 <- 4}
      if(condition==2){p0 <- 4}
      if(condition==3){p0 <- 20} # true number = half, so p0 > half
      if(condition==4){p0 <- 10} # true number = half, so p0 < half
      if(condition==5 | condition==6){p0 <- 15}
      if(condition == 7 | condition==8){p0 <- 5}
      tau0 <- p0/((ncol(x$trainX) - p0) * sqrt(nrow(x$trainX))) # tau0 is based on a prior guess of the # of relevant predictors p0
      standat <- list(N_train=nrow(x$trainX), p=ncol(x$trainX), y_train=c(x$trainY), X_train=x$trainX,
                      N_test=nrow(x$testX), X_test=x$testX, scale_global= tau0, nu_global= 1, nu_local= 1, slab_scale=2, slab_df=4)
    }
    else({
      standat <- list(N_train=nrow(x$trainX), p=ncol(x$trainX), y_train=c(x$trainY), X_train=x$trainX,
                    N_test=nrow(x$testX), X_test=x$testX)
    })
    # run
    fit.mcmc <- sampling(modFB, data=standat, iter=4000, control=list(adapt_delta=0.999, stepsize=0.001, max_treedepth=20))
    # use different controls for efficiency from condition 4 on:
    # replications that did not converge with the adapted controls were rerun with the strict settings
    # the regularized horseshoe was run with default controls first & convergence checked immediately; rerun with strict controls if divergent transitions
    #fit.mcmc <- sampling(modFB, data=standat, iter=4000, control=list(adapt_delta=0.85, max_treedepth=15))
    
    ### Convergence ###
    # save traceplot 
    nm <- paste0(prior, condition, counter)
    if(grepl("regularized_horseshoe", prior) == TRUE){
      pars <- c("sigma2", "beta\\b", "tau")
    }
    else(pars <- c("sigma2", "beta\\b", "lambda"))
    png(file=paste0("trace_", nm, ".png"), width=700, height=700)
    print(mcmc_trace(as.matrix(fit.mcmc), regex_pars=pars)) # plot
    dev.off()
    # check convergence 
    out <- summary(fit.mcmc)$summary
    rhat <- out[which(out[, "Rhat"] > 1.1), "Rhat"] # PSR > 1.1
    sp <- get_sampler_params(fit.mcmc, inc_warmup=F)
    div <- sapply(sp, function(x) sum(x[, "divergent__"])) # divergent transitions
    
    ### Extract output ###
    pars <- fit.mcmc@model_pars
    ## posterior estimates regression coefficients and hyperparameters ##
    pars.sel <- pars[-grep("linpred", pars)] # remove linear predictor from output
    if(length(grep("beta_raw", pars.sel) > 0)){ # remove beta_raw (some models are reparametrized to include standard normal beta_raw parameter)
      pars.sel <- pars.sel[-grep("beta_raw", pars.sel)]
    }
    fit.summary <- summary(fit.mcmc, pars=pars.sel, probs=seq(0, 1, 0.05))$summary # extract summary
    post.mean <- fit.summary[-grep("y_test", rownames(fit.summary)), "mean"]
    post.median <- fit.summary[-grep("y_test", rownames(fit.summary)), "50%"]
    post.draws <- rstan::extract(fit.mcmc, pars=pars.sel[-grep("y_test", pars.sel)]) # extract posterior draws from the second half of each chain (excluding burn-in)
    # estimate posterior modes based on the posterior density
    estimate_mode <- function(draws){
      d <- density(draws)
      d$x[which.max(d$y)]
    }
    post.mode <- lapply(post.draws, function(x){
      if(length(dim(x)) == 1){estimate_mode(x)}
      else(apply(x, 2, estimate_mode))
    })
    
    ## credible intervals ##
    ci <- fit.summary[-grep("y_test", rownames(fit.summary)), grep("%", colnames(fit.summary))]
    
    ## posterior standard deviations ##
    post.sd <- fit.summary[-grep("y_test", rownames(fit.summary)), "sd"]
    
    ## variable selection based on scaled neighborhood criterion ##
    sd.inter <- cbind(-post.sd[grep("beta\\b", names(post.sd))], post.sd[grep("beta\\b", names(post.sd))])
    draws.beta <- post.draws[[grep("beta\\b", names(post.draws))]]
    post.prob <- rep(NA, nrow(sd.inter))
    for(i in 1:nrow(sd.inter)){ # compute the posterior probability in [-post.sd, post.sd]
      post.prob[i] <- sum(sd.inter[i, 1] <= draws.beta[,i] & sd.inter[i, 2] >= draws.beta[,i])/nrow(draws.beta)
    }
    excl.pred.snc <- matrix(NA, nrow=11, ncol=length(post.prob)) # matrix with TRUE if predictor is not zero and thus included
    colnames(excl.pred.snc) <- rownames(sd.inter)
    rownames(excl.pred.snc) <- c("prob0", "prob0.1", "prob0.2", "prob0.3", "prob0.4", "prob0.5", "prob0.6", "prob0.7", "prob0.8", "prob0.9", "prob1")
    for(i in 1:length(post.prob)){
      sq <- seq(0, 1, 0.1)
      excl.pred.snc[, i] <- sapply(sq, function(x) post.prob[i] <= x) 
    }
    
    ## generated y values test set ##
    ygen <- fit.summary[grep("y_test", rownames(fit.summary)), "mean"]
    
    ### Return output ###
    out <- list("Replication"=paste0("rep", counter), "Rhat > 1.1"=rhat, "Number of divergent transitions"=div, "Posterior means"=post.mean, 
                "Posterior medians"=post.median, "Posterior modes"=post.mode, "Credible intervals"=ci,"Posterior standard deviations"=post.sd, 
                "Excluded predictors based on scaled neighborhood criterion"=excl.pred.snc, "Generated y-values test data"=ygen)
    return(out)
  }) # end of function to run model and extract output for each rep
  
  ### Save output ###
  save(out, file=paste0("convergence_output_", prior, "_sim", condition, ".RData"))
} # end simulation function

##############################
####### RUN SIMULATION #######            
##############################

nworkers <- detectCores() # number of cores to use
cl <- makePSOCKcluster(nworkers) # create cluster
clusterCall(cl, function() library(rstan))
clusterCall(cl, function() library(bayesplot))
out <- clusterApplyLB(cl, 1:nrow(conditions), sim.fun, cond=conditions, reps=500) # run simulation
stopCluster(cl) # shut down the nodes
