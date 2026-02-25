#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from main import bayes_trans
from verify_lib import generate_true_params
jax.config.update("jax_enable_x64", True)
import pickle
from pathlib import Path
import pandas as pd
import sys
import os

if sys.stdin.isatty():
    print("Running interactive mode.")
    assert len(sys.argv)==1
    #seed = 52
    #K = 32
    seed = 1
    #K = 4
    K = 3
else:
    print("Running script mode.")
    print(sys.argv)
    seed = int(sys.argv[1])
    K = int(sys.argv[2])

print(f"running seed {seed} with K={K}")

#from python.bayes_lib import bayes_transfer
exec(open("verify_settings.py").read())
#exec(open("python/map_lib.py").read())

if sys.stdin.isatty():
    if corr_y:
        #corr_var = 5.
        corr_var = 0.1
else:
    if corr_y:
        assert len(sys.argv)==1+3
        corr_var = float(sys.argv[3])
    else:
        assert len(sys.argv)==1+2

outpath = f"{sp}{seed}.pkl"

#if os.path.exists(outpath):
#    print(f"We already have this one. Exiting.")
#    exit()
#else:
#    print(f"Don't see an output file for this config. Proceeding.")


seedoffset = 0
#seedoffset = 2000
np.random.seed(seed+seedoffset)

if true_X:
    from glob import glob
    files = glob('proc_data/*.csv')
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f, index_col=0))
    #dfs = dfs[:5]
    K_max = len(dfs)
    Ns = [dfs[k].shape[0] for k in range(len(dfs))]
    P_max = dfs[0].shape[1] - 1 # Because first col is response.
    print(f"P_max: {P_max}")
    #assert np.all([dfs[k].shape[1]==P+1 for k in range(len(dfs))])
    psel = np.sort(np.random.choice(P_max,P,replace=False))
    ksel = np.random.choice(K_max,K,replace=False)
    Xs = [np.array(dfs[k].iloc[:,1:]) for k in range(len(dfs))]
    # Restrict to K,P as requested.
    #Xs = [Xs[k][:,psel] for k in range(K_max) if k in ksel]
    Xs = [Xs[k][:,psel] for k in ksel]
    if true_y:
        ys = [np.array(dfs[k].iloc[:,0]) for k in range(len(dfs))]
        ys = [ys[k] for k in ksel]
        N0 = ys[0].shape[0]
        Ntest = int(N0*phold)
        Ntrain = N0 - Ntest
        testinds = np.sort(np.random.choice(N0,Ntest,replace=False))
        traininds = np.setdiff1d(np.arange(N0),testinds)
        Xtest = Xs[0][testinds,:]
        ytest = ys[0][testinds]
        Xs[0] = Xs[0][traininds,:]
        ys[0] = ys[0][traininds]
else:
    Xs = [np.random.normal(size=[N,P]) for N in Ns]
Ns = [X.shape[0] for X in Xs]
E0 = np.concatenate([np.eye(P)]+[np.zeros([P,P]) for _ in range(K-1)],axis=1)

if true_y:
    assert true_X
else:
    # exec(open("verify_lib.py").read())
    betas_true, true_vals, fixed = generate_true_params(K, P, Ns, Xs, learn, omega_prior, clamp_nonzero_eta, eta_zero)
    ys = [Xs[k] @ betas_true[k] + np.random.normal(scale=np.sqrt(true_vals['sigma2']), size=Ns[k])  for k in range(K)]
    ys = [np.array(y) for y in ys]
    if corr_y:
        true_vals['psi'] = corr_var
        err_Cs = []
        for k in range(K):
            Ak = np.random.normal(size=[Ns[k],Ns[k]])
            Ck = Ak.T @ Ak
            Ck = Ck / np.linalg.norm(Ck, ord=2)
            err_Cs.append(Ck)
            Lk = np.linalg.cholesky(Ck)
            err_corr =np.sqrt(corr_var)*np.sqrt(true_vals['sigma2'])*Lk@np.random.normal(size=Ns[k])
            ys[k] += err_corr
    else:
        err_Cs = None

    print(f"true lambda_p: {true_vals['lambda_p']}")

X = np.concatenate(Xs, axis = 0)
y = np.concatenate(ys)

X0 = Xs[0]
y0 = ys[0]

## Other estimators 
errs = {}
print("Fitting Naive")
beta_naive = np.linalg.lstsq(X0,y0)[0]
print("Fitting All")
beta_all = np.linalg.lstsq(X,y)[0]

if do_R:
    print("Fitting glmnet")
    exec(open("python/li2022_wrapper.py").read())

    #glmnet = importr('glmnet')
    #def get_glmnet_coefs(X,y):
    #    res = glmnet.cv_glmnet(np.array(X), np.array(y).reshape([-1,1]))
    #    r['as.matrix'](r['coef'](res))[1:].flatten()
    #beta_glmnet = get_glmnet_coefs(X0,y0)
    #beta_glmnet = r.get_glmnet_coefs(np.array(X0),np.array(y0).reshape([-1,1])).flatten()
    #with localconverter(default_converter + numpy2ri.converter):
    #    r_X = ro.conversion.py2rpy(X0)
    #    r_y = ro.conversion.py2rpy(y0.reshape([-1,1]))
    #beta_glmnet = r.get_glmnet_coefs(r_X,r_y).flatten()
    beta_glmnet = r.get_glmnet_coefs(X0,y0.reshape([-1,1])).flatten()[1:]
    #lcv = LassoCV()
    #lcv = lcv.fit(X0,y0)
else:
    beta_glmnet = np.repeat(np.nan,P)

if true_y:
    errs['naive'] = np.mean(np.square(Xtest @ beta_naive - ytest))
    errs['all'] = np.mean(np.square(Xtest @ beta_all - ytest))
    errs['LCV'] = np.mean(np.square(Xtest @ beta_glmnet - ytest))
    errs['null'] = np.mean(np.square(np.mean(y0) - ytest))
else:
    if np.any(true_vals['eta']!=0):
        Xo = np.concatenate([Xs[k+1] for k in range(K-1) if true_vals['eta'][k]],axis=0)
        yo = np.concatenate([ys[k+1] for k in range(K-1) if true_vals['eta'][k]],axis=0)
        beta_oracle = np.linalg.lstsq(Xo,yo)[0]
    else:
        beta_oracle = np.repeat(np.nan, P)
    errs['naive'] = np.mean(np.square(beta_naive - betas_true[0]))
    errs['all'] = np.mean(np.square(beta_all - betas_true[0]))
    errs['oracle'] = np.mean(np.square(beta_oracle - betas_true[0]))
    errs['LCV'] = np.mean(np.square(beta_glmnet - betas_true[0]))
    errs['null'] = np.mean(np.square(betas_true[0]))

if do_R and K>1:
    print("Fitting Trans Lasso using Li's code")
    try:
        beta_li = r.do_trans_lasso(X, y, Ns)
        if true_y:
            errs['li2022'] = np.mean(np.square(Xtest@beta_li - ytest))
        else:
            errs['li2022'] = np.mean(np.square(beta_li - betas_true[0]))
    except Exception:
        errs['li2022'] = np.nan
    print("Fitting Trans Lasso using glmtrans")
    try:
        beta_tian = r.do_glmtrans(X, y, Ns)
        if true_y:
            errs['tian2023'] = np.mean(np.square(Xtest@beta_tian - ytest))
        else:
            errs['tian2023'] = np.mean(np.square(beta_tian - betas_true[0]))
    except Exception:
        errs['tian2023'] = np.nan
else:
    errs['li2022'] = np.nan
    errs['tian2023'] = np.nan

#print(beta_sampler)
if true_y:
    # TODO: bettter; we do this in another file if not y_true
    for v in learn:
        assert learn[v]
    fixed = {}

##print("Fixing!")
#fixed = {}
#fixed['lambda_t'] = 1e2
##fixed['lambda_p'] = 1e2
#fixed['tau2'] = 1e4
#samps, switch_rates = bayes_transfer(Xs, ys, iters = iters, fixed = fixed, debug = False, plotprefix = f"debug/{seed}", omega_prior = omega_prior, rb = rb, L = L)
#beta_samp = samps['beta']
#eta_samp_good = samps['eta']
#beta_pc_good = np.mean(beta_samp,axis=0)
#errs['pc_good'] = np.mean(np.square(Xtest @ beta_pc_good[0,:] - ytest))

if true_y:
    true_vals = None

print("Fitting Bayes")
fixed = {}
os.makedirs("debug", exist_ok=True)
#fixed['eta'] = np.zeros(K-1)
#fixed['lambda_t'] = 1.
#fixed['lambda_p'] = 100.
#fixed['tau'] = 100.
#for i in range(100):
#    print("fixed!")
#_, samps, switch_rates = bayes_trans(Xs, ys, iters = iters, fixed = fixed, debug = False, plotprefix = f"debug/{seed}", omega_prior = omega_prior, rb = rb, L = L, true_vals = true_vals)
_, samps, switch_rates = bayes_trans(Xs, ys, iters = iters, fixed = fixed, debug_tau2 = False, plotprefix = f"debug/{seed}", omega_prior = omega_prior, rb = rb, L = L, true_vals = true_vals, err_Cs = err_Cs)
beta_samp = samps['beta']
eta_samp = samps['eta']
#beta_pc = np.mean(beta_samp[:,0,:],axis=0)
beta_pc = np.mean(beta_samp,axis=0)
if true_y:
    errs['pc'] = np.mean(np.square(Xtest @ beta_pc[0,:] - ytest))
else:
    errs['pc'] = np.mean(np.square(beta_pc[0,:] - betas_true[0]))
#errs['pc_last'] = np.mean(np.square(Xtest @ beta_samp[-1,0,:] - ytest))

## MAP
print("Not fitting MAP!")
#print("Fitting MAP")
#beta_map = map_trans(Xs, ys, eta = np.round(np.mean(samps['eta'],axis=0)), lam_t = np.mean(samps['lambda_t']), lam_p = np.mean(samps['lambda_p']), tau2 = np.mean(samps['tau2']))
#if true_y:
#    errs['beta_map'] = np.mean(np.square(Xtest @ beta_map[0,:] - ytest))
#else:
#    errs['beta_map'] = np.mean(np.square(beta_map[0,:] - betas_true[0]))

#print("Fitting hybrid")
### A hybrid method
#if K > 1 and do_R:
#    nlcv = np.minimum(eta_samp.shape[0],100)
#    einds = np.random.choice(eta_samp.shape[0],nlcv,replace=False)
#    preds_transfer = np.zeros([nlcv,Ntest]) if true_y else np.zeros([nlcv,P])
#    for l in range(nlcv):
#        eta_cv = eta_samp[einds[l],:]
#        etaa_cv = np.concatenate([[1],eta_cv])
#        #lasso_big = LassoCV()
#
#        # Get overall estimate
#        Xbig = np.concatenate([Xs[k] for k in range(K) if etaa_cv[k]==1])
#        ybig = np.concatenate([ys[k] for k in range(K) if etaa_cv[k]==1])
#        #lasso_big_coefs = get_glmnet_coefs(Xbig, ybig)
#        lasso_big_coefs = r.get_glmnet_coefs(Xbig,ybig.reshape([-1,1])).flatten()[1:]
#
#        # Get transfer estimate.
#        #lasso_smol = LassoCV()
#        y0tilde = y0 - X0 @ lasso_big_coefs
#        #lasso_smol = lasso_smol.fit(X0, y0tilde)
#        lasso_smol_coefs = get_glmnet_coefs(X0, y0tilde)
#        lasso_smol_coefs = r.get_glmnet_coefs(X0,y0tilde.reshape([-1,1])).flatten()[1:]
#
#        # Get combined predictions.
#        combined_coefs = lasso_big_coefs + lasso_smol_coefs
#        if true_y:
#            #preds_transfer[l,:] = lasso_big.predict(Xtest) + lasso_smol.predict(Xtest)
#            preds_transfer[l,:] = Xtest @ combined_coefs
#        else:
#            preds_transfer[l,:] = combined_coefs  
#    trans_est = np.mean(preds_transfer, axis = 0)
#
#    if true_y:
#        errs['hyb'] = np.mean(np.square(trans_est - ytest))
#    else:
#        errs['hyb'] = np.mean(np.square(trans_est - betas_true[0]))
##errs['hyb'] = np.nan

## Get empirical CI coverage rates.
eta_pc = np.mean(samps['eta'],axis=0)

if true_y:
    results = [errs, eta_pc, switch_rates, Ns[0], ksel[0]]
else:
    cov_vars = [x for x in samps if not x in fixed and x!='eta']

    act_covs = {}
    for p in samps.keys():
        if p!='eta':
            act_covs[p] = np.nan*np.zeros_like(nom_covs)

    for i,c in enumerate(nom_covs):
        alpha = (1-c)/2
        for p in samps.keys():
            if p =='eta':
                continue
            elif p in true_vals:
                tv = true_vals[p]
            else:
                raise Exception("Unknown parameter to cover.")

            if p=='beta':
                tv = tv.reshape([K,P])

            lq = np.quantile(samps[p], alpha, axis = 0)
            uq = np.quantile(samps[p], 1-alpha, axis = 0)
            cov = np.logical_and(lq < tv, uq > tv)
            act_covs[p][i] = np.mean(cov)
    print(act_covs)
    print("true eta:")
    print(true_vals['eta'])

    results = [errs, act_covs, true_vals['eta'], eta_pc, switch_rates]

Path(sp).mkdir(parents=True, exist_ok=True)
with open(outpath, 'wb') as f:
    pickle.dump(results, f)

print("est eta:")
print(jnp.mean(samps['eta'],axis=0))

################################################################################
################################################################################
################################################################################
## Compare regression coefs!

print(errs)

fig = plt.figure()
plt.hist(np.mean(samps['beta'], axis = 0)[0,:])
plt.savefig('beta_hist.pdf')
plt.close()

fig = plt.figure()
plt.hist(Xs[0] @ np.mean(samps['beta'], axis = 0)[0,:])
plt.savefig('insamp_hist.pdf')
plt.close()

if true_X:
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(Xs[0] @ np.mean(samps['beta'], axis = 0)[0,:], ys[0])
    plt.title("Predicted vs Actual (IS)")
    plt.xlabel("Pred")
    plt.ylabel("Obs")
    plt.subplot(1,2,2)
    preds = Xtest @ np.mean(samps['beta'], axis = 0)[0,:]
    plt.scatter(preds, ytest)
    plt.title("Predicted vs Actual (OoS)")
    plt.xlabel("Pred")
    plt.ylabel("Obs")
    plt.savefig('pred_hist.pdf')
    plt.close()

    last_preds = Xtest @ samps['beta'][-1,0,:]

