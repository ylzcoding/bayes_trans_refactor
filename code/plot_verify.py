#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
# from tensorflow_probability.substrates import jax as tfp
# tfpd = tfp.distributions
from scipy.stats import invgauss
from scipy.linalg import block_diag
jax.config.update("jax_enable_x64", True)
import pickle
from pathlib import Path
from glob import glob
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import sys

if sys.stdin.isatty():
    assert len(sys.argv)==1
    seed = 52
    #K = 36
    #K = 32
    K = 3
else:
    assert len(sys.argv)==1+1
    K = int(sys.argv[2])
    print(f"running seed {seed} with K={K}")

exec(open("verify_settings.py").read())

files = glob(f"{sp}*")
nruns = len(files)

print(f"Found {nruns} files.")
num = [int(f.split('/')[-1].split('.pkl')[0]) for f in files]
print(f"missing: {np.setdiff1d(np.arange(1,np.max(num)+1),num)}")
print(f"max guy: {np.max(num)}")

nums = [int(x.split('/')[-1].split('.')[0]) for x in files]
bignum = np.setdiff1d(np.arange(max(nums))+1, nums)

eta_prob = pd.DataFrame(np.zeros([nruns,K-1]))
if not true_y:
    covs = {}
    eta_true = pd.DataFrame(np.zeros([nruns,K-1]))
    inc = 1
else:
    Ns = []
    inc = 3

for fi,file in enumerate(files):
    seed = file.split("/")[-1].split('.pkl')[0]
    with open(file, 'rb') as f:
        if true_y:
            errs_f, eta_pc_f, switch_rates_f, N_f, K_f = pickle.load(f)
        else:
            errs_f, act_covs_f, eta_true_f, eta_pc_f, switch_rates_f  = pickle.load(f)
    if fi==0:
        ncomp = len(errs_f)
        if not true_y:
            prms = list(act_covs_f.keys())
            for p in prms:
                covs[p] = pd.DataFrame(np.zeros([nruns,len(nom_covs)]))
            nprms = len(act_covs_f)
        comps = list(errs_f.keys())
        errs = pd.DataFrame(np.zeros([nruns,ncomp+inc]))
        cols = ['seed'] + comps
        if true_y:
            cols += ['N','targ']
        errs.columns = cols

    pack = [seed] + [errs_f[comp] for comp in comps]
    if true_y:
        pack += [N_f, K_f]
    errs.iloc[fi,:] = pack
    eta_prob.iloc[fi,:] = eta_pc_f
    if not true_y:
        for p in prms:
            covs[p].iloc[fi,:] = act_covs_f[p]
        eta_true.iloc[fi,:] = eta_true_f

errs.index = errs['seed']
errs = errs.drop('seed',axis=1)
eta_prob.index = errs.index

if true_y:
    ncols = 2
    nrows = 1
else:
    ncols = 2
    nfigs = nprms+3
    nrows = int(np.ceil(nfigs/ncols))

print(f"Percent with no eta movement: {np.mean(np.all(eta_prob==0, axis=1))}")

fig = plt.figure(figsize=[2.5*ncols,2.5*nrows])

## MSE
plt.subplot(nrows,ncols,1)
if true_y:
    errsnpc = errs.drop(['N','targ'],axis=1)
else:
    errsnpc = errs
plt.boxplot([row[~np.isnan(row)] for _,row in errsnpc.T.iterrows()], tick_labels = errsnpc.columns)
plt.yscale('log')
if K>1:
    baseline = errs['li2022'] if true_y else errs['naive']
else:
    baseline = errs['LCV'] 
plt.title(f"MSE Better percent: {np.round(np.mean(baseline>errs['pc']),2)}")
if 'null' in errs.columns:
    ll,ul = plt.gca().get_xlim()
    plt.hlines(np.median(errs['null']),ll,ul, linestyle='--')
    plt.hlines(np.median(errs['pc']),ll,ul, linestyle='--')

comp = 'li2022' if K>1 else 'LCV'
if comp in errs.columns:
    plt.subplot(nrows,ncols,2)
    #plt.boxplot(errs, tick_labels = errs.columns)
    diff = errs[comp] - errs['pc']
    plt.boxplot(diff[~np.isnan(diff)])
    #plt.boxplot([row[~np.isnan(row)] for _,row in errs.T.iterrows()], tick_labels = errs.columns)
    plt.title("MSE Reduction")

if not true_y:
    ## Eta prob
    plt.subplot(nrows,ncols,3)
    e = np.array(eta_true).flatten().astype(int)
    p = np.array(eta_prob).flatten()
    knn = KNeighborsRegressor(n_neighbors=int(np.sqrt(nruns)))
    knn.fit(p.reshape([-1,1]),e)
    pred = np.linspace(0,1,num=50)
    #smooth = knn.predict(p.reshape([-1,1]))
    smooth = knn.predict(pred.reshape([-1,1]))
    plt.scatter(pred,smooth)
    plt.xlabel("Predicted")
    plt.ylabel("Empirical")
    plt.title(r"$\eta$ estimation")
    plt.ylim(0,1)
    plt.plot([0,1],[0,1],color='gray',linestyle='--')

    ## Coverage
    for pi,p in enumerate(prms):
        plt.subplot(nrows,ncols,4+pi)
        mu = np.mean(covs[p], axis=0)
        sig = np.std(covs[p], axis=0)/np.sqrt(nruns)
        col = 'blue'
        plt.plot(nom_covs, mu, color = col)
        plt.plot(nom_covs, mu-2*sig, color = col, linestyle = ':')
        plt.plot(nom_covs, mu+2*sig, color = col, linestyle = ':')
        plt.plot([0,1],[0,1],color='gray',linestyle='--')
        plt.xlabel("Nominal")
        plt.ylabel("Actual")
        #plt.title(rf"$\{p}$-Coverage")
        plt.title(f"{p}-Coverage")

plt.tight_layout()
plt.savefig("verify.pdf")
plt.close()

iswin = errs['li2022'] > errs['pc']
wins = errs.index[np.where(errs['li2022'] > errs['pc'])]
print(wins)
print(np.mean(iswin))

if true_y:
    relative_mse = (errs['li2022'] - errs['pc']) / errs['li2022']
    #all_rmse = (errs['li2022'] - errs['all']) / errs['li2022']
    #naive_rmse = (errs['li2022'] - errs['naive']) / errs['li2022']
    rmse = pd.DataFrame([relative_mse, errs['li2022'], errs['pc'], eta_prob.sum(axis=1), errs['all'], errs['naive'], errs['N'],errs['targ']]).T
    rmse.columns = ['RMSE','limse','memse','eta_tot','all','naive','N','targ']
    print(rmse.sort_values('RMSE'))

if true_y:
    fig = plt.figure()
    plt.scatter(rmse['N'], rmse['RMSE'])
    #plt.scatter(errs['li2022'], errs['pc'], c = ['red' if errs['N'][i] > np.median(errs['N']) else 'green' for i in range(errs.shape[0])])
    #plt.xscale('log')
    #ll,ul = plt.gca().get_xlim()
    plt.hlines(0,np.min(errs['N']),np.max(errs['N']))
    plt.savefig("temp.pdf")
    plt.close()

    bw = errs.index[np.argmax(relative_mse)]

print(errs)

fig = plt.figure(figsize=[1.5*ncomp,1.5*ncomp])
for ci1, c1 in enumerate(comps):
    for ci2, c2 in enumerate(comps):
        plt.subplot(ncomp,ncomp,ci1*ncomp+ci2+1)
        plt.scatter(errs[c1], errs[c2])
        plt.title(f"{c1} vs {c2}")
        plt.xscale('log')
        plt.yscale('log')
plt.tight_layout()
plt.savefig("splom.pdf")
plt.close()

#print("bad MSE for our method:")
#bads = errs['pc']>100
#np.where(bads)
#print(errs.loc[bads,:])

#eta_prob.iloc[np.where(bads)[0][0],:]

triv = np.minimum(eta_prob,1-eta_prob)
meant = np.mean(triv, axis=1)
mint = np.min(triv, axis=1)

fig = plt.figure()
plt.subplot(1,2,1)
plt.scatter(meant, errs['pc'])
plt.yscale('log')
plt.subplot(1,2,2)
plt.scatter(mint, errs['pc'])
plt.savefig("triv.pdf")
plt.close()

if true_y:
    fig = plt.figure()
    plt.scatter(errs['N'], errs['li2022']/errs['pc'])
    plt.xlabel("N_t")
    plt.ylabel("mse")
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig("rel_vs_N.pdf")
    plt.close()

#np.where(np.all(eta_prob > 1-1e-2,axis=))
