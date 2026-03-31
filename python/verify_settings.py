#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#sp = "./sim_out/verify/" 
#sp = f"./sim_out/verify_{K}_{corr_var}/" 
sp = f"./sim_out/verify_{K}/" 

iters = 5000
#iters=2000
#iters = 20
for i in range(10):
    print("5k iters.")

#P = 20
##P = 5
#true_X = False
##true_X = True
##print("True X!")
#true_y = False
#do_R = False
#for i in range(10):
#    print("No R!")

#P = 399
#P = 10
#P = 50
#P = 25
P = 10

true_X = False
true_y = False

#corr_y = True
corr_y = False

#nsmol = 5
#nbig = 10
nsmol = 25
#nbig = 200
nbig = 25
Ns = [nsmol] + [nbig for _ in range(K-1)]

nom_covs = np.linspace(0,1)

phold = 0.2

prefix = 'true_y' if true_y else 'sim_y'

learn = {}
learn['beta'] = True
learn['sigma2'] = True
learn['omega'] = True
learn['lambda_p'] = True
learn['tau2'] = True
learn['rho'] = True
learn['eta'] = True
learn['lambda_t'] = True
#learn['sigma2'] = False
#learn['omega'] = False
#learn['lambda_p'] = False
#learn['tau2'] = False
#learn['rho'] = False
#learn['eta'] = False
#learn['lambda_t'] = False

#print("Some not learning!")

eta_zero = False
#eta_zero = True
#clamp_nonzero_eta = True
clamp_nonzero_eta = False

PROJ_B = True

#omega_prior = 'ig'
omega_prior = 'tpb'
for i in range(10):
    print("tpb-hs!")

#rb_omega = True
##rb_omega = False
##rb_betasig = True
#rb_betasig = False
#rb = 'none'
#rb = 'omega'
#rb = 'betasig'
#rb = 'alt'
rb = 'none'

#L = 1
#for i in range(10):
#    print("L=1")
L = 5
#L = 10
#L = 4
#L = 50

#do_R = False
do_R = False
