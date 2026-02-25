import numpy as np
import jax.numpy as jnp
from linear_algebra import get_B, get_beta_samp

## Sample params

def generate_true_params(K, P, Ns, Xs, learn:dict, omega_prior:str, clamp_nonzero_eta:bool, eta_zero:bool):
    fixed = {}
    true_vals = {}

    ## Sample params
    if learn['lambda_t']:
        true_vals['lambda_t'] = np.abs(np.random.standard_cauchy())
    else:
        true_vals['lambda_t'] = 0.1
        fixed['lambda_t'] = true_vals['lambda_t'] 

    if learn['lambda_p']:
        true_vals['lambda_p'] = np.square(np.random.standard_cauchy())
    else:
        true_vals['lambda_p'] = 1e-2
        fixed['lambda_p'] = true_vals['lambda_p'] 

    if learn['sigma2']:
        true_vals['sigma2'] = np.random.gamma(shape=1.,scale=1.)
    else:
        true_vals['sigma2'] = 1.
        fixed['sigma2'] = true_vals['sigma2']

    true_vals['rho'] = np.random.beta(1,1) 
    if not learn['rho']:
        fixed['rho'] = true_vals['rho']
    
    true_vals['eta'] = np.array([np.random.choice([False,True],1,p=[1-true_vals['rho'],true_vals['rho']])[0] for _ in range(K-1)])
    while jnp.sum(true_vals['eta'])==0 and clamp_nonzero_eta:
        true_vals['eta'] = np.array([np.random.choice([False,True],1,p=[1-true_vals['rho'],true_vals['rho']])[0] for _ in range(K-1)])
    if eta_zero:
        true_vals['eta'] = np.zeros(K-1)
    if not learn['eta']:
        fixed['eta'] = true_vals['eta']

    true_vals['tau2'] = np.abs(np.random.standard_cauchy())
    if not learn['tau2']:
        fixed['tau2'] = true_vals['tau2']

    ### Auxilliary params
    if omega_prior=='exp':
        true_vals['omega'] = np.random.exponential(2/np.square(true_vals['lambda_t']), size = P)
    elif omega_prior=='ig':
        g_rate = 1/(2*np.square(true_vals['lambda_t']))
        true_vals['omega'] = 1/np.random.gamma(0.5, 1/g_rate, size = P)
    if not learn['omega']:
        fixed['omega'] = true_vals['omega']

    xi_1 = np.random.normal(size=P)
    xi_2 = np.random.normal(size=K*P)

    ## Form matrices
    grams = jnp.stack([Xs[k].T @ Xs[k] for k in range(K)])
    B_true = get_B(grams, true_vals['eta'][None,:], jnp.array(Ns), true_vals['tau2'])[0,:,:]

    ### Pull the trigger, efficiently
    grams_aug_true = jnp.stack([true_vals['lambda_p']*jnp.eye(P) for k in range(K)])
    grams_aug_chols = jnp.linalg.cholesky(grams_aug_true)
    grams_aug_invchols = jnp.linalg.inv(grams_aug_chols)
    prior_mean = jnp.zeros(K*P)
    
    true_vals['beta'] = get_beta_samp(grams_aug_invchols, B_true, true_vals['omega'], true_vals['lambda_p'], xi_1, xi_2, prior_mean, true_vals['sigma2'])
    
    betas_true = [true_vals['beta'][k*P:(k+1)*P] for k in range(K)]
    if not learn['beta']:
        fixed['beta'] = true_vals['beta'].reshape([K,P])

    return betas_true, true_vals, fixed


### Pull the trigger, naively
#prior_mean = jnp.zeros(K*P)
#if PROJ_B:
#    svB = np.linalg.svd(B_true, full_matrices = False)
#    UB = svB[2].T
#    ImPB = np.eye(K*P) - UB @ UB.T
#    if np.sum(true_vals['eta'])==0:
#        ImPB = np.eye(K*P)
#    else:
#        ImPB = np.eye(K*P) - UB @ UB.T
#    prior_var = true_vals['sigma2'] * np.linalg.inv(true_vals['lambda_p']*ImPB + B_true.T @ np.diag(1/true_vals['omega']) @ B_true)
#else:
#    assert False
#L = np.linalg.cholesky(prior_var)
#true_vals['beta'] = L @ np.random.normal(size=K*P)
#betas_true = [true_vals['beta'][k*P:(k+1)*P] for k in range(K)]
#if not learn_beta:
#    fixed['beta'] = true_vals['beta'].reshape([K,P])

