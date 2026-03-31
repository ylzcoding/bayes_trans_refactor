
fixed = {}
true_vals = {}

## Sample params
if learn['lambda_t']:
    true_vals['lambda_t'] = np.abs(np.random.standard_cauchy())
    #print("Fixed lam_p!")
    #true_vals['lambda_t'] = 100000.
else:
    true_vals['lambda_t'] = 0.1
    fixed['lambda_t'] = true_vals['lambda_t'] 

if learn['lambda_p']:
    true_vals['lambda_p'] = np.square(np.random.standard_cauchy())
    #print("Fixed lam_p!")
    #true_vals['lambda_p'] = 100000.
else:
    true_vals['lambda_p'] = 1e-2
    fixed['lambda_p'] = true_vals['lambda_p'] 

# TODO: Technically not the true sampling dist.
if learn['sigma2']:
    true_vals['sigma2'] = np.random.gamma(shape=1.,scale=1.)
else:
    true_vals['sigma2'] = 1.
    fixed['sigma2'] = true_vals['sigma2']

true_vals['rho'] = np.random.beta(1,1) 
if not learn['rho']:
    fixed['rho'] = true_vals['rho']
print(f"true_vals['rho']: {true_vals['rho']}")
#true_vals['eta'] = jnp.zeros(K-1)
true_vals['eta'] = np.array([np.random.choice([False,True],1,p=[1-true_vals['rho'],true_vals['rho']])[0] for _ in range(K-1)])
while jnp.sum(true_vals['eta'])==0 and clamp_nonzero_eta:
    true_vals['eta'] = np.array([np.random.choice([False,True],1,p=[1-true_vals['rho'],true_vals['rho']])[0] for _ in range(K-1)])
if eta_zero:
    true_vals['eta'] = np.zeros(K-1)
    print("Warning: eta is always zero!")
if not learn['eta']:
    fixed['eta'] = true_vals['eta']
print(f"true_vals['eta']: {true_vals['eta']}")

true_vals['tau2'] = np.abs(np.random.standard_cauchy())
if not learn['tau2']:
    fixed['tau2'] = true_vals['tau2']
print(f"true_vals['tau2']: {true_vals['tau2']}")

###
## Auxilliary params
if omega_prior=='exp':
    true_vals['omega'] = np.random.exponential(2/np.square(true_vals['lambda_t']), size = P)
elif omega_prior=='ig':
    #true_vals['omega'] = 1/np.random.gamma(0.5, 1/(2/np.square(true_vals['lambda_t'])), size = P)
    g_rate = 1/(2*np.square(true_vals['lambda_t']))
    true_vals['omega'] = 1/np.random.gamma(0.5, 1/g_rate, size = P)
elif omega_prior=='dl':
    a_dl = 1.0 / P
    # T_j ~ Gamma(a, scale=2.0)
    true_vals['T'] = np.random.gamma(a_dl, 2.0, size=P)
    if not learn['lambda_t']:
        true_vals['T'] = true_vals['T'] * (true_vals['lambda_t'] / np.sum(true_vals['T']))
    else:
        true_vals['lambda_t'] = np.sum(true_vals['T'])
    true_vals['phi'] = true_vals['T'] / true_vals['lambda_t']
    true_vals['omega_dl'] = np.random.exponential(2.0, size=P)
    true_vals['omega'] = true_vals['omega_dl'] * np.square(true_vals['T'])
elif omega_prior=='tpb':
    a_tpb = 0.5 # a_tpb = 0.5 for horseshoe prior
    b_tpb = 0.5
    # v_j ~ Gamma(a, 1), u_j ~ Gamma(b, 1)
    true_vals['v'] = np.random.gamma(a_tpb, 1.0, size=P)
    true_vals['u'] = np.random.gamma(b_tpb, 1.0, size=P)
    # omega_j = lambda_t^2 * (v_j / u_j)
    true_vals['omega'] = np.square(true_vals['lambda_t']) * (true_vals['v'] / true_vals['u'])
elif omega_prior=='r2d2':
    b = 0.5
    # target size Ns[0] -> a_pi
    a_pi = 1.0 / ( (P**(b/2.0)) * (Ns[0]**(b/2.0)) * np.log(np.maximum(Ns[0], 2)) )
    a = a_pi * P
    if not learn['lambda_t']:
        true_vals['w'] = np.square(true_vals['lambda_t'])
    else:
        X_bp = np.random.gamma(a, 1.0)
        Y_bp = np.random.gamma(b, 1.0)
        true_vals['w'] = X_bp / Y_bp
        true_vals['lambda_t'] = np.sqrt(true_vals['w'])
        
    true_vals['T'] = np.random.gamma(a_pi, 1.0, size=P)
    true_vals['phi'] = true_vals['T'] / np.sum(true_vals['T'])
    true_vals['psi_r2d2'] = np.random.exponential(scale=2.0, size=P)
    true_vals['omega'] = true_vals['psi_r2d2'] * true_vals['phi'] * true_vals['w']

if not learn['omega']:
    fixed['omega'] = true_vals['omega']
    if omega_prior=='dl':
        fixed['T'] = true_vals['T']
        fixed['omega_dl'] = true_vals['omega_dl']
        fixed['phi'] = true_vals['phi']
    elif omega_prior == 'tpb':
        fixed['v'] = true_vals['v']
        fixed['u'] = true_vals['u']
    elif omega_prior == 'r2d2':
        fixed['T'] = true_vals['T']
        fixed['phi'] = true_vals['phi']
        fixed['psi_r2d2'] = true_vals['psi_r2d2']
        fixed['w'] = true_vals['w']

xi_1 = np.random.normal(size=P)
xi_2 = np.random.normal(size=K*P)

## Form matrices
grams = jnp.stack([Xs[k].T @ Xs[k] for k in range(K)])
B_true = get_B(grams, true_vals['eta'][None,:], jnp.array(Ns), true_vals['tau2'])[0,:,:]

### Pull the trigger, efficiently
grams_aug_true = jnp.stack([true_vals['lambda_p']*jnp.eye(P) for k in range(K)])
grams_aug_chols = jnp.linalg.cholesky(grams_aug_true)
grams_aug_invchols = jnp.linalg.inv(grams_aug_chols)
grams_aug_inv = jnp.linalg.inv(grams_aug_true)
prior_mean = jnp.zeros(K*P)
#true_vals['beta'] = get_beta_samp(grams_aug_chols, grams_aug_invchols, B_true, true_vals['omega'], xi_1, xi_2, prior_mean, true_vals['sigma2'])
true_vals['beta'] = get_beta_samp(grams_aug_invchols, B_true, true_vals['omega'], true_vals['lambda_p'], xi_1, xi_2, prior_mean, true_vals['sigma2'])
betas_true = [true_vals['beta'][k*P:(k+1)*P] for k in range(K)]
if not learn['beta']:
    fixed['beta'] = true_vals['beta'].reshape([K,P])

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
