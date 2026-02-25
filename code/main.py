import numpy as np
import jax.numpy as jnp
import jax
from tqdm import tqdm
from functools import partial


from preprocessing import prepare_data
from linear_algebra import get_B, get_beta_samp, woodbury_mean
from eta_nld import _eta_nld
import mcmc_steps as steps
import visualization as viz

def d2t(deltas):
    L = len(deltas)+1
    itemps = np.ones(L)
    for l in range(L-1):
        itemps[l+1] = itemps[l] * np.exp(-np.exp(deltas[l]))
    temps = 1/itemps
    return(temps)


def bayes_trans(Xs, ys, iters=10000, fixed={}, plotprefix=None, 
                desired_ar=0.574, desired_sr=0.2, gamma_exp=1., 
                verbose=True, debug_tau2 = False, omega_prior='exp', rb='none', L=5, true_vals=None, err_Cs = None):
    

    # =========================================================================
    # PHASE 1: PRECOMPUTATION & INITIALIZATION
    # =========================================================================

    D = prepare_data(Xs, ys)
    P, K, Ns = D['P'], D['K'], D['Ns']
    Xa, ya, Na = D['Xa'], D['ya'], D['Na']
    grams_all = D['grams_all']
    grams_chols = D['grams_chols']
    grams_invchols = D['grams_invchols']
    grams_inv_all = D['grams_inv_all']
    Xty = D['Xty']

    docorr = err_Cs is not None

    burnin = np.minimum(5000, iters // 5)  
    assert rb in ['none','omega','betasig','alt']

    # 2. initialization (vw)

    # working variables
    vw = {
        'beta': np.random.normal(size=[K,P]),
        'eta': jnp.zeros([K-1]),
        'sigma2': jnp.array(1.),
        'lambda_t': jnp.array(1.),
        'lambda_p': jnp.array(1.),
        'rho': jnp.array(0.5),
        'omega': jnp.ones(([P])),
        'tau2': jnp.array(1.)
    }
    if docorr:
        vw['psi'] = jnp.array(1.)

    # 1.2 Parallel Tempering (PT) status initialization
    etas = jnp.tile(vw['eta'][None,:], [L,1])     # L etas
    deltas = np.zeros(L-1)                       
    delta_tracking = np.zeros([iters, L-1])
    who = np.zeros(iters)                         
    didswitch = np.zeros(iters)                   
    etas_accepts = np.zeros([iters, L])          

    tau2_prop_scale = 1.                        
    tau2_accepts = 0.
    tau2_rejects = 0.

    if docorr: #TODO: unified scale initializtion?
        psi_prop_scale = 1.
        psi_accepts = 0.
        psi_rejects = 0.

    tracking = {} # initialize tracking dict for all variables
    for v in vw:
        tracking[v] = np.nan * np.zeros([iters] + list(vw[v].shape))
        tracking[v][0] = fixed[v] if v in fixed else vw[v]

    for v in fixed:
        vw[v] = jnp.array(tracking[v][0])

    B = get_B(grams_all, vw['eta'][None,:], Ns, vw['tau2'])[0,:,:] 
    z_samp = B @ vw['beta'].flatten()

    eta_nld_omega = jax.jit(lambda etas, grams_all, Xa, ya, vw, Ns, tau2: _eta_nld(etas, grams_all, Xa, ya, vw, Ns, tau2, rb='omega', omega_prior=omega_prior, P=P, K=K))
    eta_nld_betasig = jax.jit(lambda etas, grams_all, Xa, ya, vw, Ns, tau2: _eta_nld(etas, grams_all, Xa, ya, vw, Ns, tau2, rb='betasig', omega_prior=omega_prior, P=P, K=K))
    eta_nld_none = jax.jit(lambda etas, grams_all, Xa, ya, vw, Ns, tau2: _eta_nld(etas, grams_all, Xa, ya, vw, Ns, tau2, rb='none', omega_prior=omega_prior, P=P, K=K))

    # =========================================================================
    # THE MCMC MAIN LOOP 
    # =========================================================================

    for i in tqdm(range(1, iters), leave=False, disable=not verbose):
        
        # Rao-Blackwellization
        if rb=='betasig' or (rb=='alt' and i % 3 == 0):
            eta_nld_curr = eta_nld_betasig
        elif rb=='omega' or (rb=='alt' and i % 3 == 1):
            eta_nld_curr = eta_nld_omega
        elif rb=='none' or (rb=='alt' and i % 3 == 2):
            eta_nld_curr = eta_nld_none

        if docorr:
            from scipy.linalg import block_diag
            LXs = [Xs[k] for k in range(K)]
            Lys = [ys[k] for k in range(K)]
            for k in range(K):
                err_mat = np.linalg.inv(vw['psi'] * err_Cs[k] + np.eye(Ns[k]))
                Lk = np.linalg.cholesky(err_mat)
                LXs[k] = Lk.T @ Xs[k]
                Lys[k] = Lk.T @ ys[k]

            grams_all = jnp.stack([LXs[k].T @ LXs[k] for k in range(K)])
            Xty = jnp.stack([LXs[k].T @ Lys[k] for k in range(K)])[:,:,np.newaxis]
            Xa = block_diag(*LXs)
            ya = np.concatenate(Lys)
        
        grams_aug_all = jnp.stack([grams_all[k,:,:] + vw['lambda_p']*jnp.eye(P) for k in range(K)])
        grams_aug_chols = jnp.linalg.cholesky(grams_aug_all)
        grams_aug_invchols = jnp.linalg.inv(grams_aug_chols)
        grams_aug_inv_all = jnp.linalg.inv(grams_aug_all)
        bha_aug = grams_aug_inv_all @ Xty

        # Gibbs / MH

        # Update Omega
        if 'omega' not in fixed:
            vw['omega'] = steps.sample_omega(vw, z_samp, omega_prior)

        # Update beta
        if 'beta' not in fixed:
            beta_mean = woodbury_mean(B, vw['lambda_p'], vw['omega'], grams_aug_inv_all, bha_aug)
            beta_mean = beta_mean.flatten()
            xi_1 = np.random.normal(size=P)
            xi_2 = np.random.normal(size=K*P)
            vw['beta'] = get_beta_samp(grams_aug_invchols, B, vw['omega'], vw['lambda_p'], xi_1, xi_2, beta_mean, vw['sigma2'])
            z_samp = B @ vw['beta']
            vw['beta'] = vw['beta'].reshape([K,P])
            
            if np.any(~np.isfinite(vw['beta'])): raise Exception("nan beta")

        # Update eta
        if (K > 1) and 'eta' not in fixed:
            
            etas_prop = jnp.copy(etas)
            inds_flip = np.random.choice(K-1, L, replace=True) 
            for l in range(L):
                etas_prop = etas_prop.at[l, inds_flip[l]].set(1 - etas_prop[l, inds_flip[l]]) #

            etas_send = np.zeros([2*L, K-1])
            etas_send[0::2, :] = etas
            etas_send[1::2, :] = etas_prop

            nlp, resids = eta_nld_curr(etas_send, grams_all, Xa, ya, vw, Ns, vw['tau2'])
            nlp = nlp.reshape([L, 2]) 
            temps = d2t(deltas)
            nlpt = nlp / temps[:, None]
            lalpha = -nlpt[:, 1] + nlpt[:, 0] 
            lu = np.log(np.random.uniform(size=L))
            samps = (lalpha > lu).astype(int)
            
            etas_accepts[i, :] = samps
            samp = samps[0]
            
            nlpa = np.array([nlp[l, samps[l]] for l in range(L)])
            etas = np.array([etas_send.reshape([L, 2, K-1])[l, samps[l], :] for l in range(L)])

            if L > 1:
                ll = np.random.choice(L-1, 1)[0] 
                who[i] = ll

                lpnew = -nlpa[ll+1]/temps[ll] - nlpa[ll]/temps[ll+1]
                lpold = -nlpa[ll]/temps[ll]   - nlpa[ll+1]/temps[ll+1]
                lbeta = lpnew - lpold
                
                lu = np.log(np.random.uniform())
                isswitch = lbeta > lu
                didswitch[i] = isswitch
                
                if isswitch:
                    temp = np.copy(etas[ll, :])
                    etas[ll, :] = etas[ll+1, :]
                    etas[ll+1, :] = temp
                ## NOTE: Adaptation.
                ## Adjust proposal var.
                #print(f"Old delta: {}")
                beta_acc = np.minimum(np.exp(lbeta), 1.)
                gamma_it = np.power(1. / (i+1), gamma_exp)
                deltas[ll] = deltas[ll] + gamma_it * (beta_acc - desired_sr)
                delta_tracking[i, :] = deltas 

            vw['eta'] = etas[0, :]
            B = get_B(grams_all, vw['eta'][None,:], Ns, vw['tau2'])[0,:,:]  # TODO: is recalculating here faster than just returning earlier?

        else:
            _, resids = eta_nld_curr(etas=vw['eta'][None,:], vw=vw, tau2=vw['tau2'])
            samp = 0

        if K > 1:
            beta_cB = resids[samp, :] # beta component in kernel(B)（for lambda_p/sigma2）
            z_samp = B @ vw['beta'].flatten()
        else:
            beta_cB = vw['beta'] # if only one task, then Range(B) is trivial and beta itself is the "kernel component"
            z_samp = np.zeros(P)

        # Update Lambda_t & Lambda_p
        if 'lambda_t' not in fixed:
            vw['lambda_t'] = steps.sample_lambda_t(vw, omega_prior, P)
        if 'lambda_p' not in fixed:
            vw['lambda_p'] = steps.sample_lambda_p(vw, beta_cB, K, P)

        # Update Sigma2
        if 'sigma2' not in fixed:
            vw['sigma2'] = steps.sample_sigma2(vw, ya, Xa, z_samp, beta_cB, Na, P, K)

        # Update Rho (Sparsity Prior for Eta)
        if 'rho' not in fixed:
            rho_alpha = 1. + np.sum(vw['eta'])
            rho_beta = 1. + (K-1) - np.sum(vw['eta'])
            vw['rho'] = np.random.beta(rho_alpha, rho_beta)

        # Update tau2 via Metropolis-Hastings
        if 'tau2' not in fixed:
            tau2_prop, lalpha = steps.sample_tau2_mh(vw, tau2_prop_scale, eta_nld_curr, grams_all, Xa, ya, Ns)

            ## NOTE: MH Step
            #debug_tau2 = True
            lu = np.log(np.random.uniform())
            isaccept = lalpha > lu 
            if isaccept:
                if debug_tau2:
                    print("tau2 Accept!")
                tau2_accepts += 1
                vw['tau2'] = tau2_prop
            else:
                if debug_tau2:
                    print("tau2 Reject!")
                tau2_rejects += 1
            if debug_tau2:
                print(f"tau2 lalpha: {lalpha}")
            ## NOTE: MH Step

            ## NOTE: Adaptation.
            ## Adjust proposal var.
            alpha = np.minimum(np.exp(lalpha),1.) 
            gamma_it = np.power(1./(i+1),gamma_exp)  
            tau2_prop_scale = np.exp(np.log(tau2_prop_scale) + gamma_it * (alpha - desired_ar))  
            if debug_tau2:
                print(f"tau2 prop scale: {tau2_prop_scale}")
                print(f"tau2 prop was: {tau2_prop}")
                print(f"mean eta val: {np.mean(vw['eta'])}")
            ## NOTE: Adaptation.

        if docorr and 'psi' not in fixed:
            def lp_psi(psi, vw):
                def get_Lk(psi, k):
                    # print("delete this pls.") 
                    err_mat = np.linalg.inv(psi*err_Cs[k]+np.eye(Ns[k]))
                    Lk = np.linalg.cholesky(err_mat)
                    return Lk
                qf = 0
                logdet = 0
                for k in range(len(Xs)):
                    rk = ys[k] - Xs[k]@vw['beta'][k,:]
                    Lk = get_Lk(psi, k)
                    qf += np.sum(np.square(Lk@rk))/(2*vw['sigma2'])
                    logdet += np.sum(np.log(np.diag(Lk)))
                ll = -qf + logdet
                lpri = np.log(psi)
                lp = ll + lpri
                return float(lp)
            
            psi_prop = vw['psi'] + psi_prop_scale*np.random.normal()

            cur_ld = lp_psi(vw['psi'], vw)
            if psi_prop > 0:
                prop_ld = lp_psi(psi_prop, vw)
            else:
                prop_ld = -np.inf
            
            back_lprb = 0.
            prop_lprb = 0.
 
            met_num = prop_ld + back_lprb
            met_den = cur_ld + prop_lprb

            if not np.isfinite(met_den):
                lalpha = -np.inf
            else:
                lalpha = (met_num - met_den)

            ## NOTE: MH Step
            debug_psi = True 
            # debug_psi = False
            lu = np.log(np.random.uniform())
            isaccept = lalpha > lu
            if isaccept:
                if debug_psi:
                    print("psi Accept!")
                psi_accepts += 1
                vw['psi'] = psi_prop
            else:
                if debug_psi:
                    print("psi Reject!")
                psi_rejects += 1
            if debug_psi:
                print(f"psi lalpha: {lalpha}")
            ## NOTE: MH Step

            ## NOTE: Adaptation.
            ## Adjust proposal var.
            alpha = np.minimum(np.exp(lalpha),1.)
            gamma_it = np.power(1./(i+1),gamma_exp)
            psi_prop_scale = np.exp(np.log(psi_prop_scale) + gamma_it * (alpha - desired_ar))
            if debug_psi:
                print(f"psi prop scale: {psi_prop_scale}")
                print(f"psi prop was: {psi_prop}")
                print(f"mean eta val: {np.mean(vw['eta'])}")
            ## NOTE: Adaptation.


        for v in tracking:
            tracking[v][i] = jnp.copy(vw[v])


    # =========================================================================
    # Collect and Visualize Results
    # =========================================================================


    if plotprefix is not None:
        viz.plot_traces(
            tracking=tracking, 
            burnin=burnin, 
            iters=iters, 
            plotprefix=plotprefix, 
            true_vals=true_vals
        )
        
        viz.plot_indicators(
            tracking=tracking, 
            K=K, 
            iters=iters, 
            plotprefix=plotprefix
        )
        if L > 1:
            viz.plot_temperatures(
                delta_tracking=delta_tracking, 
                iters=iters, 
                L=L, 
                plotprefix=plotprefix, 
                d2t=d2t
            )


    # discard Burn-in samples
    for v in tracking:
        tracking[v] = np.take(tracking[v], np.arange(burnin, iters), 0)

    if not 'tau2' in fixed:
        print(f"tau2 rate: {tau2_accepts / (tau2_accepts+tau2_rejects)}")

    switch_rates = [] 
    for l in range(L-1):
        switch_rates.append(np.mean(didswitch[who==l]))

    if verbose:
        print("Accept rates:")
        print(np.mean(etas_accepts, axis=0))
        print("Switching rates:")
        print(switch_rates)

    beta0_hat = np.mean(tracking['beta'], axis=0)[0, :]

    return beta0_hat, tracking, switch_rates





  