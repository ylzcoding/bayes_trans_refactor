import numpy as np
import jax.numpy as jnp
# from tensorflow_probability.substrates import jax as tfp 
# tfpd = tfp.distributions

"""
Gibbs sampler
"""

def sample_omega(vw, z_samp, omega_prior):
    """Update auxiliary variable omega"""
    if omega_prior == 'exp':
        nu_ig = np.sqrt(vw['sigma2']*np.square(vw['lambda_t']/(z_samp+1e-8)))
        lam_ig = np.square(vw['lambda_t'])
        omega_presamp = np.random.wald(mean=nu_ig, scale=lam_ig)
        return 1.0 / (omega_presamp + 1e-8)
    elif omega_prior == 'ig':
        shape_g = 1.0
        rate_g = 0.5*(1/jnp.square(vw['lambda_t']) + jnp.square(z_samp)/vw['sigma2'])
        return 1.0 / np.random.gamma(shape_g, 1/rate_g)
    return vw['omega']

def sample_lambda_t(vw, omega_prior, P):
    """Update lambda_t"""
    A = 1.0 #  Cauchy scale.
    if omega_prior == 'exp':
        phi_samp = 1/jnp.square(vw['lambda_t'])
        a_samp = 1/np.random.gamma(1., 1/(A+1/phi_samp))
        phi_samp = 1/np.random.gamma(P+0.5, 1/(1/a_samp + 0.5*jnp.sum(vw['omega'])))
        return 1/jnp.sqrt(phi_samp)
    elif omega_prior == 'ig':
        phi_samp = np.square(vw['lambda_t'])
        a_samp = 1/np.random.gamma(1., 1/(A+1/phi_samp))
        g_rate = 1/a_samp + 0.5*jnp.sum(1/vw['omega'])
        phi_samp = 1/np.random.gamma((P+1)/2, 1/g_rate)
        return jnp.sqrt(phi_samp)
    return vw['lambda_t']

def sample_lambda_p(vw, beta_cB, K, P):
    """Update lambda_p"""
    phi_samp = 1.0 / vw['lambda_p']
    A = 1.0 # Cauchy Scale
    a_samp = 1.0 / np.random.gamma(1.0, 1.0 / (A + 1.0 / phi_samp))
    beta_proj_norm2 = jnp.sum(jnp.square(beta_cB)) / vw['sigma2']
    dof = (K - 1) * P
    phi_samp = 1.0 / np.random.gamma(dof/2.0 + 0.5, 1.0 / (beta_proj_norm2/2.0 + 1.0 / a_samp))
    
    return 1.0 / phi_samp



def sample_tau2_mh(vw, tau2_prop_scale, eta_nld, grams_all, Xa, ya, Ns):
    """
    Use Metropolis-Hastings to update tau2
    """
    tau2_pd = 'rw_normal'  
    symmetric_proposals = ['rw_normal','rw_cauchy']  
    
    if tau2_pd=='rw_cauchy':
        tau2_prop = vw['tau2'] + tau2_prop_scale*np.random.standard_cauchy()  
    elif tau2_pd=='rw_normal':
        tau2_prop = vw['tau2'] + tau2_prop_scale*np.random.normal() 
    elif tau2_pd=='unif':
        tau2_prop = np.power(10.,np.random.uniform(-2,6))
    else:
        raise Exception("Unknown tau2 prop dist!")

    cur_nll, _ = eta_nld(vw['eta'][None,:], grams_all, Xa, ya, vw, Ns, vw['tau2'])  
    if tau2_prop > 0:
        prop_nll, _ = eta_nld(vw['eta'][None,:], grams_all, Xa, ya, vw, Ns, tau2_prop)
    else:
        prop_nll = [np.inf]

    # cur_lpd = tfpd.Cauchy(0,1).log_prob(vw['tau2'])  
    # prop_lpd = tfpd.Cauchy(0,1).log_prob(tau2_prop)  try to get rid of tfp dependency since it causes some issues with jax and jit
    cur_lpd = -jnp.log(jnp.pi) - jnp.log1p(vw['tau2']**2)
    prop_lpd = -jnp.log(jnp.pi) - jnp.log1p(tau2_prop**2)

    if tau2_pd in symmetric_proposals:
        back_lprb = 0.  
        prop_lprb = 0.  
    elif tau2_pd=='unif':
        back_lprb = np.log10(vw['tau2'])
        prop_lprb = np.log10(tau2_prop)
    else:
        raise Exception("Unknown tau2 prop dist!")

    met_num = -prop_nll[0] + prop_lpd + back_lprb  
    met_den = -cur_nll[0] + cur_lpd + prop_lprb 

    if not np.isfinite(met_den):
        lalpha = -np.inf
    else:
        lalpha = (met_num - met_den)  # log-accept ratio of tau2

    return tau2_prop, lalpha

def sample_sigma2(vw, ya, Xa, z_samp, beta_cB, Na, P, K):

    shape_prec = (Na+P*K) / 2.0
    sse = jnp.sum(jnp.square(ya - Xa @ vw['beta'].flatten()))
    ssp1 = jnp.sum(jnp.square(z_samp) / vw['omega'])
    beta_proj_norm2 = jnp.sum(jnp.square(beta_cB))
    ssp2 = beta_proj_norm2 * vw['lambda_p']
    scale_prec = 1.0 / ((sse + ssp1 + ssp2) / 2.0)
    sigma2_new = 1.0 / np.random.gamma(shape=shape_prec, scale=scale_prec)
    
    return sigma2_new