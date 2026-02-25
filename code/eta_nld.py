import jax
import jax.numpy as jnp
import numpy as np
from linear_algebra import get_B


def _eta_nld(etas, grams_all, Xa, ya, vw, Ns, tau2, rb, omega_prior, P, K):
    """
    Negative Log Posterior Density given eta (Source Selection)。
    etas: shape [n_eta, K-1
    """

    Bval = get_B(grams_all, etas, Ns, tau2)

    if rb == 'betasig':
        svd = jnp.linalg.svd(Bval,full_matrices=False)
        Ub = svd[2]
        ### Slow but explicit code.
        #pri_prec1 = vw['lambda_p']*(jnp.eye(K*P)-jnp.matrix_transpose(Ub)@Ub)
        #pri_prec2 = jnp.matrix_transpose(Bval)@jnp.diag(1/vw['omega'])@Bval
        #pri_prec = pri_prec1 + pri_prec2
        #pri_prec_logdet = jnp.linalg.slogdet(pri_prec)[1]
        ### Slow but explicit code.

        # TODO: DRY
        ## fast but implicit code.
        disp = -0.5*jnp.sum(jnp.log(vw['omega'])) 
        #Bdetterm = jnp.linalg.slogdet(sob @ jnp.matrix_transpose(sob))[1]
        detBBT = jnp.linalg.slogdet(Bval @ jnp.matrix_transpose(Bval))[1]  # log|B B^T|
        #C_rB = 0.5*detBBT + disp
        C_rB = 0.5*detBBT + disp
        # C_rb gives the log normalizing constant for the distribution of beta in the range of B.
        C_kB = (K-1)*P/2 * jnp.repeat(jnp.log(vw['lambda_p']), etas.shape[0])
        # C_rb gives the log normalizing constant for the distribution of beta in the kernel of B.
        logC = C_rB + C_kB  
        pri_prec_logdet = 2*logC  
        ## fast but implicit code.
        # TODO: DRY

        ### Slow but explicit code.
        #GG = block_diag(*[grams_all[k,:,:] for k in range(K)])
        #post_prec = GG+pri_prec
        #post_prec_logdet = jnp.linalg.slogdet(post_prec)[1]
        ### Slow but explicit code.

        ## fast but implicit code.
        # Just matrix det lemma.
        grams_aug_all = grams_all + vw['lambda_p']*jnp.eye(P)[None,:,:]  # [K,P,P]
        Btens = Bval.transpose([0,2,1]).reshape([etas.shape[0],K,P,P])  
        VtAU = Bval @ (jnp.linalg.inv(grams_aug_all[None,:,:,:]) @ Btens).reshape([etas.shape[0],K*P,P])  
        W = jnp.diag(1/vw['omega'])[None,:,:] - vw['lambda_p']*jnp.linalg.inv(Bval@jnp.matrix_transpose(Bval)) 
        ld = lambda X: jnp.linalg.slogdet(X)[1] 
        ldA = jnp.sum(ld(grams_aug_all))  # sum_k log|A_k|
        DELTA = jnp.linalg.inv(W)+VtAU  
        post_prec_logdet = ld(DELTA) + ld(W) + ldA[None]  
        #VtAU = Bval @ block_diag(*grams_all) @ jnp.matrix_transpose(Bval)
        ## fast but implicit code.

        logdet_term = 0.5*(pri_prec_logdet - post_prec_logdet)

        ### Slow but explicit code.
        #yhat = (Xa[None,:,:] @ jnp.linalg.solve(post_prec, Xa.T @ ya)[:,:,None])[:,:,0]
        ### Slow but explicit code.

        ## fast but implicit code.
        Xty = Xa.T @ ya  
        AiXty = jnp.linalg.solve(grams_aug_all, Xty.reshape([K,P,1])) 
        AiXty = AiXty.reshape([K*P,1])  
        #Ai = jnp.linalg.inv(block_diag(*grams_all) + vw['lambda_p']*jnp.eye(K*P))
        #AiXty = Ai @ Xty
        mm = jnp.matrix_transpose(Bval)@ (jnp.linalg.solve(DELTA, (Bval @ AiXty[None,:,:])))  
        Aimm = jnp.linalg.solve(grams_aug_all[None,:], mm.reshape([etas.shape[0],K,P,1]))  
        #jnp.matrix_transpose(Bval) @ (DELTA @ (Bval @ (Ai @ Xty))[:,:,None])
        #Aimm = Ai[None,:,:] @ mm
        #diff = AiXty[None,:,None] - Aimm
        #yhat = Xa[None,:,:] @ diff
        Aimm = Aimm.reshape([etas.shape[0],K*P,1])
        yhat = (Xa[None,:,:] @ (AiXty[None,:,:] - Aimm))[:,:,0] 
        ## fast but implicit code.

        n_term = -Xa.shape[0]/2*jnp.log(jnp.sum(ya[None,:]*(ya[None,:]-yhat),axis=1))  

        ll = logdet_term + n_term  
        nll = -ll 

        v = jnp.tile(vw['beta'].flatten(),[etas.shape[0],1])[:,None,:] 
        sol = jax.vmap(lambda X: jnp.linalg.lstsq(X[:-1,:].T,X[-1,:])[0])(jnp.concatenate([Bval,v],axis=1))  
        resids = v[:,0,:] - (jnp.matrix_transpose(Bval) @ sol[:,:,None])[:,:,0] 
    else:
    ## Form the (log of the) term in front of the exponential, involving determinants.
        if rb == 'omega':
            if omega_prior == 'exp':
                disp = P*(jnp.log(vw['lambda_t']) - 0.5*jnp.log(vw['sigma2']) - jnp.log(2))
            elif omega_prior == 'ig':
                disp = P*(jnp.log(vw['lambda_t']) - 0.5*jnp.log(vw['sigma2']) - jnp.log(np.pi))
            else:
                raise Exception("Oh noes!")
        else:
            disp = -0.5*jnp.sum(jnp.log(vw['omega'])) - P/2.*jnp.log(2*np.pi*vw['sigma2'])

        detBBT = jnp.linalg.slogdet(Bval @ jnp.matrix_transpose(Bval))[1]
        C_rB = 0.5*detBBT + disp
        C_kB = (K-1)*P/2 * (jnp.repeat(jnp.log(vw['lambda_p']), etas.shape[0]) - jnp.log(2*np.pi*vw['sigma2']))
        logC = C_rB + C_kB

        v = jnp.tile(vw['beta'].flatten(),[etas.shape[0],1])[:,None,:]
        sol = jax.vmap(lambda X: jnp.linalg.lstsq(X[:-1,:].T,X[-1,:])[0])(jnp.concatenate([Bval,v],axis=1))
        
        resids = v[:,0,:] - (jnp.matrix_transpose(Bval) @ sol[:,:,None])[:,:,0]
        neg_n_kB = 0.5 * vw['lambda_p'] * jnp.sum(v[:,0,:]*resids,axis=1) / vw['sigma2']
        
        if rb == 'omega':
            z = Bval @ jnp.matrix_transpose(v)
            z = z[:,:,0]
            if omega_prior == 'exp':
                neg_n_rB = vw['lambda_t']/jnp.sqrt(vw['sigma2'])*jnp.sum(jnp.abs(z),axis=1)
            elif omega_prior == 'ig':
                neg_n_rB = jnp.sum(jnp.log1p(jnp.square(vw['lambda_t'])/vw['sigma2']*jnp.square(z)),axis=1)
            else:
                    raise Exception("Unknown omega_prior.")
        else:
            neg_n_rB = jnp.matrix_transpose(Bval) @ (jnp.diag(1/vw['omega'])[None,:,:] @ (Bval@jnp.matrix_transpose(v)))
            neg_n_rB = neg_n_rB[:,:,0]
            neg_n_rB = 0.5 * jnp.sum(v[:,0,:]*neg_n_rB,axis=1) / vw['sigma2']
            
        neg_norm = neg_n_kB + neg_n_rB
        # neg_n_kB gives the opposite of the term in the exponent for the distribution of the part of beta in the kernel of B. 
        # neg_n_rB same but for the part in the range.
        ####
        nll = -logC + neg_norm

    # Note: tfpd.Bernoulli is assumed, or implement manually:
    # log_prob = eta * log(rho) + (1-eta) * log(1-rho)
    prior_contrib = -jnp.sum(etas * jnp.log(vw['rho']) + (1-etas)*jnp.log(1-vw['rho']), axis=1)
    
    return nll + prior_contrib, resids