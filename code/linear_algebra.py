import jax
import jax.numpy as jnp
import numpy as np

@jax.jit
def get_B(grams_all, etas, Ns, tau2):
    """
    Get Transfer Matrix B。
    grams_all: shape (K, P, P)， Gram maxtrix for each source(X_1'X_1, X_2'X_2, ..., X_K'X_K)
    etas: shape (L, K-1)
    Ns: shape (K,)，# of samples for each task
    """
    K = grams_all.shape[0]
    P = grams_all.shape[1]
    # E_0 = [I_P, 0_P, ..., 0_P]
    E0 = jnp.concatenate([jnp.eye(P)]+[jnp.zeros([P,P]) for _ in range(K-1)], axis=1)

    # only consider source tasks selected by eta
    grams_rel = grams_all[None,1:,:,:] * etas[:,:,None,None]
    
    # (Sum X'X + tau2*I) * T = X'X_source
    cat_grams = jnp.matrix_transpose(grams_rel).reshape([etas.shape[0],(K-1)*P,P]).transpose([0,2,1]) # concat X'X sources into shape (L, P, (K-1)*P)
    sum_grams = jnp.sum(grams_rel, axis=1)
    
    rel_Ns = jnp.sum(Ns[None,1:]*etas, axis=1)
    sum_grams = sum_grams + (jnp.sum(etas,axis=1)==0)[:,None,None]*jnp.eye(P)[None,:,:] 
    sum_grams += tau2 * jnp.eye(grams_all.shape[1])[None,:,:] * rel_Ns[:,None,None]
    
    # solve for T: shape (L, P, (K-1)*P) 
    # T = (\sum X'X + \tau^2 I)^{-1} \cdot [X_1'X_1, ...]$
    T = jnp.linalg.solve(sum_grams, cat_grams)
    T = jnp.concatenate([jnp.zeros([etas.shape[0],P,P]), T], axis=2) # concat with zeros for target task
    
    B = E0[None,:,:] - T
    return B

@jax.jit
def woodbury_mean(B, lambda_p, omega, grams_aug_inv_all, bha_aug):
    """
    lambda_p: kernel space regularization
    omega: auxiliary variable
    grams_aug_inv_all: pre-computed (X_k^T X_k + \lambda_p I)^{-1} for all tasks, shape (K, P, P)
    bha_aug: pre-computed (X_k^T X_k + \lambda_p I)^{-1} X_k^T y_k for all tasks, shape (K, P)
    """
    K = grams_aug_inv_all.shape[0]
    P = grams_aug_inv_all.shape[1]
    b1 = B @ bha_aug.flatten()
    Ba = B.reshape([P,K,P]).transpose([1,0,2])
    
    BXtXiBt = jnp.sum(Ba @ grams_aug_inv_all @ jnp.matrix_transpose(Ba), axis=0)
    S = jnp.linalg.inv(jnp.diag(1/omega)-lambda_p*jnp.linalg.pinv(B@B.T)) + BXtXiBt 
    
    b2 = jnp.linalg.solve(S, b1)
    b3 = B.T @ b2
    b4 = grams_aug_inv_all @ b3.reshape([K,P,1])
    b5 = bha_aug - b4

    return b5

@jax.jit
def get_beta_samp(grams_invchols, B, omega_samp, lambda_samp, xi_1, xi_2, beta_mean, sigma2_samp):
    """
    grams_invchols: shape (K, P, P)
    xi_1: random normal vector for Range(B)
    xi_2: random normal vector for Kernel(B)
    beta_mean: results from woodbury_mean, shape (K, P)
    """
    K = grams_invchols.shape[0]
    P = grams_invchols.shape[1]
    
    LiBT = (grams_invchols @ B.transpose([1,0]).reshape([K,P,P])).reshape([K*P,P])

    svLiB = jnp.linalg.svd(LiBT, full_matrices=False)
    U1_pre = svLiB[0]
    Sigma = jnp.diag(svLiB[1])
    V = svLiB[2].T
    
    BBT = B @ B.T
    BBTpi = jnp.linalg.inv(BBT)
    
    DELTA = Sigma @ (V.T @ (jnp.diag(1/omega_samp)- lambda_samp*BBTpi)@V) @ Sigma
    lamd, W = jnp.linalg.eigh(DELTA)
    U1 = U1_pre @ W

    D = jnp.diag(1/jnp.sqrt(1+lamd))
    a1 = U1 @ (D @ xi_1)
    a2 = xi_2 - U1 @ (U1.T @ xi_2)
    a = a1 + a2
    
    LiTa = (jnp.matrix_transpose(grams_invchols) @ a.reshape([K,P,1])).flatten()
    beta_working = jnp.sqrt(sigma2_samp) * LiTa + beta_mean

    return beta_working


