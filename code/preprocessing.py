import numpy as np
import jax.numpy as jnp
from scipy.linalg import block_diag


def prepare_data(Xs, ys):

    Ns = jnp.array([X.shape[0] for X in Xs])
    Ps = [X.shape[1] for X in Xs]
    assert np.all([P==Ps[0] for P in Ps]), "All tasks must have same feature dimension"
    P = Ps[0]
    K = len(Ns)

    Xa = block_diag(*Xs)
    ya = np.concatenate(ys)
    Na = np.sum(Ns)

    # reduced QR
    grams_all_np = np.empty((K, P, P))
    grams_chols_np = np.empty((K, P, P))
    Xty_np = np.empty((K, P, 1))

    for k in range(K):
        Qk, Rk = np.linalg.qr(np.asarray(Xs[k]), mode='reduced')
        dsign = np.sign(np.diag(Rk))
        dsign[dsign == 0] = 1.0
        Rk = dsign[:, None] * Rk
        
        grams_chol_k = Rk.T  # Cholesky-like factor
        grams_chols_np[k] = grams_chol_k
        grams_all_np[k] = grams_chol_k @ Rk
        Xty_np[k, :, 0] = grams_chol_k @ (Qk.T @ np.asarray(ys[k]))

    data_struct = {
        'Ns': Ns, 'P': P, 'K': K, 'Na': Na,
        'Xa': jnp.array(Xa), 'ya': jnp.array(ya),
        'grams_all': jnp.array(grams_all_np),
        'grams_chols': jnp.array(grams_chols_np),
        'Xty': jnp.array(Xty_np)
    }
    
    # pre-compute inverses for later use in sampling
    data_struct['grams_invchols'] = jnp.linalg.inv(data_struct['grams_chols'])
    data_struct['grams_inv_all'] = jnp.linalg.inv(data_struct['grams_all'])

    return data_struct