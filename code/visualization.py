import matplotlib.pyplot as plt
import numpy as np

def plot_traces(tracking, burnin, iters, plotprefix, true_vals=None):

    if plotprefix is None:
        return

    toplot = 4 
    nvars = [int(np.prod(tracking[v].shape[1:])) for v in tracking]  
    varseach = [np.minimum(toplot,nvars[vi]) for vi,v in enumerate(tracking)] 
    plotseach = [2*varseach[vi] for vi,v in enumerate(tracking)] 
    nplots = np.sum(plotseach)  
    ncols = 4  
    nrows = int(np.ceil(nplots/ncols)) 
    plt.figure(figsize=[1.5*ncols,1.5*nrows])
    plottrans = {'sigma2' : np.log10, 'lambda_t' : np.log10, 'lambda_p' : np.log10, 'omega' : np.log10, 'tau2' : np.log10}
    
    for vi,v in enumerate(tracking):
        samps = tracking[v].reshape([iters,nvars[vi]]) 
        trans = plottrans[v] if v in plottrans else lambda x: x 
        for p in range(varseach[vi]):
            title = fr"$\{v}^{p}$"
            plt.subplot(nrows,ncols,1+int(np.sum(plotseach[:vi]))+2*p)
            plt.hist(trans(samps[burnin:,p]))
            if true_vals is not None and v in true_vals:
                ll,ul = plt.gca().get_ylim()  
                try:
                    tv = true_vals[v][p]
                except Exception:
                    tv=true_vals[v]
                tv = trans(tv)  
                plt.vlines(tv,ll,ul,color='orange',linestyle='--')
            plt.title(title)
            plt.subplot(nrows,ncols,1+int(np.sum(plotseach[:vi]))+2*p+1)
            plt.plot(trans(samps[:,p]))
            ll,ul = plt.gca().get_ylim() 
            plt.vlines(burnin,ll,ul,color='gray',linestyle='--')
            if true_vals is not None and v in true_vals:
                ll,ul = plt.gca().get_xlim() 
                plt.hlines(tv,ll,ul,color='orange',linestyle='--')
            plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{plotprefix}_plot.pdf")
    plt.close()


def plot_indicators(tracking, K, iters, plotprefix):
  
    if plotprefix is None:
        return

    # Visualize dataset indicator trajectory.
    # Not exactly MCA because that seems to give weird results here.
    # Z = tracking['eta'] / (np.sum(tracking['eta'])+1e-3)
    # r = np.mean(Z,axis=1) + 1/Z.shape[0]
    # c = np.mean(Z,axis=0) + 1/Z.shape[1]
    # tosvd = np.diag(1/np.sqrt(r)) @ (Z - r[:,None,]@c[None,:]) @ np.diag(1/np.sqrt(c))
    
    mu = np.mean(tracking['eta'],axis=0)  
    sig = np.std(tracking['eta'], axis=0)+1e-6  
    tosvd = (tracking['eta'] - mu[None,:]) / sig[None,:] 
    Zsvd = np.linalg.svd(tosvd,full_matrices=False) 
    projr = Zsvd[2].T[:,:2] 
    z = tosvd @ projr  

    nrand = 20 
    rand = np.random.binomial(1,0.5,size=[nrand,K-1]) 
    rand = (rand - mu[None,:]) / sig[None,:] 
    zrand = rand @ projr  

    bound = np.stack([np.zeros(K-1),np.ones(K-1)])  
    bound = (bound - mu[None,:]) / sig[None,:] 
    zbound = bound @ projr  

    # Yellow is Later (yater)
    if K > 2:
        fig = plt.figure()
        zj = z 
        plt.scatter(zj[:,0],zj[:,1], color = plt.colormaps['autumn'](np.arange(iters)/iters))
        plt.scatter(zrand[:,0],zrand[:,1], color = 'blue', label = 'rand')
        plt.scatter(zbound[0,0],zbound[0,1], color = 'cyan', label = 'zeros')
        plt.scatter(zbound[1,0],zbound[1,1], color = 'purple', label = 'ones')
        plt.legend()
        plt.savefig(f"{plotprefix}_inds.pdf")
        plt.close()

def plot_temperatures(delta_tracking, iters, L, plotprefix, d2t):
    plt.figure()
    temps_tracking = np.apply_along_axis(d2t, 1, delta_tracking) 
    for l in range(L):
        p = plt.plot(temps_tracking[:,l])
        plt.hlines(temps_tracking[0,l], 0, iters, color = p[0].get_color(), linestyle = '--')
    plt.yscale('log')
    plt.savefig(f"{plotprefix}_switch.pdf")
    plt.close()