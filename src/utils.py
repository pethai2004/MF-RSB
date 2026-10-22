
import numpy as np


class Config:
    
    dtype = np.float32
    
    dim = 1
    T = 1.
    mesh = (25, ), # mush be len(mesh) == dim
    Nt = 20,
    kappa = .03
    
    order_ept = 2
    splitting = 'Strang' # Strang / Lie / Yosida
    ln_solver = None # matrix solver ? including this??

def compute_reflect(x):
    return 

def compute_normal(x, sdf_fn,eps=1e-6): 
    # [Ns, dim=1,2,3] -> [N, dim],  central diff for nabla chi / ||nabla chi||
    chi = sdf_fn(x) 
    nt = np.zeros_like(chi)
    for i in range(x.shape[-1]):
        dx = np.zeros_like(chi)
        dx[:,i] = eps #
        nt[:,i] = ((sdf_fn(chi+dx)-sdf_fn(chi-dx)) / (2*eps)) 
    return nt / (np.max(np.linalg.norm(nt, axis=1, keepdims=True), 1e-12))
    
        

def path_sim(pde, config: Config): # forward Euler-Maruyama scheme
    pass