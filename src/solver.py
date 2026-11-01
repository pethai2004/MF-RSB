

import time 

import numpy as np 
import scipy
import gmsh

from src.utils import *
from src.core import *


def solve_mfrsb(mesh: Mesh, funcE, stencil, c: Config): # impl nested Fixed-point recursion loop in Alg 1
    
    # initialization 
    dt = c.T/c.Nt 
    E = None 
    mu0, muT = None, None 
    rho = np.full((c.Nt+1, mesh.N), 1/mesh.mass.sum()) # uniform
    f, h = np.zeros_like(rho, c.dtype), np.zeros_like(rho, c.dtype) # f=log(phi), g=log(phi_hat)
    
    # meta = MetaSolver() # mornitor: mass conservation, value near boundary, max/min value of rho, phi, varphi, 
    
    for n in range(1, c.N_iter+1):
        
        ## (1) precompute fixed quantities 
        ops = []
        logW, r, b = np.empty_like(rho), np.empty_like(rho), np.empty((*rho.shape, c.dim))
        for k in range(c.Nt+1):
            logW, r, b = funcE.field(mesh, rho[k], f[k], c.kappa)
            ops += [stencil.build(mesh, b[k], dt, c.kappa)]
        ops = ops[:-1] # remove extra ops
        
        # (2) operator G in for fixed rho^n
        fn, hn = one_step_ops(f, W, r, ops, dt, mu0, muT) 
        
        
        # 2.1 backward propagate + initial Sinkhorn
        
        
        # 2.2 forward prop + terminal Sinkhorn
        
        
        
        
        # main Fixed-point updates
        rho_half = normalize_rho(W+fn+hn, mesh.mass) 
        
        ## (3) damping rho^{n+1} = F(rho^n; varphi, varphi_hat)
        rho_next = (1-c.theta) * rho + c.theta * rho_half # 3.1 smooth out
        
        # diagnostic
        rho_diff = float(np.max(abs(rho_next - rho_half)))



        
    
    return 
            
            