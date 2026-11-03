import time 

import numpy as np 
import scipy
import gmsh

from src.utils import *
from src.core import *

# impl nested Fixed-point recursion loop in Alg 1
def solve_mfrsb(mesh: Mesh, E: Ekern, SL : SLStenc0, c: Config): 
    
    # initialization 
    dt = c.T/c.Nt 
    lgmu0, lgmuT = np.log(None), np.log(None) 
    rho = np.full((c.Nt+1, mesh.N), 1/mesh.mass.sum()) # uniform
    f, h = np.zeros_like(rho, c.dtype), np.zeros_like(rho, c.dtype) # f=log(phi), g=log(phi_hat)
    
    # meta = { # A
    #     "rho_diff" : [], "mu0_err" : [], "muT_err" : [], "max_pnt" : [], "EOC" :[],
    #     "err_adj" : [], "CPU_time_n" : [], "contract_fact" : [], "AT" : [], "A0T" : []
    # }
    for n in range(1, c.N_iter+1):
        
        ## (1) precompute fixed quantities 
        ops = []
        logW, r, b = np.empty_like(rho), np.empty_like(rho), np.empty((*rho.shape, c.dim))
        for k in range(c.Nt+1):
            logW, r, b = E.field(mesh, rho[k], f[k], c.kappa)
            ops += [SL.build(mesh, b[k], dt, c.kappa)]
        ops = ops[:-1] # remove extra ops
        
        fn, hn = np.empty_like(f), np.empty_like(h)
        
        fn[-1] = f[-1]
        for k in range(len(SL)-1, -1,-1):
            f[k] = .5*dt*r[k] + ops[k].log_bwd(f[k+1] + .5*dt*r[k+1])
        h[0] = lgmu0 - logW[0] - fn[0]
        
        for k, op in enumerate(ops):
            h[k+1] = .5*dt*r[k+1] + op.log_fwd(h[k] + .5*dt*r[k])
        f[-1] = lgmuT - logW[-1] - hn[-1]
        
        # main Fixed-point updates
        rho_half = normalize_rho(W+fn+hn, mesh.mass) 
        
        ## (3) damping rho^{n+1} = F(rho^n; varphi, varphi_hat)
        rho_next = (1-c.theta) * rho + c.theta * rho_half # 3.1 smooth out
        
        # diagnostic
        rho_diff = float(np.max(abs(rho_next - rho_half)))



        