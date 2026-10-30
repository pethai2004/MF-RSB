

import time 

import numpy as np 
import scipy
import gmsh

from src.utils import *
from src.core import *


def precompute_ops(): # impl fixed quantity
    pass 

def one_step_ops(): # impl inner ops (eq : cite []), S_back -> M_T-> S_fwd -> M_0
    
    # backward ops 
    
    
    # first marginal scale
    
    
    # forward ops
    
    
    # second marginal scale
    pass


def normalize_rho():
    pass 

def solve_mfrsb(mesh: Mesh, c: Config): # impl nested Fixed-point recursion loop in Alg 1
    
    # initialization 
    dt = c.T/c.Nt 
    E = None 
    mu0, muT = None, None 
    rho = np.full((c.Nt+1, mesh.N), 1/mesh.mass.sum()) # uniform
    f, h = np.zeros_like(rho, c.dtype), np.zeros_like(rho, c.dtype) # f=log(phi), g=log(phi_hat)
    
    # meta = MetaSolver() # mornitor: mass conservation, value near boundary, max/min value of rho, phi, varphi, 
    
    for n in range(1, c.N_iter+1):
        
        W, _, r, ops = precompute_ops()
        fn, hn = one_step_ops(f, W, r, ops, dt, mu0, muT)
        
        rho_half = normalize_rho(W+fn+hn, mesh.mass)
        rho_next = (1-c.theta) * rho + c.theta * rho_half
        
        rho_diff = float(np.max(abs(rho_next - rho_half)))



        
    
    return 
            
            