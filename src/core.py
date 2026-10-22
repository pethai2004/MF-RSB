from pathlib import Path

import numpy as np 
import scipy

from src.utils import *

class BC: # all BC must impl : reflect, normal, and sdf
    
    def __init__(self, c: Config):
        self.c = c 
        
    def reflect(self, x): 
        pass
    
    def sdf(self, x):
        # recursively call max one each _sdf
        
        pass 
class _FEM: 
    
    def __init__(self, c):
        pass
    
    def ass(self):
        pass 
    
    def grad(self):
        pass
    
    def lap(self):
        pass
    
class PDE: # subclass "forward, backward" to define specific PDEs

    rho = None
    vphi = None #varphi
    mphi = None #varphi_hat
    
    def __init__(self, config: Config):
        self._FEM = _FEM()
    
    def _forward(self): pass
    
    def _backward(self): pass 

def approx_ept(order=2): # compute expectation : 
    # order either 2/4/6 
    pass 

def alg_step(pde, config): # one step of the algorithm
    pass

def path_sim(pde: PDE, config: Config): # forward Euler-Maruyama scheme
    pass

def solve_mon(pde: PDE):
    pass 