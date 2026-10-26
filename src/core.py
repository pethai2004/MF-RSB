from pathlib import Path
import time 


import numpy as np 
import scipy
import gmsh

from src.utils import *


def solve_mfrsb(nodes, elements, geom, c: Config ): # main solver: Pichard-Sinkhorn MF-RSB problem
    # nodes : [N, dim], ele (int) # [N, 1], 
    fem = _FEM()



class FBPDE: # main FB-PDEs 

    def __init__(self, c: Config):
        self.c = c
        self._fem = _FEM()
        self.record = None
        
    def forward(self): pass   # time marching forward PDEs

    def backward(self) : pass #vvc

    def solve(self):
        
        print(f"=============== Initialize Solver ===============") 
        print(f"===> Node: {self._fem.n_nodes}, Elements: {self._fem.n_eles}")
        
        Nt = self.c.Nt
        for n in range(1, Nt+1): # fixed rho
            
            ct0 = time.time() 
            
            for k in range(Nt+1): # precompute fixed nonlinear quantities
                pass 
            
            for k in range(self, Nt-1, -1, -1): # backward propagate varphi_hat
                
                pass 
            
            
            for k in range(Nt): # forward propagate varphi
                
                pass 
            
        print(f"=============== Done Solver ===============")
        
        
    
    
            
            
            

class _FEM: 
    
    def __init__(self, nodes, ele, c: Config):
        self.X, self.ele = nodes, ele
        self.n_nodes, self.n_eles = len(nodes), len(ele)
    
    def grad(self): pass 
    
    def interp(self): # linear interp
        pass  
    
    def path_skorokhod(self): # characteristic path with reflection at BC.
        pass
    