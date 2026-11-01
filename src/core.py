from pathlib import Path
from dataclasses import dataclass
import time 

import numpy as np 
import matplotlib
import scipy
import gmsh

from src.utils import *

@dataclass #(frozen=True)
class Mesh:
    nodes : np.ndarray # [N,dim]
    elements : np.ndarray # [M, tri=2/3/4]
    
    def __post_init__(self):

        self.nodes = np.asarray(self.nodes)
        self.elements = np.asarray(self.elements, int)
        
        # (TODO) : how about assert and assume already preprocessed
        valid_idx, invm = np.unique(self.elements, True)
        if len(valid_idx) < len(self.nodes): 
            print(f"Found disconnected nodes")
            self.nodes = self.nodes[valid_idx]
            self.elements = invm.reshape(self.elements.shape)

        self.N, self.M = len(self.nodes), len(self.elements)
        self.edges = np.sort(np.concatenate(  # sorting is requred to correctly extract the boundary edges.
            [self.elements[:, [0,1]], self.elements[:, [1,2]], self.elements[:, [2,0]]]
        , 0), 1)
        unqiue_e, count = np.unique(self.edges, axis=0,return_counts=True)
        self.bc_edges = unqiue_e[count == 1]
        self.bc_idx = np.unique(self.bc_edges)
        self.bc = self.nodes[self.bc_idx]
        
        x = self.nodes[self.elements] # [M, tri=2/3/4, dim] 
        # [N, dim, dim] : each [[dx/dxi dx/deta], [dy/dxi, dy/deta]]
        J = np.stack((x[:, 1]-x[:,0], x[:, 2]-x[:,0]), -1) 
        detJ = np.linalg.det(J)
        if np.any(np.abs(detJ)<1e-14): raise ValueError("degenerate mesh founded")
        self.area = abs(detJ) / 2
        self.invJ = np.linalg.inv(J)
        # - [j00 + j10, j01 + j11], [j00, j01], and [j10, j11]  
        Nxa, Nxb = self.invJ[:, 0, :], self.invJ[:, 1, :]
        Nxc = -(Nxa + Nxb)
        self.basis_grad = np.stack([Nxa, Nxb, Nxc], 1)

        self.mass = np.bincount(self.elements.ravel(), weights=np.repeat(self.area / 3.0, 3), minlength=self.N)
        assert np.all(self.mass > 0)
        self.tri = matplotlib.tri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.elements)
        self.find = self.tri.get_trifinder()
        
        
    def interp(self, x):  # x : [N, dim=2]
        
        e = self.find(x[:,0], x[:,1]) # triangle `e` which `x` belongs to
        dist = x - 
    
    def grad(self, x):
        return 
    
@dataclass 
class SLStenc:
    
    x0 : np.ndarray # traceback characteristic point
    x1 : np.ndarray # landing point node
    w : np.ndarray # weights
    mass : np.ndarray 
    reflected : int = 0
    
    @classmethod
    def build(cls, mesh, b, dt, kappa):
    
        w_m = [2/3, 1/12, 1/12, 1/12, 1/12] # see weight def eq. 17 for 2dim

