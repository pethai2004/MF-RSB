from pathlib import Path
from dataclasses import dataclass
import time 

import numpy as np 
import scipy
import gmsh

from src.utils import *

@dataclass(frozen=True)
class Mesh:
    nodes : np.ndarray # [N,dim]
    elements : np.ndarray # [M, tri=2/3/4]
    
    def __post_init__(self, c: Config):
        assert c.dim == 2, 'only impl 2dim for now'
        self.N, self.M = len(self.nodes), len(self.elements)
        self.nodes = np.asarray(self.nodes, c.dtype)
        self.elements = np.asarray(self.elements, int)
        self.edges = np.sort(np.concatenate(  # sorting is requred to correctly extract the boundary edges.
            self.elements[:, [0,1]], self.elements[:, [1,2]], self.elements[:, [2,0]]
        ), 1)
        x = self.nodes[self.elements] # [M, tri=2/3/4, dim] 
        # [N, dim, dim] : each [[dx/dxi dx/deta], [dy/dxi, dy/deta]]
        J = np.stack((x[:, 1]-x[:,0], x[:, 2]-x[:,0]), -1) 
        detJ = np.linalg.det(J)
        assert np.any(abs(detJ) > 1e-14)
        
        self.area = abs(detJ) / 2
        self.invJ = np.linalg.inv(J)
        # - [j00 + j10, j01 + j11], [j00, j01], and [j10, j11]  
        self.basis_grad = np.concatenate([-self.invJ.sum(1, keepdims=True), self.invJ], 1)

        eb, nc = np.unique(self.edges, axis=0, return_counts=True)
        self.edge_bc = eb[nc == 1]
        self.node_bc_idx = np.unique(self.edge_bc)
        self.node_bc = self.nodes[self.node_bc_idx]
        
        self.m = np.bincount(self.elements.ravel(), np.repeat(self.area/3, 3), self.N) 
        assert not np.any(self.mass <=0)
        
    def interpolate(self, x): # P1 basis interp
        return None
    
def stencil(dim=2): # 
    assert dim==2 
    
    w_m = [2/3, 1/12, 1/12, 1/12, 1/12] # see weight def eq. 17 for 2dim
    
class FEM:
    
    def __init__(self, c: Config):
        self.c = c 

