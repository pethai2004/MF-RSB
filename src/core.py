from dataclasses import dataclass, field
from typing import List, Tuple
import time 

import numpy as np 
from numpy import linalg
import matplotlib
import scipy
from scipy.sparse import csr_matrix
import gmsh

from src.utils import *

@dataclass
class DOM: # (maybe [TODO]: other outer boundary such as hexagon/curve)
    n_resolves : int =3
    bounds : tuple[int] = (-2, 2, -1, 1) # [x0, x1, y0, y1]
    
    def _sdf_gradchi(self, x, dists, glists) : return dists, glists
    def sdf_gradchi(self, x: ndarray): 
        # compute SDF and its gradients, x: [num, dim=2] 
        assert x.ndim==2
        dists, grads = [], []
        dists += [x[:,0]-self.bounds[0], self.bounds[1]-x[:,0], # l,r,b,t
                x[:,1]-self.bounds[2], self.bounds[3]-x[:,1]]
        wall_grads = np.array([[1,0], [-1,0], [0,1],[0,-1]], float)
        grads = [np.broadcast_to(wall_grads[i], (x.shape[0], 2)).copy() for i in range(4)] # dim=2
        dists, grads = self._sdf_gradchi(x, dists, grads) # downstream impl
        dists = np.column_stack(dists) # [N=x.shape[0], num "obs pieces"] 
        #e.g. 3circles with 2 rects will give 1+1+4+4 final dim
        minidx = dists.argmin(1) # closest piece obs
        chi = dists[np.arange(x.shape[0]), minidx]
        
        grad_chi = np.zeros((x.shape[0], 2)) # dim=2
        for j, G in enumerate(grads):
            mask = (minidx == j)
            if np.any(mask): grad_chi[mask] = G[mask]
        
        return chi, grad_chi # [num, 1] ? [num, dim=2]
    # [TODO: try implement substeping instead]
    def reflect(self, x: ndarray): # [num_point, dim=2]
        # compute Skorakhod reflects (eq 17): for chi(x) < 0 : y_hat = y-2*chi(x)*grad_chi
        x_hat = x.copy()
        for _ in range(self.n_resolves):
            chi, grad_chi = self.sdf_gradchi(x_hat)
            mask = (chi<0.) # (out)
            if not np.any(mask) : break # no penetration
            x_hat[mask] = x_hat[mask] - 2.*chi[mask, None]*grad_chi[mask]
            
        return x_hat 
    
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
        self.basis_grad = np.stack([Nxa, Nxb, Nxc], 1) # [N, 3]
        
        # i.e. mass = zero(N) and for each loc in range(3): add.at(mass, elements[:,loc], area/3)
        self.mass = np.bincount(self.elements.ravel(), weights=np.repeat(self.area / 3.0, 3), minlength=self.N)
        assert np.all(self.mass > 0)
        self.M_lump, self.M_lump_inv = np.diag(self.mass), np.diag(1. / self.mass)
        
        self.tri = matplotlib.tri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.elements)
        self.find = self.tri.get_trifinder()
        self.tree = scipy.spatial.cKDTree(self.nodes) # NB search
        
        self.Dx, self.Dy = self.ass_react() # precompute D_l = M_lump^-1 * G_l for l={x,y}
        
    def interp(self, x):  # x : [N, dim=2]
        
        e = self.find(x[:,0], x[:,1]) # triangle `e` which `x` belongs to

        
    # def grad(self, x):
    #     return 
    
    def ass_react(self) -> tuple[csr_matrix, csr_matrix]:
        # precompute derivaitve metric for non-local reaction, return D_l = M_lump^-1 * G_l for l={x,y}
        rs, cs, Zx, Zy = [], [], [], [] # [TODO try use ndim sparse matrix lib]
        # for each vertex pair (i,j) in {0,1,2}^2
        # compute int_K beta_i * partial_x beta_j dx = areaK/3 * partial_x beta_j, for x = {x0, x1} in R^2
        for i in range(3): 
            for j in range(3):
                rs.append(self.elements[:, i])
                cs.append(self.elements[:, j])
                Zx.append(self.area / 3.0 * self.basis_grad[:, j, 0]) # x0
                Zy.append(self.area / 3.0 * self.basis_grad[:, j, 1]) # x1
        rs, cs = np.concatenate(rs), np.concatenate(cs)
        Zx, Zy = np.concatenate(Zx), np.concatenate(Zy)
        Gx = csr_matrix((Zx, (rs, cs)), shape=(self.N, self.N))
        Gy = csr_matrix((Zy, (rs, cs)), shape=(self.N, self.N))
        Dx, Dy = (self.M_lump_inv @ Gx).tocoo(), (self.M_lump_inv @ Gy).tocoo()
        
        return Gx, Gy # [shape (N, N) for each dim=1,2 etc]
    
    def compute_relative_grad(self, f):  # f is logpsi so no need to log! again
        # compute nabla log phi via (D_l phi / phi)_i in R^node along l in {x,y}
        ratx, raty = np.zeros((self.N)), np.zeros((self.N)) #for dim l in {x,y} in R^2
        # (D_l)_ij * exp(f_j - f_i) = (D_l)_ij * phi_j / phi_i
        ratx = self.Dx.data * np.exp(f[self.Dx.col] - f[self.Dx.row])
        raty = self.Dy.data * np.exp(f[self.Dy.col] - f[self.Dy.row])
        # sum_j (D_l)_ij * exp(f_j - f_i) for each i, and dim l in {x,y}
        np.add.at(ratx, self.Dx.row, ratx) 
        np.add.at(raty, self.Dy.row, raty)
        return ratx, raty

@dataclass 
class SLStenc0: # 3-branches SL-stencil for approximate ops U
    
    x0 : np.ndarray # traceback characteristic point
    x1 : np.ndarray # landing point node
    w : np.ndarray # weights
    mass : np.ndarray 
    reflected : int = 0
    
    def __post_init__(self):
        pass

    
    
    
    @classmethod
    def build(cls, mesh: Mesh, b, dt, kappa): # construct SL-transition matrix
    
        w_ms = [2/3, 1/12, 1/12, 1/12, 1/12] # see weight def eq. 17 for 2dim
        alpha = np.sqrt(6*2*kappa*dt) # i.e. 6*d
        jumps = np.array([[0,0,[alpha,0],[-alpha,0],[0,alpha],[[0,-alpha]]]])
        
        ys, y_tilds, ws, count = [], [], [], [], 0
        for z, p in zip(jumps, w_ms):
            feet, y_tild = mesh.reflect()
            node, beta = mesh.interp()
            
@dataclass 
class Ekern: # interaction kernel E (implement entire drift b(t,x,rho_t))
    gamma : float = .02
    sigma : float = .3
    block : int = 256
    
    def field(self, mesh, rho, logphi, kappa): return np.zeros(mesh.N) 
    
############################ additional subclass DOM ############################

@dataclass 
class DOMobs1(DOM):
    circles = [((0, 0), 0.3)] # ((x, y), radius)
    rrects = [] # corner-smoothed rectangle, [(x0, x1, y0, y1, rf), ...]
    
    def __post_init__(self):
        self.rrect_data = []
        for x0, x1, y0, y1, rf in self.rounded_rects:
            c = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=float)
            h = np.maximum(0.0, 0.5 * np.array([x1 - x0, y1 - y0]) - rf)
            self.rrect_data.append((c, h, float(rf)))

    def _sdf_gradchi(self, x, dists, grads):
        
        for (xc, yc), r in self.circles: 
            diff = x - np.array([xc,yc])
            norm = linalg.norm(diff, 1) 
            dists += [norm-r]
            grads += [diff / np.maximum(norm, 1e-14)[:, None]]
        
        # 2. C^{1,1} filleted rectangles (exact signed distance + outward normal)
        for c, h, rf in self.rrect_data:
            diff = x - c
            q = np.abs(diff) - h
            q_max = np.maximum(q, 0.0)
            out_dist = np.linalg.norm(q_max, axis=1)
            in_dist = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
            
            dists += [out_dist + in_dist - rf]

            sgn = np.where(diff >= 0.0, 1.0, -1.0)
            ext_mask = out_dist > 1e-14
            
            g = np.zeros((x.shape[0], 2), dtype=float)
            g[ext_mask] = sgn[ext_mask] * q_max[ext_mask] / out_dist[ext_mask, None]
            
            int_mask = ~ext_mask
            closer_x = q[int_mask, 0] >= q[int_mask, 1]
            g[int_mask, 0] = np.where(closer_x, sgn[int_mask, 0], 0.0)
            g[int_mask, 1] = np.where(~closer_x, sgn[int_mask, 1], 0.0)
            grads += [g]
            
        return dists, grads
    
# [how about put this in utils.py?]
make_dom_1 = lambda h_min=0.02, h_max=0.04: build_dom_example(
        bounds=(0.0, 2.0, 0.0, 1.0),
        circles=[((0.95, 0.50), 0.32)],
        rounded_rects=[], h_min=h_min, h_max=h_max
)
make_dom_2 = lambda h_min=0.02, h_max=0.04: build_dom_example(
    bounds=(0.0, 2.8, 0.0, 1.0), 
        circles=[
            ((0.58, 0.26), 0.09),
            ((0.58, 0.74), 0.09),
        ],
        rounded_rects=[
            (0.92, 1.10, -0.15, 0.68, 0.09),
            (1.42, 1.60,  0.32, 1.15, 0.09),
            (1.92, 2.12,  0.24, 0.76, 0.09),
        ], h_min=h_min, h_max=h_max
)
# 1d with boundary [-1, 1] (simple)
make_dom_3 = None

