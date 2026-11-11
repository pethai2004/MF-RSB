from dataclasses import dataclass, field
from typing import List, Tuple, Callable
import time 
from concurrent.futures import ThreadPoolExecutor
## [TODO : try adding parallization]

import numpy as np 
from numpy import linalg, ndarray
import scipy
from scipy.sparse import csr_matrix, diags
from scipy.special import logsumexp
import matplotlib

from src.utils import *

## Include: core FEM, reflection, interpolation, Mesh, DOM.

dim = 2 # [TODO : implement dim = 1 and 3 ?]
tri = 3 #  [TODO : remove this global??]


@dataclass
class DOM: # (maybe [TODO]: other outer boundary such as hexagon/curve)
    n_resolves : int = 3
    bounds = (-2, 2, -1, 1) # [x0, x1, y0, y1]
    
    def _sdf_gradchi(self, x, dists, glists) : return dists, glists
    
    def sdf_gradchi(self, x: ndarray): 
        # compute SDF and its gradients, x: [num, dim=2] 
        assert x.ndim==dim
        dists, grads = [], []
        dists += [x[:,0]-self.bounds[0], self.bounds[1]-x[:,0], # l,r,b,t
                x[:,1]-self.bounds[2], self.bounds[3]-x[:,1]]
        wall_grads = np.array([[1,0], [-1,0], [0,1],[0,-1]], float)
        grads = [np.broadcast_to(wall_grads[i], (x.shape[0], dim)).copy() 
                    for i in range(4)]
        
        dists, grads = self._sdf_gradchi(x, dists, grads) # downstream impl
        dists = np.column_stack(dists) # [N=x.shape[0], num "obs pieces"] 
        #e.g. 3circles with 2 rects will give 1+1+4+4 final dim
        minidx = dists.argmin(1) # closest piece obs
        chi = dists[np.arange(x.shape[0]), minidx]
        
        grad_chi = np.zeros((x.shape[0], dim)) 
        for j, G in enumerate(grads):
            mask = (minidx == j)
            if np.any(mask): grad_chi[mask] = G[mask]
        
        return chi, grad_chi 
    
    def reflect(self, x: ndarray): # [num_point, dim=2]
        # compute Skorakhod reflects (eq 17): for chi(x) < 0 : y_hat = y-2*chi(x)*grad_chi
        x_hat = x.copy()
        for _ in range(self.n_resolves):
            chi, grad_chi = self.sdf_gradchi(x_hat)
            mask = (chi<0.) # (out)
            if not np.any(mask) : break # no penetration
            x_hat[mask] -= 2.*chi[mask, None]*grad_chi[mask]

        return x_hat, float(np.maximum(0., -chi).max())

def _reflect(cls, x0, x1, max_bounces=15): #[TODO] imple func, and make dispatch to this
    # this is similar to `DOM.reflect` but will exactly handle the particle tunnelling
    # for large step size which `DOM.reflect` completely cannot detect. 
    # the path will be traced from start to account for this, unlike its predecesor.
    
    pass
@dataclass
class Mesh:
    nodes : np.ndarray # [N,dim]
    elements : np.ndarray # [M, tri=2/3/4]
    dom : DOM
    
    def __post_init__(self):

        self.nodes = np.asarray(self.nodes)
        self.elements = np.asarray(self.elements, int)
        
        # (TODO) : how about assert and assume already preprocessed
        valid_idx, invm = np.unique(self.elements, return_inverse=True)
        assert not len(valid_idx) < len(self.nodes)
        
        self.N, self.M = len(self.nodes), len(self.elements) 
        self.edges = np.sort(np.concatenate(  # sorting is requred to correctly extract the boundary edges.
            [self.elements[:, [0,1]], self.elements[:, [1,2]], self.elements[:, [2,0]]]
        , 0), 1)
        unqiue_e, count = np.unique(self.edges, axis=0,return_counts=True)
        self.bc_edges = unqiue_e[count == 1]
        self.bc_idx = np.unique(self.bc_edges)
        self.bc = self.nodes[self.bc_idx]
        
        x = self.nodes[self.elements] # [M, tri=3, dim]
        # [N, dim, dim] : each [[dx/dxi dx/deta], [dy/dxi, dy/deta]]
        J = np.stack((x[:, 1]-x[:,0], x[:, 2]-x[:,0]), -1) 
        detJ = np.linalg.det(J)
        if np.any(np.abs(detJ)<1e-14): raise ValueError("degenerate mesh founded")
        self.area = abs(detJ) / 2
        self.invJ = np.linalg.inv(J)
        # - [j00 + j10, j01 + j11], [j00, j01], and [j10, j11]  
        Nxa, Nxb = self.invJ[:, 0, :], self.invJ[:, 1, :]
        Nxc = -(Nxa + Nxb)
        self.basis_grad = np.stack([Nxc, Nxa, Nxb], 1) # [M, 3, 2]
        
        # i.e. mass = zero(N) and for each loc in range(3): add.at(mass, elements[:,loc], area/3)
        self.mass = np.bincount(self.elements.ravel(), 
                                weights=np.repeat(self.area / 3.0, 3), 
                                minlength=self.N)
        assert np.all(self.mass > 0)
        # [TODO] try using scipy.sparse.diags instead of np.diag.
        self.M_lump, self.M_lump_inv = np.diag(self.mass), np.diag(1. / self.mass)
        
        self.tri = matplotlib.tri.Triangulation(*self.nodes.T, self.elements)
        self.find = self.tri.get_trifinder()
        self.tree = scipy.spatial.cKDTree(self.nodes) # NB search
        
        # precompute D_l = M_lump^-1 * G_l for l={x,y}
        # self.Dx, self.Dy = self.ass_react() 
        rows = np.repeat(self.elements, 3, axis=1).ravel() # [M*3*3]
        cols = np.tile(self.elements, (1,3)).ravel() # [M*3*3]
        w_grad = (self.area[:, None, None] / 3.0) * self.basis_grad
        vx = np.tile(w_grad[:,:,0], (1,3)).ravel() / self.mass[rows] # [M*3*3]
        vy = np.tile(w_grad[:,:,1], (1,3)).ravel() / self.mass[rows] # [M*3*3]
        self.Dx = csr_matrix((vx, (rows, cols)), shape=(self.N, self.N)).tocoo()
        self.Dy = csr_matrix((vy, (rows, cols)), shape=(self.N, self.N)).tocoo()
        
        elem_h = np.sqrt(2. * self.area)
        self.node_h = np.zeros(self.N) # local mesh size
        for loc in range(3): # tri 3
            val = elem_h/3.
            np.add.at(self.node_h, self.elements[:, loc], val)
        self.h_max = float(self.node_h.max())
        
    # def ass_react(self) -> tuple[csr_matrix, csr_matrix]:
    #     # precompute derivaitve metric for non-local reaction, return D_l = M_lump^-1 * G_l for l={x,y}
    #     rs, cs, Zx, Zy = [], [], [], [] # [TODO try use ndim sparse matrix lib]
    #     # for each vertex pair (i,j) in {0,1,2}^2
    #     # compute int_K beta_i * partial_x beta_j dx = areaK/3 * partial_x beta_j, for x = {x0, x1} in R^2
    #     for i in range(3): 
    #         for j in range(3):
    #             rs.append(self.elements[:, i])
    #             cs.append(self.elements[:, j])
    #             Zx.append(self.area / 3.0 * self.basis_grad[:, j, 0]) # x0
    #             Zy.append(self.area / 3.0 * self.basis_grad[:, j, 1]) # x1
    #     rs, cs = np.concatenate(rs), np.concatenate(cs)
    #     Zx, Zy = np.concatenate(Zx), np.concatenate(Zy)
    #     Gx = csr_matrix((Zx, (rs, cs)), shape=(self.N, self.N))
    #     Gy = csr_matrix((Zy, (rs, cs)), shape=(self.N, self.N))
    #     Dx, Dy = (self.M_lump_inv @ Gx).tocoo(), (self.M_lump_inv @ Gy).tocoo()
        
    #     return Gx, Gy # [shape (N, N) for each dim=1,2 etc]
    
    def reflect(self, x: ndarray): return self.dom.reflect(x)
    
    def interp(self, x): 
        # get P1 bais weights (barycentric coord), x: [num, dim=2]
        e = self.find(x[:, 0], x[:, 1]) # [num,]
        isout = (e == -1) # out (chi<0)
        isin = np.logical_not(isout)

        enodes = np.zeros((x.shape[0], 3), dtype=int) # [num, 3]
        lambdas = np.zeros((x.shape[0], 3), dtype=float) # [num, 3], barycentric coord: [l0, l1, l2]
        
        if np.any(isin): # [lam0, lam1] = Jinv * (x - x0) and lam2 = 1 - lam0 - lam1
            e_in = e[isin]
            diff = x[isin] - self.nodes[self.elements[e_in, 0]] # [num_in, dim=2]
            lamb1_2 = np.einsum("abc, ac->ab", self.invJ[e_in], diff)
            lamb0 = 1.0 - lamb1_2.sum(1)
            
            lll = np.column_stack([lamb0, lamb1_2[:,0], lamb1_2[:,1]])
            # should we clip and normalize? i.e. lll.clip(0, 1), 
            enodes[isin] = self.elements[e_in]
            lambdas[isin] = lll
        
        if np.any(isout):
            _, nearest_idx = self.tree.query(x[isout]) # [num_out,]
            enodes[isout] = nearest_idx[:, None]
            lambdas[isout, 0] =1. 
        
        return enodes, lambdas 
            
    def compute_relative_grad(self, f):  # f is logpsi so no need to log! again
        # compute nabla log phi via (D_l phi / phi)_i in R^node along l in {x,y}
        ratx, raty = np.zeros((self.N)), np.zeros((self.N)) #for dim l in {x,y} in R^2
        # (D_l)_ij * exp(f_j - f_i) = (D_l)_ij * phi_j / phi_i
        xs = self.Dx.data * np.exp(f[self.Dx.col] - f[self.Dx.row])
        ys = self.Dy.data * np.exp(f[self.Dy.col] - f[self.Dy.row])
        # sum_j (D_l)_ij * exp(f_j - f_i) for each i, and dim l in {x,y}
        np.add.at(ratx, self.Dx.row, xs) 
        np.add.at(raty, self.Dy.row, ys)
        return np.column_stack([ratx, raty]) # [N, 2]

# @dataclass
# class Stencil:
    
#     x0 : np.ndarray 
#     x1 : np.ndarray 
#     w : np.ndarray # weights (log): quadrature * barycentric weight
#     mass : np.ndarray 
#     pent : int = 0

#     @classmethod
#     def build(cls, mesh: Mesh, b, dt, kappa):
#         pass

@dataclass ## Create core class instead 
class SLStenc0: # 3-branches SL-stencil for approximate ops U
    
    x0 : np.ndarray 
    x1 : np.ndarray 
    w : np.ndarray # weights (log): quadrature * barycentric weight
    mass : np.ndarray 
    pent : int = 0
    
    def __post_init__(self):
        keep=(self.w > 1e-14) # filter out small weights
        self.x0, self.x1, self.w = self.x0[keep], self.x1[keep], self.w[keep]
        self.lgw = np.log(self.w)
        self.lgm= np.log(self.mass)
        self.n = len(self.mass)
        row = np.bincount(self.x0, self.w, self.n)
        self.row_err = float(np.max(np.abs(row-1.))) #

        self.nnz_ratio = float(len(self.w)/float(self.n**2))
    @classmethod
    def build(cls, mesh: Mesh, b, dt, kappa): # construct SL-transition matrix
    
        w_ms = [2/3, 1/12, 1/12, 1/12, 1/12] # see weight def eq. 17 for 2dim
        alpha = np.sqrt(6*2*kappa*dt) # i.e. 6*d
        jumps = np.array([[0,0],[alpha,0],[-alpha,0],[0,alpha],[0,-alpha]])
        
        nidx = np.repeat(np.arange(mesh.N), 3) # 
        ys, y_tilds, ws, count = [], [], [], 0
        base = mesh.nodes +dt*b
        for z, p in zip(jumps, w_ms):
            feet, pent = mesh.reflect(base+z)
            count = max(count, pent)
            vert, beta = mesh.interp(feet)
            ys += [nidx]
            y_tilds += [vert.ravel()]
            ws += [(p*beta).ravel()]
        return cls(*(np.concatenate(X) for X in [ys, y_tilds, ws]), mesh.mass, count)
    
    
    
    
    
    # [TODO] put check assertion with direct matrix-vector product
    def feyn_bwd(self, logv): # sum_{e: i_e=i} w_e * varphi_{j_e} 
        out = np.full(self.n, -np.inf)
        # for e, source in enumerate(self.x0): # try more efficient: 
        #     out[source] = np.log(np.exp(out[source] + np.exp(
        #             self.lgw[e] + logv[self.x1[e]])
        # ))
        np.logaddexp.at(out, self.x0, self.lgw + logv[self.x1])
        return out
    
    def feyn_fwd(self, logv): # i.e. M^-1 @ Pi^T @ M^-1
        # [TODO: try using -infty instead of -100]
        out = np.full(self.n, -100.) ### just replace source with targ and so on for foward sweep
        np.logaddexp.at(out, self.x1, self.lgw +logv[self.x0] +self.lgm[self.x0]) 
        return  out - self.lgm
    
@dataclass 
class Ekern: # interaction kernel E (implement entire drift b(t,x,rho_t))
    nodes : ndarray
    gamma : float = .02
    sigma : float = .3
    # block : int = 256
    
    def __post_init__(self):
        # quite big computation, [TODO] : use offload in batch
        diff = self.nodes[:, None, :] - self.nodes[None, :, :] # [N, N, dim] 
        self.E = self.gamma * np.exp(
            -.5 * np.sum(diff**2,-1)/self.sigma**2) # is [N, N] symmetricj
        self.dE = - (diff/self.sigma**2) *self.E[:,:, None] # [N, N, dim=2]
        
    def field(self, mesh: Mesh, rho, logphi, kappa): 
        
        rho_mass = rho * mesh.mass[None,:] # rho, logphi: [T, N] and mass : [N,]
        logW = - rho_mass @ self.E.T / kappa # [T, N] x [N, N] -> [T, N]
        b = - np.einsum('ijd, tj->tid', self.dE, rho_mass) # [T, N, 2], contract over source node j
        rel_grad = np.array([ # [T, N, 2]
            mesh.compute_relative_grad(lp) for lp in logphi # each grad -> (N, 2)
        ])
        r = -np.einsum('ijd, tj,tjd->ti', self.dE, rho_mass, rel_grad)
        # try clipping, but stay like this [TODO]
        r = np.clip(r, -20, 20)
        return logW, b, r


############################ additional subclass DOM ############################

@dataclass 
class DOMobs1(DOM):
    circles: List = field(default_factory=list)  # ((x, y), radius)
    rrects: List = field(default_factory=list)  # corner-smoothed rectangle, [(x0, x1, y0, y1, rf), ...]

    def __post_init__(self):
        self.rrect_data = []
        for x0, x1, y0, y1, rf in self.rrects:
            c = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=float)
            h = np.maximum(0.0, 0.5 * np.array([x1 - x0, y1 - y0]) - rf)
            self.rrect_data.append((c, h, float(rf)))

    def _sdf_gradchi(self, x, dists, grads):
        
        for (xc, yc), r in self.circles: 
            diff = x - np.array([xc,yc])
            norm = linalg.norm(diff, axis=1) 
            dists += [norm-r]
            grads += [diff / np.maximum(norm, 1e-14)[:, None]]
        
        # 2. C^{1,1} filleted rectangles (exact signed distance + outward normal)
        for c, h, rf in self.rrect_data:
            diff = x - c
            q = np.abs(diff) - h
            q_max = np.maximum(q, 0.0)
            out_dist = linalg.norm(q_max, axis=1)
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
    
