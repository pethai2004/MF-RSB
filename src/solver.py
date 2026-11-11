
import time
import numpy as np

from fem import *
from utils import *

@dataclass
class Config:

    name_path: str = '../assets/Trial'
    T: float = 1.
    Nt: int = 40 # num temporal grid
    h_min : float = .028
    h_max : float = .04
    kappa: float = .05 # sigma = sqrt(2 * kappa)
    gamma: float = .02 # interaction stength
    sigma: float = .35 # not diffusion coef, is for Gaussian interaction
    N_iter: int = 30 
    tol: float = 1e-5
    theta: float = .65 # damping 
    splitting: str = "strang" # or lie, yoshida ? 
    mc_samples : int = 15
    stencil : str = '5p' # [3p, 5p, 9p, or mc] # [TODO: implement this]
    seed: int = 555

def solve_mfrsb(mesh: Mesh, E: Ekern, SL : StencilOps, c: Config, 
                marginals_func: Callable, logger: Experiment): 
    '''Main Solver'''
    np.random.seed(c.seed)
    
    dt = c.T/c.Nt 
    mu0, muT = marginals_func(mesh.nodes, mesh.mass) 
    lgmu0, lgmuT = np.log(mu0), np.log(muT) 
    rho = np.full((c.Nt+1, mesh.N), 1/mesh.mass.sum()) # uniform
    f, h = np.zeros_like(rho), np.zeros_like(rho) # f=log(phi), g=log(phi_hat)
    
    start_time = time.perf_counter()
    prev_diff = None 
    
    for n in range(1, c.N_iter+1):
        
        inner_time_start = time.perf_counter()
        
        logW, b, r, = E.field(mesh, rho, f, c.kappa)
        ops: list[StencilOps] = [SL.build(mesh, b[k], dt, c.kappa, c.splitting, c.mc_samples, c.seed)
                                    for k in range(c.Nt)]
        
        fn, hn = np.empty_like(f), np.empty_like(h) # varphi and varphi_hat (log)
        fn[-1] = f[-1]

        for k in range(c.Nt-1, -1,-1): # TODO: try other solves?
            if c.splitting == 'strang':
                fn[k] = .5*dt*r[k] + \
                        ops[k].feyn_bwd(fn[k+1] + \
                        .5*dt*r[k+1])
            elif c.splitting == 'lie':
                fn[k] = ops[k].feyn_bwd(fn[k+1] + dt*r[k+1]) 
                
        hn[0] = lgmu0 - logW[0] - fn[0]
        
        for k, op in enumerate(ops):
            if c.splitting == 'strang':
                hn[k+1] = .5*dt*r[k+1] + \
                        op.feyn_fwd(hn[k] + \
                        .5*dt*r[k])
            elif c.splitting == 'lie':
                hn[k+1]= dt * r[k+1] + op.feyn_fwd(hn[k])
                
        fn[-1] = lgmuT - logW[-1] - hn[-1]
        
        logrho = logW + fn + hn
        logz =  logsumexp(logrho + np.log(mesh.mass)[None, :], axis=1, keepdims=True)
        rho_half = np.exp(logrho - logz)
        rho_next = (1. - c.theta) * rho + c.theta * rho_half
        
        #### DONE ###
        cpu_iter = time.perf_counter() - inner_time_start
    

        rho_diff = float(np.max(np.abs(rho_next[1:-1] - rho_next[1:-1])))
        q_val = float(rho_diff / prev_diff) if (prev_diff is not None and prev_diff > 1e-15) else np.nan
        b_norm = np.linalg.norm(b, axis=-1)
        
        b_norm = np.linalg.norm(b, axis=-1) 
        records = dict(
            n = n, drho_inf = rho_diff, w2T = compute_w2_marginal(mesh.nodes, mesh.mass, rho_next[-1], muT)
            mu0_l1 = float(mesh.mass @ np.abs(rho_next[0] - mu0)),
            muT_l1 = float(mesh.mass @ np.abs(rho_next[-1] - muT)),
            logz_max = float(np.max(np.abs(logz))),
            err_mass = float(np.max(np.abs((rho_next @ mesh.mass) - 1.0))),
            pe_h = float(np.max(b_norm * mesh.node_h[None, :]) / (2.0 * c.kappa)),
            cfl = float(np.max(b_norm / mesh.node_h[None, :]) * dt),
            q_n = q_val,
            row_err = max(o.row_err for o in ops),
            nnz_row = float(np.mean([o.P.nnz/mesh.N for o in ops])),
            cpu_time = time.pref_counter() - start_time
        )
        prev_diff = rho_diff 
        rho, f, h = rho_next, fn, hn
        
        if rho_diff <= c.tol: break ### converged! 

    # finalize
    for k in range(c.Nt-1, -1, -1):
        f[k] = .5 * dt * r[k] + ops[k].feyn_bwd(
            f[k+1] +.5 * dt * r[k+1]
        )
    
    # alpha^* = 2*kappa*nabla(varphi) where varphi = nabla(phi) / phi
    optimal_alpha = np.sqrt(2. * c.kappa) * np.array(
       [ mesh.compute_relative_grad(f[k]) for k in range(c.Nt+1)]
    )
    optimal_alpha[-1] = optimal_alpha[-2].copy()
    
    final_energy = free_energy(mesh, E, rho, c.kappa)
    
    endtime = time.perf_counter() - start_time
    
    return {'rho': rho, 'f': f, 'h': h, 'b': b, 'mu0' : mu0, 'muT': muT,
            'opt_alpha': optimal_alpha, 'dt': dt, 'T': c.T, 
            'kappa': c.kappa, 'free_energy' : final_energy, 'endtime': endtime}


def sim_traj(mesh: Mesh, sol, substeps=4, seed=555):
    pass 


def sample_density(mesh: Mesh, rho, Np, rng): # X: [n_time, n_nodes, *ndim]
    ## 
    pass 

