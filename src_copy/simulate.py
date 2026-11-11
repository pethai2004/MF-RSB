
import time, os, argparse

from fem import *
from utils import *

ASDIR = '../assets/'


### 1-dim illusttrative case, with disconnected domain and blockage between two marginals (no solution)

# def run_mz_obs_fig(config: Config, save_result=True, num_particle=200, n_snapshot=20):
    
#     logger = Experiment(config.name_path)

#     bound = (0.0, 3.0, 0.0, 1.2)
    
#     circles = [
#         ((0.60, 0.32), 0.10),
#         ((0.60, 0.88), 0.10)
#     ]

#     rounded_rects = [
#         (0.98, 1.24, -0.20, 0.82, 0.08),
#         (1.52, 1.78,  0.38, 1.40, 0.08),
#         (2.06, 2.32,  0.35, 0.85, 0.08)
#     ]
#     nodes0, eles0 = build_dom_example(
#         bound, circles, rounded_rects,
#         h_min=config.h_min, h_max=config.h_max
#     )

#     dom = DOMobs1(3, bound, circles, rounded_rects)
#     mesh = Mesh(nodes0, eles0, dom)
#     E0 = Ekern(nodes0, gamma=config.gamma, sigma=config.sigma)

#     def _make_marg(nodes, mass):
#         x, y = nodes[:, 0], nodes[:, 1]
            
#         # Source marginal mu_0: Tri-modal Gaussian cluster near the entrance
#         s0 = 0.075
#         g1 = np.exp(-((x - 0.28)**2 + (y - 0.28)**2) / (2.0 * s0**2))
#         g2 = np.exp(-((x - 0.28)**2 + (y - 0.60)**2) / (2.0 * s0**2))
#         g3 = np.exp(-((x - 0.28)**2 + (y - 0.92)**2) / (2.0 * s0**2))
#         mu0 = (g1 + g2 + g3) / 3.0 + 1e-12
#         mu0 /= np.sum(mass * mu0)
        
#         # Target marginal mu_T: Bi-modal split target past the final baffle
#         sT = 0.055
#         dx = np.maximum(0.0, np.maximum(2.55 - x, x - 2.85))
#         dy_bot = np.maximum(0.0, np.maximum(0.12 - y, y - 0.42))
#         dy_top = np.maximum(0.0, np.maximum(0.78 - y, y - 1.08))
        
#         muT = 0.5 * (
#             np.exp(-(dx**2 + dy_bot**2) / (2.0 * sT**2)) +
#             np.exp(-(dx**2 + dy_top**2) / (2.0 * sT**2))
#         ) + 1e-12
#         muT /= np.sum(mass * muT)
            
#         return mu0, muT

    
#     solution = solve_mfrsb(mesh, E0, SLStenc0, config, _make_marg, logger) 
    
#     if save_result:
#         box = None
#         trajs, max_penetration = path_sim(mesh, solution, num_particle, 3, config.seed)
#         save_snapshot2d(mesh, solution, 
#                         os.path.join(config.name_path, "snapshot.png"),
#                         trajs, box, n_snapshot, "MFRSB with Mz Obs")
#         logger.save_sol("solution_mz", mesh, solution, trajs, box)
        
#     return solution, logger


########## fig 1 (3by2): snapshot and table 1 (3by2)

    
    
########## graph 1 (2-subplots). fixed-point, x-axis (n) against |Delta rho^n| y-axis for difference 
# (a) ops split/linear solve (b) stencils



########## table 2: for inc t / inc x / (inc t, inc x) ->0 








########## 1dim case (2by2)



########## fig 3 (2d-plot, 3by2) illustrate 1 dimensional transport plan for 
# kappa= 0, 0.5, 1., gamma=0, 0,5, 1. (density+particle)
# graph: x-axis n teration  (for each params) against (a) FB res (b) log Z, (c) pair L^1, (d) term res etc

