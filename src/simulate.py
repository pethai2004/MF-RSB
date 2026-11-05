
import time, os, argparse

from core import *
from utils import *

def run_circle_obs_fig(config: Config, save_result=False, num_particle=30, n_snapshot=6):
    
    logger = Experiment(config.name_path)

    bound=(0, 2, 0, 1)
    circles= [((.9, .5), .4)]
    rounded_rects=[]
    nodes0, eles0 = build_dom_example(
        bound, circles, rounded_rects,
        h_min=config.h_min, h_max=config.h_max
    )

    dom = DOMobs1(3, bound, circles, rounded_rects)
    mesh = Mesh(nodes0, eles0, dom)
    E0 = Ekern(nodes0, gamma=config.gamma, sigma=config.sigma)

    def _make_marg(nodes, mass):
        x, y = nodes.T
        
        g1 = np.exp(-((x - 0.35) ** 2 + (y - 0.30) ** 2) / (2.0 * 0.08 ** 2))
        g2 = np.exp(-((x - 0.35) ** 2 + (y - 0.70) ** 2) / (2.0 * 0.08 ** 2))
        mu0= 0.5 * (g1 + g2) + 1e-12
        
        dx = np.maximum(0., np.maximum(1.5-x, x-1.8))
        dy = np.maximum(0., np.maximum(.35-y, y-.85))
        muT = np.exp(-(dx**2 + dy**2) / (2.*.05**2)) + 1e-12
        
        return mu0 / np.sum(mass*mu0), muT / np.sum(mass*muT)
    
    
    solution = solve_mfrsb(mesh, E0, SLStenc0, config, _make_marg, logger) 
    
    if save_result:
        box = [(1.45, 1.82, 0.20, 0.80)]
        trajs, max_penetration = path_sim(mesh, solution, num_particle, 3, config.seed)
        save_snapshot2d(mesh, solution, 
                        os.path.join(config.name_path, "snapshot.png"),
                        trajs, box, n_snapshot, "MFRSB with Circular Obs")
        logger.save_sol("solution_circle", mesh, solution, trajs, box)
        
    return solution, logger

# # Illustrative example 2:

# def run_multi_obs_fig():
#     pass 


# (Study Main 1) Convergence for temporal/spatial mesh -> 0






# (Study Main 2)






# (Study Main 3)









# (Study Main 4)