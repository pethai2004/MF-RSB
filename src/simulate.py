
import time, os, argparse

from core import *
from utils import *

T = 1.


# T0 (1d-proof of concept) on [-1, 1], with Gaussian marginal different in means and variance. 






# T1 : [Illustrative test] 2-dim, with 3x3 snapshots, initial and endpoint marginals b
# eing mixture of Gaussian and uniform over rectangular box










# T2 : [Ops Splitting : Lie-Trotter, Strang (mine), and Yosida (possibly failed, include it anyway)]


def run_trial_2():
    pass








# T3 : [Convergence with 1: meshX -> 0, 2: mesht->0, and simultaneously (meshX, mesht)->0]


def run_trial_3():
    time_seqs = [T/10, T/20, T/40, T/80, T/160]
    h_max = [0.16, 0.08, 0.04, 0.02, 0.01]

    for delT, delX in zip(time_seqs, h_max):
        # use same solver but loop only once?? 
        pass 







# T4 : Asymptotic Reduction for b=0, sigma->0, E=0



def run_trial_4():
    pass







# T5 : 