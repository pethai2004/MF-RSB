
from core import Config, _FEM
from solver import alg_step

class PDE: # subclass "forward, backward" to define specific PDEs

    rho = None
    vphi = None #varphi
    mphi = None #varphi_hat
    
    def __init__(self, config: Config):
        self._FEM = _FEM()
    
    def _forward(self): pass
    
    def _backward(self): pass 


def path_sim(pde: PDE, config: Config): # forward Euler-Maruyama scheme
    pass