class PDE:

    rho = None
    vphi = None #varphi
    mphi = None #varphi_hat
    
    def _forward(self): pass
    
    def _backward(self): pass 


def path_sim(pde: PDE, config: Config):
    pass