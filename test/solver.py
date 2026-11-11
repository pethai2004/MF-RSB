from dataclasses import dataclass
from typing import Callable
import time
import numpy as np
from scipy.special import logsumexp

from fem import *
from utils import *


@dataclass
class Config:
    name_path: str = "results"
    T: float = 1.0
    Nt: int = 40
    h_min: float = 0.028
    h_max: float = 0.04
    kappa: float = 0.12
    gamma: float = 0.02
    sigma: float = 0.35
    N_iter: int = 60
    tol: float = 1e-5
    theta: float = 0.65
    splitting: str = "strang"
    mc_samples: int = 32
    stencil: str = "gh9"
    seed: int = 555
    verbose: bool = True


def solve_mfrsb(
    mesh: Mesh,
    E: Ekern,
    SL: StencilOps,
    c: Config,
    marginals_func: Callable,
    logger: Experiment = None,
):
    """Main MFRSB Fixed-Point Solver with inline forward/backward sweeps."""
    np.random.seed(c.seed)

    dt = c.T / c.Nt
    mu0, muT = marginals_func(mesh.nodes, mesh.mass)
    lgmu0, lgmuT = np.log(mu0), np.log(muT)
    rho = np.full((c.Nt + 1, mesh.N), 1.0 / mesh.mass.sum())
    rho_half_prev = rho.copy()
    f, h = np.zeros_like(rho), np.zeros_like(rho)

    start_time = time.perf_counter()
    prev_l1 = None
    prev_raw_l1 = None
    status = "max_iter"

    for n in range(1, c.N_iter + 1):
        logW, b, r = E.field(mesh, rho, f, c.kappa)
        ops = [
            SL.build(mesh, b[k], dt, c.kappa, c.stencil, c.mc_samples, c.seed)
            for k in range(c.Nt)
        ]

        fn, hn = np.empty_like(f), np.empty_like(h)
        fn[-1] = f[-1]

        # Inline backward sweep
        for k in range(c.Nt - 1, -1, -1):
            if c.splitting == "strang":
                fn[k] = 0.5 * dt * r[k] + ops[k].feyn_bwd(fn[k + 1] + 0.5 * dt * r[k + 1])
            elif c.splitting == "lie":
                fn[k] = ops[k].feyn_bwd(fn[k + 1] + dt * r[k + 1])

        hn[0] = lgmu0 - logW[0] - fn[0]

        # Inline forward sweep
        for k, op in enumerate(ops):
            if c.splitting == "strang":
                hn[k + 1] = 0.5 * dt * r[k + 1] + op.feyn_fwd(hn[k] + 0.5 * dt * r[k])
            elif c.splitting == "lie":
                hn[k + 1] = dt * r[k + 1] + op.feyn_fwd(hn[k])

        terminal_target = lgmuT - logW[-1] - hn[-1]
        terminal_res = float(np.max(np.abs(terminal_target - fn[-1])))
        fn[-1] = terminal_target

        logrho = logW + fn + hn
        logz = logsumexp(logrho + np.log(mesh.mass)[None, :], axis=1, keepdims=True)
        rho_half = np.exp(logrho - logz)
        rho_next = (1.0 - c.theta) * rho + c.theta * rho_half

        # 1. Damped interior density residual (for Table 1)
        drho_l1 = float(np.max(np.abs(rho_next[1:-1] - rho[1:-1]) @ mesh.mass))
        drho_inf = float(np.max(np.abs(rho_next[1:-1] - rho[1:-1])))

        # 2. Undamped map increment ||T(rho^{n-1}) - T(rho^{n-2})||_L1 (unmasks true spectral radius)
        drho_raw_l1 = float(np.max(np.abs(rho_half[1:-1] - rho_half_prev[1:-1]) @ mesh.mass))

        q_l1 = (
            float(drho_l1 / prev_l1)
            if (prev_l1 is not None and prev_l1 > 1e-15)
            else np.nan
        )
        q_raw = (
            float(drho_raw_l1 / prev_raw_l1)
            if (prev_raw_l1 is not None and prev_raw_l1 > 1e-15)
            else np.nan
        )

        # Pair-density unnormalized L1 defect: max_t | int exp(logW + f + h) dx - 1 |
        pair_l1 = float(np.max(np.abs((np.exp(logrho) @ mesh.mass) - 1.0)))

        b_norm = np.linalg.norm(b, axis=-1)
        cpu_time = time.perf_counter() - start_time

        record = dict(
            n=n,
            drho_l1=drho_l1,
            drho_raw_l1=drho_raw_l1,
            drho_inf=drho_inf,
            q_l1=q_l1,
            q_raw=q_raw,
            terminal_res=terminal_res,
            pair_l1=pair_l1,
            mu0_l1=float(mesh.mass @ np.abs(rho_next[0] - mu0)),
            muT_l1=float(mesh.mass @ np.abs(rho_next[-1] - muT)),
            w2_T=compute_w2_marginal(mesh.nodes, mesh.mass, rho_next[-1], muT),
            logz_max=float(np.max(np.abs(logz))),
            err_mass=float(np.max(np.abs((rho_next @ mesh.mass) - 1.0))),
            pe_h=float(np.max(b_norm * mesh.node_h[None, :]) / (2.0 * c.kappa)),
            cfl=float(np.max(b_norm / mesh.node_h[None, :]) * dt),
            row_err=float(max(o.row_err for o in ops)),
            adj_err=float(ops[0].adjoint_error()),
            nnz_row=float(np.mean([o.P.nnz / mesh.N for o in ops])),
            cpu_time=cpu_time,
        )

        if logger is not None:
            logger.log_step(record, verbose=c.verbose)

        prev_l1 = drho_l1
        prev_raw_l1 = drho_raw_l1
        rho_half_prev = rho_half.copy()
        rho, f, h = rho_next, fn, hn

        if drho_raw_l1 <= c.tol and drho_l1 <= c.tol:
            status = "converged"
            break

    # Finalize backward potential for optimal drift
    for k in range(c.Nt - 1, -1, -1):
        if c.splitting == "strang":
            f[k] = 0.5 * dt * r[k] + ops[k].feyn_bwd(f[k + 1] + 0.5 * dt * r[k + 1])
        elif c.splitting == "lie":
            f[k] = ops[k].feyn_bwd(f[k + 1] + dt * r[k + 1])

    optimal_alpha = np.sqrt(2.0 * c.kappa) * np.array(
        [mesh.compute_relative_grad(f[k]) for k in range(c.Nt + 1)]
    )
    optimal_alpha[-1] = optimal_alpha[-2].copy()

    final_energy = free_energy(mesh, E, rho, c.kappa)
    endtime = time.perf_counter() - start_time

    return dict(
        status=status,
        history=logger.history if logger is not None else [],
        rho=rho,
        f=f,
        h=h,
        b=b,
        mu0=mu0,
        muT=muT,
        opt_alpha=optimal_alpha,
        dt=dt,
        T=c.T,
        kappa=c.kappa,
        free_energy=final_energy,
        endtime=endtime,
    )