import argparse
import time
from pathlib import Path
from dataclasses import replace
import numpy as np

from fem import DOMobs1, Mesh, Ekern, SLStenc0, quadrature, Mesh1D, SLStencil1D
from solver import Config, solve_mfrsb
from utils import *
from fem import *

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIG = ROOT / "figures"
for p in (OUT, FIG):
    p.mkdir(exist_ok=True)


def make_2d_mesh(h=0.06):
    bounds = (0.0, 3.0, 0.0, 1.2)
    circles = [((0.60, 0.32), 0.10), ((0.60, 0.88), 0.10)]
    rects = [
        (0.98, 1.24, -0.20, 0.82, 0.08),
        (1.52, 1.78, 0.38, 1.40, 0.08),
        (2.06, 2.32, 0.35, 0.85, 0.08),
    ]
    nodes, els = build_dom_example(bounds, circles, rects, h_min=0.7 * h, h_max=h)
    return Mesh(nodes, els, DOMobs1(8, bounds, circles, rects))


def make_marginals(nodes, mass):
    x, y = nodes.T
    s0, sT, floor = 0.075, 0.055, 1e-12
    g = sum(
        np.exp(-((x - 0.28) ** 2 + (y - yc) ** 2) / (2.0 * s0 * s0))
        for yc in [0.28, 0.60, 0.92]
    ) / 3.0

    dx = np.maximum(0.0, np.maximum(2.55 - x, x - 2.85))
    lo = np.maximum(0.0, np.maximum(0.12 - y, y - 0.42))
    hi = np.maximum(0.0, np.maximum(0.78 - y, y - 1.08))
    t = 0.5 * np.exp(-(dx * dx + lo * lo) / (2.0 * sT * sT)) + 0.5 * np.exp(
        -(dx * dx + hi * hi) / (2.0 * sT * sT)
    )

    g += floor
    t += floor
    return g / (mass @ g), t / (mass @ t)


def base_config():
    return Config(
        name_path=str(OUT),
        T=1.0,
        Nt=50,
        h_min=0.042,
        h_max=0.06,
        kappa=0.12,
        gamma=0.02,
        sigma=0.35,
        N_iter=40,
        tol=1e-5,
        theta=0.65,
        splitting="strang",
        stencil="gh9",
        verbose=True,
    )


# ==============================================================================
# TRIAL 0: 2D Maze Illustrative Evolution & Table 1
# ==============================================================================
def run_trial_0():
    print("\n================ [TRIAL 0: 2D MAZE & TABLE 1] ================")
    c = base_config()
    mesh = make_2d_mesh(c.h_max)
    E = Ekern(mesh.nodes, c.gamma, c.sigma)
    logger = Experiment(str(OUT))

    sol = solve_mfrsb(mesh, E, SLStenc0, c, make_marginals, logger)
    X, pdiag = path_sim(mesh, sol, Np=1600, substeps=8, seed=991)

    save_snapshot2d(mesh, sol, FIG / "fig1_density.png")
    save_snapshot2d(mesh, sol, FIG / "fig1_particles.png", X_traj=X)

    last = sol["history"][-1]
    print(
        f"\n[Draft Text Fill-In]: Converged in {last['n']} iterations, "
        f"L1 residual = {last['drho_l1']:.3e} (raw L1 = {last['drho_raw_l1']:.3e}), "
        f"CPU time = {sol['endtime']:.2f} s, Particle SW2 = {pdiag['sw2']:.3e}"
    )

    print("\n--- TABLE 1: Fixed-Point Iterates ---")
    print(
        "Iter (n) || ||drho^n||_L1 || ||drho^n||/||drho^{n-1}|| || "
        "||rho_0^n - mu_init||_L1 || ||rho_T^n - mu_targ||_L1 || "
        "W_2(rho_T^n, mu_targ) || max_k |log Z_k|"
    )
    target_iters = [1, 3, 5, 10, 15, 20, 25]
    for r in sol["history"]:
        if r["n"] in target_iters or r is sol["history"][-1]:
            q_str = f"{r['q_l1']:.3e}" if np.isfinite(r["q_l1"]) else "   ---   "
            print(
                f"{r['n']:8d} || {r['drho_l1']:13.3e} || {q_str:25s} || "
                f"{r['mu0_l1']:24.3e} || {r['muT_l1']:24.3e} || "
                f"{r['w2_T']:21.3e} || {r['logz_max']:15.3e}"
            )


# ==============================================================================
# TRIAL 1: Frozen-Field Linear Convergence & Table 2 (Eq. 1)
# ==============================================================================
def solve_frozen(mesh, nt, kappa=0.12, splitting="strang", stencil="gh9", seed=555):
    dt = 1.0 / nt
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    b = np.column_stack([np.cos(y), np.sin(x)])
    # Time-modulated reaction so Lie O(dt) and Strang O(dt^2) separate cleanly
    t_grid = np.linspace(0.0, 1.0, nt + 1)
    r_spatial = np.sin(x) * np.cos(y)
    phi_T = 1.0 + np.cos(y) * np.cos(x)
    f = np.log(phi_T).copy()

    op = SLStenc0.build(mesh, b, dt, kappa, stencil, 32, seed)
    for k in range(nt - 1, -1, -1):
        rk = (1.0 + 0.5 * np.cos(np.pi * t_grid[k])) * r_spatial
        rk1 = (1.0 + 0.5 * np.cos(np.pi * t_grid[k + 1])) * r_spatial
        if splitting == "strang":
            f = 0.5 * dt * rk + op.feyn_bwd(f + 0.5 * dt * rk1)
        else:
            f = op.feyn_bwd(f + dt * rk1)

    phi_0 = np.exp(f)
    grad_phi_0 = mesh.compute_relative_grad(phi_0)
    return phi_0, grad_phi_0


def eval_errors_on_ref(mesh_c, phi_c, grad_c, mesh_ref, phi_ref, grad_ref):
    ids = mesh_c.find(*mesh_ref.nodes.T)
    mask = ids >= 0
    idx, w = mesh_c.interp(mesh_ref.nodes[mask])

    phi_interp = np.sum(phi_c[idx] * w, axis=1)
    grad_interp = np.sum(grad_c[idx] * w[:, :, None], axis=1)

    m_ref = mesh_ref.mass[mask]
    m_scale = mesh_ref.mass.sum() / m_ref.sum()

    diff_phi = np.abs(phi_interp - phi_ref[mask])
    err_inf = float(np.max(diff_phi))
    err_2 = float(np.sqrt(m_scale * np.sum(m_ref * diff_phi ** 2)))

    diff_grad_sq = np.sum((grad_interp - grad_ref[mask]) ** 2, axis=1)
    err_grad_2 = float(np.sqrt(m_scale * np.sum(m_ref * diff_grad_sq)))
    return err_inf, err_2, err_grad_2


def run_trial_1():
    print("\n================ [TRIAL 1: FROZEN-FIELD & TABLE 2] ================")
    mesh_fine = make_2d_mesh(0.035)
    phi_320, grad_320 = solve_frozen(mesh_fine, 320)
    phi_640, grad_640 = solve_frozen(mesh_fine, 640)
    temporal_ref_diff = float(np.sqrt(mesh_fine.mass @ ((phi_320 - phi_640) ** 2)))

    mesh_045 = make_2d_mesh(0.045)
    phi_045, grad_045 = solve_frozen(mesh_045, 320)
    _, spatial_ref_diff, _ = eval_errors_on_ref(
        mesh_045, phi_045, grad_045, mesh_fine, phi_320, grad_320
    )

    print(
        f"[Draft Text Fill-In]:\n"
        f"  - Doubling temporal reference 320 -> 640 on fine mesh changes L2 field by: {temporal_ref_diff:.3e}\n"
        f"  - Refining spatial reference 0.045 -> 0.035 changes L2 field by: {spatial_ref_diff:.3e}\n"
    )

    print("--- TABLE 2: Inner Linear Solve with Frozen Field ---")
    print(
        "dt       || dx (h_max) || |I_dx| || err_inf(phi) || err_2(phi)   || "
        "err_2(grad phi) || EOC    || CPU (s)"
    )

    # Block 1: Temporal refinement on fine mesh (h_req = 0.035) so O(h^2/dt) does not mask O(dt^2)
    prev_err, prev_dt = None, None
    for nt in [5, 10, 20, 40, 80, 120, 160, 240, 360]:
     
        t0 = time.perf_counter()
        p, g = solve_frozen(mesh_fine, nt)
        cpu = time.perf_counter() - t0
        dt = 1.0 / nt
        e_inf = float(np.max(np.abs(p - phi_320)))
        e_2 = float(np.sqrt(mesh_fine.mass @ ((p - phi_320) ** 2)))
        e_g2 = float(np.sqrt(mesh_fine.mass @ np.sum((g - grad_320) ** 2, axis=1)))
        eoc = np.log(e_2 / prev_err) / np.log(dt / prev_dt) if prev_err else np.nan
        eoc_s = f"{eoc:6.2f}" if np.isfinite(eoc) else "  --- "
        print(
            f"{dt:8.4f} || {mesh_fine.h_max:10.4f} || {mesh_fine.N:6d} || "
            f"{e_inf:12.3e} || {e_2:12.3e} || {e_g2:15.3e} || {eoc_s} || {cpu:6.2f}"
        )
        prev_err, prev_dt = e_2, dt

    print("-" * 96)

    # Block 2: Spatial refinement at fixed dt = 1/160 against h_req = 0.035
    prev_err, prev_h = None, None
    for h_req in [0.18, 0.13, 0.09, 0.06, 0.03, 0.005]:
        m = make_2d_mesh(h_req)
        t0 = time.perf_counter()
        p, g = solve_frozen(m, 160)
        cpu = time.perf_counter() - t0
        e_inf, e_2, e_g2 = eval_errors_on_ref(m, p, g, mesh_fine, phi_320, grad_320)
        eoc = np.log(e_2 / prev_err) / np.log(m.h_max / prev_h) if prev_err else np.nan
        eoc_s = f"{eoc:6.2f}" if np.isfinite(eoc) else "  --- "
        print(
            f"{1/160:8.4f} || {m.h_max:10.4f} || {m.N:6d} || "
            f"{e_inf:12.3e} || {e_2:12.3e} || {e_g2:15.3e} || {eoc_s} || {cpu:6.2f}"
        )
        prev_err, prev_h = e_2, m.h_max

    print("-" * 96)

    # Block 3: Coupled refinement (dt proportional to h_max)
    prev_err, prev_h = None, None
    for h_req, nt in [(0.18, 12), (0.13, 20), (0.09, 32), (0.06, 55)]:
        m = make_2d_mesh(h_req)
        t0 = time.perf_counter()
        p, g = solve_frozen(m, nt)
        cpu = time.perf_counter() - t0
        e_inf, e_2, e_g2 = eval_errors_on_ref(m, p, g, mesh_fine, phi_320, grad_320)
        eoc = np.log(e_2 / prev_err) / np.log(m.h_max / prev_h) if prev_err else np.nan
        eoc_s = f"{eoc:6.2f}" if np.isfinite(eoc) else "  --- "
        print(
            f"{1/nt:8.4f} || {m.h_max:10.4f} || {m.N:6d} || "
            f"{e_inf:12.3e} || {e_2:12.3e} || {e_g2:15.3e} || {eoc_s} || {cpu:6.2f}"
        )
        prev_err, prev_h = e_2, m.h_max


# ==============================================================================
# TRIAL 2: Graph 1 (2 Subplots: Splitting/Secular Inversion & Stencils)
# ==============================================================================
def run_trial_2():
    print("\n================ [TRIAL 2: SPLITTING, O(h^2/dt) INVERSION & STENCILS] ================")
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.3), layout="constrained")

    # (a) Temporal Splitting (Lie vs Strang) + O(h^2/dt) secular dissipation on coarse vs fine mesh
    mesh_coarse = make_2d_mesh(0.11)
    mesh_fine = make_2d_mesh(0.04)
    p_ref_c, _ = solve_frozen(mesh_coarse, 25, splitting="strang")
    p_ref_f, g_ref_f = solve_frozen(mesh_fine, 320, splitting="strang")

    nts = [5, 10, 20, 40, 80, 160]
    dts = [1.0 / n for n in nts]

    for method, ls, mk in [("strang", "-", "o"), ("lie", "--", "s")]:
        errs_fine = []
        for nt in nts:
            p, _ = solve_frozen(mesh_fine, nt, splitting=method)
            errs_fine.append(float(np.sqrt(mesh_fine.mass @ ((p - p_ref_f) ** 2))))
        axs[0].loglog(dts, errs_fine, ls=ls, marker=mk, ms=4, lw=1.1, label=f"{method.capitalize()} ($h=0.04$)")

    # Coarse mesh curve showing error increase as dt -> 0 due to O(h^2/dt)
    errs_coarse = []
    for nt in nts:
        p_c, g_c = solve_frozen(mesh_coarse, nt, splitting="strang")
        _, e2, _ = eval_errors_on_ref(mesh_coarse, p_c, g_c, mesh_fine, p_ref_f, g_ref_f)
        errs_coarse.append(e2)
    axs[0].loglog(dts, errs_coarse, ls="-.", marker="^", ms=4, lw=1.1, color="#d32f2f", label=r"Strang ($h=0.11$, $\mathcal{O}(h^2/\Delta t)$)")

    axs[0].set(
        xlabel=r"Time step $\Delta t$",
        ylabel=r"$\mathrm{err}_2(\varphi)$",
        title=r"(a) Splitting & $\mathcal{O}(h^2/\Delta t)$ Secular Dissipation",
    )
    axs[0].legend(frameon=False, fontsize=7.5)
    axs[0].grid(True, which="both", ls=":", alpha=0.5)

    # (b) Quadrature Stencils: L2 discrepancy vs GH9 across dt
    mesh_s = make_2d_mesh(0.07)
    nts_s = [10, 20, 40, 80, 160]
    dts_s = [1.0 / n for n in nts_s]
    for stencil, ls, mk in [("axial4", ":", "^"), ("axial5", "-.", "d"), ("mc", "--", "x")]:
        errs_st = []
        for nt in nts_s:
            p_gh9, _ = solve_frozen(mesh_s, nt, stencil="gh9")
            p_st, _ = solve_frozen(mesh_s, nt, stencil=stencil)
            errs_st.append(float(np.sqrt(mesh_s.mass @ ((p_st - p_gh9) ** 2))))
        axs[1].loglog(dts_s, errs_st, ls=ls, marker=mk, ms=4, lw=1.1, label=f"{stencil.upper()} vs GH9")

    axs[1].set(
        xlabel=r"Time step $\Delta t$",
        ylabel=r"$\|\varphi_{\mathrm{stencil}} - \varphi_{\mathrm{GH9}}\|_{L^2}$",
        title="(b) Quadrature Stencil Discrepancy",
    )
    axs[1].legend(frameon=False, fontsize=7.5)
    axs[1].grid(True, which="both", ls=":", alpha=0.5)

    fig.savefig(FIG / "fig2_splitting_stencils.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG / 'fig2_splitting_stencils.png'}")


# ==============================================================================
# TRIAL 3: Fig 3 (2x2 3D Surfaces), Fig 4 (2x2 Diagnostics), Fig 5 (3x2 Extreme 1D)
# ==============================================================================
def sim_1d_paths(xx, sol, kap, n_paths=12, seed=765):
    rng = np.random.default_rng(seed)
    nt = len(sol["rho"]) - 1
    u0 = sol["mu0"]
    em = 0.5 * (u0[:-1] + u0[1:]) * np.diff(xx)
    ids = rng.choice(len(em), n_paths, p=em / em.sum())
    paths = np.empty((nt + 1, n_paths))
    paths[0] = xx[ids] + rng.uniform(size=n_paths) * np.diff(xx)[ids]
    vel = sol["b"][:, :, 0] + np.sqrt(2.0 * kap) * sol["opt_alpha"][:, :, 0]
    sub = 20
    dt_sub = sol["dt"] / sub
    for k in range(nt):
        p = paths[k].copy()
        for q in range(sub):
            v = (1.0 - q / sub) * vel[k] + (q / sub) * vel[k + 1]
            p += np.interp(p, xx, v) * dt_sub + np.sqrt(2.0 * kap * dt_sub) * rng.normal(size=n_paths)
            p = 1.0 - np.abs((p + 1.0) % 4.0 - 2.0)
        paths[k + 1] = p
    return paths


def run_trial_3():
    print("\n================ [TRIAL 3: 1D SURFACES (FIG 3), DIAGNOSTICS (FIG 4), EXTREME 3x2 (FIG 5)] ================")
    m1d = Mesh1D(181)
    xx = m1d.nodes[:, 0]
    nt = 80
    tt = np.linspace(0.0, 1.0, nt + 1)

    # Multimodal crossing marginals: bimodal asymmetric source -> trimodal target
    def marg_1d_crossing(nodes, mass):
        x = nodes[:, 0]
        a = (
            0.65 * np.exp(-((x + 0.65) ** 2) / (2.0 * 0.06 ** 2))
            + 0.35 * np.exp(-((x - 0.45) ** 2) / (2.0 * 0.07 ** 2))
            + 1e-12
        )
        b = (
            0.40 * np.exp(-((x - 0.72) ** 2) / (2.0 * 0.05 ** 2))
            + 0.35 * np.exp(-((x + 0.10) ** 2) / (2.0 * 0.06 ** 2))
            + 0.25 * np.exp(-((x + 0.75) ** 2) / (2.0 * 0.05 ** 2))
            + 1e-12
        )
        return a / (mass @ a), b / (mass @ b)

    # 1. FIGURE 3 (2x2 3D Surfaces) & FIGURE 4 (2x2 Multi-Regime Diagnostics)
    # Using theta=0.85 and tracking drho_raw_l1 so each regime shows its true spectral rate!
    fig_3d = plt.figure(figsize=(9.0, 6.6))
    fig_diag, axs_diag = plt.subplots(2, 2, figsize=(8.4, 5.2), layout="constrained")

    regimes_4 = [
        (0.12, 0.0, 0.85, "-", "o", r"$\kappa=0.12, \gamma=0.0$ (Pure SB)"),
        (0.012, 0.0, 0.85, "--", "s", r"$\kappa=0.012, \gamma=0.0$ (Low Diffusion)"),
        (0.04, 0.60, 0.75, "-.", "^", r"$\kappa=0.04, \gamma=+0.60$ (Strong Repulsion)"),
        (0.015, -0.35, 0.85, ":", "d", r"$\kappa=0.015, \gamma=-0.35$ (Attractive / Non-contractive)"),
    ]

    xgrid, tgrid = np.meshgrid(xx, tt)

    for idx, (kap, gam, th, ls, mk, label) in enumerate(regimes_4):
        c = replace(base_config(), Nt=nt, N_iter=55, kappa=kap, gamma=gam, sigma=0.22, theta=th, tol=1e-7, verbose=False)
        sol = solve_mfrsb(m1d, Ekern(m1d.nodes, gam, 0.22), SLStencil1D, c, marg_1d_crossing, Experiment(str(OUT)))
        hist = sol["history"]
        paths = sim_1d_paths(xx, sol, kap, n_paths=10, seed=765 + idx)

        ax3 = fig_3d.add_subplot(2, 2, idx + 1, projection="3d")
        ax3.plot_surface(
            xgrid, tgrid, sol["rho"], cmap="YlGnBu", alpha=0.72, linewidth=0, rstride=2, cstride=2
        )
        for j in range(10):
            ax3.plot(paths[:, j], tt, np.zeros_like(tt), lw=0.75, color="#b71c1c", alpha=0.85)
        ax3.set(xlabel="$x$", ylabel="$t$", zlabel=r"$\rho$", title=label)
        ax3.view_init(28, -58)

        ns = [r["n"] for r in hist]
        axs_diag[0, 0].semilogy(ns, [r["drho_raw_l1"] for r in hist], ls=ls, marker=mk, markevery=6, ms=4, lw=1.0, label=label)
        axs_diag[0, 1].semilogy(ns, np.maximum([r["logz_max"] for r in hist], 1e-16), ls=ls, marker=mk, markevery=6, ms=4, lw=1.0, label=label)
        axs_diag[1, 0].semilogy(ns, np.maximum([r["pair_l1"] for r in hist], 1e-16), ls=ls, marker=mk, markevery=6, ms=4, lw=1.0, label=label)
        axs_diag[1, 1].semilogy(ns, np.maximum([r["terminal_res"] for r in hist], 1e-16), ls=ls, marker=mk, markevery=6, ms=4, lw=1.0, label=label)

    fig_3d.tight_layout()
    fig_3d.savefig(FIG / "fig3_1d_surfaces.png", dpi=220, bbox_inches="tight")
    plt.close(fig_3d)

    titles = [
        (r"(a) Undamped Map Residual $\|\Delta \rho_{\mathrm{raw}}^n\|_{L^1}$", r"$\|\Delta \rho_{\mathrm{raw}}^n\|_{L^1}$"),
        (r"(b) Log-Normalization Gap $\max_k |\log Z_k|$", r"$\max_k |\log Z_k|$"),
        (r"(c) Unnormalized Pair Defect $\|\int \rho_{\mathrm{pair}} dx - 1\|_{\infty}$", r"Pair $L^1$ Defect"),
        (r"(d) Terminal Potential Residual $\|\Delta f_T^n\|_{\infty}$", r"$\|\Delta f_T^n\|_{\infty}$"),
    ]
    for ax, (t_str, y_str) in zip(axs_diag.flat, titles):
        ax.set(title=t_str, xlabel="Iteration $n$", ylabel=y_str)
        ax.grid(True, ls=":", alpha=0.5)
    axs_diag[0, 0].legend(frameon=False, fontsize=7)
    fig_diag.savefig(FIG / "fig4_regime_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig_diag)

    # 2. FIGURE 5 (3x2 2D Transport Plans + Erratic/Filamented Particle Paths across 6 Extreme Regimes)
    fig_5, axs_5 = plt.subplots(3, 2, figsize=(8.6, 7.2), layout="constrained", sharex=True, sharey=True)
    extreme_6 = [
        (0.004, 0.0, r"(a) Vanishing Noise ($\kappa=0.004, \gamma=0$): Razor Filaments"),
        (0.25, 0.0, r"(b) High Noise ($\kappa=0.25, \gamma=0$): Wall-Reflecting Dispersion"),
        (0.015, 0.80, r"(c) Strong Repulsion ($\kappa=0.015, \gamma=+0.80$): Crowd Expansion"),
        (0.015, -0.30, r"(d) Cohesive Attraction ($\kappa=0.015, \gamma=-0.30$): Focused Pinching"),
        (0.005, 0.65, r"(e) Low Noise + Repulsion ($\kappa=0.005, \gamma=+0.65$): Bifurcating Channels"),
        (0.08, -0.45, r"(f) Diffusive Aggregation ($\kappa=0.08, \gamma=-0.45$): Metastable Merging"),
    ]

    for idx, (kap, gam, title_str) in enumerate(extreme_6):
        c = replace(base_config(), Nt=nt, N_iter=45, kappa=kap, gamma=gam, sigma=0.20, theta=0.65, verbose=False)
        sol = solve_mfrsb(m1d, Ekern(m1d.nodes, gam, 0.20), SLStencil1D, c, marg_1d_crossing, Experiment(str(OUT)))
        paths = sim_1d_paths(xx, sol, kap, n_paths=14, seed=900 + idx)

        ax = axs_5.flat[idx]
        vmax = float(np.percentile(sol["rho"], 99.0))
        pcm = ax.pcolormesh(xx, tt, sol["rho"], shading="gouraud", cmap="YlGnBu", vmin=0.0, vmax=vmax)
        for j in range(14):
            ax.plot(paths[:, j], tt, color="black", lw=0.65, alpha=0.75)
            ax.scatter(paths[0, j], 0.0, color="#1976d2", s=8, zorder=5)
            ax.scatter(paths[-1, j], 1.0, color="#d32f2f", s=8, zorder=5)
        ax.set_title(title_str, fontsize=8.5)
        if idx in (4, 5):
            ax.set_xlabel("Space $x \in [-1, 1]$")
        if idx % 2 == 0:
            ax.set_ylabel("Time $t \in [0, 1]$")

    fig_5.colorbar(pcm, ax=axs_5, label=r"Density $\rho(t, x)$", shrink=0.75, pad=0.02)
    fig_5.savefig(FIG / "fig5_1d_extreme_transport.png", dpi=220, bbox_inches="tight")
    plt.close(fig_5)
    print(
        f"Saved:\n"
        f"  - {FIG / 'fig3_1d_surfaces.png'}\n"
        f"  - {FIG / 'fig4_regime_diagnostics.png'}\n"
        f"  - {FIG / 'fig5_1d_extreme_transport.png'}"
    )

    
class Mesh1DDisconnected:
    """1D disconnected domain [-1, 0] U [1, 2] with reflecting barriers at {-1, 0, 1, 2}."""
    def __init__(self, n_half=91):
        self.nh = n_half
        self.N = 2 * n_half
        x_l = np.linspace(-1.0, 0.0, n_half)
        x_r = np.linspace(1.0, 2.0, n_half)
        self.nodes = np.concatenate([x_l, x_r])[:, None]
        dx = 1.0 / (n_half - 1)
        self.dx = dx
        self.mass = np.full(self.N, dx)
        self.mass[[0, n_half - 1, n_half, -1]] *= 0.5
        self.node_h = np.full(self.N, dx)
        self.h_max = float(dx)

        # Component-wise finite-difference derivative (no artificial differentiation across the gap)
        from scipy.sparse import diags, block_diag
        D1 = diags([-np.ones(n_half - 1), np.ones(n_half - 1)], [-1, 1], shape=(n_half, n_half)).tolil() / (2.0 * dx)
        D1[0, 0], D1[0, 1] = -1.0 / dx, 1.0 / dx
        D1[-1, -2], D1[-1, -1] = -1.0 / dx, 1.0 / dx
        self.D = block_diag([D1, D1]).tocoo()

    def compute_relative_grad(self, f):
        return (self.D @ f)[:, None]


class SLStencil1DDisconnected(StencilOps):
    @classmethod
    def build(cls, mesh, b, dt, kappa, rule="gh9", ns=32, seed=555):
        z = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
        w = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
        nh = mesh.nh

        # Left component [-1, 0]: exact reflection at x = -1 and x = 0
        y_l = mesh.nodes[:nh, 0, None] + dt * b[:nh, 0, None] + np.sqrt(2.0 * kappa * dt) * z
        y_l = -np.abs((y_l + 1.0) % 2.0 - 1.0)
        s_l = (y_l + 1.0) * (nh - 1)
        i_l = np.minimum(np.floor(s_l).astype(int), nh - 2)
        a_l = s_l - i_l

        # Right component [1, 2]: exact reflection at x = 1 and x = 2
        y_r = mesh.nodes[nh:, 0, None] + dt * b[nh:, 0, None] + np.sqrt(2.0 * kappa * dt) * z
        y_r = 2.0 - np.abs((y_r - 1.0) % 2.0 - 1.0)
        s_r = (y_r - 1.0) * (nh - 1)
        i_r = np.minimum(np.floor(s_r).astype(int), nh - 2) + nh
        a_r = s_r - (i_r - nh)

        cols_l = np.stack([i_l, i_l + 1], axis=-1).ravel()
        cols_r = np.stack([i_r, i_r + 1], axis=-1).ravel()
        cols = np.concatenate([cols_l, cols_r])

        w_l = (np.stack([1.0 - a_l, a_l], axis=-1) * w[None, :, None]).ravel()
        w_r = (np.stack([1.0 - a_r, a_r], axis=-1) * w[None, :, None]).ravel()
        weights = np.concatenate([w_l, w_r])

        rows = np.repeat(np.arange(mesh.N), 6)
        return cls(rows, cols, weights, mesh.mass)


def run_disconnected_1d():
    print("\n================ [DISCONNECTED 1D DOMAIN & TERMINAL GRADIENT EXPLOSION] ================")
    m = Mesh1DDisconnected(91)
    nh = m.nh
    nt = 80
    tt = np.linspace(0.0, 1.0, nt + 1)

    def marg_disconnected(nodes, mass):
        x = nodes[:, 0]
        # mu_init concentrated on [-1, 0] around x = -0.55; mu_targ on [1, 2] around x = 1.50
        s0, sT = 0.10, 0.065
        mu0 = np.exp(-((x + 0.55) ** 2) / (2.0 * s0 ** 2)) * (x <= 0.0) + 1e-14
        # Steep Gaussian tail of mu_targ extending across the barrier creates terminal pull at x -> 0^-
        muT = np.exp(-((x - 1.50) ** 2) / (2.0 * sT ** 2)) + 1e-300
        return mu0 / (mass @ mu0), muT / (mass @ muT)

    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4), layout="constrained")

    # Panel (a): Space-time transport plan & trapped trajectories on [-1, 0] U [1, 2]
    kap_main = 0.06
    c = replace(base_config(), Nt=nt, N_iter=35, kappa=kap_main, gamma=0.02, verbose=False)
    sol = solve_mfrsb(m, Ekern(m.nodes, 0.02, 0.35), SLStencil1DDisconnected, c, marg_disconnected, Experiment(str(OUT)))

    xx_l = m.nodes[:nh, 0]
    xx_r = m.nodes[nh:, 0]
    vmax = float(np.percentile(sol["rho"], 98.0))
    pcm = axs[0].pcolormesh(xx_l, tt, sol["rho"][:, :nh], shading="gouraud", cmap="YlGnBu", vmin=0.0, vmax=vmax)
    axs[0].pcolormesh(xx_r, tt, sol["rho"][:, nh:], shading="gouraud", cmap="YlGnBu", vmin=0.0, vmax=vmax)

    # Shade blocked obstacle region (0, 1)
    axs[0].axvspan(0.0, 1.0, facecolor="#e0e0e0", edgecolor="black", hatch="///", lw=0.7, zorder=4, label="Blocked gap $(0,1)$")

    # Simulate particles initialized in [-1, 0] according to mu_init
    rng = np.random.default_rng(42)
    n_paths = 14
    u0_l = sol["mu0"][:nh]
    em_l = 0.5 * (u0_l[:-1] + u0_l[1:]) * np.diff(xx_l)
    ids = rng.choice(len(em_l), n_paths, p=em_l / em_l.sum())
    paths = np.empty((nt + 1, n_paths))
    paths[0] = xx_l[ids] + rng.uniform(size=n_paths) * np.diff(xx_l)[ids]
    vel_l = sol["b"][:, :nh, 0] + np.sqrt(2.0 * kap_main) * sol["opt_alpha"][:, :nh, 0]
    sub = 20
    dt_sub = sol["dt"] / sub
    for k in range(nt):
        p = paths[k].copy()
        for q in range(sub):
            v = (1.0 - q / sub) * vel_l[k] + (q / sub) * vel_l[k + 1]
            p += np.interp(p, xx_l, v) * dt_sub + np.sqrt(2.0 * kap_main * dt_sub) * rng.normal(size=n_paths)
            p = -np.abs((p + 1.0) % 2.0 - 1.0)  # Skorokhod reflection at x = -1 and x = 0
        paths[k + 1] = p

    for j in range(n_paths):
        axs[0].plot(paths[:, j], tt, color="black", lw=0.7, alpha=0.8, zorder=5)
        axs[0].scatter(paths[0, j], 0.0, color="#1976d2", s=9, zorder=6)
        axs[0].scatter(paths[-1, j], 1.0, color="#d32f2f", s=9, zorder=6)

    # Plot target marginal contour location on [1, 2]
    axs[0].scatter([1.50], [1.0], color="#2e7d32", marker="*", s=55, zorder=7, label=r"Target $\mu_{\mathrm{targ}}$ mode")
    axs[0].set(
        xlim=(-1.0, 2.0),
        ylim=(0.0, 1.0),
        xlabel=r"Space $x \in [-1,0] \cup [1,2]$",
        ylabel=r"Time $t \in [0,1]$",
        title=r"(a) Trapped Transport on $[-1,0] \cup [1,2]$",
    )
    axs[0].legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=7)
    fig.colorbar(pcm, ax=axs[0], label=r"$\rho(t,x)$", shrink=0.82)

    # Panel (b): Time t vs ||nabla log varphi_t|| explosion near t -> T
    for kap, ls, mk in [(0.12, "-", "o"), (0.06, "--", "s"), (0.025, "-.", "^")]:
        ck = replace(base_config(), Nt=nt, N_iter=35, kappa=kap, gamma=0.02, verbose=False)
        sol_k = solve_mfrsb(m, Ekern(m.nodes, 0.02, 0.35), SLStencil1DDisconnected, ck, marg_disconnected, Experiment(str(OUT)))
        grad_norms = [
            float(np.max(np.abs(m.compute_relative_grad(sol_k["f"][k])[:nh])))
            for k in range(nt + 1)
        ]
        axs[1].semilogy(tt, grad_norms, ls=ls, marker=mk, markevery=8, ms=4, lw=1.1, label=rf"$\kappa = {kap}$")

    axs[1].set(
        xlabel=r"Time $t \in [0,1]$",
        ylabel=r"Gradient Norm $\|\nabla \log \varphi_t\|_{L^\infty([-1,0])}$",
        title=r"(b) Terminal Gradient Explosion as $t \to T^-$",
    )
    axs[1].legend(frameon=False)
    axs[1].grid(True, which="both", ls=":", alpha=0.5)

    fig.savefig(FIG / "fig_disconnected_1d.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG / 'fig_disconnected_1d.png'}")
    
    
        
if __name__ == "__main__":

    # run_trial_0()
    run_trial_1()
    # run_trial_2()
    # run_trial_3()
    # run_disconnected_1d()