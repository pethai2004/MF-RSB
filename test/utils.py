"""Diagnostics, exact P1 sampling, particles, academic plotting, and Gmsh geometry."""
from pathlib import Path
from dataclasses import dataclass, field
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri
import gmsh

plt.rc("font", family="serif", size=9)
plt.rc("axes", labelsize=9, titlesize=9, facecolor="white", edgecolor="black", linewidth=0.6)
plt.rc("legend", fontsize=8)
plt.rc("xtick", labelsize=8)
plt.rc("ytick", labelsize=8)
plt.rc("figure", facecolor="white")

@dataclass
class Experiment:
    path: str = "results"
    history: list = field(default_factory=list)

    def __post_init__(self):
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def log_step(self, record: dict, verbose: bool = True):
        self.history.append(record)
        if verbose:
            if record["n"] == 1:
                print(
                    "Iter (n) || ||drho||_L1 || q_L1   || ||drho||_inf || "
                    "||rho0-mu0||_L1 || ||rhoT-muT||_L1 || W2(rhoT,muT) || "
                    "max|logZ| || Adjoint  || CPU (s)",
                    flush=True,
                )
            q_str = f"{record['q_l1']:.3f}" if np.isfinite(record["q_l1"]) else "  --- "
            print(
                f"{record['n']:8d} || {record['drho_l1']:11.3e} || {q_str:6s} || "
                f"{record['drho_inf']:12.3e} || {record['mu0_l1']:15.3e} || "
                f"{record['muT_l1']:15.3e} || {record['w2_T']:12.3e} || "
                f"{record['logz_max']:9.3e} || {record['adj_err']:8.2e} || "
                f"{record['cpu_time']:6.2f}",
                flush=True,
            )

    def export(self, name="history.json"):
        Path(self.path, name).write_text(json.dumps(self.history, indent=2))


def sliced_w2(nodes_a, weights_a, nodes_b, weights_b, angles=32):
    p = np.asarray(weights_a, float)
    q = np.asarray(weights_b, float)
    p = p / p.sum()
    q = q / q.sum()
    total = 0.0
    if nodes_a.shape[1] == 1:
        dirs = np.ones((1, 1))
    else:
        a = np.arange(angles) * np.pi / angles
        dirs = np.column_stack([np.cos(a), np.sin(a)])
    for direction in dirs:
        a = nodes_a @ direction
        b = nodes_b @ direction
        ia = np.argsort(a)
        ib = np.argsort(b)
        ca = np.cumsum(p[ia])
        cb = np.cumsum(q[ib])
        ca[-1] = cb[-1] = 1.0
        breaks = np.unique(np.r_[0.0, ca, cb])
        mid = (breaks[:-1] + breaks[1:]) / 2.0
        qa = a[ia[np.searchsorted(ca, mid)]]
        qb = b[ib[np.searchsorted(cb, mid)]]
        total += np.sum(np.diff(breaks) * (qa - qb) ** 2)
    return float(np.sqrt(total / len(dirs)))


def compute_w2_marginal(nodes, mass, rho_a, rho_b):
    return sliced_w2(nodes, mass * rho_a, nodes, mass * rho_b)


def free_energy(mesh, E, rho, kappa):
    m = rho * mesh.mass
    return kappa * np.sum(m * np.log(np.maximum(rho, 1e-300)), axis=1) + 0.5 * np.einsum(
        "ti,ij,tj->t", m, E.E, m
    )


def sample_p1_density(mesh, density, Np, rng):
    vals = density[mesh.elements]
    em = mesh.area * vals.mean(axis=1)
    quotas = Np * em / em.sum()
    counts = np.floor(quotas).astype(int)
    missing = Np - counts.sum()
    if missing:
        rem = quotas - counts
        counts += np.bincount(
            rng.choice(len(em), missing, p=rem / rem.sum()), minlength=len(em)
        )
    ids = np.repeat(np.arange(len(em)), counts)
    rng.shuffle(ids)
    v = vals[ids]
    cdf = np.cumsum(v / v.sum(axis=1)[:, None], axis=1)
    j = (rng.random(Np)[:, None] > cdf).sum(axis=1)
    j = np.minimum(j, 2)
    shapes = np.ones((Np, 3))
    shapes[np.arange(Np), j] = 2.0
    g = rng.gamma(shapes)
    lam = g / g.sum(axis=1)[:, None]
    return np.sum(mesh.nodes[mesh.elements[ids]] * lam[:, :, None], axis=1)


def path_sim(mesh, sol, Np=1500, substeps=8, seed=42):
    rng = np.random.default_rng(seed)
    dt = sol["dt"] / substeps
    vel = sol["b"] + np.sqrt(2.0 * sol["kappa"]) * sol["opt_alpha"]
    X = np.empty((len(vel), Np, 2))
    X[0] = sample_p1_density(mesh, sol["mu0"], Np, rng)
    hits = 0
    for k in range(len(vel) - 1):
        x = X[k].copy()
        for j in range(substeps):
            frac = j / substeps
            v = (1.0 - frac) * vel[k] + frac * vel[k + 1]
            idx, w = mesh.interp(x)
            drift = np.sum(w[:, :, None] * v[idx], axis=1)
            prop = x + dt * drift + np.sqrt(2.0 * sol["kappa"] * dt) * rng.normal(size=x.shape)
            x, h = mesh.reflect_step(x, prop)
            hits += h
        X[k + 1] = x
    return X, dict(
        collisions=hits,
        outside=int(np.sum(mesh.find(*X[-1].T) < 0)),
        sw2=sliced_w2(X[-1], np.ones(Np), mesh.nodes, mesh.mass * sol["muT"]),
    )


def draw_domain(ax, mesh, target=None):
    from matplotlib.patches import Circle, FancyBboxPatch

    for center, r in mesh.dom.circles:
        ax.add_patch(
            Circle(center, r, facecolor="#e6e6e6", edgecolor="black", lw=0.7, zorder=4)
        )
    for x0, x1, y0, y1, r in mesh.dom.rrects:
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                boxstyle=f"round,pad=0,rounding_size={r}",
                facecolor="#e6e6e6",
                edgecolor="black",
                lw=0.7,
                zorder=4,
            )
        )
    if target is not None:
        ax.tricontour(
            mesh.tri,
            target,
            levels=[0.25 * target.max()],
            colors="black",
            linestyles="--",
            linewidths=0.6,
            zorder=5,
        )
    a, b, c, d = mesh.dom.bounds
    ax.set(xlim=(a, b), ylim=(c, d), aspect="equal")
    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.set_yticks([0.0, 0.6, 1.2])


def save_snapshot2d(mesh, sol, save_path, X_traj=None, n_shots=6):
    ks = np.linspace(0, len(sol["rho"]) - 1, n_shots, dtype=int)
    fig, axs = plt.subplots(3, 2, figsize=(8.5, 5.2), layout="constrained", sharex=True, sharey=True)
    vmax = float(np.percentile(sol["rho"], 99.5))

    for ax, k in zip(axs.flat, ks):
        if X_traj is None:
            artist = ax.tripcolor(
                mesh.tri,
                sol["rho"][k],
                shading="gouraud",
                cmap="YlGnBu",
                vmin=0.0,
                vmax=vmax,
            )
        else:
            ax.scatter(
                *X_traj[k, :: max(1, X_traj.shape[1] // 400)].T,
                s=1.8,
                color="#4a6fe3",
                alpha=0.35,
                 zorder=2,
            )
            for j in range(min(8, X_traj.shape[1])):
                ax.plot(*X_traj[: k + 1, j].T, lw=0.75, color="black", alpha=0.75, zorder=3)
                ax.scatter(*X_traj[k, j], s=7, color="#d32f2f", zorder=6)
        draw_domain(ax, mesh, sol["muT"])
        ax.set_title(f"$t = {k * sol['dt']:.2f}$")

    if X_traj is None:
        fig.colorbar(artist, ax=axs, label=r"$\rho(t, x)$", shrink=0.75, pad=0.02)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_dom_example(bounds, circles, rounded_rects, h_min=0.028, h_max=0.040):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ

        xmin, xmax, ymin, ymax = bounds
        outer_tag = occ.addRectangle(xmin, ymin, 0.0, xmax - xmin, ymax - ymin)
        holes = []

        for (cx, cy), r in circles:
            holes.append((2, occ.addDisk(cx, cy, 0.0, r, r)))

        for rx0, rx1, ry0, ry1, rf in rounded_rects:
            cx = 0.5 * (rx0 + rx1)
            cy = 0.5 * (ry0 + ry1)
            hx = max(0.0, 0.5 * (rx1 - rx0) - rf)
            hy = max(0.0, 0.5 * (ry1 - ry0) - rf)

            parts = []
            if hx > 1e-8 and hy > 1e-8:
                parts.append((2, occ.addRectangle(cx - hx, cy - hy - rf, 0.0, 2.0 * hx, 2.0 * (hy + rf))))
                parts.append((2, occ.addRectangle(cx - hx - rf, cy - hy, 0.0, 2.0 * (hx + rf), 2.0 * hy)))
                for sx in (-1.0, 1.0):
                    for sy in (-1.0, 1.0):
                        parts.append((2, occ.addDisk(cx + sx * hx, cy + sy * hy, 0.0, rf, rf)))
            elif hx > 1e-8:
                parts.append((2, occ.addRectangle(cx - hx, cy - rf, 0.0, 2.0 * hx, 2.0 * rf)))
                parts.append((2, occ.addDisk(cx - hx, cy, 0.0, rf, rf)))
                parts.append((2, occ.addDisk(cx + hx, cy, 0.0, rf, rf)))
            elif hy > 1e-8:
                parts.append((2, occ.addRectangle(cx - rf, cy - hy, 0.0, 2.0 * rf, 2.0 * hy)))
                parts.append((2, occ.addDisk(cx, cy - hy, 0.0, rf, rf)))
                parts.append((2, occ.addDisk(cx, cy + hy, 0.0, rf, rf)))
            else:
                parts.append((2, occ.addDisk(cx, cy, 0.0, rf, rf)))

            if len(parts) > 1:
                fused, _ = occ.fuse([parts[0]], parts[1:])
                holes.extend(fused)
            else:
                holes.extend(parts)

        if holes:
            occ.cut([(2, outer_tag)], holes)

        occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", h_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        all_nodes = coords.reshape(-1, 3)[:, :2]
        tag_to_index = dict((tag, i) for i, tag in enumerate(node_tags))

        elem_types, _, node_indices = gmsh.model.mesh.getElements(dim=2)
        primary_type = elem_types[0]
        _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(primary_type)
        element_node_tags = node_indices[0].reshape(-1, num_nodes)
        raw_elements = np.vectorize(tag_to_index.get)(element_node_tags)
    finally:
        gmsh.finalize()

    used_nodes, inverse_idx = np.unique(raw_elements, return_inverse=True)
    nodes = all_nodes[used_nodes]
    elements = inverse_idx.reshape(raw_elements.shape)

    x0 = nodes[elements[:, 0]]
    x1 = nodes[elements[:, 1]]
    x2 = nodes[elements[:, 2]]
    detJ = (x1[:, 0] - x0[:, 0]) * (x2[:, 1] - x0[:, 1]) - (x2[:, 0] - x0[:, 0]) * (x1[:, 1] - x0[:, 1])
    cw_mask = detJ < 0.0
    if np.any(cw_mask):
        elements[cw_mask, 1], elements[cw_mask, 2] = (
            elements[cw_mask, 2].copy(),
            elements[cw_mask, 1].copy(),
        )

    return nodes, elements