import os, json
from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray
import gmsh

import matplotlib.pyplot as plt

def compute_w2_marginal(nodes, mass, rho_a, rho_b):
    """Computes fast Gaussian/moment + 1D-sliced Wasserstein-2 distance proxy on unstructured mesh."""
    p = np.maximum(rho_a * mass, 0.0)
    q = np.maximum(rho_b * mass, 0.0)
    p /= np.sum(p)
    q /= np.sum(q)
    # Exact 2D empirical Bures-Wasserstein + sliced 1D quantile transport over 16 angles
    angles = np.linspace(0.0, np.pi, 16, endpoint=False)
    w2_sq = 0.0
    for theta in angles:
        proj = nodes[:, 0] * np.cos(theta) + nodes[:, 1] * np.sin(theta)
        idx = np.argsort(proj)
        proj_s = proj[idx]
        cdf_p = np.cumsum(p[idx])
        cdf_q = np.cumsum(q[idx])
        u = np.linspace(0.0, 1.0, 200)
        qp = np.interp(u, cdf_p, proj_s)
        qq = np.interp(u, cdf_q, proj_s)
        w2_sq += (2.0 / len(angles)) * np.mean((qp - qq) ** 2)
    return float(np.sqrt(w2_sq))

def free_energy(mesh, E, rho: ndarray, kappa):
    rho_mass = rho * mesh.mass[None,:]
    ent = kappa * np.sum( rho_mass * np.log(np.maximum(rho, 1e-300)),axis=1)
    interaction = .5 * np.einsum('ti, ij, tj->t', rho_mass, E.E, rho_mass)
    
    return ent + interaction


def sample_p1_density(mesh, density: np.ndarray, Np: int, rng):
    """Sample particles over a triangulated domain."""

    # Compute and normalize element masses.
    vertex_values = density[mesh.elements]
    masses = mesh.area * np.sum(vertex_values, axis=1) / 3.0
    probabilities = masses / np.sum(masses)

    # Allocate particles to triangles.
    expected = probabilities * Np
    n_per_element = expected.astype(int)

    n_remaining = Np - n_per_element.sum()

    if n_remaining:
        residual = expected - n_per_element
        selected = rng.choice(
            len(masses),
            size=n_remaining,
            p=residual / residual.sum()
        )
        n_per_element += np.bincount(
            selected, minlength=len(masses)
        )

    element_ids = np.repeat(
        np.arange(len(masses)), n_per_element
    )
    rng.shuffle(element_ids)

    # Generate uniformly distributed barycentric coordinates.
    r = rng.uniform(size=(Np, 2))
    reflected = r.sum(axis=1) > 1.0
    r[reflected] = 1.0 - r[reflected]

    barycentric = np.empty((Np, 3))
    barycentric[:, 0] = 1.0 - r[:, 0] - r[:, 1]
    barycentric[:, 1:] = r

    # Map reference-triangle coordinates to physical triangles.
    vertices = mesh.nodes[mesh.elements[element_ids]]

    return np.sum(
        barycentric[:, :, None] * vertices,
        axis=1
    )


def path_sim(
    mesh,
    sol: dict,
    Np: int = 450,
    substeps: int = 6,
    seed: int = 42
):
    """Simulate reflected controlled diffusion trajectories."""

    rng = np.random.default_rng(seed)

    dt = sol["dt"] / substeps
    diffusion = np.sqrt(2.0 * sol["kappa"])

    velocity = sol["b"] + diffusion * sol["opt_alpha"]

    n_steps = velocity.shape[0] - 1
    n_pairs = Np // 2
    n_particles = 2 * n_pairs

    trajectories = np.empty(
        (n_steps + 1, n_particles, 2),
        dtype=float
    )

    trajectories[0] = sample_p1_density(
        mesh, sol["mu0"], n_particles, rng
    )

    max_penetration = 0.0

    for n in range(n_steps):
        positions = trajectories[n].copy()

        for j in range(substeps):
            theta = j / substeps

            # Time interpolation of the nodal velocity.
            nodal_velocity = (
                velocity[n]
                + theta * (velocity[n + 1] - velocity[n])
            )

            # Spatial P1 interpolation at particle locations.
            indices, weights = mesh.interp(positions)

            particle_velocity = np.sum(
                weights[:, :, None] * nodal_velocity[indices],
                axis=1
            )

            # Antithetic noise construction.
            noise = rng.standard_normal((n_pairs, 2))
            increments = np.vstack((noise, -noise))

            # Euler-Maruyama update.
            proposal = (
                positions
                + dt * particle_velocity
                + diffusion * np.sqrt(dt) * increments
            )

            # Return particles to the admissible domain.
            positions, penetration = mesh.reflect(proposal)

            max_penetration = max(
                max_penetration, penetration
            )

        trajectories[n + 1] = positions

    return trajectories, max_penetration
    
@dataclass 
class Experiment: # handle all test config, mornitoring, save results, verbose, etc
    
    path: str = './trial'
    history: dict = field(
        default_factory=lambda : {
            "n": [],
            r"||\Delta \rho^n||_{L^\infty}": [],
            r"q_n": [],
            r"\|\rho_0^n - \mu_0\|_{L^1}": [],
            r"\|\rho_T^n - \mu_T\|_{L^1}": [],
            r"\mathcal{W}_2(\rho_T^n, \mu_T)": [],
            r"\max_k |\log Z_k|": [],
            r"\max(-\chi)": [],
            r"\mathcal{E}_{\mathrm{mass}}": [],
            r"\mathrm{Pe}_h": [],
            r"C_{\mathrm{CFL}}": [],
            r"\mathrm{nnz} / N^2": [],
            r"t_{\mathrm{CPU}} [s]": [],
        }
    )
    def __post_init__(self):
        os.makedirs(self.path, exist_ok=True)
        
    def log_step(
            self,
            n: int,
            rho_next: np.ndarray,
            rho_prev: np.ndarray,
            prev_diff: float,
            mu0: np.ndarray,
            muT: np.ndarray,
            mesh,
            ops: list,
            b_field: np.ndarray,
            logz: np.ndarray,
            dt: float,
            kappa: float,
            cpu_time: float,
            compute_w2_fn,
        ) -> float:
        
            rho_diff = float(np.max(np.abs(rho_next[1:-1] - rho_prev[1:-1])))
            contract_q = (
                float(rho_diff / prev_diff)
                if (prev_diff is not None and prev_diff > 1e-15)
                else np.nan
            )

            mu0_err = float(mesh.mass @ np.abs(rho_next[0] - mu0))
            muT_err = float(mesh.mass @ np.abs(rho_next[-1] - muT))
            w2_T = compute_w2_fn(mesh.nodes, mesh.mass, rho_next[-1], muT)
            pd_gap = float(np.max(np.abs(logz)))

            max_pnt = float(max(op.pent for op in ops))
            err_mass = float(
                max(
                    np.max(np.abs((rho_next @ mesh.mass) - 1.0)),
                    max(op.row_err for op in ops),
                )
            )

            b_norm = np.linalg.norm(b_field, axis=-1)
            pe_h = float(np.max(b_norm * mesh.node_h[None, :]) / (2.0 * kappa))
            cfl = float(np.max(b_norm / mesh.node_h[None, :]) * dt)
            nnz_ratio = float(np.mean([op.nnz_ratio for op in ops]))

            self.history["n"].append(n)
            self.history[r"||\Delta \rho^n||_{L^\infty}"].append(rho_diff)
            self.history[r"q_n"].append(contract_q)
            self.history[r"\|\rho_0^n - \mu_0\|_{L^1}"].append(mu0_err)
            self.history[r"\|\rho_T^n - \mu_T\|_{L^1}"].append(muT_err)
            self.history[r"\mathcal{W}_2(\rho_T^n, \mu_T)"].append(w2_T)
            self.history[r"\max_k |\log Z_k|"].append(pd_gap)
            self.history[r"\max(-\chi)"].append(max_pnt)
            self.history[r"\mathcal{E}_{\mathrm{mass}}"].append(err_mass)
            self.history[r"\mathrm{Pe}_h"].append(pe_h)
            self.history[r"C_{\mathrm{CFL}}"].append(cfl)
            self.history[r"\mathrm{nnz} / N^2"].append(nnz_ratio)
            self.history[r"t_{\mathrm{CPU}} [s]"].append(cpu_time)

            q_str = f"{contract_q:.3f}" if not np.isnan(contract_q) else "   ---"
            print(
                f"  [Iter {n:02d}] "
                f"||drho||_inf: {rho_diff:.2e} | "
                f"q: {q_str} | "
                f"L1(0,T): ({mu0_err:.2e}, {muT_err:.2e}) | "
                f"W2: {w2_T:.2e} | "
                f"MassErr: {err_mass:.1e} | "
                f"MaxPen: {max_pnt:.1e} | "
                f"CFL: {cfl:.2f} | "
                f"dt_CPU: {cpu_time:.2f}s",
                flush=True,
            )
            return rho_diff

    def export(self, name="metrics.json"):
        out_file = os.path.join(self.path, name)
        with open(out_file, "w") as fp:
            json.dump(self.records, fp, indent=2)
        print(f"  [Experiment] Exported metrics table to {out_file}")
        
        
    def save_sol(self, name: str, mesh, sol: dict, X_traj: ndarray = None, target_boxes: list = None):
        """Saves complete solution state so snapshots or GIFs can be reconstructed offline."""
        out_file = os.path.join(self.path, f"{name}.npz")
        payload = {
            "nodes": mesh.nodes,
            "elements": mesh.elements,
            "bounds": np.asarray(mesh.dom.bounds) if mesh.dom else np.array([0, 1, 0, 1]),
            "rho": sol["rho"],
            "f": sol["f"],
            "h": sol["h"],
            "b": sol["b"],
            "opt_alpha": sol["opt_alpha"],
            "free_energy": sol["free_energy"],
            "mu0": sol["mu0"],
            "muT": sol["muT"],
            "dt": np.array(sol["dt"]),
            "T": np.array(sol["T"]),
            "kappa": np.array(sol["kappa"]),
        }
        if X_traj is not None:
            payload["X_traj"] = X_traj
        if target_boxes is not None:
            payload["target_boxes"] = np.asarray(target_boxes)
        np.savez_compressed(out_file, **payload)
        print(f"  [Experiment] Saved full state to {out_file}")
        return out_file
    

def save_snapshot2d(mesh, sol: dict, save_path: str, X_traj: np.ndarray = None,
                    target_boxes: list = None, n_shots: int = 4, title: str = None):
    rho, dt, Nt = sol["rho"], sol["dt"], len(sol["rho"]) - 1
    vmax = np.percentile(rho, 99.0)
    snap_idx = np.linspace(0, Nt, n_shots, dtype=int)
    ncols = 2 if X_traj is not None else 1
    
    fig, axes = plt.subplots(n_shots, ncols, figsize=(4.0 * ncols, 2.5 * n_shots), constrained_layout=True)
    axes = np.atleast_2d(axes).reshape(n_shots, ncols)
    if title: fig.suptitle(title)

    for i, k in enumerate(snap_idx):
        t_cur = k * dt
        
        # Density Map
        ax_l = axes[i, 0]
        tpc = ax_l.tripcolor(mesh.tri, rho[k], shading="gouraud", cmap="magma_r", vmin=0.0, vmax=vmax)
        ax_l.set(aspect="equal", title=f"Density t={t_cur:.2f}")
        
        # Particles and Mesh
        if X_traj is not None:
            ax_r = axes[i, 1]
            ax_r.triplot(mesh.tri, color="gray", lw=0.2, alpha=0.5)
            ax_r.scatter(X_traj[k, :, 0], X_traj[k, :, 1], c="blue", s=4, alpha=0.6)
            
            if target_boxes:
                for bx0, bx1, by0, by1 in target_boxes:
                    ax_r.plot([bx0, bx1, bx1, bx0, bx0], [by0, by0, by1, by1, by0], 'r--', lw=1.2)
                    
            ax_r.set(aspect="equal", title=f"Particles t={t_cur:.2f}")

    fig.colorbar(tpc, ax=axes, location="right", label="Density")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] Saved snapshot to {save_path}")


############################ Domain Mesh Generation ########################

# copied from Gemini
def build_dom_example(bounds, circles, rounded_rects, h_min=0.028, h_max=0.040):
    gmsh.initialize()
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
    tag_to_index = {tag: i for i, tag in enumerate(node_tags)}

    elem_types, _, node_indices = gmsh.model.mesh.getElements(dim=2)
    primary_type = elem_types[0]
    _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(primary_type)
    element_node_tags = node_indices[0].reshape(-1, num_nodes)
    raw_elements = np.vectorize(tag_to_index.get)(element_node_tags)
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
        elements[cw_mask, 1], elements[cw_mask, 2] = elements[cw_mask, 2].copy(), elements[cw_mask, 1].copy()

    return nodes, elements


# def make_dom_2(h_min=0.026, h_max=0.038):
#     """Case 2: C^{1,1} Smooth-Corner Serpentine Maze."""
#     return build_dom_example(
#         bounds=(0.0, 2.8, 0.0, 1.0),
#         circles=[((0.58, 0.26), 0.09), ((0.58, 0.74), 0.09)],
#         rounded_rects=[
#             (0.92, 1.10, -0.15, 0.68, 0.09),
#             (1.42, 1.60,  0.32, 1.15, 0.09),
#             (1.92, 2.12,  0.24, 0.76, 0.09),
#         ],
#         h_min=h_min, h_max=h_max
#     )
# # 1d with boundary [-1, 1] (simple)
# make_dom_3 = None





# def marginals_maze(nodes, mass):
#     x, y = nodes[:, 0], nodes[:, 1]
#     sig = 0.065
#     g1 = np.exp(-((x - 0.26) ** 2 + (y - 0.22) ** 2) / (2.0 * sig ** 2))
#     g2 = np.exp(-((x - 0.26) ** 2 + (y - 0.50) ** 2) / (2.0 * sig ** 2))
#     g3 = np.exp(-((x - 0.26) ** 2 + (y - 0.78) ** 2) / (2.0 * sig ** 2))
#     mu_init = (g1 + g2 + g3) / 3.0 + 1e-12
#     mu_init /= np.sum(mass * mu_init)

#     eps = 0.045
#     dx = np.maximum(0.0, np.maximum(2.30 - x, x - 2.65))
#     dy_bot = np.maximum(0.0, np.maximum(0.10 - y, y - 0.36))
#     dy_top = np.maximum(0.0, np.maximum(0.64 - y, y - 0.90))
#     mu_targ = 0.5 * (np.exp(-(dx ** 2 + dy_bot ** 2) / (2.0 * eps ** 2)) +
#                      np.exp(-(dx ** 2 + dy_top ** 2) / (2.0 * eps ** 2))) + 1e-12
#     mu_targ /= np.sum(mass * mu_targ)
#     return mu_init, mu_targ