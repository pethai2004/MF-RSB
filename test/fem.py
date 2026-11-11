from dataclasses import dataclass, field
from typing import List
import numpy as np
from numpy import linalg, ndarray
import scipy.spatial
from scipy.sparse import csr_matrix, diags
import matplotlib.tri

dim = 2
tri = 3


@dataclass
class DOM:
    n_resolves: int = 3
    bounds: tuple = (-2.0, 2.0, -1.0, 1.0)

    def _sdf_gradchi(self, x, dists, glists):
        return dists, glists

    def sdf_gradchi(self, x: ndarray):
        assert x.ndim == dim
        dists = [
            x[:, 0] - self.bounds[0],
            self.bounds[1] - x[:, 0],
            x[:, 1] - self.bounds[2],
            self.bounds[3] - x[:, 1],
        ]
        wall_grads = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float)
        grads = [
            np.broadcast_to(wall_grads[i], (x.shape[0], dim)).copy()
            for i in range(4)
        ]

        dists, grads = self._sdf_gradchi(x, dists, grads)
        dists = np.column_stack(dists)
        minidx = dists.argmin(1)
        chi = dists[np.arange(x.shape[0]), minidx]

        grad_chi = np.zeros((x.shape[0], dim))
        for j, G in enumerate(grads):
            mask = minidx == j
            if np.any(mask):
                grad_chi[mask] = G[mask]

        return chi, grad_chi

    def reflect(self, x: ndarray):
        x_hat = x.copy()
        for _ in range(self.n_resolves):
            chi, grad_chi = self.sdf_gradchi(x_hat)
            mask = chi < 0.0
            if not np.any(mask):
                break
            x_hat[mask] -= 2.0 * chi[mask, None] * grad_chi[mask]

        return x_hat, float(np.maximum(0.0, -chi).max())


@dataclass
class Mesh:
    nodes: np.ndarray
    elements: np.ndarray
    dom: DOM

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.elements = np.asarray(self.elements, dtype=int)

        valid_idx = np.unique(self.elements)
        assert len(valid_idx) == len(self.nodes)

        self.N, self.M = len(self.nodes), len(self.elements)
        self.edges = np.sort(
            np.concatenate(
                [
                    self.elements[:, [0, 1]],
                    self.elements[:, [1, 2]],
                    self.elements[:, [2, 0]],
                ],
                axis=0,
            ),
            axis=1,
        )
        unique_e, count = np.unique(self.edges, axis=0, return_counts=True)
        self.bc_edges = unique_e[count == 1]
        self.bc_idx = np.unique(self.bc_edges)
        self.bc = self.nodes[self.bc_idx]

        x = self.nodes[self.elements]  # [M, 3, 2]
        J = np.stack((x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), axis=-1)
        detJ = np.linalg.det(J)
        if np.any(np.abs(detJ) < 1e-14):
            raise ValueError("Degenerate mesh elements found.")
        self.area = np.abs(detJ) / 2.0
        self.invJ = np.linalg.inv(J)

        Nxa, Nxb = self.invJ[:, 0, :], self.invJ[:, 1, :]
        Nxc = -(Nxa + Nxb)
        self.basis_grad = np.stack([Nxc, Nxa, Nxb], axis=1)  # [M, 3, 2]

        self.mass = np.bincount(
            self.elements.ravel(),
            weights=np.repeat(self.area / 3.0, 3),
            minlength=self.N,
        )
        assert np.all(self.mass > 0)
        self.M_lump = diags(self.mass)
        self.M_lump_inv = diags(1.0 / self.mass)

        self.tri = matplotlib.tri.Triangulation(*self.nodes.T, self.elements)
        self.find = self.tri.get_trifinder()
        self.tree = scipy.spatial.cKDTree(self.nodes, balanced_tree=True)

        rows = np.repeat(self.elements, 3, axis=1).ravel()
        cols = np.tile(self.elements, (1, 3)).ravel()
        w_grad = (self.area[:, None, None] / 3.0) * self.basis_grad
        vx = np.tile(w_grad[:, :, 0], (1, 3)).ravel() / self.mass[rows]
        vy = np.tile(w_grad[:, :, 1], (1, 3)).ravel() / self.mass[rows]
        self.Dx = csr_matrix((vx, (rows, cols)), shape=(self.N, self.N)).tocoo()
        self.Dy = csr_matrix((vy, (rows, cols)), shape=(self.N, self.N)).tocoo()

        # Element diameters (maximum edge length per triangle)
        e01 = linalg.norm(x[:, 1] - x[:, 0], axis=1)
        e12 = linalg.norm(x[:, 2] - x[:, 1], axis=1)
        e20 = linalg.norm(x[:, 0] - x[:, 2], axis=1)
        elem_h = np.maximum(e01, np.maximum(e12, e20))
        self.h_max = float(elem_h.max())

        self.node_h = np.zeros(self.N)
        counts = np.zeros(self.N)
        for loc in range(3):
            np.add.at(self.node_h, self.elements[:, loc], elem_h)
            np.add.at(counts, self.elements[:, loc], 1.0)
        self.node_h /= counts

        # Boundary wall segments (using self.bc_edges, NOT self.edges)
        self.wall_a = self.nodes[self.bc_edges[:, 0]]
        self.wall_e = self.nodes[self.bc_edges[:, 1]] - self.wall_a
        self.wall_mid = self.wall_a + 0.5 * self.wall_e

        self.wall_normal = np.column_stack([self.wall_e[:, 1], -self.wall_e[:, 0]])
        self.wall_normal /= linalg.norm(self.wall_normal, axis=1)[:, None]

        center = np.mean(self.nodes[self.elements], axis=1)
        edge_to_ele = {}
        for j, ele in enumerate(self.elements):
            for i, k in [(0, 1), (1, 2), (2, 0)]:
                edge_to_ele[tuple(sorted((ele[i], ele[k])))] = j
        self.wall_owner = np.array([edge_to_ele[tuple(e)] for e in self.bc_edges])

        flip = np.sum(self.wall_normal * (center[self.wall_owner] - self.wall_mid), axis=1) > 0
        self.wall_normal[flip] *= -1

        self.wall_tree = scipy.spatial.cKDTree(self.wall_mid, balanced_tree=True)
        self.wall_radius = float(linalg.norm(self.wall_e, axis=1).max() / 2.0)

    def reflect(self, x: ndarray):
        return self.dom.reflect(x)

    def interp(self, x):
        e = self.find(x[:, 0], x[:, 1])
        isout = e == -1
        isin = ~isout

        enodes = np.zeros((x.shape[0], 3), dtype=int)
        lambdas = np.zeros((x.shape[0], 3), dtype=float)

        if np.any(isin):
            e_in = e[isin]
            diff = x[isin] - self.nodes[self.elements[e_in, 0]]
            lamb1_2 = np.einsum("abc,ac->ab", self.invJ[e_in], diff)
            lamb0 = 1.0 - lamb1_2.sum(1)
            lll = np.column_stack([lamb0, lamb1_2[:, 0], lamb1_2[:, 1]])
            lll = np.maximum(lll, 0.0)
            lll /= lll.sum(1)[:, None]
            enodes[isin] = self.elements[e_in]
            lambdas[isin] = lll

        if np.any(isout):
            out_pts = x[isout]
            _, cand = self.wall_tree.query(out_pts, k=min(4, len(self.wall_a)))
            cand = np.atleast_2d(cand)
            start_a, edges = self.wall_a[cand], self.wall_e[cand]

            t_proj = np.clip(
                np.sum((out_pts[:, None] - start_a) * edges, axis=2)
                / np.sum(edges * edges, axis=2),
                0.0,
                1.0,
            )
            proj_pts = start_a + t_proj[:, :, None] * edges

            best_idx = linalg.norm(out_pts[:, None] - proj_pts, axis=2).argmin(axis=1)
            best_walls = cand[np.arange(len(out_pts)), best_idx]
            snapped_pts = proj_pts[np.arange(len(out_pts)), best_idx]

            owners = self.wall_owner[best_walls]
            lam_out = np.einsum(
                "abc,ac->ab",
                self.invJ[owners],
                snapped_pts - self.nodes[self.elements[owners, 0]],
            )
            weights = np.maximum(np.column_stack([1.0 - lam_out.sum(1), lam_out]), 0.0)
            lambdas[isout] = weights / weights.sum(1)[:, None]
            enodes[isout] = self.elements[owners]

        return enodes, lambdas

    def reflect_step(self, start, proposal, max_bounces=8):
        """Exact ray-segment reflection from `start` to `proposal`.
        Returns (final_pos, total_hits).
        """
        pos = np.asarray(start, float).copy()
        disp = np.asarray(proposal, float) - pos
        hits = 0

        def cross2d(v1, v2):
            return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

        for _ in range(max_bounces):
            radius = linalg.norm(disp, axis=1) / 2.0 + self.wall_radius + 1e-10
            nearby = self.wall_tree.query_ball_point(pos + disp / 2.0, radius)

            p_ids = np.repeat(np.arange(len(pos)), [len(w) for w in nearby])
            if not len(p_ids):
                break
            w_ids = np.concatenate([w for w in nearby if w])

            move_vec, wall_vec = disp[p_ids], self.wall_e[w_ids]
            delta = self.wall_a[w_ids] - pos[p_ids]
            den = cross2d(move_vec, wall_vec)

            valid = np.abs(den) > 1e-15
            hit_time = np.full(len(p_ids), np.inf)
            hit_pos = np.full(len(p_ids), np.inf)
            hit_time[valid] = cross2d(delta[valid], wall_vec[valid]) / den[valid]
            hit_pos[valid] = cross2d(delta[valid], move_vec[valid]) / den[valid]

            is_hit = (
                valid
                & (hit_time >= -1e-9)
                & (hit_time < 1.0 - 1e-12)
                & (hit_pos >= -1e-9)
                & (hit_pos <= 1.0 + 1e-9)
            )
            is_hit &= np.sum(move_vec * self.wall_normal[w_ids], axis=1) > 1e-14

            first_time = np.full(len(pos), np.inf)
            np.minimum.at(first_time, p_ids[is_hit], np.maximum(hit_time[is_hit], 0.0))

            collided = np.isfinite(first_time)
            if not np.any(collided):
                break

            hits += int(np.sum(collided))

            hit_wall = np.full(len(pos), len(self.wall_e))
            is_first = np.zeros(len(hit_time), bool)
            is_first[is_hit] = np.abs(hit_time[is_hit] - first_time[p_ids[is_hit]]) < 1e-8
            np.minimum.at(hit_wall, p_ids[is_first], w_ids[is_first])

            pos[~collided] += disp[~collided]
            disp[~collided] = 0.0

            n = self.wall_normal[hit_wall[collided]]
            t_hit = first_time[collided, None]

            pos[collided] += t_hit * disp[collided] - 1e-10 * n
            rem_disp = (1.0 - t_hit) * disp[collided]
            disp[collided] = rem_disp - 2.0 * np.sum(rem_disp * n, axis=1)[:, None] * n

        pos += disp
        return pos, hits

    def compute_relative_grad(self, f):
        return np.column_stack([self.Dx @ f, self.Dy @ f])


def quadrature(spec, ns=32, seed=555):
    if spec == "axial4":
        points = np.sqrt(2.0) * np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float)
        w = np.full(4, 0.25)
    elif spec in ("axial5", "5p"):
        points = np.sqrt(6.0) * np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], float)
        w = np.array([2 / 3, 1 / 12, 1 / 12, 1 / 12, 1 / 12], float)
    elif spec in ("gh9", "9p"):
        z1d = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
        w1d = np.array([1 / 6, 2 / 3, 1 / 6])
        Z1, Z2 = np.meshgrid(z1d, z1d)
        W1, W2 = np.meshgrid(w1d, w1d)
        points = np.column_stack([Z1.ravel(), Z2.ravel()])
        w = (W1 * W2).ravel()
    elif spec in ("mc", "mc_adpt"):
        assert ns >= 2 and ns % 2 == 0
        rng = np.random.default_rng(seed)
        half_samples = rng.normal(size=(ns // 2, 2))
        points = np.vstack([half_samples, -half_samples])
        w = np.full(ns, 1.0 / ns)
    else:
        raise ValueError(f"Unknown quadrature rule: {spec}")
    return points, w


_get_quad = quadrature


@dataclass
class StencilOps:
    x0: np.ndarray
    x1: np.ndarray
    w: np.ndarray
    mass: np.ndarray
    pent: int = 0

    def __post_init__(self):
        if np.any(self.w < -1e-13):
            raise FloatingPointError("Negative stencil weight encountered.")
        self.w = np.maximum(self.w, 0.0)
        keep = self.w > 1e-14
        self.x0, self.x1, self.w = self.x0[keep], self.x1[keep], self.w[keep]
        self.n = len(self.mass)

        self.P = csr_matrix((self.w, (self.x0, self.x1)), shape=(self.n, self.n))
        self.P.sum_duplicates()
        self.P.eliminate_zeros()

        self.x0 = np.repeat(np.arange(self.n), np.diff(self.P.indptr))
        self.x1 = self.P.indices.copy()

        self.w = self.P.data.copy()
        self.lgw = np.log(self.w)
        self.lgm = np.log(self.mass)

        self.row_err = float(np.max(np.abs(np.asarray(self.P.sum(1)).ravel() - 1.0)))
        self.nnz_ratio = self.P.nnz / (self.n ** 2)

    @classmethod
    def build(cls, mesh: Mesh, b, dt, kappa, spec="gh9", ns=32, seed=555):
        jump, ws = quadrature(spec, ns, seed)
        x0 = np.tile(mesh.nodes, (len(ws), 1))
        x_half = (
            mesh.nodes[None] + dt * b[None] + np.sqrt(2.0 * kappa * dt) * jump[:, None]
        ).reshape(-1, 2)
        x_1, hits = mesh.reflect_step(x0, x_half)

        verts, betas = mesh.interp(x_1)
        rows = np.repeat(np.tile(np.arange(mesh.N), len(ws)), 3)
        Ws = (np.repeat(ws, mesh.N)[:, None] * betas).ravel()

        return cls(rows, verts.ravel(), Ws, mesh.mass, hits)

    def feyn_bwd(self, logv):
        out = np.full(self.n, -np.inf)
        np.logaddexp.at(out, self.x0, self.lgw + logv[self.x1])
        return out

    def feyn_fwd(self, logv):
        out = np.full(self.n, -np.inf)
        np.logaddexp.at(out, self.x1, self.lgw + logv[self.x0] + self.lgm[self.x0])
        return out - self.lgm

    def adjoint_error(self):
        rng = np.random.default_rng(123)
        u = rng.uniform(-0.5, 0.5, self.n)
        v = rng.uniform(-0.5, 0.5, self.n)
        lhs = float(np.sum(self.mass * np.exp(u) * np.exp(self.feyn_bwd(v))))
        rhs = float(np.sum(self.mass * np.exp(self.feyn_fwd(u)) * np.exp(v)))
        return abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))


SLStenc0 = StencilOps


@dataclass
class Ekern:
    nodes: ndarray
    gamma: float = 0.02
    sigma: float = 0.35

    def __post_init__(self):
        diff = self.nodes[:, None, :] - self.nodes[None, :, :]
        self.E = self.gamma * np.exp(-0.5 * np.sum(diff ** 2, -1) / (self.sigma ** 2))
        self.dE = -(diff / (self.sigma ** 2)) * self.E[:, :, None]

    def field(self, mesh, rho, logphi, kappa):
        rho_mass = rho * mesh.mass[None, :]
        logW = -(rho_mass @ self.E.T) / kappa
        b = -np.einsum("ijd,tj->tid", self.dE, rho_mass)
        rel_grad = np.array([mesh.compute_relative_grad(lp) for lp in logphi])
        r = -np.einsum("ijd,tj,tjd->ti", self.dE, rho_mass, rel_grad)
        return logW, b, r


@dataclass
class DOMobs1(DOM):
    circles: List = field(default_factory=list)
    rrects: List = field(default_factory=list)

    def __post_init__(self):
        self.rrect_data = []
        for x0, x1, y0, y1, rf in self.rrects:
            c = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=float)
            h = np.maximum(0.0, 0.5 * np.array([x1 - x0, y1 - y0]) - rf)
            self.rrect_data.append((c, h, float(rf)))

    def _sdf_gradchi(self, x, dists, grads):
        for (xc, yc), r in self.circles:
            diff = x - np.array([xc, yc])
            norm = linalg.norm(diff, axis=1)
            dists += [norm - r]
            grads += [diff / np.maximum(norm, 1e-14)[:, None]]

        for c, h, rf in self.rrect_data:
            diff = x - c
            q = np.abs(diff) - h
            q_max = np.maximum(q, 0.0)
            out_dist = linalg.norm(q_max, axis=1)
            in_dist = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)

            dists += [out_dist + in_dist - rf]

            sgn = np.where(diff >= 0.0, 1.0, -1.0)
            ext_mask = out_dist > 1e-14

            g = np.zeros((x.shape[0], 2), dtype=float)
            g[ext_mask] = sgn[ext_mask] * q_max[ext_mask] / out_dist[ext_mask, None]

            int_mask = ~ext_mask
            closer_x = q[int_mask, 0] >= q[int_mask, 1]
            g[int_mask, 0] = np.where(closer_x, sgn[int_mask, 0], 0.0)
            g[int_mask, 1] = np.where(~closer_x, sgn[int_mask, 1], 0.0)
            grads += [g]

        return dists, grads


@dataclass
class Mesh1D:
    N: int = 121

    def __post_init__(self):
        self.nodes = np.linspace(-1, 1, self.N)[:, None]
        dx = 2 / (self.N - 1)
        self.mass = np.full(self.N, dx)
        self.mass[[0, -1]] *= 0.5
        self.node_h = np.full(self.N, dx)
        self.h_max = float(dx)
        self.r = 0
        D = diags([-np.ones(self.N - 1), np.ones(self.N - 1)], [-1, 1], shape=(self.N, self.N)).tolil() / (2 * dx)
        D[0, 0] = -1 / dx
        D[0, 1] = 1 / dx
        D[-1, -2] = -1 / dx
        D[-1, -1] = 1 / dx
        self.D = D.tocoo()

    def compute_relative_grad(self, f):
        return (self.D @ f)[:, None]


class SLStencil1D(StencilOps):
    @classmethod
    def build(cls, mesh, b, dt, kappa, rule="gh9", ns=32, seed=555):
        z = np.array([-np.sqrt(3), 0, np.sqrt(3)])
        w = np.array([1 / 6, 2 / 3, 1 / 6])
        y = mesh.nodes[:, 0, None] + dt * b[:, 0, None] + np.sqrt(2 * kappa * dt) * z
        y = -1 + 2 - np.abs((y + 1) % 4 - 2)
        s = (y + 1) * (mesh.N - 1) / 2
        i = np.minimum(np.floor(s).astype(int), mesh.N - 2)
        a = s - i
        cols = np.stack([i, i + 1], axis=-1).ravel()
        weights = (np.stack([1 - a, a], axis=-1) * w[None, :, None]).ravel()
        rows = np.repeat(np.arange(mesh.N), 6)
        return cls(rows, cols, weights, mesh.mass)