
from dataclasses import dataclass
import numpy as np
import gmsh

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from skfem import MeshLine, MeshTri

@dataclass
class Config:
    
    dtype = np.float32
    
    dim = 1
    T = 1.
    mesh = (25, ), # mush be len(mesh) == dim
    Nt = 20,
    kappa = .03
    
    order_ept = 2
    splitting = 'Strang' # Strang / Lie / Yosida
    ln_solver = None # matrix solver ? including this??
    N_iter = 30 
    theta = 0.65
    tol = 1e-5
    kern_gamma = 0.02,
    quite = False 
    output_prefix = './trial'
    
    seed = 42
    
@dataclass
class MetaHist:

    rho_diff : np.ndarray # [Nt, 1]
    marg_err_0 : np.ndarray # [Nt, 1]
    marg_err_1 : np.ndarray 
    penetration: np.ndarray
    avg_penetration : np.ndarray
    max_penetration : np.ndarray
    
    fields_norm : np.ndarray # (Nt, 3)  # each sols norm : rho, varphi, varphi hat 
    fields_grad_norm : np.ndarray # (Nt, 3) # each sols grad norm 
    
def write_sol(field, path): # write field variable [n_times, n_space, dim] 
    if field.dim == 2: # assuming this is [n_space, dim] 
        field = field[None, ...] # add extra 

def read_field(path):
    pass


def read_geom(path): # load predefined nodes and element
    pass 

def render_2evo_gif(field, domain, save_path, smooth=False): # save gif of field [n_times, n_space, dim]
    # this render as 3-d with z axis being field value
    pass

def save_snapshot2d(field, domain, save_path, n_shots=9, smooth=False):
    # save 2d snapshot to grid
    pass

    
# def compute_reflect(x):
#     return 

# def compute_normal(x, sdf_fn,eps=1e-6): 
#     # [Ns, dim=1,2,3] -> [N, dim],  central diff for nabla chi / ||nabla chi||
#     chi = sdf_fn(x) 
#     grad = np.zeros_like(chi)
#     for i in range(x.shape[-1]):
#         dx = np.zeros_like(chi)
#         dx[:,i] = eps 
#         grad[:,i] = ((sdf_fn(chi+dx)-sdf_fn(chi-dx)) / (2*eps)) # central diff 
#     return grad / (np.max(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12))
    
        

def path_sim(pde, config: Config): # forward Euler-Maruyama scheme
    pass


############################ Domain Mesh Generation ########################

# this part is AI-generated
def _extract_mesh_2d():
    tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = coords.reshape(-1, 3)[:, :2]
    tag_map = {int(t): i for i, t in enumerate(tags)}
    elem_types, _, connectivity = gmsh.model.mesh.getElements(dim=2)
    tri_idx = next(i for i, t in enumerate(elem_types) if t == 2)
    triangle_tags = connectivity[tri_idx].reshape(-1, 3)
    elements = np.array([[tag_map[int(t)] for t in tri] for tri in triangle_tags], dtype=int)
    return nodes, elements

def superellipse(cx, cy, a, b, power=6, n=160):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    q = 2.0 / power
    return np.column_stack([cx + a * np.sign(np.cos(t)) * np.abs(np.cos(t))**q, cy + b * np.sign(np.sin(t)) * np.abs(np.sin(t))**q])

def smooth_boundary(vertices, sigma=0.025, n=600):
    vertices = np.asarray(vertices, float)
    closed = np.vstack([vertices, vertices[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.r_[0, np.cumsum(lengths)]
    perimeter = s[-1]
    t = np.linspace(0, perimeter, n, endpoint=False)
    sampled = np.column_stack([np.interp(t, s, closed[:, j]) for j in range(2)])
    sampled = gaussian_filter1d(sampled, sigma=sigma * n / perimeter, axis=0, mode="wrap")
    knots = np.r_[t, perimeter]
    curve = np.vstack([sampled, sampled[0]])
    splines = [CubicSpline(knots, curve[:, j], bc_type="periodic") for j in range(2)]
    return np.column_stack([f(t) for f in splines])

def _add_spline_loop(geo, xy, h, segments=8):
    xy = np.asarray(xy, float)
    n, step = len(xy), len(xy) // segments
    pts = [geo.addPoint(x, y, 0, h) for x, y in xy]
    curves = [geo.addSpline([pts[(k * step + j) % n] for j in range(step + 1)]) for k in range(segments)]
    return geo.addCurveLoop(curves)

def make_dom_1(n=201):
    nodes = np.linspace(-1, 1, n)
    mesh = MeshLine(nodes[None, :])
    nodes, elements = mesh.p.T, mesh.t.T
    x = nodes[:, 0]
    g = lambda mu, s: np.exp(-0.5 * ((x - mu) / s)**2)
    rho0 = g(-0.7, 0.12)
    rhoT = (g(-0.5, 0.12) + g(0.0, 0.12) + g(0.5, 0.12)) / 3.0
    mass = np.zeros(len(nodes))
    np.add.at(mass, elements.ravel(), np.repeat(np.abs(np.diff(nodes[elements], axis=1)).ravel() / 2, 2))
    return nodes, elements, rho0 / (mass @ rho0), rhoT / (mass @ rhoT)

def make_dom_2(h=0.04):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("maze")
        geo = gmsh.model.geo
        outer = _add_spline_loop(geo, superellipse(2.0, 1.0, 2.0, 1.0, power=8), h)
        walls = [(0.8, 0.81, 0.075, 0.73), (1.6, 1.19, 0.075, 0.73), (2.4, 0.81, 0.075, 0.73), (3.2, 1.19, 0.075, 0.73)]
        holes = [_add_spline_loop(geo, superellipse(cx, cy, a, b, power=4, n=120), h) for cx, cy, a, b in walls]
        geo.addPlaneSurface([outer] + holes)
        geo.synchronize()
        gmsh.model.mesh.generate(2)
        nodes, elements = _extract_mesh_2d()
    finally:
        gmsh.finalize()
    mesh = MeshTri(nodes.T, elements.T)
    nodes, elements = mesh.p.T, mesh.t.T
    g = lambda mu: np.exp(-np.sum((nodes - mu)**2, axis=1) / (2 * 0.12**2))
    rho0, rhoT = g([0.30, 1.65]), g([3.70, 0.35])
    v = nodes[elements]
    J = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]], axis=-1)
    mass = np.bincount(elements.ravel(), weights=np.repeat(np.abs(np.linalg.det(J)) / 6, 3), minlength=len(nodes))
    return nodes, elements, rho0 / (mass @ rho0), rhoT / (mass @ rhoT)

def make_dom_3(h=0.04):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("smooth_obstacles")
        geo = gmsh.model.geo
        outer_pts = [geo.addPoint(x, y, 0, h) for x, y in [(0, 0), (4, 0), (4, 1), (0, 1)]]
        lines = [geo.addLine(outer_pts[i], outer_pts[(i + 1) % 4]) for i in range(4)]
        outer_loop = geo.addCurveLoop(lines)
        circle_pts = superellipse(0.85, 0.50, 0.3, 0.3, power=2, n=120)
        L_pts = smooth_boundary([[1.7, 0.1], [2.3, 0.1], [2.3, 0.4], [2.0, 0.4], [2.0, 0.8], [1.7, 0.8]], sigma=0.025)
        L_up_pts = smooth_boundary([[2.8, 0.1], [3.1, 0.1], [3.1, 0.5], [3.4, 0.6], [3.4, 0.7], [2.8, 0.7]], sigma=0.025)
        hole1 = _add_spline_loop(geo, circle_pts, h)
        hole2 = _add_spline_loop(geo, L_pts, h)
        hole3 = _add_spline_loop(geo, L_up_pts, h)
        geo.addPlaneSurface([outer_loop, hole1, hole2, hole3])
        geo.synchronize()
        gmsh.model.mesh.generate(2)
        nodes, elements = _extract_mesh_2d()
    finally:
        gmsh.finalize()
    x, y = nodes.T
    rho0 = ((0.1 <= x) & (x <= 0.5) & (0.6 <= y) & (y <= 0.9)).astype(float) / (0.4 * 0.3)
    g2d = lambda mu, var: np.exp(-0.5 * np.sum((nodes - mu)**2 / var, axis=1)) / (2 * np.pi * np.sqrt(np.prod(var)))
    rhoT = 0.6 * g2d([3.60, 0.18], [0.015, 0.002]) + 0.4 * g2d([3.85, 0.22], [0.008, 0.003])
    return nodes, elements, rho0, rhoT