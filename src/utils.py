import os
from dataclasses import dataclass
import matplotlib
import numpy as np
from numpy import ndarray
import gmsh

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from skfem import MeshLine, MeshTri

@dataclass 
class Experiment: # handle all test config, mornitoring, save results, verbose, etc
    
    path: str = './trial'
    dim: int = 2 # only support 2 for now
    seed: int = 42
    
    kappa: float = .03 # diffusion -> sigma = sqrt(2*kappa)
    gamma: float = .02 # iteracting kernel strength
    
    N_iter: int  = 30 # max fixed-point iteration
    tol: float = 1e-5 # fixed-point tolerate break
    
    
    def __post_init__(self):
        os.makedirs(self.path)

    def export(self, ): 
        pass 
    
    def add(self, n, X):
        pass 


        



def plot_mesh2d(nodes, ele, figsize=(12, 6)): # "workable"
    triang = matplotlib.tri.Triangulation(nodes[:, 0], nodes[:, 1], ele)

    fig, ax = plt.subplots(figsize=figsize)
    ax.triplot(triang, color="red", markersize=1.5, linewidth=0.1, 
            marker=".", markerfacecolor="blue")

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