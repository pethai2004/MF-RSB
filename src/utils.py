
import numpy as np
import gmsh

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



class MeshGen:

    def __init__(self, name="domain"):

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        self.occ = gmsh.model.occ

    # --- Primitives ---
    def circle(self, c, r): 
        return (2, self.occ.addDisk(c[0], c[1], 0, r, r))

    def rectangle(self, corner, dx, dy):
        return (2, self.occ.addRectangle(corner[0], corner[1], 0, dx, dy))

    def hexagon(self, c, r):
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        pts = np.column_stack([c[0] + r * np.cos(angles), c[1] + r * np.sin(angles)])
        return self.build_piecewise(pts)

    def cylinder(self, c, r, dz):
        return (3, self.occ.addCylinder(c[0], c[1], c[2], 0, 0, dz, r))

    def build_piecewise(self, points, smooth=False, extrude_dz=0.0):
        pts = np.asarray(points)
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])

        gmsh_pts = [self.occ.addPoint(p[0], p[1], p[2] if len(p) > 2 else 0.0) for p in pts[:-1]]
        
        if smooth:
            wire = self.occ.addWire([self.occ.addBSpline(gmsh_pts + [gmsh_pts[0]])])
        else:
            lines = [self.occ.addLine(gmsh_pts[i], gmsh_pts[(i + 1) % len(gmsh_pts)]) for i in range(len(gmsh_pts))]
            wire = self.occ.addWire(lines)

        surf = (2, self.occ.addPlaneSurface([wire]))
        if extrude_dz > 0:
            return next(item for item in self.occ.extrude([surf], 0, 0, extrude_dz) if item[0] == 3)
        return surf

    def cut_holes(self, outer, holes):
        holes = [h for h in holes if h is not None]
        return self.occ.cut([outer], holes)[0][0] if holes else outer


def generate_mesh(gen_instance, dim=2, h_min=0.1, h_max=0.25):

    gen_instance.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", h_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
    gmsh.model.mesh.generate(dim)

    # Extract node coordinates
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = coords.reshape(-1, 3)[:, :dim]
    tag_to_index = {tag: i for i, tag in enumerate(node_tags)}

    # Extract primary elements for target dimension dynamically
    elem_types, _, node_indices = gmsh.model.mesh.getElements(dim=dim)
    if not elem_types:
        raise RuntimeError(f"No mesh elements found for dimension {dim}.")

    primary_type = elem_types[0]
    _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(primary_type)
    element_node_tags = node_indices[0].reshape(-1, num_nodes)

    elements = np.vectorize(tag_to_index.get)(element_node_tags)

    gmsh.finalize()
    return nodes, elements

def visualize_mesh():
    gmsh.fltk.run()
    gmsh.finalize()
    
    
    
############################ Specific simulation case utils ########################

# each of these create nodes and element mesh for each experiment specific domain.
def make_data_1(h=0.02, p1, p2, p3): 
    return

def make_data_2(): 
    return

def make_data_2(): 
    return