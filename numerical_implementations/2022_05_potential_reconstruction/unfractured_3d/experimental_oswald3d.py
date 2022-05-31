import porepy as pp
import numpy as np
import mdestimates as mde
import matplotlib.pyplot as plt
import scipy.sparse as sps
import itertools

from analytical import ExactSolution
from true_errors import TrueError

#%% Study parameters
recon_methods = ["vohralik"]
mesh_size = 0.3
errors = {method: {} for method in recon_methods}
for method in recon_methods:
    errors[method]["majorant"] = []
    errors[method]["true_error"] = []
    errors[method]["i_eff"] = []

#%% Create mesh
domain = {
    "xmin": 0.0,
    "xmax": 1.0,
    "ymin": 0.0,
    "ymax": 1.0,
    "zmin": 0.0,
    "zmax": 1.0,
}
network_3d = pp.FractureNetwork3d(domain=domain)
mesh_args = {
    "mesh_size_bound": mesh_size,
    "mesh_size_frac": mesh_size,
    "mesh_size_min": mesh_size/10,
}

gb = network_3d.mesh(mesh_args)
g = gb.grids_of_dimension(3)[0]
d = gb.node_props(g)


#%%
# Sanity check
if g.dim != 3:
    raise ValueError("cell_edge_map only makes sense for three-dimensional grids")

# Retrieve number of cells, nodes, and faces for quick access
nc: int = g.num_cells
nn: int = g.num_nodes
nf: int = g.num_faces

# Let us first obtain the cell-edges mapping

# Retrieve cell_nodes mapping in array form
nodes_of_cell: np.ndarray = sps.find(g.cell_nodes())[0].reshape((nc, 4))

# Manually create edges. There are 6 edges per cell in 3D
edges_c = np.empty((6, 2, nc), dtype=np.int32)
edges_c[0] = np.vstack([nodes_of_cell[:, 0], nodes_of_cell[:, 1]])
edges_c[1] = np.vstack([nodes_of_cell[:, 0], nodes_of_cell[:, 2]])
edges_c[2] = np.vstack([nodes_of_cell[:, 0], nodes_of_cell[:, 3]])
edges_c[3] = np.vstack([nodes_of_cell[:, 1], nodes_of_cell[:, 2]])
edges_c[4] = np.vstack([nodes_of_cell[:, 1], nodes_of_cell[:, 3]])
edges_c[5] = np.vstack([nodes_of_cell[:, 2], nodes_of_cell[:, 3]])

# Sort the pair of nodes in ascending order to avoid troubles
edges_c_sorted: np.ndarray = np.empty((6, 2, nc), dtype=np.int32)
for edge in range(6):
    edges_c_sorted[edge] = np.sort(edges_c[edge], axis=0)

# We can now create a list of edges
edge_list = []
for edge in range(6):
    for cell in range(nc):
        edge_list.append(tuple(edges_c_sorted[edge][:, cell]))

# Many of these edges will repeat, so we have to make the list unique
unique_edge_list = np.unique(edge_list, axis=0)
ne: int = len(unique_edge_list)

# Having the unique set of edges, we can now create a "nodes -> edge" mapping
col_0 = np.array(unique_edge_list[:, 0], dtype=np.int32)
col_1 = np.array(unique_edge_list[:, 1], dtype=np.int32)
row = np.array(range(ne), dtype=np.int32)
data = np.ones(ne, dtype=bool)
nodes_edge_0 = sps.csc_matrix((data, (row, col_0)), shape=(ne, nn), dtype=bool)
nodes_edge_1 = sps.csc_matrix((data, (row, col_1)), shape=(ne, nn), dtype=bool)
edge_nodes = (nodes_edge_0 + nodes_edge_1).T

# Create a dictionary to store node -> edge mapping
n2e = dict()
for idx, edge in enumerate(unique_edge_list):
    n2e.update({tuple(edge): idx})
ne: int = len(unique_edge_list)  # number of edges

# Obtain the edge-cell connectivity.
# TODO: Avoid looping for better performance
edge_cells = np.zeros((nc, ne), dtype=bool)
for edge in range(6):
    for node_pair, cell in zip(edges_c_sorted[edge].T, range(nc)):
        edge_cells[cell, n2e[tuple(node_pair)]] = True
cell_edges = sps.csc_matrix(edge_cells).T

# We do more or less the same for the face -> edge mapping

# Retrieve face_nodes mapping in array form
nodes_of_face: np.ndarray = sps.find(g.face_nodes)[0].reshape((nf, 3))

# Manually create edges. There are 3 edges per face in 3D
edges_f = np.empty((3, 2, nf), dtype=np.int32)
edges_f[0] = np.vstack([nodes_of_face[:, 0], nodes_of_face[:, 1]])
edges_f[1] = np.vstack([nodes_of_face[:, 0], nodes_of_face[:, 2]])
edges_f[2] = np.vstack([nodes_of_face[:, 1], nodes_of_face[:, 2]])

edges_f_sorted: np.ndarray = np.empty((3, 2, nf), dtype=np.int32)
for edge in range(3):
    edges_f_sorted[edge] = np.sort(edges_f[edge], axis=0)

# Create a list of edges
edge_faces = np.zeros((nf, ne), dtype=bool)
for edge in range(3):
    for node_pair, face in zip(edges_f_sorted[edge].T, range(nf)):
        edge_faces[face, n2e[tuple(node_pair)]] = True
face_edges = sps.csc_matrix(edge_faces).T

#%
#
# #%% Declare keywords
# pp.set_state(d)
# parameter_keyword = "flow"
# subdomain_variable = "pressure"
# flux_variable = "flux"
# subdomain_operator_keyword = "diffusion"
#
# # %% Assign data
# exact = ExactSolution(gb)
# integrated_sources = exact.integrate_f(degree=10)
# bc_faces = g.get_boundary_faces()
# bc_type = bc_faces.size * ["dir"]
# bc = pp.BoundaryCondition(g, bc_faces, bc_type)
# bc_values = exact.dir_bc_values()
# permeability = pp.SecondOrderTensor(np.ones(g.num_cells))
#
# parameters = {
#     "bc": bc,
#     "bc_values": bc_values,
#     "source": integrated_sources,
#     "second_order_tensor": permeability,
#     "ambient_dimension": gb.dim_max(),
# }
# pp.initialize_data(g, d, "flow", parameters)
#
# # %% Discretize the model
# subdomain_discretization = pp.RT0(keyword=parameter_keyword)
# source_discretization = pp.DualScalarSource(keyword=parameter_keyword)
#
# d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 1}}
# d[pp.DISCRETIZATION] = {subdomain_variable: {
#     subdomain_operator_keyword: subdomain_discretization,
#     "source": source_discretization
# }
# }
#
# # %% Assemble and solve
# assembler = pp.Assembler(gb)
# assembler.discretize()
# A, b = assembler.assemble_matrix_rhs()
# sol = sps.linalg.spsolve(A, b)
# assembler.distribute_variable(sol)
#
# # Overwrite d[pp.STATE][subdomain_variable] to be consistent with FEM
# for g, d in gb:
#     discr = d[pp.DISCRETIZATION][subdomain_variable][
#         subdomain_operator_keyword]
#     pressure = discr.extract_pressure(g, d[pp.STATE][subdomain_variable],
#                                       d).copy()
#     flux = discr.extract_flux(g, d[pp.STATE][subdomain_variable], d).copy()
#     d[pp.STATE][subdomain_variable] = pressure
#     d[pp.STATE][flux_variable] = flux
#
# # %% Estimate errors
# source_list = [exact.f("fun")]
# for method in recon_methods:
#     estimates = mde.ErrorEstimate(gb, source_list=source_list, p_recon_method=method)
#     estimates.estimate_error()
#     estimates.transfer_error_to_state()
#     majorant = estimates.get_majorant()
#
#     te = TrueError(estimates)
#     true_error = te.true_error()
#     i_eff = majorant / true_error
#
#     print(f"Mesh size: {mesh_size}")
#     print(f"Pressure Reconstruction Method: {estimates.p_recon_method}")
#     print(f"Majorant: {majorant}")
#     print(f"True error: {true_error}")
#     print(f"Efficiency index: {i_eff}")
#     print(50 * "-")
#
#     errors[method]["majorant"].append(majorant)
#     errors[method]["true_error"].append(true_error)
#     errors[method]["i_eff"].append(i_eff)