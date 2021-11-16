import porepy as pp
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import mdestimates as mde
import sympy as sym
import mdestimates.estimates_utils as utils

from mdestimates._velocity_reconstruction import _reconstructed_face_fluxes

#%% Create a grid, with a rotated one-dimensional fracture
p = np.array([[0.50, 0.50], [0.25, 0.75]])
e = np.array([[0], [1]])
domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}
network_2d = pp.FractureNetwork2d(p, e, domain)
mesh_args = {'mesh_size_frac': 0.05, 'mesh_size_bound': 0.05}
gb = network_2d.mesh(mesh_args)

g2d = gb.grids_of_dimension(2)[0]
g1d = gb.grids_of_dimension(1)[0]
d1d = gb.node_props(g1d)
cc1 = g1d.cell_centers
fc1 = g1d.face_centers

g_rot = utils.rotate_embedded_grid(g1d)

#%% Establish an exact solution on the fracture
x, y, z = sym.symbols("x y z")

p_sym = -4*x**2*y**3
gradp_sym = [sym.diff(p_sym, x), sym.diff(p_sym, y), sym.diff(p_sym, z)]
u_sym = [-gradp_sym[0], -gradp_sym[1], -gradp_sym[2]]

p_fun = sym.lambdify((x, y, z), p_sym, "numpy")
gradp_fun = [sym.lambdify((x, y, z), gradp, "numpy") for gradp in gradp_sym]
u_fun = [sym.lambdify((x, y, z), u, "numpy") for u in u_sym]

p_cc = p_fun(g1d.cell_centers[0], g1d.cell_centers[1], g1d.cell_centers[2])
u_cc = [u(g1d.cell_centers[0], g1d.cell_centers[1], g1d.cell_centers[2]) for u in u_fun]
u_cc[2] = np.zeros(g1d.num_cells)
u_fc = [u(g1d.face_centers[0], g1d.face_centers[1], g1d.face_centers[2]) for u in u_fun]
u_fc[2] = np.zeros(g1d.num_faces)

full_flux = (u_fc * g1d.face_normals).sum(axis=0)

#%% Reconstruct flux
d1d["estimates"] = {}
d1d["estimates"]["full_flux"] = full_flux

# Useful mappings
cell_faces_map, _, _ = sps.find(g1d.cell_faces)
cell_nodes_map, _, _ = sps.find(g1d.cell_nodes())

# Cell-basis arrays
faces_cell = cell_faces_map.reshape(np.array([g1d.num_cells, g1d.dim + 1]))
opp_nodes_cell = utils.get_opposite_side_nodes(g1d)
opp_nodes_coor_cell = g_rot.nodes[:, opp_nodes_cell]
sign_normals_cell = utils.get_sign_normals(g1d, g_rot)
vol_cell = g1d.cell_volumes

# Perform actual reconstruction and obtain coefficients
coeffs = np.empty([g1d.num_cells, g1d.dim + 1])
alpha = 1 / (g1d.dim * vol_cell)
coeffs[:, 0] = alpha * np.sum(sign_normals_cell * full_flux[faces_cell], axis=1)
for dim in range(g1d.dim):
    coeffs[:, dim + 1] = -alpha * np.sum(
        (sign_normals_cell * full_flux[faces_cell] * opp_nodes_coor_cell[dim]),
        axis=1,
    )

# TEST -> Flux reconstruction
# Check if the reconstructed evaluated at the face centers normal fluxes
# match the numerical ones
recons_flux = _reconstructed_face_fluxes(g1d, g_rot, coeffs)
np.testing.assert_almost_equal(
    recons_flux,
    full_flux,
    decimal=12,
    err_msg="Flux reconstruction has failed.",
)
# END OF TEST

# Store coefficients in the data dictionary
d1d["estimates"]["recon_u"] = coeffs

# Reconstructed cell center velocities
u_recon_cc = coeffs[:, 0] * g_rot.cell_centers[0] + coeffs[:, 1]
# Rotated exact velocities
u_cc_rot = np.dot(g_rot.rotation_matrix, u_cc)[g_rot.dim_bool]


# THE ABOVE TWO QUANTITIES ARE THE ONE THAT WE NEED TO COMPARE
plt.plot(g_rot.cell_centers.flatten(), u_recon_cc.flatten(), label="Reconstructed cc "
                                                                  "velocities")
plt.plot(g_rot.cell_centers.flatten(), u_cc_rot.flatten(), label="Exact cc velocities")
plt.legend()
plt.show()

#%% Reconstruct pressure

# Retrieving topological data
nc = g1d.num_cells
nf = g1d.num_faces
nn = g1d.num_nodes

# Perform reconstruction
cell_nodes = g1d.cell_nodes()
cell_node_volumes = cell_nodes * sps.dia_matrix((g1d.cell_volumes, 0), (nc, nc))
sum_cell_nodes = cell_node_volumes * np.ones(nc)
cell_nodes_scaled = (
        sps.dia_matrix((1.0 / sum_cell_nodes, 0), (nn, nn)) * cell_node_volumes
)

# Project fluxes using RT0
# d_RT0 = d.copy()
# pp.RT0(self.kw).discretize(g, d_RT0)
# proj_flux = pp.RT0(self.kw).project_flux(g, flux, d_RT0)[: g.dim]

proj_flux = u_recon_cc

# Obtain local gradients
loc_grad = - proj_flux

# Obtaining nodal pressures
cell_nodes_map, _, _ = sps.find(g1d.cell_nodes())
cell_node_matrix = cell_nodes_map.reshape(np.array([g1d.num_cells, g1d.dim + 1]))
nodal_pressures = np.zeros(nn)

for col in range(g1d.dim + 1):
    nodes = cell_node_matrix[:, col]
    dist = g_rot.nodes[: g1d.dim, nodes] - g_rot.cell_centers[: g1d.dim]
    scaling = cell_nodes_scaled[nodes, np.arange(nc)]
    contribution = (
            np.asarray(scaling) * (p_cc + np.sum(dist * loc_grad, axis=0))
    ).ravel()
    nodal_pressures += np.bincount(nodes, weights=contribution, minlength=nn)


# Export lagrangian nodes and coordintates
cell_nodes_map, _, _ = sps.find(g1d.cell_nodes())
nodes_cell = cell_nodes_map.reshape(np.array([g1d.num_cells, g1d.dim + 1]))
point_val = nodal_pressures[nodes_cell]
point_coo = g_rot.nodes[:, nodes_cell]

# Obtain pressure coefficients
recons_p = utils.interpolate_P1(point_val, point_coo)

reconp_cc = recons_p[:, 0] * g_rot.cell_centers[0] + recons_p[:, 1]
