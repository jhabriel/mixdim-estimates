import numpy as np
import porepy as pp
import scipy.sparse as sps
import matplotlib.pyplot as plt
import mdestimates as mde
import pypardiso

from analytical_2d import ExactSolution2D
from true_errors_2d import TrueErrors2D
import mdestimates.estimates_utils as utils

# %% Study parameters
recon_methods = ["cochez", "keilegavlen", "vohralik"]
recon_methods = ["cochez", "keilegavlen"]
errors = {method: {} for method in recon_methods}
for method in recon_methods:
    errors[method]["majorant"] = []
    errors[method]["true_error"] = []
    errors[method]["i_eff"] = []
# mesh_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
mesh_size = 0.1

# Create grid bucket and extract data
domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
network_2d = pp.fracture_importer.network_2d_from_csv("network.csv", domain=domain)

# Target lengths
target_h_bound = mesh_size
target_h_fract = mesh_size
mesh_args = {"mesh_size_bound": target_h_bound, "mesh_size_frac": target_h_fract}

# Construct grid bucket
gb = network_2d.mesh(mesh_args, constraints=[1, 2])

# Get hold of grids and dictionaries
g_2d = gb.grids_of_dimension(2)[0]
g_1d = gb.grids_of_dimension(1)[0]
h_max = gb.diameter()
d_2d = gb.node_props(g_2d)
d_1d = gb.node_props(g_1d)
d_e = gb.edge_props([g_1d, g_2d])
mg = d_e["mortar_grid"]

# Populate the data dictionaries with pp.STATE
for g, d in gb:
    pp.set_state(d)

for e, d in gb.edges():
    pp.set_state(d)

# Instantiate exact solution object and obtain integrated sources
ex = ExactSolution2D(gb)
bc_vals_2d = ex.dir_bc_values()
integrated_f2d = ex.integrate_f2d()
integrated_f1d = ex.integrate_f1d()

# %% Obtain numerical solution
parameter_keyword = "flow"
max_dim = gb.dim_max()

# Set parameters in the subdomains
for g, d in gb:

    # Get hold of boundary faces and declare bc-type. We assign Dirichlet
    # bc to the bulk, and no-flux for the 2D fracture
    bc_faces = g.get_boundary_faces()
    bc_type = bc_faces.size * ["dir"]
    bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
    specified_parameters = {"bc": bc}

    # Also set the values - specified as vector of size g.num_faces
    bc_vals = np.zeros(g.num_faces)
    if g.dim == max_dim:
        bc_vals = bc_vals_2d
    specified_parameters["bc_values"] = bc_vals

    # (Integrated) source terms are given by the exact solution
    if g.dim == max_dim:
        source_term = integrated_f2d
    else:
        source_term = integrated_f1d

    specified_parameters["source"] = source_term

    # Initialize default data
    pp.initialize_default_data(g, d, parameter_keyword, specified_parameters)

# Next loop over the edges
for e, d in gb.edges():
    # Set the normal diffusivity
    data = {"normal_diffusivity": 1}

    # Add parameters: We again use keywords to identify sets of parameters.
    mg = d["mortar_grid"]
    pp.initialize_data(mg, d, parameter_keyword, data)

# Discretize model with RT0-P0
subdomain_discretization = pp.RT0(keyword=parameter_keyword)
source_discretization = pp.DualScalarSource(keyword=parameter_keyword)

# Define keywords
subdomain_variable = "pressure"
flux_variable = "flux"
subdomain_operator_keyword = "diffusion"
edge_discretization = pp.RobinCoupling(
    parameter_keyword, subdomain_discretization, subdomain_discretization
)
edge_variable = "interface_flux"
coupling_operator_keyword = "interface_diffusion"

# Loop over all subdomains in the GridBucket
for g, d in gb:
    d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 1}}
    d[pp.DISCRETIZATION] = {
        subdomain_variable: {
            subdomain_operator_keyword: subdomain_discretization,
            "source": source_discretization,
        }
    }

# Next, loop over the edges
for e, d in gb.edges():
    # Get the grids of the neighboring subdomains
    g1, g2 = gb.nodes_of_edge(e)
    # The interface variable has one degree of freedom per cell in the mortar grid
    d[pp.PRIMARY_VARIABLES] = {edge_variable: {"cells": 1}}

    # The coupling discretization links an edge discretization with
    # variables
    d[pp.COUPLING_DISCRETIZATION] = {
        coupling_operator_keyword: {
            g1: (subdomain_variable, subdomain_operator_keyword),
            g2: (subdomain_variable, subdomain_operator_keyword),
            e: (edge_variable, edge_discretization),
        }
    }

# Assemble, solve, and distribute variables
assembler = pp.Assembler(gb)
assembler.discretize()
A, b = assembler.assemble_matrix_rhs()
# sol = sps.linalg.spsolve(A, b)
sol = pypardiso.spsolve(A, b)
assembler.distribute_variable(sol)

# Overwrite d[pp.STATE][subdomain_variable] to be consistent with FEM
for g, d in gb:
    discr = d[pp.DISCRETIZATION][subdomain_variable][subdomain_operator_keyword]
    pressure = discr.extract_pressure(g, d[pp.STATE][subdomain_variable], d).copy()
    flux = discr.extract_flux(g, d[pp.STATE][subdomain_variable], d).copy()
    d[pp.STATE][subdomain_variable] = pressure
    d[pp.STATE][flux_variable] = flux

# %% Testing diffusive error
estimates = mde.ErrorEstimate(gb, lam_name=edge_variable, p_recon_method="vohralik")
# estimates.estimate_error()
# estimates.transfer_error_to_state()

# Populating data dicitionaries with the key: self.estimates_kw
estimates.init_estimates_data_keyword()

print("Performing velocity reconstruction...", end="")
vel_rec = mde.VelocityReconstruction(estimates)
# 1.1: Compute full flux
vel_rec.compute_full_flux()
# 1.2: Reconstruct velocity
vel_rec.reconstruct_velocity()
print("\u2713")

print("Performing pressure reconstruction...", end="")
p_rec = mde.PressureReconstruction(estimates)
# 2.1: Reconstruct pressure
p_rec.reconstruct_pressure()
print("\u2713")

print("Computing upper bounds...", end="")
# 3.1 Diffusive errors
diffusive_error = mde.DiffusiveError(estimates)
diffusive_error.compute_diffusive_error()

# %% DEV P2 diffusive error

# Loop through all the edges of the grid bucket
for e, d in estimates.gb.edges():
    edge = e
    d_e = d
    # Obtain the interface diffusive flux error
    # diffusive_error = estimates.interface_diffusive_error(e, d_e)
    # d_e[estimates.estimates_kw]["diffusive_error"] = diffusive_error

# Get mortar grid and check dimensionality
mg = d_e["mortar_grid"]
if mg.dim != 1:
    raise ValueError("Expected one-dimensional mortar grid")

# Get hold of higher- and lower-dimensional neighbors and their dictionaries
g_l, g_h = estimates.gb.nodes_of_edge(edge)
d_h = estimates.gb.node_props(g_h)
d_l = estimates.gb.node_props(g_l)

# Retrieve normal diffusivity
normal_diff = d_e[pp.PARAMETERS][estimates.kw]["normal_diffusivity"]
if isinstance(normal_diff, int) or isinstance(normal_diff, float):
    k = normal_diff * np.ones([mg.num_cells, 1])
else:
    k = normal_diff.reshape(mg.num_cells, 1)

# Face-cell map between higher- and lower-dimensional subdomains
frac_faces = sps.find(mg.primary_to_mortar_avg().T)[0]
frac_cells = sps.find(mg.secondary_to_mortar_avg().T)[0]

gh_rot = mde.RotatedGrid(g_h)
gl_rot = mde.RotatedGrid(g_l)
rotation_matrix = gl_rot.rotation_matrix
dim_bool = gl_rot.dim_bool

# Obtain the cells corresponding to the frac_faces
cells_of_frac_faces, _, _ = sps.find(g_h.cell_faces[frac_faces].T)

# Retrieve the coefficients of the polynomials corresponding to those cells
if "recon_p" in d_h[estimates.estimates_kw]:
    p_high = d_h[estimates.estimates_kw]["recon_p"]
else:
    raise ValueError("Pressure must be reconstructed first")
p_high = p_high[cells_of_frac_faces]

# Get nodes of the fracture faces
nodes_of_frac_faces = np.reshape(
    sps.find(g_h.face_nodes.T[frac_faces].T)[0], [frac_faces.size, g_h.dim]
)


def get_lagrangian_coordinates(grid) -> np.ndarray:
    # Obtain the coordinates of the nodes of the fracture faces
    lagran_coo_nodes = grid.nodes[:, nodes_of_frac_faces]
    lagran_coo_fc = grid.face_centers[:, frac_faces.reshape((frac_faces.size, 1))]
    lagran_coo = np.dstack((lagran_coo_nodes, lagran_coo_fc))
    return lagran_coo


# Evaluate the polynomials at the relevant Lagrangian nodes
point_coo_rot = get_lagrangian_coordinates(gh_rot)
point_val = utils.eval_p2(p_high, point_coo_rot)

# Rotate the coordinates of the Lagrangian nodes w.r.t. the lower-dimensional grid
point_coo = get_lagrangian_coordinates(g_h)
point_edge_coo_rot = np.empty_like(point_coo)
for element in range(frac_faces.size):
    point_edge_coo_rot[:, element] = np.dot(rotation_matrix, point_coo[:, element])
point_edge_coo_rot = point_edge_coo_rot[dim_bool]

# Construct a polynomial (of reduced dimensionality) using the rotated coordinates
trace_pressure = utils.interpolate_p2(point_val, point_edge_coo_rot)

# Test if the values of the original polynomial match the new one
point_val_rot = utils.eval_p2(trace_pressure, point_edge_coo_rot)
np.testing.assert_almost_equal(point_val, point_val_rot, decimal=12)

#     # %% Obtain error estimates (and transfer them to d[pp.STATE])
#     for method in recon_methods:
#         estimates = mde.ErrorEstimate(gb, lam_name=edge_variable, p_recon_method=method)
#         estimates.estimate_error()
#         estimates.transfer_error_to_state()
#
#         # %% Retrieve errors
#         te = TrueErrors2D(gb=gb, estimates=estimates)
#
#         diffusive_sq_2d = d_2d[pp.STATE]["diffusive_error"]
#         diffusive_sq_1d = d_1d[pp.STATE]["diffusive_error"]
#         diffusive_sq_lmortar = d_e[pp.STATE]["diffusive_error"][int(mg.num_cells / 2) :]
#         diffusive_sq_rmortar = d_e[pp.STATE]["diffusive_error"][: int(mg.num_cells / 2)]
#         diffusive_error = np.sqrt(
#             diffusive_sq_2d.sum()
#             + diffusive_sq_1d.sum()
#             + diffusive_sq_lmortar.sum()
#             + diffusive_sq_rmortar.sum()
#         )
#
#         residual_sq_2d = te.residual_error_2d_local_poincare()
#         residual_sq_1d = te.residual_error_1d_local_poincare()
#         residual_error = np.sqrt(residual_sq_2d.sum() + residual_sq_1d.sum())
#
#         majorant = diffusive_error + residual_error
#         true_error = te.pressure_error()
#         i_eff = majorant / true_error
#
#         print(50 * "-")
#         print(f"Mesh size: {mesh_size}")
#         print(f"Number of cells: {gb.num_cells()}")
#         print(f"Pressure Reconstruction Method: {estimates.p_recon_method}")
#         print(f"Majorant: {majorant}")
#         print(f"True error: {true_error}")
#         print(f"Efficiency index: {i_eff}")
#         print(50 * "-")
#
#         errors[method]["majorant"].append(majorant)
#         errors[method]["true_error"].append(true_error)
#         errors[method]["i_eff"].append(i_eff)
#
# #%% Plotting
# plt.rcParams.update({'font.size': 13})
# fig, (ax1, ax2) = plt.subplots(
#     nrows=1,
#     ncols=2,
#     gridspec_kw={'width_ratios': [2, 1]},
#     figsize=(11, 5)
# )
#
# ax1.plot(
#     np.log2(1/np.array(mesh_sizes)),
#     np.log2(np.array(errors["cochez"]["majorant"])),
#     linewidth=3,
#     linestyle="-",
#     color="red",
#     marker=".",
#     markersize="10",
#     label="Majorant PR1",
# )
#
# ax1.plot(
#     np.log2(1/np.array(mesh_sizes)),
#     np.log2(np.array(errors["cochez"]["true_error"])),
#     linewidth=2,
#     linestyle="--",
#     color="red",
#     marker=".",
#     markersize="10",
#     label="True error PR1",
# )
#
# ax1.plot(
#     np.log2(1/np.array(mesh_sizes)),
#     np.log2(np.array(errors["keilegavlen"]["majorant"])),
#     linewidth=3,
#     linestyle="-",
#     color="blue",
#     marker=".",
#     markersize="10",
#     label="Majorant PR2",
# )
#
# ax1.plot(
#     np.log2(1/np.array(mesh_sizes)),
#     np.log2(np.array(errors["keilegavlen"]["true_error"])),
#     linewidth=2,
#     linestyle="--",
#     color="blue",
#     marker=".",
#     markersize="10",
#     label="True error PR2",
# )
#
# # ax1.plot(
# #     np.log2(1/np.array(mesh_sizes)),
# #     np.log2(np.array(errors["vohralik"]["majorant"])),
# #     linewidth=3,
# #     linestyle="-",
# #     color="green",
# #     marker=".",
# #     markersize="10",
# #     label="Majorant PR3",
# # )
# #
# # ax1.plot(
# #     np.log2(1/np.array(mesh_sizes)),
# #     np.log2(np.array(errors["vohralik"]["true_error"])),
# #     linewidth=2,
# #     linestyle="--",
# #     color="green",
# #     marker=".",
# #     markersize="10",
# #     label="True error PR3",
# # )
#
# # # Plot reference line
# # x1 = 0
# # y1 = 2
# # y2 = -2
# # x2 = x1 - y2 + y1
# # ax1.plot(
# #     [x1, x2],
# #     [y1, y2],
# #     linewidth=3,
# #     linestyle="-",
# #     color="black",
# #     label="Linear"
# # )
#
# ax1.set_xlabel(r"$\mathrm{log_2}\left(1/h\right)$")
# ax1.set_ylabel(r"$\mathrm{log_2\left(error\right)}$")
# ax1.legend()
#
# ax2.plot(
#     np.log2(1/np.array(mesh_sizes)),
#     np.array(errors["cochez"]["i_eff"]),
#     linewidth=3,
#     linestyle="-",
#     color="red",
#     marker=".",
#     markersize="10",
#     label="PR1",
# )
#
# ax2.plot(
#     np.log2(1/np.array(mesh_sizes)),
#     np.array(errors["keilegavlen"]["i_eff"]),
#     linewidth=3,
#     linestyle="-",
#     color="blue",
#     marker=".",
#     markersize="10",
#     label="PR2",
# )
#
# # ax2.plot(
# #     np.log2(1/np.array(mesh_sizes)),
# #     np.array(errors["vohralik"]["i_eff"]),
# #     linewidth=3,
# #     linestyle="-",
# #     color="green",
# #     marker=".",
# #     markersize="10",
# #     label="PR3",
# # )
#
# ax2.set_xlabel(r"$\mathrm{log_2}\left(1/h\right)$")
# ax2.set_ylabel(r"$\mathrm{Efficiency~index}$")
# ax2.legend()
#
# plt.tight_layout()
# plt.savefig("unfractured.pdf")
# plt.show()
