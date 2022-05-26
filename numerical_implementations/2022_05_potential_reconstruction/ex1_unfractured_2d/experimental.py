import porepy as pp
import numpy as np
import mdestimates as mde
import scipy.sparse as sps

from analytical import ExactSolution
from true_errors import TrueError

#%% Create a grid
mesh_size = 0.5
domain = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
network_2d = pp.FractureNetwork2d(None, None, domain)
mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}

gb = network_2d.mesh(mesh_args)
g = gb.grids_of_dimension(2)[0]
d = gb.node_props(g)
#pp.plot_grid(g, alpha=0.1, plot_2d=True, figsize=(5, 5))

#%% Declare keywords
pp.set_state(d)
parameter_keyword = "flow"
subdomain_variable = "pressure"
flux_variable = "flux"
subdomain_operator_keyword = "diffusion"

#%% Assign data
exact = ExactSolution(gb)
integrated_sources = exact.integrate_f(degree=10)
bc_faces = g.get_boundary_faces()
bc_type = bc_faces.size * ["dir"]
bc = pp.BoundaryCondition(g, bc_faces, bc_type)
bc_values = exact.dir_bc_values()
permeability = pp.SecondOrderTensor(np.ones(g.num_cells))

parameters = {
    "bc": bc,
    "bc_values": bc_values,
    "source": integrated_sources,
    "second_order_tensor": permeability,
    "ambient_dimension": gb.dim_max(),
}
pp.initialize_data(g, d, "flow", parameters)

#%% Discretize the model
subdomain_discretization = pp.RT0(keyword=parameter_keyword)
source_discretization = pp.DualScalarSource(keyword=parameter_keyword)

d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 1}}
d[pp.DISCRETIZATION] = {subdomain_variable: {
            subdomain_operator_keyword: subdomain_discretization,
            "source": source_discretization
        }
    }

#%% Assemble and solve
assembler = pp.Assembler(gb)
assembler.discretize()
A, b = assembler.assemble_matrix_rhs()
sol = sps.linalg.spsolve(A, b)
assembler.distribute_variable(sol)

# Overwrite d[pp.STATE][subdomain_variable] to be consistent with FEM
for g, d in gb:
    discr = d[pp.DISCRETIZATION][subdomain_variable][
        subdomain_operator_keyword]
    pressure = discr.extract_pressure(g, d[pp.STATE][subdomain_variable],
                                      d).copy()
    flux = discr.extract_flux(g, d[pp.STATE][subdomain_variable], d).copy()
    d[pp.STATE][subdomain_variable] = pressure
    d[pp.STATE][flux_variable] = flux

#%% Estimate errors
source_list = [exact.f("fun")]
estimates = mde.ErrorEstimate(
    gb,
    source_list=source_list,
    p_recon_method="keilegavlen"
)
estimates.estimate_error()
estimates.transfer_error_to_state()
estimates.print_summary()

majorant = estimates.get_majorant()
#print(majorant)

#Plot errors
# pp.plot_grid(g, d[pp.STATE]["diffusive_error"], plot_2d=True,
#              title="Diffusive error squared")
#
# pp.plot_grid(g, d[pp.STATE]["residual_error"], plot_2d=True,
#              title="Residual error squared")


# %% Check true errors
te = TrueError(estimates)
true_error = te.true_error()
i_eff = majorant / true_error
print("Method: ", estimates.p_recon_method)
print(f"Majorant: {majorant}")
print(f"True error: {true_error}")
print(f"Efficiency index: {i_eff}")

#%%
p_cc = d[pp.STATE][estimates.p_name]

# Retrieving topological data
nc = g.num_cells
nf = g.num_faces
nn = g.num_nodes

# Perform reconstruction
cell_nodes = g.cell_nodes()
cell_node_volumes = cell_nodes * sps.dia_matrix((g.cell_volumes, 0), (nc, nc))
sum_cell_nodes = cell_node_volumes * np.ones(nc)
cell_nodes_scaled = (
        sps.dia_matrix((1.0 / sum_cell_nodes, 0), (nn, nn)) * cell_node_volumes
)

cell_nodes_pressure = cell_nodes * sps.dia_matrix((p_cc, 0), (nc, nc))

numerator = cell_nodes_scaled.multiply(cell_nodes_pressure)
numerator = np.array(numerator.sum(axis=1)).flatten()

denominator = np.array(cell_nodes_scaled.sum(axis=1)).flatten()

node_pressure = numerator / denominator

# Treatment of boundary conditions
bc = d[pp.PARAMETERS][estimates.kw]["bc"]
bc_values = d[pp.PARAMETERS][estimates.kw]["bc_values"]
external_dirichlet_boundary = np.logical_and(
    bc.is_dir, g.tags["domain_boundary_faces"]
)
face_vec = np.zeros(nf)
face_vec[external_dirichlet_boundary] = 1
num_dir_face_of_node = g.face_nodes * face_vec
is_dir_node = num_dir_face_of_node > 0
face_vec *= 0
face_vec[external_dirichlet_boundary] = bc_values[external_dirichlet_boundary]
node_val_dir = g.face_nodes * face_vec
node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
node_pressure[is_dir_node] = node_val_dir[is_dir_node]

# Save in the dictionary
d[estimates.estimates_kw]["node_pressure"] = node_pressure

# Export Lagrangian nodes and coordinates
cell_nodes_map, _, _ = sps.find(g.cell_nodes())
nodes_cell = cell_nodes_map.reshape(np.array([g.num_cells, g.dim + 1]))
point_val = node_pressure[nodes_cell]
point_coo = g_rot.nodes[:, nodes_cell]

