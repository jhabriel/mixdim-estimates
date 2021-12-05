import numpy as np
import porepy as pp
import scipy.sparse as sps
import sympy as sym
import quadpy as qp
import mdestimates as mde

import mdestimates.estimates_utils as utils
import model_md_poincare as mutils


#%% Create mesh
h = 1/40
gb = mutils.make_constrained_mesh(h=h)
g_2d = gb.grids_of_dimension(2)[0]
g_1d = gb.grids_of_dimension(1)[0]
d_2d = gb.node_props(g_2d)
d_1d = gb.node_props(g_1d)
d_e = gb.edge_props((g_2d, g_1d))
mg = d_e["mortar_grid"]

#%% Set states
for _, d in gb:
    pp.set_state(d)

for _, d in gb.edges():
    pp.set_state(d)

#%% Retrieve boundary and cell indices
cell_idx_list, regions_2d = mutils.get_2d_cell_indices(g_2d)
bound_idx_list = mutils.get_2d_boundary_indices(g_2d)

#%% Retrieve analytical expressions
p2d_sym_list, p2d_numpy_list, p2d_cc = mutils.get_exact_2d_pressure(g_2d)
gradp2d_sym_list, gradp2d_numpy_list, gradp2d_cc = mutils.get_exact_2d_pressure_gradient(
    g_2d, p2d_sym_list
)
u2d_sym_list, u2d_numpy_list, u2d_cc = mutils.get_exact_2d_velocity(g_2d, gradp2d_sym_list)
f2d_sym_list, f2d_numpy_list, f2d_cc = mutils.get_exact_2d_source_term(g_2d, u2d_sym_list)
bc_vals_2d = mutils.get_2d_boundary_values(g_2d, bound_idx_list, p2d_numpy_list)

#%% Compute exact source terms
integrated_f2d = mutils.integrate_source_2d(g_2d, f2d_numpy_list, cell_idx_list)
integrated_f1d = - (1/20) * 2 * g_1d.cell_volumes

#%% Set parameters
parameter_keyword = "flow"

# Loop over the nodes
for g, d in gb:
    if g.dim == gb.dim_max():
        # All Dirichlet for the bulk
        bc_faces = g.get_boundary_faces()
        bc_type = bc_faces.size * ["dir"]
        bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)
        # Specificy bc values
        bc_values = bc_vals_2d
        # Specifiy source terms
        source = integrated_f2d
        # Specified parameters dictionary
        specified_parameters = {"bc": bc, "bc_values": bc_values, "source": source}
    else:
        # Specifiy source terms
        source = integrated_f1d
        # Specified parameters dictionary
        specified_parameters = {"source": source}
    pp.initialize_default_data(g, d, parameter_keyword, specified_parameters)

# Loop over the edges
for e, d in gb.edges():
    data = {"normal_diffusivity": 1.0}
    mg = d["mortar_grid"]
    pp.initialize_data(mg, d, parameter_keyword, data)

#%% Discretization
subdomain_discretization = pp.Mpfa(keyword=parameter_keyword)
source_discretization = pp.ScalarSource(keyword=parameter_keyword)
subdomain_variable = "pressure"
flux_variable = "flux"
subdomain_operator_keyword = "diffusion"
edge_discretization = pp.RobinCoupling(
    parameter_keyword, subdomain_discretization, subdomain_discretization
)
edge_variable = "interface_flux"
coupling_operator_keyword = "interface_diffusion"

# Loop over the nodes
for g, d in gb:
    d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 0}}
    d[pp.DISCRETIZATION] = {
        subdomain_variable: {
            subdomain_operator_keyword: subdomain_discretization,
            "source": source_discretization,
        }
    }

# Loop over the edges
for e, d in gb.edges():
    # Get the grids of the neighboring subdomains
    g1, g2 = gb.nodes_of_edge(e)
    # The interface variable has one degree of freedom per cell in the mortar grid
    d[pp.PRIMARY_VARIABLES] = {edge_variable: {"cells": 1}}
    # The coupling discretization links an edge discretization with variables
    d[pp.COUPLING_DISCRETIZATION] = {
        coupling_operator_keyword: {
            g1: (subdomain_variable, subdomain_operator_keyword),
            g2: (subdomain_variable, subdomain_operator_keyword),
            e: (edge_variable, edge_discretization),
        }
    }

#%% Obtain numerical solution
assembler = pp.Assembler(gb)
assembler.discretize()
A, b = assembler.assemble_matrix_rhs()
sol = sps.linalg.spsolve(A, b)
assembler.distribute_variable(sol)

#%% Solve eigenvalue problem
A1 = gb.num_cells() * A
eig_min = np.real(sps.linalg.eigs(A1, k=1, which="SM")[0])[0]
C_p = 1/np.sqrt(eig_min)
print(f"Minimum eigvenvalue is: {eig_min}")
print(f"Poincare constant is: {C_p}")