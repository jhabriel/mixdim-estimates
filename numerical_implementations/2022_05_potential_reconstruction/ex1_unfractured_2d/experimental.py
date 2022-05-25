import porepy as pp
import numpy as np
import mdestimates as mde
import scipy.sparse as sps

from analytical import ExactSolution
from true_errors import TrueError

#%% Create a grid
mesh_size = 0.05
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
estimates = mde.ErrorEstimate(gb, source_list=source_list)
estimates.estimate_error()
estimates.transfer_error_to_state()
estimates.print_summary()

majorant = estimates.get_majorant()
#print(majorant)

# Plot errors
# pp.plot_grid(g, d[pp.STATE]["diffusive_error"], plot_2d=True,
#              title="Diffusive error squared")
#
# pp.plot_grid(g, d[pp.STATE]["residual_error"], plot_2d=True,
#              title="Residual error squared")


# %% Check true errors
true_error = TrueError(estimates)