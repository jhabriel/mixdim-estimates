import porepy as pp
import numpy as np
import mdestimates as mde
import matplotlib.pyplot as plt
import scipy.sparse as sps
import pypardiso

from analytical import ExactSolution
from true_errors import TrueError
import mdestimates.estimates_utils as utils

#%% Study parameters
recon_methods = ["cochez", "keilegavlen", "vohralik"]
mesh_sizes = [0.2, 0.15, 0.1, 0.08, 0.06, 0.05]
errors = {method: {} for method in recon_methods}
for method in recon_methods:
    errors[method]["majorant"] = []
    errors[method]["true_error"] = []
    errors[method]["i_eff"] = []


for mesh_size in mesh_sizes:
    #%% Create a grid
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
        "mesh_size_min": mesh_size,
    }

    gb = network_3d.mesh(mesh_args)
    g = gb.grids_of_dimension(3)[0]
    d = gb.node_props(g)

    #%% Declare keywords
    pp.set_state(d)
    parameter_keyword = "flow"
    subdomain_variable = "pressure"
    flux_variable = "flux"
    subdomain_operator_keyword = "diffusion"

    # %% Assign data
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

    # %% Discretize the model
    subdomain_discretization = pp.RT0(keyword=parameter_keyword)
    source_discretization = pp.DualScalarSource(keyword=parameter_keyword)

    d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 1}}
    d[pp.DISCRETIZATION] = {subdomain_variable: {
        subdomain_operator_keyword: subdomain_discretization,
        "source": source_discretization
    }
    }

    # %% Assemble and solve
    assembler = pp.Assembler(gb)
    assembler.discretize()
    A, b = assembler.assemble_matrix_rhs()
    # sol = sps.linalg.spsolve(A, b)
    sol = pypardiso.spsolve(A, b)
    assembler.distribute_variable(sol)

    # Overwrite d[pp.STATE][subdomain_variable] to be consistent with FEM
    for g, d in gb:
        discr = d[pp.DISCRETIZATION][subdomain_variable][subdomain_operator_keyword]
        pressure = discr.extract_pressure(g, d[pp.STATE][subdomain_variable], d)
        flux = discr.extract_flux(g, d[pp.STATE][subdomain_variable], d)
        d[pp.STATE][subdomain_variable] = pressure
        d[pp.STATE][flux_variable] = flux

    # %% Estimate errors
    source_list = [exact.f("fun")]
    for method in recon_methods:
        estimates = mde.ErrorEstimate(gb, source_list=source_list, p_recon_method=method)
        estimates.estimate_error()
        estimates.transfer_error_to_state()
        majorant = estimates.get_majorant()

        te = TrueError(estimates)
        true_error = te.true_error()
        i_eff = majorant / true_error

        print(f"Mesh size: {mesh_size}")
        print(f"Number of cells: {g.num_cells}")
        print(f"Pressure Reconstruction Method: {estimates.p_recon_method}")
        print(f"Majorant: {majorant}")
        print(f"True error: {true_error}")
        print(f"Efficiency index: {i_eff}")
        print(50 * "-")

        errors[method]["majorant"].append(majorant)
        errors[method]["true_error"].append(true_error)
        errors[method]["i_eff"].append(i_eff)

# #%% Exporting to Paraview
# reconstructed_p1 = te.reconstructed_p_p1()
# reconstructed_p2 = te.reconstructed_p_p2()
# postprocessed_p2 = te.postprocessed_p_p2()
# d[pp.STATE]["reconstructed_p_p2"] = reconstructed_p2
# d[pp.STATE]["postprocessed_p_p2"] = postprocessed_p2
# d[pp.STATE]["exact_p"] = te.p("cc")
#
# exporter = pp.Exporter(gb, file_name="unfractured_3d", folder_name="out")
# exporter.write_vtu([
#     "pressure",
#     "reconstructed_p_p2",
#     "postprocessed_p_p2",
#     "exact_p",
#     "diffusive_error",
#     "residual_error"
# ])
#

#%% Plotting
plt.rcParams.update({'font.size': 13})
fig, (ax1, ax2) = plt.subplots(
    nrows=1,
    ncols=2,
    gridspec_kw={'width_ratios': [2, 1]},
    figsize=(11, 5)
)

ax1.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["cochez"]["majorant"])),
    linewidth=3,
    linestyle="-",
    color="red",
    marker=".",
    markersize="10",
    label="Majorant PR1",
)

ax1.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["cochez"]["true_error"])),
    linewidth=2,
    linestyle="--",
    color="red",
    marker=".",
    markersize="10",
    label="True error PR1",
)

ax1.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["keilegavlen"]["majorant"])),
    linewidth=3,
    linestyle="-",
    color="blue",
    marker=".",
    markersize="10",
    label="Majorant PR2",
)

ax1.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["keilegavlen"]["true_error"])),
    linewidth=2,
    linestyle="--",
    color="blue",
    marker=".",
    markersize="10",
    label="True error PR2",
)

ax1.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["vohralik"]["majorant"])),
    linewidth=3,
    linestyle="-",
    color="green",
    marker=".",
    markersize="10",
    label="Majorant PR3",
)

ax1.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["vohralik"]["true_error"])),
    linewidth=2,
    linestyle="--",
    color="green",
    marker=".",
    markersize="10",
    label="True error PR3",
)

# Plot reference line
x1 = 0
y1 = 2
y2 = -2
x2 = x1 - y2 + y1
ax1.plot(
    [x1, x2],
    [y1, y2],
    linewidth=3,
    linestyle="-",
    color="black",
    label="Linear"
)

ax1.set_xlabel(r"$\mathrm{log_2}\left(1/h\right)$")
ax1.set_ylabel(r"$\mathrm{log_2\left(error\right)}$")
ax1.legend()

ax2.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.array(errors["cochez"]["i_eff"]),
    linewidth=3,
    linestyle="-",
    color="red",
    marker=".",
    markersize="10",
    label="PR1",
)

ax2.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.array(errors["keilegavlen"]["i_eff"]),
    linewidth=3,
    linestyle="-",
    color="blue",
    marker=".",
    markersize="10",
    label="PR2",
)

ax2.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.array(errors["vohralik"]["i_eff"]),
    linewidth=3,
    linestyle="-",
    color="green",
    marker=".",
    markersize="10",
    label="PR3",
)

ax2.set_xlabel(r"$\mathrm{log_2}\left(1/h\right)$")
ax2.set_ylabel(r"$\mathrm{Efficiency~index}$")
ax2.legend()

plt.tight_layout()
plt.savefig("unfractured.pdf")
plt.show()

