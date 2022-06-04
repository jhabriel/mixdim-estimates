import porepy as pp
import numpy as np
import mdestimates as mde
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.sparse as sps

from analytical import ExactSolution
from true_errors import TrueError

#%% Study parameters
recon_methods = ["cochez", "keilegavlen", "vohralik"]
mesh_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
errors = {method: {} for method in recon_methods}
for method in recon_methods:
    errors[method]["majorant"] = []
    errors[method]["true_error"] = []
    errors[method]["i_eff"] = []

for mesh_size in mesh_sizes:

    #%% Create a grid
    domain = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
    network_2d = pp.FractureNetwork2d(None, None, domain)
    mesh_args = {"mesh_size_bound": mesh_size, "mesh_size_frac": mesh_size}

    gb = network_2d.mesh(mesh_args)
    g = gb.grids_of_dimension(2)[0]
    d = gb.node_props(g)

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
        discr = d[pp.DISCRETIZATION][subdomain_variable][subdomain_operator_keyword]
        pressure = discr.extract_pressure(g, d[pp.STATE][subdomain_variable], d)
        flux = discr.extract_flux(g, d[pp.STATE][subdomain_variable], d)
        d[pp.STATE][subdomain_variable] = pressure
        d[pp.STATE][flux_variable] = flux

    #%% Estimate errors
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
        print(f"Pressure Reconstruction Method: {estimates.p_recon_method}")
        print(f"Majorant: {majorant}")
        print(f"True error: {true_error}")
        print(f"Efficiency index: {i_eff}")
        print(50 * "-")

        errors[method]["majorant"].append(majorant)
        errors[method]["true_error"].append(true_error)
        errors[method]["i_eff"].append(i_eff)

#%% Obtain relative efficiency indices
true_errors = np.empty((len(mesh_sizes), len(recon_methods)))
for mesh_idx, _ in enumerate(mesh_sizes):
    for mthd_idx, method in enumerate(recon_methods):
        true_errors[mesh_idx, mthd_idx] = errors[method]["true_error"][mesh_idx]
min_true_errors = np.min(true_errors, axis=1)

for method in recon_methods:
    errors[method]["rel_i_eff"] = list(errors[method]["majorant"] / min_true_errors)

#%% Plotting
#%% Plotting
plt.rcParams.update({'font.size': 14})
tab10 = colors.ListedColormap(plt.cm.tab10.colors[:6])
fig, (ax1, ax2) = plt.subplots(
    nrows=1,
    ncols=2,
    gridspec_kw={'width_ratios': [3, 2]},
    figsize=(11, 5)
)

ax1.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["cochez"]["majorant"])),
    linewidth=2,
    linestyle="--",
    color=tab10.colors[0],
    marker=".",
    markersize="10",
    label=r"$\mathfrak{M}_{\mathrm{PR1}}$",
)

ax1.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["cochez"]["true_error"])),
    linewidth=4,
    linestyle="-",
    alpha=0.5,
    color=tab10.colors[0],
    marker=".",
    markersize="10",
    label=r"$\mathfrak{T}_{\mathrm{PR1}}$",
)

ax1.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["keilegavlen"]["majorant"])),
    linewidth=2,
    linestyle="--",
    color=tab10.colors[1],
    marker=".",
    markersize="10",
    label=r"$\mathfrak{M}_{\mathrm{PR2}}$",
)

ax1.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["keilegavlen"]["true_error"])),
    linewidth=4,
    linestyle="-",
    color=tab10.colors[1],
    alpha=0.5,
    marker=".",
    markersize="10",
    label=r"$\mathfrak{T}_{\mathrm{PR2}}$",
)

ax1.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["vohralik"]["majorant"])),
    linewidth=2,
    linestyle="--",
    color=tab10.colors[2],
    marker=".",
    markersize="10",
    label=r"$\mathfrak{M}_{\mathrm{PR3}}$",
)

ax1.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["vohralik"]["true_error"])),
    linewidth=4,
    linestyle="-",
    color=tab10.colors[2],
    alpha=0.5,
    marker=".",
    markersize="10",
    label=r"$\mathfrak{T}_{\mathrm{PR3}}$",
)

# Plot reference line
x1 = 7.3
y1 = -13
y2 = -9
x2 = x1 - y2 + y1
ax1.plot(
    [x1, x2],
    [y1, y2],
    linewidth=2,
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
    linewidth=2,
    linestyle="--",
    color=tab10.colors[0],
    marker=".",
    markersize="9",
    label=r"$I_{\mathrm{PR1}}$",
)

ax2.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.array(errors["cochez"]["rel_i_eff"]),
    linewidth=4,
    linestyle="-",
    color=tab10.colors[0],
    alpha=0.5,
    marker=".",
    markersize="10",
    label=r"$I_{\mathrm{rel},\mathrm{PR1}}$",
)

ax2.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.array(errors["keilegavlen"]["i_eff"]),
    linewidth=2,
    linestyle="--",
    color=tab10.colors[1],
    marker=".",
    markersize="10",
    label=r"$I_{\mathrm{PR2}}$",
)

ax2.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.array(errors["keilegavlen"]["rel_i_eff"]),
    linewidth=4,
    linestyle="-",
    alpha=0.5,
    color=tab10.colors[1],
    marker=".",
    markersize="10",
    label=r"$I_{\mathrm{rel},\mathrm{PR2}}$",
)

ax2.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.array(errors["vohralik"]["i_eff"]),
    linewidth=2,
    linestyle="--",
    color=tab10.colors[2],
    marker=".",
    markersize="10",
    label=r"$I_{\mathrm{PR3}}$",
)

ax2.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.array(errors["vohralik"]["rel_i_eff"]),
    linewidth=4,
    linestyle="-",
    alpha=0.5,
    marker=".",
    markersize="10",
    color=tab10.colors[2],
    label=r"$I_{\mathrm{rel},\mathrm{PR3}}$",
)


ax2.set_xlabel(r"$\mathrm{log_2}\left(1/h\right)$")
ax2.set_ylabel(r"$\mathrm{Efficiency~index}$")
ax2.legend()

plt.tight_layout()
plt.savefig("unfractured_2d.pdf")
plt.show()

