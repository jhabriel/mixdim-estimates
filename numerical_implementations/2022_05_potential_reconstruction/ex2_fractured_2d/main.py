import numpy as np
import porepy as pp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mdestimates as mde
import pypardiso

from analytical_2d import ExactSolution2D
from true_errors_2d import TrueErrors2D

# %% Study parameters
recon_methods = ["cochez", "keilegavlen", "vohralik"]
# recon_methods = ["cochez", "keilegavlen"]
errors = {method: {} for method in recon_methods}
for method in recon_methods:
    errors[method]["majorant"] = []
    errors[method]["true_error"] = []
    errors[method]["i_eff"] = []
mesh_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]

# Loop through the mesh sizes
for mesh_size in mesh_sizes:

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

    # %% Obtain error estimates (and transfer them to d[pp.STATE])
    for method in recon_methods:
        estimates = mde.ErrorEstimate(gb, lam_name=edge_variable, p_recon_method=method)
        estimates.estimate_error()
        estimates.transfer_error_to_state()

        # %% Retrieve errors
        te = TrueErrors2D(gb=gb, estimates=estimates)

        diffusive_sq_2d = d_2d[pp.STATE]["diffusive_error"]
        diffusive_sq_1d = d_1d[pp.STATE]["diffusive_error"]
        diffusive_sq_lmortar = d_e[pp.STATE]["diffusive_error"][int(mg.num_cells / 2) :]
        diffusive_sq_rmortar = d_e[pp.STATE]["diffusive_error"][: int(mg.num_cells / 2)]
        diffusive_error = np.sqrt(
            diffusive_sq_2d.sum()
            + diffusive_sq_1d.sum()
            + diffusive_sq_lmortar.sum()
            + diffusive_sq_rmortar.sum()
        )

        residual_sq_2d = te.residual_error_2d()
        residual_sq_1d = te.residual_error_1d()
        residual_error = np.sqrt(residual_sq_2d.sum() + residual_sq_1d.sum())

        majorant = diffusive_error + residual_error
        true_error = te.pressure_error()
        i_eff = majorant / true_error

        print(50 * "-")
        print(f"Mesh size: {mesh_size}")
        print(f"Number of cells: {gb.num_cells()}")
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
x1 = 7
y1 = -8
y2 = -4.5
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
plt.savefig("fractured_2d.pdf")
plt.show()
