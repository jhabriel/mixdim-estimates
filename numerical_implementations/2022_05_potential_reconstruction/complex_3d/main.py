# Importing modules
import numpy as np
import porepy as pp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# import scipy.sparse as sps
import mdestimates as mde
import pypardiso

# %% Study parameters
# recon_methods = ["cochez", "keilegavlen", "vohralik"]
recon_methods = ["cochez", "keilegavlen", "vohralik"]
errors = {method: {} for method in recon_methods}
for method in recon_methods:
    errors[method]["majorant"] = []
mesh_sizes = [0.5, 0.3, 0.1]

for mesh_size in mesh_sizes:

    #%% Create grid bucket
    # Load fracture network: Fracture + Ghost Fractures
    network_3d = pp.fracture_importer.network_3d_from_csv("network.csv")

    # Create mesh_arg dictionary
    mesh_args = {
        "mesh_size_frac": mesh_size,
        "mesh_size_bound": mesh_size,
        "mesh_size_min": mesh_size / 10,
    }

    # Construct grid bucket
    gb = network_3d.mesh(mesh_args)

    # Get hold of grids and dictionaries
    g_3d = gb.grids_of_dimension(3)[0]
    d_3d = gb.node_props(g_3d)

    if mesh_size == 0.5:
        exporter = pp.Exporter(gb, file_name="complex_3d", folder_name="out")
        exporter.write_vtu(gb)

    #%% Obtain numerical solution
    parameter_keyword = "flow"
    max_dim = gb.dim_max()
    #  For convinience, store values of bounding box
    x_min: float = gb.bounding_box()[0][0]
    y_min: float = gb.bounding_box()[0][1]
    z_min: float = gb.bounding_box()[0][2]
    x_max: float = gb.bounding_box()[1][0]
    y_max: float = gb.bounding_box()[1][1]
    z_max: float = gb.bounding_box()[1][2]

    # Set parameters in the subdomains
    for g, d in gb:

        # Permeability assignment
        if g.dim == 3:
            kxx = np.ones(g.num_cells)
        else:
            kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx=kxx)
        specified_parameters = {'second_order_tensor': perm}

        # Add boundary conditions: Linear pressure drop from left (p=1) to right (p=0), the rest no flow.
        if g.dim == 3:
            left = np.where(np.abs(g.face_centers[1]) < 1e-5)[0]
            right = np.where(np.abs(g.face_centers[1] - y_max) < 1e-5)[0]

            # Define BoundaryCondition object
            bc_faces = np.hstack((left, right))
            bc_type = bc_faces.size * ['dir']
            bc = pp.BoundaryCondition(g, faces=bc_faces, cond=bc_type)

            # Register the assigned value
            specified_parameters['bc'] = bc

            # Also set the values - specified as vector of size g.num_faces
            bc_values = np.zeros(g.num_faces)
            bc_values[left] = 1
            bc_values[right] = 0
            specified_parameters['bc_values'] = bc_values

        # Initialize subdomain data dictionary
        pp.initialize_default_data(g, d, "flow", specified_parameters)

        # Assing interface parameters
    for e, d in gb.edges():
        # Set the normal diffusivity
        data = {"normal_diffusivity": 1}

        # Initialize edge data dictionaries
        mg = d["mortar_grid"]
        pp.initialize_data(mg, d, "flow", data)

    # Assign discretization
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

    # First, loop over the nodes
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

        # The coupling discretization links an edge discretization with variables
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
    sol = pypardiso.spsolve(A, b)
    assembler.distribute_variable(sol)

    # Overwrite d[pp.STATE][subdomain_variable] to be consistent with FEM methods
    for g, d in gb:
        discr = d[pp.DISCRETIZATION][subdomain_variable][subdomain_operator_keyword]
        pressure = discr.extract_pressure(g, d[pp.STATE][subdomain_variable], d)
        flux = discr.extract_flux(g, d[pp.STATE][subdomain_variable], d)
        d[pp.STATE][subdomain_variable] = pressure
        d[pp.STATE][flux_variable] = flux

    if mesh_size == 0.5:
        exporter.write_vtu(["pressure"])

    for method in recon_methods:

        # %% Obtain error estimates (and transfer them to d[pp.STATE])
        print(50 * "-")
        print(f"Mesh size: {mesh_size}")
        print(f"Number of cells: {gb.num_cells()}")
        print(f"Pressure Reconstruction Method: {method}")
        estimates = mde.ErrorEstimate(gb, lam_name=edge_variable, p_recon_method=method)
        estimates.estimate_error()
        estimates.transfer_error_to_state()
        majorant = estimates.get_majorant()
        errors[method]["majorant"].append(majorant)
        print(f"Majorant: {majorant}")
        print(50 * "-")

#%% Plotting
plt.rcParams.update({'font.size': 14})
tab10 = colors.ListedColormap(plt.cm.tab10.colors[:3])
fig, ax = plt.subplots()

ax.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["cochez"]["majorant"])),
    linewidth=2,
    linestyle="--",
    color=tab10.colors[0],
    marker=".",
    markersize="10",
    label=r"$\mathfrak{M}_{\mathrm{PR1}}$",
)

ax.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["keilegavlen"]["majorant"])),
    linewidth=2,
    linestyle="--",
    color=tab10.colors[1],
    marker=".",
    markersize="10",
    label=r"$\mathfrak{M}_{\mathrm{PR1}}$",
)

ax.plot(
    np.log2(1/np.array(mesh_sizes)),
    np.log2(np.array(errors["vohralik"]["majorant"])),
    linewidth=2,
    linestyle="--",
    color=tab10.colors[2],
    marker=".",
    markersize="10",
    label=r"$\mathfrak{M}_{\mathrm{PR1}}$",
)

# Plot reference line
x1 = 0
y1 = 6
y2 = 3
x2 = x1 - y2 + y1
ax.plot(
    [x1, x2],
    [y1, y2],
    linewidth=2,
    linestyle="-",
    color="black",
    label="Linear"
)

ax.set_xlabel(r"$\mathrm{log_2}\left(1/h\right)$")
ax.set_ylabel(r"$\mathrm{log_2\left(error\right)}$")
ax.legend()
#
# ax2.plot(
#     np.log2(1/np.array(mesh_sizes)),
#     np.array(errors["cochez"]["i_eff"]),
#     linewidth=2,
#     linestyle="--",
#     color=tab10.colors[0],
#     marker=".",
#     markersize="9",
#     label=r"$I_{\mathrm{PR1}}$",
# )
#
# ax2.plot(
#     np.log2(1/np.array(mesh_sizes)),
#     np.array(errors["keilegavlen"]["i_eff"]),
#     linewidth=2,
#     linestyle="--",
#     color=tab10.colors[1],
#     marker=".",
#     markersize="10",
#     label=r"$I_{\mathrm{PR2}}$",
# )
#
# ax2.plot(
#     np.log2(1/np.array(mesh_sizes)),
#     np.array(errors["vohralik"]["i_eff"]),
#     linewidth=2,
#     linestyle="--",
#     color=tab10.colors[2],
#     marker=".",
#     markersize="10",
#     label=r"$I_{\mathrm{PR3}}$",
# )
#
# ax2.set_xlabel(r"$\mathrm{log_2}\left(1/h\right)$")
# ax2.set_ylabel(r"$\mathrm{Efficiency~index}$")
# ax2.legend()
#
plt.tight_layout()
plt.savefig("complex_3d.pdf")
plt.show()