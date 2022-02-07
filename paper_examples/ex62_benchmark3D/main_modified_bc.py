from time import time
import helpers
import porepy as pp
import numpy as np
import scipy.sparse.linalg as spla
import itertools
import mdestimates as mde
import pickle


def low_zones(g):
    """
    Returns indices corresponding to the lower zones of the domain

    Parameters
    ----------
    g : pp.Grid
        Grid.

    Returns
    -------
    bool
        Containting the indices.

    """

    if g.dim < 3:
        return np.zeros(g.num_cells, dtype=bool)

    zone_0 = np.logical_and(g.cell_centers[0, :] > 0.5, g.cell_centers[1, :] < 0.5)

    zone_1 = np.logical_and.reduce(
        tuple(
            [
                g.cell_centers[0, :] > 0.75,
                g.cell_centers[1, :] > 0.5,
                g.cell_centers[1, :] < 0.75,
                g.cell_centers[2, :] > 0.5,
            ]
        )
    )

    zone_2 = np.logical_and.reduce(
        tuple(
            [
                g.cell_centers[0, :] > 0.625,
                g.cell_centers[0, :] < 0.75,
                g.cell_centers[1, :] > 0.5,
                g.cell_centers[1, :] < 0.625,
                g.cell_centers[2, :] > 0.5,
                g.cell_centers[2, :] < 0.75,
            ]
        )
    )

    return np.logical_or.reduce(tuple([zone_0, zone_1, zone_2]))


def set_parameters_conductive(gb):
    """
    Sets parameters for the benchmark problem

    Parameters
    ----------
    gb : Gridbucket

    Returns
    -------
    None.

    """

    data = {"km": 1, "km_low": 1e-1, "kf": 1e4, "aperture": 1e-4}

    tol = 1e-8

    for g, d in gb:
        d["is_tangential"] = True
        d["low_zones"] = low_zones(g)
        d["Aavatsmark_transmissibilities"] = True

        unity = np.ones(g.num_cells)
        empty = np.empty(0)

        if g.dim == 2:
            d["frac_num"] = g.frac_num * unity
        else:
            d["frac_num"] = -1 * unity

        # set the permeability
        if g.dim == 3:
            kxx = data["km"] * unity
            kxx[d["low_zones"]] = data["km_low"]
            perm = pp.SecondOrderTensor(kxx=kxx)

        elif g.dim == 2:
            kxx = data["kf"] * unity
            perm = pp.SecondOrderTensor(kxx=kxx)
        else:  # dim == 1
            kxx = data["kf"] * unity
            perm = pp.SecondOrderTensor(kxx=kxx)

        # Assign apertures
        aperture = np.power(data["aperture"], 3 - g.dim)

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc_val = np.zeros(g.num_faces)

        if b_faces.size != 0:

            b_face_centers = g.face_centers[:, b_faces]
            b_inflow = np.logical_and.reduce(
                tuple(b_face_centers[i, :] < 0.25 - tol for i in range(3))
            )
            b_outflow = np.logical_and.reduce(
                tuple(b_face_centers[i, :] > 0.875 + tol for i in range(3))
            )

            labels = np.array(["neu"] * b_faces.size)
            labels[b_outflow] = "dir"
            labels[b_inflow] = "dir"
            bc = pp.BoundaryCondition(g, b_faces, labels)

            outflow_faces = b_faces[b_outflow]
            inflow_faces = b_faces[b_inflow]
            bc_val[outflow_faces] = 0
            bc_val[inflow_faces] = 1

        else:
            bc = pp.BoundaryCondition(g, empty, empty)

        specified_parameters_f = {
            "second_order_tensor": perm,
            "aperture": aperture * unity,
            "bc": bc,
            "bc_values": bc_val,
        }
        pp.initialize_default_data(g, d, "flow", specified_parameters_f)

    # Assign coupling permeability, the aperture is read from the lower dimensional grid
    for _, d in gb.edges():
        mg = d["mortar_grid"]
        kn = 2 * data["kf"] * np.ones(mg.num_cells) / data["aperture"]
        d[pp.PARAMETERS] = pp.Parameters(
            mg, ["flow", "transport"], [{"normal_diffusivity": kn}, {}]
        )
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}, "transport": {}}

    return None


#%% Input data
numerical_methods = ["RT0", "MVEM", "MPFA", "TPFA"]
mesh_resolutions = ["coarse", "intermediate", "fine"]
subdomain_variable = "pressure"
subdomain_operator_keyword = "diffusive"
flux_variable = "flux"
edge_variable = "mortar_flux"

#%% Initialize dictionary
out = {k: {} for k in numerical_methods}
for i in itertools.product(numerical_methods, mesh_resolutions):
    out[i[0]][i[1]] = {
        "error_node_3d": [],
        "error_node_2d": [],
        "error_node_1d": [],
        "error_edge_2d": [],
        "error_edge_1d": [],
        "error_edge_0d": [],
        "majorant_pressure": [],
        "majorant_velocity": [],
        "majorant_combined": [],
    }

#%% Import grid buckets
file_to_read = open("grids.pkl", "rb")
grid_buckets = pickle.load(file_to_read)
folder = "geiger_3d"
conductive = True  # Benchmark for the case of conductive domains

for i in itertools.product(numerical_methods, mesh_resolutions):

    # Print simulation info in the console
    print("Solving with", i[0], "for", i[1], "mesh size.")

    msg = 'Mesh resolution should be either "coarse", '
    msg += '"intermediate", or "fine".'

    # Create grid bucket
    tic = time()
    if i[1] == "coarse":
        gb = grid_buckets["coarse"]
    elif i[1] == "intermediate":
        gb = grid_buckets["intermediate"]
    elif i[1] == "fine":
        gb = grid_buckets["fine"]
    else:
        raise ValueError(msg)
    print(f"3D cells: {gb.grids_of_dimension(3)[0].num_cells}")
    print(f"Grid construction done. Time {time() - tic}")


    #%% Solve the problem
    set_parameters_conductive(gb)

    if i[0] == "RT0":
        assembler, block_info = helpers.setup_flow_assembler(gb, pp.RT0("flow"))
    elif i[0] == "MVEM":
        assembler, block_info = helpers.setup_flow_assembler(gb, pp.MVEM("flow"))
    elif i[0] == "MPFA":
        assembler, block_info = helpers.setup_flow_assembler(gb, pp.Mpfa("flow"))
    elif i[0] == "TPFA":
        assembler, block_info = helpers.setup_flow_assembler(gb, pp.Tpfa("flow"))
    else:
        raise ValueError(
            'Numerical method should be either "RT0", "MVEM", "MPFA", or "TPFA".'
        )

    tic = time()
    assembler.discretize()
    print(f"Discretization finished. Time {time() - tic}")
    tic = time()
    A, b = assembler.assemble_matrix_rhs()
    x = spla.spsolve(A, b)
    assembler.distribute_variable(x)

    #%% Overwrite d[pp.STATE][subdomain_variable] to be consistent with FEM methods
    for g, d in gb:
        discr = d[pp.DISCRETIZATION][subdomain_variable][subdomain_operator_keyword]
        pressure = discr.extract_pressure(g, d[pp.STATE][subdomain_variable], d).copy()
        flux = discr.extract_flux(g, d[pp.STATE][subdomain_variable], d).copy()
        d[pp.STATE][subdomain_variable] = pressure
        d[pp.STATE][flux_variable] = flux

    print(f"Problem succesfully solved. Time {time() - tic}")

    #%% Estimate errors
    tic = time()
    estimates = mde.ErrorEstimate(
        gb, lam_name=edge_variable, sd_operator_name=subdomain_operator_keyword
    )
    estimates.estimate_error()
    estimates.transfer_error_to_state()
    scaled_majorant = estimates.get_scaled_majorant()
    print(f"Errors succesfully estimated. Time {time() - tic}")
    estimates.print_summary(scaled=True)
    print("\n")

    if i[0] == "RT0" and i[1] == "fine":
        paraview = pp.Exporter(gb, "rt0_fine", folder_name="out")
        paraview.write_vtu(["pressure", "diffusive_error"])

    #%% Compute errors
    error_node_3d = 0
    error_node_2d = 0
    error_node_1d = 0
    error_edge_2d = 0
    error_edge_1d = 0
    error_edge_0d = 0

    # Get subdomain errors
    for g, d in gb:
        if g.dim == 3:
            error_node_3d += estimates.get_scaled_local_errors(g, d)
        elif g.dim == 2:
            error_node_2d += estimates.get_scaled_local_errors(g, d)
        elif g.dim == 1:
            error_node_1d += estimates.get_scaled_local_errors(g, d)
        else:
            continue

    # Get interface errors
    for e, d_e in gb.edges():
        mg = d_e["mortar_grid"]
        if mg.dim == 2:
            error_edge_2d += estimates.get_scaled_local_errors(mg, d_e)
        elif mg.dim == 1:
            error_edge_1d += estimates.get_scaled_local_errors(mg, d_e)
        elif mg.dim == 0:
            error_edge_0d += estimates.get_scaled_local_errors(mg, d_e)

    # Populate dictionary with proper fields
    out[i[0]][i[1]]["error_node_3d"] = error_node_3d
    out[i[0]][i[1]]["error_node_2d"] = error_node_2d
    out[i[0]][i[1]]["error_node_1d"] = error_node_1d
    out[i[0]][i[1]]["error_edge_2d"] = error_edge_2d
    out[i[0]][i[1]]["error_edge_1d"] = error_edge_1d
    out[i[0]][i[1]]["error_edge_0d"] = error_edge_0d
    out[i[0]][i[1]]["majorant_pressure"] = scaled_majorant
    out[i[0]][i[1]]["majorant_velocity"] = scaled_majorant
    out[i[0]][i[1]]["majorant_combined"] = 2 * scaled_majorant

#%% Export
# Permutations
rows = len(numerical_methods) * len(mesh_resolutions)

# Initialize lists
numerical_method_name = []
mesh_resolution_name = []
col_3d_node = []
col_2d_node = []
col_1d_node = []
col_2d_edge = []
col_1d_edge = []
col_0d_edge = []
col_majorant_pressure = []
col_majorant_velocity = []
col_majorant_combined = []

# Populate lists
for i in itertools.product(numerical_methods, mesh_resolutions):
    numerical_method_name.append(i[0])
    mesh_resolution_name.append(i[1])
    col_3d_node.append(out[i[0]][i[1]]["error_node_3d"])
    col_2d_node.append(out[i[0]][i[1]]["error_node_2d"])
    col_1d_node.append(out[i[0]][i[1]]["error_node_1d"])
    col_2d_edge.append(out[i[0]][i[1]]["error_edge_2d"])
    col_1d_edge.append(out[i[0]][i[1]]["error_edge_1d"])
    col_0d_edge.append(out[i[0]][i[1]]["error_edge_0d"])
    col_majorant_pressure.append(out[i[0]][i[1]]["majorant_pressure"])
    col_majorant_velocity.append(out[i[0]][i[1]]["majorant_velocity"])
    col_majorant_combined.append(out[i[0]][i[1]]["majorant_combined"])


# Prepare for exporting
export = np.zeros(
    rows,
    dtype=[
        ("var1", "U6"),
        ("var2", "U6"),
        ("var3", float),
        ("var4", float),
        ("var5", float),
        ("var6", float),
        ("var7", float),
        ("var8", float),
        ("var9", float),
        ("var10", float),
        ("var11", float),
    ],
)

export["var1"] = numerical_method_name
export["var2"] = mesh_resolution_name
export["var3"] = col_3d_node
export["var4"] = col_2d_node
export["var5"] = col_1d_node
export["var6"] = col_2d_edge
export["var7"] = col_1d_edge
export["var8"] = col_0d_edge
export["var9"] = col_majorant_pressure
export["var10"] = col_majorant_velocity
export["var11"] = col_majorant_combined

# Header
header = "num_method mesh_size eta_omega_3d eta_omega_2d eta_omega_1d"
header += " eta_gamma_2d eta_gamma_1d eta_gamma_0d M_p M_u M_pu"

# Write into txt
np.savetxt(
    "convergence_bench3d.txt",
    export,
    delimiter=",",
    fmt="%4s %8s %2.2e %2.2e %2.2e %2.2e %2.2e %2.2e %2.2e %2.2e %2.2e",
    header=header,
)
