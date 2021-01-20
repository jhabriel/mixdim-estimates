from time import time
import helpers
import porepy as pp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import itertools
import mdestimates as mde

def make_grid(num):
    """
    Creates mixed-dimensional grid

    Parameters
    ----------
    num : Integer
        0 for coarse, 1 for intermediate, and 2 for fine.

    Returns
    -------
    pp.Griddbucket
        Grid bucket corresponding to the desired option.

    """

    if num == 0:
        fn = "geiger_geo/mesh500.geo"
    elif num == 1:
        fn = "geiger_geo/mesh4k.geo"
    elif num == 2:
        fn = "geiger_geo/mesh32k.geo"

    return pp.fracture_importer.dfm_from_gmsh(fn, 3)

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
        return np.zeros(g.num_cells, dtype=np.bool)

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
            bc = pp.BoundaryCondition(g, b_faces, labels)

            f_faces = b_faces[b_inflow]
            bc_val[f_faces] = -aperture * g.face_areas[f_faces]
            bc_val[b_faces[b_outflow]] = 1

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
numerical_methods = ["TPFA", "MPFA", "RT0", "MVEM"]
mesh_resolutions = ["coarse", "intermediate", "fine"]
subdomain_variable = "pressure"
subdomain_operator_keyword = "diffusive"
flux_variable = "flux"
edge_variable = "mortar_flux"
    
#%% Initialize dictionary
out = {k: {} for k in numerical_methods}
for i in itertools.product(numerical_methods, mesh_resolutions):
    out[i[0]][i[1]] = {
        "error_node_3d": [ ],
        "error_node_2d": [ ],
        "error_node_1d": [ ],
        "error_edge_2d": [ ],
        "error_edge_1d": [ ],
        "error_edge_0d": [ ],
        "error_global": [ ]
        }

#%% Generate grid bucket
folder = "geiger_3d"
conductive = True # Benchmark for the case of conductive domains
for i in itertools.product(numerical_methods, mesh_resolutions):
    
    # Print simulation info in the console
    print("Solving with", i[0], "for", i[1], "mesh size.")

    msg = 'Mesh resolution should be either "coarse", '
    msg += '"intermediate", or "fine".'

    # Create grid bucket
    tic = time()
    if i[1] == "coarse":
        gb = make_grid(0)
    elif i[1] == "intermediate":
        gb = make_grid(1)
    elif i[1] == "fine":
        gb = make_grid(2)    
    else:
        raise ValueError(msg)
    
    print(f"Grid construction done. Time {time() - tic}")  
    
    #%% Solve the problem
    set_parameters_conductive(gb)
        
    if i[0] == "TPFA":
        assembler, block_info = helpers.setup_flow_assembler(gb, pp.Tpfa("flow"))
    elif i[0] == "MPFA":
        assembler, block_info = helpers.setup_flow_assembler(gb, pp.Mpfa("flow"))
    elif i[0] == "RT0":
         assembler, block_info = helpers.setup_flow_assembler(gb, pp.RT0("flow"))
    elif i[0] == "MVEM":
         assembler, block_info = helpers.setup_flow_assembler(gb, pp.MVEM("flow"))
    else:
         raise ValueError('Numerical method should be either "TPFA", "MPFA", "RT0", or "MVEM".')
   
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
    estimates = mde.ErrorEstimate(gb, 
                                 lam_name=edge_variable, 
                                 sd_operator_name=subdomain_operator_keyword,
                                 )
    estimates.estimate_error()
    estimates.transfer_error_to_state()
    scaled_majorant = estimates.get_scaled_majorant()
    print(f"Errors succesfully estimated. Time {time() - tic}")
    estimates.print_summary(scaled=True)
    print("\n")
    
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
            #perm = d[pp.PARAMETERS]["flow"]["second_order_tensor"].values
            #perm = perm[0][0]
            #error_node_3d += ( (1/perm) * d[pp.STATE]["diffusive_error"]).sum() ** 0.5
            error_node_3d += estimates.get_scaled_local_errors(g, d)
            #d[pp.STATE]['scaled_error'] = (1/perm) * d[pp.STATE]["diffusive_error"]
        elif g.dim == 2:
            #perm = d[pp.PARAMETERS]["flow"]["second_order_tensor"].values
            #perm = perm[0][0]
            #error_node_2d += ( (1/perm) * d[pp.STATE]["diffusive_error"]).sum() ** 0.5
            error_node_2d += estimates.get_scaled_local_errors(g, d)
            #d[pp.STATE]['scaled_error'] = (1/perm) * d[pp.STATE]["diffusive_error"]
        elif g.dim == 1:
            #perm = d[pp.PARAMETERS]["flow"]["second_order_tensor"].values
            #perm = perm[0][0]
            #error_node_1d += ( (1/perm) * d[pp.STATE]["diffusive_error"]).sum() ** 0.5
            error_node_1d += estimates.get_scaled_local_errors(g, d)
            #d[pp.STATE]['scaled_error'] = (1/perm) * d[pp.STATE]["diffusive_error"]
        else:
            continue
    
    # Get interface errors
    for e, d_e in gb.edges():
        mg = d_e['mortar_grid']
        if mg.dim == 2:
            #knormal = d[pp.PARAMETERS]["flow"]["normal_diffusivity"]
            #error_edge_2d += ( (1/knormal) * d[pp.STATE]["diffusive_error"]).sum() ** 0.5
            error_edge_2d += estimates.get_scaled_local_errors(mg, d_e)
            #d[pp.STATE]['scaled_error'] = (1/knormal) * d[pp.STATE]["diffusive_error"]
        elif mg.dim == 1:
            #knormal = d[pp.PARAMETERS]["flow"]["normal_diffusivity"]
            #error_edge_1d += ( (1/knormal) * d[pp.STATE]["diffusive_error"]).sum() ** 0.5
            error_edge_1d += estimates.get_scaled_local_errors(mg, d_e)
            #d[pp.STATE]['scaled_error'] = (1/knormal) * d[pp.STATE]["diffusive_error"]
        elif mg.dim == 0:
            #knormal = d[pp.PARAMETERS]["flow"]["normal_diffusivity"]
            #error_edge_0d += ( (1/knormal) * d[pp.STATE]["diffusive_error"]).sum() ** 0.5
            error_edge_0d += estimates.get_scaled_local_errors(mg, d_e)
            #d[pp.STATE]['scaled_error'] = (1/knormal) * d[pp.STATE]["diffusive_error"]
    
    # # Get global errors
    # error_global = estimates.compute_global_error() ** 0.5
    # scaling_factors = [ ]
    # for g, d in gb:
    #     V = g.cell_volumes
    #     if g.dim != 0:
    #         perm = d[pp.PARAMETERS]["flow"]["second_order_tensor"].values
    #         perm = perm[0][0]
    #         scaling_factors.append(np.max(perm))
    # for e, d in gb.edges():
    #     mg = d['mortar_grid']
    #     V = mg.cell_volumes
    #     k_norm = d[pp.PARAMETERS]["flow"]['normal_diffusivity']
    #     scaling_factors.append(np.max(k_norm))
    
    # scale_factor = np.max(scaling_factors)    
    # error_global_scaled = error_global / np.sqrt(scale_factor)
    
    # Populate dictionary with proper fields
    out[i[0]][i[1]]["error_node_3d"] = error_node_3d
    out[i[0]][i[1]]["error_node_2d"] = error_node_2d
    out[i[0]][i[1]]["error_node_1d"] = error_node_1d
    out[i[0]][i[1]]["error_edge_2d"] = error_edge_2d
    out[i[0]][i[1]]["error_edge_1d"] = error_edge_1d
    out[i[0]][i[1]]["error_edge_0d"] = error_edge_0d  
    out[i[0]][i[1]]["scaled_majorant"] = scaled_majorant

    # # Print info in the console
    # print("SUMMARY OF ERRORS")
    # print('Error node 3D:', error_node_3d)
    # print('Error node 2D:', error_node_2d)
    # print('Error node 1D:', error_node_1d)
    # print('Error edge 2D:', error_edge_2d)
    # print('Error edge 1D:', error_edge_1d)
    # print('Error edge 0D:', error_edge_0d)
    # print('Scaled global error:', error_global_scaled)
    # print('Global error:', error_global)
    # print('\n')

#%% Export
# Permutations
rows = len(numerical_methods) * len(mesh_resolutions)

# Intialize lists
numerical_method_name = [ ]
mesh_resolution_name = [ ]
col_3d_node = [ ]
col_2d_node = [ ]
col_1d_node = [ ]
col_2d_edge = [ ]
col_1d_edge = [ ]
col_0d_edge = [ ]
col_scaled_majorant = [ ]

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
    col_scaled_majorant.append(out[i[0]][i[1]]["scaled_majorant"])

# Prepare for exporting
export = np.zeros(rows, 
              dtype=[('var1', 'U6'), ('var2', 'U6'), 
                      ('var3', float), ('var4', float), 
                      ('var5', float), ('var6', float), 
                      ('var7', float), ('var8', float),
                      ('var9', float),
                      ])

export['var1'] = numerical_method_name
export['var2'] = mesh_resolution_name
export['var3'] = col_3d_node
export['var4'] = col_2d_node
export['var5'] = col_1d_node
export['var6'] = col_2d_edge
export['var7'] = col_1d_edge
export['var8'] = col_0d_edge
export['var9'] = col_scaled_majorant

# Header
header = "num_method mesh_size eta_omega_3d eta_omega_2d eta_omega_1d" 
header += " eta_gamma_2d eta_gamma_1d eta_gamma_0d scaled_majorant"

# Write into txt
np.savetxt('convergence_geiger.txt', 
            export,
            delimiter=',',
            fmt="%4s %8s %2.2e %2.2e %2.2e %2.2e %2.2e %2.2e %2.2e",
            header=header
            )

#%% Export to LaTeX

# Permutations
rows = len(numerical_methods) * len(mesh_resolutions)

# Intialize lists
ampersend = []
for i in range(rows): ampersend.append('&')
numerical_method_name = [ ]
mesh_resolution_name = [ ]
col_3d_node = [ ]
col_2d_node = [ ]
col_1d_node = [ ]
col_2d_edge = [ ]
col_1d_edge = [ ]
col_0d_edge = [ ]
col_major = [ ]

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
    col_scaled_majorant.append(out[i[0]][i[1]]["majorant_scaled"])

# Prepare for exporting
export = np.zeros(rows, 
              dtype=[('var1', 'U6'), ('var2', 'U6'), 
                      ('var3', float), ('amp1', 'U6'), 
                      ('var4', float), ('amp2', 'U6'),
                      ('var5', float), ('amp3', 'U6'),
                      ('var6', float), ('amp4', 'U6'),
                      ('var7', float), ('amp5', 'U6'),
                      ('var8', float), ('amp6', 'U6'),
                      ('var9', float), 
                      ])

export['var1'] = numerical_method_name
export['var2'] = mesh_resolution_name
export['var3'] = col_3d_node
export['amp1'] = ampersend
export['var4'] = col_2d_node
export['amp2'] = ampersend
export['var5'] = col_1d_node
export['amp3'] = ampersend
export['var6'] = col_2d_edge
export['amp4'] = ampersend
export['var7'] = col_1d_edge
export['amp5'] = ampersend
export['var8'] = col_0d_edge
export['amp6'] = ampersend
export['var9'] = col_scaled_majorant

# Format string
fmt = "%4s %8s %2.2e %1s %2.2e %1s %2.2e %1s %2.2e %1s %2.2e "
fmt += "%1s %2.2e %1s %2.2e"

# Header
header = "num_method mesh_size eta_omega_3d & eta_omega_2d & eta_omega_1d & "
heaader =+ "eta_gamma_2d & eta_gamma_1d & eta_gamma_0d & scaled_majorant"

# Write into txt
np.savetxt('convergence_geiger_tex.txt', 
            export,
            delimiter=',',
            fmt=fmt,
            header=header
            )

#%% Exporting to Paraview
g_3d = gb.grids_of_dimension(3)[0]
tol = 1e-8
b_faces = g_3d.tags["domain_boundary_faces"].nonzero()[0]
bc_val = np.zeros(g_3d.num_faces)
if b_faces.size != 0:

    b_face_centers = g_3d.face_centers[:, b_faces]
    b_inflow = np.logical_and.reduce(
        tuple(b_face_centers[i, :] < 0.25 - tol for i in range(3))
    )
    b_outflow = np.logical_and.reduce(
        tuple(b_face_centers[i, :] > 0.875 + tol for i in range(3))
    )

cells_inflow = np.unique(sps.find(g_3d.cell_faces[b_faces[b_inflow]])[1])
cells_outflow  = np.unique(sps.find(g_3d.cell_faces[b_faces[b_outflow]])[1])
cell_boundaries = np.zeros(g_3d.num_cells)
cell_boundaries[cells_inflow] = -2
cell_boundaries[cells_outflow] = -1

for g, d in gb:
    if g.dim == gb.dim_max():
        d[pp.STATE]['cell_bound'] = cell_boundaries
    else:
        d[pp.STATE]['cell_bound'] = np.zeros(g.num_cells)

exporter = pp.Exporter(gb, "fine", folder_name="out")
exporter.write_vtu(['pressure', 'diffusive_error', 'scaled_error', "cell_bound"])




















