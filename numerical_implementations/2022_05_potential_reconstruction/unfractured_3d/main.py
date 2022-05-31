import porepy as pp
import numpy as np
import mdestimates as mde
import matplotlib.pyplot as plt
import scipy.sparse as sps
import itertools

from analytical import ExactSolution
from true_errors import TrueError

#%% Study parameters
recon_methods = ["vohralik"]
mesh_sizes = [0.3]
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
        "mesh_size_min": mesh_size/10,
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
        print(f"Pressure Reconstruction Method: {estimates.p_recon_method}")
        print(f"Majorant: {majorant}")
        print(f"True error: {true_error}")
        print(f"Efficiency index: {i_eff}")
        print(50 * "-")

        errors[method]["majorant"].append(majorant)
        errors[method]["true_error"].append(true_error)
        errors[method]["i_eff"].append(i_eff)