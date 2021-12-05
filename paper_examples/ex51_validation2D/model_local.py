import numpy as np
import porepy as pp
import scipy.sparse as sps
import mdestimates as mde

from analytical_2d import ExactSolution2D
from true_errors_2d import TrueErrors2D


def model_local(gb, method):
    """
    Runs main model for 1d/2d validation from the paper

    Parameters
    ----------
    gb : PorePy Object
        Grid bucket
    method: String
        Numerical method, e.g.: 'TPFA', 'MPFA', 'RTO', 'MVEM'

    Returns
    -------
    out: dictionary containing the relevant information

    """

    # %% Method type
    def fv(scheme):
        """
        Checks wheter a numerical method is FV or not

        Parameters
        ----------
        scheme : string
            Numerical method.

        Returns
        -------
        bool
            True if the numerical method is FV, false otherwise.

        """
        if scheme in ["mpfa", "MPFA", "tpfa", "TPFA"]:
            return True
        elif scheme in ["rt0", "RT0", "mvem", "MVEM"]:
            return False
        else:
            raise ValueError("Method unrecognized")

    # Get hold of grids and dictionaries
    g_2d = gb.grids_of_dimension(2)[0]
    g_1d = gb.grids_of_dimension(1)[0]
    h_max = gb.diameter()
    d_2d = gb.node_props(g_2d)
    d_1d = gb.node_props(g_1d)
    d_e = gb.edge_props([g_1d, g_2d])
    mg = d_e["mortar_grid"]

    # Get hold of mesh sizes
    h_1 = 0.5 / g_1d.num_cells
    h_gamma = 0.5 / (mg.num_cells / 2)

    # Mappings
    cell_faces_map, _, _ = sps.find(g_2d.cell_faces)
    cell_nodes_map, _, _ = sps.find(g_2d.cell_nodes())

    # Populate the data dictionaries with pp.STATE
    for g, d in gb:
        pp.set_state(d)

    for e, d in gb.edges():
        pp.set_state(d)

    # Retrieve true error object and exact expressions
    ex = ExactSolution2D(gb)
    # cell_idx_list = ex.cell_idx
    # bound_idx_list = ex.bc_idx

    # p2d_sym_list = ex.p2d("sym")
    # p2d_numpy_list = ex.p2d("fun")
    # p2d_cc = ex.p2d("cc")

    # gradp2d_sym_list = ex.gradp2d("sym")
    # gradp2d_numpy_list = ex.gradp2d("fun")
    # gradp2d_cc = ex.gradp2d("cc")

    # u2d_sym_list = ex.u2d("sym")
    # u2d_numpy_list = ex.u2d("fun")
    # u2d_cc = ex.u2d("cc")

    # f2d_sym_list = ex.f2d("sym")
    # f2d_numpy_list = ex.f2d("fun")
    # f2d_cc = ex.f2d("cc")

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

    # Discretize model according to the numerical method
    if method in ["mpfa", "MPFA"]:
        subdomain_discretization = pp.Mpfa(keyword=parameter_keyword)
    elif method in ["tpfa", "TPFA"]:
        subdomain_discretization = pp.Tpfa(keyword=parameter_keyword)
    elif method in ["rt0", "RT0"]:
        subdomain_discretization = pp.RT0(keyword=parameter_keyword)
    elif method in ["mvem", "MVEM"]:
        subdomain_discretization = pp.MVEM(keyword=parameter_keyword)
    else:
        raise ValueError("Method not implemented")

    # Discretize source term according to the method family
    if fv(method):
        source_discretization = pp.ScalarSource(keyword=parameter_keyword)
    else:
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
    if fv(method):  # FV methods
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {subdomain_variable: {"cells": 1, "faces": 0}}
            d[pp.DISCRETIZATION] = {
                subdomain_variable: {
                    subdomain_operator_keyword: subdomain_discretization,
                    "source": source_discretization,
                }
            }
    else:  # FEM methods
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
        # The interface variable has one degree of freedom per cell in
        # the mortar grid
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
    sol = sps.linalg.spsolve(A, b)
    assembler.distribute_variable(sol)

    # Overwrite d[pp.STATE][subdomain_variable] to be consistent with FEM
    for g, d in gb:
        discr = d[pp.DISCRETIZATION][subdomain_variable][subdomain_operator_keyword]
        pressure = discr.extract_pressure(g, d[pp.STATE][subdomain_variable], d).copy()
        flux = discr.extract_flux(g, d[pp.STATE][subdomain_variable], d).copy()
        d[pp.STATE][subdomain_variable] = pressure
        d[pp.STATE][flux_variable] = flux

    # %% Obtain error estimates (and transfer them to d[pp.STATE])
    estimates = mde.ErrorEstimate(gb, lam_name=edge_variable)
    estimates.estimate_error()
    estimates.transfer_error_to_state()

    # %% Retrieve errors
    te = TrueErrors2D(gb=gb, estimates=estimates)

    diffusive_error_squared_2d = d_2d[pp.STATE]["diffusive_error"]
    diffusive_error_squared_1d = d_1d[pp.STATE]["diffusive_error"]
    diffusive_error_squared_mortar = d_e[pp.STATE]["diffusive_error"]
    diffusive_error = (
        diffusive_error_squared_2d.sum()
        + diffusive_error_squared_1d.sum()
        + diffusive_error_squared_mortar.sum()
    ) ** 0.5

    residual_error_squared_2d = te.residual_error_2d_local_poincare()
    residual_error_squared_1d = te.residual_error_1d_local_poincare()
    residual_error = (
        residual_error_squared_2d.sum() + residual_error_squared_1d.sum()
    ) ** 0.5

    majorant_pressure = diffusive_error + residual_error
    majorant_velocity = majorant_pressure
    majorant_combined = majorant_pressure + majorant_velocity + residual_error

    # Distinguishing between subdomain and mortar errors
    bulk_error = (
        diffusive_error_squared_2d.sum() + residual_error_squared_2d.sum()
    ) ** 0.5
    fracture_error = (
        diffusive_error_squared_1d.sum() + residual_error_squared_1d.sum()
    ) ** 0.5
    mortar_error = diffusive_error_squared_mortar.sum() ** 0.5

    # %% Obtain true errors

    # Pressure error
    # pressure_error_squared_2d = te.pressure_error_squared_2d()
    # pressure_error_squared_1d = te.pressure_error_squared_1d()
    # pressure_error_squared_mortar = te.pressure_error_squared_mortar()
    true_pressure_error = te.pressure_error()

    # Velocity error
    # velocity_error_squared_2d = te.velocity_error_squared_2d()
    # velocity_error_squared_1d = te.velocity_error_squared_1d()
    # velocity_error_squared_mortar = te.velocity_error_squared_mortar()
    true_velocity_error = te.velocity_error()

    # True combined error
    true_combined_error = true_pressure_error + true_velocity_error + residual_error

    # %% Compute efficiency indices
    i_eff_p = majorant_pressure / true_pressure_error
    i_eff_u = majorant_velocity / true_velocity_error
    i_eff_pu = majorant_combined / true_combined_error

    print(50 * "-")
    print(f"Majorant pressure: {majorant_pressure}")
    print(f"Majorant velocity: {majorant_velocity}")
    print(f"Majorant combined: {majorant_combined}")
    print(f"Bulk error: {bulk_error}")
    print(f"Fracture error: {fracture_error}")
    print(f"Mortar error: {mortar_error}")
    print(f"True error (pressure): {true_pressure_error}")
    print(f"True error (velocity): {true_velocity_error}")
    print(f"True error (combined): {true_combined_error}")
    print(f"Efficiency index (pressure): {i_eff_p}")
    print(f"Efficiency index (velocity): {i_eff_u}")
    print(f"Efficiency index (combined): {i_eff_pu}")
    print(50 * "-")

    # Prepare return dictionary
    out = {}

    out["majorant_pressure"] = majorant_pressure
    out["majorant_velocity"] = majorant_velocity
    out["majorant_combined"] = majorant_combined
    out["true_pressure_error"] = true_pressure_error
    out["true_velocity_error"] = true_velocity_error
    out["true_combined_error"] = true_combined_error
    out["efficiency_pressure"] = i_eff_p
    out["efficiency_velocity"] = i_eff_u
    out["efficiency_combined"] = i_eff_pu

    out["bulk"] = {}
    out["bulk"]["mesh_size"] = h_max
    out["bulk"]["error"] = bulk_error
    out["bulk"]["diffusive_error"] = diffusive_error_squared_2d.sum() ** 0.5
    out["bulk"]["residual_error"] = residual_error_squared_2d.sum() ** 0.5

    out["frac"] = {}
    out["frac"]["mesh_size"] = h_1
    out["frac"]["error"] = fracture_error
    out["frac"]["diffusive_error"] = diffusive_error_squared_1d.sum() ** 0.5
    out["frac"]["residual_error"] = residual_error_squared_1d.sum() ** 0.5

    out["mortar"] = {}
    out["mortar"]["mesh_size"] = h_gamma
    out["mortar"]["error"] = mortar_error

    return out
