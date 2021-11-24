# Importing modules
import numpy as np
import porepy as pp
import scipy.sparse as sps
import sympy as sym
import quadpy as qp
import mdestimates as mde

import mdestimates.estimates_utils as utils
from mdestimates._velocity_reconstruction import _internal_source_term_contribution as mortar_jump

from analytical_3d import ExactSolution3D
from true_errors_3d import TrueErrors3D


def model_local(gb, method):
    """
    Runs main model for Validation 5.2 from the paper

    Parameters
    ----------
    gb : PorePy Object
        Grid bucket
    method: String
        Numerical method, e.g.: 'TPFA', 'MPFA', 'RTO', 'MVEM'

    Returns
    -------
    out: Dictionary
        Containing the relevant export data

    """

    #%% Method type
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
            raise TypeError("Method unrecognized")

    # Get hold of grids and dictionaries
    g_3d = gb.grids_of_dimension(3)[0]
    g_2d = gb.grids_of_dimension(2)[0]
    h_max = gb.diameter()
    d_3d = gb.node_props(g_3d)
    d_2d = gb.node_props(g_2d)
    d_e = gb.edge_props([g_2d, g_3d])
    mg = d_e["mortar_grid"]

    # Exact solutions
    ex = ExactSolution3D(gb)
    cell_idx_list = ex.cell_idx
    bound_idx_list = ex.bc_idx

    p3d_sym_list = ex.p3d("sym")
    p3d_numpy_list = ex.p3d("fun")
    p3d_cc = ex.p3d("cc")

    gradp3d_sym_list = ex.gradp3d("sym")
    gradp3d_numpy_list = ex.gradp3d("fun")
    gradp3d_cc = ex.gradp3d("cc")

    u3d_sym_list = ex.u3d("sym")
    u3d_numpy_list = ex.u3d("fun")
    u3d_cc = ex.u3d("cc")

    f3d_sym_list = ex.f3d("sym")
    f3d_numpy_list = ex.f3d("fun")
    f3d_cc = ex.f3d("cc")

    bc_vals_3d = ex.dir_bc_values()

    integrated_f3d = ex.integrate_f3d()
    integrated_f2d = ex.integrate_f2d()

    #%% Obtain numerical solution
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
            bc_vals = bc_vals_3d
        specified_parameters["bc_values"] = bc_vals

        # (Integrated) source terms are given by the exact solution
        if g.dim == max_dim:
            source_term = integrated_f3d
        else:
            source_term = integrated_f2d

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
    sol = sps.linalg.spsolve(A, b)
    assembler.distribute_variable(sol)

    # Overwrite d[pp.STATE][subdomain_variable] to be consistent with FEM methods
    for g, d in gb:
        discr = d[pp.DISCRETIZATION][subdomain_variable][subdomain_operator_keyword]
        pressure = discr.extract_pressure(g, d[pp.STATE][subdomain_variable], d).copy()
        flux = discr.extract_flux(g, d[pp.STATE][subdomain_variable], d).copy()
        d[pp.STATE][subdomain_variable] = pressure
        d[pp.STATE][flux_variable] = flux

    # %% Obtain error estimates (and transfer them to d[pp.STATE])
    # NOTE: Residual errors must be obtained separately
    estimates = mde.ErrorEstimate(gb, lam_name=edge_variable)
    estimates.estimate_error()
    estimates.transfer_error_to_state()

    # %% Retrieve errors
    te = TrueErrors3D(gb=gb, estimates=estimates)

    diffusive_error_squared_3d = d_3d[pp.STATE]["diffusive_error"]
    diffusive_error_squared_2d = d_2d[pp.STATE]["diffusive_error"]
    diffusive_error_squared_mortar = d_e[pp.STATE]["diffusive_error"]
    diffusive_error = (diffusive_error_squared_3d.sum()
                       + diffusive_error_squared_2d.sum()
                       + diffusive_error_squared_mortar.sum()) ** 0.5

    residual_error_squared_3d = te.residual_error_3d_local_poincare()
    residual_error_squared_2d = te.residual_error_2d_local_poincare()
    residual_error = (residual_error_squared_3d.sum()
                      + residual_error_squared_2d.sum()) ** 0.5

    majorant = diffusive_error + residual_error

    # Distinguishing between subdomain and mortar errors
    bulk_error = (diffusive_error_squared_3d.sum()
                  + residual_error_squared_3d.sum()) ** 0.5
    fracture_error = (diffusive_error_squared_2d.sum()
                      + residual_error_squared_2d.sum()) ** 0.5
    mortar_error = diffusive_error_squared_mortar.sum() ** 0.5

    # %% Obtain true errors

    # Pressure error
    pressure_error_squared_3d = te.pressure_error_squared_3d()
    pressure_error_squared_2d = te.pressure_error_squared_2d()
    pressure_error_squared_mortar = te.pressure_error_squared_mortar()
    true_pressure_error = te.pressure_error()

    # Velocity error
    velocity_error_squared_3d = te.velocity_error_squared_3d()
    velocity_error_squared_2d = te.velocity_error_squared_2d()
    velocity_error_squared_mortar = te.velocity_error_squared_mortar()
    true_velocity_error = te.velocity_error()

    # True combined error
    true_combined_error = (true_pressure_error + true_velocity_error
                           + residual_error)

    # %% Compute efficiency indices
    i_eff_p = majorant / true_pressure_error
    i_eff_u = majorant / true_velocity_error
    i_eff_pu = (3 * majorant) / true_combined_error

    print(50 * "-")
    print(f'Majorant: {majorant}')
    print(f'Bulk error: {bulk_error}')
    print(f'Fracture error: {fracture_error}')
    print(f'Mortar error: {mortar_error}')
    print(f"True error (pressure): {true_pressure_error}")
    print(f"True error (velocity): {true_velocity_error}")
    print(f"True error (combined): {true_combined_error}")
    print(f"Efficiency index (pressure): {i_eff_p}")
    print(f"Efficiency index (velocity): {i_eff_u}")
    print(f"Efficiency index (combined): {i_eff_pu}")
    print(50 * "-")

    # Prepare return dictionary
    out = {}

    out["majorant"] = majorant
    out["true_pressure_error"] = true_pressure_error
    out["true_velocity_error"] = true_velocity_error
    out["true_combined_error"] = true_combined_error
    out["efficiency_pressure"] = i_eff_p
    out["efficiency_velocity"] = i_eff_u
    out["efficiency_combined"] = i_eff_pu

    out["bulk"] = {}
    out["bulk"]["mesh_size"] = np.max(g_3d.cell_diameters())
    out["bulk"]["error"] = bulk_error
    out["bulk"]["diffusive_error"] = diffusive_error_squared_3d.sum() ** 0.5
    out["bulk"]["residual_error"] = residual_error_squared_3d.sum() ** 0.5

    out["frac"] = {}
    out["frac"]["mesh_size"] = np.max(g_2d.cell_diameters())
    out["frac"]["error"] = fracture_error
    out["frac"]["diffusive_error"] = diffusive_error_squared_2d.sum() ** 0.5
    out["frac"]["residual_error"] = residual_error_squared_2d.sum() ** 0.5

    out["mortar"] = {}
    out["mortar"]["mesh_size"] = np.max(mg.cell_diameters())
    out["mortar"]["error"] = mortar_error

    return out
