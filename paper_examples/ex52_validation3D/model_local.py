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
    kwe = estimates.estimates_kw

    bulk_diffusive_squared = d_3d[kwe]["diffusive_error"].sum()
    fracture_diffusive_squared = d_2d[kwe]["diffusive_error"].sum()
    mortar_diffusive_squared = d_e[kwe]["diffusive_error"].sum()
    diffusive_error = np.sqrt(
        bulk_diffusive_squared + fracture_diffusive_squared + mortar_diffusive_squared
    )  # T_1 in the paper

    # %% Obtain residual error
    def compute_residual_error(g, d, estimates):
        """
        Computes residual errors for each subdomain grid

        Parameters
        ----------
        g: Grid
        d: Data dictionary
        estimates: Estimates object

        Returns
        -------
        Residual error (squared) for each cell of the subdomain.
        """

        # Retrieve reconstructed velocity
        recon_u = d[estimates.estimates_kw]["recon_u"].copy()

        # Retrieve permeability
        perm = d[pp.PARAMETERS][estimates.kw]["second_order_tensor"].values
        k = perm[0][0].reshape(g.num_cells, 1)

        # Obtain (square of the) constant multiplying the norm:
        # (C_{p,K} h_K / ||k^{-1/2}||_K)^2 = k_K h_K^2 / pi^2
        const = k * g.cell_diameters().reshape(g.num_cells, 1) ** 2 / np.pi ** 2

        # Obtain coefficients of the full flux and compute its divergence
        u = utils.poly2col(recon_u)
        if g.dim == 3:
            div_u = 3 * u[0]
        elif g.dim == 2:
            div_u = 2 * u[0]

        # Obtain contribution from mortar jump to local mass conservation
        jump_in_mortars = (mortar_jump(estimates, g) / g.cell_volumes).reshape(g.num_cells, 1)

        # Declare integration method and get hold of elements in QuadPy format
        if g.dim == 3:
            int_method = qp.t3.get_good_scheme(6)  # since f is quadratic, we need at least order 4
            elements = utils.get_quadpy_elements(g, g)
        elif g.dim == 2:
            int_method = qp.t2.get_good_scheme(6)
            elements = utils.get_quadpy_elements(g, utils.rotate_embedded_grid(g))

        # We now declare the different integrand regions and compute the norms
        integral = np.zeros(g.num_cells)
        if g.dim == 3:
            for (f, idx) in zip(f3d_numpy_list, cell_idx_list):
                # Declare integrand
                def integrand(x):
                    return (f(x[0], x[1], x[2]) - div_u + jump_in_mortars) ** 2

                # Integrate, and add the contribution of each subregion
                integral += int_method.integrate(integrand, elements) * idx
        elif g.dim == 2:
            # Declare integrand
            def integrand(x):
                f_1d = -2 * np.ones_like(x)
                return (f_1d - div_u + jump_in_mortars) ** 2

            integral = int_method.integrate(integrand, elements)

        # Finally, obtain residual error
        residual_error = const.flatten() * integral

        return residual_error

    bulk_residual_squared = compute_residual_error(g_3d, d_3d, estimates).sum()
    fracture_residual_squared = compute_residual_error(g_2d, d_2d, estimates).sum()
    residual_error = np.sqrt(bulk_residual_squared + fracture_residual_squared)  # T_2 in the paper

    #%% Evaluation of the majorant
    majorant = diffusive_error + residual_error

    # Distinguishing between subdomain and mortar errors
    bulk_error = np.sqrt(bulk_diffusive_squared + bulk_residual_squared)
    fracture_error = np.sqrt(fracture_diffusive_squared + fracture_residual_squared)
    mortar_error = np.sqrt(mortar_diffusive_squared)

    print("------------------------------------------------")
    print(f'Majorant: {majorant}')
    print(f'Bulk error: {bulk_error}')
    print(f'Fracture error: {fracture_error}')
    print(f'Mortar error: {mortar_error}')
    print("------------------------------------------------")

    #%% Evaluate reconstructed quantities
    def get_cc_reconp(estimates, cell_idx_list):
        """
        Get hold of cell-centered evaluated reconstructed pressures

        Parameters
        ----------
        estimates : Error Estimates Object
            Error estimates object after mde.ErrorEstimate() has been applied
        cell_idx_list : List of length 9
            Containing the boolean cell indices of the subregions of the 3D domain

        Returns
        -------
        NumPy nd-Array
            Cell-centered evaluated reconstructed pressure of the 3D domain.
        NumPy nd-Array
            Cell-centered evaluated reconstructed pressure of the 2D domain.

        """

        # Get hold of estimates keyword
        kw_e = estimates.estimates_kw

        for g, d in gb:

            # Get hold of reconstructed pressure
            recon_p = d[kw_e]["recon_p"].copy()
            p = utils.poly2col(recon_p)

            # Obtain cell-centered coordinates
            x = g.cell_centers

            # Evaluate the 3D-reconstructed pressure
            if g.dim == 3:
                rp_cc_3d = np.zeros([g.num_cells, 1])
                for idx in cell_idx_list:
                    rp_cc_3d += (
                        p[0] * x[0].reshape(g.num_cells, 1)
                        + p[1] * x[1].reshape(g.num_cells, 1)
                        + p[2] * x[2].reshape(g.num_cells, 1)
                        + p[3]
                    ) * idx.reshape(g.num_cells, 1)
            # Evaluate the 2D-reconstructed pressure
            else:
                rp_cc_2d = (
                    p[0] * x[0].reshape(g.num_cells, 1)
                    + p[1] * x[1].reshape(g.num_cells, 1)
                    + p[2]
                )

        return rp_cc_3d.flatten(), rp_cc_2d.flatten()

    def get_cc_reconvel(estimates, cell_idx_list):
        """
        Get hold of cell-centered evaluated reconstructed velocities

        Parameters
        ----------
        estimates : Error Estimates Object
            Error estimates object after mde.ErrorEstimate() has been applied
        cell_idx_list : List of length 9
            Containing the boolean cell indices of the subregions of the 3D domain

        Returns
        -------
        NumPy nd-Array
            Cell-centered evaluated reconstructed velocity of the 3D domain.
        NumPy nd-Array
            Cell-centered evaluated reconstructed velocity of the 2D domain.

        """

        # Get hold of estimates keyword
        kw_e = estimates.estimates_kw

        for g, d in gb:

            # Get hold of reconstructed pressure
            recon_u = d[kw_e]["recon_u"].copy()
            u = utils.poly2col(recon_u)

            # Obtain cell-centered coordinates
            x = g.cell_centers

            # Evaluate the 3D-reconstructed pressure
            if g.dim == 3:
                ru_cc_3d_x = np.zeros([g.num_cells, 1])
                ru_cc_3d_y = np.zeros([g.num_cells, 1])
                ru_cc_3d_z = np.zeros([g.num_cells, 1])

                for idx in cell_idx_list:
                    ru_cc_3d_x += (
                        u[0] * x[0].reshape(g.num_cells, 1) + u[1]
                    ) * idx.reshape(g.num_cells, 1)
                    ru_cc_3d_y += (
                        u[0] * x[1].reshape(g.num_cells, 1) + u[2]
                    ) * idx.reshape(g.num_cells, 1)
                    ru_cc_3d_z += (
                        u[0] * x[2].reshape(g.num_cells, 1) + u[3]
                    ) * idx.reshape(g.num_cells, 1)

                ru_cc_3d = np.array(
                    [ru_cc_3d_x.flatten(), ru_cc_3d_y.flatten(), ru_cc_3d_z.flatten()]
                )

            # Evaluate the 2D-reconstructed pressure
            else:
                ru_cc_2d_x = (u[0] * x[0].reshape(g.num_cells, 1) + u[1]).flatten()
                ru_cc_2d_y = (u[0] * x[1].reshape(g.num_cells, 1) + u[2]).flatten()
                ru_cc_2d_z = np.zeros(g.num_cells)
                ru_cc_2d = np.array([ru_cc_2d_x, ru_cc_2d_y, ru_cc_2d_z])

        return ru_cc_3d, ru_cc_2d

    # Get hold of cell-centered reconstructed pressure for the 3D and 2D domain
    reconp_cc_3d, reconp_cc_2d = get_cc_reconp(estimates, cell_idx_list)

    # Get hold of cell-centered reconstructed velocity for the 3D and 2D domain
    reconu_cc_3d, reconu_cc_2d = get_cc_reconvel(estimates, cell_idx_list)

    #%% Compute true errors for the pressure, i.e., ||| p - p_h |||
    def compute_pressure_3d_true_error(
        g, d, estimates, gradp3d_numpy_list, cell_idx_list
    ):
        """
        Computes true "pressure" error for the 3D subdomain

        Parameters
        ----------
        g : PorePy Grid
            Three-dimensional grid.
        d : Dictionary
            Dictionary corresponding to the 3D-grid.
        estimates : Error estimates object
            Error estimates object as obtained with mde.ErrorEstimate.
        gradp3d_numpy_list : List
            List of NumPy lambda functions for each subregion.
        cell_idx_list : List
            List of Numpy boolean array for each subregion.

        Raises
        ------
        ValueError
            If dimension of the grid is different from 3.

        Returns
        -------
        integral: NumPy array of size g_3d.num_cells
            (Squared) of the true errors for each element of the grid.

        """

        # Check if dimension is 3
        if g.dim != 3:
            raise ValueError("Dimension should be 3")

        # Get hold of keyword for accessing error estimates subdict
        kwe = estimates.estimates_kw

        # Get hold of reconstructed pressure and create list of coeffcients
        recon_p = d[kwe]["recon_p"].copy()
        pr = utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = qp.t3.get_good_scheme(3)
        elements = utils.get_quadpy_elements(g, g)

        # Compute the true error for each subregion
        integral = np.zeros(g.num_cells)

        for (gradp, idx) in zip(gradp3d_numpy_list, cell_idx_list):

            # Declare integrand and add subregion contribution
            def integrand(x):
                gradp_exact_x = gradp[0](x[0], x[1], x[2])
                gradp_exact_y = gradp[1](x[0], x[1], x[2])
                gradp_exact_z = gradp[2](x[0], x[1], x[2])

                gradp_recon_x = pr[0] * np.ones_like(x[0])
                gradp_recon_y = pr[1] * np.ones_like(x[1])
                gradp_recon_z = pr[2] * np.ones_like(x[2])

                int_x = (gradp_exact_x - gradp_recon_x) ** 2
                int_y = (gradp_exact_y - gradp_recon_y) ** 2
                int_z = (gradp_exact_z - gradp_recon_z) ** 2

                return int_x + int_y + int_z

            integral += method.integrate(integrand, elements) * idx

        return integral

    def compute_pressure_2d_true_error(g, d, estimates):
        """
        Computes the true "pressure" error for the 2D domain (the fracture)

        Parameters
        ----------
        g : PorePy Grid
            Two-dimensional grid.
        d : Dictionary
            Dictionary corresponding to the 2D-grid.
        estimates : Error estimates object
            Error estimates object as obtained with mde.ErrorEstimate.

        Raises
        ------
        ValueError
            If dimension of the grid is different from 2.

        Returns
        -------
        NumPy nd-Array of size g_2d.num_cells
            (Squared) of the true errors for each element of the grid.
        """

        # Check if dimension is 2
        if g.dim != 2:
            raise ValueError("Dimension should be 2")

        # Get hold of keyword for accessing error estimates subdict
        kwe = estimates.estimates_kw

        # Get hold of reconstructed pressure and create list of coefficients
        recon_p = d[kwe]["recon_p"].copy()
        pr = utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = qp.t2.get_good_scheme(3)
        g_rot = utils.rotate_embedded_grid(g)
        elements = utils.get_quadpy_elements(g, g_rot)

        # Compute the true error
        def integrand(x):

            gradp_exact_x = np.zeros_like(x[0])
            gradp_exact_y = np.zeros_like(x[1])

            gradp_recon_x = pr[0] * np.ones_like(x[0])
            gradp_recon_y = pr[1] * np.ones_like(x[1])

            int_x = (gradp_exact_x - gradp_recon_x) ** 2
            int_y = (gradp_exact_y - gradp_recon_y) ** 2

            return int_x + int_y

        return method.integrate(integrand, elements)

    def compute_pressure_mortar_true_error(d_e, estimates):
        """
        Computes the true "pressure" error for the mortar grid

        Parameters
        ----------
        d_e : Dictionary
            Dictionary of the interface
        estimates: Error estimate object
            Error estimate object as obtained with mde.ErrorEstimate()

        Raises
        ------
        ValueError
            If the dimension of the mortar grid is different from 2.

        Returns
        -------
        true_error_mortar: NumPy nd-array of shape (mg.num_cells, 1)
            True error (squared) for each element of the mortar grid.

        """

        # Import functions
        from mdestimates._error_evaluation import (
            _get_high_pressure_trace,
            _get_low_pressure,
        )

        def compute_sidegrid_error(estimates, side_tuple):
            """
            This functions projects a mortar quantity to the side grids, and then
            performs the integration on the given side grid.

            Parameters
            ----------
            side_tuple : Tuple
                Containing the sidegrids

            Returns
            -------
            true_error_side : NumPy nd-Array of size (sidegrid.num_cells, 1)
                True error (squared) for each element of the side grid.

            """

            # Get projector and sidegrid object
            projector = side_tuple[0]
            sidegrid = side_tuple[1]

            # Rotate side-grid
            sidegrid_rot = utils.rotate_embedded_grid(sidegrid)

            # Obtain QuadPy elements
            elements = utils.get_quadpy_elements(sidegrid, sidegrid_rot)

            # Project relevant quantities to the side grids
            deltap_side = projector * deltap
            k_side = projector * k
            n = projector.shape[0]
            true_jump = -np.ones(n).reshape(n, 1)

            # Declare integrand
            def integrand(x):
                p_jump = utils.eval_P1(deltap_side, x)
                return (k_side ** 0.5 * (true_jump - p_jump)) ** 2

            # Compute integral
            true_error_side = method.integrate(integrand, elements)

            return true_error_side

        # Get hold of mortar grid and check the dimensionality
        mg = d_e["mortar_grid"]
        if mg.dim != 2:
            raise ValueError("Expected two-dimensional grid")

        # Obtain higher- and lower-dimensional grids and dictionaries
        g_l, g_h = gb.nodes_of_edge(e)
        d_h = gb.node_props(g_h)
        d_l = gb.node_props(g_l)

        # Retrieve normal diffusivity
        normal_diff = d_e[pp.PARAMETERS]["flow"]["normal_diffusivity"]
        if isinstance(normal_diff, int) or isinstance(normal_diff, float):
            k = normal_diff * np.ones([mg.num_cells, 1])
        else:
            k = normal_diff.reshape(mg.num_cells, 1)

        # Face-cell map between higher- and lower-dimensional subdomains
        frac_faces = sps.find(mg.primary_to_mortar_avg().T)[0]
        frac_cells = sps.find(mg.secondary_to_mortar_avg().T)[0]

        # Obtain the trace of the higher-dimensional pressure
        tracep_high = _get_high_pressure_trace(estimates, g_l, g_h, d_h, frac_faces)

        # Obtain the lower-dimensional pressure
        p_low = _get_low_pressure(estimates, g_l, d_l, frac_cells)

        # Now, we can work with the pressure difference
        deltap = p_low - tracep_high

        # Declare integration method
        method = qp.t2.get_good_scheme(3)

        # Retrieve side-grids tuples
        sides = mg.project_to_side_grids()

        # Compute the errors for each sidegrid
        mortar_error = []
        for side in sides:
            mortar_error.append(compute_sidegrid_error(estimates, side))

        # Concatenate into one numpy array
        true_error_mortar = np.concatenate(mortar_error)

        return true_error_mortar

    #%% Compute true errors for the velocity, i.e., ||| u - u_h |||_*
    def compute_velocity_3d_true_error(g, d, estimates, u3d_numpy_list, cell_idx_list):
        """
        Computes the true "velocity" error for the 3D subdomain

        Parameters
        ----------
        g : PorePy Grid
            Three-dimensional grid.
        d : Dictionary
            Dictionary corresponding to the 3D-grid.
        estimates : Error estimates object
            Error estimates object as obtained with mde.ErrorEstimate.
        u3d_numpy_list : List
            List of NumPy lambda functions for each subregion.
        cell_idx_list : List
            List of Numpy boolean array for each subregion.

        Raises
        ------
        ValueError
            If dimension of the grid is different from 3.

        Returns
        -------
        integral: NumPy array of size g_3d.num_cells
            (Squared) of the true errors for each element of the grid.

        """

        # Check if dimension is 3
        if g.dim != 3:
            raise ValueError("Dimension should be 3")

        # Get hold of keyword for accessing error estimates subdict
        kwe = estimates.estimates_kw

        # Get hold of approximated velocities and create list of coeffcients
        recon_u = d[kwe]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        # Obtain elements and declare integration method
        method = qp.t3.get_good_scheme(3)
        elements = utils.get_quadpy_elements(g, g)

        # Compute the true error for each subregion
        integral = np.zeros(g.num_cells)

        for (vel, idx) in zip(u3d_numpy_list, cell_idx_list):

            # Declare integrand and add subregion contribution
            def integrand(x):
                vel_exact_x = vel[0](x[0], x[1], x[2])
                vel_exact_y = vel[1](x[0], x[1], x[2])
                vel_exact_z = vel[2](x[0], x[1], x[2])

                vel_recon_x = u[0] * x[0] + u[1]
                vel_recon_y = u[0] * x[1] + u[2]
                vel_recon_z = u[0] * x[2] + u[3]

                int_x = (vel_exact_x - vel_recon_x) ** 2
                int_y = (vel_exact_y - vel_recon_y) ** 2
                int_z = (vel_exact_z - vel_recon_z) ** 2

                return int_x + int_y + int_z

            integral += method.integrate(integrand, elements) * idx

        return integral

    def compute_velocity_2d_true_error(g, d, estimates):
        """
        Computes the true "velocity" error for the 2D subdomain (the fracture)

        Parameters
        ----------
        g : PorePy Grid
            Two-dimensional grid.
        d : Dictionary
            Dictionary corresponding to the 2D-grid.
        estimates : Error estimates object
            Error estimates object as obtained with mde.ErrorEstimate.

        Raises
        ------
        ValueError
            If dimension of the grid is different from 2.

        Returns
        -------
        NumPy nd-Array of size g_2d.num_cells
            (Squared) of the true errors for each element of the grid.
        """

        # Check if dimension is 2
        if g.dim != 2:
            raise ValueError("Dimension should be 2")

        # Get hold of keyword for accessing error estimates subdict
        kwe = estimates.estimates_kw

        # Get hold of reconstructed pressure and create list of coeffcients
        recon_u = d[kwe]["recon_u"].copy()
        u = utils.poly2col(recon_u)

        # Obtain elements and declare integration method
        method = qp.t2.get_good_scheme(3)
        g_rot = utils.rotate_embedded_grid(g)
        elements = utils.get_quadpy_elements(g, g_rot)

        # Compute the true error
        def integrand(x):

            vel_exact_x = np.zeros_like(x[0])
            vel_exact_y = np.zeros_like(x[1])

            vel_recon_x = u[0] * x[0] + u[1]
            vel_recon_y = u[0] * x[1] + u[2]

            int_x = (vel_exact_x - vel_recon_x) ** 2
            int_y = (vel_exact_y - vel_recon_y) ** 2

            return int_x + int_y

        return method.integrate(integrand, elements)

    def compute_velocity_mortar_true_error(d_e, estimates):
        """
        Computes the true "velocity" error for the mortar grid

        Parameters
        ----------
        d_e : Dictionary
            Dictionary of the interface
        estimates: Error estimate object
            Error estimate object as obtained with mde.ErrorEstimate()

        Raises
        ------
        ValueError
            If the dimension of the mortar grid is different from 2.

        Returns
        -------
        true_error_mortar: NumPy nd-array of shape (mg.num_cells, 1)
            True error (squared) for each element of the mortar grid.

        """

        # Get mortar grid
        mg = d_e["mortar_grid"]

        # Sanity check
        if mg.dim != 2:
            raise ValueError("Mortar grid must be two-dimensional")

        # Obtain difference between exact and approximated mortar fluxes
        V = mg.cell_volumes
        lmbda = d_e[pp.STATE][estimates.lam_name].copy()
        mortar_flux = lmbda / V
        lbmda_diff = (1.0 - mortar_flux) ** 2
        true_error_mortar = lbmda_diff * V

        return true_error_mortar

    #%% Obtain true errors
    # Pressure true errors -> tpe = true pressure error
    tpe_bulk_squared = compute_pressure_3d_true_error(g_3d, d_3d, estimates, gradp3d_numpy_list, cell_idx_list).sum()
    tpe_fracture_squared = compute_pressure_2d_true_error(g_2d, d_2d, estimates).sum()
    tpe_mortar_squared = compute_pressure_mortar_true_error(d_e, estimates).sum()
    true_pressure_error = np.sqrt(tpe_bulk_squared + tpe_fracture_squared + tpe_mortar_squared)

    # Velocity true errors -> tve = true velocity error
    tve_bulk_squared = compute_velocity_3d_true_error(g_3d, d_3d, estimates, u3d_numpy_list, cell_idx_list).sum()
    tve_fracture_squared = compute_velocity_2d_true_error(g_2d, d_2d, estimates).sum()
    tve_mortar_squared = compute_velocity_mortar_true_error(d_e, estimates).sum()
    true_velocity_error = np.sqrt(tve_bulk_squared + tve_fracture_squared + tve_mortar_squared)

    # True error for the primal-dual variable
    true_combined_error = true_pressure_error + true_velocity_error + residual_error

    # %% Compute efficiency indices
    i_eff_p = majorant / true_pressure_error  # (Eq. 4.27)
    i_eff_u = majorant / true_velocity_error  # (Eq. 4.28)
    i_eff_pu = (3 * majorant) / true_combined_error  # (Eq. 4.29)

    print(f"Efficiency index (pressure): {i_eff_p}")
    print(f"Efficiency index (velocity): {i_eff_u}")
    print(f"Efficiency index (combined): {i_eff_pu}")

    #%% Return
    return (
        h_max,
        bulk_error,
        np.sqrt(tpe_bulk_squared),
        np.sqrt(tve_bulk_squared),
        g_3d.num_cells,
        fracture_error,
        np.sqrt(tpe_fracture_squared),
        np.sqrt(tve_fracture_squared),
        g_2d.num_cells,
        mortar_error,
        np.sqrt(tpe_mortar_squared),
        np.sqrt(tve_mortar_squared),
        mg.num_cells,
        majorant,
        true_pressure_error,
        true_velocity_error,
        i_eff_p,
        i_eff_u,
        i_eff_pu,
    )
