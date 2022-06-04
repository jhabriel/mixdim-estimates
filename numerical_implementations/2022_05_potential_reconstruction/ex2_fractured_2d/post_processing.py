# Testing whether Vohralik's postprocessing gives zero diffusive error on subdominas and
# mortars

import numpy as np
import porepy as pp
import scipy.sparse as sps
import matplotlib.pyplot as plt
import mdestimates as mde
import pypardiso
import quadpy as qp

from analytical_2d import ExactSolution2D
from true_errors_2d import TrueErrors2D
import mdestimates.estimates_utils as utils

from typing import Tuple
Edge = Tuple[pp.Grid, pp.Grid]

# %% Study parameters
# Create grid bucket and extract data
domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
network_2d = pp.fracture_importer.network_2d_from_csv("network.csv", domain=domain)

# Target lengths
mesh_size = 0.1
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

# %% Testing diffusive error
estimates = mde.ErrorEstimate(gb, lam_name=edge_variable, p_recon_method="vohralik")
estimates.init_estimates_data_keyword()

# Reconstruct velocity
vel_rec = mde.VelocityReconstruction(estimates)
vel_rec.compute_full_flux()
vel_rec.reconstruct_velocity()

# Reconstruct pressure
p_rec = mde.PressureReconstruction(estimates)
p_rec.reconstruct_pressure()

#%% Subdomain diffusive error (diffusive in the context of Vohralik's)
def subdomain_diffusive_error(estimates,
                              g: pp.Grid,
                              g_rot: mde.RotatedGrid,
                              d: dict
                              ) -> np.ndarray:
    """
    Computes the square of the subdomain diffusive errors.

    Parameters
    ----------
        g (pp.Grid): PorePy grid.
        g_rot (mde.RotatedGrid): Rotated pseudo-grid.
        d (dict): Data dictionary.

    Raises
    ------
    ValueError:
        If the reconstructed pressure is not in the data dictionary.
        If the reconstructed velocity is not in the data dictionary.
        If the grid dimension is not 1, 2, or 3.

    Returns
    -------
        diffusive_error (np.ndarray): Square of the diffusive flux error for the  grid.
        The size of the array is: g.num_cells.

    Technical note
    --------------
        The square of the diffusive flux error is given locally for an element E by:

                || K_E^(-1/2) u_rec,E + K_E^(1/2) grad(p_rec,E) ||_E^2,

        where K_E is the permeability, u_rec,E is the reconstructed velocity, and
        grad(p_rec,E) is the gradient of the reconstructed pressure.
    """

    # Sanity checks
    if g.dim not in [1, 2, 3]:
        raise ValueError("Error not defined for the given grid dimension.")

    if "postprocessed_p" not in d[estimates.estimates_kw]:
        raise ValueError("Pressure must be postprocessed first.")

    if "recon_u" not in d[estimates.estimates_kw]:
        raise ValueError("Velocity must be postprocessed first.")

    # Retrieve postprocessed pressure
    post_p = d[estimates.estimates_kw]["postprocessed_p"]

    # Retrieve reconstructed velocity
    recon_u = d[estimates.estimates_kw]["recon_u"]

    # Retrieve permeability
    perm = d[pp.PARAMETERS][estimates.kw]["second_order_tensor"].values
    k = perm[0][0].reshape(g.num_cells, 1)

    # Get quadpy elements and declare integration method
    # Locally, we have at most P1 polynomials, but since we measure the square of the
    # error, we need at integration methods of order 2 to compute the exact integrals
    elements = utils.get_quadpy_elements(g, g_rot)
    if g.dim == 1:
        method = qp.c1.newton_cotes_closed(4)
    elif g.dim == 2:
        method = qp.t2.get_good_scheme(4)
    else:
        method = qp.t3.get_good_scheme(4)

    # Obtain coefficients
    p = utils.poly2col(post_p)
    u = utils.poly2col(recon_u)

    # Declare integrands and prepare for integration
    def integrand(x):

        # One-dimensional subdomains
        # gradp reconstructed in x
        # Recall that
        # p(x)|K = c0x^2 + c1x + c2
        # with "gradient"
        # gradp(x)|K = 2c0x + c1
        if g.dim == 1:
            veloc_x = u[0] * x + u[1]

            if estimates.p_degree == 1:  # P1
                gradp_x = p[0] * np.ones_like(x)
            else:  # P2
                gradp_x = 2 * p[0] * x + p[1] * np.ones_like(x)

            int_x = (k ** (-0.5) * veloc_x + k ** 0.5 * gradp_x) ** 2

            return int_x

        # Two-dimensional subdomains
        # gradp reconstructed in x and y
        # Recall that
        # p(x, y)|K = c0x^2 + c1xy + c2x + c3y^2 + c4y + c5
        # with gradient
        # gradp(x,y)|K = [2c0x + c1y + c2, c1x + 2c3y + c4]
        elif g.dim == 2:
            veloc_x = u[0] * x[0] + u[1]
            veloc_y = u[0] * x[1] + u[2]

            if estimates.p_degree == 1:  # P1
                gradp_x = p[0] * np.ones_like(x[0])
                gradp_y = p[1] * np.ones_like(x[1])
            else:  # P2
                gradp_x = 2 * p[0] * x[0] + p[1] * x[1] + p[2] * np.ones_like(x[0])
                gradp_y = p[1] * x[0] + 2 * p[3] * x[1] + p[4] * np.ones_like(x[1])

            int_x = (k ** (-0.5) * veloc_x + k ** 0.5 * gradp_x) ** 2
            int_y = (k ** (-0.5) * veloc_y + k ** 0.5 * gradp_y) ** 2

            return int_x + int_y

        # Three-dimensional subdomains
        # gradp reconstructed in x, y, and z
        # Recall that:
        # p(x,y,z)|K = c0x^2 + c1xy + c2xz + c3x + c4y^2 + c5yz + c6y + c7z^2 + c8z + c9
        # with gradient:
        #                  [ 2c0x + c1y + c2z + c3 ]
        # gradp(x,y,z)|K = [ c1x + 2c4y + c5z + c6 ]
        #                  [ c2x + c5y + 2c7z + c8 ]
        else:
            veloc_x = u[0] * x[0] + u[1]
            veloc_y = u[0] * x[1] + u[2]
            veloc_z = u[0] * x[2] + u[3]

            if estimates.p_degree == 1:  # P1
                gradp_x = p[0] * np.ones_like(x[0])
                gradp_y = p[1] * np.ones_like(x[1])
                gradp_z = p[2] * np.ones_like(x[2])
            else:  # P2
                gradp_x = (
                        2 * p[0] * x[0]
                        + p[1] * x[1]
                        + p[2] * x[2]
                        + p[3] * np.ones_like(x[0])
                )
                gradp_y = (
                        p[1] * x[0]
                        + 2 * p[4] * x[1]
                        + p[5] * x[2]
                        + p[6] * np.ones_like(x[1])
                )
                gradp_z = (
                        p[2] * x[0]
                        + p[5] * x[1]
                        + 2 * p[7] * x[2]
                        + p[8] * np.ones_like(x[2])
                )

            int_x = (k ** (-0.5) * veloc_x + k ** 0.5 * gradp_x) ** 2
            int_y = (k ** (-0.5) * veloc_y + k ** 0.5 * gradp_y) ** 2
            int_z = (k ** (-0.5) * veloc_z + k ** 0.5 * gradp_z) ** 2

            return int_x + int_y + int_z

    # Compute the integral
    diffusive_error = method.integrate(integrand, elements)

    return diffusive_error

diffusive_2d = subdomain_diffusive_error(estimates, g_2d, mde.RotatedGrid(g_2d), d_2d)
diffusive_1d = subdomain_diffusive_error(estimates, g_1d, mde.RotatedGrid(g_1d), d_1d)

#%% Inteface diffusive errors (diffusive in the context of Vohralik's)
# Utility functions
def _frac_faces_lagrangian_coo(
        estimates,
        g_h: pp.Grid,
        frac_faces: np.ndarray,
        rotated_coo: bool = False
) -> np.ndarray:

    if estimates.p_degree == 1:  # P1 polynomials
        # Get nodes of the fracture faces
        frac_faces_nodes = sps.find(g_h.face_nodes.T[frac_faces].T)[0]
        nodes_of_frac_faces = frac_faces_nodes.reshape((frac_faces.size, g_h.dim))
        # Obtain coordinates of Lagrangian nodes at the nodes of the fracture faces
        if rotated_coo:
            gh_rot = mde.RotatedGrid(g_h)
            lagran_coo = gh_rot.nodes[:, nodes_of_frac_faces]
        else:
            lagran_coo = g_h.nodes[:, nodes_of_frac_faces]
    else:
        if g_h.dim == 3:
            raise NotImplementedError("P2 elements not implemented for 3D")
        else:
            # Get nodes of the fracture faces
            frac_faces_nodes = sps.find(g_h.face_nodes.T[frac_faces].T)[0]
            nodes_of_frac_faces = frac_faces_nodes.reshape((frac_faces.size, g_h.dim))
            frac_faces_reshaped = frac_faces.reshape((frac_faces.size, 1))
            # Obtain the coordinates of the nodes of the fracture faces and the
            # coordinates of the fracture faces centers
            if rotated_coo:
                gh_rot = mde.RotatedGrid(g_h)
                lagran_coo_nodes = gh_rot.nodes[:, nodes_of_frac_faces]
                lagran_coo_fc = gh_rot.face_centers[:, frac_faces_reshaped]
                lagran_coo = np.dstack((lagran_coo_nodes, lagran_coo_fc))
            else:
                lagran_coo_nodes = g_h.nodes[:, nodes_of_frac_faces]
                lagran_coo_fc = g_h.face_centers[:, frac_faces_reshaped]
                lagran_coo = np.dstack((lagran_coo_nodes, lagran_coo_fc))

    return lagran_coo

def _get_high_pressure_trace(estimates,
                             g_l: pp.Grid,
                             g_h: pp.Grid,
                             d_h: dict,
                             frac_faces: np.ndarray
                             ) -> np.ndarray:

    # Rotate both grids, and obtain rotation matrix and effective dimension
    gh_rot = mde.RotatedGrid(g_h)
    gl_rot = mde.RotatedGrid(g_l)
    rotation_matrix = gl_rot.rotation_matrix
    dim_bool = gl_rot.dim_bool

    # Obtain the cells corresponding to the frac_faces
    cells_of_frac_faces, _, _ = sps.find(g_h.cell_faces[frac_faces].T)

    # Retrieve the coefficients of the polynomials corresponding to those cells
    if "postprocessed_p" in d_h[estimates.estimates_kw]:
        p_high = d_h[estimates.estimates_kw]["postprocessed_p"]
    else:
        raise ValueError("Pressure must be reconstructed first")
    p_high = p_high[cells_of_frac_faces]

    # NOTE: Use the rotated coordinates to perform the evaluation of the pressure,
    # but use the original coordinates to rotate the edge using the rotation matrix of
    # the lower-dimensional grid as reference.

    # Evaluate the polynomials at the relevant Lagrangian nodes according to poly degree
    point_coo_rot = _frac_faces_lagrangian_coo(estimates, g_h, frac_faces, rotated_coo=True)
    if estimates.p_degree == 1:
        point_val = utils.eval_p1(p_high, point_coo_rot)
    else:
        point_val = utils.eval_p2(p_high, point_coo_rot)

    # Rotate the coordinates of the Lagrangian nodes w.r.t. the lower-dimensional grid
    point_coo = _frac_faces_lagrangian_coo(estimates, g_h, frac_faces)
    point_edge_coo_rot = np.empty_like(point_coo)
    for element in range(frac_faces.size):
        point_edge_coo_rot[:, element] = np.dot(rotation_matrix, point_coo[:, element])
    point_edge_coo_rot = point_edge_coo_rot[dim_bool]

    # Construct a polynomial (of reduced dimensionality) using the rotated coordinates
    if estimates.p_degree == 1:
        trace_pressure = utils.interpolate_p1(point_val, point_edge_coo_rot)
    else:
        trace_pressure = utils.interpolate_p2(point_val, point_edge_coo_rot)

    # Test if the values of the original polynomial match the new one
    if estimates.p_degree == 1:
        point_val_rot = utils.eval_p1(trace_pressure, point_edge_coo_rot)
    else:
        point_val_rot = utils.eval_p2(trace_pressure, point_edge_coo_rot)

    np.testing.assert_almost_equal(point_val, point_val_rot, decimal=12)

    return trace_pressure

def _get_low_pressure(estimates, d_l: dict, frac_cells: np.ndarray) -> np.ndarray:
    """
    Obtain coefficients of the projected lower-dimensional pressure.

    Parameters
    ----------
        d_l (dict): Lower-dimensional data dictionary.
        frac_cells (np.ndarray): Lower-dimensional fracture cells.

    Raises
    ------
        ValueError
            (*) If the pressure has not been reconstructed.

    Returns
    -------
        p_low (np.ndarray): Coefficients of the projected lower-dimensional pressure.
    """

    # Retrieve lower-dimensional reconstructed pressure coefficients
    if "postprocessed_p" in d_l[estimates.estimates_kw]:
        p_low = d_l[estimates.estimates_kw]["postprocessed_p"]
    else:
        raise ValueError("Pressure must be reconstructed first")
    p_low = p_low[frac_cells]

    return p_low

def _get_normal_velocity(estimates, d_e: dict) -> np.ndarray:
    """
    Obtain the normal velocities for each mortar cell.

    Parameters
    ----------
        d_e (dict): Edge data dictionary.

    Raises
    ------
        ValueError
            If the mortar fluxes are not in the data dictionary

    Returns
    -------
        normal_velocity (np.ndarray): normal velocities (mg.num_cells, 1).

    Note
    ----
        The normal velocities are the mortar fluxes scaled by the mortar cell volumes.

    """

    # Retrieve mortar fluxes from edge dictionary
    if estimates.lam_name in d_e[pp.STATE]:
        mortar_flux: np.ndarray = d_e[pp.STATE][estimates.lam_name].copy()
    else:
        raise ValueError("Mortar fluxes not found in the data dicitionary")

    # Get hold of mortar grid and obtain the volumes of the mortar cells
    mg: pp.MortarGrid = d_e["mortar_grid"]
    cell_volumes = mg.cell_volumes

    # Obtain the normal velocities and reshape into a column array
    normal_velocity = mortar_flux / cell_volumes
    normal_velocity = normal_velocity.reshape(mortar_flux.size, 1)

    return normal_velocity

def interface_diffusive_error_1d(estimates, edge: Edge, d_e: dict) -> np.ndarray:
    """
    Computes diffusive flux error (squared) for one-dimensional mortar grids.

    Parameters
    ----------
        edge (Edge): PorePy edge.
        d_e (dict): Edge data dictionary.

    Raises
    ------
        ValueError
            If the dimension of the mortar grid is different from 1.

    Returns
    -------
        diffusive_error (np.ndarray): Diffusive error (squared) for each cell of the
            mortar grid. The size of the array is: mg.num_cells.
    """

    def compute_sidegrid_error(side_tuple: Tuple) -> np.ndarray:
        """
        Projects a mortar quantity to a side grid and perform numerical integration.

        Parameters
        ----------
            side_tuple (Tuple): Containing the sidegrids

        Returns
        -------
            diffusive_error_side (np.ndarray): Diffusive error (squared) for each element
                of the side grid.
        """

        # Get projector and sidegrid object
        projector = side_tuple[0]
        sidegrid = side_tuple[1]

        # Rotate side-grid
        sidegrid_rot = mde.RotatedGrid(sidegrid)

        # Obtain quadpy elements
        elements = utils.get_quadpy_elements(sidegrid, sidegrid_rot)

        # Project relevant quanitites to the side grid
        deltap_side = projector * deltap
        normalvel_side = projector * normal_vel
        k_side = projector * k

        # Declare integrand
        def integrand(x):
            coors = x[np.newaxis, :, :]  # this is needed for 1D grids
            if estimates.p_degree == 1:
                p_jump = utils.eval_p1(deltap_side, coors)
            else:
                p_jump = utils.eval_p2(deltap_side, coors)
            return (k_side ** (-0.5) * normalvel_side + k_side ** 0.5 * p_jump) ** 2

        # Compute integral
        diffusive_error_side = method.integrate(integrand, elements)

        return diffusive_error_side

    # Get mortar grid and check dimensionality
    mg = d_e["mortar_grid"]
    if mg.dim != 1:
        raise ValueError("Expected one-dimensional mortar grid")

    # Get hold of higher- and lower-dimensional neighbors and their dictionaries
    g_l, g_h = estimates.gb.nodes_of_edge(edge)
    d_h = estimates.gb.node_props(g_h)
    d_l = estimates.gb.node_props(g_l)

    # Retrieve normal diffusivity
    normal_diff = d_e[pp.PARAMETERS][estimates.kw]["normal_diffusivity"]
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
    p_low = _get_low_pressure(estimates, d_l, frac_cells)

    # Now, we can work with the pressure difference
    deltap = p_low - tracep_high

    # Obtain normal velocities
    normal_vel = _get_normal_velocity(estimates, d_e)

    # Declare integration method
    method = qp.c1.newton_cotes_closed(4)

    # Retrieve side-grids tuples
    sides = mg.project_to_side_grids()

    # Compute the errors for each sidegrid
    diffusive = []
    for side in sides:
        diffusive.append(compute_sidegrid_error(side))

    # Concatenate into one numpy array
    diffusive_error = np.concatenate(diffusive)

    return diffusive_error

diffusive_mortar = interface_diffusive_error_1d(estimates, e, d_e)

